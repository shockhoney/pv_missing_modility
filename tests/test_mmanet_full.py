from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from models.comparisons.mmanet_full import (
    FusionTail,
    MMANetImageModel,
    ModalityStem,
    SELayer,
)


def _batch(size: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(101)
    palm = torch.randn(size, 3, 64, 64, generator=generator)
    vein = torch.randn(size, 3, 64, 64, generator=generator)
    labels = torch.arange(size) % 6
    return palm, vein, labels


def test_official_stem_and_fusion_tensor_layout() -> None:
    stem = ModalityStem()
    tail = FusionTail(in_channels=256, num_classes=6).eval()
    assert stem.network[0].kernel_size == (3, 3)
    assert isinstance(stem.se_layer, SELayer)
    assert tail.layer3[0].conv1.in_channels == 256
    assert tail.layer3[0].conv1.out_channels == 256
    assert not hasattr(tail, "input_adapter")
    with torch.no_grad():
        output = tail(torch.randn(2, 256, 8, 8))
    torch.testing.assert_close(output["logits"], tail.classifier(output["embedding"]))
    torch.testing.assert_close(output["representation"].norm(dim=1), torch.ones(2))


def test_teacher_pretraining_and_freeze_are_separate_stages() -> None:
    torch.manual_seed(3)
    model = MMANetImageModel(num_classes=6)
    palm, vein, labels = _batch()
    teacher = model.teacher_loss(palm, vein, labels)
    teacher["total"].backward()
    assert model.teacher.palm_stem.network[0].weight.grad is not None
    assert model.deployment.palm_stem.network[0].weight.grad is None
    model.zero_grad(set_to_none=True)
    model.freeze_teacher()
    assert all(not parameter.requires_grad for parameter in model.teacher.parameters())


def test_mad_matches_official_repository_broadcasting() -> None:
    generator = torch.Generator().manual_seed(13)
    student = torch.randn(4, 8, 2, 2, generator=generator)
    teacher = torch.randn(4, 8, 2, 2, generator=generator)
    logits = torch.randn(4, 6, generator=generator)
    actual = MMANetImageModel.margin_aware_distillation(student, teacher, logits)
    student_flat = student.flatten(1)
    teacher_flat = teacher.flatten(1)
    student_relation = F.normalize(student_flat @ student_flat.t(), p=2, dim=1)
    teacher_relation = F.normalize(teacher_flat @ teacher_flat.t(), p=2, dim=1)
    discrepancy = F.mse_loss(student_relation, teacher_relation, reduction="none")
    probability = F.softmax(logits, dim=1)
    entropy = -(probability * probability.log()).sum(dim=1)
    expected = torch.sum(discrepancy * (entropy / entropy.sum()))

    torch.testing.assert_close(actual, expected)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_mad_stays_finite_under_cuda_fp16_autocast() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(29)
    student = (
        torch.randn(4, 512, 7, 7, device=device, generator=generator) * 100.0
    ).requires_grad_()
    teacher = torch.randn(4, 512, 7, 7, device=device, generator=generator) * 100.0
    logits = torch.randn(4, 6, device=device, generator=generator)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = MMANetImageModel.margin_aware_distillation(
            student, teacher, logits
        )

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_released_classification_recipe_uses_task_only_warmup() -> None:
    model = MMANetImageModel(num_classes=6, warmup_epochs=5).train()
    model.freeze_teacher()
    palm, vein, labels = _batch()
    output = model.deployment_loss(
        palm,
        vein,
        labels,
        palm_present=torch.ones(4, dtype=torch.bool),
        vein_present=torch.ones(4, dtype=torch.bool),
        epoch=5,
    )
    torch.testing.assert_close(output["mad"], torch.zeros_like(output["mad"]))
    torch.testing.assert_close(output["mar"], torch.zeros_like(output["mar"]))
    torch.testing.assert_close(output["total"], output["task"])


def test_mar_uses_all_warmup_histograms_and_activates_after_epoch_five() -> None:
    torch.manual_seed(17)
    model = MMANetImageModel(num_classes=6, warmup_epochs=5)
    complete = torch.tensor([0, 20, 0, 0, 0, 0])
    palm = torch.tensor([20, 0, 0, 0, 0, 0])
    vein = complete.clone()
    for epoch in range(1, 6):
        model.record_mar_epoch(epoch, palm, vein, complete)
    assert model.weak_modality == "palm"
    model.freeze_teacher()
    palm_images, vein_images, labels = _batch()
    output = model.deployment_loss(
        palm_images,
        vein_images,
        labels,
        palm_present=torch.ones(4, dtype=torch.bool),
        vein_present=torch.zeros(4, dtype=torch.bool),
        epoch=6,
    )
    assert bool(output["mar_active"])
    assert output["mar"].item() > 0
    expected = output["task"] + 30.0 * output["mad"] + 0.5 * output["mar"]
    torch.testing.assert_close(output["total"], expected)
    output["total"].backward()
    assert model.deployment.palm_stem.network[0].weight.grad is not None
    assert model.regularizer.classifier.weight.grad is not None
    assert model.teacher.palm_stem.network[0].weight.grad is None


def test_mar_matches_official_full_batch_denominator() -> None:
    model = MMANetImageModel(num_classes=6, warmup_epochs=1).train()
    complete = torch.tensor([0, 20, 0, 0, 0, 0])
    palm_hist = torch.tensor([20, 0, 0, 0, 0, 0])
    model.record_mar_epoch(1, palm_hist, complete, complete)
    model.freeze_teacher()
    model.deployment.eval()
    model.regularizer.eval()
    palm, vein, labels = _batch()
    palm_present = torch.tensor([True, False, False, False])
    vein_present = ~palm_present
    deployment = model.deployment(palm, vein, palm_present, vein_present)
    auxiliary = model.regularizer(deployment["fused_stem"])
    expected = F.cross_entropy(
        auxiliary["logits"][:1], labels[:1], reduction="sum"
    ) / labels.numel()
    actual = model.deployment_loss(
        palm, vein, labels, palm_present, vein_present, epoch=2
    )
    torch.testing.assert_close(actual["mar"], expected)


def test_deployment_supports_complete_and_both_singletons() -> None:
    torch.manual_seed(23)
    model = MMANetImageModel(num_classes=6).eval()
    palm, vein, _ = _batch(2)
    with torch.no_grad():
        complete = model.representation(palm, vein)
        palm_only = model.representation(palm, vein, True, False)
        vein_only = model.representation(palm, vein, False, True)
    assert complete.shape == palm_only.shape == vein_only.shape == (2, 512)
    for representation in (complete, palm_only, vein_only):
        torch.testing.assert_close(representation.norm(dim=1), torch.ones(2))
