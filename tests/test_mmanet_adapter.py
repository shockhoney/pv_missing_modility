from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from models.comparisons.mmanet import MMANetAdapter


def _adapter(**kwargs: object) -> MMANetAdapter:
    torch.manual_seed(7)
    palm_weight = torch.randn(6, 256)
    vein_weight = torch.randn(6, 256)
    return MMANetAdapter(
        num_classes=6,
        palm_teacher_weight=palm_weight,
        vein_teacher_weight=vein_weight,
        **kwargs,
    )


def _batch(size: int = 4) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(11)
    palm_map = torch.randn(size, 256, 7, 7, generator=generator)
    vein_map = torch.randn(size, 256, 7, 7, generator=generator)
    palm_embedding = torch.randn(size, 256, generator=generator)
    vein_embedding = torch.randn(size, 256, generator=generator)
    labels = torch.arange(size) % 6
    return palm_map, vein_map, palm_embedding, vein_embedding, labels


def test_representations_support_scalar_and_batch_masks_and_reject_all_missing() -> None:
    model = _adapter().eval()
    palm, vein, _, _, _ = _batch(3)

    complete = model.representation(palm, vein)
    palm_only = model.representation(palm, None, True, False)
    vein_only = model.representation(None, vein, False, True)
    mixed = model.representation(
        palm,
        vein,
        torch.tensor([True, True, False]),
        torch.tensor([True, False, True]),
    )
    assert complete.shape == palm_only.shape == vein_only.shape == mixed.shape == (3, 256)
    for representation in (complete, palm_only, vein_only, mixed):
        torch.testing.assert_close(
            representation.norm(dim=1), torch.ones(3), atol=1e-6, rtol=1e-6
        )
    torch.testing.assert_close(mixed[1], palm_only[1])
    torch.testing.assert_close(mixed[2], vein_only[2])
    with pytest.raises(ValueError, match="at least one modality"):
        model.representation(palm, vein, False, False)


def test_fixed_teacher_uses_cached_embeddings_and_arcface_cosine_weights() -> None:
    model = _adapter().train()
    palm, vein, palm_embedding, vein_embedding, labels = _batch()
    original_palm_weight = model.palm_teacher_weight.clone()
    original_vein_weight = model.vein_teacher_weight.clone()

    output = model(
        palm,
        vein,
        labels=labels,
        epoch=1,
        palm_embedding=palm_embedding,
        vein_embedding=vein_embedding,
    )
    expected_representation = F.normalize(palm_embedding + vein_embedding, dim=1)
    expected_logits = 16.0 * (
        F.linear(F.normalize(palm_embedding, dim=1), F.normalize(original_palm_weight, dim=1))
        + F.linear(F.normalize(vein_embedding, dim=1), F.normalize(original_vein_weight, dim=1))
    )
    torch.testing.assert_close(output["teacher_representation"], expected_representation)
    torch.testing.assert_close(output["teacher_logits"], expected_logits)
    assert not model.palm_teacher_weight.requires_grad
    assert not model.vein_teacher_weight.requires_grad

    output["loss_dict"]["total"].backward()
    assert model.palm_teacher_weight.grad is None
    assert model.vein_teacher_weight.grad is None
    torch.testing.assert_close(model.palm_teacher_weight, original_palm_weight)
    torch.testing.assert_close(model.vein_teacher_weight, original_vein_weight)


def test_mad_matches_official_entropy_weighted_gram_mse() -> None:
    generator = torch.Generator().manual_seed(13)
    student = torch.randn(4, 8, generator=generator)
    teacher = torch.randn(4, 8, generator=generator)
    logits = torch.randn(4, 6, generator=generator)

    actual = MMANetAdapter.margin_aware_distillation(student, teacher, logits)
    student_gram = F.normalize(student @ student.t(), p=2, dim=1)
    teacher_gram = F.normalize(teacher @ teacher.t(), p=2, dim=1)
    mse = F.mse_loss(student_gram, teacher_gram, reduction="none")
    probability = F.softmax(logits, dim=1)
    entropy = -(probability * probability.log()).sum(dim=1)
    expected = torch.sum(mse * (entropy / entropy.sum()))
    torch.testing.assert_close(actual, expected)


def test_mar_collects_only_training_histograms_locks_weak_and_activates_epoch_six() -> None:
    model = _adapter().train()
    palm, vein, palm_embedding, vein_embedding, labels = _batch()

    # Make palm the unambiguously weak singleton under the official histogram
    # distance: palm predicts only class 0, vein and complete only class 1.
    for epoch in range(2, 6):
        model.begin_epoch(epoch)
        palm_logits = torch.tensor([[9.0, 0, 0, 0, 0, 0]]).repeat(4, 1)
        vein_logits = torch.tensor([[0.0, 9, 0, 0, 0, 0]]).repeat(4, 1)
        complete_logits = vein_logits.clone()
        model._update_prediction_histograms(palm_logits, vein_logits, complete_logits)
        model.end_epoch()

    assert model.weak_modality == "palm"
    assert model.mar_distribution_distance[0] > model.mar_distribution_distance[1]
    assert int(model.mar_epoch_observations.sum()) == 12

    output = model(
        palm,
        vein,
        labels=labels,
        epoch=6,
        palm_embedding=palm_embedding,
        vein_embedding=vein_embedding,
    )
    assert output["mar_active"]
    assert output["weak_modality"] == "palm"
    losses = output["loss_dict"]
    expected = losses["deployment"] + 30.0 * losses["mad"] + 0.5 * losses["mar"]
    torch.testing.assert_close(losses["total"], expected)
    assert losses["mar"].item() > 0.0

    state = model.state_dict()
    restored = _adapter()
    restored.load_state_dict(state)
    assert restored.weak_modality == "palm"
    torch.testing.assert_close(
        restored.mar_distribution_distance, model.mar_distribution_distance
    )


def test_warmup_total_excludes_mad_mar_but_all_student_paths_receive_gradients() -> None:
    model = _adapter().train()
    palm, vein, palm_embedding, vein_embedding, labels = _batch()
    output = model(
        palm,
        vein,
        labels=labels,
        epoch=3,
        palm_embedding=palm_embedding,
        vein_embedding=vein_embedding,
    )
    losses = output["loss_dict"]
    assert not output["mar_active"]
    assert losses["mar"].item() == 0.0
    torch.testing.assert_close(losses["total"], losses["deployment"])
    assert losses["mad"].item() >= 0.0

    losses["total"].backward()
    for module in (
        model.palm_special,
        model.vein_special,
        model.shared_fusion,
        model.deployment_head,
    ):
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in module.parameters()
        )
    # The auxiliary head is deliberately dormant until MAR activation.
    assert all(parameter.grad is None for parameter in model.auxiliary_features.parameters())


@pytest.mark.parametrize(
    ("operation", "match"),
    [
        (lambda model: model.representation(torch.randn(2, 255, 7, 7), None, True, False), "shape"),
        (lambda model: model.representation(torch.randn(2, 256, 7, 7), None), "vein_map"),
        (
            lambda model: model(
                torch.randn(2, 256, 7, 7),
                torch.randn(2, 256, 7, 7),
                labels=torch.tensor([0, 1]),
            ),
            "epoch",
        ),
    ],
)
def test_invalid_inputs_fail_loudly(operation: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        operation(_adapter())  # type: ignore[operator]
