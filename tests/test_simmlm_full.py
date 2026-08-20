from __future__ import annotations

import torch
import torch.nn.functional as F

from models.comparisons.simmlm_full import SimMLMImageModel


def _batch(size: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(811)
    palm = torch.randn(size, 3, 64, 64, generator=generator)
    vein = torch.randn(size, 3, 64, 64, generator=generator)
    labels = torch.arange(size) % 5
    return palm, vein, labels


def test_logit_level_gate_masks_missing_modalities_exactly() -> None:
    torch.manual_seed(5)
    model = SimMLMImageModel(embedding_dim=32, num_classes=5).eval()
    palm, vein, _ = _batch()
    palm_mask = torch.tensor([True, True, False, False])
    vein_mask = torch.tensor([True, False, True, True])
    assert model.router_head.in_features == 256
    with torch.no_grad():
        output = model(palm, vein, palm_mask, vein_mask)
    weights = output["router_weights"]
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(4))
    assert weights[1].tolist() == [1.0, 0.0]
    assert weights[2].tolist() == [0.0, 1.0]
    expected = weights[:, :1] * output["palm_logits"] + weights[:, 1:] * output["vein_logits"]
    torch.testing.assert_close(output["logits"], expected)
    torch.testing.assert_close(output["representation"].norm(dim=1), torch.ones(4))


def test_missing_expert_is_not_forwarded_or_used_to_update_batchnorm() -> None:
    torch.manual_seed(6)
    model = SimMLMImageModel(embedding_dim=32, num_classes=5).train()
    palm, vein, _ = _batch()
    running_mean = model.vein_expert.encoder.backbone.bn1.running_mean.clone()
    final_mean = model.vein_expert.encoder.bn.running_mean.clone()

    output = model(palm, vein, True, False)

    torch.testing.assert_close(
        model.vein_expert.encoder.backbone.bn1.running_mean, running_mean
    )
    torch.testing.assert_close(model.vein_expert.encoder.bn.running_mean, final_mean)
    torch.testing.assert_close(output["vein_embedding"], torch.zeros_like(output["vein_embedding"]))
    torch.testing.assert_close(output["vein_logits"], torch.zeros_like(output["vein_logits"]))


def test_one_present_sample_uses_safe_expert_batchnorm_path() -> None:
    model = SimMLMImageModel(embedding_dim=32, num_classes=5).train()
    palm, vein, _ = _batch()
    palm_present = torch.tensor([False, True, True, True])
    output = model(palm, vein, palm_present, ~palm_present)
    output["logits"].sum().backward()
    assert model.vein_expert.classifier.weight.grad is not None


def test_cooperative_loss_is_paper_mofe_equation_and_backpropagates() -> None:
    torch.manual_seed(7)
    model = SimMLMImageModel(embedding_dim=32, num_classes=5).train()
    palm, vein, labels = _batch()
    fewer_is_palm = torch.tensor([True, False, True, False])
    loss = model.cooperative_loss(palm, vein, labels, fewer_is_palm)
    more_each = F.cross_entropy(loss["more_logits"], labels, reduction="none")
    fewer_each = F.cross_entropy(loss["fewer_logits"], labels, reduction="none")
    expected_mofe = F.relu(more_each - fewer_each).mean()
    expected_total = more_each.mean() + fewer_each.mean() + 0.1 * expected_mofe
    torch.testing.assert_close(loss["mofe"], expected_mofe)
    torch.testing.assert_close(loss["total"], expected_total)
    loss["total"].backward()
    for parameter in (
        model.palm_expert.encoder.backbone.conv1.weight,
        model.vein_expert.encoder.backbone.conv1.weight,
        model.palm_router.features[0].weight,
        model.vein_router.features[0].weight,
        model.router_head.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_independent_expert_stage_excludes_other_expert() -> None:
    torch.manual_seed(9)
    model = SimMLMImageModel(embedding_dim=32, num_classes=5).train()
    palm, _, labels = _batch()
    loss = model.expert_loss("palm", palm, labels)["total"]
    loss.backward()
    assert model.palm_expert.encoder.backbone.conv1.weight.grad is not None
    assert model.palm_expert.classifier.weight.grad is not None
    assert model.vein_expert.encoder.backbone.conv1.weight.grad is None
    assert model.router_head.weight.grad is None


def test_all_missing_is_rejected() -> None:
    model = SimMLMImageModel(embedding_dim=32, num_classes=5).eval()
    palm, vein, _ = _batch(2)
    try:
        model(palm, vein, False, False)
    except ValueError as exc:
        assert "at least one modality" in str(exc)
    else:
        raise AssertionError("all-missing batch should fail")
