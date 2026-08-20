from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.comparisons.ssfd import (
    BidirectionalCrossModalFeatureTransformation,
    ResNetSSFDNet,
    ResNetSharedSpecificEncoder,
    SSFDAdapter,
)


def _adapter(**kwargs: object) -> SSFDAdapter:
    torch.manual_seed(7)
    options = {"num_classes": 6, "cmft_hidden_dim": 16, "dropout": 0.0}
    options.update(kwargs)
    return SSFDAdapter(**options)


class _TinyImageEncoder(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.shared = nn.Linear(6, feature_dim)
        self.specific = nn.Linear(6, feature_dim)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        base = self.features(images).flatten(1)
        return {"shared": self.shared(base), "specific": self.specific(base)}


def test_adapter_deployment_representations_have_stable_order_and_norm() -> None:
    model = _adapter().eval()
    palm = torch.randn(4, 256)
    vein = torch.randn(4, 256)

    palm_parts = model.encode_parts(palm, "palm")
    vein_parts = model.encode_parts(vein, "palmvein")
    assert set(palm_parts) == {"shared", "specific"}
    assert palm_parts["shared"].shape == (4, 128)
    assert vein_parts["specific"].shape == (4, 128)

    raw = model.complete_representation(palm, vein, normalize=False)
    torch.testing.assert_close(
        raw,
        torch.cat(
            [
                palm_parts["shared"], palm_parts["specific"],
                vein_parts["shared"], vein_parts["specific"],
            ],
            dim=1,
        ),
    )
    complete = model.complete_representation(palm, vein)
    palm_available = model.missing_representation(palm, "palm")
    vein_available = model.missing_representation(vein, "vein")
    assert complete.shape == palm_available.shape == vein_available.shape == (4, 512)
    torch.testing.assert_close(complete, F.normalize(raw, dim=1))
    for representation in (complete, palm_available, vein_available):
        torch.testing.assert_close(
            representation.norm(dim=1), torch.ones(4), atol=1e-6, rtol=1e-6
        )

    # PM names a missing palm (vein available); VM is the inverse.
    torch.testing.assert_close(model.missing_representation(vein, "PM"), vein_available)
    torch.testing.assert_close(model.missing_representation(palm, "VM"), palm_available)
    torch.testing.assert_close(model(palm=palm), palm_available)
    torch.testing.assert_close(model(vein=vein), vein_available)
    torch.testing.assert_close(model(palm=palm, vein=vein), complete)
    assert model(palm=palm, vein=vein, return_logits=True).shape == (4, 6)


def test_bidirectional_cmft_has_paper_dimensions_and_shared_default() -> None:
    cmft = BidirectionalCrossModalFeatureTransformation(
        feature_dim=128, hidden_dim=2048, dropout=0.5
    )
    assert cmft.share_weights
    assert cmft.shared_transform is not None
    linears = [
        module for module in cmft.shared_transform.modules()
        if isinstance(module, nn.Linear)
    ]
    assert [(layer.in_features, layer.out_features) for layer in linears] == [
        (128, 2048), (2048, 2048), (2048, 128)
    ]
    assert all(torch.count_nonzero(layer.weight) > 0 for layer in linears)
    assert all(torch.count_nonzero(layer.bias) == 0 for layer in linears)
    assert sum(parameter.numel() for parameter in cmft.parameters()) == sum(
        parameter.numel() for layer in linears for parameter in layer.parameters()
    )

    outputs = cmft(torch.randn(2, 128), torch.randn(2, 128))
    assert outputs["palm_from_vein"].shape == (2, 128)
    assert outputs["vein_from_palm"].shape == (2, 128)

    directed = BidirectionalCrossModalFeatureTransformation(
        feature_dim=8, hidden_dim=16, dropout=0.0, share_weights=False
    )
    assert directed.palm_to_vein_transform is not directed.vein_to_palm_transform


def test_five_losses_are_scalar_weighted_raw_and_differentiable() -> None:
    model = _adapter().train()
    palm = torch.randn(5, 256)
    vein = torch.randn(5, 256)
    labels = torch.tensor([0, 1, 2, 3, 4])

    losses = model.loss_dict(palm, vein, labels)
    assert set(losses) == {
        "classification", "triplet", "transformation",
        "inter_consistency", "intra_consistency", "total",
    }
    assert all(loss.ndim == 0 and torch.isfinite(loss) for loss in losses.values())
    expected = losses["classification"] + 0.3 * (
        losses["triplet"] + losses["transformation"]
        + losses["inter_consistency"] + losses["intra_consistency"]
    )
    torch.testing.assert_close(losses["total"], expected)

    # The paper losses use raw latent scales, not normalized surrogate logits.
    assert not torch.allclose(
        model.encode_parts(palm, "palm")["shared"].norm(dim=1),
        torch.ones(5),
    )

    losses["total"].backward()
    for module in (
        model.palm_shared_head, model.palm_specific_head,
        model.vein_shared_head, model.vein_specific_head,
        model.cmft, model.identity_classifier,
    ):
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in module.parameters()
        )


def _frozen_classifier(feature_dim: int, num_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(0.5), nn.Linear(feature_dim, num_classes)
    )


def test_pretrained_identity_classifiers_remain_frozen_and_in_eval_mode() -> None:
    palm_classifier = _frozen_classifier(256, 6)
    vein_classifier = _frozen_classifier(256, 6)
    palm_weight = palm_classifier[-1].weight.detach().clone()
    vein_weight = vein_classifier[-1].weight.detach().clone()
    model = _adapter(
        palm_classifier=palm_classifier,
        vein_classifier=vein_classifier,
    ).train()

    assert not palm_classifier.training
    assert not vein_classifier.training
    assert all(not parameter.requires_grad for parameter in palm_classifier.parameters())
    assert all(not parameter.requires_grad for parameter in vein_classifier.parameters())

    losses = model.loss_dict(
        torch.randn(4, 256), torch.randn(4, 256), torch.tensor([0, 1, 2, 3])
    )
    losses["total"].backward()
    assert all(parameter.grad is None for parameter in palm_classifier.parameters())
    assert all(parameter.grad is None for parameter in vein_classifier.parameters())
    torch.testing.assert_close(palm_classifier[-1].weight, palm_weight)
    torch.testing.assert_close(vein_classifier[-1].weight, vein_weight)


def test_resnet_shared_specific_encoder_emits_configured_parts() -> None:
    torch.manual_seed(11)
    encoder = ResNetSharedSpecificEncoder(embedding_size=16, use_se=False).eval()
    with torch.inference_mode():
        parts = encoder(torch.randn(1, 3, 32, 32))
    assert parts["shared"].shape == (1, 8)
    assert parts["specific"].shape == (1, 8)
    assert not torch.equal(parts["shared"], parts["specific"])


def test_complete_image_model_is_end_to_end_and_requires_real_teachers() -> None:
    feature_dim = 4
    embedding_size = 2 * feature_dim
    palm_encoder = _TinyImageEncoder(feature_dim)
    vein_encoder = _TinyImageEncoder(feature_dim)
    palm_teacher = _frozen_classifier(embedding_size, 3)
    vein_teacher = _frozen_classifier(embedding_size, 3)
    model = ResNetSSFDNet(
        num_classes=3,
        embedding_size=embedding_size,
        cmft_hidden_dim=8,
        dropout=0.0,
        palm_classifier=palm_teacher,
        vein_classifier=vein_teacher,
        palm_encoder=palm_encoder,
        vein_encoder=vein_encoder,
    ).train()
    palm = torch.randn(2, 3, 16, 16)
    vein = torch.randn(2, 3, 16, 16)
    labels = torch.tensor([0, 1])

    assert model.complete_representation(palm, vein).shape == (2, 16)
    assert model.missing_representation(palm, "palm").shape == (2, 16)
    assert model.missing_representation(vein, "vein").shape == (2, 16)
    assert model.classification_logits(palm, vein).shape == (2, 3)
    losses = model.loss_dict(palm, vein, labels)
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in palm_encoder.parameters())
    assert any(parameter.grad is not None for parameter in vein_encoder.parameters())
    assert all(parameter.grad is None for parameter in palm_teacher.parameters())
    assert all(parameter.grad is None for parameter in vein_teacher.parameters())

    missing_teachers = ResNetSSFDNet(
        num_classes=3,
        embedding_size=embedding_size,
        cmft_hidden_dim=8,
        dropout=0.0,
        palm_encoder=_TinyImageEncoder(feature_dim),
        vein_encoder=_TinyImageEncoder(feature_dim),
    )
    with pytest.raises(RuntimeError, match="pre-trained frozen"):
        missing_teachers.loss_dict(palm, vein, labels)


@pytest.mark.parametrize(
    ("operation", "exception", "match"),
    [
        (lambda model: model.encode_parts(torch.randn(2, 255), "palm"), ValueError, "shape"),
        (lambda model: model.encode_parts(torch.randn(2, 256), "iris"), ValueError, "Unsupported"),
        (
            lambda model: model.complete_representation(
                torch.randn(2, 256), torch.randn(3, 256)
            ),
            ValueError,
            "batch sizes",
        ),
        (lambda model: model(), ValueError, "At least one"),
        (
            lambda model: model.encode_parts(torch.ones(2, 256, dtype=torch.long), "palm"),
            TypeError,
            "floating-point",
        ),
    ],
)
def test_invalid_inputs_fail_loudly(operation: object, exception: type[Exception], match: str) -> None:
    with pytest.raises(exception, match=match):
        operation(_adapter())  # type: ignore[operator]
