from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.backbones import ResNet18Encoder
from models.comparisons.hcmig import (
    HCMIGAdapter,
    MDSFF,
    PatchGAN70Discriminator,
    ResNet9Generator,
)
from utils.full_hcmig_experiment import (
    representation_callback,
    train_generation_epoch,
    train_recognition_epoch,
)


def _paired_images(size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    palm = torch.rand(1, 3, size, size, generator=generator) * 2.0 - 1.0
    vein = torch.rand(1, 3, size, size, generator=generator) * 2.0 - 1.0
    return palm, vein


def _has_finite_nonzero_gradient(module: torch.nn.Module) -> bool:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    return bool(gradients) and all(torch.isfinite(gradient).all() for gradient in gradients) and any(
        torch.count_nonzero(gradient).item() > 0 for gradient in gradients
    )


def test_architecture_and_generated_modalities_have_expected_shape() -> None:
    torch.manual_seed(11)
    model = HCMIGAdapter(base_channels=4).eval()
    palm, vein = _paired_images()

    assert sum(isinstance(module, type(model.texture_v2p.network[10])) for module in model.texture_v2p.modules()) == 9
    assert isinstance(model.palm_discriminator, PatchGAN70Discriminator)
    assert isinstance(model.texture_p2v, ResNet9Generator)

    with torch.no_grad():
        generated_vein = model.generate_missing(palm, "palm")
        generated_palm = model.generate_missing(vein, "palmvein")

    assert generated_vein.shape == palm.shape
    assert generated_palm.shape == vein.shape
    assert torch.isfinite(generated_vein).all()
    assert torch.isfinite(generated_palm).all()
    assert generated_vein.min() >= -1.0 and generated_vein.max() <= 1.0
    assert generated_palm.min() >= -1.0 and generated_palm.max() <= 1.0

    with pytest.raises(ValueError, match="available_modality"):
        model.generate_missing(palm, "iris")


def test_generator_losses_are_finite_weighted_and_reach_all_generators() -> None:
    torch.manual_seed(13)
    model = HCMIGAdapter(base_channels=4).train()
    palm, vein = _paired_images()

    losses = model.generator_loss_dict(palm, vein)
    assert set(losses) == {
        "total",
        "cycle",
        "adversarial",
        "cms",
        "fourier",
        "fourier_structure",
        "fourier_texture",
    }
    assert all(loss.ndim == 0 and torch.isfinite(loss) for loss in losses.values())
    expected = (
        losses["cycle"]
        + losses["adversarial"]
        + losses["cms"]
        + 0.1 * (losses["fourier_structure"] + losses["fourier_texture"])
    )
    torch.testing.assert_close(losses["total"], expected)

    losses["total"].backward()
    assert _has_finite_nonzero_gradient(model.texture_v2p)
    assert _has_finite_nonzero_gradient(model.texture_p2v)
    assert _has_finite_nonzero_gradient(model.structure_v2p)
    assert _has_finite_nonzero_gradient(model.structure_p2v)


def test_discriminator_losses_are_finite_and_do_not_backpropagate_generators() -> None:
    torch.manual_seed(17)
    model = HCMIGAdapter(base_channels=4).train()
    palm, vein = _paired_images()

    losses = model.discriminator_loss_dict(palm, vein)
    assert set(losses) == {
        "total",
        "palm",
        "vein",
        "palm_real",
        "palm_fake",
        "vein_real",
        "vein_fake",
    }
    assert all(loss.ndim == 0 and torch.isfinite(loss) for loss in losses.values())
    torch.testing.assert_close(losses["total"], losses["palm"] + losses["vein"])

    losses["total"].backward()
    assert _has_finite_nonzero_gradient(model.palm_discriminator)
    assert _has_finite_nonzero_gradient(model.vein_discriminator)
    assert all(parameter.grad is None for parameter in model.generator_parameters())


def test_radial_frequency_masks_are_centered_and_complementary() -> None:
    model = HCMIGAdapter(base_channels=2, fft_radius_ratio=0.1)
    low, high = model.radial_frequency_masks(32, 64)

    assert low.shape == high.shape == (1, 1, 32, 64)
    torch.testing.assert_close(low + high, torch.ones_like(low), rtol=0.0, atol=0.0)
    assert low[0, 0, 16, 32].item() == 1.0
    assert low[0, 0, 0, 0].item() == 0.0
    assert torch.count_nonzero(low).item() > 1
    assert torch.count_nonzero(high).item() > torch.count_nonzero(low).item()


def test_normal_weight_initialization_is_applied() -> None:
    torch.manual_seed(19)
    model = HCMIGAdapter(base_channels=8)
    weight = model.texture_v2p.network[1].weight.detach()

    assert abs(weight.mean().item()) < 0.005
    assert 0.015 < weight.std().item() < 0.025



def test_resnet_dual_encoder_topology_and_independence() -> None:
    model = HCMIGAdapter(base_channels=2, num_classes=5)

    assert isinstance(model.palm_encoder, ResNet18Encoder)
    assert isinstance(model.vein_encoder, ResNet18Encoder)
    assert model.palm_encoder is not model.vein_encoder
    assert model.palm_encoder.local_dim == 256
    assert model.vein_encoder.local_dim == 256
    assert isinstance(model.mdsff.palm_importance.network[-1], torch.nn.Linear)


def test_multinomial_sparse_fusion_is_channelwise_and_eval_is_expected_value() -> None:
    model = MDSFF(num_classes=3, embedding_size=16, dropout=0.0)
    palm = torch.full((2, model.feature_dim), 3.0)
    vein = torch.full((2, model.feature_dim), -2.0)
    probabilities = torch.tensor([[0.25, 0.75], [0.6, 0.4]])

    model.train()
    fused, mask = model.sparse_fuse(
        palm,
        vein,
        probabilities,
        stochastic=True,
        generator=torch.Generator().manual_seed(23),
    )
    assert mask.shape == (2, 2, model.feature_dim)
    torch.testing.assert_close(
        mask.sum(dim=1), torch.ones_like(mask[:, 0]), rtol=0.0, atol=0.0
    )
    assert torch.all((fused == 3.0) | (fused == -2.0))

    model.eval()
    expected, soft_mask = model.sparse_fuse(
        palm, vein, probabilities, stochastic=False
    )
    torch.testing.assert_close(expected[0], torch.full_like(expected[0], -0.75))
    torch.testing.assert_close(soft_mask[:, :, 0], probabilities)


def test_recognition_objective_is_weighted_and_trains_both_modalities() -> None:
    torch.manual_seed(29)
    model = HCMIGAdapter(
        base_channels=2,
        num_classes=3,
        recognition_dropout=0.0,
    ).train()
    palm, vein = _paired_images()
    palm = torch.cat([palm, palm.flip(-1)], dim=0)
    vein = torch.cat([vein, vein.flip(-2)], dim=0)
    labels = torch.tensor([0, 2])

    losses = model.recognition_loss_dict(
        palm,
        vein,
        labels,
        stochastic=True,
        generator=torch.Generator().manual_seed(31),
    )
    assert set(losses) == {
        "total",
        "fused_cls",
        "palm_cls",
        "vein_cls",
        "confidence",
    }
    expected = (
        losses["fused_cls"]
        + 0.5 * losses["palm_cls"]
        + 0.5 * losses["vein_cls"]
        + 0.1 * losses["confidence"]
    )
    torch.testing.assert_close(losses["total"], expected)
    losses["total"].backward()

    assert _has_finite_nonzero_gradient(model.palm_encoder)
    assert _has_finite_nonzero_gradient(model.vein_encoder)
    assert _has_finite_nonzero_gradient(model.mdsff.palm_importance)
    assert _has_finite_nonzero_gradient(model.mdsff.vein_importance)
    assert _has_finite_nonzero_gradient(model.mdsff.palm_confidence)
    assert _has_finite_nonzero_gradient(model.mdsff.vein_confidence)
    assert all(parameter.grad is None for parameter in model.generator_parameters())


def test_complete_and_missing_recognition_return_same_embedding_contract() -> None:
    torch.manual_seed(37)
    model = HCMIGAdapter(
        base_channels=2,
        num_classes=4,
        recognition_dropout=0.0,
    ).eval()
    palm, vein = _paired_images()

    with torch.no_grad():
        complete = model.recognize(palm_domain=palm, vein_domain=vein)
        vein_missing = model.recognize(palm_domain=palm)
        palm_missing = model.recognize(vein_domain=vein)

    assert complete["normalized_embedding"].shape == (1, model.mdsff.embedding_dim)
    assert vein_missing["normalized_embedding"].shape == complete["normalized_embedding"].shape
    assert palm_missing["normalized_embedding"].shape == complete["normalized_embedding"].shape
    assert complete["generated_modality"].item() == 0
    assert palm_missing["generated_modality"].item() == 1
    assert vein_missing["generated_modality"].item() == 2
    assert torch.isfinite(vein_missing["normalized_embedding"]).all()


def test_fourier_equations_are_directional() -> None:
    model = HCMIGAdapter(base_channels=2, num_classes=3)
    palm, vein = _paired_images()
    changed_palm = palm.roll(shifts=3, dims=-1)
    changed_vein = vein.roll(shifts=5, dims=-2)

    structure, _ = model._fourier_losses(palm, palm, changed_vein, vein)
    _, texture = model._fourier_losses(changed_palm, palm, vein, vein)
    torch.testing.assert_close(structure, torch.zeros_like(structure))
    torch.testing.assert_close(texture, torch.zeros_like(texture))


def test_stage_freezing_and_parameter_groups_are_disjoint() -> None:
    model = HCMIGAdapter(base_channels=2, num_classes=3)
    generator_ids = {id(parameter) for parameter in model.generator_parameters()}
    discriminator_ids = {
        id(parameter) for parameter in model.discriminator_parameters()
    }
    recognition_ids = {id(parameter) for parameter in model.recognition_parameters()}
    assert not generator_ids & discriminator_ids
    assert not generator_ids & recognition_ids
    assert not discriminator_ids & recognition_ids

    model.set_training_stage("recognition")
    assert all(not parameter.requires_grad for parameter in model.generator_parameters())
    assert all(not parameter.requires_grad for parameter in model.discriminator_parameters())
    assert all(parameter.requires_grad for parameter in model.recognition_parameters())

    model.set_training_stage("generation")
    assert all(parameter.requires_grad for parameter in model.generator_parameters())
    assert all(parameter.requires_grad for parameter in model.discriminator_parameters())
    assert all(not parameter.requires_grad for parameter in model.recognition_parameters())



def test_full_trainer_single_batch_smoke_covers_both_stages() -> None:
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = HCMIGAdapter(
        base_channels=2,
        num_classes=3,
        recognition_embedding_size=16,
        recognition_dropout=0.0,
    )
    images = torch.rand(3, 3, 32, 32) * 2.0 - 1.0
    labels = torch.tensor([0, 1, 2])
    masks = torch.ones(3, 2)
    loader = DataLoader(
        TensorDataset(images, images.flip(-1), labels, masks),
        batch_size=3,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    generator_optimizer = torch.optim.SGD(
        model.generator_parameters(), lr=1e-4, weight_decay=5e-3
    )
    discriminator_optimizer = torch.optim.SGD(
        model.discriminator_parameters(), lr=1e-4, weight_decay=5e-3
    )
    generation = train_generation_epoch(
        model,
        loader,
        generator_optimizer,
        discriminator_optimizer,
        device,
        scaler,
        micro_batch_size=1,
        gradient_clip=1.0,
    )
    recognition_optimizer = torch.optim.SGD(
        model.recognition_parameters(), lr=1e-4, weight_decay=5e-3
    )
    recognition = train_recognition_epoch(
        model,
        loader,
        recognition_optimizer,
        device,
        scaler,
        gradient_clip=1.0,
    )
    model.eval()
    callback = representation_callback(
        model, device=device, stochastic=True, seed=47
    )
    probe_masks = torch.tensor(
        [[1, 1], [1, 0], [0, 1]], dtype=torch.bool
    )
    with torch.inference_mode():
        embeddings = callback(images, images.flip(-1), probe_masks)

    assert generation["generator_total"] > 0.0
    assert generation["discriminator_total"] > 0.0
    assert recognition["total"] > 0.0
    assert embeddings.shape == (3, model.mdsff.embedding_dim)
    torch.testing.assert_close(
        torch.linalg.vector_norm(embeddings, dim=1),
        torch.ones(3),
        atol=1e-5,
        rtol=1e-5,
    )
