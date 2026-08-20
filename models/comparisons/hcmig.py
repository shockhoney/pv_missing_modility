"""Paper-derived complete two-stage HCMIG implementation.

This module implements both hierarchical cross-modal image generation and the
paper's multimodal dynamic sparse feature fusion (MDSFF) recognition stage.

HCMIGAdapter image inputs and generated outputs use three channels in the
``[-1, 1]`` domain.  Its lower-level MDSFF module accepts ImageNet-normalized
images so it can also be used directly with the project data loaders.

The recognition encoders are the project's pretrained palm/vein ResNet-18
backbones (``models.backbones.ResNet18Encoder``) instead of the paper's VGG16;
the generation stage keeps the paper's ResNet-9/PatchGAN structure.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import ResNet18Encoder


__all__ = [
    "HCMIG",
    "ConfidencePredictor",
    "HCMIGAdapter",
    "MDSFF",
    "ModalImportanceHead",
    "PatchGAN70Discriminator",
    "ResNet9Generator",
]


class ResidualBlock(nn.Module):
    """CycleGAN residual block with reflection padding and InstanceNorm."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels, affine=False, track_running_stats=False),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images + self.block(images)


class ResNet9Generator(nn.Module):
    """Standard CycleGAN ResNet generator with nine residual blocks."""

    def __init__(self, input_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        if input_channels <= 0 or base_channels <= 0:
            raise ValueError("input_channels and base_channels must be positive")

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, c1, kernel_size=7, bias=False),
            nn.InstanceNorm2d(c1, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(c2, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(c3, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
        ]
        layers.extend(ResidualBlock(c3) for _ in range(9))
        layers.extend(
            [
                nn.ConvTranspose2d(
                    c3,
                    c2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(c2, affine=False, track_running_stats=False),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    c2,
                    c1,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(c1, affine=False, track_running_stats=False),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(3),
                nn.Conv2d(c1, input_channels, kernel_size=7),
                nn.Tanh(),
            ]
        )
        self.network = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


class PatchGAN70Discriminator(nn.Module):
    """The 70x70 PatchGAN discriminator used by pix2pix/CycleGAN."""

    def __init__(self, input_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        if input_channels <= 0 or base_channels <= 0:
            raise ValueError("input_channels and base_channels must be positive")

        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, c1, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(c2, affine=False, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(c3, affine=False, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c3, c4, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(c4, affine=False, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c4, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


def _normal_init(module: nn.Module) -> None:
    """CycleGAN initialization: convolutional weights N(0, 0.02)."""

    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ModalImportanceHead(nn.Module):
    """MLP class predictor used for modality importance evaluation."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or num_classes <= 0:
            raise ValueError("input_dim and num_classes must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        hidden_dim = input_dim if hidden_dim is None else int(hidden_dim)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    def probabilities(self, features: torch.Tensor) -> torch.Tensor:
        # Equations (13)-(14) explicitly use sigmoid rather than softmax.
        return torch.sigmoid(self(features))


class ConfidencePredictor(nn.Module):
    """Reconstruct the unavailable-at-test-time true-class probability."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        hidden_dim = input_dim if hidden_dim is None else int(hidden_dim)
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


class MDSFF(nn.Module):
    """Multimodal Dynamic Sparse Feature Fusion, equations (11)-(21)."""

    def __init__(
        self,
        num_classes: int,
        *,
        embedding_size: int = 256,
        embedding_dim: int | None = None,
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.5,
        palm_encoder: ResNet18Encoder | None = None,
        vein_encoder: ResNet18Encoder | None = None,
        lambda_fused_cls: float = 1.0,
        lambda_palm_cls: float = 0.5,
        lambda_vein_cls: float = 0.5,
        lambda_confidence: float = 0.1,
        confidence_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if embedding_size <= 0 or embedding_size % 2:
            raise ValueError("embedding_size must be positive and even")
        if confidence_epsilon <= 0.0:
            raise ValueError("confidence_epsilon must be positive")
        loss_weights = (
            lambda_fused_cls,
            lambda_palm_cls,
            lambda_vein_cls,
            lambda_confidence,
        )
        if any(weight < 0.0 for weight in loss_weights):
            raise ValueError(
                "recognition loss weights must be non-negative"
            )

        self.palm_encoder = palm_encoder or ResNet18Encoder(
            input_channel=3, embedding_size=embedding_size
        )
        self.vein_encoder = vein_encoder or ResNet18Encoder(
            input_channel=3, embedding_size=embedding_size, use_se=True
        )
        feature_dim = self.palm_encoder.local_dim
        embedding_dim = feature_dim if embedding_dim is None else int(embedding_dim)
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.palm_importance = ModalImportanceHead(
            feature_dim,
            num_classes,
            hidden_dim=mlp_hidden_dim,
            dropout=dropout,
        )
        self.vein_importance = ModalImportanceHead(
            feature_dim,
            num_classes,
            hidden_dim=mlp_hidden_dim,
            dropout=dropout,
        )
        self.palm_confidence = ConfidencePredictor(
            feature_dim, hidden_dim=mlp_hidden_dim, dropout=dropout
        )
        self.vein_confidence = ConfidencePredictor(
            feature_dim, hidden_dim=mlp_hidden_dim, dropout=dropout
        )
        if embedding_dim == feature_dim:
            self.fusion_projection: nn.Module = nn.Identity()
        else:
            self.fusion_projection = nn.Sequential(
                nn.Linear(feature_dim, embedding_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        self.fused_classifier = nn.Linear(embedding_dim, num_classes)

        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.embedding_dim = int(embedding_dim)
        self.lambda_fused_cls = float(lambda_fused_cls)
        self.lambda_palm_cls = float(lambda_palm_cls)
        self.lambda_vein_cls = float(lambda_vein_cls)
        self.lambda_confidence = float(lambda_confidence)
        self.confidence_epsilon = float(confidence_epsilon)
        self._initialize_mlp_weights()

    def _initialize_mlp_weights(self) -> None:
        backbone_ids = {
            id(module)
            for encoder in (self.palm_encoder, self.vein_encoder)
            for module in encoder.modules()
        }
        for module in self.modules():
            if id(module) in backbone_ids:
                continue
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def recognition_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.parameters()

    @staticmethod
    def _validate_features(
        palm: torch.Tensor, vein: torch.Tensor
    ) -> None:
        if palm.ndim != 2 or vein.ndim != 2 or palm.shape != vein.shape:
            raise ValueError(
                "palm and vein features must have the same "
                "[batch, channels] shape"
            )
        if not palm.is_floating_point() or not vein.is_floating_point():
            raise TypeError(
                "palm and vein features must be floating-point tensors"
            )
        if palm.device != vein.device:
            raise ValueError(
                "palm and vein features must be on the same device"
            )

    def _importance_outputs(
        self,
        palm_features: torch.Tensor,
        vein_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        palm_logits = self.palm_importance(palm_features)
        vein_logits = self.vein_importance(vein_features)
        palm_class_probabilities = torch.sigmoid(palm_logits)
        vein_class_probabilities = torch.sigmoid(vein_logits)
        palm_confidence = self.palm_confidence(palm_features)
        vein_confidence = self.vein_confidence(vein_features)
        confidence = torch.stack(
            [palm_confidence, vein_confidence], dim=1
        )
        probabilities = confidence.clamp_min(self.confidence_epsilon)
        probabilities = probabilities / probabilities.sum(
            dim=1, keepdim=True
        )
        return {
            "palm_logits": palm_logits,
            "vein_logits": vein_logits,
            "palm_class_probabilities": palm_class_probabilities,
            "vein_class_probabilities": vein_class_probabilities,
            "palm_max_probability": palm_class_probabilities.max(
                dim=1
            ).values,
            "vein_max_probability": vein_class_probabilities.max(
                dim=1
            ).values,
            "palm_confidence": palm_confidence,
            "vein_confidence": vein_confidence,
            "modality_probabilities": probabilities,
        }

    def sparse_fuse(
        self,
        palm_features: torch.Tensor,
        vein_features: torch.Tensor,
        probabilities: torch.Tensor,
        *,
        stochastic: bool,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse features and return the [B, 2, C] mask."""

        self._validate_features(palm_features, vein_features)
        if palm_features.size(1) != self.feature_dim:
            raise ValueError(
                f"feature dimension must be {self.feature_dim}, "
                f"got {palm_features.size(1)}"
            )
        if probabilities.shape != (palm_features.size(0), 2):
            raise ValueError("probabilities must have shape [batch, 2]")
        if (
            not torch.isfinite(probabilities).all()
            or (probabilities < 0).any()
        ):
            raise ValueError(
                "probabilities must be finite and non-negative"
            )
        probabilities = probabilities / probabilities.sum(
            dim=1, keepdim=True
        ).clamp_min(self.confidence_epsilon)
        stacked = torch.stack([palm_features, vein_features], dim=1)
        soft_mask = probabilities.unsqueeze(-1).expand(
            -1, -1, self.feature_dim
        )
        if not stochastic:
            return (stacked * soft_mask).sum(dim=1), soft_mask

        sampled_modalities = torch.multinomial(
            probabilities,
            num_samples=self.feature_dim,
            replacement=True,
            generator=generator,
        )
        hard_mask = F.one_hot(
            sampled_modalities, num_classes=2
        ).permute(0, 2, 1)
        hard_mask = hard_mask.to(dtype=palm_features.dtype)
        # The published multinomial draw is discrete.  Its probabilities are
        # trained by the modality classification and confidence-reconstruction
        # objectives rather than an unreported straight-through estimator.
        return (stacked * hard_mask).sum(dim=1), hard_mask

    def forward_from_features(
        self,
        palm_features: torch.Tensor,
        vein_features: torch.Tensor,
        *,
        stochastic: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        self._validate_features(palm_features, vein_features)
        if palm_features.size(1) != self.feature_dim:
            raise ValueError(
                f"feature dimension must be {self.feature_dim}, "
                f"got {palm_features.size(1)}"
            )
        importance = self._importance_outputs(
            palm_features, vein_features
        )
        use_sampling = self.training if stochastic is None else bool(stochastic)
        fused_features, sparse_mask = self.sparse_fuse(
            palm_features,
            vein_features,
            importance["modality_probabilities"],
            stochastic=use_sampling,
            generator=generator,
        )
        embedding = self.fusion_projection(fused_features)
        return {
            **importance,
            "palm_features": palm_features,
            "vein_features": vein_features,
            "fused_features": fused_features,
            "embedding": embedding,
            "normalized_embedding": F.normalize(embedding, dim=1),
            "fused_logits": self.fused_classifier(embedding),
            "sparse_mask": sparse_mask,
        }

    def forward(
        self,
        palm_images: torch.Tensor,
        vein_images: torch.Tensor,
        *,
        stochastic: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        if palm_images.shape != vein_images.shape:
            raise ValueError(
                "palm_images and vein_images must have the same shape"
            )
        return self.forward_from_features(
            self.palm_encoder(palm_images),
            self.vein_encoder(vein_images),
            stochastic=stochastic,
            generator=generator,
        )

    def loss_dict(
        self,
        palm_images: torch.Tensor,
        vein_images: torch.Tensor,
        labels: torch.Tensor,
        *,
        stochastic: bool = True,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        output = self(
            palm_images,
            vein_images,
            stochastic=stochastic,
            generator=generator,
        )
        if labels.ndim != 1 or labels.size(0) != palm_images.size(0):
            raise ValueError("labels must have shape [batch]")
        labels = labels.to(device=palm_images.device, dtype=torch.long)
        if (
            labels.numel()
            and (labels.min() < 0 or labels.max() >= self.num_classes)
        ):
            raise ValueError(
                "labels are outside the configured class range"
            )
        fused_cls = F.cross_entropy(output["fused_logits"], labels)
        palm_cls = F.cross_entropy(output["palm_logits"], labels)
        vein_cls = F.cross_entropy(output["vein_logits"], labels)
        palm_true_probability = output[
            "palm_class_probabilities"
        ].gather(1, labels[:, None]).squeeze(1)
        vein_true_probability = output[
            "vein_class_probabilities"
        ].gather(1, labels[:, None]).squeeze(1)
        confidence = F.mse_loss(
            output["palm_confidence"],
            palm_true_probability.detach(),
        )
        confidence = confidence + F.mse_loss(
            output["vein_confidence"],
            vein_true_probability.detach(),
        )
        total = (
            self.lambda_fused_cls * fused_cls
            + self.lambda_palm_cls * palm_cls
            + self.lambda_vein_cls * vein_cls
            + self.lambda_confidence * confidence
        )
        return {
            "total": total,
            "fused_cls": fused_cls,
            "palm_cls": palm_cls,
            "vein_cls": vein_cls,
            "confidence": confidence,
        }


class HCMIGAdapter(nn.Module):
    """Hierarchical bidirectional texture and structure image generation.

    Args:
        base_channels: Width of every generator/discriminator.  The paper
            configuration and training default is 64.  A smaller width may be
            supplied by unit tests without changing the nine-block topology.
        fft_radius_ratio: Radius of the centered Fourier low-pass mask relative
            to the shorter spatial side.
        num_classes: Training identities for the MDSFF classifiers.
        recognition_embedding_size: Embedding width of the two pretrained
            ResNet-18 recognition encoders.
    """

    VALID_MODALITIES = frozenset({"palm", "palmprint", "vein", "palmvein"})

    def __init__(
        self,
        base_channels: int = 64,
        fft_radius_ratio: float = 0.1,
        *,
        num_classes: int = 600,
        recognition_embedding_size: int = 256,
        recognition_embedding_dim: int | None = None,
        recognition_hidden_dim: int | None = None,
        recognition_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if not 0.0 < fft_radius_ratio < 0.5:
            raise ValueError("fft_radius_ratio must be in (0, 0.5)")

        generator_kwargs = {"input_channels": 3, "base_channels": base_channels}
        discriminator_kwargs = {
            "input_channels": 3,
            "base_channels": base_channels,
        }
        self.texture_vein_to_palm = ResNet9Generator(**generator_kwargs)
        self.texture_palm_to_vein = ResNet9Generator(**generator_kwargs)
        self.structure_vein_to_palm = ResNet9Generator(**generator_kwargs)
        self.structure_palm_to_vein = ResNet9Generator(**generator_kwargs)
        self.palm_discriminator = PatchGAN70Discriminator(**discriminator_kwargs)
        self.vein_discriminator = PatchGAN70Discriminator(**discriminator_kwargs)
        for module in self.generation_modules():
            module.apply(_normal_init)

        self.mdsff = MDSFF(
            num_classes=num_classes,
            embedding_size=recognition_embedding_size,
            embedding_dim=recognition_embedding_dim,
            mlp_hidden_dim=recognition_hidden_dim,
            dropout=recognition_dropout,
            lambda_fused_cls=1.0,
            lambda_palm_cls=0.5,
            lambda_vein_cls=0.5,
            lambda_confidence=0.1,
        )

        self.fft_radius_ratio = float(fft_radius_ratio)
        self.lambda_cycle = 1.0
        self.lambda_adversarial = 1.0
        self.lambda_cms = 1.0
        self.lambda_fourier = 0.1
        self.register_buffer(
            "imagenet_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=False,
        )

    @property
    def texture_v2p(self) -> ResNet9Generator:
        return self.texture_vein_to_palm

    @property
    def texture_p2v(self) -> ResNet9Generator:
        return self.texture_palm_to_vein

    @property
    def structure_v2p(self) -> ResNet9Generator:
        return self.structure_vein_to_palm

    @property
    def structure_p2v(self) -> ResNet9Generator:
        return self.structure_palm_to_vein

    @property
    def palm_encoder(self) -> ResNet18Encoder:
        return self.mdsff.palm_encoder

    @property
    def vein_encoder(self) -> ResNet18Encoder:
        return self.mdsff.vein_encoder

    def generation_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.texture_vein_to_palm,
            self.texture_palm_to_vein,
            self.structure_vein_to_palm,
            self.structure_palm_to_vein,
            self.palm_discriminator,
            self.vein_discriminator,
        )

    def generator_parameters(self) -> Iterator[nn.Parameter]:
        """Iterate over only the four generator parameter sets."""

        for module in (
            self.texture_vein_to_palm,
            self.texture_palm_to_vein,
            self.structure_vein_to_palm,
            self.structure_palm_to_vein,
        ):
            yield from module.parameters()

    def discriminator_parameters(self) -> Iterator[nn.Parameter]:
        """Iterate over only the two discriminator parameter sets."""

        yield from self.palm_discriminator.parameters()
        yield from self.vein_discriminator.parameters()

    def recognition_parameters(self) -> Iterator[nn.Parameter]:
        """Iterate over all stage-2 MDSFF parameters."""

        yield from self.mdsff.parameters()

    def set_training_stage(
        self,
        stage: Literal["generation", "recognition", "all"],
    ) -> None:
        """Freeze the unused stage to avoid needless gradients and state."""

        if stage not in {"generation", "recognition", "all"}:
            raise ValueError(
                "stage must be 'generation', 'recognition', or 'all'"
            )
        generation_trainable = stage in {"generation", "all"}
        recognition_trainable = stage in {"recognition", "all"}
        for module in self.generation_modules():
            module.requires_grad_(generation_trainable)
        self.mdsff.requires_grad_(recognition_trainable)

    @staticmethod
    def _validate_images(images: torch.Tensor, name: str) -> None:
        if images.ndim != 4 or images.size(1) != 3:
            raise ValueError(f"{name} must have shape [batch, 3, height, width]")
        if images.size(-2) < 32 or images.size(-1) < 32:
            raise ValueError(f"{name} spatial dimensions must both be at least 32")
        if images.size(-2) % 4 or images.size(-1) % 4:
            raise ValueError(f"{name} spatial dimensions must be divisible by 4")
        if not images.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")

    @classmethod
    def _validate_pair(
        cls, palm_domain: torch.Tensor, vein_domain: torch.Tensor
    ) -> None:
        cls._validate_images(palm_domain, "palm_domain")
        cls._validate_images(vein_domain, "vein_domain")
        if palm_domain.shape != vein_domain.shape:
            raise ValueError("palm_domain and vein_domain must have the same shape")
        if palm_domain.device != vein_domain.device:
            raise ValueError("palm_domain and vein_domain must be on the same device")

    def generate_missing(
        self, source: torch.Tensor, available_modality: str
    ) -> torch.Tensor:
        """Generate the absent modality from the available image.

        ``available_modality='palm'`` returns a generated palm-vein image;
        ``available_modality='vein'`` returns a generated palmprint image.
        ``palmprint`` and ``palmvein`` are accepted as explicit synonyms.
        """

        self._validate_images(source, "source")
        modality = available_modality.strip().lower()
        if modality not in self.VALID_MODALITIES:
            supported = ", ".join(sorted(self.VALID_MODALITIES))
            raise ValueError(
                f"Unsupported available_modality={available_modality!r}; "
                f"expected one of {supported}"
            )
        if modality in {"palm", "palmprint"}:
            textured = self.texture_palm_to_vein(source)
            return self.structure_palm_to_vein(textured)
        textured = self.texture_vein_to_palm(source)
        return self.structure_vein_to_palm(textured)

    def radial_frequency_masks(
        self,
        height: int,
        width: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return complementary centered low/high FFT masks of shape [1,1,H,W]."""

        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        if not (dtype.is_floating_point or dtype.is_complex):
            raise TypeError("frequency mask dtype must be floating point")
        mask_dtype = torch.float32 if dtype.is_complex else dtype
        rows = torch.arange(height, device=device, dtype=mask_dtype) - height // 2
        columns = torch.arange(width, device=device, dtype=mask_dtype) - width // 2
        distance = torch.sqrt(rows[:, None].square() + columns[None, :].square())
        radius = self.fft_radius_ratio * min(height, width)
        low = (distance <= radius).to(dtype=mask_dtype)[None, None]
        high = torch.ones_like(low) - low
        return low, high

    @staticmethod
    def _masked_complex_l1(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean L1 over selected real and imaginary Fourier coefficients."""

        difference = prediction - target
        selected = mask.sum().clamp_min(1.0)
        normalizer = selected * difference.size(0) * difference.size(1)
        real_l1 = (difference.real.abs() * mask).sum() / normalizer
        imaginary_l1 = (difference.imag.abs() * mask).sum() / normalizer
        return 0.5 * (real_l1 + imaginary_l1)

    def _fourier_losses(
        self,
        generated_palm: torch.Tensor,
        palm_domain: torch.Tensor,
        generated_vein: torch.Tensor,
        vein_domain: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # cuFFT does not support every spatial size for half precision.  The
        # explicit float conversion also keeps frequency losses numerically
        # stable under automatic mixed precision while retaining gradients.
        images = (generated_palm, palm_domain, generated_vein, vein_domain)
        spectra = [
            torch.fft.fftshift(
                torch.fft.fft2(
                    image.float() if image.dtype in {torch.float16, torch.bfloat16} else image,
                    norm="ortho",
                ),
                dim=(-2, -1),
            )
            for image in images
        ]
        low, high = self.radial_frequency_masks(
            generated_palm.size(-2),
            generated_palm.size(-1),
            device=generated_palm.device,
            dtype=spectra[0].real.dtype,
        )
        generated_palm_fft, palm_fft, generated_vein_fft, vein_fft = spectra

        # Equations (8)-(9) are directional: L_S supervises generated
        # palmprint high frequencies, while L_T supervises generated palmvein
        # low frequencies.  Applying both bands to both directions changes the
        # published objective and was the main error in the previous adapter.
        structure = self._masked_complex_l1(
            generated_palm_fft, palm_fft, high
        )
        texture = self._masked_complex_l1(
            generated_vein_fft, vein_fft, low
        )
        return structure, texture

    @staticmethod
    def _real_target(logits: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(logits)

    @staticmethod
    def _fake_target(logits: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(logits)

    def generator_loss_dict(
        self, palm_domain: torch.Tensor, vein_domain: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute all bidirectional HCMIG generator losses.

        The adversarial term supervises the intermediate texture-transferred
        images as in the paper.  CMS and both high/low Fourier terms supervise
        the final structure-generated images against paired targets.
        """

        self._validate_pair(palm_domain, vein_domain)

        palm_texture = self.texture_vein_to_palm(vein_domain)
        vein_texture = self.texture_palm_to_vein(palm_domain)
        reconstructed_vein = self.texture_palm_to_vein(palm_texture)
        reconstructed_palm = self.texture_vein_to_palm(vein_texture)
        generated_palm = self.structure_vein_to_palm(palm_texture)
        generated_vein = self.structure_palm_to_vein(vein_texture)

        cycle = F.l1_loss(reconstructed_palm, palm_domain)
        cycle = cycle + F.l1_loss(reconstructed_vein, vein_domain)

        palm_logits = self.palm_discriminator(palm_texture)
        vein_logits = self.vein_discriminator(vein_texture)
        adversarial = F.binary_cross_entropy_with_logits(
            palm_logits, self._real_target(palm_logits)
        )
        adversarial = adversarial + F.binary_cross_entropy_with_logits(
            vein_logits, self._real_target(vein_logits)
        )

        cms = F.l1_loss(generated_palm, palm_domain)
        cms = cms + F.l1_loss(generated_vein, vein_domain)
        fourier_structure, fourier_texture = self._fourier_losses(
            generated_palm,
            palm_domain,
            generated_vein,
            vein_domain,
        )
        fourier = fourier_structure + fourier_texture
        total = (
            self.lambda_cycle * cycle
            + self.lambda_adversarial * adversarial
            + self.lambda_cms * cms
            + self.lambda_fourier * fourier
        )
        return {
            "total": total,
            "cycle": cycle,
            "adversarial": adversarial,
            "cms": cms,
            "fourier": fourier,
            "fourier_structure": fourier_structure,
            "fourier_texture": fourier_texture,
        }

    def discriminator_loss_dict(
        self, palm_domain: torch.Tensor, vein_domain: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute detached-fake BCEWithLogits losses for both PatchGANs."""

        self._validate_pair(palm_domain, vein_domain)
        with torch.no_grad():
            fake_palm = self.texture_vein_to_palm(vein_domain)
            fake_vein = self.texture_palm_to_vein(palm_domain)

        palm_real_logits = self.palm_discriminator(palm_domain)
        palm_fake_logits = self.palm_discriminator(fake_palm.detach())
        vein_real_logits = self.vein_discriminator(vein_domain)
        vein_fake_logits = self.vein_discriminator(fake_vein.detach())

        palm_real = F.binary_cross_entropy_with_logits(
            palm_real_logits, self._real_target(palm_real_logits)
        )
        palm_fake = F.binary_cross_entropy_with_logits(
            palm_fake_logits, self._fake_target(palm_fake_logits)
        )
        vein_real = F.binary_cross_entropy_with_logits(
            vein_real_logits, self._real_target(vein_real_logits)
        )
        vein_fake = F.binary_cross_entropy_with_logits(
            vein_fake_logits, self._fake_target(vein_fake_logits)
        )
        palm = 0.5 * (palm_real + palm_fake)
        vein = 0.5 * (vein_real + vein_fake)
        total = palm + vein
        return {
            "total": total,
            "palm": palm,
            "vein": vein,
            "palm_real": palm_real,
            "palm_fake": palm_fake,
            "vein_real": vein_real,
            "vein_fake": vein_fake,
        }


    def domain_to_recognition_input(
        self, images: torch.Tensor
    ) -> torch.Tensor:
        """Convert generation-domain images to VGG/ImageNet normalization."""

        self._validate_images(images, "images")
        zero_to_one = (images.clamp(-1.0, 1.0) + 1.0) * 0.5
        return (zero_to_one - self.imagenet_mean) / self.imagenet_std

    def recognition_forward(
        self,
        palm_domain: torch.Tensor,
        vein_domain: torch.Tensor,
        *,
        stochastic: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run MDSFF for a complete pair in the generation image domain."""

        self._validate_pair(palm_domain, vein_domain)
        return self.mdsff(
            self.domain_to_recognition_input(palm_domain),
            self.domain_to_recognition_input(vein_domain),
            stochastic=stochastic,
            generator=generator,
        )

    def recognition_loss_dict(
        self,
        palm_domain: torch.Tensor,
        vein_domain: torch.Tensor,
        labels: torch.Tensor,
        *,
        stochastic: bool = True,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the MDSFF objective in equation (21)."""

        self._validate_pair(palm_domain, vein_domain)
        return self.mdsff.loss_dict(
            self.domain_to_recognition_input(palm_domain),
            self.domain_to_recognition_input(vein_domain),
            labels,
            stochastic=stochastic,
            generator=generator,
        )

    def recognize(
        self,
        *,
        palm_domain: torch.Tensor | None = None,
        vein_domain: torch.Tensor | None = None,
        stochastic: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Recognize complete or one-missing-modality samples end-to-end."""

        if palm_domain is None and vein_domain is None:
            raise ValueError("at least one modality must be supplied")
        generated_modality = 0
        if palm_domain is None:
            assert vein_domain is not None
            palm_domain = self.generate_missing(vein_domain, "vein")
            generated_modality = 1
        elif vein_domain is None:
            vein_domain = self.generate_missing(palm_domain, "palm")
            generated_modality = 2
        output = self.recognition_forward(
            palm_domain,
            vein_domain,
            stochastic=stochastic,
            generator=generator,
        )
        output["palm_domain"] = palm_domain
        output["vein_domain"] = vein_domain
        output["generated_modality"] = palm_domain.new_tensor(
            generated_modality, dtype=torch.int64
        )
        return output

    def forward(
        self,
        palm_domain: torch.Tensor | None = None,
        vein_domain: torch.Tensor | None = None,
        *,
        stochastic: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.recognize(
            palm_domain=palm_domain,
            vein_domain=vein_domain,
            stochastic=stochastic,
            generator=generator,
        )

# Short method name retained for the common comparison registry.
HCMIG = HCMIGAdapter
