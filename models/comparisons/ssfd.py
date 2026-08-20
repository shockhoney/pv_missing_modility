"""SSFD-Net for palmprint/palmvein recognition with a missing modality.

The implementation follows equations (1)--(16) of Pan et al., *Digital
Signal Processing* 159 (2025) 105003. The paper does not provide a public
source repository, so ambiguous details are explicit configuration choices.

``ResNetSSFDNet`` is the image-level reproduction with the project's
pretrained palm/vein ResNet-18 encoders (replacing the paper's from-scratch
VGG16 Dp/Dv teachers, which are unavailable for the locked protocol). The
frozen per-modality ArcFace identity heads trained with ``train_encoder.py``
serve as Dp/Dv for the identity-consistency objectives (5)--(11).
``SSFDAdapter`` is retained only for controlled experiments that
intentionally fix external encoders. The fused feature order is always
``[palm_shared, palm_specific, vein_shared, vein_specific]``.

Paper losses operate on raw features and logits. Deployment representations
are unit-normalized by default for the project's cosine-matching protocol.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import ResNet18Encoder


METHOD_NAME: Final = "SSFD-Net"
ARCHITECTURE_VERSION: Final = "ssfd_paper_resnet_v3"

Classifier = Callable[[torch.Tensor], torch.Tensor]
FeatureParts = dict[str, torch.Tensor]


class CrossModalFeatureTransformation(nn.Module):
    """One CMFT MLP: two 2048-wide hidden layers in the paper setup."""

    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 2048,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, specific: torch.Tensor) -> torch.Tensor:
        if specific.ndim != 2 or specific.size(1) != self.feature_dim:
            raise ValueError(
                f"specific must have shape [batch, {self.feature_dim}], "
                f"got {tuple(specific.shape)}"
            )
        # Equation (9) uses raw Euclidean distance, so do not normalize here.
        return self.network(specific)


class BidirectionalCrossModalFeatureTransformation(nn.Module):
    """Apply CMFT in both directions.

    Equation (9) uses one ``CMFT`` symbol for both directions, hence shared
    weights are the faithful default. Independent directed mappings remain an
    explicit ablation through ``share_weights=False``.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 2048,
        dropout: float = 0.5,
        share_weights: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.share_weights = bool(share_weights)
        if self.share_weights:
            self.shared_transform = CrossModalFeatureTransformation(
                feature_dim, hidden_dim, dropout
            )
            self.palm_to_vein_transform = None
            self.vein_to_palm_transform = None
        else:
            self.shared_transform = None
            self.palm_to_vein_transform = CrossModalFeatureTransformation(
                feature_dim, hidden_dim, dropout
            )
            self.vein_to_palm_transform = CrossModalFeatureTransformation(
                feature_dim, hidden_dim, dropout
            )

    def palm_from_vein(self, vein_specific: torch.Tensor) -> torch.Tensor:
        module = self.shared_transform if self.share_weights else self.vein_to_palm_transform
        assert module is not None
        return module(vein_specific)

    def vein_from_palm(self, palm_specific: torch.Tensor) -> torch.Tensor:
        module = self.shared_transform if self.share_weights else self.palm_to_vein_transform
        assert module is not None
        return module(palm_specific)

    def forward(
        self,
        palm_specific: torch.Tensor,
        vein_specific: torch.Tensor,
    ) -> FeatureParts:
        return {
            "palm_from_vein": self.palm_from_vein(vein_specific),
            "vein_from_palm": self.vein_from_palm(palm_specific),
        }


class ResNetSharedSpecificEncoder(nn.Module):
    """Shared/specific part encoder backed by a project ResNet-18 encoder.

    The project encoder maps an image to a ``embedding_size``-wide pooled
    embedding; its ``shared_head`` and ``specific_head`` each project one half
    of that embedding, yielding two ``embedding_size // 2`` parts as required
    by SSFD equations (3)--(4). The backbone weights come from the project's
    per-modality identity checkpoints, so no Stage-A teacher pretraining is
    needed.
    """

    def __init__(
        self,
        embedding_size: int = 256,
        *,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        if embedding_size <= 0 or embedding_size % 2:
            raise ValueError("embedding_size must be positive and even")
        self.embedding_size = int(embedding_size)
        self.feature_dim = self.embedding_size // 2
        self.backbone = ResNet18Encoder(
            input_channel=3,
            embedding_size=self.embedding_size,
            use_se=use_se,
        )

    def load_encoder_state(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        strict: bool = True,
    ) -> None:
        self.backbone.load_state_dict(state_dict, strict=strict)

    def forward(self, images: torch.Tensor) -> FeatureParts:
        if images.ndim != 4 or images.size(1) != 3:
            raise ValueError(
                "images must have shape [batch, 3, height, width], "
                f"got {tuple(images.shape)}"
            )
        if min(images.shape[-2:]) < 32:
            raise ValueError("ResNet-18 inputs must be at least 32x32")
        if not images.is_floating_point():
            raise TypeError("images must be floating-point tensors")
        shared, specific = self.backbone.parts(images)
        return {"shared": shared, "specific": specific}


class _SSFDBase(nn.Module):
    """Shared equations and deployment paths for image and feature models."""

    LOSS_WEIGHTS: Final = {
        "classification": 1.0,
        "triplet": 0.3,
        "transformation": 0.3,
        "inter_consistency": 0.3,
        "intra_consistency": 0.3,
    }
    _PALM_ALIASES: Final = frozenset({"palm", "palmprint", "p"})
    _VEIN_ALIASES: Final = frozenset({"vein", "palmvein", "v"})

    def __init__(
        self,
        *,
        feature_dim: int,
        num_classes: int,
        cmft_hidden_dim: int,
        dropout: float,
        triplet_margin: float,
        palm_classifier: Classifier | None,
        vein_classifier: Classifier | None,
        share_cmft_weights: bool,
        allow_identity_fallback: bool,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or num_classes <= 1:
            raise ValueError("feature_dim must be positive and num_classes must exceed one")
        if triplet_margin < 0.0:
            raise ValueError("triplet_margin must be non-negative")
        self.feature_dim = int(feature_dim)
        self.representation_dim = 4 * self.feature_dim
        self.num_classes = int(num_classes)
        self.triplet_margin = float(triplet_margin)
        self.allow_identity_fallback = bool(allow_identity_fallback)
        self.cmft = BidirectionalCrossModalFeatureTransformation(
            feature_dim=self.feature_dim,
            hidden_dim=cmft_hidden_dim,
            dropout=dropout,
            share_weights=share_cmft_weights,
        )
        self.identity_classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.representation_dim, self.num_classes)
        )
        nn.init.xavier_uniform_(self.identity_classifier[-1].weight)
        nn.init.zeros_(self.identity_classifier[-1].bias)

        # Caller-owned Dp/Dv are assigned after internal initialization.
        self.palm_classifier = palm_classifier
        self.vein_classifier = vein_classifier
        self.freeze_identity_classifiers()

    @staticmethod
    def _freeze_classifier(classifier: Classifier | None) -> None:
        if isinstance(classifier, nn.Module):
            classifier.requires_grad_(False)
            classifier.eval()

    def freeze_identity_classifiers(self) -> None:
        self._freeze_classifier(self.palm_classifier)
        self._freeze_classifier(self.vein_classifier)

    def train(self, mode: bool = True) -> "_SSFDBase":
        super().train(mode)
        self.freeze_identity_classifiers()
        return self

    @classmethod
    def _canonical_modality(cls, modality: str) -> str:
        normalized = str(modality).strip().lower()
        if normalized in cls._PALM_ALIASES:
            return "palm"
        if normalized in cls._VEIN_ALIASES:
            return "vein"
        raise ValueError(f"Unsupported modality: {modality!r}")

    @classmethod
    def _available_modality(cls, modality: str) -> str:
        normalized = str(modality).strip().lower().replace("-", "_")
        if normalized in {"pm", "palm_missing", "palmprint_missing"}:
            return "vein"
        if normalized in {"vm", "vein_missing", "palmvein_missing"}:
            return "palm"
        if normalized.endswith("_available"):
            normalized = normalized.removesuffix("_available")
        return cls._canonical_modality(normalized)

    def encode_parts(self, value: torch.Tensor, modality: str) -> FeatureParts:
        raise NotImplementedError

    def _validate_parts(self, parts: Mapping[str, torch.Tensor], name: str) -> None:
        if set(parts) != {"shared", "specific"}:
            raise ValueError(f"{name} encoder must return shared and specific features")
        shared, specific = parts["shared"], parts["specific"]
        if shared.ndim != 2 or shared.size(1) != self.feature_dim:
            raise ValueError(f"{name} shared feature must have shape [batch, {self.feature_dim}]")
        if specific.ndim != 2 or tuple(specific.shape) != tuple(shared.shape):
            raise ValueError(f"{name} specific feature must have shape [batch, {self.feature_dim}]")
        if not shared.is_floating_point() or not specific.is_floating_point():
            raise TypeError(f"{name} features must be floating-point tensors")

    @staticmethod
    def _join(parts: tuple[torch.Tensor, ...], *, normalize: bool) -> torch.Tensor:
        representation = torch.cat(parts, dim=1)
        return F.normalize(representation, dim=1) if normalize else representation

    def _complete_from_parts(
        self,
        palm_parts: Mapping[str, torch.Tensor],
        vein_parts: Mapping[str, torch.Tensor],
        *,
        normalize: bool,
    ) -> torch.Tensor:
        return self._join(
            (
                palm_parts["shared"], palm_parts["specific"],
                vein_parts["shared"], vein_parts["specific"],
            ),
            normalize=normalize,
        )

    def complete_representation(
        self,
        palm: torch.Tensor,
        vein: torch.Tensor,
        *,
        normalize: bool = True,
    ) -> torch.Tensor:
        palm_parts = self.encode_parts(palm, "palm")
        vein_parts = self.encode_parts(vein, "vein")
        if palm_parts["shared"].size(0) != vein_parts["shared"].size(0):
            raise ValueError("palm and vein batch sizes must match")
        return self._complete_from_parts(palm_parts, vein_parts, normalize=normalize)

    def missing_representation(
        self,
        available: torch.Tensor,
        modality: str,
        *,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Reconstruct the absent modality according to equations (14)--(16)."""
        available_modality = self._available_modality(modality)
        parts = self.encode_parts(available, available_modality)
        if available_modality == "palm":
            vein_specific = self.cmft.vein_from_palm(parts["specific"])
            return self._join(
                (parts["shared"], parts["specific"], parts["shared"], vein_specific),
                normalize=normalize,
            )
        palm_specific = self.cmft.palm_from_vein(parts["specific"])
        return self._join(
            (parts["shared"], palm_specific, parts["shared"], parts["specific"]),
            normalize=normalize,
        )

    def classification_logits(
        self,
        palm: torch.Tensor | None = None,
        vein: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if palm is not None and vein is not None:
            representation = self.complete_representation(palm, vein, normalize=False)
        elif palm is not None:
            representation = self.missing_representation(palm, "palm", normalize=False)
        elif vein is not None:
            representation = self.missing_representation(vein, "vein", normalize=False)
        else:
            raise ValueError("At least one modality must be provided")
        return self.identity_classifier(representation)

    def _identity_logits(self, feature: torch.Tensor, modality: str) -> torch.Tensor:
        canonical = self._canonical_modality(modality)
        classifier = self.palm_classifier if canonical == "palm" else self.vein_classifier
        if classifier is not None:
            logits = classifier(feature)
        elif self.allow_identity_fallback:
            zeros = torch.zeros_like(feature)
            padded = (
                torch.cat((feature, zeros), dim=1)
                if canonical == "palm"
                else torch.cat((zeros, feature), dim=1)
            )
            logits = self.identity_classifier(padded)
        else:
            raise RuntimeError(
                "SSFD identity-consistency losses require pre-trained frozen "
                f"{canonical} and counterpart identity classifiers"
            )
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise ValueError("Identity classifiers must return a 2-D logits tensor")
        # Equations (8) and (10) compare D outputs directly; no normalization.
        return logits

    @staticmethod
    def _batch_l2(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(left - right, ord=2, dim=1).mean()

    def loss_dict(
        self,
        palm: torch.Tensor,
        vein: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute equations (5)--(11) with the paper's loss weights."""
        palm_parts = self.encode_parts(palm, "palm")
        vein_parts = self.encode_parts(vein, "vein")
        batch = palm_parts["shared"].size(0)
        if vein_parts["shared"].size(0) != batch:
            raise ValueError("palm and vein batch sizes must match")
        if labels.ndim != 1 or labels.size(0) != batch:
            raise ValueError("labels must have shape [batch]")
        labels = labels.to(device=palm_parts["shared"].device, dtype=torch.long)

        palm_shared, palm_specific = palm_parts["shared"], palm_parts["specific"]
        vein_shared, vein_specific = vein_parts["shared"], vein_parts["specific"]
        complete = self._complete_from_parts(palm_parts, vein_parts, normalize=False)
        classification = F.cross_entropy(self.identity_classifier(complete), labels)

        shared_distance = torch.linalg.vector_norm(palm_shared - vein_shared, ord=2, dim=1)
        palm_negative = torch.linalg.vector_norm(palm_shared - palm_specific, ord=2, dim=1)
        vein_negative = torch.linalg.vector_norm(vein_shared - vein_specific, ord=2, dim=1)
        triplet = (
            F.relu(shared_distance - palm_negative + self.triplet_margin)
            + F.relu(shared_distance - vein_negative + self.triplet_margin)
        ).mean()

        translated = self.cmft(palm_specific, vein_specific)
        palm_from_vein = translated["palm_from_vein"]
        vein_from_palm = translated["vein_from_palm"]
        transformation = 0.5 * (
            self._batch_l2(palm_specific, palm_from_vein)
            + self._batch_l2(vein_specific, vein_from_palm)
        )

        # Eq. (8): swap shared features, retain the target-specific feature.
        inter_consistency = self._batch_l2(
            self._identity_logits(torch.cat((palm_shared, vein_specific), dim=1), "vein"),
            self._identity_logits(torch.cat((vein_shared, palm_specific), dim=1), "palm"),
        )

        # Eq. (10): real and translated specific features preserve identity.
        intra_consistency = self._batch_l2(
            self._identity_logits(torch.cat((palm_shared, palm_specific), dim=1), "palm"),
            self._identity_logits(torch.cat((palm_shared, palm_from_vein), dim=1), "palm"),
        ) + self._batch_l2(
            self._identity_logits(torch.cat((vein_shared, vein_specific), dim=1), "vein"),
            self._identity_logits(torch.cat((vein_shared, vein_from_palm), dim=1), "vein"),
        )

        losses = {
            "classification": classification,
            "triplet": triplet,
            "transformation": transformation,
            "inter_consistency": inter_consistency,
            "intra_consistency": intra_consistency,
        }
        losses["total"] = sum(
            self.LOSS_WEIGHTS[name] * value for name, value in losses.items()
        )
        return losses

    def forward(
        self,
        palm: torch.Tensor | None = None,
        vein: torch.Tensor | None = None,
        *,
        return_logits: bool = False,
        normalize: bool = True,
    ) -> torch.Tensor:
        if return_logits:
            return self.classification_logits(palm=palm, vein=vein)
        if palm is not None and vein is not None:
            return self.complete_representation(palm, vein, normalize=normalize)
        if palm is not None:
            return self.missing_representation(palm, "palm", normalize=normalize)
        if vein is not None:
            return self.missing_representation(vein, "vein", normalize=normalize)
        raise ValueError("At least one modality must be provided")


class ResNetSSFDNet(_SSFDBase):
    """Complete image-level SSFD-Net with two project ResNet-18 encoders.

    Dp and Dv accept concatenated ``[shared, specific]`` features of shape
    ``[B, embedding_size]``. Classification and deployment work without them,
    but ``loss_dict`` fails loudly if either is absent rather than silently
    replacing the paper's identity-consistency objectives with a surrogate.
    """

    def __init__(
        self,
        num_classes: int = 432,
        embedding_size: int = 256,
        cmft_hidden_dim: int = 512,
        dropout: float = 0.5,
        triplet_margin: float = 0.1,
        palm_classifier: Classifier | None = None,
        vein_classifier: Classifier | None = None,
        palm_encoder: nn.Module | None = None,
        vein_encoder: nn.Module | None = None,
        share_cmft_weights: bool = True,
    ) -> None:
        if embedding_size <= 0 or embedding_size % 2:
            raise ValueError("embedding_size must be positive and even")
        super().__init__(
            feature_dim=embedding_size // 2,
            num_classes=num_classes,
            cmft_hidden_dim=cmft_hidden_dim,
            dropout=dropout,
            triplet_margin=triplet_margin,
            palm_classifier=palm_classifier,
            vein_classifier=vein_classifier,
            share_cmft_weights=share_cmft_weights,
            allow_identity_fallback=False,
        )
        self.embedding_size = int(embedding_size)
        self.palm_encoder = palm_encoder or ResNetSharedSpecificEncoder(
            embedding_size=self.embedding_size, use_se=False
        )
        self.vein_encoder = vein_encoder or ResNetSharedSpecificEncoder(
            embedding_size=self.embedding_size, use_se=True
        )

    def encode_parts(self, images: torch.Tensor, modality: str) -> FeatureParts:
        canonical = self._canonical_modality(modality)
        encoder = self.palm_encoder if canonical == "palm" else self.vein_encoder
        parts = encoder(images)
        if not isinstance(parts, Mapping):
            raise ValueError("SSFD image encoders must return a feature mapping")
        result = {"shared": parts["shared"], "specific": parts["specific"]}
        self._validate_parts(result, canonical)
        return result


class SSFDAdapter(_SSFDBase):
    """Feature-level front end retained for controlled-backbone studies.

    This is not the paper's VGG16 reproduction. If Dp/Dv are omitted, the
    legacy multimodal-head fallback remains solely so existing cached-feature
    experiments stay runnable.
    """

    def __init__(
        self,
        input_dim: int = 256,
        part_dim: int = 128,
        num_classes: int = 432,
        cmft_hidden_dim: int = 2048,
        dropout: float = 0.5,
        triplet_margin: float = 0.1,
        palm_classifier: Classifier | None = None,
        vein_classifier: Classifier | None = None,
        share_cmft_weights: bool = True,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)
        super().__init__(
            feature_dim=part_dim,
            num_classes=num_classes,
            cmft_hidden_dim=cmft_hidden_dim,
            dropout=dropout,
            triplet_margin=triplet_margin,
            palm_classifier=palm_classifier,
            vein_classifier=vein_classifier,
            share_cmft_weights=share_cmft_weights,
            allow_identity_fallback=True,
        )
        self.part_dim = self.feature_dim
        self.palm_shared_head = self._part_head(dropout)
        self.palm_specific_head = self._part_head(dropout)
        self.vein_shared_head = self._part_head(dropout)
        self.vein_specific_head = self._part_head(dropout)
        self._reset_part_heads()

    def _part_head(self, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.input_dim, self.feature_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
        )

    def _reset_part_heads(self) -> None:
        for root in (
            self.palm_shared_head, self.palm_specific_head,
            self.vein_shared_head, self.vein_specific_head,
        ):
            linear = root[0]
            assert isinstance(linear, nn.Linear)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    def _validate_embedding(self, embedding: torch.Tensor, name: str) -> None:
        if embedding.ndim != 2 or embedding.size(1) != self.input_dim:
            raise ValueError(
                f"{name} must have shape [batch, {self.input_dim}], "
                f"got {tuple(embedding.shape)}"
            )
        if not embedding.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")

    def encode_parts(self, embedding: torch.Tensor, modality: str) -> FeatureParts:
        self._validate_embedding(embedding, "embedding")
        canonical = self._canonical_modality(modality)
        if canonical == "palm":
            shared = self.palm_shared_head(embedding)
            specific = self.palm_specific_head(embedding)
        else:
            shared = self.vein_shared_head(embedding)
            specific = self.vein_specific_head(embedding)
        result = {"shared": shared, "specific": specific}
        self._validate_parts(result, canonical)
        return result


__all__ = [
    "ARCHITECTURE_VERSION",
    "METHOD_NAME",
    "BidirectionalCrossModalFeatureTransformation",
    "CrossModalFeatureTransformation",
    "ResNetSSFDNet",
    "ResNetSharedSpecificEncoder",
    "SSFDAdapter",
]
