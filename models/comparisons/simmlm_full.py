"""Paper-faithful image-level SimMLM for palm biometric recognition.

The official ICCV 2025 implementation mixes modality-expert logits after a
learned mask-aware gate. It uses independent expert pretraining followed by
cooperative More-vs-Fewer (MoFe) training. The earlier comparison adapter in
``simmlm.py`` is retained only for backward compatibility with old checkpoints.
"""

from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import build_encoder
from models.comparisons.simmlm import _presence_mask


ARCHITECTURE_VERSION: Final = "simmlm_dmome_logit_image_v3"
OFFICIAL_COMMIT: Final = "c9be74ba913db08013b8e573ce6cbda25da9ea1a"


class ImageExpert(nn.Module):
    """One independently pre-trainable modality expert."""

    def __init__(self, modality: str, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = build_encoder(modality, embedding_size=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(images)
        return embedding, self.classifier(embedding)

    def forward_available(
        self, images: torch.Tensor, present: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the expert only for samples where its modality exists.

        DMoME excludes a missing expert rather than forwarding a zero image
        through it. Apart from wasted compute, doing the latter corrupts an
        expert's BatchNorm statistics during cooperative training.
        """

        if present.ndim != 1 or present.numel() != images.size(0):
            raise ValueError("present must contain one mask value per sample")
        present = present.to(device=images.device, dtype=torch.bool)
        embedding = images.new_zeros(images.size(0), self.encoder.bn.num_features)
        logits = images.new_zeros(images.size(0), self.classifier.out_features)
        if not torch.any(present):
            return embedding, logits

        selected = images[present]
        feature_map = self.encoder.forward_features(selected)
        pooled = self.encoder.global_pool(feature_map).flatten(1)
        # BatchNorm1d cannot estimate variance from the legal one-present-item
        # edge case. Reuse running statistics only for that case.
        if self.encoder.bn.training and pooled.size(0) == 1:
            selected_embedding = F.batch_norm(
                pooled,
                self.encoder.bn.running_mean,
                self.encoder.bn.running_var,
                self.encoder.bn.weight,
                self.encoder.bn.bias,
                training=False,
                momentum=0.0,
                eps=self.encoder.bn.eps,
            )
        else:
            selected_embedding = self.encoder.bn(pooled)
        selected_logits = self.classifier(selected_embedding)
        embedding[present] = selected_embedding
        logits[present] = selected_logits
        return embedding, logits


class RouterBranch(nn.Module):
    """Supplementary Table A1 two-layer CNN producing a 128-d feature."""

    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(64, output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feature = self.features(images).flatten(1)
        return F.relu(self.projection(feature), inplace=False)


class SimMLMImageModel(nn.Module):
    """Two-stage DMoME with the official logit-level mixture and MoFe loss.

    The paper does not define a verification embedding. For this repository's
    identity-disjoint gallery/probe protocol, :meth:`representation`
    concatenates the two gated expert embeddings in disjoint blocks. This
    biometric extension does not alter the paper's classification objective.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_classes: int = 432,
        router_feature_dim: int = 128,
        ranking_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or num_classes <= 1 or router_feature_dim <= 0:
            raise ValueError("Invalid SimMLM dimensions")
        if ranking_weight < 0:
            raise ValueError("ranking_weight must be non-negative")
        self.embedding_dim = int(embedding_dim)
        self.representation_dim = 2 * self.embedding_dim
        self.num_classes = int(num_classes)
        self.ranking_weight = float(ranking_weight)
        self.palm_expert = ImageExpert("palm", self.embedding_dim, self.num_classes)
        self.vein_expert = ImageExpert("vein", self.embedding_dim, self.num_classes)
        self.palm_router = RouterBranch(router_feature_dim)
        self.vein_router = RouterBranch(router_feature_dim)
        # Supplementary Table A1 concatenates the two 128-d features and maps
        # that 256-d vector directly to two gating values.
        self.router_head = nn.Linear(2 * router_feature_dim, 2)
        nn.init.xavier_uniform_(self.router_head.weight)
        nn.init.zeros_(self.router_head.bias)

    @staticmethod
    def _validate_images(images: torch.Tensor, name: str) -> None:
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError(f"{name} must have shape [B, 3, H, W]")
        if images.size(1) != 3:
            raise ValueError(f"{name} must have three channels")
        if not images.is_floating_point():
            raise TypeError(f"{name} must be floating point")

    def _prepare_images(
        self,
        palm: torch.Tensor | None,
        vein: torch.Tensor | None,
        palm_present: bool | torch.Tensor,
        vein_present: bool | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reference = palm if palm is not None else vein
        if reference is None:
            raise ValueError("At least one modality image is required")
        self._validate_images(reference, "reference")
        batch, device = reference.size(0), reference.device
        palm_mask = _presence_mask(
            palm_present, batch_size=batch, device=device, name="palm_present"
        )
        vein_mask = _presence_mask(
            vein_present, batch_size=batch, device=device, name="vein_present"
        )
        if torch.any(~palm_mask & ~vein_mask):
            raise ValueError("Every sample must contain at least one modality")
        if palm is None:
            if torch.any(palm_mask):
                raise ValueError("palm is required when palm_present is true")
            palm = torch.zeros_like(reference)
        if vein is None:
            if torch.any(vein_mask):
                raise ValueError("vein is required when vein_present is true")
            vein = torch.zeros_like(reference)
        self._validate_images(palm, "palm")
        self._validate_images(vein, "vein")
        if palm.shape != vein.shape or palm.device != vein.device:
            raise ValueError("Palm and vein images must share shape and device")
        palm = palm * palm_mask[:, None, None, None].to(palm.dtype)
        vein = vein * vein_mask[:, None, None, None].to(vein.dtype)
        return palm, vein, palm_mask, vein_mask

    def encode(
        self,
        palm: torch.Tensor | None,
        vein: torch.Tensor | None,
        palm_present: bool | torch.Tensor = True,
        vein_present: bool | torch.Tensor = True,
    ) -> dict[str, torch.Tensor]:
        palm, vein, palm_mask, vein_mask = self._prepare_images(
            palm, vein, palm_present, vein_present
        )
        palm_embedding, palm_logits = self.palm_expert.forward_available(palm, palm_mask)
        vein_embedding, vein_logits = self.vein_expert.forward_available(vein, vein_mask)
        masks = torch.stack((palm_mask, vein_mask), dim=1)
        router_input = torch.cat(
            (
                self.palm_router(palm),
                self.vein_router(vein),
            ),
            dim=1,
        )
        router_logits = self.router_head(router_input).masked_fill(
            ~masks, float("-inf")
        )
        weights = F.softmax(router_logits, dim=1)
        logits = weights[:, :1] * palm_logits + weights[:, 1:] * vein_logits
        representation = F.normalize(
            torch.cat(
                (weights[:, :1] * palm_embedding, weights[:, 1:] * vein_embedding),
                dim=1,
            ),
            dim=1,
        )
        return {
            "representation": representation,
            "logits": logits,
            "router_weights": weights,
            "palm_embedding": palm_embedding,
            "vein_embedding": vein_embedding,
            "palm_logits": palm_logits,
            "vein_logits": vein_logits,
            "palm_present": palm_mask,
            "vein_present": vein_mask,
        }

    def representation(
        self,
        palm: torch.Tensor | None,
        vein: torch.Tensor | None,
        palm_present: bool | torch.Tensor = True,
        vein_present: bool | torch.Tensor = True,
    ) -> torch.Tensor:
        return self.encode(palm, vein, palm_present, vein_present)["representation"]

    def expert_loss(
        self, modality: str, images: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if modality == "palm":
            expert = self.palm_expert
        elif modality == "vein":
            expert = self.vein_expert
        else:
            raise ValueError("modality must be palm or vein")
        embedding, logits = expert(images)
        loss = F.cross_entropy(logits, labels.long())
        return {"embedding": embedding, "logits": logits, "total": loss}

    def cooperative_loss(
        self,
        palm: torch.Tensor,
        vein: torch.Tensor,
        labels: torch.Tensor,
        fewer_is_palm: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Apply official Eq. (3)/(4) to one sampled pair per sample."""

        labels = labels.long().reshape(-1)
        if fewer_is_palm.ndim != 1 or fewer_is_palm.numel() != labels.numel():
            raise ValueError("fewer_is_palm must contain one boolean per sample")
        more = self.encode(palm, vein, True, True)
        fewer_is_palm = fewer_is_palm.to(device=labels.device, dtype=torch.bool)
        fewer = self.encode(palm, vein, fewer_is_palm, ~fewer_is_palm)
        more_each = F.cross_entropy(more["logits"], labels, reduction="none")
        fewer_each = F.cross_entropy(fewer["logits"], labels, reduction="none")
        task_more = more_each.mean()
        task_fewer = fewer_each.mean()
        mofe = F.relu(more_each - fewer_each).mean()
        total = task_more + task_fewer + self.ranking_weight * mofe
        return {
            "task_more": task_more,
            "task_fewer": task_fewer,
            "mofe": mofe,
            "total": total,
            "more_logits": more["logits"],
            "fewer_logits": fewer["logits"],
        }

    def forward(
        self,
        palm: torch.Tensor | None,
        vein: torch.Tensor | None,
        palm_present: bool | torch.Tensor = True,
        vein_present: bool | torch.Tensor = True,
    ) -> dict[str, torch.Tensor]:
        return self.encode(palm, vein, palm_present, vein_present)


__all__ = [
    "ARCHITECTURE_VERSION",
    "ImageExpert",
    "OFFICIAL_COMMIT",
    "RouterBranch",
    "SimMLMImageModel",
]
