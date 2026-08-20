"""Feature-level SimMLM adapter for the controlled Tongji comparison.

The frozen palmprint and palm-vein encoders are treated as SimMLM's already
pretrained modality experts.  This module implements the cooperative DMoME
stage on their 256-dimensional outputs: a trainable projection per modality,
a mask-aware router, and a shared identity classifier.  The MoFe objective
compares the complete input with each of its two single-modality subsets.
"""

from __future__ import annotations

from typing import Any, Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import build_encoder


METHOD_NAME: Final = "SimMLM"
IMPLEMENTATION_TYPE: Final = "official-method feature-level adapted"
ARCHITECTURE_VERSION: Final = "simmlm_dmome_feature_adapter_v1"
FULL_ARCHITECTURE_VERSION: Final = "simmlm_dmome_logit_image_v2"


def _presence_mask(
    value: bool | torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Expand a scalar or per-sample modality indicator to ``[batch]``."""

    if isinstance(value, bool):
        return torch.full((batch_size,), value, dtype=torch.bool, device=device)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a bool or torch.Tensor")
    value = value.to(device=device, dtype=torch.bool)
    if value.numel() == 1:
        return value.reshape(1).expand(batch_size)
    if value.numel() != batch_size:
        raise ValueError(
            f"{name} has {value.numel()} values, expected {batch_size}"
        )
    return value.reshape(batch_size)


class _ResidualExpert(nn.Module):
    """One trainable projection on top of a frozen modality embedding."""

    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, embedding_dim)
        self.normalization = nn.LayerNorm(embedding_dim)
        self.activation = nn.ReLU(inplace=False)
        self.residual = (
            nn.Identity()
            if input_dim == embedding_dim
            else nn.Linear(input_dim, embedding_dim, bias=False)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        projected = self.activation(self.normalization(self.projection(values)))
        return F.normalize(self.residual(values) + projected, dim=1)


class SimMLMAdapter(nn.Module):
    """Dynamic Mixture of Modality Experts for two frozen embeddings.

    Args:
        input_dim: Dimension of each frozen unimodal encoder output.
        embedding_dim: Dimension of each expert and the fused representation.
        num_classes: Number of training identities.
        router_hidden_dim: Hidden width of the DMoME router.
        ranking_weight: Weight of the More-vs-Fewer (MoFe) ranking loss.

    ``palm_present`` and ``vein_present`` may each be a Python boolean, a
    scalar tensor, or a per-sample tensor.  A missing modality is zeroed before
    entering both its expert and the router, and its routing logit is set to
    negative infinity before softmax.  Consequently it receives exactly zero
    weight, while an available single modality receives exactly unit weight.
    """

    def __init__(
        self,
        input_dim: int = 256,
        embedding_dim: int = 256,
        num_classes: int = 432,
        router_hidden_dim: int = 128,
        ranking_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or embedding_dim <= 0 or router_hidden_dim <= 0:
            raise ValueError("Feature and router dimensions must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must exceed one")
        if ranking_weight < 0.0:
            raise ValueError("ranking_weight must be non-negative")

        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)
        self.representation_dim = self.embedding_dim
        self.num_classes = int(num_classes)
        self.router_hidden_dim = int(router_hidden_dim)
        self.ranking_weight = float(ranking_weight)

        self.palm_expert = _ResidualExpert(self.input_dim, self.embedding_dim)
        self.vein_expert = _ResidualExpert(self.input_dim, self.embedding_dim)
        self.router = nn.Sequential(
            nn.Linear(2 * self.input_dim + 2, self.router_hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(self.router_hidden_dim, 2),
        )
        self.classifier = nn.Linear(self.embedding_dim, self.num_classes)
        self._initialize_trainable_layers()

    def _initialize_trainable_layers(self) -> None:
        """Use a deterministic, explicit Xavier policy for every linear."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _validate_embedding(
        values: torch.Tensor, name: str, input_dim: int
    ) -> None:
        if values.ndim != 2 or values.size(1) != input_dim:
            raise ValueError(
                f"{name} must have shape [batch, {input_dim}], "
                f"got {tuple(values.shape)}"
            )
        if not values.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")

    def _prepare_optional_pair(
        self,
        palm: torch.Tensor | None,
        vein: torch.Tensor | None,
        palm_present: bool | torch.Tensor,
        vein_present: bool | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reference = palm if palm is not None else vein
        if reference is None:
            raise ValueError("At least one modality embedding is required")
        self._validate_embedding(reference, "reference", self.input_dim)
        batch_size, device = reference.size(0), reference.device
        palm_mask = _presence_mask(
            palm_present,
            batch_size=batch_size,
            device=device,
            name="palm_present",
        )
        vein_mask = _presence_mask(
            vein_present,
            batch_size=batch_size,
            device=device,
            name="vein_present",
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
        self._validate_embedding(palm, "palm", self.input_dim)
        self._validate_embedding(vein, "vein", self.input_dim)
        if palm.shape != vein.shape:
            raise ValueError("Palm and vein embeddings must have identical shapes")
        if palm.device != vein.device:
            raise ValueError("Palm and vein embeddings must be on the same device")

        dtype = self.classifier.weight.dtype
        palm = palm.to(dtype=dtype) * palm_mask[:, None].to(dtype=dtype)
        vein = vein.to(dtype=dtype) * vein_mask[:, None].to(dtype=dtype)
        return palm, vein, palm_mask, vein_mask

    def _route_prepared(
        self,
        palm: torch.Tensor,
        vein: torch.Tensor,
        palm_mask: torch.Tensor,
        vein_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        masks = torch.stack((palm_mask, vein_mask), dim=1)
        router_input = torch.cat((palm, vein, masks.to(dtype=palm.dtype)), dim=1)
        router_logits = self.router(router_input)
        router_logits = router_logits.masked_fill(~masks, float("-inf"))
        weights = F.softmax(router_logits, dim=1)

        palm_expert = self.palm_expert(palm)
        vein_expert = self.vein_expert(vein)
        fused = (
            weights[:, 0:1] * palm_expert
            + weights[:, 1:2] * vein_expert
        )
        return F.normalize(fused, dim=1), weights, palm_expert, vein_expert

    def encode(
        self,
        palm: torch.Tensor | None,
        vein: torch.Tensor | None,
        palm_present: bool | torch.Tensor = True,
        vein_present: bool | torch.Tensor = True,
    ) -> dict[str, torch.Tensor]:
        """Return the fused representation, expert outputs and route weights."""

        palm, vein, palm_mask, vein_mask = self._prepare_optional_pair(
            palm, vein, palm_present, vein_present
        )
        representation, weights, palm_expert, vein_expert = self._route_prepared(
            palm, vein, palm_mask, vein_mask
        )
        return {
            "representation": representation,
            "router_weights": weights,
            "palm_expert": palm_expert,
            "vein_expert": vein_expert,
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
        """Return a unit 256-d representation for any valid modality mask."""

        return self.encode(
            palm, vein, palm_present, vein_present
        )["representation"]

    def _training_variants(
        self, palm: torch.Tensor, vein: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        variants = {
            "complete": self.encode(palm, vein, True, True),
            "palm": self.encode(palm, None, True, False),
            "vein": self.encode(None, vein, False, True),
        }
        representations = {
            name: encoded["representation"] for name, encoded in variants.items()
        }
        weights = {
            name: encoded["router_weights"] for name, encoded in variants.items()
        }
        return representations, weights

    def _losses_from_variants(
        self,
        representations: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        batch_size = representations["complete"].size(0)
        labels = labels.to(
            device=representations["complete"].device, dtype=torch.long
        ).reshape(-1)
        if labels.numel() != batch_size:
            raise ValueError("There must be one label per sample")

        logits = {
            name: self.classifier(values)
            for name, values in representations.items()
        }
        per_sample = {
            name: F.cross_entropy(values, labels, reduction="none")
            for name, values in logits.items()
        }
        complete_task = per_sample["complete"].mean()
        palm_task = per_sample["palm"].mean()
        vein_task = per_sample["vein"].mean()
        # Each comparable pair follows the official blended-loss convention:
        # concatenate its more/less samples and average the task loss over 2B.
        # Averaging the (complete, palm) and (complete, vein) pairs therefore
        # counts the complete-modality loss twice.
        task = (2.0 * complete_task + palm_task + vein_task) / 4.0

        # SimMLM's MoFe objective: more modalities should never have a larger
        # task loss than a directly comparable strict subset.
        more_vs_palm = F.relu(
            per_sample["complete"] - per_sample["palm"]
        ).mean()
        more_vs_vein = F.relu(
            per_sample["complete"] - per_sample["vein"]
        ).mean()
        mofe = (more_vs_palm + more_vs_vein) / 2.0
        total = task + self.ranking_weight * mofe
        losses = {
            "complete_task": complete_task,
            "palm_task": palm_task,
            "vein_task": vein_task,
            "task": task,
            "more_vs_palm": more_vs_palm,
            "more_vs_vein": more_vs_vein,
            "mofe": mofe,
            "total": total,
        }
        return logits, losses

    def loss_dict(
        self,
        palm: torch.Tensor,
        vein: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute complete + two single-modal task losses and MoFe."""

        representations, _ = self._training_variants(palm, vein)
        _, losses = self._losses_from_variants(representations, labels)
        return losses

    def forward(
        self,
        palm: torch.Tensor | None,
        vein: torch.Tensor | None,
        palm_present: bool | torch.Tensor = True,
        vein_present: bool | torch.Tensor = True,
        *,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        selected = self.encode(palm, vein, palm_present, vein_present)
        selected_logits = self.classifier(selected["representation"])
        output: dict[str, Any] = {
            **selected,
            "task_logits": selected_logits,
            "logits": {"selected": selected_logits},
            "representations": {"selected": selected["representation"]},
            "routing": {"selected": selected["router_weights"]},
            "loss_dict": {},
        }

        if labels is not None:
            if palm is None or vein is None:
                raise ValueError(
                    "Both modalities are required to compute SimMLM training losses"
                )
            representations, weights = self._training_variants(palm, vein)
            logits, losses = self._losses_from_variants(representations, labels)
            output["representations"] = representations
            output["routing"] = weights
            output["logits"] = logits
            output["loss_dict"] = losses
        return output


__all__ = [
    "ARCHITECTURE_VERSION",
    "IMPLEMENTATION_TYPE",
    "METHOD_NAME",
    "SimMLMAdapter",
]
