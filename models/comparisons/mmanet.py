"""Feature-level MMANet adapter for the controlled Tongji comparison.

The common, frozen palmprint and palm-vein ResNet-18 encoders live outside
this module.  MMANet therefore receives their ``[B, 256, 7, 7]`` maps and
only replaces the paper-specific multimodal portion: modality-specific
branches, a shared deployment branch, margin-aware distillation (MAD), and
modality-aware regularization (MAR).

The full-modality teacher is deliberately fixed.  Its representation and
logits are computed from frozen encoder embeddings and frozen ArcFace class
weights, so adapting MMANet never adds a second training run or observes an
evaluation identity.
"""

from __future__ import annotations

from typing import Any, Final

import torch
import torch.nn as nn
import torch.nn.functional as F


METHOD_NAME: Final = "MMANet"
IMPLEMENTATION_TYPE: Final = "official-module feature-level adapted"


def _presence_mask(
    value: bool | torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Return a scalar or per-sample modality flag as a ``[B]`` bool mask."""

    if isinstance(value, bool):
        return torch.full((batch_size,), value, dtype=torch.bool, device=device)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a bool or torch.Tensor")
    value = value.to(device=device, dtype=torch.bool)
    if value.numel() == 1:
        return value.reshape(1).expand(batch_size)
    if value.numel() != batch_size:
        raise ValueError(f"{name} has {value.numel()} values, expected {batch_size}")
    return value.reshape(batch_size)


class MMANetAdapter(nn.Module):
    """MMANet's MAD/MAR deployment network on frozen dual-encoder features.

    The singleton order is fixed to ``0=palm`` and ``1=vein``.  MAR statistics
    are collected only from training forward passes in epochs 2--5.  Calling
    :meth:`end_epoch` for epoch 5 locks the singleton with the largest
    accumulated KL distance from the complete-modality prediction histogram;
    from epoch 6 onward only that singleton receives auxiliary supervision.
    All schedule state is held in persistent buffers and is checkpoint-safe.
    """

    PALM_ONLY: Final = 0
    VEIN_ONLY: Final = 1
    COMPLETE: Final = 2
    TEACHER_SCALE: Final = 32.0

    def __init__(
        self,
        input_channels: int = 256,
        embedding_dim: int = 256,
        num_classes: int = 432,
        palm_teacher_weight: torch.Tensor | None = None,
        vein_teacher_weight: torch.Tensor | None = None,
        mad_weight: float = 30.0,
        mar_weight: float = 0.5,
        warmup_epochs: int = 5,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or embedding_dim <= 0:
            raise ValueError("input_channels and embedding_dim must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must exceed one")
        if mad_weight < 0.0 or mar_weight < 0.0:
            raise ValueError("MAD and MAR weights must be non-negative")
        if warmup_epochs < 2:
            raise ValueError("warmup_epochs must be at least 2")

        self.input_channels = int(input_channels)
        self.embedding_dim = int(embedding_dim)
        self.special_channels = self.embedding_dim // 2
        if self.special_channels * 2 != self.embedding_dim:
            raise ValueError("embedding_dim must be even")
        self.num_classes = int(num_classes)
        self.mad_weight = float(mad_weight)
        self.mar_weight = float(mar_weight)
        self.warmup_epochs = int(warmup_epochs)

        # Separate modality-specific trunks feed one shared deployment trunk.
        self.palm_special = nn.Conv2d(
            self.input_channels, self.special_channels, kernel_size=1, bias=False
        )
        self.vein_special = nn.Conv2d(
            self.input_channels, self.special_channels, kernel_size=1, bias=False
        )
        self.shared_fusion = nn.Sequential(
            nn.Conv2d(
                self.embedding_dim,
                self.embedding_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.deployment_head = nn.Linear(self.embedding_dim, self.num_classes)

        # One auxiliary regularization branch is shared by both singleton
        # modalities, matching MMANet's shared auxiliary backbone.
        self.auxiliary_features = nn.Sequential(
            nn.Conv2d(
                self.special_channels,
                self.embedding_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.auxiliary_classifier = nn.Linear(self.embedding_dim, self.num_classes)

        self._initialize_trainable_layers()
        self.register_buffer(
            "palm_teacher_weight",
            self._fixed_teacher_weight(palm_teacher_weight, "palm_teacher_weight"),
        )
        self.register_buffer(
            "vein_teacher_weight",
            self._fixed_teacher_weight(vein_teacher_weight, "vein_teacher_weight"),
        )

        # Histogram rows are palm-only, vein-only and complete.  Float64 keeps
        # accumulation stable and mirrors the offline histogram calculation in
        # the official implementation.
        self.register_buffer(
            "mar_epoch_histograms",
            torch.zeros(3, self.num_classes, dtype=torch.float64),
        )
        self.register_buffer("mar_distribution_distance", torch.zeros(2, dtype=torch.float64))
        self.register_buffer("mar_epoch_observations", torch.zeros(3, dtype=torch.long))
        self.register_buffer("weak_singleton_index", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("mar_current_epoch", torch.tensor(0, dtype=torch.long))
        self.register_buffer("mar_epoch_finalized", torch.tensor(True, dtype=torch.bool))

    def _initialize_trainable_layers(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _fixed_teacher_weight(
        self, value: torch.Tensor | None, name: str
    ) -> torch.Tensor:
        expected = (self.num_classes, self.embedding_dim)
        if value is None:
            # A deterministic-under-seed, frozen fallback keeps the adapter
            # independently testable.  Formal experiments pass checkpoint
            # ArcFace weights and never use this branch.
            value = torch.empty(expected)
            nn.init.xavier_uniform_(value)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor or None")
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {expected}")
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        return value.detach().float().clone()

    @staticmethod
    def _reference_map(
        palm_map: torch.Tensor | None, vein_map: torch.Tensor | None
    ) -> torch.Tensor:
        reference = palm_map if palm_map is not None else vein_map
        if reference is None:
            raise ValueError("At least one modality feature map is required")
        if not isinstance(reference, torch.Tensor) or reference.ndim != 4:
            raise ValueError("Feature maps must have shape [B, C, H, W]")
        if not reference.is_floating_point():
            raise TypeError("Feature maps must be floating point")
        return reference

    def _prepare_inputs(
        self,
        palm_map: torch.Tensor | None,
        vein_map: torch.Tensor | None,
        palm_present: bool | torch.Tensor,
        vein_present: bool | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reference = self._reference_map(palm_map, vein_map)
        batch_size, _, height, width = reference.shape
        device = reference.device
        palm_mask = _presence_mask(
            palm_present, batch_size=batch_size, device=device, name="palm_present"
        )
        vein_mask = _presence_mask(
            vein_present, batch_size=batch_size, device=device, name="vein_present"
        )
        if torch.any(~palm_mask & ~vein_mask):
            raise ValueError("Every sample must contain at least one modality")

        expected = (batch_size, self.input_channels, height, width)
        if palm_map is None:
            if torch.any(palm_mask):
                raise ValueError("palm_map is required when palm_present is true")
            palm_map = torch.zeros_like(reference)
        if vein_map is None:
            if torch.any(vein_mask):
                raise ValueError("vein_map is required when vein_present is true")
            vein_map = torch.zeros_like(reference)
        for value, name in ((palm_map, "palm_map"), (vein_map, "vein_map")):
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {expected}")
            if value.device != device:
                raise ValueError("Both modality maps must be on the same device")
            if not value.is_floating_point():
                raise TypeError(f"{name} must be floating point")

        dtype = self.palm_special.weight.dtype
        palm_map = palm_map.to(dtype=dtype)
        vein_map = vein_map.to(dtype=dtype)
        palm_special = self.palm_special(palm_map)
        vein_special = self.vein_special(vein_map)
        palm_special = palm_special * palm_mask[:, None, None, None]
        vein_special = vein_special * vein_mask[:, None, None, None]
        return palm_special, vein_special, palm_mask, vein_mask

    @staticmethod
    def _pool_unit(feature_map: torch.Tensor) -> torch.Tensor:
        return F.normalize(feature_map.mean(dim=(-2, -1)), dim=1)

    def _deployment_from_special(
        self, palm_special: torch.Tensor, vein_special: torch.Tensor
    ) -> torch.Tensor:
        fused = torch.cat((palm_special, vein_special), dim=1)
        return self._pool_unit(self.shared_fusion(fused))

    def representation(
        self,
        palm_map: torch.Tensor | None,
        vein_map: torch.Tensor | None,
        palm_present: bool | torch.Tensor = True,
        vein_present: bool | torch.Tensor = True,
    ) -> torch.Tensor:
        """Return the unit 256-D deployment representation used for matching."""

        palm_special, vein_special, _, _ = self._prepare_inputs(
            palm_map, vein_map, palm_present, vein_present
        )
        return self._deployment_from_special(palm_special, vein_special)

    def _three_deployment_representations(
        self, palm_map: torch.Tensor, vein_map: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        palm_special, vein_special, _, _ = self._prepare_inputs(
            palm_map, vein_map, True, True
        )
        zeros_palm = torch.zeros_like(palm_special)
        zeros_vein = torch.zeros_like(vein_special)
        complete = self._deployment_from_special(palm_special, vein_special)
        palm_only = self._deployment_from_special(palm_special, zeros_vein)
        vein_only = self._deployment_from_special(zeros_palm, vein_special)
        return complete, palm_only, vein_only, palm_special, vein_special

    def _validate_embedding(
        self, embedding: torch.Tensor, *, batch_size: int, name: str
    ) -> torch.Tensor:
        expected = (batch_size, self.embedding_dim)
        if not isinstance(embedding, torch.Tensor) or tuple(embedding.shape) != expected:
            actual = tuple(embedding.shape) if isinstance(embedding, torch.Tensor) else type(embedding)
            raise ValueError(f"{name} has shape {actual}, expected {expected}")
        if not embedding.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        return embedding.to(device=self.palm_teacher_weight.device, dtype=self.palm_teacher_weight.dtype)

    @torch.no_grad()
    def teacher_outputs(
        self,
        palm_map: torch.Tensor,
        vein_map: torch.Tensor,
        *,
        palm_embedding: torch.Tensor | None = None,
        vein_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute fixed complete-modality teacher representation and logits."""

        reference = self._reference_map(palm_map, vein_map)
        batch_size = reference.size(0)
        if palm_map is None or vein_map is None:
            raise ValueError("The full-modality teacher requires both modality maps")
        if palm_embedding is None:
            palm_embedding = palm_map.mean(dim=(-2, -1))
        if vein_embedding is None:
            vein_embedding = vein_map.mean(dim=(-2, -1))
        palm_embedding = self._validate_embedding(
            palm_embedding, batch_size=batch_size, name="palm_embedding"
        )
        vein_embedding = self._validate_embedding(
            vein_embedding, batch_size=batch_size, name="vein_embedding"
        )
        # The fixed teacher representation is the normalized complete pair.
        # Keep the encoder embeddings unmodified until after their sum; the
        # ArcFace cosine logits below normalize each modality independently.
        teacher_representation = F.normalize(palm_embedding + vein_embedding, dim=1)
        palm_embedding = F.normalize(palm_embedding, dim=1)
        vein_embedding = F.normalize(vein_embedding, dim=1)

        palm_weight = F.normalize(self.palm_teacher_weight, dim=1)
        vein_weight = F.normalize(self.vein_teacher_weight, dim=1)
        palm_logits = self.TEACHER_SCALE * F.linear(palm_embedding, palm_weight)
        vein_logits = self.TEACHER_SCALE * F.linear(vein_embedding, vein_weight)
        teacher_logits = 0.5 * (palm_logits + vein_logits)
        return teacher_representation, teacher_logits

    @staticmethod
    def margin_aware_distillation(
        student_feature: torch.Tensor,
        teacher_feature: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Official MAD: entropy-weighted elementwise Gram-matrix MSE."""

        if student_feature.size(0) != teacher_feature.size(0):
            raise ValueError("Student and teacher batch sizes must match")
        if teacher_logits.ndim != 2 or teacher_logits.size(0) != student_feature.size(0):
            raise ValueError("teacher_logits must have shape [batch, classes]")
        student_flat = student_feature.reshape(student_feature.size(0), -1)
        teacher_flat = teacher_feature.reshape(teacher_feature.size(0), -1)
        student_gram = F.normalize(student_flat @ student_flat.t(), p=2, dim=1)
        teacher_gram = F.normalize(teacher_flat @ teacher_flat.t(), p=2, dim=1)
        elementwise_mse = F.mse_loss(student_gram, teacher_gram, reduction="none")

        probability = F.softmax(teacher_logits, dim=1)
        tiny = torch.finfo(probability.dtype).tiny
        entropy = -(probability * probability.clamp_min(tiny).log()).sum(dim=1)
        entropy_probability = entropy / entropy.sum().clamp_min(tiny)
        # The official tensor broadcasting weights Gram-matrix columns.
        return torch.sum(elementwise_mse * entropy_probability)

    def _auxiliary_logits(self, special: torch.Tensor) -> torch.Tensor:
        auxiliary = self._pool_unit(self.auxiliary_features(special))
        return self.auxiliary_classifier(auxiliary)

    @property
    def weak_modality(self) -> str | None:
        index = int(self.weak_singleton_index.item())
        return {self.PALM_ONLY: "palm", self.VEIN_ONLY: "vein"}.get(index)

    @torch.no_grad()
    def reset_mar_statistics(self) -> None:
        self.mar_epoch_histograms.zero_()
        self.mar_distribution_distance.zero_()
        self.mar_epoch_observations.zero_()
        self.weak_singleton_index.fill_(-1)
        self.mar_current_epoch.zero_()
        self.mar_epoch_finalized.fill_(True)

    @torch.no_grad()
    def begin_epoch(self, epoch: int) -> None:
        """Open a training epoch without discarding resumable same-epoch state."""

        epoch = int(epoch)
        if epoch <= 0:
            raise ValueError("epoch must be positive")
        current = int(self.mar_current_epoch.item())
        if current == epoch and not bool(self.mar_epoch_finalized.item()):
            return
        if current and current != epoch and not bool(self.mar_epoch_finalized.item()):
            self.end_epoch()
        self.mar_current_epoch.fill_(epoch)
        self.mar_epoch_histograms.zero_()
        self.mar_epoch_observations.zero_()
        self.mar_epoch_finalized.fill_(False)

    @torch.no_grad()
    def _update_prediction_histograms(
        self,
        palm_logits: torch.Tensor,
        vein_logits: torch.Tensor,
        complete_logits: torch.Tensor,
    ) -> None:
        epoch = int(self.mar_current_epoch.item())
        if not 2 <= epoch <= self.warmup_epochs:
            return
        for row, logits in enumerate((palm_logits, vein_logits, complete_logits)):
            predictions = logits.detach().argmax(dim=1)
            histogram = torch.bincount(predictions, minlength=self.num_classes)
            self.mar_epoch_histograms[row].add_(histogram.to(torch.float64))
            self.mar_epoch_observations[row].add_(predictions.numel())

    @staticmethod
    def _official_histogram_kl(singleton: torch.Tensor, complete: torch.Tensor) -> torch.Tensor:
        # Official Eq. 12/13 code applies softmax to the already-normalized
        # histograms, then KL(singleton || complete) via kl_div.
        return F.kl_div(
            F.log_softmax(complete.unsqueeze(0), dim=1),
            F.softmax(singleton.unsqueeze(0), dim=1),
            reduction="batchmean",
        )

    @torch.no_grad()
    def end_epoch(self) -> None:
        """Finalize warm-up histogram KL and lock the weak singleton at epoch 5."""

        if bool(self.mar_epoch_finalized.item()):
            return
        epoch = int(self.mar_current_epoch.item())
        if 2 <= epoch <= self.warmup_epochs and torch.all(self.mar_epoch_observations > 0):
            distributions = self.mar_epoch_histograms / self.mar_epoch_observations[:, None]
            for singleton in (self.PALM_ONLY, self.VEIN_ONLY):
                distance = self._official_histogram_kl(
                    distributions[singleton], distributions[self.COMPLETE]
                )
                self.mar_distribution_distance[singleton].add_(distance)
        if epoch == self.warmup_epochs:
            self.weak_singleton_index.copy_(
                self.mar_distribution_distance.argmax().to(dtype=torch.long)
            )
        self.mar_epoch_finalized.fill_(True)

    def _ensure_epoch(self, epoch: int) -> None:
        current = int(self.mar_current_epoch.item())
        finalized = bool(self.mar_epoch_finalized.item())
        if current != int(epoch) or finalized:
            self.begin_epoch(int(epoch))

    def forward(
        self,
        palm_map: torch.Tensor | None,
        vein_map: torch.Tensor | None,
        palm_present: bool | torch.Tensor = True,
        vein_present: bool | torch.Tensor = True,
        *,
        labels: torch.Tensor | None = None,
        epoch: int | None = None,
        palm_embedding: torch.Tensor | None = None,
        vein_embedding: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Run inference or compute complete/singleton MMANet training losses."""

        if labels is None:
            representation = self.representation(
                palm_map, vein_map, palm_present, vein_present
            )
            logits = self.deployment_head(representation)
            return {
                "representation": representation,
                "inference_representation": representation,
                "logits": logits,
                "task_logits": logits,
                "loss_dict": {},
                "mar_active": False,
                "weak_modality": self.weak_modality,
            }

        if palm_map is None or vein_map is None:
            raise ValueError("MMANet training requires complete paired feature maps")
        if epoch is None:
            raise ValueError("epoch is required when labels are provided")
        labels = labels.to(device=palm_map.device, dtype=torch.long).reshape(-1)
        if labels.numel() != palm_map.size(0):
            raise ValueError("There must be one label per sample")
        if self.training:
            self._ensure_epoch(int(epoch))

        complete, palm_only, vein_only, palm_special, vein_special = (
            self._three_deployment_representations(palm_map, vein_map)
        )
        complete_logits = self.deployment_head(complete)
        palm_logits = self.deployment_head(palm_only)
        vein_logits = self.deployment_head(vein_only)
        teacher_representation, teacher_logits = self.teacher_outputs(
            palm_map,
            vein_map,
            palm_embedding=palm_embedding,
            vein_embedding=vein_embedding,
        )

        deployment_loss = torch.stack(
            (
                F.cross_entropy(complete_logits, labels),
                F.cross_entropy(palm_logits, labels),
                F.cross_entropy(vein_logits, labels),
            )
        ).mean()
        mad_loss = torch.stack(
            tuple(
                self.margin_aware_distillation(
                    student, teacher_representation, teacher_logits
                )
                for student in (complete, palm_only, vein_only)
            )
        ).mean()

        if self.training:
            self._update_prediction_histograms(palm_logits, vein_logits, complete_logits)
        mar_active = int(epoch) > self.warmup_epochs and self.weak_modality is not None
        if mar_active:
            weak_special = (
                palm_special
                if int(self.weak_singleton_index.item()) == self.PALM_ONLY
                else vein_special
            )
            auxiliary_logits = self._auxiliary_logits(weak_special)
            mar_loss = F.cross_entropy(auxiliary_logits, labels)
        else:
            auxiliary_logits = complete_logits.new_empty((0, self.num_classes))
            mar_loss = deployment_loss.new_zeros(())

        if mar_active:
            total = (
                deployment_loss
                + self.mad_weight * mad_loss
                + self.mar_weight * mar_loss
            )
        else:
            total = deployment_loss
        losses = {
            "deployment": deployment_loss,
            "classification": deployment_loss,
            "mad": mad_loss,
            "mar": mar_loss,
            "total": total,
        }
        return {
            "representation": complete,
            "inference_representation": complete,
            "logits": complete_logits,
            "task_logits": complete_logits,
            "complete_logits": complete_logits,
            "palm_logits": palm_logits,
            "vein_logits": vein_logits,
            "auxiliary_logits": auxiliary_logits,
            "teacher_representation": teacher_representation,
            "teacher_logits": teacher_logits,
            "loss_dict": losses,
            "mar_active": mar_active,
            "weak_modality": self.weak_modality,
        }


__all__ = ["MMANetAdapter"]
