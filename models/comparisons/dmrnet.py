"""DMRNet for the controlled Tongji comparison.

The common palmprint and palm-vein encoders live outside this module. This
adapter receives their final ``[B, C, H, W]`` maps and implements the part of
DMRNet that starts at modality dropout: uniform sampling from every non-empty
modality combination, spatial probabilistic embeddings, distribution
regularization, and hard-combination regularization (HCR).

The implementation follows the ECCV 2024 paper and the official classification
code. In particular, ``mean_head`` and ``logvar_head`` are 1x1 convolution +
batch-normalization heads on a 2D fused map, the task predictor consumes a
reparameterized sample during training, and inference always uses the mean.
"""

from __future__ import annotations

from typing import Any, Final

import torch
import torch.nn as nn
import torch.nn.functional as F


METHOD_NAME: Final = "DMRNet"
IMPLEMENTATION_TYPE: Final = "official-module spatial-feature adapted"


def _presence_mask(
    value: bool | torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Return a scalar or per-sample presence flag as a ``[B]`` bool mask."""

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


class DMRNetAdapter(nn.Module):
    """Spatial DMRNet on top of two modality-specific encoder maps.

    Combination codes use the bit convention from the paper:
    ``1=palm-only``, ``2=vein-only`` and ``3=complete``. Warm-up statistics
    store the sum of every predicted variance element and its exact element
    count, making Eq. (10) an unbiased whole-data accumulation even when the
    final batch has a different size.

    ``training_step`` is the faithful training entry point: it assigns each
    paired sample a uniformly random non-empty combination. ``forward`` also
    accepts explicit masks for evaluation and for runners which prepare the
    combinations themselves.
    """

    PALM_ONLY: Final = 1
    VEIN_ONLY: Final = 2
    COMPLETE: Final = 3
    NUM_NONEMPTY_COMBINATIONS: Final = 3

    def __init__(
        self,
        input_channels: int = 256,
        embedding_dim: int = 256,
        num_classes: int = 432,
        alpha: float = 1e-3,
        beta: float = 0.5,
        hcr_top_v: int = 2,
        hcr_warmup_epochs: int = 5,
        logvar_min: float = -10.0,
        logvar_max: float = 10.0,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or embedding_dim <= 0 or num_classes <= 1:
            raise ValueError(
                "Channel and embedding counts must be positive and num_classes must exceed one"
            )
        if alpha < 0.0 or beta < 0.0:
            raise ValueError("alpha and beta must be non-negative")
        if not 1 <= hcr_top_v <= self.NUM_NONEMPTY_COMBINATIONS:
            raise ValueError("hcr_top_v must be between 1 and 3")
        if hcr_warmup_epochs < 0:
            raise ValueError("hcr_warmup_epochs cannot be negative")
        if logvar_min >= logvar_max:
            raise ValueError("logvar_min must be smaller than logvar_max")

        self.input_channels = int(input_channels)
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.hcr_top_v = int(hcr_top_v)
        self.hcr_warmup_epochs = int(hcr_warmup_epochs)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

        # Cached inputs are already final spatial maps from two full encoders;
        # concatenation is therefore the adapted fusion representation z.
        fused_channels = self.input_channels * 2
        self.mean_head = nn.Sequential(
            nn.Conv2d(fused_channels, self.embedding_dim, kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim),
        )
        self.logvar_head = nn.Sequential(
            nn.Conv2d(fused_channels, self.embedding_dim, kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim),
        )
        self.classifier = nn.Linear(self.embedding_dim, self.num_classes)
        self._initialize_layers()

        # Index zero is the invalid all-missing pattern. Float64 accumulation
        # avoids precision loss over an entire training set and is checkpointed.
        self.register_buffer(
            "combination_variance_sum", torch.zeros(4, dtype=torch.float64)
        )
        self.register_buffer(
            "combination_variance_element_count", torch.zeros(4, dtype=torch.long)
        )
        self.register_buffer(
            "combination_observation_count", torch.zeros(4, dtype=torch.long)
        )

    def _initialize_layers(self) -> None:
        """Use the initialization applied by the official AV implementation."""

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @property
    def hcr_predictor(self) -> nn.Linear:
        """The HCR predictor is exactly the task predictor (paper Sec. 3.3)."""

        return self.classifier

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
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
                actual = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value)
                raise ValueError(f"{name} has shape {actual}, expected {expected}")
            if value.device != device:
                raise ValueError("Both modality maps must be on the same device")
            if not value.is_floating_point():
                raise TypeError(f"{name} must be floating point")

        dtype = self.mean_head[0].weight.dtype
        palm_map = palm_map.to(dtype=dtype) * palm_mask[:, None, None, None]
        vein_map = vein_map.to(dtype=dtype) * vein_mask[:, None, None, None]
        return palm_map, vein_map, palm_mask, vein_mask

    @staticmethod
    def combination_codes(
        palm_present: torch.Tensor, vein_present: torch.Tensor
    ) -> torch.Tensor:
        """Encode masks as the bit sum used in Eq. (10)."""

        return palm_present.long() + 2 * vein_present.long()

    @classmethod
    def sample_combination_codes(
        cls,
        batch_size: int,
        device: torch.device | str,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Uniformly sample all non-empty combinations, independently per item."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        # The comparison runner owns a CPU generator even when the model is on
        # CUDA. Sample on the generator's device, then transfer the tiny code
        # vector, so checkpointed RNG replay remains device-safe.
        sample_device = device if generator is None else generator.device
        codes = torch.randint(
            cls.PALM_ONLY,
            cls.COMPLETE + 1,
            (int(batch_size),),
            device=sample_device,
            generator=generator,
        )
        return codes.to(device=device)

    @classmethod
    def random_modality_masks(
        cls,
        batch_size: int,
        device: torch.device | str,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample codes and return ``(palm_mask, vein_mask, codes)``."""

        codes = cls.sample_combination_codes(
            batch_size, device, generator=generator
        )
        return codes.bitwise_and(1).bool(), codes.bitwise_and(2).bool(), codes

    def distribution(
        self,
        palm_map: torch.Tensor | None,
        vein_map: torch.Tensor | None,
        palm_present: bool | torch.Tensor = True,
        vein_present: bool | torch.Tensor = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate spatial ``mu`` and ``log(sigma^2)`` for an input combination."""

        palm_map, vein_map, palm_mask, vein_mask = self._prepare_inputs(
            palm_map, vein_map, palm_present, vein_present
        )
        fused = torch.cat((palm_map, vein_map), dim=1)
        mean = self.mean_head(fused)
        # The official code has no clamp. Bounding the downstream value avoids
        # exp overflow while preserving its parameterization in the useful range.
        logvar = self.logvar_head(fused).clamp(
            min=self.logvar_min, max=self.logvar_max
        )
        return mean, logvar, self.combination_codes(palm_mask, vein_mask)

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Differentiably sample ``mu + epsilon * sigma``."""

        if mean.shape != logvar.shape:
            raise ValueError("mean and logvar must have identical shapes")
        return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)

    @staticmethod
    def pool(feature_map: torch.Tensor) -> torch.Tensor:
        """Official task mapping ``g``: spatial average pooling and flattening."""

        if feature_map.ndim != 4:
            raise ValueError("feature_map must have shape [B, C, H, W]")
        return F.adaptive_avg_pool2d(feature_map, 1).flatten(1)

    @classmethod
    def pool_and_normalize(cls, feature_map: torch.Tensor) -> torch.Tensor:
        """Unit representation used only by the biometric matcher."""

        return F.normalize(cls.pool(feature_map), dim=1)

    @staticmethod
    def kl_divergence(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Paper Eq. (9): batch mean of the per-sample, all-element KL sum."""

        if mean.shape != logvar.shape:
            raise ValueError("mean and logvar must have identical shapes")
        if mean.ndim < 2:
            raise ValueError("mean and logvar must include batch and feature axes")
        mean_float = mean.float()
        logvar_float = logvar.float()
        per_element = 0.5 * (
            mean_float.square() + logvar_float.exp() - 1.0 - logvar_float
        )
        return per_element.flatten(1).sum(dim=1).mean()

    def representation(
        self,
        palm_map: torch.Tensor | None,
        vein_map: torch.Tensor | None,
        palm_present: bool | torch.Tensor = True,
        vein_present: bool | torch.Tensor = True,
        sample: bool | None = None,
    ) -> torch.Tensor:
        """Return a unit retrieval representation; evaluation always uses ``mu``."""

        mean, logvar, _ = self.distribution(
            palm_map, vein_map, palm_present, vein_present
        )
        should_sample = self.training if sample is None else bool(sample)
        spatial = self.reparameterize(mean, logvar) if should_sample else mean
        return self.pool_and_normalize(spatial)

    @torch.no_grad()
    def reset_hcr_statistics(self) -> None:
        self.combination_variance_sum.zero_()
        self.combination_variance_element_count.zero_()
        self.combination_observation_count.zero_()

    @torch.no_grad()
    def update_hcr_statistics(
        self, logvar: torch.Tensor, combination_codes: torch.Tensor
    ) -> None:
        """Accumulate every ``sigma^2`` element by combination (paper Eq. 10)."""

        if not isinstance(logvar, torch.Tensor) or logvar.ndim != 4:
            raise ValueError("logvar must have shape [B, C, H, W]")
        codes = combination_codes.to(
            device=logvar.device, dtype=torch.long
        ).reshape(-1)
        if codes.numel() != logvar.size(0):
            raise ValueError("There must be one combination code per sample")
        if torch.any((codes < self.PALM_ONLY) | (codes > self.COMPLETE)):
            raise ValueError("Combination codes must be one of 1, 2 or 3")

        variance = logvar.detach().double().exp().flatten(1)
        sample_sums = variance.sum(dim=1)
        self.combination_variance_sum.index_add_(0, codes, sample_sums)
        elements_per_sample = variance.size(1)
        self.combination_variance_element_count.index_add_(
            0,
            codes,
            torch.full_like(codes, elements_per_sample, dtype=torch.long),
        )
        self.combination_observation_count.index_add_(
            0, codes, torch.ones_like(codes, dtype=torch.long)
        )

    def variance_by_combination(self) -> torch.Tensor:
        """Return Eq. (10) for codes 0..3; unseen combinations are ``NaN``."""

        counts = self.combination_variance_element_count
        means = self.combination_variance_sum / counts.clamp_min(1)
        return torch.where(
            counts > 0, means, torch.full_like(means, float("nan"))
        )

    def hard_combination_codes(self) -> torch.Tensor:
        """Return the observed top-V combinations ranked by accumulated variance."""

        counts = self.combination_variance_element_count[1:]
        valid = torch.nonzero(counts > 0, as_tuple=False).flatten() + 1
        if valid.numel() == 0:
            return valid
        scores = self.variance_by_combination()[valid]
        top_count = min(self.hcr_top_v, valid.numel())
        # stable=True gives deterministic lower-code tie breaking.
        order = torch.argsort(scores, descending=True, stable=True)
        return valid[order[:top_count]]

    def _hcr_mask(self, combination_codes: torch.Tensor) -> torch.Tensor:
        hard_codes = self.hard_combination_codes().to(combination_codes.device)
        if hard_codes.numel() == 0:
            return torch.zeros_like(combination_codes, dtype=torch.bool)
        return (combination_codes[:, None] == hard_codes[None, :]).any(dim=1)

    def _shared_hcr_forward(
        self,
        task_embedding: torch.Tensor,
        labels: torch.Tensor,
        combination_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Branch again through the task predictor for hard combinations.

        The official model computes ``out`` and ``auxi_out`` by calling the same
        linear layer on the same sampled embedding. This creates the auxiliary
        HCR loss branch while sharing both predictor parameters and the gradient
        path into the sampled spatial representation. Dividing the selected loss
        sum by the original batch size matches Eq. (11--12): non-hard samples
        contribute zero rather than shrinking the denominator.
        """

        hcr_mask = self._hcr_mask(combination_codes)
        if not torch.any(hcr_mask):
            empty = task_embedding.new_empty((0, self.num_classes))
            return self.classifier.weight.sum() * 0.0, empty, hcr_mask

        # fr and ft are deliberately the same module, exactly as in the
        # official ConcatFusion and SURF auxiliary-share implementations.
        hard_logits = self.hcr_predictor(task_embedding[hcr_mask])
        hard_loss = F.cross_entropy(
            hard_logits, labels[hcr_mask], reduction="sum"
        ) / labels.numel()
        return hard_loss, hard_logits, hcr_mask

    def training_step(
        self,
        palm_map: torch.Tensor,
        vein_map: torch.Tensor,
        labels: torch.Tensor,
        *,
        epoch: int,
        generator: torch.Generator | None = None,
        sample: bool | None = None,
        update_hcr_stats: bool = True,
    ) -> dict[str, Any]:
        """Apply official per-sample random modality dropout and compute losses."""

        reference = self._reference_map(palm_map, vein_map)
        if palm_map is None or vein_map is None:
            raise ValueError("DMRNet training requires complete paired feature maps")
        if palm_map.shape != vein_map.shape:
            raise ValueError("Paired feature maps must have identical shapes")
        palm_present, vein_present, _ = self.random_modality_masks(
            reference.size(0), reference.device, generator=generator
        )
        return self.forward(
            palm_map,
            vein_map,
            palm_present,
            vein_present,
            labels=labels,
            epoch=epoch,
            sample=sample,
            update_hcr_stats=update_hcr_stats,
        )

    def forward(
        self,
        palm_map: torch.Tensor | None,
        vein_map: torch.Tensor | None,
        palm_present: bool | torch.Tensor = True,
        vein_present: bool | torch.Tensor = True,
        *,
        labels: torch.Tensor | None = None,
        epoch: int | None = None,
        sample: bool | None = None,
        update_hcr_stats: bool = True,
    ) -> dict[str, Any]:
        """Run explicit combinations or calculate the complete DMRNet loss."""

        reference = self._reference_map(palm_map, vein_map)
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
        mean, logvar, codes = self.distribution(
            palm_map, vein_map, palm_mask, vein_mask
        )
        should_sample = self.training if sample is None else bool(sample)
        sampled_map = self.reparameterize(mean, logvar) if should_sample else mean

        # The official classifier sees unnormalized GAP features. Unit
        # normalization is added only at the biometric matching boundary.
        task_embedding = self.pool(sampled_map)
        task_representation = F.normalize(task_embedding, dim=1)
        inference_embedding = self.pool(mean)
        inference_representation = F.normalize(inference_embedding, dim=1)
        logits = self.classifier(task_embedding)

        current_epoch = 0 if epoch is None else int(epoch)
        if (
            self.training
            and labels is not None
            and update_hcr_stats
            and current_epoch <= self.hcr_warmup_epochs
        ):
            self.update_hcr_statistics(logvar, codes)

        hard_codes = self.hard_combination_codes()
        hcr_statistics_complete = bool(
            torch.all(self.combination_variance_element_count[1:] > 0).item()
        )
        hcr_active = (
            self.training
            and current_epoch > self.hcr_warmup_epochs
            and hcr_statistics_complete
        )
        output: dict[str, Any] = {
            "representation": task_representation,
            "inference_representation": inference_representation,
            "sampled_representation": task_representation,
            "task_embedding": task_embedding,
            "inference_embedding": inference_embedding,
            "mean": mean,
            "mu": mean,
            "logvar": logvar,
            "std": torch.exp(0.5 * logvar),
            "variance": logvar.exp(),
            "task_logits": logits,
            "logits": logits,
            "combination_codes": codes,
            "palm_present": palm_mask,
            "vein_present": vein_mask,
            "hard_combination_codes": hard_codes,
            "hcr_statistics_complete": hcr_statistics_complete,
            "hcr_active": hcr_active,
            "hcr_logits": logits.new_empty((0, self.num_classes)),
            "hcr_sample_mask": torch.zeros_like(codes, dtype=torch.bool),
            "loss_dict": {},
        }

        if labels is not None:
            labels = labels.to(device=logits.device, dtype=torch.long).reshape(-1)
            if labels.numel() != logits.size(0):
                raise ValueError("There must be one label per sample")
            task_loss = F.cross_entropy(logits, labels)
            distribution_loss = self.kl_divergence(mean, logvar)
            if hcr_active:
                hcr_loss, hcr_logits, hcr_mask = self._shared_hcr_forward(
                    task_embedding, labels, codes
                )
            else:
                hcr_loss = task_loss.new_zeros(())
                hcr_logits = logits.new_empty((0, self.num_classes))
                hcr_mask = torch.zeros_like(codes, dtype=torch.bool)
            total = task_loss + self.alpha * distribution_loss
            if hcr_active:
                total = total + self.beta * hcr_loss
            output["hcr_logits"] = hcr_logits
            output["hcr_sample_mask"] = hcr_mask
            output["loss_dict"] = {
                "task": task_loss,
                "classification": task_loss,
                "kl": distribution_loss,
                "distribution": distribution_loss,
                "hcr": hcr_loss,
                "total": total,
            }
        return output


__all__ = ["DMRNetAdapter"]
