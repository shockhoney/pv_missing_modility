"""Balanced staged DCCA and Transformer recovery without legacy fallback."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dcca_specformer_components import (
    RecoveryBackbone,
    _templates,
)


ARCHITECTURE_VERSION = "balanced_staged_dcca_specformer_v9_1"


class BalancedFusionGate(nn.Module):
    """Keep both shared and recovered-specific evidence active for every sample."""

    def __init__(
        self,
        dropout: float,
        min_recovery_weight: float = 0.15,
        max_recovery_weight: float = 0.75,
        initial_recovery_weight: float = 0.30,
    ):
        super().__init__()
        if not 0.0 < min_recovery_weight < max_recovery_weight < 1.0:
            raise ValueError("Recovery bounds must satisfy 0 < min < max < 1")
        self.min_recovery_weight = float(min_recovery_weight)
        if not min_recovery_weight < initial_recovery_weight < max_recovery_weight:
            raise ValueError("Initial recovery weight must be inside its bounds")
        self.max_recovery_weight = float(max_recovery_weight)
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        probability = (initial_recovery_weight - min_recovery_weight) / (
            max_recovery_weight - min_recovery_weight
        )
        nn.init.constant_(
            self.net[-1].bias, math.log(probability / (1.0 - probability))
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(self.net(features).squeeze(1))
        width = self.max_recovery_weight - self.min_recovery_weight
        return self.min_recovery_weight + width * probability


class DCCASpecFormerRecovery(RecoveryBackbone):
    """Trainable shared alignment plus bounded Transformer recovery fusion."""

    def __init__(
        self,
        input_dim: int = 256,
        shared_dim: int = 192,
        specific_dim: int = 128,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.1,
        max_gate: float = 0.75,
        min_recovery_weight: float = 0.15,
        retrieval_dropout: float = 0.10,
        branch_floor: float = 0.0,
        max_proxy_identities: int = 512,
    ):
        if not 0.0 <= retrieval_dropout < 1.0:
            raise ValueError("retrieval_dropout must be in [0, 1)")
        if not 0.0 <= branch_floor < 1.0 / 3.0:
            raise ValueError("branch_floor must be in [0, 1/3)")
        super().__init__(
            input_dim=input_dim,
            shared_dim=shared_dim,
            specific_dim=specific_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            dropout=dropout,
        )
        self.retrieval_dropout = float(retrieval_dropout)
        self.branch_floor = float(branch_floor)
        self.max_proxy_identities = int(max_proxy_identities)
        self.safe_gate = BalancedFusionGate(
            dropout=dropout,
            min_recovery_weight=min_recovery_weight,
            max_recovery_weight=max_gate,
        )
        self.identity_proxies = nn.Parameter(
            torch.empty(max_proxy_identities, shared_dim)
        )
        nn.init.normal_(self.identity_proxies, std=0.02)
        self.register_buffer(
            "active_identity_labels",
            torch.full((max_proxy_identities,), -1, dtype=torch.long),
        )
        self.register_buffer("active_identity_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("recovery_stage_ready", torch.tensor(False, dtype=torch.bool))
        hidden = max(specific_dim, input_dim // 2)
        self.cycle_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, input_dim),
                )
                for _ in range(2)
            ]
        )
        # Retrieval corruption needs a trainable residual with useful initial scale;
        self.recovery_decoder.residual_logit.data.fill_(math.log(0.1 / 0.9))

    @property
    def min_recovery_weight(self) -> float:
        return self.safe_gate.min_recovery_weight

    @property
    def max_recovery_weight(self) -> float:
        return self.safe_gate.max_recovery_weight

    @torch.no_grad()
    def initialize_identity_proxies(
        self,
        palm: torch.Tensor,
        vein: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        identities = labels.long().unique(sorted=True)
        if identities.numel() > self.max_proxy_identities:
            raise ValueError("Training identities exceed proxy capacity")
        self.active_identity_labels.fill_(-1)
        self.active_identity_labels[: identities.numel()].copy_(identities)
        self.active_identity_count.fill_(identities.numel())
        palm_shared = self.palm_projector(palm)
        vein_shared = self.vein_projector(vein)
        joint = F.normalize(palm_shared + vein_shared, dim=1)
        templates, template_labels = _templates(joint, labels)
        if not torch.equal(template_labels, identities):
            raise ValueError("Proxy identity order differs from training labels")
        self.identity_proxies[: identities.numel()].copy_(templates)

    def proxy_logits(
        self, shared: torch.Tensor, labels: torch.Tensor, temperature: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = int(self.active_identity_count.item())
        if count < 1:
            raise RuntimeError("Identity proxies have not been initialized")
        identity_labels = self.active_identity_labels[:count]
        targets = torch.searchsorted(identity_labels, labels.long())
        if not torch.equal(identity_labels[targets], labels.long()):
            raise ValueError("Batch contains an identity outside the proxy table")
        proxies = F.normalize(self.identity_proxies[:count], dim=1)
        logits = F.normalize(shared, dim=1) @ proxies.t() / float(temperature)
        return logits, targets

    def cycle_reconstruct(
        self, recovered_target: torch.Tensor, direction: int
    ) -> torch.Tensor:
        return F.normalize(self.cycle_heads[direction](recovered_target), dim=1)

    def _balanced_base_weights(self, direction: int) -> torch.Tensor:
        probabilities = torch.softmax(self.base_weight_logits[direction], dim=0)
        return self.branch_floor + (1.0 - 3.0 * self.branch_floor) * probabilities

    def score_from_encoding(
        self,
        available: dict[str, torch.Tensor],
        available_modality: str,
        memory: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor | str]:
        target_modality, direction = self._direction(available_modality)
        labels = memory["labels"]
        branches = torch.stack(
            [
                available["embedding"] @ memory[f"{available_modality}_embedding"].t(),
                available["shared"] @ memory[f"{available_modality}_shared"].t(),
                available["shared"] @ memory[f"{target_modality}_shared"].t(),
            ],
            dim=2,
        )
        base_weights = self._balanced_base_weights(direction)
        base_scores = (branches * base_weights.view(1, 1, 3)).sum(dim=2)
        temperature = self.log_temperatures[direction].exp().clamp(0.003, 0.2)
        posterior = torch.softmax(base_scores / temperature, dim=1)
        posterior_used = posterior
        retrieval_drop_mask = torch.zeros(
            posterior.size(0), dtype=torch.bool, device=posterior.device
        )
        if self.training and self.retrieval_dropout > 0.0:
            retrieval_drop_mask = (
                torch.rand(posterior.size(0), device=posterior.device)
                < self.retrieval_dropout
            )
            uniform = torch.full_like(posterior, 1.0 / posterior.size(1))
            posterior_used = torch.where(
                retrieval_drop_mask.unsqueeze(1), uniform, posterior
            )
        retrieved_target = posterior_used @ memory[f"{target_modality}_embedding"]
        retrieved_specific = posterior_used @ memory[f"{target_modality}_specific"]
        recovered, predicted_specific, log_variance = self.recovery_decoder(
            available["tokens"],
            available["shared"],
            retrieved_target,
            retrieved_specific,
            direction,
        )
        recovered_scores = recovered @ memory[f"{target_modality}_embedding"].t()
        posterior_entropy = -(
            posterior * posterior.clamp_min(1e-12).log()
        ).sum(dim=1) / math.log(posterior.size(1))
        top_two = base_scores.topk(k=2, dim=1).values
        margin = top_two[:, 0] - top_two[:, 1]
        standardized_margin = margin / base_scores.std(
            dim=1, unbiased=False
        ).clamp_min(1e-3)
        base_top = base_scores.max(dim=1)
        recovered_top = recovered_scores.max(dim=1)
        agreement = base_top.indices.eq(recovered_top.indices).float()
        direction_features = F.one_hot(
            torch.full(
                (base_scores.size(0),),
                direction,
                device=base_scores.device,
                dtype=torch.long,
            ),
            num_classes=2,
        ).to(base_scores)
        gate_features = torch.cat(
            [
                posterior_entropy.unsqueeze(1),
                posterior.max(dim=1).values.unsqueeze(1),
                standardized_margin.unsqueeze(1),
                base_top.values.unsqueeze(1),
                recovered_top.values.unsqueeze(1),
                (recovered_top.values - base_top.values).unsqueeze(1),
                agreement.unsqueeze(1),
                log_variance.exp().unsqueeze(1),
                direction_features,
            ],
            dim=1,
        )
        recovery_weight = self.safe_gate(gate_features)
        shared_weight = 1.0 - recovery_weight
        fused_scores = (
            shared_weight.unsqueeze(1) * base_scores
            + recovery_weight.unsqueeze(1) * recovered_scores
        )
        cycle = self.cycle_reconstruct(recovered, direction)
        return {
            "shared": available["shared"],
            "shared_raw": available["shared_raw"],
            "specific": available["specific"],
            "predicted_specific": predicted_specific,
            "mean": recovered,
            "cycle": cycle,
            "log_variance": log_variance,
            "posterior": posterior,
            "posterior_confidence": posterior.max(dim=1).values,
            "posterior_entropy": posterior_entropy,
            "recovery_reliability": margin,
            "retrieval_drop_mask": retrieval_drop_mask,
            "retrieval_dropout_fraction": retrieval_drop_mask.float().mean(),
            "gate_features": gate_features,
            "learned_gate": recovery_weight,
            "recovery_gate": recovery_weight,
            "recovery_weight": recovery_weight,
            "shared_weight": shared_weight,
            "base_weights": base_weights.expand(base_scores.size(0), -1),
            "base_branch_scores": branches,
            "base_scores": base_scores,
            "recovered_scores": recovered_scores,
            "fused_scores": fused_scores,
            "candidate_labels": labels,
            "target_modality": target_modality,
            "temperature": temperature,
        }
