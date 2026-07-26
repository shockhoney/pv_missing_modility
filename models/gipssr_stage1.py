"""Stage-1 shared alignment and recovery initialization for GIPSSR-Net."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cuef import ConflictAwareUncertaintyEvidentialFusion
from models.gipssr_components import GIPSSRCore, _templates


STAGE1_ARCHITECTURE_VERSION = "gipssr_cuef_stage1_v2"


class GIPSSRStage1(GIPSSRCore):
    """IGDCA shared alignment plus bounded Transformer recovery fusion."""

    def __init__(
        self,
        input_dim: int = 256,
        shared_dim: int = 192,
        specific_dim: int = 128,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.1,
        max_recovery_weight: float = 0.75,
        min_recovery_weight: float = 0.15,
        retrieval_dropout: float = 0.10,
        branch_floor: float = 0.02,
        max_proxy_identities: int = 512,
        ablation: str = "full",
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
            ablation=ablation,
        )
        self.retrieval_dropout = float(retrieval_dropout)
        self.branch_floor = float(branch_floor)
        self.max_proxy_identities = int(max_proxy_identities)
        self.cuef = ConflictAwareUncertaintyEvidentialFusion(
            dropout=dropout,
            min_recovery_weight=min_recovery_weight,
            max_recovery_weight=max_recovery_weight,
            base_branch_floor=branch_floor,
            ablation=ablation,
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
        return self.cuef.min_recovery_weight

    @property
    def max_recovery_weight(self) -> float:
        return self.cuef.max_recovery_weight

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
        if self.ablation == "without_igdca":
            self.identity_proxies[: identities.numel()].zero_()
            return
        palm_shared = self.palm_igdca(palm)
        vein_shared = self.vein_igdca(vein)
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

    def fuse_score_branches(
        self,
        base_branch_scores: torch.Tensor,
        recovered_scores: torch.Tensor,
        log_variance: torch.Tensor,
        direction: int,
    ) -> dict[str, torch.Tensor]:
        branches = torch.cat(
            [base_branch_scores, recovered_scores.unsqueeze(2)], dim=2
        )
        external_uncertainty = branches.new_zeros(
            branches.size(0), branches.size(2)
        )
        external_uncertainty[:, 3] = log_variance.exp()
        return self.cuef(
            branches,
            direction,
            external_uncertainty=external_uncertainty,
        )

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
        retrieval_fusion = self.cuef(branches, direction)
        base_scores = retrieval_fusion["fused_scores"]
        posterior = torch.softmax(base_scores / 0.05, dim=1)
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
        fusion = self.fuse_score_branches(
            branches, recovered_scores, log_variance, direction
        )
        branch_weights = fusion["weights"]
        recovery_weight = branch_weights[:, 3]
        shared_weight = 1.0 - recovery_weight
        base_weights = branch_weights[:, :3] / shared_weight.unsqueeze(1).clamp_min(1e-8)
        fused_scores = fusion["fused_scores"]
        posterior_entropy = -(
            posterior * posterior.clamp_min(1e-12).log()
        ).sum(dim=1) / math.log(posterior.size(1))
        top_two = base_scores.topk(k=2, dim=1).values
        margin = top_two[:, 0] - top_two[:, 1]
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
            "recovery_weight": recovery_weight,
            "shared_weight": shared_weight,
            "base_weights": base_weights,
            "branch_weights": branch_weights,
            "base_branch_scores": branches,
            "calibrated_branch_scores": fusion["calibrated_scores"],
            "base_scores": base_scores,
            "recovered_scores": recovered_scores,
            "fused_scores": fused_scores,
            "fusion_evidence": fusion["evidence"],
            "fusion_uncertainty": fusion["uncertainty"],
            "fusion_conflict": fusion["conflict"],
            "fusion_conflict_scale": fusion["conflict_scale"],
            "calibration_scale": fusion["calibration_scale"],
            "retrieval_weights": retrieval_fusion["weights"],
            "candidate_labels": labels,
            "target_modality": target_modality,
        }
