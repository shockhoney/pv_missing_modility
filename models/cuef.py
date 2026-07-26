"""CUEF: conflict-aware uncertainty-calibrated evidential score fusion."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConflictAwareUncertaintyEvidentialFusion(nn.Module):
    """Fuse fixed branch evidence without depending on gallery cardinality.

    Each score branch is summarized into a fixed-size evidence token. Branch-wise
    calibration aligns score scales, a one-layer token Transformer models branch
    conflict, and learned evidence precision yields bounded sample-wise weights.
    """

    BRANCHES = ("available", "shared_same", "shared_cross", "recovered")
    FUSION_ABLATIONS = {
        "without_cuef_calibration",
        "without_cuef_conflict",
        "without_cuef_uncertainty",
    }

    def __init__(
        self,
        dropout: float,
        min_recovery_weight: float = 0.15,
        max_recovery_weight: float = 0.75,
        base_branch_floor: float = 0.0,
        model_dim: int = 64,
        heads: int = 4,
        ablation: str = "full",
    ) -> None:
        super().__init__()
        if not 0.0 < min_recovery_weight < max_recovery_weight < 1.0:
            raise ValueError("Recovery bounds must satisfy 0 < min < max < 1")
        if not 0.0 <= base_branch_floor < 1.0 / 3.0:
            raise ValueError("Base branch floor must be in [0, 1/3)")
        if model_dim % heads:
            raise ValueError("CUEF model_dim must be divisible by heads")
        self.min_recovery_weight = float(min_recovery_weight)
        self.max_recovery_weight = float(max_recovery_weight)
        self.base_branch_floor = float(base_branch_floor)
        self.ablation = ablation
        self.use_calibration = ablation != "without_cuef_calibration"
        self.use_conflict = ablation != "without_cuef_conflict"
        self.use_uncertainty = ablation != "without_cuef_uncertainty"

        # Differentiable cohort normalization aligns per-probe branch scales; the
        # learned residual scale depends only on missing direction and branch.
        self.calibration_log_scale = nn.Parameter(torch.zeros(2, 4))
        self.branch_embedding = nn.Parameter(torch.empty(4, model_dim))
        self.direction_embedding = nn.Parameter(torch.empty(2, model_dim))
        self.feature_projection = nn.Sequential(
            nn.Linear(9, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )
        block = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.conflict_encoder = nn.TransformerEncoder(
            block, num_layers=1, norm=nn.LayerNorm(model_dim)
        )
        self.evidence_head = nn.Linear(model_dim, 1)
        self.uncertainty_head = nn.Linear(model_dim, 1)
        self.conflict_scale_logit = nn.Parameter(torch.tensor(0.0))
        # Identity-preserving initialization keeps shared-same dominant for retrieval
        # while assigning the recovered branch a nonzero 0.30 prior.
        initial_recovery_probability = (
            (0.30 - min_recovery_weight)
            / (max_recovery_weight - min_recovery_weight)
        )
        recovery_prior = math.log(
            initial_recovery_probability / (1.0 - initial_recovery_probability)
        )
        self.branch_prior = nn.Parameter(
            torch.tensor(
                [
                    [-4.0, 4.0, -4.0, recovery_prior],
                    [-4.0, 4.0, -4.0, recovery_prior],
                ]
            )
        )
        nn.init.trunc_normal_(self.branch_embedding, std=0.02)
        nn.init.trunc_normal_(self.direction_embedding, std=0.02)
        nn.init.normal_(self.evidence_head.weight, std=0.02)
        nn.init.zeros_(self.evidence_head.bias)
        nn.init.normal_(self.uncertainty_head.weight, std=0.02)
        nn.init.constant_(self.uncertainty_head.bias, -2.0)

    def _calibrate(
        self, scores: torch.Tensor, direction: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = scores.size(2)
        if self.use_calibration:
            centered = scores - scores.mean(dim=1, keepdim=True)
            spread = scores.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-4)
            scale = self.calibration_log_scale[direction, :count].exp().clamp(0.25, 4.0)
            calibrated = 0.05 * centered / spread * scale.view(1, 1, count)
        else:
            scale = scores.new_ones(count)
            calibrated = scores
        return calibrated, scale

    @staticmethod
    def _score_statistics(
        calibrated: torch.Tensor,
        external_uncertainty: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, identities, branches = calibrated.shape
        if identities < 2:
            raise ValueError("CUEF requires at least two gallery identities")
        top = calibrated.topk(2, dim=1)
        top1 = top.values[:, 0]
        margin = top.values[:, 0] - top.values[:, 1]
        mean = calibrated.mean(dim=1)
        std = calibrated.std(dim=1, unbiased=False).clamp_min(1e-6)
        probabilities = torch.softmax(calibrated / 0.05, dim=1)
        confidence = probabilities.max(dim=1).values
        entropy = -(
            probabilities * probabilities.clamp_min(1e-8).log()
        ).sum(dim=1) / math.log(identities)
        consensus = probabilities.mean(dim=2, keepdim=True).clamp_min(1e-8)
        conflict = (
            probabilities
            * (probabilities.clamp_min(1e-8).log() - consensus.log())
        ).sum(dim=1)
        top_indices = top.indices[:, 0]
        agreement = (
            top_indices.unsqueeze(2).eq(top_indices.unsqueeze(1)).float().mean(dim=2)
        )
        features = torch.stack(
            [
                top1,
                margin,
                mean,
                std,
                confidence,
                1.0 - entropy,
                agreement,
                conflict,
                external_uncertainty.clamp_min(0.0).log1p(),
            ],
            dim=2,
        )
        if features.shape != (batch, branches, 9):
            raise RuntimeError("Unexpected CUEF evidence feature shape")
        return features, conflict

    def forward(
        self,
        scores: torch.Tensor,
        direction: int,
        external_uncertainty: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if scores.ndim != 3 or not 2 <= scores.size(2) <= 4:
            raise ValueError("CUEF scores must have shape [batch, identities, 2..4]")
        batch, _, count = scores.shape
        if external_uncertainty is None:
            external_uncertainty = scores.new_zeros(batch, count)
        if external_uncertainty.shape != (batch, count):
            raise ValueError("CUEF external uncertainty has incompatible shape")
        calibrated, scale = self._calibrate(scores, direction)
        features, conflict = self._score_statistics(
            calibrated, external_uncertainty
        )
        if not self.use_conflict:
            features = features.clone()
            features[:, :, 7] = 0.0
            conflict = conflict.new_zeros(conflict.shape)
        if not self.use_uncertainty:
            features = features.clone()
            features[:, :, 8] = 0.0

        tokens = self.feature_projection(features)
        tokens = tokens + self.branch_embedding[:count].unsqueeze(0)
        tokens = tokens + self.direction_embedding[direction].view(1, 1, -1)
        if self.use_conflict:
            tokens = self.conflict_encoder(tokens)
        evidence = self.evidence_head(tokens).squeeze(2)
        evidence = evidence + self.branch_prior[direction, :count]
        predicted_uncertainty = F.softplus(
            self.uncertainty_head(tokens).squeeze(2)
        )
        if self.use_uncertainty:
            predicted_uncertainty = predicted_uncertainty + external_uncertainty
        else:
            predicted_uncertainty = predicted_uncertainty.new_zeros(
                predicted_uncertainty.shape
            )
        conflict_scale = F.softplus(self.conflict_scale_logit)
        logits = evidence - predicted_uncertainty.log1p()
        if self.use_conflict:
            logits = logits - conflict_scale * conflict

        if count == 4:
            recovery_probability = torch.sigmoid(logits[:, 3])
            recovery_weight = self.min_recovery_weight + (
                self.max_recovery_weight - self.min_recovery_weight
            ) * recovery_probability
            base_probability = torch.softmax(logits[:, :3], dim=1)
            floor = self.base_branch_floor
            base_probability = floor + (1.0 - 3.0 * floor) * base_probability
            weights = torch.cat(
                [
                    (1.0 - recovery_weight).unsqueeze(1) * base_probability,
                    recovery_weight.unsqueeze(1),
                ],
                dim=1,
            )
        else:
            weights = torch.softmax(logits, dim=1)
            if count == 3 and self.base_branch_floor > 0.0:
                floor = self.base_branch_floor
                weights = floor + (1.0 - 3.0 * floor) * weights
        fused = (calibrated * weights.unsqueeze(1)).sum(dim=2)
        if self.use_calibration:
            fused = fused - fused.mean(dim=1, keepdim=True)
            fused = 0.05 * fused / fused.std(
                dim=1, unbiased=False, keepdim=True
            ).clamp_min(1e-4)
        return {
            "fused_scores": fused,
            "calibrated_scores": calibrated,
            "weights": weights,
            "evidence": evidence,
            "uncertainty": predicted_uncertainty,
            "conflict": conflict,
            "calibration_scale": scale,
            "conflict_scale": conflict_scale,
        }
