"""Shared representation and metric helpers for Tongji comparison methods."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.evaluation import score_matrix_metrics
from utils.scenarios import PALMPRINT_MISSING, PALMVEIN_MISSING


SCENARIOS = (PALMPRINT_MISSING, PALMVEIN_MISSING)


def identity_templates(
    representations: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average samples per sorted identity and return unit-normalized templates."""

    labels = labels.to(device=representations.device, dtype=torch.long)
    identities = labels.unique(sorted=True)
    templates = torch.stack(
        [representations[labels == identity].mean(dim=0) for identity in identities]
    )
    return F.normalize(templates, dim=1), identities


def representation_scores(
    probes: torch.Tensor, templates: torch.Tensor
) -> torch.Tensor:
    return F.normalize(probes, dim=1) @ F.normalize(templates, dim=1).t()


def metrics_from_representations(
    probes: torch.Tensor,
    probe_labels: torch.Tensor,
    templates: torch.Tensor,
    template_labels: torch.Tensor,
) -> dict:
    return score_matrix_metrics(
        representation_scores(probes, templates).detach().float().cpu(),
        template_labels.detach().cpu(),
        probe_labels.detach().cpu(),
        topk=(1, 5),
        far_points=(1e-3, 1e-4),
        warn_far_resolution=False,
    )


def validation_rank(results: dict) -> tuple[float, float, float, float]:
    """Match the existing GIPSSR identity-disjoint validation selection rule."""

    fused = [results[scenario]["fused"] for scenario in SCENARIOS]
    return (
        sum(item["eer"] for item in fused) / len(fused),
        max(item["eer"] for item in fused),
        -sum(item["tar_at_far"][1e-4] for item in fused) / len(fused),
        -sum(item["tar_at_far"][1e-3] for item in fused) / len(fused),
    )
