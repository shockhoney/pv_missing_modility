from __future__ import annotations

import math

import torch

from utils.full_comparison_common import metric_rank
from utils.scenarios import PALMPRINT_MISSING, PALMVEIN_MISSING
from utils.evaluation import score_matrix_metrics


def test_score_matrix_metrics_perfect_separation() -> None:
    metrics = score_matrix_metrics(
        torch.tensor([[0.9, 0.1], [0.2, 0.8]]),
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        topk=(1,),
        far_points=(1e-3, 1e-4),
        warn_far_resolution=False,
    )

    assert metrics["eer"] == 0.0
    assert metrics["topk"][1] == 1.0
    assert metrics["tar_at_far"][1e-3] == 1.0
    assert metrics["tar_at_far"][1e-4] == 1.0
    assert metrics["num_gallery_identities"] == 2
    assert metrics["num_probes"] == 2
    assert metrics["num_genuine_scores"] == 2
    assert metrics["num_impostor_scores"] == 2
    assert metrics["far_count_resolution"] == 0.5


def test_score_matrix_metrics_complete_misranking() -> None:
    metrics = score_matrix_metrics(
        torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        topk=(1,),
        far_points=(1e-3, 1e-4),
        warn_far_resolution=False,
    )

    assert metrics["eer"] == 1.0
    assert metrics["topk"][1] == 0.0
    assert metrics["tar_at_far"][1e-3] == 0.0
    assert metrics["tar_at_far"][1e-4] == 0.0


def test_score_matrix_metrics_count_resolution_matches_score_grain() -> None:
    metrics = score_matrix_metrics(
        torch.tensor(
            [
                [1.0, 0.1, 0.2],
                [0.3, 1.0, 0.4],
                [0.5, 0.6, 1.0],
            ]
        ),
        torch.tensor([10, 20, 30]),
        torch.tensor([10, 20, 30]),
        topk=(1, 2),
        far_points=(1e-3, 1e-4),
        warn_far_resolution=False,
    )

    assert metrics["num_genuine_scores"] == 3
    assert metrics["num_impostor_scores"] == 6
    assert math.isclose(metrics["far_count_resolution"], 1.0 / 6.0)
    assert math.isclose(metrics["minimum_nonzero_far"], 1.0 / 6.0)
    assert metrics["topk"] == {1: 1.0, 2: 1.0}


def test_full_checkpoint_rank_balances_both_missing_scenarios() -> None:
    results = {
        PALMPRINT_MISSING: {
            "fused": {
                "eer": 0.1,
                "tar_at_far": {1e-3: 0.4, 1e-4: 0.2},
            }
        },
        PALMVEIN_MISSING: {
            "fused": {
                "eer": 0.3,
                "tar_at_far": {1e-3: 0.2, 1e-4: 0.1},
            }
        },
    }

    rank = metric_rank(results)

    assert math.isclose(rank[0], 0.2)
    assert math.isclose(rank[1], 0.3)
    assert math.isclose(rank[2], -0.15)
    assert math.isclose(rank[3], -0.3)
