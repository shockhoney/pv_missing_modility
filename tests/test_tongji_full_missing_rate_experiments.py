import copy

import pytest
import torch

from run_missing_rate_experiments import mask_sha256
from run_tongji_full_missing_rate_experiments import (
    SCHEMA_VERSION,
    evaluate_score_matrices,
    masks_from_reference,
    result_is_current,
)


def _reference():
    masks = {}
    previous = []
    for rate, indices in ((0.5, [1, 3]), (1.0, [0, 1, 2, 3])):
        mask = torch.zeros(4, dtype=torch.bool)
        mask[indices] = True
        masks[str(int(rate * 100))] = {
            "requested_rate": rate,
            "actual_rate": len(indices) / 4,
            "missing_count": len(indices),
            "total_probe_count": 4,
            "missing_indices": indices,
            "mask_sha256": mask_sha256(mask),
            "selection_order_sha256": "x",
        }
        previous = indices
    return {"mask_seed": 42, "masks": masks}


def test_retained_masks_are_reconstructed_and_audited(monkeypatch):
    import run_tongji_full_missing_rate_experiments as module

    monkeypatch.setattr(module, "RATES", (0.5, 1.0))
    masks = masks_from_reference(_reference())
    assert masks["50"]["mask"].tolist() == [False, True, False, True]
    assert torch.all(masks["50"]["mask"] <= masks["100"]["mask"])

    broken = copy.deepcopy(_reference())
    broken["masks"]["50"]["missing_indices"] = [0, 3]
    with pytest.raises(ValueError, match="SHA-256"):
        masks_from_reference(broken)


def test_score_mixing_keeps_all_probes_and_uses_only_selected_rows():
    labels = torch.tensor([0, 1, 0, 1])
    candidates = torch.tensor([0, 1])
    complete = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    missing = complete.flip(1)
    mask = torch.tensor([False, True, False, True])
    masks = {"50": {"mask": mask}}
    scores = {
        "complete": complete,
        "palmprint_missing": missing,
        "palmvein_missing": missing,
    }
    results = evaluate_score_matrices(scores, labels, candidates, masks)
    for scenario in ("palmprint_missing", "palmvein_missing"):
        assert results[scenario]["50"]["num_probes"] == 4
        assert results[scenario]["50"]["topk"][1] == 0.5


def test_resume_gate_binds_checkpoint_protocol_and_masks(tmp_path):
    mask = torch.tensor([True, False])
    masks = {
        "100": {
            "mask": mask,
            "mask_sha256": mask_sha256(mask),
            "missing_count": 1,
            "missing_indices": [0],
            "total_probe_count": 2,
        }
    }
    protocol = {"gallery_sha256": "g", "probe_sha256": "p"}
    path = tmp_path / "dmrnet.json"
    path.write_text(
        __import__("json").dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "method": "dmrnet",
                "checkpoint_sha256": "c",
                "protocol": protocol,
                "masks": {
                    "100": {name: value for name, value in masks["100"].items() if name != "mask"}
                },
            }
        ),
        encoding="utf-8",
    )
    assert result_is_current(
        path,
        method="dmrnet",
        checkpoint_hash="c",
        protocol=protocol,
        masks=masks,
    )
    assert not result_is_current(
        path,
        method="dmrnet",
        checkpoint_hash="different",
        protocol=protocol,
        masks=masks,
    )
