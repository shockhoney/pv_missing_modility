import torch

from run_missing_rate_experiments import (
    RATES,
    cohort_normalize_scores,
    complete_multimodal_scores,
    half_up_missing_count,
    mix_score_rows,
    nested_missing_masks,
)


def test_half_up_counts_match_three_locked_test_protocols():
    assert [half_up_missing_count(240, rate) for rate in RATES] == [24, 48, 120, 192, 240]
    assert [half_up_missing_count(116, rate) for rate in RATES] == [12, 23, 58, 93, 116]
    assert [half_up_missing_count(200, rate) for rate in RATES] == [20, 40, 100, 160, 200]


def test_masks_are_reproducible_nested_and_dataset_specific():
    sample_order = [(f"palm-{i}", f"vein-{i}", i // 2) for i in range(116)]
    first = nested_missing_masks(
        116, dataset="cumt", seed=42, sample_order=sample_order
    )
    second = nested_missing_masks(
        116, dataset="cumt", seed=42, sample_order=sample_order
    )
    other = nested_missing_masks(
        116, dataset="tongji", seed=42, sample_order=sample_order
    )
    previous = torch.zeros(116, dtype=torch.bool)
    for key in ("10", "20", "50", "80", "100"):
        assert torch.equal(first[key]["mask"], second[key]["mask"])
        assert not torch.any(previous & ~first[key]["mask"])
        assert first[key]["mask"].sum().item() == first[key]["missing_count"]
        previous = first[key]["mask"]
    assert not torch.equal(first["20"]["mask"], other["20"]["mask"])


def test_complete_score_fusion_has_cuef_row_calibration():
    palm = torch.tensor([[1.0, 0.0, -1.0], [0.2, 0.4, 0.9]])
    vein = torch.tensor([[0.2, 0.4, 0.8], [0.7, 0.1, -0.2]])
    complete = complete_multimodal_scores(palm, vein)
    assert torch.allclose(complete.mean(dim=1), torch.zeros(2), atol=1e-7)
    assert torch.allclose(
        complete.std(dim=1, unbiased=False), torch.full((2,), 0.05), atol=1e-7
    )
    assert torch.allclose(complete, cohort_normalize_scores((palm + vein) / 2))


def test_mixed_scores_use_missing_rows_only():
    complete = torch.arange(12, dtype=torch.float32).view(4, 3)
    missing = -complete
    mask = torch.tensor([False, True, False, True])
    mixed = mix_score_rows(complete, missing, mask)
    assert torch.equal(mixed[~mask], complete[~mask])
    assert torch.equal(mixed[mask], missing[mask])
    assert torch.equal(
        mix_score_rows(complete, missing, torch.ones(4, dtype=torch.bool)), missing
    )
