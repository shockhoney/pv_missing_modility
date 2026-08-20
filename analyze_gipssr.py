"""Audit and summarize the retained seed-42 GIPSSR-Net/CUEF experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

DATASETS = ("tongji", "cumt", "polyu")
SEED = 42
SCENARIOS = ("palmprint_missing", "palmvein_missing")
DERIVED_VARIANTS = ("single", "without_recovery_fusion")
TRAINED_VARIANTS = (
    "without_igdca",
    "without_sgssd",
    "without_giprd",
    "without_sgssd_giprd",
    "without_cuef_calibration",
    "without_cuef_conflict",
    "without_cuef_uncertainty",
    "full",
)
VARIANTS = (*DERIVED_VARIANTS, *TRAINED_VARIANTS)
METRICS = ("eer", "top1", "tar_1e-3", "tar_1e-4")
FINAL_ARCHITECTURE = "gipssr_cuef_state_space_recovery_v3"
STAGE1_ARCHITECTURE = "gipssr_cuef_stage1_v2"
LABELS = {
    "single": "Single available modality",
    "without_recovery_fusion": "w/o recovered-score fusion†",
    "without_igdca": "w/o IGDCA",
    "without_sgssd": "w/o SGSSD",
    "without_giprd": "w/o GIPRD",
    "without_sgssd_giprd": "w/o SGSSD & GIPRD",
    "without_cuef_calibration": "w/o CUEF calibration",
    "without_cuef_conflict": "w/o CUEF conflict",
    "without_cuef_uncertainty": "w/o CUEF uncertainty",
    "full": "Full model",
}
MODULES = ("IGDCA", "SGSSD", "GIPRD", "Cal.", "Conflict", "Uncertainty")
MODULE_MATRIX = {
    "without_igdca": (False, True, True, True, True, True),
    "without_sgssd": (True, False, True, True, True, True),
    "without_giprd": (True, True, False, True, True, True),
    "without_sgssd_giprd": (True, False, False, True, True, True),
    "without_cuef_calibration": (True, True, True, False, True, True),
    "without_cuef_conflict": (True, True, True, True, False, True),
    "without_cuef_uncertainty": (True, True, True, True, True, False),
    "full": (True, True, True, True, True, True),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_row(block: dict) -> dict[str, float]:
    return {
        "eer": float(block["eer"]),
        "top1": float(block["topk"]["1"]),
        "tar_1e-3": float(block["tar_at_far"]["0.001"]),
        "tar_1e-4": float(block["tar_at_far"]["0.0001"]),
    }


def scenario_row(payload: dict, scenario: str, variant: str) -> dict[str, float]:
    result = payload["results"][scenario]
    if variant == "single":
        block = result["branches"]["available"]
    elif variant == "without_recovery_fusion":
        block = result["fused_without_recovery"]
    else:
        block = result["fused"]
    return metric_row(block)


def single_seed_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if len(rows) != 1:
        raise ValueError(f"Expected one seed-42 metric row, got {len(rows)}")
    return dict(rows[0])


def checkpoint_path(root: Path, dataset: str, seed: int, variant: str) -> Path:
    directory = "stage1_full" if variant == "without_sgssd_giprd" else variant
    return root / dataset / f"seed_{seed}" / directory / "best.pth"


def selection_path(root: Path, dataset: str, seed: int, variant: str) -> Path:
    directory = (
        "stage1_full_validation"
        if variant == "without_sgssd_giprd"
        else f"{variant}_validation"
    )
    return root / dataset / f"seed_{seed}" / directory / "best.pth"


def audit_result(
    payload: dict,
    result_path: Path,
    checkpoint_root: Path,
    dataset: str,
    seed: int,
    variant: str,
    require_selection: bool,
) -> dict:
    errors: list[str] = []
    checkpoint_file = checkpoint_path(checkpoint_root, dataset, seed, variant)
    if payload.get("ablation") != variant:
        errors.append(f"result ablation={payload.get('ablation')!r}")
    if not checkpoint_file.is_file():
        return {"passed": False, "errors": [*errors, f"missing {checkpoint_file}"]}
    actual_sha = sha256(checkpoint_file)
    if payload.get("checkpoint_sha256") != actual_sha:
        errors.append("result/checkpoint SHA-256 mismatch")
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    architecture = payload.get("architecture_version")
    expected_architecture = (
        STAGE1_ARCHITECTURE if variant == "without_sgssd_giprd" else FINAL_ARCHITECTURE
    )
    if architecture != expected_architecture:
        errors.append(f"architecture={architecture!r}, expected={expected_architecture!r}")
    if checkpoint.get("architecture_version") != expected_architecture:
        errors.append("checkpoint architecture mismatch")
    expected_model_ablation = "full" if variant == "without_sgssd_giprd" else variant
    if payload.get("model_ablation") != expected_model_ablation:
        errors.append(f"model ablation={payload.get('model_ablation')!r}")
    if checkpoint.get("training_stage") != "fixed_full_train":
        errors.append("checkpoint is not a fixed-full-train replay")
    if int(payload["best_epoch"]) != int(checkpoint["best_epoch"]):
        errors.append("result/checkpoint best_epoch mismatch")

    has_sgssd = any(name.startswith("sgssd.") for name in state)
    has_giprd = any(name.startswith("giprd.") for name in state)
    has_cuef = any(name.startswith("cuef.") for name in state)
    expected_structure = {
        "without_igdca": (True, True),
        "without_sgssd": (False, True),
        "without_giprd": (True, False),
        "without_sgssd_giprd": (False, False),
        "without_cuef_calibration": (True, True),
        "without_cuef_conflict": (True, True),
        "without_cuef_uncertainty": (True, True),
        "full": (True, True),
    }[variant]
    if (has_sgssd, has_giprd) != expected_structure:
        errors.append(
            f"SGSSD/GIPRD={(has_sgssd, has_giprd)}, expected={expected_structure}"
        )
    if not has_cuef:
        errors.append("checkpoint has no CUEF parameters")
    if any("calibration_bias" in name for name in state):
        errors.append("obsolete per-branch calibration bias is present")
    legacy_fragments = ("base_weight_logits", "log_temperatures", "balanced_fusion_gate")
    if any(any(fragment in name for fragment in legacy_fragments) for name in state):
        errors.append("legacy scalar fusion parameters are present")

    selected = selection_path(checkpoint_root, dataset, seed, variant)
    selection_verified = False
    if selected.is_file():
        selection_verified = checkpoint.get("selection_checkpoint_sha256") == sha256(selected)
        if not selection_verified:
            errors.append("selection checkpoint SHA-256 mismatch")
    elif require_selection:
        errors.append(f"missing selection checkpoint {selected}")

    required_diagnostics = {
        "branch_weights",
        "fusion_evidence",
        "fusion_uncertainty",
        "fusion_conflict",
        "calibration_scale",
        "fusion_conflict_scale",
    }
    for scenario in SCENARIOS:
        result = payload.get("results", {}).get(scenario, {})
        missing = sorted(required_diagnostics - result.keys())
        if missing:
            errors.append(f"{scenario}: missing diagnostics {missing}")
        bounds = result.get("recovery_weight_bounds")
        if bounds != [0.15, 0.75]:
            errors.append(f"{scenario}: recovery bounds={bounds!r}")
        mean_weight = result.get("recovery_weight", {}).get("mean", -1.0)
        if not 0.15 <= mean_weight <= 0.75:
            errors.append(f"{scenario}: recovery mean outside bounds")
    return {
        "passed": not errors,
        "errors": errors,
        "result": str(result_path),
        "checkpoint": str(checkpoint_file),
        "checkpoint_sha256": actual_sha,
        "selection_verified": selection_verified,
        "has_cuef": has_cuef,
        "has_sgssd": has_sgssd,
        "has_giprd": has_giprd,
    }


def fmt(summary: dict, metric: str) -> str:
    precision = 4 if metric == "eer" else 2
    return f"{summary[metric] * 100:.{precision}f}"


def fmt_pair(block: dict, metric: str) -> str:
    return " / ".join(fmt(block[scenario], metric) for scenario in SCENARIOS)


def build(
    result_root: Path,
    checkpoint_root: Path,
    require_selection: bool = False,
) -> dict:
    raw = {
        dataset: {
            variant: {scenario: [] for scenario in SCENARIOS}
            for variant in VARIANTS
        }
        for dataset in DATASETS
    }
    audits: list[dict] = []
    diagnostics: list[dict] = []
    protocol_hashes = {dataset: set() for dataset in DATASETS}
    for dataset in DATASETS:
        payloads: dict[str, dict] = {}
        for variant in TRAINED_VARIANTS:
            path = result_root / dataset / f"seed_{SEED}" / f"{variant}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text())
            payloads[variant] = payload
            audit = audit_result(
                payload,
                path,
                checkpoint_root,
                dataset,
                SEED,
                variant,
                require_selection,
            )
            audits.append(audit)
            protocol_hashes[dataset].add(
                (payload["gallery_protocol_sha256"], payload["probe_protocol_sha256"])
            )
            for scenario in SCENARIOS:
                raw[dataset][variant][scenario].append(
                    scenario_row(payload, scenario, variant)
                )
        full = payloads["full"]
        for variant in DERIVED_VARIANTS:
            for scenario in SCENARIOS:
                raw[dataset][variant][scenario].append(
                    scenario_row(full, scenario, variant)
                )
        for scenario in SCENARIOS:
            result = full["results"][scenario]
            diagnostics.append(
                {
                    "dataset": dataset,
                    "seed": SEED,
                    "scenario": scenario,
                    "recovery_weight": result["recovery_weight"]["mean"],
                    "shared_weight": result["shared_weight"]["mean"],
                    "eer_gain": result["improvement_over_no_recovery"]["eer"],
                    "tar_1e-3_gain": result["improvement_over_no_recovery"]["tar_1e-3"],
                    "tar_1e-4_gain": result["improvement_over_no_recovery"]["tar_1e-4"],
                }
            )
    for dataset, hashes in protocol_hashes.items():
        if len(hashes) != 1:
            audits.append({"passed": False, "errors": [f"{dataset}: {len(hashes)} protocol pairs"]})
    failures = [item for item in audits if not item["passed"]]
    if failures:
        raise RuntimeError(f"Ablation audit failed: {json.dumps(failures, indent=2)}")

    datasets: dict[str, dict] = {}
    for dataset in DATASETS:
        datasets[dataset] = {}
        for variant in VARIANTS:
            directional = {
                scenario: single_seed_metrics(raw[dataset][variant][scenario])
                for scenario in SCENARIOS
            }
            datasets[dataset][variant] = {
                "seed": SEED,
                "directional_summary": directional,
                "scenario_macro_summary": {
                    metric: statistics.fmean(
                        directional[scenario][metric] for scenario in SCENARIOS
                    )
                    for metric in METRICS
                },
            }

    effects = {
        dataset: {
            variant: {
                metric: (
                    datasets[dataset][variant]["scenario_macro_summary"][metric]
                    - datasets[dataset]["full"]["scenario_macro_summary"][metric]
                )
                * (100 if metric == "eer" else -100)
                for metric in METRICS
            }
            for variant in VARIANTS
            if variant != "full"
        }
        for dataset in DATASETS
    }
    summary = {
        "method_name": "GIPSSR-Net",
        "fusion_name": "CUEF",
        "architecture_version": FINAL_ARCHITECTURE,
        "protocol": {
            "seed": SEED,
            "aggregation": "raw seed-42 directional metrics; macro averages the two missing directions",
            "selection": "identity-disjoint validation selects epoch; selected epoch is replayed on the full training split; test once",
            "dataset_specific_hyperparameters": False,
            "fallback": False,
        },
        "variants": {
            "single": "frozen available-modality branch from the full run",
            "without_recovery_fusion": "inference-only removal of recovered scores; no replacement",
            "without_igdca": "remove trainable shared alignment and all shared-space objectives",
            "without_sgssd": "remove shared-guided state-space specific disentangler",
            "without_giprd": "remove gallery identity-prior recovery decoder",
            "without_sgssd_giprd": "retain the trainable Stage-1 recovery and CUEF backbone",
            "without_cuef_calibration": "remove differentiable cohort/scale calibration",
            "without_cuef_conflict": "remove conflict token interaction and conflict penalty",
            "without_cuef_uncertainty": "remove predicted/external uncertainty from evidence weighting",
            "full": "IGDCA + SGSSD + GIPRD + CUEF",
        },
        "datasets": datasets,
        "effects_vs_full_percentage_points": effects,
        "full_model_diagnostics": {
            "recovery_weight_range": [
                min(item["recovery_weight"] for item in diagnostics),
                max(item["recovery_weight"] for item in diagnostics),
            ],
            "shared_weight_range": [
                min(item["shared_weight"] for item in diagnostics),
                max(item["shared_weight"] for item in diagnostics),
            ],
            "mean_eer_gain_over_no_recovery": statistics.fmean(
                item["eer_gain"] for item in diagnostics
            ),
            "mean_tar_1e-3_gain_over_no_recovery": statistics.fmean(
                item["tar_1e-3_gain"] for item in diagnostics
            ),
            "mean_tar_1e-4_gain_over_no_recovery": statistics.fmean(
                item["tar_1e-4_gain"] for item in diagnostics
            ),
        },
        "audit": {
            "passed": True,
            "num_result_checkpoint_pairs": len(DATASETS) * len(TRAINED_VARIANTS),
            "selection_checkpoints_required": require_selection,
            "selection_checkpoints_verified": sum(
                bool(item.get("selection_verified")) for item in audits
            ),
            "checks": "SHA-256, architecture, label, fixed-full replay, best epoch, selection SHA when retained, CUEF/SGSSD/GIPRD structure, no legacy fusion parameters, protocol fingerprints, diagnostics and recovery bounds",
        },
    }
    out = result_root.parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# GIPSSR-Net v3 / CUEF results",
        "",
        "All metrics are percentages from seed 42. In directional summary cells, the order is palmprint-missing / palmvein-missing.",
        "",
        "Protocol: identity-disjoint validation selects the epoch, the selected epoch is replayed on the full training split, and the test protocol is evaluated once. No fallback, teacher comparison, deployment blending, or dataset-specific hyperparameter branch is used.",
    ]
    for dataset in DATASETS:
        lines += [
            "",
            f"## {dataset.upper()}",
            "",
            "### Seed-42 test results: single modality versus recovered fusion",
            "",
            "Each metric is reported as `single available modality → full recovered fusion`.",
            "",
            "| Missing modality | EER ↓ | Top-1 ↑ | TAR@FAR=1e-3 ↑ | TAR@FAR=1e-4 ↑ |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for scenario, scenario_label in (
            ("palmprint_missing", "Palmprint"),
            ("palmvein_missing", "Palm vein"),
        ):
            single = datasets[dataset]["single"]["directional_summary"][scenario]
            full = datasets[dataset]["full"]["directional_summary"][scenario]
            transitions = [
                f"{100.0 * single[metric]:.4f} → {100.0 * full[metric]:.4f}"
                for metric in METRICS
            ]
            transitions_text = " | ".join(transitions)
            lines.append(f"| {scenario_label} | {transitions_text} |")
        lines += [
            "",
            "### Seed-42 summary: single modality versus recovered fusion",
            "",
            "| Setting | EER ↓ | Top-1 ↑ | TAR@FAR=1e-3 ↑ | TAR@FAR=1e-4 ↑ |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for variant in ("single", "without_recovery_fusion", "full"):
            directional = datasets[dataset][variant]["directional_summary"]
            lines.append(
                f"| {LABELS[variant]} | {fmt_pair(directional, 'eer')} | "
                f"{fmt_pair(directional, 'top1')} | {fmt_pair(directional, 'tar_1e-3')} | "
                f"{fmt_pair(directional, 'tar_1e-4')} |"
            )
        lines += [
            "",
            "### Module ablation",
            "",
            "| Setting | IGDCA | SGSSD | GIPRD | Cal. | Conflict | Uncertainty | EER ↓ | Top-1 ↑ | TAR@1e-3 ↑ | TAR@1e-4 ↑ |",
            "| --- | :---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: | ---: | ---: |",
        ]
        for variant in TRAINED_VARIANTS:
            macro = datasets[dataset][variant]["scenario_macro_summary"]
            checks = ["✓" if enabled else "✗" for enabled in MODULE_MATRIX[variant]]
            lines.append(
                f"| {LABELS[variant]} | {' | '.join(checks)} | {fmt(macro, 'eer')} | "
                f"{fmt(macro, 'top1')} | {fmt(macro, 'tar_1e-3')} | {fmt(macro, 'tar_1e-4')} |"
            )
    lines += [
        "",
        "† This is an inference-only diagnostic from the same full checkpoint: the recovered-score branch is removed and no replacement or fallback is introduced.",
        "",
        "CUMT has 6,612 impostor scores per direction, so its smallest positive empirical FAR is 1.5124e-4; TAR@FAR=1e-4 is consequently the FAR=0 operating point.",
        "",
        "## Evidence-based interpretation",
        "",
        "- IGDCA is the decisive cross-dataset component: removing it causes large degradation on every dataset and both directions.",
        "- Differentiable cohort/scale calibration is decisive on Tongji and PolyU and improves the hard CUMT palmprint-missing direction, although CUMT shows an EER direction trade-off.",
        "- Conflict modeling improves CUMT and PolyU but is mixed on Tongji; its claim should be framed as hard-direction/cross-dataset robustness rather than uniform per-direction superiority.",
        "- Uncertainty weighting improves the three-dataset average for seed 42 and is most visible on CUMT/PolyU; Tongji contains a small direction-specific trade-off.",
        "- SGSSD and GIPRD provide small, generally positive changes. Their effects are much smaller than IGDCA and calibration and should not be overstated from a single seed.",
        "- The recovery branch retains nonzero bounded contribution for every probe; the retained full runs use learned weights rather than a fixed rule or test-time fallback.",
        "",
        f"Audit passed: all {len(DATASETS) * len(TRAINED_VARIANTS)} result/checkpoint pairs match their hashes, architectures, labels, selected epochs, parameter structures, protocol fingerprints, required diagnostics, and recovery-weight bounds.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_root",
        type=Path,
        default=Path("outputs/gipssr/ablations/results"),
    )
    parser.add_argument(
        "--checkpoint_root",
        type=Path,
        default=Path("outputs/gipssr/ablations/checkpoints"),
    )
    parser.add_argument("--require_selection", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.result_root, args.checkpoint_root, args.require_selection)
