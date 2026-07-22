"""Summarize the retained staged-DCCA recovery experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASETS = ("tongji", "cumt", "polyu")
SCENARIOS = ("palmprint_missing", "palmvein_missing")


def mean(rows, section, field):
    return sum(row[section][field] for row in rows) / len(rows)


def load_rows(root: Path):
    rows = []
    for dataset in DATASETS:
        payload = json.loads((root / dataset / "test_metrics.json").read_text())
        for scenario in SCENARIOS:
            result = payload["results"][scenario]
            single = result["branches"]["available"]
            fused = result["fused"]
            rows.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "single": {
                        "eer": single["eer"],
                        "top1": single["topk"]["1"],
                        "tar_1e-3": single["tar_at_far"]["0.001"],
                        "tar_1e-4": single["tar_at_far"]["0.0001"],
                    },
                    "fused": {
                        "eer": fused["eer"],
                        "top1": fused["topk"]["1"],
                        "tar_1e-3": fused["tar_at_far"]["0.001"],
                        "tar_1e-4": fused["tar_at_far"]["0.0001"],
                    },
                    "no_recovery_eer": result["fused_without_recovery"]["eer"],
                    "recovered_only_eer": result["branches"]["recovered"]["eer"],
                    "shared_weight": result["shared_weight"]["mean"],
                    "recovery_weight": result["recovery_weight"]["mean"],
                    "recovery_weight_bounds": result["recovery_weight_bounds"],
                    "far_count_resolution": fused["far_count_resolution"],
                }
            )
    return rows


def build(root: Path):
    rows = load_rows(root)
    fields = ("eer", "top1", "tar_1e-3", "tar_1e-4")
    summary = {
        section: {field: mean(rows, section, field) for field in fields}
        for section in ("single", "fused")
    }
    summary["mean_recovery_eer_gain"] = sum(
        row["no_recovery_eer"] - row["fused"]["eer"] for row in rows
    ) / len(rows)
    summary["recovery_weight_range"] = [
        min(row["recovery_weight"] for row in rows),
        max(row["recovery_weight"] for row in rows),
    ]
    payload = {
        "architecture_version": "balanced_staged_dcca_specformer_v9_1",
        "summary": summary,
        "rows": rows,
    }
    (root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Staged DCCA-Transformer v9.1 results",
        "",
        "| Dataset | Missing | Single EER | Fused EER | Single Top-1 | Fused Top-1 | Fused TAR@1e-3 | Fused TAR@1e-4 | Shared / recovery weight |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        missing = "Palmprint" if row["scenario"] == "palmprint_missing" else "Palm-vein"
        lines.append(
            f"| {row['dataset'].upper()} | {missing} | "
            f"{row['single']['eer']*100:.4f}% | {row['fused']['eer']*100:.4f}% | "
            f"{row['single']['top1']*100:.2f}% | {row['fused']['top1']*100:.2f}% | "
            f"{row['fused']['tar_1e-3']*100:.2f}% | "
            f"{row['fused']['tar_1e-4']*100:.2f}% | "
            f"{row['shared_weight']*100:.2f}% / {row['recovery_weight']*100:.2f}% |"
        )
    fused = summary["fused"]
    single = summary["single"]
    lines += [
        "",
        f"Six-direction mean: single EER {single['eer']*100:.4f}%, "
        f"fused EER {fused['eer']*100:.4f}%, fused Top-1 {fused['top1']*100:.2f}%, "
        f"fused TAR@1e-3 {fused['tar_1e-3']*100:.2f}%, "
        f"fused TAR@1e-4 {fused['tar_1e-4']*100:.2f}%.",
        "",
        "CUMT TAR@1e-4 is evaluated at the empirical FAR=0 point because its "
        "impostor-count resolution is 1.5124e-4.",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/dcca_specformer/v9_1")
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args().root)
