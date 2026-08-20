"""Summarize Tongji validation-only hyperparameter metrics."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(
    "outputs/gipssr/hyperparameter_experiments/tongji/seed_42/validation_results"
)
ROWS = (
    ("K=3", "k_3"),
    ("K=5", "default"),
    ("K=8", "k_8"),
    ("alpha=0.10", "alpha_0.10"),
    ("alpha=0.25", "default"),
    ("alpha=0.50", "alpha_0.50"),
    ("wmax=0.55", "wmax_0.55"),
    ("wmax=0.75", "default"),
    ("wmax=0.95", "wmax_0.95"),
)
SCENARIOS = ("palmprint_missing", "palmvein_missing")


def main() -> None:
    payloads = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in ROOT.glob("*.json")
    }
    print(
        "setting,PM_EER,PM_Top1,PM_TAR1e-3,PM_TAR1e-4,"
        "VM_EER,VM_Top1,VM_TAR1e-3,VM_TAR1e-4"
    )
    for label, name in ROWS:
        values = []
        for scenario in SCENARIOS:
            metrics = payloads[name]["results"][scenario]["fused"]
            values.extend(
                (
                    100.0 * metrics["eer"],
                    100.0 * metrics["topk"]["1"],
                    100.0 * metrics["tar_at_far"]["0.001"],
                    100.0 * metrics["tar_at_far"]["0.0001"],
                )
            )
        print(label + "," + ",".join(f"{value:.12f}" for value in values))

    gallery_hashes = {item["gallery_protocol_sha256"] for item in payloads.values()}
    probe_hashes = {item["probe_protocol_sha256"] for item in payloads.values()}
    print(f"unique_protocol_hashes={len(gallery_hashes)}/{len(probe_hashes)}")
    for name, payload in sorted(payloads.items()):
        counts = []
        for scenario in SCENARIOS:
            metrics = payload["results"][scenario]["fused"]
            counts.append(
                (
                    scenario,
                    metrics["num_gallery_identities"],
                    metrics["num_probes"],
                    metrics["num_impostor_scores"],
                )
            )
        print(name, payload["training_stage"], payload["best_epoch"], counts)


if __name__ == "__main__":
    main()
