"""Resumable Tongji missing-rate evaluation for GIPSSR and five full baselines.

The missing rate is the fraction of the *complete test probe population* for
which one target modality is replaced by the corresponding single-modality
inference path.  Gallery samples are always complete.  The exact seed-42,
nested 10/20/50/80/100 percent masks retained by the GIPSSR experiment are
reused so every method is measured on the same probe rows.

This program performs no training.  It writes one atomic JSON file as soon as
each checkpoint is evaluated and skips hash-matched files on a resumed run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import torch

from run_missing_rate_experiments import (
    MASK_VERSION,
    RATES,
    basic_metrics,
    evaluate_scores,
    mask_sha256,
    mix_score_rows,
    rate_key,
)
from run_tongji_full_comparisons import METHOD_ORDER, best_checkpoint, result_path
from test_full_comparison import _saved_input_size, load_method_checkpoint
from utils.checkpoint_io import file_sha256
from utils.comparison_metrics import identity_templates, representation_scores
from utils.full_comparison_common import (
    SCENARIOS,
    atomic_json,
    collect_representations,
    paired_loader,
)
from utils.runtime import resolve_device, set_random_seed


ROOT: Final = Path(__file__).resolve().parent
SCHEMA_VERSION: Final = "tongji_full_missing_rates_v1"
PROTOCOL_NAME: Final = "tongji_session1_id_disjoint_closed_set_v1"
SEED: Final = 42
ALL_METHODS: Final = ("gipssr", *METHOD_ORDER)
DEFAULT_COMPARISON_ROOT: Final = (
    ROOT / "outputs/gipssr/full_comparisons/tongji/seed_42"
)
DEFAULT_GIPSSR_RESULT: Final = (
    ROOT / "outputs/gipssr/missing_rate_experiments/tongji/seed_42/results.json"
)
DEFAULT_OUTPUT_ROOT: Final = DEFAULT_COMPARISON_ROOT / "missing_rates"
GALLERY: Final = ROOT / "data_txt/tongji/ssfd_gallery_full.txt"
PROTOCOL: Final = ROOT / "data_txt/tongji/ssfd_test_protocol.txt"
MANIFEST: Final = ROOT / "data_txt/tongji/manifest.json"
METRICS: Final = ("eer", "top1", "tar_1e-3", "tar_1e-4")


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_methods(values: list[str]) -> tuple[str, ...]:
    requested: list[str] = []
    for value in values:
        requested.extend(item.strip().lower() for item in value.split(","))
    invalid = sorted(set(requested) - set(ALL_METHODS))
    if invalid:
        raise ValueError(f"Unsupported methods: {invalid}")
    selected = set(requested)
    if not selected:
        raise ValueError("At least one method is required")
    return tuple(method for method in ALL_METHODS if method in selected)


def masks_from_reference(reference: dict) -> dict[str, dict]:
    """Restore and fully audit the retained public GIPSSR mask payload."""

    if int(reference.get("mask_seed", -1)) != SEED:
        raise ValueError("GIPSSR missing-rate result does not use mask seed 42")
    public = reference.get("masks")
    if not isinstance(public, dict):
        raise ValueError("GIPSSR missing-rate result has no masks")
    expected_keys = tuple(rate_key(rate) for rate in RATES)
    if set(public) != set(expected_keys):
        raise ValueError("GIPSSR missing-rate grid differs from the locked grid")

    masks: dict[str, dict] = {}
    previous: torch.Tensor | None = None
    for key in expected_keys:
        saved = dict(public[key])
        total = int(saved["total_probe_count"])
        indices = [int(index) for index in saved["missing_indices"]]
        if total <= 0 or len(indices) != len(set(indices)):
            raise ValueError(f"Invalid {key}% retained mask population")
        if any(index < 0 or index >= total for index in indices):
            raise ValueError(f"Out-of-range {key}% retained mask index")
        mask = torch.zeros(total, dtype=torch.bool)
        mask[indices] = True
        if int(mask.sum()) != int(saved["missing_count"]):
            raise ValueError(f"{key}% retained mask count mismatch")
        if mask_sha256(mask) != saved["mask_sha256"]:
            raise ValueError(f"{key}% retained mask SHA-256 mismatch")
        if previous is not None and torch.any(previous & ~mask):
            raise ValueError("Retained missing-rate masks are not nested")
        saved["mask"] = mask
        masks[key] = saved
        previous = mask
    return masks


def public_masks(masks: dict[str, dict]) -> dict[str, dict]:
    return {
        key: {name: value for name, value in block.items() if name != "mask"}
        for key, block in masks.items()
    }


def _protocol(reference: dict) -> dict:
    manifest = _read_json(MANIFEST)
    gallery_hash = file_sha256(GALLERY)
    probe_hash = file_sha256(PROTOCOL)
    if manifest.get("protocol") != PROTOCOL_NAME:
        raise ValueError("Unexpected Tongji protocol manifest")
    expected = manifest.get("sha256", {})
    if expected.get(GALLERY.name) != gallery_hash:
        raise ValueError("Tongji gallery hash differs from manifest")
    if expected.get(PROTOCOL.name) != probe_hash:
        raise ValueError("Tongji probe hash differs from manifest")
    if reference.get("gallery_protocol_sha256") != gallery_hash:
        raise ValueError("GIPSSR and full baselines use different galleries")
    if reference.get("probe_protocol_sha256") != probe_hash:
        raise ValueError("GIPSSR and full baselines use different probes")
    return {
        "name": PROTOCOL_NAME,
        "session": manifest.get("session"),
        "gallery_path": str(GALLERY.resolve()),
        "gallery_sha256": gallery_hash,
        "probe_path": str(PROTOCOL.resolve()),
        "probe_sha256": probe_hash,
        "manifest_path": str(MANIFEST.resolve()),
        "manifest_sha256": file_sha256(MANIFEST),
        "test_identities": int(manifest["identity_counts"]["test"]),
        "gallery_samples_per_identity": int(
            manifest["samples_per_identity"]["test_gallery"]
        ),
        "probe_samples_per_identity": int(
            manifest["samples_per_identity"]["test_probe"]
        ),
    }


def _fresh_callback(method: str, model, saved: dict, device: torch.device):
    if method == "ssfd":
        from utils.full_ssfd_experiment import representation_callback

        return representation_callback(model)
    if method == "dmrnet":
        from utils.full_dmrnet_experiment import representation_callback

        return representation_callback(model)
    if method == "hcmig":
        from utils.full_hcmig_experiment import representation_callback

        stochastic = bool(saved["configuration"].get("stochastic_eval", True))
        return representation_callback(
            model,
            device=device,
            stochastic=stochastic,
            seed=SEED + 1009,
        )
    if method == "simmlm":
        from utils.full_simmlm_experiment import representation_callback

        return representation_callback(model)
    from utils.full_mmanet_experiment import representation_callback

    return representation_callback(model)


def _loaders(input_size: int, batch_size: int, num_workers: int):
    gallery = paired_loader(
        GALLERY,
        input_size=input_size,
        batch_size=batch_size,
        num_workers=num_workers,
        train=False,
        seed=SEED,
    )
    probes = {
        split: paired_loader(
            PROTOCOL,
            input_size=input_size,
            batch_size=batch_size,
            num_workers=num_workers,
            train=False,
            split_filter=split,
            seed=SEED,
        )
        for split in ("complete", *SCENARIOS)
    }
    return gallery, probes


def _metric_value(block: dict, metric: str) -> float:
    if metric == "eer":
        return float(block["eer"])
    if metric == "top1":
        values = block["topk"]
        return float(values.get("1", values.get(1)))
    values = block["tar_at_far"]
    key = "0.001" if metric == "tar_1e-3" else "0.0001"
    return float(values.get(key, values.get(float(key))))


def audit_hundred_percent(formal: dict, results: dict) -> dict:
    audit: dict[str, dict] = {}
    for scenario in SCENARIOS:
        expected = formal["results"][scenario]["fused"]
        actual = results[scenario]["100"]
        audit[scenario] = {}
        for metric in METRICS:
            difference = abs(
                _metric_value(expected, metric) - _metric_value(actual, metric)
            )
            audit[scenario][metric] = difference
            if difference > 1e-7:
                raise ValueError(
                    f"100% {scenario}/{metric} differs from formal test by "
                    f"{difference:g}"
                )
    return audit


def evaluate_score_matrices(
    scores: dict[str, torch.Tensor],
    labels: torch.Tensor,
    template_labels: torch.Tensor,
    masks: dict[str, dict],
) -> dict[str, dict]:
    results = {scenario: {} for scenario in SCENARIOS}
    for scenario in SCENARIOS:
        for key, block in masks.items():
            mixed = mix_score_rows(scores["complete"], scores[scenario], block["mask"])
            results[scenario][key] = evaluate_scores(
                mixed, template_labels, labels
            )
    return results


@torch.inference_mode()
def evaluate_baseline(
    method: str,
    checkpoint: Path,
    formal: dict,
    masks: dict[str, dict],
    protocol: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    set_random_seed(SEED)
    loaded_method, model, saved, missing_callback = load_method_checkpoint(
        checkpoint,
        method=method,
        device=device,
        seed=SEED,
    )
    if loaded_method != method:
        raise RuntimeError("Loaded checkpoint method changed unexpectedly")
    input_size = _saved_input_size(method, saved)
    batch_size = int(formal.get("batch_size", 4 if method in {"ssfd", "hcmig"} else 64))
    gallery_loader, probes = _loaders(input_size, batch_size, args.num_workers)

    # Match test_full_comparison's RNG consumption exactly for the two 100%
    # missing paths: gallery -> palmprint_missing -> palmvein_missing.
    gallery, gallery_labels = collect_representations(
        gallery_loader, missing_callback, device
    )
    templates, template_labels = identity_templates(gallery, gallery_labels)
    representations: dict[str, torch.Tensor] = {}
    labels: torch.Tensor | None = None
    for scenario in SCENARIOS:
        values, observed = collect_representations(
            probes[scenario], missing_callback, device
        )
        if labels is not None and not torch.equal(labels, observed):
            raise RuntimeError("Missing-direction probe label orders differ")
        labels = observed
        representations[scenario] = values

    # A fresh seeded HCMIG callback makes the stochastic gallery identical,
    # while allowing complete probes to be evaluated without perturbing the
    # formal missing-path RNG stream above.  Other methods are deterministic.
    complete_callback = _fresh_callback(method, model, saved, device)
    complete_gallery, complete_gallery_labels = collect_representations(
        gallery_loader, complete_callback, device
    )
    if not torch.equal(gallery_labels, complete_gallery_labels):
        raise RuntimeError("Complete and missing gallery label orders differ")
    if not torch.allclose(gallery, complete_gallery, atol=1e-6, rtol=1e-5):
        raise RuntimeError("Fresh callback did not reproduce gallery representations")
    complete, complete_labels = collect_representations(
        probes["complete"], complete_callback, device
    )
    if labels is None or not torch.equal(labels, complete_labels):
        raise RuntimeError("Complete/missing probe label orders differ")
    if labels.numel() != next(iter(masks.values()))["total_probe_count"]:
        raise RuntimeError("Probe count differs from retained missing masks")

    scores = {
        "complete": representation_scores(complete, templates).float().cpu(),
        **{
            scenario: representation_scores(values, templates).float().cpu()
            for scenario, values in representations.items()
        },
    }
    results = evaluate_score_matrices(scores, labels, template_labels, masks)
    audit = audit_hundred_percent(formal, results)
    complete_metrics = evaluate_scores(
        scores["complete"], template_labels, labels
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "tongji",
        "method": method,
        "seed": SEED,
        "mask_seed": SEED,
        "mask_version": MASK_VERSION,
        "rates": list(RATES),
        "protocol": protocol,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "architecture_version": formal.get("architecture_version"),
        "official_commit": formal.get("official_commit"),
        "best_epoch": formal.get("best_epoch"),
        "formal_100_percent_result": str(result_path(args.comparison_root, method).resolve()),
        "formal_100_percent_result_sha256": file_sha256(
            result_path(args.comparison_root, method)
        ),
        "hundred_percent_recomputation_audit": audit,
        "partial_missing_protocol": (
            "complete gallery; a shared nested seed-42 subset of the 240 complete "
            "probes uses the target method's matching single-modality path; all "
            "other probes use its complete multimodal path"
        ),
        "masks": public_masks(masks),
        "complete_0_percent": complete_metrics,
        "results": results,
        "evaluated_at_utc": _utc_now(),
    }


def gipssr_payload(reference: dict, protocol: dict) -> dict:
    """Bind the already-computed GIPSSR seed-42 result into this comparison."""

    checkpoint = Path(reference["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if file_sha256(checkpoint) != reference["checkpoint_sha256"]:
        raise ValueError("GIPSSR reference/checkpoint SHA-256 mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "tongji",
        "method": "gipssr",
        "seed": SEED,
        "mask_seed": SEED,
        "mask_version": MASK_VERSION,
        "rates": list(RATES),
        "protocol": protocol,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "architecture_version": reference.get("architecture_version"),
        "best_epoch": reference.get("best_epoch"),
        "source_result": str(DEFAULT_GIPSSR_RESULT.resolve()),
        "source_result_sha256": file_sha256(DEFAULT_GIPSSR_RESULT),
        "result_reused_without_retraining_or_reinference": True,
        "partial_missing_protocol": (
            "same complete gallery and shared nested seed-42 probe masks; "
            "retained GIPSSR seed-42 score-matrix results"
        ),
        "masks": reference["masks"],
        "complete_0_percent": reference["complete_0_percent"],
        "results": reference["results"],
        "evaluated_at_utc": _utc_now(),
    }


def result_is_current(
    path: Path,
    *,
    method: str,
    checkpoint_hash: str,
    protocol: dict,
    masks: dict[str, dict],
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _read_json(path)
        return (
            payload.get("schema_version") == SCHEMA_VERSION
            and payload.get("method") == method
            and payload.get("checkpoint_sha256") == checkpoint_hash
            and payload.get("protocol", {}).get("gallery_sha256")
            == protocol["gallery_sha256"]
            and payload.get("protocol", {}).get("probe_sha256")
            == protocol["probe_sha256"]
            and payload.get("masks") == public_masks(masks)
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _available_payloads(output_root: Path) -> dict[str, dict]:
    payloads = {}
    for method in ALL_METHODS:
        path = output_root / f"{method}.json"
        if path.is_file():
            payloads[method] = _read_json(path)
    return payloads


def long_rows(payloads: dict[str, dict]) -> list[dict]:
    rows = []
    for method in ALL_METHODS:
        if method not in payloads:
            continue
        payload = payloads[method]
        for scenario in SCENARIOS:
            for percent in (0, 10, 20, 50, 80, 100):
                block = (
                    payload["complete_0_percent"]
                    if percent == 0
                    else payload["results"][scenario][str(percent)]
                )
                metrics = basic_metrics(block)
                rows.append(
                    {
                        "dataset": "tongji",
                        "protocol": PROTOCOL_NAME,
                        "seed": SEED,
                        "mask_seed": SEED,
                        "method": method,
                        "scenario": scenario,
                        "missing_rate_percent": percent,
                        "missing_count": 0 if percent == 0 else payload["masks"][str(percent)]["missing_count"],
                        "total_probes": payload["masks"]["100"]["total_probe_count"],
                        "eer_percent": metrics["eer"] * 100.0,
                        "rank1_percent": metrics["top1"] * 100.0,
                        "tar_far_1e-3_percent": metrics["tar_1e-3"] * 100.0,
                        "tar_far_1e-4_percent": metrics["tar_1e-4"] * 100.0,
                    }
                )
    return rows


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def render_report(payloads: dict[str, dict], pending: list[str]) -> str:
    lines = [
        "# Tongji 不同缺失率统一识别结果",
        "",
        (
            "同一协议、同一 seed-42 嵌套 probe mask；gallery 始终双模态完整。"
            "缺失率 10/20/50/80/100% 分别缺失 24/48/120/192/240 个 probe，"
            "0% 为完整双模态参考。"
        ),
        "",
    ]
    labels = {
        SCENARIOS[0]: "掌纹缺失（掌静脉可用）",
        SCENARIOS[1]: "掌静脉缺失（掌纹可用）",
    }
    for scenario in SCENARIOS:
        lines.extend(
            [
                f"## {labels[scenario]}",
                "",
                "| 方法 | 缺失率 | EER ↓ (%) | Rank-1 ↑ (%) | TAR@FAR=1e-3 ↑ (%) | TAR@FAR=1e-4 ↑ (%) |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in long_rows(payloads):
            if row["scenario"] != scenario:
                continue
            lines.append(
                f"| {row['method']} | {row['missing_rate_percent']}% | "
                f"{row['eer_percent']:.4f} | {row['rank1_percent']:.4f} | "
                f"{row['tar_far_1e-3_percent']:.4f} | "
                f"{row['tar_far_1e-4_percent']:.4f} |"
            )
        lines.append("")
    if pending:
        lines.extend(["待 checkpoint 完成后续跑：" + "、".join(pending) + "。", ""])
    return "\n".join(lines)


def write_summary(output_root: Path, pending: list[str]) -> None:
    payloads = _available_payloads(output_root)
    rows = long_rows(payloads)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "tongji",
        "protocol": PROTOCOL_NAME,
        "seed": SEED,
        "mask_seed": SEED,
        "rates_percent": [0, 10, 20, 50, 80, 100],
        "completed_methods": [method for method in ALL_METHODS if method in payloads],
        "pending_methods": pending,
        "method_results": {
            method: str((output_root / f"{method}.json").resolve())
            for method in ALL_METHODS
            if method in payloads
        },
        "row_count": len(rows),
        "updated_at_utc": _utc_now(),
    }
    atomic_json(output_root / "summary.json", summary)
    _atomic_csv(output_root / "summary.csv", rows)
    report = render_report(payloads, pending)
    report_path = output_root / "summary.md"
    temporary = report_path.with_suffix(".md.tmp")
    temporary.write_text(report, encoding="utf-8")
    os.replace(temporary, report_path)


def run(args: argparse.Namespace) -> dict[str, dict]:
    reference = _read_json(args.gipssr_result)
    masks = masks_from_reference(reference)
    protocol = _protocol(reference)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    pending: list[str] = []

    if "gipssr" in args.methods:
        path = output_root / "gipssr.json"
        payload = gipssr_payload(reference, protocol)
        if args.force or not result_is_current(
            path,
            method="gipssr",
            checkpoint_hash=payload["checkpoint_sha256"],
            protocol=protocol,
            masks=masks,
        ):
            atomic_json(path, payload)
            print(f"[Saved] {path}", flush=True)
        else:
            print(f"[Skip current] {path}", flush=True)
        write_summary(output_root, pending)

    selected_baselines = [method for method in args.methods if method != "gipssr"]
    if selected_baselines:
        device = resolve_device(args.device, require_available=True, announce=True)
    else:
        device = torch.device("cpu")
    for method in selected_baselines:
        checkpoint = best_checkpoint(args.comparison_root, method)
        formal_path = result_path(args.comparison_root, method)
        if not checkpoint.is_file() or not formal_path.is_file():
            pending.append(method)
            write_summary(output_root, pending)
            message = f"[Pending] {method}: checkpoint or formal result is not ready"
            if args.allow_missing:
                print(message, flush=True)
                continue
            raise FileNotFoundError(message)
        checkpoint_hash = file_sha256(checkpoint)
        path = output_root / f"{method}.json"
        if not args.force and result_is_current(
            path,
            method=method,
            checkpoint_hash=checkpoint_hash,
            protocol=protocol,
            masks=masks,
        ):
            print(f"[Skip current] {path}", flush=True)
            write_summary(output_root, pending)
            continue
        formal = _read_json(formal_path)
        if formal.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError(f"{method} formal result is not bound to current checkpoint")
        if formal.get("gallery_protocol_sha256") != protocol["gallery_sha256"]:
            raise ValueError(f"{method} gallery protocol mismatch")
        if formal.get("probe_protocol_sha256") != protocol["probe_sha256"]:
            raise ValueError(f"{method} probe protocol mismatch")
        print(f"[Evaluate] {method}", flush=True)
        payload = evaluate_baseline(
            method, checkpoint, formal, masks, protocol, args, device
        )
        atomic_json(path, payload)
        print(f"[Saved] {path}", flush=True)
        write_summary(output_root, pending)
        del payload
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pending = [
        method
        for method in METHOD_ORDER
        if method in args.methods and not (output_root / f"{method}.json").is_file()
    ]
    write_summary(output_root, pending)
    return _available_payloads(output_root)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable Tongji missing-rate evaluation of GIPSSR + full baselines."
    )
    parser.add_argument("--methods", nargs="+", default=list(ALL_METHODS))
    parser.add_argument("--comparison-root", type=Path, default=DEFAULT_COMPARISON_ROOT)
    parser.add_argument("--gipssr-result", type=Path, default=DEFAULT_GIPSSR_RESULT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--allow-missing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    args.methods = _normalize_methods(args.methods)
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    return args


if __name__ == "__main__":
    run(parse_args())
