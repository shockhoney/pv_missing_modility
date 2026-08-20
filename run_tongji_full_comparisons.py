"""Resume, evaluate and summarize all five full Tongji reproductions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from train_full_comparison import METHODS
from utils.checkpoint_io import file_sha256, safe_torch_load
from utils.full_comparison_common import SCENARIOS, atomic_json


ROOT: Final = Path(__file__).resolve().parent
METHOD_ORDER: Final = ("ssfd", "dmrnet", "hcmig", "simmlm", "mmanet")
DEFAULT_OUTPUT_ROOT: Final = ROOT / "outputs/gipssr/full_comparisons/tongji/seed_42"
DEFAULT_EPOCHS: Final = 100
GALLERY: Final = ROOT / "data_txt/tongji/ssfd_gallery_full.txt"
PROTOCOL: Final = ROOT / "data_txt/tongji/ssfd_test_protocol.txt"
MANIFEST: Final = ROOT / "data_txt/tongji/manifest.json"
RUNNER_SCHEMA: Final = "tongji_full_comparison_run_v2"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sequential resumable training/testing of five full baselines."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHOD_ORDER),
        help="Space- or comma-separated subset, executed in canonical order.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Override every method's main and pretraining-stage epochs; "
            "the paper reproduction default is 100 for each stage."
        ),
    )
    parser.add_argument(
        "--generator-epochs",
        type=int,
        default=None,
        help=(
            "Override generator-stage epochs independently (used for the "
            "quick HCMIG generation-stage run)."
        ),
    )
    parser.add_argument(
        "--hcmig-generation-epoch-cap",
        type=int,
        default=None,
        help="Cap the HCMIG generation stage early and continue with recognition.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--min-epochs", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force-test",
        action="store_true",
        help="Recompute test metrics even if hashes show the JSON is current.",
    )
    return parser


def _normalize_methods(values: list[str]) -> tuple[str, ...]:
    requested: list[str] = []
    for value in values:
        requested.extend(part.strip().lower() for part in value.split(","))
    invalid = sorted({name for name in requested if name not in METHODS})
    if invalid:
        raise ValueError(f"Unsupported methods: {invalid}")
    selected = set(requested)
    if not selected:
        raise ValueError("At least one method is required")
    return tuple(name for name in METHOD_ORDER if name in selected)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.seed != 42:
        raise ValueError("Full comparison experiments are locked to seed 42")
    args.methods = _normalize_methods(args.methods)
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("epochs override must be positive")
    if args.generator_epochs is not None and args.generator_epochs <= 0:
        raise ValueError("generator_epochs override must be positive")
    if (
        args.hcmig_generation_epoch_cap is not None
        and args.hcmig_generation_epoch_cap < 0
    ):
        raise ValueError("hcmig_generation_epoch_cap must be non-negative")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if args.min_epochs <= 0:
        raise ValueError("min_epochs must be positive")
    return args


def method_dir(output_root: Path, method: str) -> Path:
    return output_root / method


def best_checkpoint(output_root: Path, method: str) -> Path:
    return method_dir(output_root, method) / "best.pth"


def result_path(output_root: Path, method: str) -> Path:
    return output_root / "results" / f"{method}.json"


def train_command(args: argparse.Namespace, method: str) -> list[str]:
    epochs = args.epochs or DEFAULT_EPOCHS
    generator_epochs = args.generator_epochs or epochs
    command = [
        sys.executable,
        str(ROOT / "train_full_comparison.py"),
        "--method", method,
        "--output-dir", str(method_dir(Path(args.output_root), method)),
        "--device", args.device,
        "--seed", str(args.seed),
        "--num-workers", str(args.num_workers),
        "--epochs", str(epochs),
        "--teacher-epochs", str(epochs),
        "--expert-epochs", str(epochs),
        "--generator-epochs", str(generator_epochs),
        "--early-stopping-patience", str(args.early_stopping_patience),
        "--min-epochs", str(args.min_epochs),
    ]
    if args.hcmig_generation_epoch_cap is not None:
        command.extend(
            ["--hcmig-generation-epoch-cap", str(args.hcmig_generation_epoch_cap)]
        )
    return command


def test_command(args: argparse.Namespace, method: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "test_full_comparison.py"),
        "--method", method,
        "--checkpoint", str(best_checkpoint(Path(args.output_root), method)),
        "--metrics-path", str(result_path(Path(args.output_root), method)),
        "--gallery-list", str(GALLERY),
        "--protocol-list", str(PROTOCOL),
        "--manifest", str(MANIFEST),
        "--device", args.device,
        "--seed", str(args.seed),
        "--num-workers", str(args.num_workers),
    ]


def run_command(command: list[str], log_path: Path, *, dry_run: bool) -> None:
    print(f"[Command] {' '.join(command)}", flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_utc_now()}] {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def result_is_current(result: Path, checkpoint: Path) -> bool:
    if not result.is_file() or not checkpoint.is_file():
        return False
    try:
        payload = _read_json(result)
        return (
            payload.get("checkpoint_sha256") == file_sha256(checkpoint)
            and payload.get("gallery_protocol_sha256") == file_sha256(GALLERY)
            and payload.get("probe_protocol_sha256") == file_sha256(PROTOCOL)
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _manifest_base(args: argparse.Namespace) -> dict:
    epochs = args.epochs or DEFAULT_EPOCHS
    return {
        "schema_version": RUNNER_SCHEMA,
        "dataset": "tongji",
        "protocol_manifest": str(MANIFEST),
        "protocol_manifest_sha256": file_sha256(MANIFEST),
        "gallery_sha256": file_sha256(GALLERY),
        "probe_sha256": file_sha256(PROTOCOL),
        "methods": list(args.methods),
        "canonical_order": list(METHOD_ORDER),
        "paper_default_epochs_per_stage": DEFAULT_EPOCHS,
        "effective_epochs_per_stage": epochs,
        "generator_epochs_per_stage": args.generator_epochs or epochs,
        "hcmig_generation_epoch_cap": args.hcmig_generation_epoch_cap,
        "early_stopping": {
            "patience": args.early_stopping_patience,
            "min_epochs": args.min_epochs,
            "scope": "validation_selected_main_stage_only",
        },
        "seed": args.seed,
        "device": args.device,
        "runs": {},
        "started_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
    }


def _load_or_create_manifest(path: Path, args: argparse.Namespace) -> dict:
    if not path.is_file():
        return _manifest_base(args)
    payload = _read_json(path)
    base = _manifest_base(args)
    if payload.get("schema_version") != RUNNER_SCHEMA:
        return base
    payload.update(
        {key: value for key, value in base.items()
         if key not in {"runs", "started_at_utc"}}
    )
    payload.setdefault("runs", {})
    return payload


def _metric(payload: dict, scenario: str, section: str, key) -> float:
    value = payload["results"][scenario]["fused"][section]
    if isinstance(value, dict):
        value = value.get(str(key), value.get(key))
    return float(value)


def _training_metadata(output_root: Path, method: str) -> dict:
    best = safe_torch_load(best_checkpoint(output_root, method), "cpu")
    last_path = method_dir(output_root, method) / "last.pth"
    last = safe_torch_load(last_path, "cpu") if last_path.is_file() else {}
    stopping = last.get("early_stopping")
    actual_epochs = last.get("actual_epochs")
    if actual_epochs is None:
        actual_epochs = {"main": int(last.get("epoch", 0))}
    return {
        "best_epoch": int(best.get("best_epoch", best.get("epoch", 0))),
        "actual_epochs": actual_epochs,
        "early_stopped": bool(
            isinstance(stopping, dict) and stopping.get("stopped")
        ),
        "stop_reason": (
            stopping.get("stop_reason") if isinstance(stopping, dict) else None
        ),
    }


def _summary_rows(output_root: Path, methods: tuple[str, ...]) -> list[dict]:
    rows: list[dict] = []
    for method in methods:
        path = result_path(output_root, method)
        if not path.is_file():
            continue
        payload = _read_json(path)
        metadata = _training_metadata(output_root, method)
        row = {
            "method": method,
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "best_epoch": metadata["best_epoch"],
            "actual_epochs": json.dumps(
                metadata["actual_epochs"], sort_keys=True
            ),
            "early_stopped": metadata["early_stopped"],
            "stop_reason": metadata["stop_reason"],
        }
        for scenario in SCENARIOS:
            prefix = "PM" if scenario == "palmprint_missing" else "VM"
            row[f"{prefix}_eer_pct"] = 100.0 * _metric(payload, scenario, "eer", None)
            row[f"{prefix}_rank1_pct"] = 100.0 * _metric(payload, scenario, "topk", 1)
            row[f"{prefix}_tar_far_1e-3_pct"] = 100.0 * _metric(
                payload, scenario, "tar_at_far", 1e-3
            )
            row[f"{prefix}_tar_far_1e-4_pct"] = 100.0 * _metric(
                payload, scenario, "tar_at_far", 1e-4
            )
        rows.append(row)
    return rows


def write_summaries(output_root: Path, methods: tuple[str, ...]) -> None:
    rows = _summary_rows(output_root, methods)
    if not rows:
        return
    fields = list(rows[0])
    csv_path = output_root / "summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)

    columns = (
        "Method", "Best epoch", "Actual epochs", "Early stopped",
        "PM EER (%)", "VM EER (%)", "PM Rank-1 (%)",
        "VM Rank-1 (%)", "PM TAR@1e-3 (%)", "VM TAR@1e-3 (%)",
    )
    lines = [
        "# Tongji full-comparison results",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = (
            row["method"],
            str(row["best_epoch"]),
            row["actual_epochs"],
            str(row["early_stopped"]),
            f"{row['PM_eer_pct']:.4f}", f"{row['VM_eer_pct']:.4f}",
            f"{row['PM_rank1_pct']:.4f}", f"{row['VM_rank1_pct']:.4f}",
            f"{row['PM_tar_far_1e-3_pct']:.4f}",
            f"{row['VM_tar_far_1e-3_pct']:.4f}",
        )
        lines.append("| " + " | ".join(values) + " |")
    markdown = output_root / "summary.md"
    temporary = markdown.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(markdown)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = Path(args.output_root).resolve()
    args.output_root = str(output_root)
    if args.dry_run:
        for method in args.methods:
            run_command(train_command(args, method), Path(), dry_run=True)
            run_command(test_command(args, method), Path(), dry_run=True)
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest = _load_or_create_manifest(manifest_path, args)
    atomic_json(manifest_path, manifest)
    for method in args.methods:
        state = manifest["runs"].setdefault(method, {})
        state.update({"status": "training", "updated_at_utc": _utc_now()})
        manifest["updated_at_utc"] = _utc_now()
        atomic_json(manifest_path, manifest)
        try:
            run_command(
                train_command(args, method),
                output_root / "logs" / f"{method}_train.log",
                dry_run=False,
            )
            checkpoint = best_checkpoint(output_root, method)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            training_metadata = _training_metadata(output_root, method)
            state.update(
                {
                    "status": "testing",
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": file_sha256(checkpoint),
                    **training_metadata,
                    "updated_at_utc": _utc_now(),
                }
            )
            atomic_json(manifest_path, manifest)
            result = result_path(output_root, method)
            if args.force_test or not result_is_current(result, checkpoint):
                run_command(
                    test_command(args, method),
                    output_root / "logs" / f"{method}_test.log",
                    dry_run=False,
                )
            state.update(
                {
                    "status": "complete",
                    "result": str(result),
                    "result_sha256": file_sha256(result),
                    "completed_at_utc": _utc_now(),
                    "updated_at_utc": _utc_now(),
                }
            )
        except Exception as error:
            state.update(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "updated_at_utc": _utc_now(),
                }
            )
            atomic_json(manifest_path, manifest)
            raise
        manifest["updated_at_utc"] = _utc_now()
        atomic_json(manifest_path, manifest)
        write_summaries(output_root, args.methods)
    print(f"[Complete] summaries: {output_root / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
