"""Run the seven unique Tongji one-factor-at-a-time experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from utils.checkpoint_io import file_sha256, safe_torch_load


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs/gipssr/hyperparameter_experiments/tongji/seed_42"
CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"
RESULT_ROOT = OUTPUT_ROOT / "results"
LOG_ROOT = OUTPUT_ROOT / "logs"

WARM_START = ROOT / "outputs/gipssr/ablations/checkpoints/tongji/seed_42/stage1_full/best.pth"
DEFAULT_CHECKPOINT = ROOT / "outputs/gipssr/ablations/checkpoints/tongji/seed_42/full/best.pth"
DEFAULT_RESULT = ROOT / "outputs/gipssr/ablations/results/tongji/seed_42/full.json"

TRAIN_LIST = ROOT / "data_txt/tongji/ssfd_train_full.txt"
VAL_GALLERY_LIST = ROOT / "data_txt/tongji/ssfd_val_gallery_full.txt"
VAL_PROTOCOL_LIST = ROOT / "data_txt/tongji/ssfd_val_protocol.txt"
TEST_GALLERY_LIST = ROOT / "data_txt/tongji/ssfd_gallery_full.txt"
TEST_PROTOCOL_LIST = ROOT / "data_txt/tongji/ssfd_test_protocol.txt"
PALM_CHECKPOINT = ROOT / "outputs/encoders/palm_best.pth"
VEIN_CHECKPOINT = ROOT / "outputs/encoders/vein_best.pth"

ARCHITECTURE = "gipssr_cuef_state_space_recovery_v3"
SEED = 42


@dataclass(frozen=True)
class Experiment:
    name: str
    factor: str
    display_value: str
    topk_candidates: int
    max_refinement: float
    max_recovery_weight: float
    reuse_default_training: bool = False


EXPERIMENTS = (
    Experiment("k_3", "K", "3", 3, 0.25, 0.75),
    Experiment("default", "K", "5", 5, 0.25, 0.75, True),
    Experiment("k_8", "K", "8", 8, 0.25, 0.75),
    Experiment("alpha_0.10", "alpha", "0.10", 5, 0.10, 0.75),
    Experiment("alpha_0.50", "alpha", "0.50", 5, 0.50, 0.75),
    Experiment("wmax_0.55", "wmax", "0.55", 5, 0.25, 0.55),
    Experiment("wmax_0.95", "wmax", "0.95", 5, 0.25, 0.95),
)


COMMON_TRAIN_ARGS = (
    "--train_list", str(TRAIN_LIST.relative_to(ROOT)),
    "--val_gallery_list", str(VAL_GALLERY_LIST.relative_to(ROOT)),
    "--val_protocol_list", str(VAL_PROTOCOL_LIST.relative_to(ROOT)),
    "--palm_ckpt", str(PALM_CHECKPOINT.relative_to(ROOT)),
    "--vein_ckpt", str(VEIN_CHECKPOINT.relative_to(ROOT)),
    "--device", "cuda",
    "--seed", str(SEED),
    "--input_size", "224",
    "--embedding_size", "256",
    "--shared_dimensions", "192",
    "--specific_dimensions", "128",
    "--transformer_layers", "2",
    "--transformer_heads", "4",
    "--dropout", "0.1",
    "--epochs", "12",
    "--eval_every", "1",
    "--batch_identities", "32",
    "--instances_per_identity", "2",
    "--steps_per_epoch", "0",
    "--episodic_gallery_fraction", "0.5",
    "--memory_batch_size", "256",
    "--extract_batch_size", "128",
    "--num_workers", "4",
    "--learning_rate", "3e-4",
    "--min_learning_rate", "1e-5",
    "--weight_decay", "1e-4",
    "--gradient_clip", "1.0",
    "--contrastive_temperature", "0.07",
    "--identity_temperature", "0.05",
    "--evidence_target_temperature", "0.05",
    "--hard_margin", "0.1",
    "--cca_dimensions", "64",
    "--cca_ridge", "1e-3",
    "--cca_eigen_floor", "1e-6",
    "--analytic_eigen_floor", "1.0",
    "--nr_interval", "10",
    "--nr_dimensions", "16",
    "--nr_noise_scale", "1.0",
    "--recovery_warmup_epochs", "4",
    "--metric_weight", "1.0",
    "--specific_metric_weight", "0.25",
    "--dcca_weight", "0.2",
    "--nr_weight", "0.01",
    "--reconstruction_weight", "0.5",
    "--identity_weight", "0.5",
    "--rank_weight", "1.0",
    "--evidence_weight", "0.2",
    "--anchor_weight", "0.2",
    "--ablation", "full",
    "--role_queries", "4",
    "--candidate_dropout", "0.20",
    "--orthogonal_weight", "0.10",
    "--minimum_candidate_epochs", "2",
    "--proxy_weight", "0.5",
    "--proxy_temperature", "0.05",
    "--pair_alignment_weight", "0.5",
    "--cycle_weight", "0.25",
    "--pauc_weight", "0.5",
    "--pauc_margin", "0.05",
    "--pauc_temperature", "0.05",
    "--retrieval_dropout", "0.10",
    "--min_recovery_weight", "0.15",
    "--branch_floor", "0.02",
)


CONTROL_KEYS = {
    "topk_candidates",
    "max_refinement",
    "max_recovery_weight",
}
PATH_KEYS = {"save_dir", "selection_ckpt", "warm_start_ckpt"}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def check_inputs() -> None:
    for path in (
        WARM_START,
        DEFAULT_CHECKPOINT,
        DEFAULT_RESULT,
        TRAIN_LIST,
        VAL_GALLERY_LIST,
        VAL_PROTOCOL_LIST,
        TEST_GALLERY_LIST,
        TEST_PROTOCOL_LIST,
        PALM_CHECKPOINT,
        VEIN_CHECKPOINT,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)


def experiment_args(experiment: Experiment) -> list[str]:
    return [
        "--warm_start_ckpt", str(WARM_START),
        "--topk_candidates", str(experiment.topk_candidates),
        "--max_refinement", str(experiment.max_refinement),
        "--max_recovery_weight", str(experiment.max_recovery_weight),
        *COMMON_TRAIN_ARGS,
    ]


def selection_checkpoint(experiment: Experiment) -> Path:
    return CHECKPOINT_ROOT / experiment.name / "selection" / "best.pth"


def final_checkpoint(experiment: Experiment) -> Path:
    if experiment.reuse_default_training:
        return DEFAULT_CHECKPOINT
    return CHECKPOINT_ROOT / experiment.name / "final" / "best.pth"


def result_path(experiment: Experiment) -> Path:
    return RESULT_ROOT / f"{experiment.name}.json"


def run_command(command: list[str], log_path: Path, dry_run: bool) -> None:
    rendered = " ".join(command)
    print(f"\n[Command] {rendered}", flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[Command] {rendered}\n")
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


def assert_value(actual, expected, label: str) -> None:
    if isinstance(expected, float):
        if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{label}: expected {expected!r}, found {actual!r}")
    elif actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, found {actual!r}")


def validate_checkpoint(path: Path, experiment: Experiment, expected_stage: str) -> dict:
    checkpoint = safe_torch_load(path, "cpu")
    assert_value(checkpoint.get("architecture_version"), ARCHITECTURE, f"{path}: architecture")
    assert_value(checkpoint.get("training_stage"), expected_stage, f"{path}: training stage")
    saved_args = checkpoint.get("args", {})
    assert_value(saved_args.get("seed"), SEED, f"{path}: seed")
    assert_value(saved_args.get("ablation"), "full", f"{path}: ablation")
    for key in CONTROL_KEYS:
        assert_value(saved_args.get(key), getattr(experiment, key), f"{path}: {key}")
    return checkpoint


def compare_final_invariants(experiments: tuple[Experiment, ...]) -> None:
    baseline = safe_torch_load(DEFAULT_CHECKPOINT, "cpu")["args"]
    excluded = CONTROL_KEYS | PATH_KEYS
    for experiment in experiments:
        candidate = safe_torch_load(final_checkpoint(experiment), "cpu")["args"]
        keys = (set(baseline) | set(candidate)) - excluded
        differences = {
            key: (baseline.get(key), candidate.get(key))
            for key in sorted(keys)
            if baseline.get(key) != candidate.get(key)
        }
        if differences:
            raise ValueError(f"Non-controlled argument differences for {experiment.name}: {differences}")


def evaluation_command(checkpoint: Path, metrics_path: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "test_gipssr.py"),
        "--ckpt", str(checkpoint),
        "--metrics_path", str(metrics_path),
        "--gallery_list", str(TEST_GALLERY_LIST),
        "--protocol_list", str(TEST_PROTOCOL_LIST),
        "--palm_ckpt", str(PALM_CHECKPOINT.relative_to(ROOT)),
        "--vein_ckpt", str(VEIN_CHECKPOINT.relative_to(ROOT)),
        "--device", "cuda",
        "--seed", str(SEED),
        "--input_size", "224",
        "--embedding_size", "256",
        "--extract_batch_size", "128",
        "--memory_batch_size", "256",
        "--num_workers", "8",
        "--top_k", "1", "5",
        "--far_points", "1e-3", "1e-4",
    ]


def validate_result(path: Path, checkpoint: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert_value(payload.get("architecture_version"), ARCHITECTURE, f"{path}: architecture")
    assert_value(payload.get("checkpoint_sha256"), file_sha256(checkpoint), f"{path}: checkpoint hash")
    assert_value(payload.get("gallery_protocol_sha256"), file_sha256(TEST_GALLERY_LIST), f"{path}: gallery hash")
    assert_value(payload.get("probe_protocol_sha256"), file_sha256(TEST_PROTOCOL_LIST), f"{path}: probe hash")
    for scenario in ("palmprint_missing", "palmvein_missing"):
        fused = payload["results"][scenario]["fused"]
        for key in ("eer", "topk", "tar_at_far"):
            if key not in fused:
                raise ValueError(f"{path}: missing {scenario}.fused.{key}")
    return payload


def metric_row(experiment: Experiment, payload: dict) -> dict[str, str | float | int]:
    row: dict[str, str | float | int] = {
        "experiment": experiment.name,
        "factor": experiment.factor,
        "value": experiment.display_value,
        "K": experiment.topk_candidates,
        "alpha": experiment.max_refinement,
        "wmax": experiment.max_recovery_weight,
    }
    for prefix, scenario in (("PM", "palmprint_missing"), ("VM", "palmvein_missing")):
        fused = payload["results"][scenario]["fused"]
        row[f"{prefix}_EER_percent"] = 100.0 * fused["eer"]
        row[f"{prefix}_Top1_percent"] = 100.0 * fused["topk"]["1"]
        row[f"{prefix}_TAR_1e-3_percent"] = 100.0 * fused["tar_at_far"]["0.001"]
        row[f"{prefix}_TAR_1e-4_percent"] = 100.0 * fused["tar_at_far"]["0.0001"]
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(payloads: dict[str, dict]) -> None:
    by_name = {experiment.name: experiment for experiment in EXPERIMENTS}
    unique_rows = [metric_row(experiment, payloads[experiment.name]) for experiment in EXPERIMENTS]
    write_csv(OUTPUT_ROOT / "metrics_unique.csv", unique_rows)

    fill_order = (
        ("Top-K candidates", "3", "k_3"),
        ("Top-K candidates", "5", "default"),
        ("Top-K candidates", "8", "k_8"),
        ("Refinement scale alpha", "0.10", "alpha_0.10"),
        ("Refinement scale alpha", "0.25", "default"),
        ("Refinement scale alpha", "0.50", "alpha_0.50"),
        ("Recovery bound wmax", "0.55", "wmax_0.55"),
        ("Recovery bound wmax", "0.75", "default"),
        ("Recovery bound wmax", "0.95", "wmax_0.95"),
    )
    fill_rows = []
    for parameter, value, name in fill_order:
        row = metric_row(by_name[name], payloads[name])
        row["parameter"] = parameter
        row["value"] = value
        ordered = {"parameter": row.pop("parameter"), "value": row.pop("value"), **row}
        fill_rows.append(ordered)
    write_csv(OUTPUT_ROOT / "metrics_fill_table.csv", fill_rows)

    existing_default = validate_result(DEFAULT_RESULT, DEFAULT_CHECKPOINT)
    old_row = metric_row(by_name["default"], existing_default)
    new_row = metric_row(by_name["default"], payloads["default"])
    for key in old_row:
        if key.startswith(("PM_", "VM_")):
            assert_value(new_row[key], old_row[key], f"default re-evaluation: {key}")

    manifest = {
        "dataset": "tongji",
        "seed": SEED,
        "design": "one-factor-at-a-time",
        "default": {"K": 5, "alpha": 0.25, "wmax": 0.75},
        "default_training_reused": rel(DEFAULT_CHECKPOINT),
        "warm_start": {
            "path": rel(WARM_START),
            "sha256": file_sha256(WARM_START),
        },
        "encoders": {
            "palm": {"path": rel(PALM_CHECKPOINT), "sha256": file_sha256(PALM_CHECKPOINT)},
            "vein": {"path": rel(VEIN_CHECKPOINT), "sha256": file_sha256(VEIN_CHECKPOINT)},
        },
        "protocols": {
            "train": {"path": rel(TRAIN_LIST), "sha256": file_sha256(TRAIN_LIST)},
            "validation_gallery": {"path": rel(VAL_GALLERY_LIST), "sha256": file_sha256(VAL_GALLERY_LIST)},
            "validation_probe": {"path": rel(VAL_PROTOCOL_LIST), "sha256": file_sha256(VAL_PROTOCOL_LIST)},
            "test_gallery": {"path": rel(TEST_GALLERY_LIST), "sha256": file_sha256(TEST_GALLERY_LIST)},
            "test_probe": {"path": rel(TEST_PROTOCOL_LIST), "sha256": file_sha256(TEST_PROTOCOL_LIST)},
        },
        "experiments": [
            {
                **asdict(experiment),
                "checkpoint": rel(final_checkpoint(experiment)),
                "checkpoint_sha256": file_sha256(final_checkpoint(experiment)),
                "result": rel(result_path(experiment)),
            }
            for experiment in EXPERIMENTS
        ],
    }
    (OUTPUT_ROOT / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", choices=[experiment.name for experiment in EXPERIMENTS], nargs="+")
    args = parser.parse_args()
    check_inputs()
    chosen = tuple(
        experiment
        for experiment in EXPERIMENTS
        if args.only is None or experiment.name in args.only
    )

    for experiment in chosen:
        print(
            f"\n===== {experiment.name}: K={experiment.topk_candidates}, "
            f"alpha={experiment.max_refinement:.2f}, "
            f"wmax={experiment.max_recovery_weight:.2f} =====",
            flush=True,
        )
        if experiment.reuse_default_training:
            validate_checkpoint(DEFAULT_CHECKPOINT, experiment, "fixed_full_train")
            print(f"[Reuse] existing default training: {DEFAULT_CHECKPOINT}", flush=True)
            continue

        selection = selection_checkpoint(experiment)
        if selection.exists():
            validate_checkpoint(selection, experiment, "identity_validation_selection")
            print(f"[Resume] valid selection checkpoint: {selection}", flush=True)
        else:
            command = [
                sys.executable,
                str(ROOT / "train_gipssr.py"),
                *experiment_args(experiment),
                "--save_dir", str(selection.parent),
                "--cache_dir", "outputs/gipssr/cache/tongji_validation",
            ]
            run_command(command, LOG_ROOT / f"{experiment.name}_selection.log", args.dry_run)
            if not args.dry_run:
                validate_checkpoint(selection, experiment, "identity_validation_selection")

        final = final_checkpoint(experiment)
        if final.exists():
            validate_checkpoint(final, experiment, "fixed_full_train")
            print(f"[Resume] valid final checkpoint: {final}", flush=True)
        else:
            command = [
                sys.executable,
                str(ROOT / "train_gipssr.py"),
                *experiment_args(experiment),
                "--fixed_full_train",
                "--selection_ckpt", str(selection),
                "--save_dir", str(final.parent),
                "--cache_dir", "outputs/gipssr/cache/tongji_full",
            ]
            run_command(command, LOG_ROOT / f"{experiment.name}_final.log", args.dry_run)
            if not args.dry_run:
                validate_checkpoint(final, experiment, "fixed_full_train")

    if args.dry_run:
        return

    compare_final_invariants(chosen)
    payloads = {}
    for experiment in chosen:
        checkpoint = final_checkpoint(experiment)
        metrics = result_path(experiment)
        if metrics.exists():
            payloads[experiment.name] = validate_result(metrics, checkpoint)
            print(f"[Resume] valid metrics: {metrics}", flush=True)
        else:
            run_command(
                evaluation_command(checkpoint, metrics),
                LOG_ROOT / f"{experiment.name}_evaluation.log",
                False,
            )
            payloads[experiment.name] = validate_result(metrics, checkpoint)

    if len(chosen) == len(EXPERIMENTS):
        write_outputs(payloads)
        print(f"[Done] experiment artifacts: {OUTPUT_ROOT}", flush=True)
    else:
        print("[Info] partial --only run complete; aggregate CSV files are deferred.", flush=True)


if __name__ == "__main__":
    main()
