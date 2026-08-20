"""Evaluate a self-contained full-comparison checkpoint on Tongji test data."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import torch

from train_full_comparison import METHODS
from utils.checkpoint_io import file_sha256, safe_torch_load
from utils.full_comparison_common import (
    SCENARIOS,
    atomic_json,
    evaluate_gallery_probe,
    paired_loader,
)
from utils.runtime import resolve_device, set_random_seed


ROOT: Final = Path(__file__).resolve().parent
DEFAULT_GALLERY: Final = "data_txt/tongji/ssfd_gallery_full.txt"
DEFAULT_PROTOCOL: Final = "data_txt/tongji/ssfd_test_protocol.txt"
DEFAULT_MANIFEST: Final = "data_txt/tongji/manifest.json"
RESULT_SCHEMA_VERSION: Final = "tongji_full_comparison_test_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test one full reproduction on the shared Tongji protocol."
    )
    parser.add_argument("--checkpoint", "--ckpt", dest="checkpoint", required=True)
    parser.add_argument("--method", choices=("auto", *METHODS), default="auto")
    parser.add_argument(
        "--gallery-list", "--gallery_list",
        dest="gallery_list",
        default=DEFAULT_GALLERY,
    )
    parser.add_argument(
        "--protocol-list", "--protocol_list",
        dest="protocol_list",
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--metrics-path", "--metrics_path",
        dest="metrics_path",
        default=None,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-size", "--input_size", dest="input_size", type=int, default=None)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=None)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=4)
    parser.add_argument(
        "--hcmig-stochastic",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.seed != 42:
        raise ValueError("Full comparison evaluation is locked to seed 42")
    if args.input_size is not None and args.input_size <= 0:
        raise ValueError("input_size must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.metrics_path is None:
        args.metrics_path = str(Path(args.checkpoint).parent / "test_results.json")
    return args


def _checkpoint_method(payload: dict, requested: str) -> str:
    saved = str(payload.get("method", "")).lower()
    if saved not in METHODS:
        raise ValueError(f"Checkpoint has unsupported method: {saved!r}")
    if requested != "auto" and requested != saved:
        raise ValueError(
            f"Requested method {requested!r} does not match checkpoint {saved!r}"
        )
    return saved


def _saved_input_size(method: str, payload: dict) -> int:
    container = payload.get("config") if method == "ssfd" else payload.get("args")
    if method == "hcmig":
        container = payload.get("configuration")
    if not isinstance(container, dict):
        raise ValueError(f"{method} checkpoint lacks saved input configuration")
    return int(container.get("input_size", 224))


def load_method_checkpoint(
    checkpoint_path: str | Path,
    *,
    method: str = "auto",
    device: str | torch.device = "cpu",
    seed: int = 42,
    hcmig_stochastic: bool | None = None,
):
    """Return ``(method, model, payload, representation_callback)``."""

    if seed != 42:
        raise ValueError("Full comparison evaluation is locked to seed 42")
    preview = safe_torch_load(checkpoint_path, "cpu")
    if not isinstance(preview, dict):
        raise TypeError("Checkpoint payload must be a mapping")
    method = _checkpoint_method(preview, method)
    target_device = torch.device(device)

    if method == "ssfd":
        from utils.full_ssfd_experiment import (
            load_trained_ssfd,
            representation_callback,
        )

        model, _, _ = load_trained_ssfd(checkpoint_path, device=target_device)
        callback = representation_callback(model)
        payload = preview
    elif method == "dmrnet":
        from utils.full_dmrnet_experiment import (
            load_checkpoint_model,
            representation_callback,
        )

        model, payload = load_checkpoint_model(checkpoint_path, target_device)
        callback = representation_callback(model)
    elif method == "hcmig":
        from utils.full_hcmig_experiment import (
            load_checkpoint_model,
            representation_callback,
        )

        model, payload = load_checkpoint_model(checkpoint_path, target_device)
        configuration = payload["configuration"]
        stochastic = (
            bool(configuration.get("stochastic_eval", True))
            if hcmig_stochastic is None
            else bool(hcmig_stochastic)
        )
        callback = representation_callback(
            model,
            device=target_device,
            stochastic=stochastic,
            seed=seed + 1009,
        )
    elif method == "simmlm":
        from utils.full_simmlm_experiment import (
            load_checkpoint_model,
            representation_callback,
        )

        model, payload = load_checkpoint_model(checkpoint_path, target_device)
        callback = representation_callback(model)
    else:
        from utils.full_mmanet_experiment import (
            load_checkpoint_model,
            representation_callback,
        )

        model, payload = load_checkpoint_model(checkpoint_path, target_device)
        callback = representation_callback(model)
    model.eval()
    return method, model, payload, callback


def _load_manifest(path: str | Path) -> tuple[dict, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset") != "tongji":
        raise ValueError("Evaluation manifest is not Tongji")
    return payload, file_sha256(path)


def _protocol_metadata(args: argparse.Namespace) -> dict:
    gallery = Path(args.gallery_list)
    probe = Path(args.protocol_list)
    for path in (gallery, probe):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest, manifest_hash = _load_manifest(args.manifest)
    gallery_hash = file_sha256(gallery)
    probe_hash = file_sha256(probe)
    expected = manifest.get("sha256", {})
    for path, actual in ((gallery, gallery_hash), (probe, probe_hash)):
        if path.name in expected and expected[path.name] != actual:
            raise ValueError(f"Tongji protocol hash mismatch: {path}")
    bundle = hashlib.sha256(
        f"{manifest_hash}:{gallery_hash}:{probe_hash}".encode("ascii")
    ).hexdigest()
    return {
        "dataset": "tongji",
        "name": manifest.get("protocol"),
        "session": manifest.get("session"),
        "manifest_path": str(Path(args.manifest).resolve()),
        "manifest_sha256": manifest_hash,
        "gallery_path": str(gallery.resolve()),
        "gallery_sha256": gallery_hash,
        "probe_path": str(probe.resolve()),
        "probe_sha256": probe_hash,
        "bundle_sha256": bundle,
    }


def evaluate_checkpoint(args: argparse.Namespace) -> dict:
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    device = resolve_device(args.device, require_available=True, announce=True)
    set_random_seed(args.seed)
    method, model, saved, callback = load_method_checkpoint(
        checkpoint,
        method=args.method,
        device=device,
        seed=args.seed,
        hcmig_stochastic=args.hcmig_stochastic,
    )
    input_size = args.input_size or _saved_input_size(method, saved)
    batch_size = args.batch_size or (4 if method in {"ssfd", "hcmig"} else 64)
    gallery_loader = paired_loader(
        args.gallery_list,
        input_size=input_size,
        batch_size=batch_size,
        num_workers=args.num_workers,
        train=False,
    )
    probes = {
        scenario: paired_loader(
            args.protocol_list,
            input_size=input_size,
            batch_size=batch_size,
            num_workers=args.num_workers,
            train=False,
            split_filter=scenario,
        )
        for scenario in SCENARIOS
    }
    results = evaluate_gallery_probe(gallery_loader, probes, callback, device)
    protocol = _protocol_metadata(args)
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "method": method,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "architecture_version": saved.get(
            "architecture_version", saved.get("model_architecture_version")
        ),
        "official_commit": saved.get("official_commit"),
        "best_epoch": saved.get("best_epoch", saved.get("epoch")),
        "protocol": protocol,
        "gallery_protocol_sha256": protocol["gallery_sha256"],
        "probe_protocol_sha256": protocol["probe_sha256"],
        "input_size": input_size,
        "batch_size": batch_size,
        "seed": args.seed,
        "results": results,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.metrics_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = evaluate_checkpoint(args)
    for scenario in SCENARIOS:
        metrics = payload["results"][scenario]["fused"]
        print(
            f"[{payload['method']} {scenario}] EER={metrics['eer']*100:.4f}% "
            f"Rank-1={metrics['topk'][1]*100:.4f}%",
            flush=True,
        )
    print(f"[Saved] {args.metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
