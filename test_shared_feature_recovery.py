"""Evaluate shared identity feature recovery on the unchanged Tongji protocol."""

from __future__ import annotations

import argparse
import json
import os

import torch

from models.shared_feature_recovery import (
    ARCHITECTURE_VERSION,
    RegularizedSharedIdentityProjector,
)
from utils.checkpoint import load_encoder_from_checkpoint
from utils.checkpoint_io import file_sha256, safe_torch_load
from utils.evaluation import format_gallery_probe_metrics, gallery_probe_scores, score_matrix_metrics
from utils.feature_extraction import (
    extract_paired_features,
    extract_single_features,
    paired_feature_loader,
    single_feature_loader,
)
from utils.runtime import resolve_device, set_random_seed
from utils.scenarios import PALMPRINT_MISSING, PALMVEIN_MISSING


def verify_fingerprints(checkpoint, args):
    for name, path in (("palm_encoder", args.palm_ckpt), ("vein_encoder", args.vein_ckpt)):
        expected = checkpoint.get(f"{name}_sha256")
        if expected is not None and file_sha256(path) != expected:
            raise ValueError(f"{name} checkpoint fingerprint differs from recovery fitting")


def print_result(name, metrics):
    print(f"[{name}]")
    print("\n".join(format_gallery_probe_metrics(metrics)))


def evaluate_scenario(scenario, projector, gallery, probe, probe_labels, weights, dimensions, args):
    if scenario == PALMVEIN_MISSING:
        available, target = "palm", "vein"
    elif scenario == PALMPRINT_MISSING:
        available, target = "vein", "palm"
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")
    available_scores, labels = gallery_probe_scores(
        gallery[available], gallery["labels"], probe
    )
    gallery_available_shared = projector.transform(
        gallery[available], available, dimensions
    )
    gallery_target_shared = projector.transform(gallery[target], target, dimensions)
    probe_shared = projector.transform(probe, available, dimensions)
    same_scores, same_labels = gallery_probe_scores(
        gallery_available_shared, gallery["labels"], probe_shared
    )
    cross_scores, cross_labels = gallery_probe_scores(
        gallery_target_shared, gallery["labels"], probe_shared
    )
    if not torch.equal(labels, same_labels) or not torch.equal(labels, cross_labels):
        raise ValueError("Candidate label orders differ across shared feature domains")
    cross_weight = float(weights["cross_gallery_weight"])
    alpha = float(weights["alpha"])
    shared_scores = (1.0 - cross_weight) * same_scores + cross_weight * cross_scores
    fused_scores = (available_scores + alpha * shared_scores) / (1.0 + alpha)
    kwargs = dict(
        candidate_labels=labels,
        probe_labels=probe_labels,
        topk=args.top_k,
        far_points=args.far_points,
    )
    available_metrics = score_matrix_metrics(available_scores, **kwargs)
    shared_metrics = score_matrix_metrics(shared_scores, **kwargs)
    fused_metrics = score_matrix_metrics(fused_scores, **kwargs)
    print(f"\n===== {scenario} =====")
    print_result("available only", available_metrics)
    print_result("recovered shared identity only", shared_metrics)
    print(
        f"[fusion parameters] alpha={alpha:g} cross_gallery_weight={cross_weight:g} "
        f"shared_dimensions={dimensions}"
    )
    print_result("fused", fused_metrics)
    improvements = {
        "eer": available_metrics["eer"] - fused_metrics["eer"],
        "tar_1e-3": fused_metrics["tar_at_far"][1e-3] - available_metrics["tar_at_far"][1e-3],
        "tar_1e-4": fused_metrics["tar_at_far"][1e-4] - available_metrics["tar_at_far"][1e-4],
        "top1": fused_metrics["topk"][1] - available_metrics["topk"][1],
    }
    print(
        f"[improvement] EER={improvements['eer'] * 100:+.2f}pp "
        f"TAR@1e-3={improvements['tar_1e-3'] * 100:+.2f}pp "
        f"TAR@1e-4={improvements['tar_1e-4'] * 100:+.2f}pp "
        f"Top1={improvements['top1'] * 100:+.2f}pp"
    )
    print(
        f"[goal] eer_drop>=0.40pp={improvements['eer'] >= 0.004} "
        f"tar_1e-3_gain>=5pp={improvements['tar_1e-3'] >= 0.05} "
        f"tar_1e-4_gain>=5pp={improvements['tar_1e-4'] >= 0.05} "
        f"top1_not_lower={improvements['top1'] >= -1e-9}"
    )
    return {"available": available_metrics, "shared": shared_metrics, "fused": fused_metrics}


def evaluate(args):
    device = resolve_device(args.device, require_available=True, announce=True)
    set_random_seed(args.seed)
    checkpoint = safe_torch_load(args.ckpt, device)
    if checkpoint.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("Checkpoint is not a shared identity recovery model")
    verify_fingerprints(checkpoint, args)
    saved_args = checkpoint.get("args", {})
    args.input_size = saved_args.get("input_size", args.input_size)
    args.embedding_size = saved_args.get("embedding_size", args.embedding_size)
    configuration = checkpoint["configuration"]
    projector = RegularizedSharedIdentityProjector(
        args.embedding_size, unit_input=configuration["unit_input"]
    ).to(device)
    projector.load_state_dict(checkpoint["model"])
    projector.eval()
    palm_encoder = load_encoder_from_checkpoint(
        args.palm_ckpt, "palm", args.embedding_size, device
    )
    vein_encoder = load_encoder_from_checkpoint(
        args.vein_ckpt, "vein", args.embedding_size, device
    )
    gallery = extract_paired_features(
        palm_encoder,
        vein_encoder,
        paired_feature_loader(
            args.gallery_list, None, args.input_size, args.extract_batch_size, args.num_workers
        ),
        device,
        "Extract complete shared-recovery Gallery",
    )
    gallery = {
        key: value.to(device) if key != "labels" else value
        for key, value in gallery.items()
    }
    results = {}
    for scenario in (PALMPRINT_MISSING, PALMVEIN_MISSING):
        modality = "vein" if scenario == PALMPRINT_MISSING else "palm"
        encoder = vein_encoder if modality == "vein" else palm_encoder
        probe, labels = extract_single_features(
            encoder,
            single_feature_loader(
                args.protocol_list,
                modality,
                scenario,
                args.input_size,
                args.extract_batch_size,
                args.num_workers,
            ),
            device,
            f"Extract {scenario} shared-recovery Probes",
        )
        results[scenario] = evaluate_scenario(
            scenario,
            projector,
            gallery,
            probe.to(device),
            labels,
            checkpoint["fusion_weights"][scenario],
            int(configuration["dimensions"]),
            args,
        )
    if args.metrics_path:
        directory = os.path.dirname(args.metrics_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.metrics_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "checkpoint_sha256": file_sha256(args.ckpt),
                    "gallery_protocol_sha256": file_sha256(args.gallery_list),
                    "probe_protocol_sha256": file_sha256(args.protocol_list),
                    "results": results,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
        print(f"[Info] saved metrics to {args.metrics_path}")
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Evaluate shared identity feature recovery")
    parser.add_argument("--gallery_list", default="data_txt/tongji/ssfd_gallery_full.txt")
    parser.add_argument("--protocol_list", default="data_txt/tongji/ssfd_test_protocol.txt")
    parser.add_argument("--ckpt", default="outputs/shared_feature_recovery/best.pth")
    parser.add_argument("--palm_ckpt", default="outputs/encoders/palm_best.pth")
    parser.add_argument("--vein_ckpt", default="outputs/encoders/vein_best.pth")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--embedding_size", type=int, default=256)
    parser.add_argument("--extract_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--far_points", type=float, nargs="+", default=[1e-3, 1e-4])
    parser.add_argument(
        "--metrics_path", default="outputs/shared_feature_recovery/test_metrics.json"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args())
