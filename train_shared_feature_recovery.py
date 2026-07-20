"""Fit and validate cross-modally shared identity recovery on frozen embeddings."""

from __future__ import annotations

import argparse
import itertools
import math

import torch

from models.shared_feature_recovery import (
    ARCHITECTURE_VERSION,
    RegularizedSharedIdentityProjector,
)
from utils.checkpoint import load_encoder_from_checkpoint
from utils.checkpoint_io import file_sha256, save_checkpoint
from utils.evaluation import gallery_probe_scores, score_matrix_metrics
from utils.feature_extraction import extract_paired_features, paired_feature_loader
from utils.runtime import resolve_device, set_random_seed
from utils.scenarios import PALMPRINT_MISSING, PALMVEIN_MISSING


def metric_summary(metrics):
    return (
        f"EER={metrics['eer'] * 100:.2f}% "
        f"TAR@1e-3={metrics['tar_at_far'][1e-3] * 100:.2f}% "
        f"TAR@1e-4={metrics['tar_at_far'][1e-4] * 100:.2f}% "
        f"Top1={metrics['topk'][1] * 100:.2f}%"
    )


def selection_value(metrics, baseline):
    improvements = (
        (baseline["eer"] - metrics["eer"]) / 0.004,
        (metrics["tar_at_far"][1e-3] - baseline["tar_at_far"][1e-3]) / 0.05,
        (metrics["tar_at_far"][1e-4] - baseline["tar_at_far"][1e-4]) / 0.05,
    )
    return min(improvements) + 0.05 * sum(improvements)


def score_metrics(scores, candidate_labels, probe_labels):
    return score_matrix_metrics(
        scores,
        candidate_labels,
        probe_labels,
        topk=(1, 5),
        far_points=(1e-3, 1e-4),
        warn_far_resolution=False,
    )


def search_fusion(available, shared_same, shared_cross, labels, probe_labels, baseline, args):
    best = None
    alphas = [index * args.alpha_step for index in range(round(args.alpha_max / args.alpha_step) + 1)]
    cross_weights = [
        index * args.cross_step for index in range(round(1.0 / args.cross_step) + 1)
    ]
    for cross_weight, alpha in itertools.product(cross_weights, alphas):
        shared = (1.0 - cross_weight) * shared_same + cross_weight * shared_cross
        fused = (available + alpha * shared) / (1.0 + alpha)
        metrics = score_metrics(fused, labels, probe_labels)
        if metrics["topk"][1] + 1e-9 < baseline["topk"][1]:
            continue
        value = selection_value(metrics, baseline)
        candidate = {
            "alpha": alpha,
            "cross_gallery_weight": cross_weight,
            "selection_value": value,
            "metrics": metrics,
        }
        if best is None or value > best["selection_value"]:
            best = candidate
    if best is None:
        raise RuntimeError("No shared-feature fusion preserved the single-modality Top-1")
    return best


def evaluate_configuration(projector, dimensions, gallery, probes, baselines, scores, labels, args):
    palm_gallery = projector.transform(gallery["palm"], "palm", dimensions)
    vein_gallery = projector.transform(gallery["vein"], "vein", dimensions)
    palm_probe = projector.transform(probes["palm"], "palm", dimensions)
    vein_probe = projector.transform(probes["vein"], "vein", dimensions)
    palm_same, _ = gallery_probe_scores(palm_gallery, gallery["labels"], palm_probe)
    vein_same, _ = gallery_probe_scores(vein_gallery, gallery["labels"], vein_probe)
    palm_cross, _ = gallery_probe_scores(vein_gallery, gallery["labels"], palm_probe)
    vein_cross, _ = gallery_probe_scores(palm_gallery, gallery["labels"], vein_probe)
    return {
        PALMVEIN_MISSING: search_fusion(
            scores["palm"], palm_same, palm_cross, labels, probes["labels"], baselines["palm"], args
        ),
        PALMPRINT_MISSING: search_fusion(
            scores["vein"], vein_same, vein_cross, labels, probes["labels"], baselines["vein"], args
        ),
    }


def fit(args):
    device = resolve_device(args.device, require_available=True, announce=True)
    set_random_seed(args.seed)
    palm_encoder = load_encoder_from_checkpoint(args.palm_ckpt, "palm", args.embedding_size, device)
    vein_encoder = load_encoder_from_checkpoint(args.vein_ckpt, "vein", args.embedding_size, device)
    training = extract_paired_features(
        palm_encoder,
        vein_encoder,
        paired_feature_loader(
            args.train_list, None, args.input_size, args.extract_batch_size, args.num_workers
        ),
        device,
        "Extract shared-recovery training features",
    )
    gallery = extract_paired_features(
        palm_encoder,
        vein_encoder,
        paired_feature_loader(
            args.val_gallery_list, None, args.input_size, args.extract_batch_size, args.num_workers
        ),
        device,
        "Extract shared-recovery validation Gallery",
    )
    probes = extract_paired_features(
        palm_encoder,
        vein_encoder,
        paired_feature_loader(
            args.val_protocol_list,
            "complete",
            args.input_size,
            args.extract_batch_size,
            args.num_workers,
        ),
        device,
        "Extract shared-recovery validation Probes",
    )
    palm_scores, labels = gallery_probe_scores(
        gallery["palm"], gallery["labels"], probes["palm"]
    )
    vein_scores, vein_labels = gallery_probe_scores(
        gallery["vein"], gallery["labels"], probes["vein"]
    )
    if not torch.equal(labels, vein_labels):
        raise ValueError("Palm and vein validation label orders differ")
    scores = {"palm": palm_scores, "vein": vein_scores}
    baselines = {
        "palm": score_metrics(palm_scores, labels, probes["labels"]),
        "vein": score_metrics(vein_scores, labels, probes["labels"]),
    }
    print(f"[Validation baseline palm] {metric_summary(baselines['palm'])}")
    print(f"[Validation baseline vein] {metric_summary(baselines['vein'])}")

    best = None
    for unit_input, eigen_floor in itertools.product((False, True), args.eigen_floors):
        projector = RegularizedSharedIdentityProjector(args.embedding_size, unit_input=unit_input)
        projector.fit(training["palm"], training["vein"], eigen_floor)
        for dimensions in args.shared_dimensions:
            validation = evaluate_configuration(
                projector, dimensions, gallery, probes, baselines, scores, labels, args
            )
            minimum = min(result["selection_value"] for result in validation.values())
            total = sum(result["selection_value"] for result in validation.values())
            rank = (minimum, total)
            if best is not None and rank <= best["rank"]:
                continue
            best = {
                "rank": rank,
                "unit_input": unit_input,
                "eigen_floor": eigen_floor,
                "dimensions": dimensions,
                "validation": validation,
                "model": {key: value.cpu() for key, value in projector.state_dict().items()},
                "correlation_mean": float(
                    projector.canonical_correlations[:dimensions].mean().item()
                ),
            }
    if best is None:
        raise RuntimeError("Shared identity configuration search produced no result")
    print(
        f"[Best shared recovery] unit_input={best['unit_input']} "
        f"eigen_floor={best['eigen_floor']:g} dimensions={best['dimensions']} "
        f"mean_correlation={best['correlation_mean']:.4f} min_score={best['rank'][0]:.3f}"
    )
    for scenario, result in best["validation"].items():
        print(
            f"[Validation {scenario}] {metric_summary(result['metrics'])} "
            f"alpha={result['alpha']:.2f} cross_weight={result['cross_gallery_weight']:.2f} "
            f"score={result['selection_value']:.3f}"
        )
    save_checkpoint(
        args.save_path,
        {
            "architecture_version": ARCHITECTURE_VERSION,
            "training_stage": "closed_form_shared_recovery",
            "model": best["model"],
            "args": vars(args),
            "configuration": {
                "unit_input": best["unit_input"],
                "eigen_floor": best["eigen_floor"],
                "dimensions": best["dimensions"],
                "mean_canonical_correlation": best["correlation_mean"],
            },
            "validation": {
                "baselines": baselines,
                **best["validation"],
            },
            "fusion_weights": {
                scenario: {
                    "alpha": result["alpha"],
                    "cross_gallery_weight": result["cross_gallery_weight"],
                }
                for scenario, result in best["validation"].items()
            },
            "best_selection_score": best["rank"][0],
            "palm_encoder_sha256": file_sha256(args.palm_ckpt),
            "vein_encoder_sha256": file_sha256(args.vein_ckpt),
            "train_list_sha256": file_sha256(args.train_list),
            "val_gallery_list_sha256": file_sha256(args.val_gallery_list),
            "val_protocol_list_sha256": file_sha256(args.val_protocol_list),
        },
    )
    print(f"[Info] saved {args.save_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Fit cross-modally shared identity recovery")
    parser.add_argument("--train_list", default="data_txt/tongji/ssfd_train_full.txt")
    parser.add_argument("--val_gallery_list", default="data_txt/tongji/ssfd_val_gallery_full.txt")
    parser.add_argument("--val_protocol_list", default="data_txt/tongji/ssfd_val_protocol.txt")
    parser.add_argument("--palm_ckpt", default="outputs/encoders/palm_best.pth")
    parser.add_argument("--vein_ckpt", default="outputs/encoders/vein_best.pth")
    parser.add_argument("--save_path", default="outputs/shared_feature_recovery/best.pth")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--embedding_size", type=int, default=256)
    parser.add_argument("--extract_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--eigen_floors", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    parser.add_argument("--shared_dimensions", type=int, nargs="+", default=[16, 32, 64, 96, 128, 192, 256])
    parser.add_argument("--alpha_max", type=float, default=2.0)
    parser.add_argument("--alpha_step", type=float, default=0.05)
    parser.add_argument("--cross_step", type=float, default=0.25)
    args = parser.parse_args(argv)
    if args.alpha_step <= 0 or args.alpha_max < 0:
        parser.error("Fusion alpha bounds must be non-negative")
    if args.cross_step <= 0 or not math.isclose(round(1 / args.cross_step) * args.cross_step, 1.0):
        parser.error("cross_step must divide 1.0 exactly")
    if any(not 1 <= value <= args.embedding_size for value in args.shared_dimensions):
        parser.error("shared_dimensions must be in [1, embedding_size]")
    return args


if __name__ == "__main__":
    fit(parse_args())
