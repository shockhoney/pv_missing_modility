"""Evaluate gallery-conditioned feature recovery with a strict no-recovery ablation."""

from __future__ import annotations

import argparse
import json
import math
import os

import torch

from models.shared_feature_recovery import (
    ARCHITECTURE_VERSION,
    TrainableSharedFeatureRecovery,
)
from utils.checkpoint import load_encoder_from_checkpoint
from utils.checkpoint_io import file_sha256, safe_torch_load
from utils.evaluation import format_gallery_probe_metrics, score_matrix_metrics
from utils.feature_extraction import (
    extract_paired_features,
    extract_single_features,
    paired_feature_loader,
    single_feature_loader,
)
from utils.runtime import resolve_device, set_random_seed
from utils.scenarios import PALMPRINT_MISSING, PALMVEIN_MISSING


def verify_fingerprints(checkpoint, args):
    fingerprints = checkpoint.get("fingerprints", {})
    for name, path in (("palm_encoder", args.palm_ckpt), ("vein_encoder", args.vein_ckpt)):
        expected = fingerprints.get(f"{name}_sha256")
        if expected is not None and file_sha256(path) != expected:
            raise ValueError(f"{name} checkpoint fingerprint differs from recovery training")


def print_result(name, metrics):
    print(f"[{name}]")
    print("\n".join(format_gallery_probe_metrics(metrics)))


def distribution_summary(values):
    values = values.detach().float()
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "q10": float(torch.quantile(values, 0.1).item()),
        "median": float(torch.quantile(values, 0.5).item()),
        "q90": float(torch.quantile(values, 0.9).item()),
        "q95": float(torch.quantile(values, 0.95).item()),
        "q99": float(torch.quantile(values, 0.99).item()),
        "max": float(values.max().item()),
    }


def score_margin(scores, candidate_labels, probe_labels):
    candidate_labels = candidate_labels.to(scores.device)
    probe_labels = probe_labels.to(scores.device)
    positive_mask = candidate_labels.unsqueeze(0).eq(probe_labels.unsqueeze(1))
    floor = torch.finfo(scores.dtype).min
    positive = scores.masked_fill(~positive_mask, floor).max(dim=1).values
    negative = scores.masked_fill(positive_mask, floor).max(dim=1).values
    return positive - negative


@torch.inference_mode()
def evaluate_scenario(scenario, model, gallery, probe, probe_labels, args):
    if scenario == PALMPRINT_MISSING:
        available, target = "vein", "palm"
    elif scenario == PALMVEIN_MISSING:
        available, target = "palm", "vein"
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")
    output = model.recover_with_gallery(
        probe,
        available,
        gallery[available],
        gallery[target],
        gallery["labels"],
    )
    labels = output["candidate_labels"]
    branch_scores = torch.cat(
        [output["base_branch_scores"], output["recovered_scores"].unsqueeze(2)],
        dim=2,
    )
    metric_kwargs = dict(
        candidate_labels=labels,
        probe_labels=probe_labels,
        topk=args.top_k,
        far_points=args.far_points,
    )
    branches = {
        name: score_matrix_metrics(branch_scores[:, :, index], **metric_kwargs)
        for index, name in enumerate(model.BRANCHES)
    }
    base_metrics = score_matrix_metrics(output["base_scores"], **metric_kwargs)
    fused_metrics = score_matrix_metrics(output["fused_scores"], **metric_kwargs)
    improvement = {
        "eer": base_metrics["eer"] - fused_metrics["eer"],
        "tar_1e-3": fused_metrics["tar_at_far"][1e-3]
        - base_metrics["tar_at_far"][1e-3],
        "tar_1e-4": fused_metrics["tar_at_far"][1e-4]
        - base_metrics["tar_at_far"][1e-4],
        "top1": fused_metrics["topk"][1] - base_metrics["topk"][1],
    }
    posterior_entropy = -(
        output["posterior"] * output["posterior"].clamp_min(1e-12).log()
    ).sum(dim=1)
    normalized_entropy = posterior_entropy / math.log(output["posterior"].size(1))
    base_margin = score_margin(output["base_scores"], labels, probe_labels)
    recovered_margin = score_margin(output["recovered_scores"], labels, probe_labels)
    fused_margin = score_margin(output["fused_scores"], labels, probe_labels)
    margin_summary = {
        "recovered_minus_base_mean": float((recovered_margin - base_margin).mean().item()),
        "fused_minus_base_mean": float((fused_margin - base_margin).mean().item()),
        "fused_improved_fraction": float((fused_margin > base_margin).float().mean().item()),
    }
    print(f"\n===== {scenario} (gallery-conditioned band-selective recovery v6) =====")
    for name, metrics in branches.items():
        print_result(name, metrics)
    print_result("fusion without recovery", base_metrics)
    calibration = model.calibration(available)
    print(
        f"[recovery calibration] temperature={calibration['temperature'].item():.4f} "
        f"alpha={calibration['recovery_alpha'].item():.4f} "
        f"margin_floor={calibration['margin_floor'].item():.4f} "
        f"margin_ceiling={calibration['margin_ceiling'].item():.4f} "
        f"margin_slope={calibration['margin_slope'].item():.4f} "
        f"base_weights={output['base_weights'][0].tolist()}"
    )
    print_result("fusion with recovery", fused_metrics)
    print(
        f"[recovery contribution] EER={improvement['eer'] * 100:+.4f}pp "
        f"TAR@1e-3={improvement['tar_1e-3'] * 100:+.2f}pp "
        f"TAR@1e-4={improvement['tar_1e-4'] * 100:+.2f}pp "
        f"Top1={improvement['top1'] * 100:+.2f}pp"
    )
    result = {
        "branches": branches,
        "fused_without_recovery": base_metrics,
        "fused": fused_metrics,
        "improvement_over_no_recovery": improvement,
        "posterior_confidence": distribution_summary(output["posterior_confidence"]),
        "posterior_normalized_entropy": distribution_summary(normalized_entropy),
        "recovery_reliability": distribution_summary(output["recovery_reliability"]),
        "recovery_gate": distribution_summary(output["recovery_gate"]),
        "recovery_gate_activation": {
            "fraction_gt_0.01": float((output["recovery_gate"] > 0.01).float().mean().item()),
            "fraction_gt_0.05": float((output["recovery_gate"] > 0.05).float().mean().item()),
        },
        "base_weights": {
            name: float(output["base_weights"][0, index].item())
            for index, name in enumerate(model.BRANCHES[:3])
        },
        "specific_margin": margin_summary,
    }
    print(f"[posterior confidence] {result['posterior_confidence']}")
    print(f"[recovery reliability] {result['recovery_reliability']}")
    print(f"[recovery gate] {result['recovery_gate']}")
    print(f"[recovery gate activation] {result['recovery_gate_activation']}")
    print(f"[specific margin] {margin_summary}")
    return result


def evaluate(args):
    device = resolve_device(args.device, require_available=True, announce=True)
    set_random_seed(args.seed)
    checkpoint = safe_torch_load(args.ckpt, device)
    if checkpoint.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("Checkpoint is not a gallery-conditioned band-selective recovery v6 model")
    verify_fingerprints(checkpoint, args)
    configuration = checkpoint["configuration"]
    model = TrainableSharedFeatureRecovery(
        input_dim=int(configuration["input_dim"]),
        shared_dim=int(configuration["shared_dim"]),
        dropout=float(configuration["dropout"]),
        unit_input=bool(configuration.get("unit_input", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    saved_args = checkpoint.get("args", {})
    input_size = int(saved_args.get("input_size", args.input_size))
    embedding_size = int(saved_args.get("embedding_size", args.embedding_size))
    palm_encoder = load_encoder_from_checkpoint(
        args.palm_ckpt, "palm", embedding_size, device
    )
    vein_encoder = load_encoder_from_checkpoint(
        args.vein_ckpt, "vein", embedding_size, device
    )
    gallery = extract_paired_features(
        palm_encoder,
        vein_encoder,
        paired_feature_loader(
            args.gallery_list,
            None,
            input_size,
            args.extract_batch_size,
            args.num_workers,
        ),
        device,
        "Extract gallery-conditioned recovery Gallery",
    )
    gallery = {key: value.to(device) for key, value in gallery.items()}
    results = {}
    for scenario, modality, encoder in (
        (PALMPRINT_MISSING, "vein", vein_encoder),
        (PALMVEIN_MISSING, "palm", palm_encoder),
    ):
        probe, labels = extract_single_features(
            encoder,
            single_feature_loader(
                args.protocol_list,
                modality,
                scenario,
                input_size,
                args.extract_batch_size,
                args.num_workers,
            ),
            device,
            f"Extract {scenario} recovery Probes",
        )
        results[scenario] = evaluate_scenario(
            scenario, model, gallery, probe.to(device), labels, args
        )
    if args.metrics_path:
        directory = os.path.dirname(args.metrics_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.metrics_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "architecture_version": checkpoint["architecture_version"],
                    "checkpoint_sha256": file_sha256(args.ckpt),
                    "training_mode": configuration["training_mode"],
                    "selection_rule": checkpoint.get("selection_rule"),
                    "gallery_protocol_sha256": file_sha256(args.gallery_list),
                    "probe_protocol_sha256": file_sha256(args.protocol_list),
                    "calibration": checkpoint.get("calibration"),
                    "results": results,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
        print(f"[Info] saved metrics to {args.metrics_path}")
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Evaluate gallery-conditioned feature recovery")
    parser.add_argument("--gallery_list", default="data_txt/tongji/ssfd_gallery_full.txt")
    parser.add_argument("--protocol_list", default="data_txt/tongji/ssfd_test_protocol.txt")
    parser.add_argument(
        "--recovery_ckpt",
        "--ckpt",
        dest="ckpt",
        default="outputs/shared_feature_recovery/recovery_v6/tongji/best.pth",
    )
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
        "--output",
        "--metrics_path",
        dest="metrics_path",
        default="outputs/shared_feature_recovery/recovery_v6/tongji/test_metrics.json",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args())
