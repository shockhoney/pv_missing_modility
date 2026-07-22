"""Evaluate differentiable-CCA spatial Transformer recovery."""

from __future__ import annotations

import argparse
import json
import os

import torch

from models.dcca_specformer import ARCHITECTURE_VERSION, DCCASpecFormerRecovery
from utils.checkpoint import load_encoder_from_checkpoint
from utils.checkpoint_io import file_sha256, safe_torch_load
from utils.evaluation import format_gallery_probe_metrics, score_matrix_metrics
from utils.feature_extraction import paired_feature_loader, single_feature_loader
from utils.runtime import resolve_device, set_random_seed
from utils.scenarios import PALMPRINT_MISSING, PALMVEIN_MISSING
from utils.spatial_feature_extraction import (
    extract_paired_spatial_features,
    extract_single_spatial_features,
)


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


def print_result(name, metrics):
    print(f"[{name}]")
    print("\n".join(format_gallery_probe_metrics(metrics)))


def verify_fingerprints(checkpoint, args):
    fingerprints = checkpoint.get("fingerprints", {})
    for name, path in (("palm_encoder", args.palm_ckpt), ("vein_encoder", args.vein_ckpt)):
        expected = fingerprints.get(f"{name}_sha256")
        if expected is not None and file_sha256(path) != expected:
            raise ValueError(f"{name} checkpoint fingerprint differs from recovery training")


@torch.inference_mode()
def evaluate_direction(model, memory, probes, available, args):
    output = model.recover_with_gallery(
        probes[available], probes[f"{available}_spatial"], available, memory
    )
    labels = output["candidate_labels"]
    metric_kwargs = dict(
        candidate_labels=labels,
        probe_labels=probes["labels"],
        topk=args.top_k,
        far_points=args.far_points,
    )
    branch_scores = torch.cat(
        [output["base_branch_scores"], output["recovered_scores"].unsqueeze(2)], dim=2
    )
    branches = {
        name: score_matrix_metrics(branch_scores[:, :, index], **metric_kwargs)
        for index, name in enumerate(model.BRANCHES)
    }
    base = score_matrix_metrics(output["base_scores"], **metric_kwargs)
    fused = score_matrix_metrics(output["fused_scores"], **metric_kwargs)
    improvement = {
        "eer": base["eer"] - fused["eer"],
        "tar_1e-3": fused["tar_at_far"][1e-3] - base["tar_at_far"][1e-3],
        "tar_1e-4": fused["tar_at_far"][1e-4] - base["tar_at_far"][1e-4],
        "top1": fused["topk"][1] - base["topk"][1],
    }
    print(f"\n===== {available} available / {output['target_modality']} missing ({ARCHITECTURE_VERSION}) =====")
    for name, metrics in branches.items():
        print_result(name, metrics)
    print_result("fusion without recovery", base)
    print_result("fusion with recovery", fused)
    print(
        f"[parameters] temperature={output['temperature'].item():.5f} "
        f"shared_weight={output['shared_weight'].mean().item():.4f} "
        f"recovery_weight={output['recovery_weight'].mean().item():.4f} "
        f"bounds=[{model.min_recovery_weight:.2f}, {model.max_recovery_weight:.2f}] "
        f"fallback_model=None"
    )
    print(
        f"[recovery contribution] EER={improvement['eer']*100:+.4f}pp "
        f"TAR@1e-3={improvement['tar_1e-3']*100:+.2f}pp "
        f"TAR@1e-4={improvement['tar_1e-4']*100:+.2f}pp "
        f"Top1={improvement['top1']*100:+.2f}pp"
    )
    return {
        "branches": branches,
        "fused_without_recovery": base,
        "fused": fused,
        "improvement_over_no_recovery": improvement,
        "posterior_confidence": distribution_summary(output["posterior_confidence"]),
        "posterior_normalized_entropy": distribution_summary(output["posterior_entropy"]),
        "predicted_variance": distribution_summary(output["log_variance"].exp()),
        "learned_gate": distribution_summary(output["learned_gate"]),
        "recovery_gate": distribution_summary(output["recovery_gate"]),
        "shared_weight": distribution_summary(output["shared_weight"]),
        "recovery_weight": distribution_summary(output["recovery_weight"]),
        "recovery_weight_bounds": [
            model.min_recovery_weight, model.max_recovery_weight
        ],
        "recovery_gate_activation": {
            "fraction_gt_0.01": float((output["recovery_gate"] > 0.01).float().mean().item()),
            "fraction_gt_0.05": float((output["recovery_gate"] > 0.05).float().mean().item()),
        },
        "base_weights": {
            name: float(output["base_weights"][0, index].item())
            for index, name in enumerate(model.BRANCHES[:3])
        },
        "temperature": float(output["temperature"].item()),
        "fallback_model": None,
    }


def evaluate(args):
    device = resolve_device(args.device, require_available=True, announce=True)
    set_random_seed(args.seed)
    checkpoint = safe_torch_load(args.ckpt, device)
    saved_args = checkpoint.get("args", {})
    if checkpoint.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError(f"Checkpoint architecture is not {ARCHITECTURE_VERSION}")
    verify_fingerprints(checkpoint, args)
    config = checkpoint["configuration"]
    model = DCCASpecFormerRecovery(
        input_dim=int(config["input_dim"]),
        shared_dim=int(config["shared_dim"]),
        specific_dim=int(config["specific_dim"]),
        min_recovery_weight=float(saved_args.get("min_recovery_weight", 0.15)),
        retrieval_dropout=float(saved_args.get("retrieval_dropout", 0.10)),
        branch_floor=float(saved_args.get("branch_floor", 0.0)),
        transformer_layers=int(config["transformer_layers"]),
        transformer_heads=int(config["transformer_heads"]),
        dropout=float(config["dropout"]),
        max_gate=float(config["max_gate"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    input_size = int(saved_args.get("input_size", args.input_size))
    embedding_size = int(saved_args.get("embedding_size", args.embedding_size))
    palm_encoder = load_encoder_from_checkpoint(args.palm_ckpt, "palm", embedding_size, device)
    vein_encoder = load_encoder_from_checkpoint(args.vein_ckpt, "vein", embedding_size, device)
    gallery = extract_paired_spatial_features(
        palm_encoder,
        vein_encoder,
        paired_feature_loader(
            args.gallery_list, None, input_size, args.extract_batch_size, args.num_workers
        ),
        device,
        "Extract DCCA-SpecFormer Gallery",
    )
    gallery = {key: value.to(device) for key, value in gallery.items()}
    gallery["labels"] = gallery["labels"].long()
    memory = model.build_gallery_memory(gallery, chunk_size=args.memory_batch_size)
    results = {}
    for scenario, modality, encoder in (
        (PALMPRINT_MISSING, "vein", vein_encoder),
        (PALMVEIN_MISSING, "palm", palm_encoder),
    ):
        embeddings, spatial, labels = extract_single_spatial_features(
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
            f"Extract {scenario} DCCA-SpecFormer probes",
        )
        probes = {
            modality: embeddings.to(device),
            f"{modality}_spatial": spatial.to(device),
            "labels": labels.to(device=device, dtype=torch.long),
        }
        results[scenario] = evaluate_direction(model, memory, probes, modality, args)
    if args.metrics_path:
        directory = os.path.dirname(args.metrics_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.metrics_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "architecture_version": checkpoint["architecture_version"],
                    "checkpoint_sha256": file_sha256(args.ckpt),
                    "training_stage": checkpoint.get("training_stage"),
                    "best_epoch": checkpoint.get("best_epoch"),
                    "selection_rule": checkpoint.get("selection_rule"),
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
    parser = argparse.ArgumentParser("Evaluate DCCA-SpecFormer recovery")
    parser.add_argument("--gallery_list", default="data_txt/tongji/ssfd_gallery_full.txt")
    parser.add_argument("--protocol_list", default="data_txt/tongji/ssfd_test_protocol.txt")
    parser.add_argument("--recovery_ckpt", "--ckpt", dest="ckpt", default="outputs/dcca_specformer/v9_1/tongji/best.pth")
    parser.add_argument("--palm_ckpt", default="outputs/encoders/palm_best.pth")
    parser.add_argument("--vein_ckpt", default="outputs/encoders/vein_best.pth")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--embedding_size", type=int, default=256)
    parser.add_argument("--extract_batch_size", type=int, default=128)
    parser.add_argument("--memory_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--far_points", type=float, nargs="+", default=[1e-3, 1e-4])
    parser.add_argument("--output", "--metrics_path", dest="metrics_path", default="outputs/dcca_specformer/v9_1/tongji/test_metrics.json")
    return parser.parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args())
