"""Shared data loading, metrics, and orchestration for staged DCCA training."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from models.dcca_specformer import ARCHITECTURE_VERSION
from utils.checkpoint import load_encoder_from_checkpoint
from utils.checkpoint_io import file_sha256, safe_torch_load, save_checkpoint_atomic
from utils.evaluation import score_matrix_metrics
from utils.runtime import resolve_device, set_random_seed
from utils.scenarios import PALMPRINT_MISSING, PALMVEIN_MISSING
from utils.spatial_feature_extraction import load_or_extract_paired_spatial_cache


DIRECTIONS = (
    (PALMPRINT_MISSING, "vein", "palm"),
    (PALMVEIN_MISSING, "palm", "vein"),
)


def cpu_state_dict(module):
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def score_metrics(scores, candidate_labels, probe_labels):
    return score_matrix_metrics(
        scores.detach().float().cpu(),
        candidate_labels.detach().cpu(),
        probe_labels.detach().cpu(),
        topk=(1, 5),
        far_points=(1e-3, 1e-4),
        warn_far_resolution=False,
    )


def distribution_summary(values):
    values = values.detach().float()
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "median": float(torch.quantile(values, 0.5).item()),
        "q90": float(torch.quantile(values, 0.9).item()),
        "q99": float(torch.quantile(values, 0.99).item()),
        "max": float(values.max().item()),
    }


def supervised_contrastive_loss(features, labels, temperature):
    features = F.normalize(features, dim=1)
    logits = features @ features.t() / temperature
    self_mask = torch.eye(features.size(0), dtype=torch.bool, device=features.device)
    positives = labels[:, None].eq(labels[None, :]) & ~self_mask
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    denominator = torch.exp(logits).masked_fill(self_mask, 0.0).sum(dim=1)
    log_probability = logits - denominator.clamp_min(1e-12).log().unsqueeze(1)
    return -(log_probability * positives).sum(dim=1).div(positives.sum(dim=1)).mean()


def identity_balanced_batches(labels, identities_per_batch, instances, steps, generator):
    labels = labels.cpu()
    identities = labels.unique(sorted=True)
    members = [torch.nonzero(labels == identity, as_tuple=False).flatten() for identity in identities]
    for _ in range(steps):
        count = min(identities_per_batch, len(members))
        selected = torch.randperm(len(members), generator=generator)[:count]
        batch = []
        for identity_index in selected.tolist():
            values = members[identity_index]
            if values.numel() >= instances:
                choice = values[torch.randperm(values.numel(), generator=generator)[:instances]]
            else:
                choice = values[torch.randint(values.numel(), (instances,), generator=generator)]
            batch.append(choice)
        order = torch.randperm(count * instances, generator=generator)
        yield torch.cat(batch)[order]


def episodic_split(labels, generator, gallery_fraction):
    gallery, query = [], []
    for identity in labels.unique(sorted=True).cpu().tolist():
        members = torch.nonzero(labels.cpu() == identity, as_tuple=False).flatten()
        members = members[torch.randperm(members.numel(), generator=generator)]
        gallery_count = min(members.numel() - 2, max(1, round(members.numel() * gallery_fraction)))
        gallery.append(members[:gallery_count])
        query.append(members[gallery_count:])
    return torch.cat(gallery), torch.cat(query)


def select_rows(values, indices, device):
    return {
        key: value[indices].to(device=device, non_blocking=True)
        for key, value in values.items()
    }


def to_device(values, device):
    return {
        key: value.to(device=device, non_blocking=True)
        for key, value in values.items()
    }


def hard_margin(scores, targets):
    row = torch.arange(scores.size(0), device=scores.device)
    positive = scores[row, targets]
    negative = scores.masked_fill(
        F.one_hot(targets, scores.size(1)).bool(), torch.finfo(scores.dtype).min
    ).max(dim=1).values
    return positive - negative




def metrics_for_output(model, output, probe_labels):
    labels = output["candidate_labels"]
    branch_scores = torch.cat(
        [output["base_branch_scores"], output["recovered_scores"].unsqueeze(2)], dim=2
    )
    return {
        "branches": {
            name: score_metrics(branch_scores[:, :, index], labels, probe_labels)
            for index, name in enumerate(model.BRANCHES)
        },
        "fused_without_recovery": score_metrics(output["base_scores"], labels, probe_labels),
        "fused": score_metrics(output["fused_scores"], labels, probe_labels),
        "recovery_gate": distribution_summary(output["recovery_gate"]),
        "learned_gate": distribution_summary(output["learned_gate"]),
        "posterior_confidence": distribution_summary(output["posterior_confidence"]),
        "posterior_entropy": distribution_summary(output["posterior_entropy"]),
        "predicted_variance": distribution_summary(output["log_variance"].exp()),
        "base_weights": output["base_weights"][0].detach().cpu().tolist(),
        "temperature": float(output["temperature"].item()),
        "fallback_model": None,
    }


@torch.inference_mode()
def evaluate_model(model, gallery, probes):
    model.eval()
    memory = model.build_gallery_memory(gallery)
    results = {}
    for scenario, available, _ in DIRECTIONS:
        output = model.recover_with_gallery(
            probes[available], probes[f"{available}_spatial"], available, memory
        )
        results[scenario] = metrics_for_output(model, output, probes["labels"])
    return results


def validation_rank(results):
    fused = [results[scenario]["fused"] for scenario, _, _ in DIRECTIONS]
    return (
        sum(x["eer"] for x in fused) / len(fused),
        max(x["eer"] for x in fused),
        -sum(x["tar_at_far"][1e-4] for x in fused) / len(fused),
        -sum(x["tar_at_far"][1e-3] for x in fused) / len(fused),
    )


def load_sets(args, palm_encoder, vein_encoder, device):
    specs = {"train": (args.train_list, None, "train_spatial.pt", "Extract training spatial features")}
    if not args.fixed_full_train:
        specs.update(
            {
                "val_gallery": (
                    args.val_gallery_list, None, "val_gallery_spatial.pt", "Extract validation gallery spatial features"
                ),
                "val_probe": (
                    args.val_protocol_list, "complete", "val_probe_spatial.pt", "Extract validation probe spatial features"
                ),
            }
        )
    values, metadata = {}, {}
    for name, (source, split, filename, description) in specs.items():
        features, cache_metadata = load_or_extract_paired_spatial_cache(
            os.path.join(args.cache_dir, filename),
            source,
            split,
            palm_encoder,
            vein_encoder,
            args.palm_ckpt,
            args.vein_ckpt,
            device,
            args.input_size,
            args.embedding_size,
            args.extract_batch_size,
            args.num_workers,
            description,
            force=args.force_recache,
        )
        values[name] = to_device(features, device)
        values[name]["labels"] = values[name]["labels"].long()
        metadata[name] = cache_metadata
    return values, metadata



def train(args):
    device = resolve_device(args.device, require_available=True, announce=True)
    set_random_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    palm_encoder = load_encoder_from_checkpoint(args.palm_ckpt, "palm", args.embedding_size, device)
    vein_encoder = load_encoder_from_checkpoint(args.vein_ckpt, "vein", args.embedding_size, device)
    feature_sets, cache_metadata = load_sets(args, palm_encoder, vein_encoder, device)
    del palm_encoder, vein_encoder
    training = feature_sets["train"]
    model = build_model(args, training, device)
    epochs = args.epochs
    schedule_epochs = args.epochs
    selection_checkpoint = None
    if args.fixed_full_train:
        if not args.selection_ckpt:
            raise ValueError("fixed_full_train requires --selection_ckpt")
        selection_checkpoint = safe_torch_load(args.selection_ckpt, "cpu")
        if selection_checkpoint.get("architecture_version") != ARCHITECTURE_VERSION:
            raise ValueError(f"selection_ckpt architecture is not {ARCHITECTURE_VERSION}")
        epochs = int(selection_checkpoint["best_epoch"])
        schedule_epochs = int(selection_checkpoint["args"]["epochs"])
        print(f"[Selection] epochs={epochs}; no fallback or deployment blend")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed + 17)
    history = []
    best = None
    best_state = None
    best_validation = None
    best_epoch = epochs
    for epoch in range(1, epochs + 1):
        started = time.time()
        progress = epoch / max(1, schedule_epochs)
        lr = args.min_learning_rate + 0.5 * (args.learning_rate - args.min_learning_rate) * (
            1.0 + math.cos(math.pi * progress)
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        losses, cca_stats = train_epoch(model, training, optimizer, epoch, args, generator)
        record = {
            "epoch": epoch,
            "learning_rate": lr,
            "losses": losses,
            "cca": cca_stats,
            "elapsed_seconds": time.time() - started,
        }
        if not args.fixed_full_train and epoch % args.eval_every == 0:
            validation = evaluate_model(
                model, feature_sets["val_gallery"], feature_sets["val_probe"]
            )
            rank = validation_rank(validation)
            record["validation"] = validation
            record["validation_rank"] = rank
            if best is None or rank < best:
                best = rank
                best_state = cpu_state_dict(model)
                best_validation = copy.deepcopy(validation)
                best_epoch = epoch
        history.append(record)
        validation_text = ""
        if "validation" in record:
            validation_text = " ".join(
                f"{scenario}:EER={record['validation'][scenario]['fused']['eer']*100:.4f}%/"
                f"base={record['validation'][scenario]['fused_without_recovery']['eer']*100:.4f}%/"
                f"gate={record['validation'][scenario]['recovery_gate']['mean']:.3f}"
                for scenario, _, _ in DIRECTIONS
            )
        print(
            f"[Epoch {epoch:03d}/{epochs:03d}] lr={lr:.2e} total={losses['total']:.4f} "
            f"dcca={losses['dcca']:.4f} rec={losses['reconstruction']:.4f} "
            f"time={record['elapsed_seconds']:.1f}s {validation_text}"
        )
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    fingerprints = {
        "palm_encoder_sha256": file_sha256(args.palm_ckpt),
        "vein_encoder_sha256": file_sha256(args.vein_ckpt),
        "train_list_sha256": file_sha256(args.train_list),
    }
    payload = {
        "architecture_version": ARCHITECTURE_VERSION,
        "training_stage": "fixed_full_train" if args.fixed_full_train else "identity_validation_selection",
        "model": cpu_state_dict(model),
        "args": vars(args),
        "configuration": {
            "input_dim": args.embedding_size,
            "shared_dim": args.shared_dimensions,
            "specific_dim": args.specific_dimensions,
            "transformer_layers": args.transformer_layers,
            "transformer_heads": args.transformer_heads,
            "dropout": args.dropout,
            "max_gate": args.max_gate,
            "min_recovery_weight": getattr(args, "min_recovery_weight", 0.15),
            "retrieval_dropout": getattr(args, "retrieval_dropout", 0.10),
            "branch_floor": getattr(args, "branch_floor", 0.0),
            "branches": list(model.BRANCHES),
        },
        "epoch": epochs,
        "best_epoch": best_epoch,
        "selection_rule": "identity-disjoint minimum mean EER of the end-to-end neural output",
        "validation": best_validation,
        "history": history,
        "fingerprints": fingerprints,
        "feature_cache_metadata": cache_metadata,
        "selection_checkpoint_sha256": None
        if selection_checkpoint is None
        else file_sha256(args.selection_ckpt),
    }
    output = os.path.join(args.save_dir, "best.pth")
    save_checkpoint_atomic(output, payload)
    with open(os.path.join(args.save_dir, "training_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "architecture_version": ARCHITECTURE_VERSION,
                "best_epoch": best_epoch,
                "fallback_model": None,
                "validation": best_validation,
                "history": history,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"[Done] saved={output}")
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Train DCCA-SpecFormer missing-modality recovery")
    parser.add_argument("--train_list", default="data_txt/tongji/ssfd_train_full.txt")
    parser.add_argument("--val_gallery_list", default="data_txt/tongji/ssfd_val_gallery_full.txt")
    parser.add_argument("--val_protocol_list", default="data_txt/tongji/ssfd_val_protocol.txt")
    parser.add_argument("--palm_ckpt", default="outputs/encoders/palm_best.pth")
    parser.add_argument("--vein_ckpt", default="outputs/encoders/vein_best.pth")
    parser.add_argument("--selection_ckpt", default=None)
    parser.add_argument("--save_dir", default="outputs/dcca_specformer/hiasr_v10/tongji_validation")
    parser.add_argument("--cache_dir", default="outputs/dcca_specformer/cache/tongji_validation")
    parser.add_argument("--fixed_full_train", action="store_true")
    parser.add_argument("--force_recache", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--embedding_size", type=int, default=256)
    parser.add_argument("--shared_dimensions", type=int, default=192)
    parser.add_argument("--specific_dimensions", type=int, default=128)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--transformer_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_gate", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--batch_identities", type=int, default=32)
    parser.add_argument("--instances_per_identity", type=int, default=2)
    parser.add_argument("--steps_per_epoch", type=int, default=0)
    parser.add_argument("--episodic_gallery_fraction", type=float, default=0.5)
    parser.add_argument("--memory_batch_size", type=int, default=256)
    parser.add_argument("--extract_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--contrastive_temperature", type=float, default=0.07)
    parser.add_argument("--identity_temperature", type=float, default=0.05)
    parser.add_argument("--gate_target_temperature", type=float, default=0.05)
    parser.add_argument("--hard_margin", type=float, default=0.1)
    parser.add_argument("--cca_dimensions", type=int, default=64)
    parser.add_argument("--cca_ridge", type=float, default=1e-3)
    parser.add_argument("--cca_eigen_floor", type=float, default=1e-6)
    parser.add_argument("--analytic_eigen_floor", type=float, default=1.0)
    parser.add_argument("--nr_interval", type=int, default=10)
    parser.add_argument("--nr_dimensions", type=int, default=16)
    parser.add_argument("--nr_noise_scale", type=float, default=1.0)
    parser.add_argument("--recovery_warmup_epochs", type=int, default=4)
    parser.add_argument("--metric_weight", type=float, default=1.0)
    parser.add_argument("--specific_metric_weight", type=float, default=0.25)
    parser.add_argument("--dcca_weight", type=float, default=0.2)
    parser.add_argument("--nr_weight", type=float, default=0.01)
    parser.add_argument("--reconstruction_weight", type=float, default=0.5)
    parser.add_argument("--identity_weight", type=float, default=0.5)
    parser.add_argument("--rank_weight", type=float, default=1.0)
    parser.add_argument("--safe_weight", type=float, default=1.0)
    parser.add_argument("--gate_weight", type=float, default=0.2)
    parser.add_argument("--anchor_weight", type=float, default=0.2)
    return parser.parse_args(argv)


