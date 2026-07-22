"""Fit shared projectors and calibrate closed-set prototype recovery.

Frozen single-modal encoders provide 256D features. The recovery module never
hallucinates an unidentifiable target-private vector from the source alone.
Instead, it forms an identity posterior in the available/shared domain and
recovers a target-modality prototype from the enrolled complete gallery.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F

from models.shared_feature_recovery import (
    ARCHITECTURE_VERSION,
    RegularizedSharedIdentityProjector,
    TrainableSharedFeatureRecovery,
)
from utils.checkpoint import load_encoder_from_checkpoint
from utils.checkpoint_io import file_sha256, safe_torch_load, save_checkpoint_atomic
from utils.evaluation import score_matrix_metrics
from utils.feature_extraction import load_or_extract_paired_feature_cache
from utils.runtime import resolve_device, set_random_seed
from utils.scenarios import PALMPRINT_MISSING, PALMVEIN_MISSING


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
        "q10": float(torch.quantile(values, 0.1).item()),
        "median": float(torch.quantile(values, 0.5).item()),
        "q90": float(torch.quantile(values, 0.9).item()),
        "q95": float(torch.quantile(values, 0.95).item()),
        "q99": float(torch.quantile(values, 0.99).item()),
        "max": float(values.max().item()),
    }


def normalized_class_templates(features, class_indices, num_classes):
    sums = features.new_zeros(num_classes, features.size(1))
    sums.index_add_(0, class_indices, features)
    counts = torch.bincount(class_indices, minlength=num_classes).to(features).unsqueeze(1)
    return F.normalize(sums / counts, dim=1)


def identity_balanced_batches(class_indices, identities_per_batch, instances, steps, generator):
    class_indices = class_indices.cpu()
    num_classes = int(class_indices.max().item()) + 1
    members = [
        torch.nonzero(class_indices == index, as_tuple=False).flatten()
        for index in range(num_classes)
    ]
    for _ in range(steps):
        selected = torch.randperm(num_classes, generator=generator)[:identities_per_batch]
        batch = []
        for identity in selected.tolist():
            values = members[identity]
            if values.numel() >= instances:
                choice = values[
                    torch.randperm(values.numel(), generator=generator)[:instances]
                ]
            else:
                choice = values[
                    torch.randint(values.numel(), (instances,), generator=generator)
                ]
            batch.append(choice)
        order = torch.randperm(identities_per_batch * instances, generator=generator)
        yield torch.cat(batch)[order]


def supervised_contrastive_loss(features, labels, temperature):
    features = F.normalize(features, dim=1)
    logits = features @ features.t() / temperature
    self_mask = torch.eye(features.size(0), dtype=torch.bool, device=features.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_prob = logits - exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12).log()
    return -(log_prob * positive_mask).sum(dim=1).div(positive_mask.sum(dim=1)).mean()


def hard_identity_loss(query, templates, class_indices, margin):
    scores = F.normalize(query, dim=1) @ F.normalize(templates, dim=1).t()
    row = torch.arange(scores.size(0), device=scores.device)
    positive = scores[row, class_indices]
    negative = scores.masked_fill(
        F.one_hot(class_indices, scores.size(1)).bool(),
        torch.finfo(scores.dtype).min,
    ).max(dim=1).values
    return F.relu(margin + negative - positive).mean()


@torch.inference_mode()
def evaluate_direction(model, gallery, probes, available, target):
    output = model.recover_with_gallery(
        probes[available],
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
    branches = {
        name: score_metrics(branch_scores[:, :, index], labels, probes["labels"])
        for index, name in enumerate(model.BRANCHES)
    }
    return {
        "branches": branches,
        "fused_without_recovery": score_metrics(
            output["base_scores"], labels, probes["labels"]
        ),
        "fused": score_metrics(output["fused_scores"], labels, probes["labels"]),
        "posterior_confidence": distribution_summary(output["posterior_confidence"]),
        "recovery_reliability": distribution_summary(output["recovery_reliability"]),
        "recovery_gate": distribution_summary(output["recovery_gate"]),
        "base_weights": {
            name: float(output["base_weights"][0, index].item())
            for index, name in enumerate(model.BRANCHES[:3])
        },
    }


@torch.inference_mode()
def evaluate(model, gallery, probes):
    model.eval()
    return {
        scenario: evaluate_direction(model, gallery, probes, available, target)
        for scenario, available, target in DIRECTIONS
    }


def copy_projectors_from_checkpoint(model, checkpoint_path, device):
    checkpoint = safe_torch_load(checkpoint_path, device)
    state = checkpoint["model"]
    for modality in ("palm", "vein"):
        prefix = f"{modality}_projector."
        projector_state = {
            name[len(prefix) :]: value
            for name, value in state.items()
            if name.startswith(prefix)
        }
        if not projector_state:
            raise ValueError(f"No {modality} projector found in {checkpoint_path}")
        getattr(model, f"{modality}_projector").load_state_dict(
            projector_state, strict=True
        )
        enabled_name = f"{modality}_refiner_enabled"
        if enabled_name in state:
            getattr(model, enabled_name).copy_(state[enabled_name])
    return {
        "source": checkpoint_path,
        "sha256": file_sha256(checkpoint_path),
        "source_architecture": checkpoint.get("architecture_version"),
        "source_epoch": checkpoint.get("epoch"),
    }


@torch.no_grad()
def shared_validation_rollback(model, initial_projectors, initial_validation, gallery, probes):
    trained_validation = evaluate(model, gallery, probes)
    decisions = {}
    for modality, scenario in (("vein", PALMPRINT_MISSING), ("palm", PALMVEIN_MISSING)):
        baseline = initial_validation[scenario]["branches"]["shared_same"]["eer"]
        trained = trained_validation[scenario]["branches"]["shared_same"]["eer"]
        keep = trained < baseline
        if not keep:
            getattr(model, f"{modality}_projector").load_state_dict(
                initial_projectors[modality], strict=True
            )
            getattr(model, f"{modality}_refiner_enabled").fill_(False)
        decisions[modality] = {
            "scenario": scenario,
            "initial_eer": baseline,
            "trained_eer": trained,
            "kept_trainable_residual": keep,
        }
    return decisions


@torch.no_grad()
def apply_projector_policy(model, initial_projectors, policy):
    decisions = {}
    for modality in ("palm", "vein"):
        keep = bool(policy[modality]["kept_trainable_residual"])
        if not keep:
            getattr(model, f"{modality}_projector").load_state_dict(
                initial_projectors[modality], strict=True
            )
        getattr(model, f"{modality}_refiner_enabled").fill_(keep)
        decisions[modality] = {
            "source": "identity_disjoint_calibration_checkpoint",
            "kept_trainable_residual": keep,
        }
    return decisions


def train_projectors(model, training, class_indices, args, gallery, probes, policy):
    reference_shared = {
        "palm": model.project(training["palm"], "palm").detach().clone(),
        "vein": model.project(training["vein"], "vein").detach().clone(),
    }
    initial_projectors = {
        "palm": cpu_state_dict(model.palm_projector),
        "vein": cpu_state_dict(model.vein_projector),
    }
    initial_validation = None if gallery is None else evaluate(model, gallery, probes)
    parameters = list(model.palm_projector.refiner.parameters()) + list(
        model.vein_projector.refiner.parameters()
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.projector_refiner_lr,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed + 1)
    history = []
    num_classes = int(class_indices.max().item()) + 1
    steps = args.steps_per_epoch or math.ceil(
        training["labels"].numel()
        / (args.batch_identities * args.instances_per_identity)
    )
    for epoch in range(1, args.shared_epochs + 1):
        started = time.time()
        progress = epoch / max(1, args.shared_epochs)
        factor = args.min_lr_ratio + (1.0 - args.min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
        for group in optimizer.param_groups:
            group["lr"] = args.projector_refiner_lr * factor
        model.train()
        palm_bank = normalized_class_templates(
            model.project(training["palm"], "palm").detach(),
            class_indices,
            num_classes,
        )
        vein_bank = normalized_class_templates(
            model.project(training["vein"], "vein").detach(),
            class_indices,
            num_classes,
        )
        totals = {"supcon": 0.0, "hard": 0.0, "preserve": 0.0, "total": 0.0}
        for cpu_indices in identity_balanced_batches(
            class_indices,
            args.batch_identities,
            args.instances_per_identity,
            steps,
            generator,
        ):
            indices = cpu_indices.to(training["palm"].device)
            labels = class_indices[indices]
            palm_shared = model.project(training["palm"][indices], "palm")
            vein_shared = model.project(training["vein"][indices], "vein")
            supcon = 0.5 * (
                supervised_contrastive_loss(palm_shared, labels, args.temperature)
                + supervised_contrastive_loss(vein_shared, labels, args.temperature)
            )
            hard = 0.5 * (
                hard_identity_loss(palm_shared, palm_bank, labels, args.hard_margin)
                + hard_identity_loss(vein_shared, vein_bank, labels, args.hard_margin)
            )
            preserve = 0.5 * (
                (1.0 - F.cosine_similarity(
                    palm_shared, reference_shared["palm"][indices], dim=1
                )).mean()
                + (1.0 - F.cosine_similarity(
                    vein_shared, reference_shared["vein"][indices], dim=1
                )).mean()
            )
            total = supcon + hard + args.preserve_weight * preserve
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
            optimizer.step()
            for name, value in (
                ("supcon", supcon), ("hard", hard), ("preserve", preserve), ("total", total)
            ):
                totals[name] += float(value.detach().item())
        record = {
            "epoch": epoch,
            "losses": {name: value / steps for name, value in totals.items()},
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(f"[Shared {epoch:03d}] {record['losses']} time={record['elapsed_seconds']:.1f}s")
    model.eval()
    if gallery is not None:
        decisions = shared_validation_rollback(
            model, initial_projectors, initial_validation, gallery, probes
        )
    else:
        decisions = apply_projector_policy(model, initial_projectors, policy)
    return history, decisions


def alpha_grid(args):
    count = int(round((args.alpha_max - args.alpha_min) / args.alpha_step))
    return [args.alpha_min + index * args.alpha_step for index in range(count + 1)]


@torch.inference_mode()
def calibrate_direction(model, gallery, probes, available, target, temperatures, args):
    best = None
    for available_weight in args.available_weight_grid:
        for cross_weight in args.cross_weight_grid:
            same_weight = 1.0 - available_weight - cross_weight
            if same_weight < 0:
                continue
            base_weights = torch.tensor(
                [available_weight, same_weight, cross_weight],
                device=gallery["palm"].device,
            )
            for temperature in temperatures:
                model.set_calibration(
                    available, base_weights, temperature, 0.5, 0.0, 0.2, 1.0
                )
                output = model.recover_with_gallery(
                    probes[available],
                    available,
                    gallery[available],
                    gallery[target],
                    gallery["labels"],
                )
                labels = output["candidate_labels"]
                base_scores = output["base_scores"]
                recovered_scores = output["recovered_scores"]
                reliability = output["recovery_reliability"]
                base = score_metrics(base_scores, labels, probes["labels"])
                for margin_floor in args.margin_floor_grid:
                    for margin_ceiling in args.margin_ceiling_grid:
                        if margin_ceiling <= margin_floor:
                            continue
                        for margin_slope in args.margin_slope_grid:
                            unit_gate = torch.sigmoid(
                                (reliability - margin_floor) / margin_slope
                            ) * torch.sigmoid(
                                (margin_ceiling - reliability) / margin_slope
                            )
                            for alpha in alpha_grid(args):
                                gate = alpha * unit_gate
                                fused_scores = (
                                    (1.0 - gate.unsqueeze(1)) * base_scores
                                    + gate.unsqueeze(1) * recovered_scores
                                )
                                fused = score_metrics(
                                    fused_scores, labels, probes["labels"]
                                )
                                if fused["eer"] > base["eer"] or (
                                    fused["eer"] == base["eer"]
                                    and base["eer"] > 0.0
                                    and not args.allow_validation_eer_tie
                                ):
                                    continue
                                rank = (
                                    fused["eer"],
                                    -fused["tar_at_far"][1e-4],
                                    -fused["tar_at_far"][1e-3],
                                    -fused["topk"][1],
                                    alpha,
                                    temperature,
                                    margin_floor,
                                    margin_ceiling,
                                    margin_slope,
                                )
                                if best is None or rank < best[0]:
                                    best = (
                                        rank,
                                        {
                                            "base_weights": base_weights.detach().cpu().tolist(),
                                            "temperature": float(temperature),
                                            "recovery_alpha": float(alpha),
                                            "margin_floor": float(margin_floor),
                                            "margin_ceiling": float(margin_ceiling),
                                            "margin_slope": float(margin_slope),
                                        },
                                    )
    if best is None:
        raise RuntimeError(
            f"No calibration for {available}->{target} improved validation EER"
        )
    selected = best[1]
    model.set_calibration(
        available,
        torch.tensor(selected["base_weights"], device=gallery["palm"].device),
        selected["temperature"],
        selected["recovery_alpha"],
        selected["margin_floor"],
        selected["margin_ceiling"],
        selected["margin_slope"],
    )
    selected["validation"] = evaluate_direction(
        model, gallery, probes, available, target
    )
    return selected


def load_feature_sets(args, palm_encoder, vein_encoder, device):
    specs = {
        "train": (args.train_list, None, "train_features.pt", "Extract training features"),
    }
    if not args.fixed_full_train:
        specs.update(
            {
                "val_gallery": (
                    args.val_gallery_list, None, "val_gallery_features.pt", "Extract validation gallery"
                ),
                "val_probe": (
                    args.val_protocol_list, "complete", "val_probe_features.pt", "Extract validation probes"
                ),
            }
        )
    values, metadata = {}, {}
    for name, (source, split, filename, description) in specs.items():
        features, cache_metadata = load_or_extract_paired_feature_cache(
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
        values[name] = {
            key: value.to(device) if key != "labels" else value.to(device=device, dtype=torch.long)
            for key, value in features.items()
        }
        metadata[name] = cache_metadata
    return values, metadata


def checkpoint_payload(model, args, fingerprints, caches, history, policy, calibration, source):
    return {
        "architecture_version": ARCHITECTURE_VERSION,
        "training_stage": "gallery_conditioned_prototype_recovery",
        "model": cpu_state_dict(model),
        "args": vars(args),
        "configuration": {
            "input_dim": args.embedding_size,
            "shared_dim": args.shared_dimensions,
            "dropout": args.dropout,
            "unit_input": False,
            "training_mode": "fixed_full_train" if args.fixed_full_train else "identity_validation_calibrated",
            "branches": list(model.BRANCHES),
            "recovery_type": "closed_set_gallery_conditioned_target_prototype",
            "confidence_type": "band_pass_available_top1_top2_cosine_margin",
        },
        "epoch": len(history),
        "best_epoch": len(history),
        "selection_rule": "selective_margin_band_strict_eer_gain_then_low_far_tar",
        "fingerprints": fingerprints,
        "feature_cache_metadata": caches,
        "projector_history": history,
        "projector_policy": policy,
        "calibration": calibration,
        "projector_source": source,
    }


def train(args):
    device = resolve_device(args.device, require_available=True, announce=True)
    set_random_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    fingerprints = {
        "palm_encoder_sha256": file_sha256(args.palm_ckpt),
        "vein_encoder_sha256": file_sha256(args.vein_ckpt),
        "train_list_sha256": file_sha256(args.train_list),
    }
    palm_encoder = load_encoder_from_checkpoint(
        args.palm_ckpt, "palm", args.embedding_size, device
    )
    vein_encoder = load_encoder_from_checkpoint(
        args.vein_ckpt, "vein", args.embedding_size, device
    )
    feature_sets, cache_metadata = load_feature_sets(
        args, palm_encoder, vein_encoder, device
    )
    del palm_encoder, vein_encoder
    training = feature_sets["train"]
    label_values = training["labels"].unique(sorted=True)
    class_indices = torch.searchsorted(label_values, training["labels"])
    model = TrainableSharedFeatureRecovery(
        input_dim=args.embedding_size,
        shared_dim=args.shared_dimensions,
        dropout=args.dropout,
        unit_input=False,
    ).to(device)
    cca = RegularizedSharedIdentityProjector(args.embedding_size, unit_input=False).to(device)
    cca.fit(training["palm"], training["vein"], args.eigen_floor)
    model.initialize_from_cca(cca)
    calibration_checkpoint = None
    policy = None
    if args.fixed_full_train:
        if not args.calibration_ckpt:
            raise ValueError("fixed_full_train requires --calibration_ckpt")
        calibration_checkpoint = safe_torch_load(args.calibration_ckpt, device)
        if calibration_checkpoint.get("architecture_version") != ARCHITECTURE_VERSION:
            raise ValueError("Calibration checkpoint is not a v6 recovery checkpoint")
        policy = calibration_checkpoint["projector_policy"]
    if args.projector_ckpt:
        projector_source = copy_projectors_from_checkpoint(
            model, args.projector_ckpt, device
        )
        history = []
        policy = policy or {
            modality: {
                "source": args.projector_ckpt,
                "kept_trainable_residual": bool(
                    getattr(model, f"{modality}_refiner_enabled").item()
                ),
            }
            for modality in ("palm", "vein")
        }
        print(f"[Projector migration] {projector_source}")
    else:
        history, policy = train_projectors(
            model,
            training,
            class_indices,
            args,
            feature_sets.get("val_gallery"),
            feature_sets.get("val_probe"),
            policy,
        )
        projector_source = {"source": "trained_from_frozen_encoder_features"}
    if args.fixed_full_train:
        model.load_state_dict(
            {
                **model.state_dict(),
                **{
                    name: value
                    for name, value in calibration_checkpoint["model"].items()
                    if name.startswith("p2v_")
                    or name.startswith("v2p_")
                    or name.endswith("margin_floor")
                    or name.endswith("margin_ceiling")
                    or name.endswith("margin_slope")
                },
            },
            strict=True,
        )
        calibration = calibration_checkpoint["calibration"]
    else:
        gallery = feature_sets["val_gallery"]
        probes = feature_sets["val_probe"]
        calibration = {
            PALMPRINT_MISSING: calibrate_direction(
                model, gallery, probes, "vein", "palm", args.v2p_temperature_grid, args
            ),
            PALMVEIN_MISSING: calibrate_direction(
                model, gallery, probes, "palm", "vein", args.p2v_temperature_grid, args
            ),
        }
        final_validation = evaluate(model, gallery, probes)
        for scenario in (PALMPRINT_MISSING, PALMVEIN_MISSING):
            calibration[scenario]["validation"] = final_validation[scenario]
            result = final_validation[scenario]
            print(
                f"[Calibration {scenario}] base_EER={result['fused_without_recovery']['eer'] * 100:.4f}% "
                f"recovered_EER={result['branches']['recovered']['eer'] * 100:.4f}% "
                f"fused_EER={result['fused']['eer'] * 100:.4f}% "
                f"gate={result['recovery_gate']['mean']:.4f}"
            )
    payload = checkpoint_payload(
        model,
        args,
        fingerprints,
        cache_metadata,
        history,
        policy,
        calibration,
        projector_source,
    )
    output = os.path.join(args.save_dir, "best.pth")
    save_checkpoint_atomic(output, payload)
    print(f"[Done] saved={output}")
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Train gallery-conditioned feature recovery")
    parser.add_argument("--train_list", default="data_txt/tongji/ssfd_train_full.txt")
    parser.add_argument("--val_gallery_list", default="data_txt/tongji/ssfd_val_gallery_full.txt")
    parser.add_argument("--val_protocol_list", default="data_txt/tongji/ssfd_val_protocol.txt")
    parser.add_argument("--palm_ckpt", default="outputs/encoders/palm_best.pth")
    parser.add_argument("--vein_ckpt", default="outputs/encoders/vein_best.pth")
    parser.add_argument("--projector_ckpt", default=None)
    parser.add_argument("--calibration_ckpt", default=None)
    parser.add_argument("--save_dir", default="outputs/shared_feature_recovery/recovery_v6/tongji_validation")
    parser.add_argument("--cache_dir", default="outputs/shared_feature_recovery/cache")
    parser.add_argument("--fixed_full_train", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--embedding_size", type=int, default=256)
    parser.add_argument("--shared_dimensions", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eigen_floor", type=float, default=1.0)
    parser.add_argument("--shared_epochs", type=int, default=20)
    parser.add_argument("--batch_identities", type=int, default=32)
    parser.add_argument("--instances_per_identity", type=int, default=2)
    parser.add_argument("--steps_per_epoch", type=int, default=0)
    parser.add_argument("--extract_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--projector_refiner_lr", type=float, default=5e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_clip", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--hard_margin", type=float, default=0.1)
    parser.add_argument("--preserve_weight", type=float, default=2.0)
    parser.add_argument("--available_weight_grid", type=float, nargs="+", default=[0.0])
    parser.add_argument("--cross_weight_grid", type=float, nargs="+", default=[0.0])
    parser.add_argument("--p2v_temperature_grid", type=float, nargs="+", default=[0.05])
    parser.add_argument("--v2p_temperature_grid", type=float, nargs="+", default=[0.01])
    parser.add_argument("--alpha_min", type=float, default=0.05)
    parser.add_argument("--alpha_max", type=float, default=0.50)
    parser.add_argument("--alpha_step", type=float, default=0.01)
    parser.add_argument(
        "--margin_floor_grid", type=float, nargs="+", default=[0.0, 0.01, 0.02]
    )
    parser.add_argument(
        "--margin_ceiling_grid",
        type=float,
        nargs="+",
        default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
    )
    parser.add_argument(
        "--margin_slope_grid",
        type=float,
        nargs="+",
        default=[0.005, 0.01, 0.02],
    )
    parser.add_argument("--allow_validation_eer_tie", action="store_true")
    parser.add_argument("--force_recache", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.shared_dimensions <= args.embedding_size:
        parser.error("shared_dimensions must be in [1, embedding_size]")
    if args.batch_identities < 2 or args.instances_per_identity < 2:
        parser.error("identity-balanced batches require at least two identities and instances")
    if not 0 < args.alpha_min <= args.alpha_max <= 1 or args.alpha_step <= 0:
        parser.error("alpha grid is invalid")
    if any(value < 0 for value in args.margin_floor_grid):
        parser.error("margin floors must be non-negative")
    if any(value <= 0 for value in args.margin_ceiling_grid):
        parser.error("margin ceilings must be positive")
    if any(value <= 0 for value in args.margin_slope_grid):
        parser.error("margin slopes must be positive")
    return args


if __name__ == "__main__":
    train(parse_args())
