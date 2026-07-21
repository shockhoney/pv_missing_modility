"""Train bidirectional probabilistic feature recovery on frozen encoder features.

The test protocol is deliberately absent from this entry point.  Checkpoint
selection normally uses only the two validation EERs.  The fixed full-training
mode uses a previously fixed epoch and directional policy without validation.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F

from models.shared_feature_recovery import (
    TRAINABLE_ARCHITECTURE_VERSION,
    RegularizedSharedIdentityProjector,
    TrainableSharedFeatureRecovery,
)
from utils.checkpoint import load_encoder_from_checkpoint
from utils.checkpoint_io import file_sha256, safe_torch_load, save_checkpoint_atomic
from utils.evaluation import gallery_probe_scores, score_matrix_metrics
from utils.feature_extraction import load_or_extract_paired_feature_cache
from utils.runtime import resolve_device, set_random_seed
from utils.scenarios import PALMPRINT_MISSING, PALMVEIN_MISSING


SCENARIOS = (PALMPRINT_MISSING, PALMVEIN_MISSING)


def metric_summary(metrics):
    return (
        f"EER={metrics['eer'] * 100:.4f}% "
        f"TAR@1e-3={metrics['tar_at_far'][1e-3] * 100:.2f}% "
        f"TAR@1e-4={metrics['tar_at_far'][1e-4] * 100:.2f}% "
        f"Top1={metrics['topk'][1] * 100:.2f}%"
    )


def score_metrics(scores, candidate_labels, probe_labels):
    return score_matrix_metrics(
        scores.detach().float().cpu(),
        candidate_labels.detach().cpu(),
        probe_labels.detach().cpu(),
        topk=(1, 5),
        far_points=(1e-3, 1e-4),
        warn_far_resolution=False,
    )


def cpu_state_dict(module):
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def normalized_class_templates(features, class_indices, num_classes):
    sums = features.new_zeros(num_classes, features.size(1))
    sums.index_add_(0, class_indices, features)
    counts = torch.bincount(class_indices, minlength=num_classes).to(features).unsqueeze(1)
    if torch.any(counts == 0):
        raise ValueError("Every training identity must have at least one sample")
    return F.normalize(sums / counts, dim=1)


def identity_balanced_batches(class_indices, identities_per_batch, instances, steps, generator):
    class_indices = class_indices.cpu()
    num_classes = int(class_indices.max().item()) + 1
    members = [torch.nonzero(class_indices == index, as_tuple=False).flatten() for index in range(num_classes)]
    if any(values.numel() == 0 for values in members):
        raise ValueError("Identity-balanced sampling found an empty identity")
    for _ in range(steps):
        if identities_per_batch <= num_classes:
            selected = torch.randperm(num_classes, generator=generator)[:identities_per_batch]
        else:
            selected = torch.randint(num_classes, (identities_per_batch,), generator=generator)
        batch = []
        for identity in selected.tolist():
            values = members[identity]
            if values.numel() >= instances:
                choice = values[torch.randperm(values.numel(), generator=generator)[:instances]]
            else:
                choice = values[torch.randint(values.numel(), (instances,), generator=generator)]
            batch.append(choice)
        yield torch.cat(batch)[torch.randperm(identities_per_batch * instances, generator=generator)]


def supervised_contrastive_loss(features, labels, temperature):
    features = F.normalize(features, dim=1)
    logits = features @ features.t() / temperature
    self_mask = torch.eye(features.size(0), dtype=torch.bool, device=features.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    if torch.any(positive_mask.sum(dim=1) == 0):
        raise ValueError("Supervised contrastive batches require a positive for every sample")
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_prob = logits - exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12).log()
    return -(log_prob * positive_mask).sum(dim=1).div(positive_mask.sum(dim=1)).mean()


def positive_and_hard_negative(scores, class_indices):
    row = torch.arange(scores.size(0), device=scores.device)
    positive = scores[row, class_indices]
    negative = scores.masked_fill(
        F.one_hot(class_indices, scores.size(1)).bool(),
        torch.finfo(scores.dtype).min,
    ).max(dim=1).values
    return positive, negative


def hard_identity_loss(query, templates, class_indices, margin):
    scores = F.normalize(query, dim=1) @ F.normalize(templates, dim=1).t()
    positive, negative = positive_and_hard_negative(scores, class_indices)
    return F.relu(margin + negative - positive).mean(), positive - negative


def gaussian_nll(mean, logvar, target):
    return 0.5 * (logvar + (target - mean).square() * torch.exp(-logvar)).mean()


def recovery_direction_losses(
    model,
    available,
    target,
    available_modality,
    target_templates,
    class_indices,
    args,
):
    output = model.recover(available, available_modality)
    mean = output["mean"]
    logvar = output["logvar"]
    cosine_error = 1.0 - F.cosine_similarity(mean, target, dim=1)
    nll = gaussian_nll(mean, logvar, target)
    cosine = cosine_error.mean()
    hard, recovered_margin = hard_identity_loss(
        mean, target_templates, class_indices, args.hard_margin
    )
    target_shared = model.project(mean, output["target_modality"])
    cycle = (1.0 - F.cosine_similarity(target_shared, output["shared"], dim=1)).mean()
    reliability_target = torch.exp(-cosine_error.detach() / args.reliability_temperature)
    gate = F.smooth_l1_loss(output["reliability"], reliability_target)
    return output, {
        "nll": nll,
        "cos": cosine,
        "hard": hard,
        "cycle": cycle,
        "gate": gate,
        "recovered_margin": recovered_margin,
    }


def fused_direction_losses(
    output,
    available,
    available_shared_templates,
    target_shared_templates,
    available_templates,
    target_templates,
    class_indices,
    args,
):
    branches = torch.stack(
        [
            F.normalize(available, dim=1) @ available_templates.t(),
            output["shared"] @ available_shared_templates.t(),
            output["shared"] @ target_shared_templates.t(),
            F.normalize(output["mean"], dim=1) @ target_templates.t(),
        ],
        dim=2,
    )
    fused = (branches * output["weights"].unsqueeze(1)).sum(dim=2)
    fused_positive, fused_negative = positive_and_hard_negative(fused, class_indices)
    available_positive, available_negative = positive_and_hard_negative(branches[:, :, 0], class_indices)
    fused_margin = fused_positive - fused_negative
    available_margin = available_positive - available_negative
    return {
        "fused_hard": F.relu(args.hard_margin + fused_negative - fused_positive).mean(),
        "safe": F.relu(available_margin - fused_margin).mean(),
        "fused_ce": F.cross_entropy(fused / args.score_temperature, class_indices),
    }


@torch.no_grad()
def build_training_banks(model, training, class_indices, num_classes):
    was_training = model.training
    model.eval()
    banks = {
        "palm": normalized_class_templates(training["palm"], class_indices, num_classes),
        "vein": normalized_class_templates(training["vein"], class_indices, num_classes),
        "palm_shared": normalized_class_templates(
            model.project(training["palm"], "palm"), class_indices, num_classes
        ),
        "vein_shared": normalized_class_templates(
            model.project(training["vein"], "vein"), class_indices, num_classes
        ),
    }
    model.train(was_training)
    return banks


@torch.inference_mode()
def evaluate_direction(model, gallery, probes, available, target):
    available_scores, labels = gallery_probe_scores(
        gallery[available], gallery["labels"], probes[available]
    )
    output = model.recover(probes[available], available)
    gallery_available_shared = model.project(gallery[available], available)
    gallery_target_shared = model.project(gallery[target], target)
    same_scores, same_labels = gallery_probe_scores(
        gallery_available_shared, gallery["labels"], output["shared"]
    )
    cross_scores, cross_labels = gallery_probe_scores(
        gallery_target_shared, gallery["labels"], output["shared"]
    )
    recovered_scores, recovered_labels = gallery_probe_scores(
        gallery[target], gallery["labels"], output["mean"]
    )
    if not (
        torch.equal(labels, same_labels)
        and torch.equal(labels, cross_labels)
        and torch.equal(labels, recovered_labels)
    ):
        raise ValueError("Candidate label orders differ across validation branches")
    branches = torch.stack(
        [available_scores, same_scores, cross_scores, recovered_scores], dim=2
    )
    fused_scores = (branches * output["weights"].unsqueeze(1)).sum(dim=2)
    branch_names = TrainableSharedFeatureRecovery.BRANCHES
    return {
        "metrics": score_metrics(fused_scores, labels, probes["labels"]),
        "branches": {
            name: score_metrics(branches[:, :, index], labels, probes["labels"])
            for index, name in enumerate(branch_names)
        },
        "reliability": {
            "mean": float(output["reliability"].mean().item()),
            "q10": float(torch.quantile(output["reliability"], 0.1).item()),
            "median": float(torch.quantile(output["reliability"], 0.5).item()),
            "q90": float(torch.quantile(output["reliability"], 0.9).item()),
        },
        "mean_weights": {
            name: float(output["weights"][:, index].mean().item())
            for index, name in enumerate(branch_names)
        },
    }


@torch.inference_mode()
def validate(model, gallery, probes):
    model.eval()
    return {
        PALMPRINT_MISSING: evaluate_direction(model, gallery, probes, "vein", "palm"),
        PALMVEIN_MISSING: evaluate_direction(model, gallery, probes, "palm", "vein"),
    }


@torch.no_grad()
def projector_validation_safe_rollback(
    model,
    initial_projectors,
    initial_validation,
    gallery,
    probes,
    min_delta,
):
    """Keep a residual projector only when its direction improves validation EER."""

    trained_validation = validate(model, gallery, probes)
    decisions = {}
    directions = (
        ("vein", PALMPRINT_MISSING, model.vein_projector, model.vein_refiner_enabled),
        ("palm", PALMVEIN_MISSING, model.palm_projector, model.palm_refiner_enabled),
    )
    for modality, scenario, projector, enabled in directions:
        baseline_eer = initial_validation[scenario]["branches"]["shared_same"]["eer"]
        trained_eer = trained_validation[scenario]["branches"]["shared_same"]["eer"]
        keep = trained_eer < baseline_eer - min_delta
        if not keep:
            projector.load_state_dict(initial_projectors[modality], strict=True)
            enabled.fill_(False)
        decisions[modality] = {
            "scenario": scenario,
            "initial_eer": baseline_eer,
            "trained_eer": trained_eer,
            "kept_trainable_residual": keep,
        }
    return decisions


@torch.no_grad()
def apply_prevalidated_projector_policy(model, initial_projectors):
    """Apply the directional policy fixed by the earlier train/validation run."""

    model.vein_projector.load_state_dict(initial_projectors["vein"], strict=True)
    model.vein_refiner_enabled.fill_(False)
    model.palm_refiner_enabled.fill_(True)
    return {
        "vein": {
            "source": "prevalidated_432_train_48_validation_protocol",
            "kept_trainable_residual": False,
        },
        "palm": {
            "source": "prevalidated_432_train_48_validation_protocol",
            "kept_trainable_residual": True,
        },
    }


def selection_rank(validation, epoch):
    palm_eer = validation[PALMPRINT_MISSING]["metrics"]["eer"]
    vein_eer = validation[PALMVEIN_MISSING]["metrics"]["eer"]
    return (max(palm_eer, vein_eer), 0.5 * (palm_eer + vein_eer), int(epoch))


def set_stage_trainability(model, stage):
    for parameter in model.parameters():
        parameter.requires_grad = False
    modules = []
    if bool(model.palm_refiner_enabled.item()):
        modules.append(model.palm_projector.refiner)
    if bool(model.vein_refiner_enabled.item()):
        modules.append(model.vein_projector.refiner)
    if stage >= 2:
        modules += [model.p2v, model.v2p, model.p2v_reliability, model.v2p_reliability]
    if stage >= 3:
        modules += [model.p2v_gate, model.v2p_gate]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True


def stage_for_epoch(epoch, args):
    if epoch <= args.shared_warmup_epochs:
        return 1
    if epoch <= args.recovery_end_epoch:
        return 2
    return 3


def stage_bounds(stage, args):
    if stage == 1:
        return 1, args.shared_warmup_epochs
    if stage == 2:
        return args.shared_warmup_epochs + 1, args.recovery_end_epoch
    return args.recovery_end_epoch + 1, args.epochs


def stage_lr_factor(epoch, stage, args):
    first, last = stage_bounds(stage, args)
    local_epoch = epoch - first + 1
    length = max(1, last - first + 1)
    warmup = min(args.lr_warmup_epochs, length)
    if local_epoch <= warmup:
        return local_epoch / max(1, warmup)
    progress = (local_epoch - warmup) / max(1, length - warmup)
    return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def update_learning_rates(optimizer, epoch, stage, args):
    factor = stage_lr_factor(epoch, stage, args)
    for group in optimizer.param_groups:
        name = group["name"]
        if name == "projector_base":
            base = args.projector_lr * (0.5 if stage == 3 else 1.0)
        elif name == "projector_refiner":
            base = (args.projector_refiner_lr if stage == 1 else args.projector_lr)
            if stage == 3:
                base *= 0.5
        elif name == "recoverer":
            base = args.recoverer_lr * (0.5 if stage == 3 else 1.0)
        else:
            base = args.gate_lr
        group["lr"] = base * factor


def build_optimizer(model, args):
    projector_base = list(model.palm_projector.base.parameters()) + list(
        model.vein_projector.base.parameters()
    )
    projector_refiner = list(model.palm_projector.refiner.parameters()) + list(
        model.vein_projector.refiner.parameters()
    )
    recoverer = list(model.p2v.parameters()) + list(model.v2p.parameters())
    reliability = list(model.p2v_reliability.parameters()) + list(
        model.v2p_reliability.parameters()
    )
    gates = list(model.p2v_gate.parameters()) + list(model.v2p_gate.parameters())
    return torch.optim.AdamW(
        [
            {"params": projector_base, "name": "projector_base"},
            {"params": projector_refiner, "name": "projector_refiner"},
            {"params": recoverer, "name": "recoverer"},
            {"params": reliability, "name": "reliability"},
            {"params": gates, "name": "gate"},
        ],
        lr=args.projector_lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )


def average_loss_dict(totals, count):
    return {name: value / max(1, count) for name, value in totals.items()}


def train_epoch(
    model,
    optimizer,
    training,
    reference_shared,
    class_indices,
    num_classes,
    epoch,
    args,
    generator,
):
    stage = stage_for_epoch(epoch, args)
    set_stage_trainability(model, stage)
    update_learning_rates(optimizer, epoch, stage, args)
    model.train()
    banks = build_training_banks(model, training, class_indices, num_classes)
    steps = args.steps_per_epoch or math.ceil(
        training["labels"].numel() / (args.batch_identities * args.instances_per_identity)
    )
    totals = {}
    for cpu_indices in identity_balanced_batches(
        class_indices,
        args.batch_identities,
        args.instances_per_identity,
        steps,
        generator,
    ):
        indices = cpu_indices.to(training["palm"].device)
        palm = training["palm"][indices]
        vein = training["vein"][indices]
        labels = class_indices[indices]
        palm_shared = model.project(palm, "palm")
        vein_shared = model.project(vein, "vein")
        supcon = 0.5 * (
            supervised_contrastive_loss(palm_shared, labels, args.temperature)
            + supervised_contrastive_loss(vein_shared, labels, args.temperature)
        )
        preserve = 0.5 * (
            (1.0 - F.cosine_similarity(palm_shared, reference_shared["palm"][indices], dim=1)).mean()
            + (1.0 - F.cosine_similarity(vein_shared, reference_shared["vein"][indices], dim=1)).mean()
        )
        if stage == 1:
            palm_hard, _ = hard_identity_loss(
                palm_shared, banks["palm_shared"], labels, args.hard_margin
            )
            vein_hard, _ = hard_identity_loss(
                vein_shared, banks["vein_shared"], labels, args.hard_margin
            )
            shared_hard = 0.5 * (palm_hard + vein_hard)
            losses = {"supcon": supcon, "shared_hard": shared_hard, "preserve": preserve}
            total = supcon + shared_hard + args.preserve_weight * preserve
        else:
            p2v_output, p2v = recovery_direction_losses(
                model, palm, vein, "palm", banks["vein"], labels, args
            )
            v2p_output, v2p = recovery_direction_losses(
                model, vein, palm, "vein", banks["palm"], labels, args
            )
            losses = {
                name: 0.5 * (p2v[name] + v2p[name])
                for name in ("nll", "cos", "hard", "cycle", "gate")
            }
            losses["supcon"] = supcon
            losses["preserve"] = preserve
            total = (
                losses["nll"]
                + losses["cos"]
                + args.supcon_weight * losses["supcon"]
                + losses["hard"]
                + args.cycle_weight * losses["cycle"]
                + args.gate_calibration_weight * losses["gate"]
                + args.anchor_weight * model.anchor_loss()
                + args.preserve_weight * losses["preserve"]
            )
            if stage == 3:
                p2v_fused = fused_direction_losses(
                    p2v_output,
                    palm,
                    banks["palm_shared"],
                    banks["vein_shared"],
                    banks["palm"],
                    banks["vein"],
                    labels,
                    args,
                )
                v2p_fused = fused_direction_losses(
                    v2p_output,
                    vein,
                    banks["vein_shared"],
                    banks["palm_shared"],
                    banks["vein"],
                    banks["palm"],
                    labels,
                    args,
                )
                for name in ("fused_hard", "safe", "fused_ce"):
                    losses[name] = 0.5 * (p2v_fused[name] + v2p_fused[name])
                total = (
                    total
                    + losses["fused_hard"]
                    + args.safe_weight * losses["safe"]
                    + args.fused_ce_weight * losses["fused_ce"]
                )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"Non-finite gradient norm at epoch {epoch}")
        optimizer.step()
        losses["total"] = total.detach()
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().item())
    return stage, average_loss_dict(totals, steps)


def checkpoint_payload(
    model,
    optimizer,
    args,
    epoch,
    validation,
    best_rank,
    best_epoch,
    history,
    fingerprints,
    caches,
    include_optimizer,
):
    payload = {
        "architecture_version": TRAINABLE_ARCHITECTURE_VERSION,
        "training_stage": "backprop_probabilistic_feature_recovery",
        "model": cpu_state_dict(model),
        "args": vars(args),
        "configuration": {
            "input_dim": args.embedding_size,
            "shared_dim": args.shared_dimensions,
            "hidden_dim": args.hidden_dim,
            "gate_hidden_dim": args.gate_hidden_dim,
            "dropout": args.dropout,
            "unit_input": False,
            "eigen_floor": args.eigen_floor,
            "ridge": args.ridge,
            "training_mode": (
                "fixed_full_train_no_validation"
                if args.fixed_full_train
                else "validation_selected"
            ),
            "branches": list(TrainableSharedFeatureRecovery.BRANCHES),
        },
        "epoch": int(epoch),
        "validation": validation,
        "selection_rank": list(best_rank) if best_rank is not None else None,
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "selection_rule": (
            "fixed_epoch_with_prevalidated_directional_projector_policy"
            if args.fixed_full_train
            else "lexicographic(max_direction_eer, mean_direction_eer, epoch)"
        ),
        "history": history,
        "fingerprints": fingerprints,
        "feature_cache_metadata": caches,
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    return payload


def verify_resume(checkpoint, args, fingerprints):
    if checkpoint.get("architecture_version") != TRAINABLE_ARCHITECTURE_VERSION:
        raise ValueError("Resume checkpoint architecture version is incompatible")
    saved = checkpoint.get("configuration", {})
    if saved.get("input_dim") != args.embedding_size or saved.get("shared_dim") != args.shared_dimensions:
        raise ValueError("Resume checkpoint feature dimensions are incompatible")
    expected_mode = (
        "fixed_full_train_no_validation"
        if args.fixed_full_train
        else "validation_selected"
    )
    if saved.get("training_mode", "validation_selected") != expected_mode:
        raise ValueError("Resume checkpoint training mode is incompatible")
    if checkpoint.get("fingerprints") != fingerprints:
        raise ValueError("Resume checkpoint encoder or protocol fingerprints differ")


def load_feature_sets(args, palm_encoder, vein_encoder, device):
    specs = {
        "train": (args.train_list, None, "train_features.pt", "Extract training features"),
    }
    if not args.fixed_full_train:
        specs.update(
            {
                "val_gallery": (
                    args.val_gallery_list,
                    None,
                    "val_gallery_features.pt",
                    "Extract validation gallery features",
                ),
                "val_probe": (
                    args.val_protocol_list,
                    "complete",
                    "val_probe_features.pt",
                    "Extract validation probe features",
                ),
            }
        )
    values, metadata = {}, {}
    for name, (path, split, filename, description) in specs.items():
        features, cache_metadata = load_or_extract_paired_feature_cache(
            os.path.join(args.cache_dir, filename),
            path,
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


def train(args):
    device = resolve_device(args.device, require_available=True, announce=True)
    set_random_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    fingerprints = {
        "palm_encoder_sha256": file_sha256(args.palm_ckpt),
        "vein_encoder_sha256": file_sha256(args.vein_ckpt),
        "train_list_sha256": file_sha256(args.train_list),
    }
    if not args.fixed_full_train:
        fingerprints.update(
            {
                "val_gallery_list_sha256": file_sha256(args.val_gallery_list),
                "val_protocol_list_sha256": file_sha256(args.val_protocol_list),
            }
        )
    palm_encoder = load_encoder_from_checkpoint(
        args.palm_ckpt, "palm", args.embedding_size, device
    )
    vein_encoder = load_encoder_from_checkpoint(
        args.vein_ckpt, "vein", args.embedding_size, device
    )
    for encoder in (palm_encoder, vein_encoder):
        for parameter in encoder.parameters():
            parameter.requires_grad = False
    feature_sets, cache_metadata = load_feature_sets(
        args, palm_encoder, vein_encoder, device
    )
    del palm_encoder, vein_encoder
    training = feature_sets["train"]
    gallery = feature_sets.get("val_gallery")
    probes = feature_sets.get("val_probe")
    label_values = training["labels"].unique(sorted=True)
    class_indices = torch.searchsorted(label_values, training["labels"])
    if not torch.equal(label_values[class_indices], training["labels"]):
        raise ValueError("Failed to map training labels to class indices")
    num_classes = label_values.numel()
    cca = RegularizedSharedIdentityProjector(args.embedding_size, unit_input=False).to(device)
    cca.fit(training["palm"], training["vein"], args.eigen_floor)
    model = TrainableSharedFeatureRecovery(
        input_dim=args.embedding_size,
        shared_dim=args.shared_dimensions,
        hidden_dim=args.hidden_dim,
        gate_hidden_dim=args.gate_hidden_dim,
        dropout=args.dropout,
        unit_input=False,
    ).to(device)
    model.initialize_from_cca(
        cca,
        training["palm"],
        training["vein"],
        ridge=args.ridge,
        reliability_temperature=args.reliability_temperature,
    )
    with torch.no_grad():
        reference_shared = {
            "palm": model.project(training["palm"], "palm").detach().clone(),
            "vein": model.project(training["vein"], "vein").detach().clone(),
        }
    initial_projectors = {
        "palm": cpu_state_dict(model.palm_projector),
        "vein": cpu_state_dict(model.vein_projector),
    }
    initial_validation = (
        None if args.fixed_full_train else validate(model, gallery, probes)
    )
    optimizer = build_optimizer(model, args)
    start_epoch = 1
    best_rank = None
    best_epoch = None
    history = []
    stale_epochs = 0
    if args.resume:
        checkpoint = safe_torch_load(args.resume, device)
        verify_resume(checkpoint, args, fingerprints)
        model.load_state_dict(checkpoint["model"], strict=True)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        rank = checkpoint.get("selection_rank")
        best_rank = tuple(rank) if rank is not None else None
        best_epoch = checkpoint.get("best_epoch")
        history = checkpoint.get("history", [])
        patience_start = max(
            args.recovery_end_epoch,
            int(best_epoch or args.recovery_end_epoch),
        )
        stale_epochs = max(0, int(checkpoint["epoch"]) - patience_start)
        print(f"[Resume] epoch={checkpoint['epoch']} best_epoch={best_epoch}")
    with torch.inference_mode():
        smoke = model.recover(training["palm"][:2], "palm")
        if smoke["mean"].shape != (2, args.embedding_size):
            raise ValueError("Recovery smoke test produced an invalid mean shape")
        if not all(torch.isfinite(value).all() for value in smoke.values() if torch.is_tensor(value)):
            raise FloatingPointError("Recovery smoke test produced non-finite tensors")
        if not torch.allclose(smoke["weights"].sum(dim=1), torch.ones(2, device=device), atol=1e-6):
            raise ValueError("Dynamic gate weights do not sum to one")
    print(
        f"[Init] pairs={training['labels'].numel()} identities={num_classes} "
        f"shared_dim={args.shared_dimensions} mean_CCA={cca.canonical_correlations[:args.shared_dimensions].mean().item():.4f} "
        f"mode={'fixed_full_train_no_validation' if args.fixed_full_train else 'validation_selected'}"
    )
    generator = torch.Generator().manual_seed(args.seed + start_epoch)
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        stage, losses = train_epoch(
            model,
            optimizer,
            training,
            reference_shared,
            class_indices,
            num_classes,
            epoch,
            args,
            generator,
        )
        rollback_decisions = None
        if epoch == args.shared_warmup_epochs:
            if args.fixed_full_train:
                rollback_decisions = apply_prevalidated_projector_policy(
                    model, initial_projectors
                )
            else:
                rollback_decisions = projector_validation_safe_rollback(
                    model,
                    initial_projectors,
                    initial_validation,
                    gallery,
                    probes,
                    args.early_stop_min_delta,
                )
            print(f"[Projector safety rollback] {rollback_decisions}")
        validation = None
        improved = False
        if not args.fixed_full_train and epoch > args.shared_warmup_epochs:
            validation = validate(model, gallery, probes)
            rank = selection_rank(validation, epoch)
            if best_rank is None or (
                rank[0] < best_rank[0] - args.early_stop_min_delta
                or (
                    abs(rank[0] - best_rank[0]) <= args.early_stop_min_delta
                    and rank[1:] < best_rank[1:]
                )
            ):
                best_rank = rank
                best_epoch = epoch
                stale_epochs = 0
                improved = True
            else:
                stale_epochs = stale_epochs + 1 if epoch > args.recovery_end_epoch else 0
        elif args.fixed_full_train and epoch == args.epochs:
            best_epoch = epoch
            best_rank = None
            improved = True
        record = {
            "epoch": epoch,
            "stage": stage,
            "losses": losses,
            "validation": validation,
            "projector_safe_rollback": rollback_decisions,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        loss_text = " ".join(f"{name}={value:.4f}" for name, value in losses.items())
        print(f"[Epoch {epoch:03d} stage={stage}] {loss_text} time={record['elapsed_seconds']:.1f}s")
        if validation is not None:
            for scenario in SCENARIOS:
                result = validation[scenario]
                print(
                    f"  [Val {scenario}] {metric_summary(result['metrics'])} "
                    f"q={result['reliability']['mean']:.3f} weights={result['mean_weights']}"
                )
        last_path = os.path.join(args.save_dir, "last.pth")
        save_checkpoint_atomic(
            last_path,
            checkpoint_payload(
                model,
                optimizer,
                args,
                epoch,
                validation,
                best_rank,
                best_epoch,
                history,
                fingerprints,
                cache_metadata,
                include_optimizer=True,
            ),
        )
        if improved:
            best_path = os.path.join(args.save_dir, "best.pth")
            save_checkpoint_atomic(
                best_path,
                checkpoint_payload(
                    model,
                    optimizer,
                    args,
                    epoch,
                    validation,
                    best_rank,
                    best_epoch,
                    history,
                    fingerprints,
                    cache_metadata,
                    include_optimizer=False,
                ),
            )
            if args.fixed_full_train:
                print(f"  [Fixed checkpoint] epoch={best_epoch} saved={best_path}")
            else:
                print(f"  [Best] rank={best_rank} saved={best_path}")
        if (
            stage == 3
            and validation is not None
            and stale_epochs >= args.early_stop_patience
        ):
            print(f"[Early stop] no validation EER improvement for {stale_epochs} epochs")
            break
    if best_epoch is None:
        raise RuntimeError("Training ended before a checkpoint was selected or fixed")
    print(f"[Done] best_epoch={best_epoch} selection_rank={best_rank}")
    return os.path.join(args.save_dir, "best.pth")


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Train probabilistic shared feature recovery")
    parser.add_argument("--train_list", default="data_txt/tongji/ssfd_train_full.txt")
    parser.add_argument("--val_gallery_list", default="data_txt/tongji/ssfd_val_gallery_full.txt")
    parser.add_argument("--val_protocol_list", default="data_txt/tongji/ssfd_val_protocol.txt")
    parser.add_argument("--palm_ckpt", default="outputs/encoders/palm_best.pth")
    parser.add_argument("--vein_ckpt", default="outputs/encoders/vein_best.pth")
    parser.add_argument("--save_dir", default="outputs/shared_feature_recovery/trainable_v2")
    parser.add_argument("--cache_dir", default="outputs/shared_feature_recovery/cache")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--fixed_full_train",
        action="store_true",
        help="disable validation and save the fixed final epoch using the prevalidated rollback policy",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--embedding_size", type=int, default=256)
    parser.add_argument("--shared_dimensions", type=int, default=192)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--gate_hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eigen_floor", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--shared_warmup_epochs", type=int, default=20)
    parser.add_argument("--recovery_end_epoch", type=int, default=70)
    parser.add_argument("--batch_identities", type=int, default=32)
    parser.add_argument("--instances_per_identity", type=int, default=2)
    parser.add_argument("--steps_per_epoch", type=int, default=0)
    parser.add_argument("--extract_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--projector_lr", type=float, default=1e-4)
    parser.add_argument("--projector_refiner_lr", type=float, default=5e-4)
    parser.add_argument("--recoverer_lr", type=float, default=1e-3)
    parser.add_argument("--gate_lr", type=float, default=1e-3)
    parser.add_argument("--lr_warmup_epochs", type=int, default=5)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_clip", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--reliability_temperature", type=float, default=0.1)
    parser.add_argument("--score_temperature", type=float, default=0.05)
    parser.add_argument("--hard_margin", type=float, default=0.1)
    parser.add_argument("--supcon_weight", type=float, default=0.5)
    parser.add_argument("--cycle_weight", type=float, default=0.25)
    parser.add_argument("--gate_calibration_weight", type=float, default=0.1)
    parser.add_argument("--safe_weight", type=float, default=0.5)
    parser.add_argument("--fused_ce_weight", type=float, default=0.2)
    parser.add_argument("--anchor_weight", type=float, default=1e-2)
    parser.add_argument("--preserve_weight", type=float, default=2.0)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--force_recache", action="store_true")
    parser.add_argument("--selection_metric", choices=("eer_worst",), default="eer_worst")
    args = parser.parse_args(argv)
    if not 1 <= args.shared_dimensions <= args.embedding_size:
        parser.error("shared_dimensions must be in [1, embedding_size]")
    if args.fixed_full_train:
        if not 0 <= args.shared_warmup_epochs < args.epochs <= args.recovery_end_epoch:
            parser.error(
                "fixed_full_train boundaries must satisfy 0 <= warmup < epochs <= recovery_end"
            )
    elif not 0 <= args.shared_warmup_epochs < args.recovery_end_epoch < args.epochs:
        parser.error(
            "stage boundaries must satisfy 0 <= warmup < recovery_end < epochs"
        )
    if args.batch_identities < 2 or args.instances_per_identity < 2:
        parser.error("identity-balanced batches need >=2 identities and >=2 instances")
    if args.steps_per_epoch < 0:
        parser.error("steps_per_epoch must be non-negative")
    if args.temperature <= 0 or args.reliability_temperature <= 0 or args.score_temperature <= 0:
        parser.error("temperatures must be positive")
    if args.early_stop_patience <= 0 or args.early_stop_min_delta < 0:
        parser.error("early-stop settings are invalid")
    return args


if __name__ == "__main__":
    train(parse_args())
