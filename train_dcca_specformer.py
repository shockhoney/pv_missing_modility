"""Train the staged DCCA and Transformer missing-modality recovery model."""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from utils import dcca_training_support as training
from models.dcca_specformer import ARCHITECTURE_VERSION, DCCASpecFormerRecovery
from utils.analytic_cca import RegularizedSharedIdentityProjector
from utils.stable_differentiable_cca import deep_cca_loss, nr_dcca_loss


def low_far_pauc_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    margin: float,
    temperature: float,
    fractions: tuple[float, ...] = (0.01, 0.05),
) -> torch.Tensor:
    """Pair positives against the hardest negative score quantiles."""

    if scores.size(1) < 2:
        raise ValueError("pAUC loss requires at least two candidate identities")
    row = torch.arange(scores.size(0), device=scores.device)
    positives = scores[row, targets].unsqueeze(1)
    mask = F.one_hot(targets, scores.size(1)).bool()
    negatives = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    losses = []
    available = scores.size(1) - 1
    for fraction in fractions:
        count = max(1, min(available, math.ceil(available * float(fraction))))
        hard_negatives = negatives.topk(count, dim=1).values
        violations = (hard_negatives - positives + float(margin)) / float(temperature)
        losses.append(F.softplus(violations).mean() * float(temperature))
    return torch.stack(losses).mean()


def configure_stage(model: DCCASpecFormerRecovery, shared_only: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(not shared_only)
    if shared_only:
        for projector in (model.palm_projector, model.vein_projector):
            for parameter in projector.refiner.parameters():
                parameter.requires_grad_(True)
        model.identity_proxies.requires_grad_(True)
        model.recovery_stage_ready.fill_(False)
    else:
        for module in (model.palm_projector, model.vein_projector):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        model.identity_proxies.requires_grad_(False)


def directional_losses(
    output,
    source_embedding,
    target_encoding,
    target_embedding,
    target_indices,
    args,
):
    cosine_error = 1.0 - F.cosine_similarity(output["mean"], target_embedding, dim=1)
    specific_error = 1.0 - F.cosine_similarity(
        output["predicted_specific"], target_encoding["specific"], dim=1
    )
    log_variance = output["log_variance"]
    reconstruction = (
        torch.exp(-log_variance) * cosine_error + 0.5 * log_variance
    ).mean() + specific_error.mean()
    cycle = (
        1.0 - F.cosine_similarity(output["cycle"], source_embedding, dim=1)
    ).mean()
    identity = F.cross_entropy(
        output["recovered_scores"] / args.identity_temperature, target_indices
    )
    base_margin = training.hard_margin(output["base_scores"], target_indices)
    recovered_margin = training.hard_margin(output["recovered_scores"], target_indices)
    fused_margin = training.hard_margin(output["fused_scores"], target_indices)
    rank = F.relu(args.hard_margin - fused_margin).mean()
    safe = F.relu(base_margin.detach() - fused_margin).mean()
    target_gate = torch.sigmoid(
        (recovered_margin.detach() - base_margin.detach())
        / args.gate_target_temperature
    )
    normalized_gate = (
        output["learned_gate"] - args.min_recovery_weight
    ) / (args.max_gate - args.min_recovery_weight)
    gate = F.binary_cross_entropy(
        normalized_gate.clamp(1e-5, 1.0 - 1e-5), target_gate
    )
    pauc = 0.5 * (
        low_far_pauc_loss(
            output["fused_scores"],
            target_indices,
            args.pauc_margin,
            args.pauc_temperature,
        )
        + low_far_pauc_loss(
            output["recovered_scores"],
            target_indices,
            args.pauc_margin,
            args.pauc_temperature,
        )
    )
    return {
        "reconstruction": reconstruction,
        "cycle": cycle,
        "identity": identity,
        "rank": rank,
        "safe": safe,
        "gate": gate,
        "pauc": pauc,
    }


def _empty_totals():
    return {
        name: 0.0
        for name in (
            "metric",
            "proxy",
            "pair_alignment",
            "dcca",
            "nr",
            "reconstruction",
            "cycle",
            "identity",
            "rank",
            "safe",
            "gate",
            "pauc",
            "anchor",
            "retrieval_dropout",
            "total",
        )
    }


def _statistics_payload(statistics):
    if statistics is None:
        return None
    return {
        "correlation": float(statistics.correlation.item()),
        "effective_rank_palm": float(statistics.effective_rank_palm.item()),
        "effective_rank_vein": float(statistics.effective_rank_vein.item()),
    }


def train_shared_stage(model, training_set, optimizer, args, generator):
    configure_stage(model, shared_only=True)
    model.train()
    labels_cpu = training_set["labels"].cpu()
    steps = args.steps_per_epoch or math.ceil(
        labels_cpu.numel() / (args.batch_identities * args.instances_per_identity)
    )
    totals = _empty_totals()
    cca_statistics = None
    for step, indices_cpu in enumerate(
        training.identity_balanced_batches(
            labels_cpu,
            args.batch_identities,
            args.instances_per_identity,
            steps,
            generator,
        ),
        1,
    ):
        indices = indices_cpu.to(training_set["palm"].device)
        labels = training_set["labels"][indices]
        palm_raw = model.palm_projector.forward_raw(training_set["palm"][indices])
        vein_raw = model.vein_projector.forward_raw(training_set["vein"][indices])
        palm_shared = F.normalize(palm_raw, dim=1)
        vein_shared = F.normalize(vein_raw, dim=1)
        paired_labels = torch.cat([labels, labels])
        metric = training.supervised_contrastive_loss(
            torch.cat([palm_shared, vein_shared]),
            paired_labels,
            args.contrastive_temperature,
        )
        palm_logits, proxy_targets = model.proxy_logits(
            palm_shared, labels, args.proxy_temperature
        )
        vein_logits, _ = model.proxy_logits(
            vein_shared, labels, args.proxy_temperature
        )
        proxy = 0.5 * (
            F.cross_entropy(palm_logits, proxy_targets)
            + F.cross_entropy(vein_logits, proxy_targets)
        )
        pair_alignment = (
            1.0 - F.cosine_similarity(palm_shared, vein_shared, dim=1)
        ).mean()
        cca_topk = min(args.cca_dimensions, palm_raw.size(1), labels.numel() - 1)
        dcca, cca_statistics = deep_cca_loss(
            palm_raw,
            vein_raw,
            ridge=args.cca_ridge,
            eigen_floor=args.cca_eigen_floor,
            topk=cca_topk,
        )
        nr = dcca.new_zeros(())
        if args.nr_weight > 0 and step % args.nr_interval == 0:
            nr = 0.5 * (
                nr_dcca_loss(
                    model.palm_projector,
                    training_set["palm"][indices],
                    noise_scale=args.nr_noise_scale,
                    ridge=args.cca_ridge,
                    topk=args.nr_dimensions,
                )
                + nr_dcca_loss(
                    model.vein_projector,
                    training_set["vein"][indices],
                    noise_scale=args.nr_noise_scale,
                    ridge=args.cca_ridge,
                    topk=args.nr_dimensions,
                )
            )
        anchor = model.anchor_loss()
        total = (
            args.metric_weight * metric
            + args.proxy_weight * proxy
            + args.pair_alignment_weight * pair_alignment
            + args.dcca_weight * dcca
            + args.nr_weight * nr
            + args.anchor_weight * anchor
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            args.gradient_clip,
        )
        optimizer.step()
        values = {
            "metric": metric,
            "proxy": proxy,
            "pair_alignment": pair_alignment,
            "dcca": dcca,
            "nr": nr,
            "anchor": anchor,
            "total": total,
        }
        for name, value in values.items():
            totals[name] += float(value.detach().item())
    return (
        {name: value / steps for name, value in totals.items()},
        _statistics_payload(cca_statistics),
    )


def train_recovery_stage(model, training_set, optimizer, epoch, args, generator):
    configure_stage(model, shared_only=False)
    recovery_epoch = epoch - args.shared_stage_epochs
    model.recovery_stage_ready.fill_(
        recovery_epoch >= args.minimum_recovery_epochs
    )
    gallery_indices, query_indices = training.episodic_split(
        training_set["labels"], generator, args.episodic_gallery_fraction
    )
    model.eval()
    gallery = training.select_rows(
        training_set, gallery_indices, training_set["palm"].device
    )
    memory = model.build_gallery_memory(gallery, chunk_size=args.memory_batch_size)
    model.train()
    query_labels = training_set["labels"][query_indices].cpu()
    steps = args.steps_per_epoch or math.ceil(
        query_indices.numel() / (args.batch_identities * args.instances_per_identity)
    )
    totals = _empty_totals()
    for local_indices in training.identity_balanced_batches(
        query_labels,
        args.batch_identities,
        args.instances_per_identity,
        steps,
        generator,
    ):
        indices = query_indices[local_indices].to(training_set["palm"].device)
        labels = training_set["labels"][indices]
        palm = model.encode(
            training_set["palm"][indices],
            training_set["palm_spatial"][indices],
            "palm",
        )
        vein = model.encode(
            training_set["vein"][indices],
            training_set["vein_spatial"][indices],
            "vein",
        )
        specific_metric = 0.5 * (
            training.supervised_contrastive_loss(
                palm["specific"], labels, args.contrastive_temperature
            )
            + training.supervised_contrastive_loss(
                vein["specific"], labels, args.contrastive_temperature
            )
        )
        outputs = {
            "p2v": model.score_from_encoding(palm, "palm", memory),
            "v2p": model.score_from_encoding(vein, "vein", memory),
        }
        target_indices = torch.searchsorted(memory["labels"], labels)
        p2v = directional_losses(
            outputs["p2v"],
            palm["embedding"],
            vein,
            F.normalize(training_set["vein"][indices], dim=1),
            target_indices,
            args,
        )
        v2p = directional_losses(
            outputs["v2p"],
            vein["embedding"],
            palm,
            F.normalize(training_set["palm"][indices], dim=1),
            target_indices,
            args,
        )
        directional = {
            name: 0.5 * (p2v[name] + v2p[name]) for name in p2v
        }
        dropout_fraction = 0.5 * (
            outputs["p2v"]["retrieval_dropout_fraction"]
            + outputs["v2p"]["retrieval_dropout_fraction"]
        )
        recovery_ramp = min(
            1.0, recovery_epoch / max(1, args.recovery_warmup_epochs)
        )
        total = args.specific_metric_weight * specific_metric + recovery_ramp * (
            args.reconstruction_weight * directional["reconstruction"]
            + args.cycle_weight * directional["cycle"]
            + args.identity_weight * directional["identity"]
            + args.rank_weight * directional["rank"]
            + args.safe_weight * directional["safe"]
            + args.gate_weight * directional["gate"]
            + args.pauc_weight * directional["pauc"]
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            args.gradient_clip,
        )
        optimizer.step()
        values = {
            "metric": specific_metric,
            **directional,
            "retrieval_dropout": dropout_fraction,
            "total": total,
        }
        for name, value in values.items():
            totals[name] += float(value.detach().item())
    return {name: value / steps for name, value in totals.items()}, None


def train_epoch(model, training_set, optimizer, epoch, args, generator):
    if epoch <= args.shared_stage_epochs:
        return train_shared_stage(model, training_set, optimizer, args, generator)
    return train_recovery_stage(
        model, training_set, optimizer, epoch, args, generator
    )


def build_model(args, training_set, device):
    model = DCCASpecFormerRecovery(
        input_dim=args.embedding_size,
        shared_dim=args.shared_dimensions,
        specific_dim=args.specific_dimensions,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        dropout=args.dropout,
        max_gate=args.max_gate,
        min_recovery_weight=args.min_recovery_weight,
        retrieval_dropout=args.retrieval_dropout,
        branch_floor=args.branch_floor,
    ).to(device)
    cca = RegularizedSharedIdentityProjector(
        args.embedding_size, unit_input=False
    ).to(device)
    cca.fit(
        training_set["palm"],
        training_set["vein"],
        args.analytic_eigen_floor,
    )
    model.initialize_from_cca(cca)
    model.initialize_identity_proxies(
        training_set["palm"],
        training_set["vein"],
        training_set["labels"],
    )
    return model


def metrics_for_output(model, output, probe_labels):
    result = base_metrics_for_output(model, output, probe_labels)
    result.update(
        {
            "shared_weight": training.distribution_summary(output["shared_weight"]),
            "recovery_weight": training.distribution_summary(
                output["recovery_weight"]
            ),
            "retrieval_dropout_fraction": float(
                output["retrieval_dropout_fraction"].item()
            ),
            "recovery_weight_bounds": [
                model.min_recovery_weight,
                model.max_recovery_weight,
            ],
        }
    )
    return result


@torch.inference_mode()
def evaluate_model(model, gallery, probes):
    model.eval()
    memory = model.build_gallery_memory(gallery)
    results = {}
    for scenario, available, _ in training.DIRECTIONS:
        output = model.recover_with_gallery(
            probes[available],
            probes[f"{available}_spatial"],
            available,
            memory,
        )
        result = metrics_for_output(model, output, probes["labels"])
        result["recovery_stage_ready"] = bool(model.recovery_stage_ready.item())
        results[scenario] = result
    return results


def validation_rank(results):
    if not all(
        results[scenario].get("recovery_stage_ready", False)
        for scenario, _, _ in training.DIRECTIONS
    ):
        return (float("inf"), float("inf"), 0.0, 0.0)
    return base_validation_rank(results)


def parse_args():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--shared_stage_epochs", type=int, default=2)
    extra.add_argument("--minimum_recovery_epochs", type=int, default=4)
    extra.add_argument("--proxy_weight", type=float, default=0.5)
    extra.add_argument("--proxy_temperature", type=float, default=0.05)
    extra.add_argument("--pair_alignment_weight", type=float, default=0.5)
    extra.add_argument("--cycle_weight", type=float, default=0.25)
    extra.add_argument("--pauc_weight", type=float, default=0.5)
    extra.add_argument("--pauc_margin", type=float, default=0.05)
    extra.add_argument("--pauc_temperature", type=float, default=0.05)
    extra.add_argument("--retrieval_dropout", type=float, default=0.10)
    extra.add_argument("--min_recovery_weight", type=float, default=0.15)
    extra.add_argument("--branch_floor", type=float, default=0.0)
    values, remaining = extra.parse_known_args()
    args = training.parse_args(remaining)
    for name, value in vars(values).items():
        setattr(args, name, value)
    if "--max_gate" not in remaining:
        args.max_gate = 0.75
    if "--safe_weight" not in remaining:
        args.safe_weight = 0.25
    if args.epochs <= args.shared_stage_epochs:
        raise ValueError("epochs must exceed shared_stage_epochs")
    return args


base_metrics_for_output = training.metrics_for_output
base_validation_rank = training.validation_rank
training.ARCHITECTURE_VERSION = ARCHITECTURE_VERSION
training.build_model = build_model
training.train_epoch = train_epoch
training.metrics_for_output = metrics_for_output
training.evaluate_model = evaluate_model
training.validation_rank = validation_rank


if __name__ == "__main__":
    training.train(parse_args())
