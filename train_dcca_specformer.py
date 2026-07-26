"""Train the final HIASR missing-modality recovery network by backpropagation."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from utils import recovery_backbone_training as backbone
from models.dcca_specformer import ARCHITECTURE_VERSION, DCCASpecFormerRecovery
from utils import dcca_training_support as orchestration
from utils.checkpoint_io import safe_torch_load


INTERNAL_BACKBONE_VERSION = backbone.INTERNAL_BACKBONE_VERSION
backbone_directional_losses = backbone.directional_losses
backbone_metrics_for_output = backbone.metrics_for_output


def configure_stage(model: DCCASpecFormerRecovery, shared_only: bool) -> None:
    del shared_only
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.shared_disentangler, model.hierarchical_decoder):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.refinement_logit.requires_grad_(True)


def build_model(args, training_set, device):
    del training_set
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
        topk_candidates=args.topk_candidates,
        role_queries=args.role_queries,
        candidate_dropout=args.candidate_dropout,
        max_refinement=args.max_refinement,
    ).to(device)
    checkpoint = safe_torch_load(args.warm_start_ckpt, device)
    if checkpoint.get("architecture_version") != INTERNAL_BACKBONE_VERSION:
        raise ValueError("warm_start_ckpt must be an internal stable-backbone checkpoint")
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    expected_prefixes = (
        "shared_disentangler.",
        "hierarchical_decoder.",
        "refinement_logit",
    )
    unexpected_missing = [
        name
        for name in incompatible.missing_keys
        if not name.startswith(expected_prefixes)
    ]
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Warm-start mismatch: missing={unexpected_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    configure_stage(model, shared_only=False)
    return model


def directional_losses(
    output,
    source_embedding,
    target_encoding,
    target_embedding,
    target_indices,
    args,
):
    losses = backbone_directional_losses(
        output,
        source_embedding,
        target_encoding,
        target_embedding,
        target_indices,
        args,
    )
    student_margin = orchestration.hard_margin(
        output["fused_scores"], target_indices
    )
    teacher_margin = orchestration.hard_margin(
        output["teacher_fused_scores"], target_indices
    )
    teacher_safe = F.relu(teacher_margin.detach() - student_margin).mean()
    losses["safe"] = losses["safe"] + args.teacher_safe_weight * teacher_safe
    losses["reconstruction"] = (
        losses["reconstruction"]
        + args.orthogonal_weight * output["orthogonality"].mean()
    )
    return losses


def metrics_for_output(model, output, probe_labels):
    result = backbone_metrics_for_output(model, output, probe_labels)
    result.update(
        {
            "orthogonality": orchestration.distribution_summary(
                output["orthogonality"]
            ),
            "candidate_keep_fraction": float(
                output["candidate_keep_fraction"].item()
            ),
            "refinement_gate": orchestration.distribution_summary(
                output["refinement_gate"]
            ),
            "refinement_active_fraction": float(
                output["refinement_active_fraction"].item()
            ),
            "topk_candidates": model.topk_candidates,
            "role_queries": model.role_queries,
        }
    )
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--warm_start_ckpt", required=True)
    parser.add_argument("--topk_candidates", type=int, default=5)
    parser.add_argument("--role_queries", type=int, default=4)
    parser.add_argument("--candidate_dropout", type=float, default=0.20)
    parser.add_argument("--max_refinement", type=float, default=0.25)
    parser.add_argument("--orthogonal_weight", type=float, default=0.10)
    parser.add_argument("--teacher_safe_weight", type=float, default=1.0)
    parser.add_argument("--minimum_candidate_epochs", type=int, default=2)
    values, remaining = parser.parse_known_args(argv)
    args = backbone.parse_args(["--shared_stage_epochs", "0", *remaining])
    for name, value in vars(values).items():
        setattr(args, name, value)
    args.shared_stage_epochs = 0
    args.minimum_recovery_epochs = args.minimum_candidate_epochs
    if args.epochs < args.minimum_candidate_epochs:
        raise ValueError("epochs must cover minimum_candidate_epochs")
    return args


backbone.configure_stage = configure_stage
backbone.directional_losses = directional_losses
backbone.metrics_for_output = metrics_for_output
backbone.ARCHITECTURE_VERSION = ARCHITECTURE_VERSION
orchestration.ARCHITECTURE_VERSION = ARCHITECTURE_VERSION
orchestration.build_model = build_model
orchestration.train_epoch = backbone.train_epoch
orchestration.metrics_for_output = metrics_for_output
orchestration.evaluate_model = backbone.evaluate_model
orchestration.validation_rank = backbone.validation_rank


if __name__ == "__main__":
    orchestration.train(parse_args())
