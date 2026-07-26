"""Train GIPSSR-Net end to end by backpropagation."""

from __future__ import annotations

import argparse


from utils import gipssr_stage1_training as stage1
from models.gipssr import ARCHITECTURE_VERSION, GIPSSRNet
from utils import gipssr_training as orchestration
from utils.checkpoint_io import safe_torch_load


STAGE1_ARCHITECTURE_VERSION = stage1.STAGE1_ARCHITECTURE_VERSION
stage1_directional_losses = stage1.directional_losses
stage1_metrics_for_output = stage1.metrics_for_output


def configure_stage(model: GIPSSRNet, shared_only: bool) -> None:
    del shared_only
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.sgssd, model.giprd, model.cuef):
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    if model.refinement_logit is not None:
        model.refinement_logit.requires_grad_(True)


def build_model(args, training_set, device):
    del training_set
    model = GIPSSRNet(
        input_dim=args.embedding_size,
        shared_dim=args.shared_dimensions,
        specific_dim=args.specific_dimensions,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        dropout=args.dropout,
        max_recovery_weight=args.max_recovery_weight,
        min_recovery_weight=args.min_recovery_weight,
        retrieval_dropout=args.retrieval_dropout,
        branch_floor=args.branch_floor,
        topk_candidates=args.topk_candidates,
        role_queries=args.role_queries,
        candidate_dropout=args.candidate_dropout,
        max_refinement=args.max_refinement,
        ablation=args.ablation,
    ).to(device)
    checkpoint = safe_torch_load(args.warm_start_ckpt, device)
    if checkpoint.get("architecture_version") != STAGE1_ARCHITECTURE_VERSION:
        raise ValueError("warm_start_ckpt must be a GIPSSR stage-1 checkpoint")
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    expected_prefixes = (
        "sgssd.",
        "giprd.",
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
    loss_target_encoding = dict(target_encoding)
    if "hierarchical_specific" in target_encoding:
        loss_target_encoding["specific"] = target_encoding["hierarchical_specific"]
    losses = stage1_directional_losses(
        output,
        source_embedding,
        loss_target_encoding,
        target_embedding,
        target_indices,
        args,
    )
    losses["safe"] = losses["safe"].new_zeros(())
    losses["reconstruction"] = (
        losses["reconstruction"]
        + args.orthogonal_weight * output["orthogonality"].mean()
    )
    return losses


def metrics_for_output(model, output, probe_labels):
    result = stage1_metrics_for_output(model, output, probe_labels)
    result.update(
        {
            "orthogonality": orchestration.distribution_summary(
                output["orthogonality"]
            ),
            "candidate_keep_fraction": float(
                output["candidate_keep_fraction"].item()
            ),
            "unconditional_applied_fraction": (
                0.0 if model.ablation == "without_giprd" else 1.0
            ),
            "refinement_scale": orchestration.distribution_summary(
                output["refinement_scale"]
            ),
            "topk_candidates": model.topk_candidates,
            "role_queries": model.role_queries,
        }
    )
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--warm_start_ckpt", required=True)
    parser.add_argument(
        "--ablation",
        choices=("full", "without_igdca", "without_sgssd", "without_giprd", "without_sgssd_giprd", "without_cuef_calibration", "without_cuef_conflict", "without_cuef_uncertainty"),
        default="full",
    )
    parser.add_argument("--topk_candidates", type=int, default=5)
    parser.add_argument("--role_queries", type=int, default=4)
    parser.add_argument("--candidate_dropout", type=float, default=0.20)
    parser.add_argument("--max_refinement", type=float, default=0.25)
    parser.add_argument("--orthogonal_weight", type=float, default=0.10)
    parser.add_argument("--minimum_candidate_epochs", type=int, default=2)
    values, remaining = parser.parse_known_args(argv)
    args = stage1.parse_args(["--shared_stage_epochs", "0", *remaining])
    for name, value in vars(values).items():
        setattr(args, name, value)
    args.shared_stage_epochs = 0
    if args.ablation == "without_sgssd_giprd":
        raise ValueError(
            "without_sgssd_giprd uses the trained GIPSSRStage1 checkpoint directly"
        )
    args.minimum_recovery_epochs = args.minimum_candidate_epochs
    args.safe_weight = 0.0
    if args.epochs < args.minimum_candidate_epochs:
        raise ValueError("epochs must cover minimum_candidate_epochs")
    return args


stage1.configure_stage = configure_stage
stage1.directional_losses = directional_losses
stage1.metrics_for_output = metrics_for_output
stage1.ARCHITECTURE_VERSION = ARCHITECTURE_VERSION
orchestration.ARCHITECTURE_VERSION = ARCHITECTURE_VERSION
orchestration.build_model = build_model
orchestration.train_epoch = stage1.train_epoch
orchestration.metrics_for_output = metrics_for_output
orchestration.evaluate_model = stage1.evaluate_model
orchestration.validation_rank = stage1.validation_rank


if __name__ == "__main__":
    orchestration.train(parse_args())
