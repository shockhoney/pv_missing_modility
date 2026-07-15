import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models.missing_model import (
    ARCHITECTURE_VERSION,
    MissingModalityRecognizer,
    cosine_alignment_loss,
    load_missing_model_state,
)
from utils.checkpoint import load_encoder_teacher_from_checkpoint, save_checkpoint
from utils.checkpoint_io import file_sha256, safe_torch_load
from utils.datasets_txt import MissingPairTxtDataset, _sample_key, infer_num_classes
from utils.evaluation import recognition_rate
from utils.preprocess import build_palm_transform, build_vein_transform
from utils.runtime import build_data_loader, cosine_annealing_lr, resolve_device, set_optimizer_lr, set_random_seed
from utils.scenarios import COMPLETE, PALMPRINT_MISSING, PALMVEIN_MISSING, SSFD_SCENARIOS


SCENARIOS = SSFD_SCENARIOS
ALIGNMENT_STAGE = "alignment"
COMPLETE_FUSION_STAGE = "complete_fusion"
DIFFUSION_STAGE = "diffusion"
RECOVERY_STAGE = "recovery"
FUSION_STAGE = "fusion"
STAGE_ORDER = (ALIGNMENT_STAGE, COMPLETE_FUSION_STAGE, DIFFUSION_STAGE, RECOVERY_STAGE, FUSION_STAGE)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_CONFIG_KEYS = (
    "train_list_fingerprint",
    "palm_ckpt_fingerprint",
    "vein_ckpt_fingerprint",
    "input_size",
    "embedding_size",
    "attn_heads",
    "channel_reduction",
    "diffusion_steps",
    "ddim_steps",
    "diffusion_base_channels",
    "diffusion_time_dim",
    "diffusion_dropout",
    "diffusion_stats_momentum",
    "arcface_s",
    "arcface_m",
    "palm_teacher_s",
    "palm_teacher_m",
    "vein_teacher_s",
    "vein_teacher_m",
)


def project_path(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def set_input_fingerprints(args):
    args.train_list_fingerprint = file_sha256(project_path(args.train_list))
    args.palm_ckpt_fingerprint = file_sha256(project_path(args.palm_ckpt))
    args.vein_ckpt_fingerprint = file_sha256(project_path(args.vein_ckpt))


def validate_encoder_checkpoint(checkpoint, modality, args, num_classes):
    if checkpoint.get("modality") != modality:
        raise ValueError(f"Expected a {modality!r} encoder checkpoint")
    if checkpoint.get("num_classes") != num_classes:
        raise ValueError(f"{modality} encoder class count does not match train_list")

    checkpoint_args = checkpoint.get("args", {})
    for key in ("embedding_size", "input_size"):
        if checkpoint_args.get(key) != getattr(args, key):
            raise ValueError(
                f"{modality} encoder {key} mismatch: "
                f"checkpoint={checkpoint_args.get(key)!r}, current={getattr(args, key)!r}"
            )

    saved_fingerprint = checkpoint_args.get("train_list_fingerprint")
    if saved_fingerprint is not None:
        if saved_fingerprint != args.train_list_fingerprint:
            raise ValueError(f"{modality} encoder was trained with a different label protocol")
        return

    if not args.allow_legacy_encoder_ckpt:
        raise ValueError(
            f"{modality} encoder checkpoint has no training-protocol fingerprint. "
            "Retrain it or pass --allow_legacy_encoder_ckpt only after verifying the label mapping manually."
        )
    saved_train_list = checkpoint_args.get("train_list")
    if saved_train_list is None:
        raise ValueError(f"{modality} encoder checkpoint does not identify its training protocol")
    if os.path.normcase(os.path.abspath(project_path(saved_train_list))) != os.path.normcase(
        os.path.abspath(project_path(args.train_list))
    ):
        raise ValueError(f"{modality} encoder was trained from a different train_list")
    print(
        f"[Warning] accepting legacy {modality} encoder without a protocol fingerprint; "
        "only the train_list path could be verified"
    )


def make_loader(list_path, args, train=False):
    dataset = MissingPairTxtDataset(
        list_path,
        build_palm_transform(args.input_size, train=train),
        build_vein_transform(args.input_size, train=train),
    )
    if train:
        if not dataset.samples:
            raise ValueError("Missing-model train_list is empty")
        labels = sorted({sample["label"] for sample in dataset.samples})
        if labels != list(range(len(labels))):
            raise ValueError("Missing-model train_list labels must be contiguous and zero-based")
        identity_to_label = {}
        label_to_identity = {}
        for sample in dataset.samples:
            if not (sample["palm_exists"] and sample["vein_exists"]):
                raise ValueError("Missing-model training requires a complete paired palm/vein train_list")
            palm_path = sample["palm_path"]
            vein_path = sample["vein_path"]
            if _sample_key(os.path.basename(palm_path)) != _sample_key(os.path.basename(vein_path)):
                raise ValueError(f"Unpaired palm/vein sample keys: {palm_path!r}, {vein_path!r}")
            palm_identity = os.path.basename(os.path.dirname(palm_path))
            vein_identity = os.path.basename(os.path.dirname(vein_path))
            if palm_identity != vein_identity:
                raise ValueError(f"Unpaired palm/vein identities: {palm_path!r}, {vein_path!r}")
            label = sample["label"]
            if identity_to_label.setdefault(palm_identity, label) != label:
                raise ValueError(f"Identity {palm_identity!r} maps to multiple labels")
            if label_to_identity.setdefault(label, palm_identity) != palm_identity:
                raise ValueError(f"Label {label} maps to multiple identities")
    return build_data_loader(dataset, args.batch_size, args.num_workers, train=train)


def make_optimizer(model, lr, weight_decay):
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def make_model(args, num_classes, device):
    palm_encoder, palm_teacher, palm_checkpoint = load_encoder_teacher_from_checkpoint(
        project_path(args.palm_ckpt), "palm", args.embedding_size, device
    )
    vein_encoder, vein_teacher, vein_checkpoint = load_encoder_teacher_from_checkpoint(
        project_path(args.vein_ckpt), "vein", args.embedding_size, device
    )
    validate_encoder_checkpoint(palm_checkpoint, "palm", args, num_classes)
    validate_encoder_checkpoint(vein_checkpoint, "vein", args, num_classes)
    if palm_teacher.out_features != num_classes or vein_teacher.out_features != num_classes:
        raise ValueError("Teacher classifier class count does not match train_list")

    args.palm_teacher_s, args.palm_teacher_m = palm_teacher.s, palm_teacher.m
    args.vein_teacher_s, args.vein_teacher_m = vein_teacher.s, vein_teacher.m
    return MissingModalityRecognizer(
        palm_encoder,
        vein_encoder,
        num_classes,
        dim=args.embedding_size,
        heads=args.attn_heads,
        reduction=args.channel_reduction,
        arcface_s=args.arcface_s,
        arcface_m=args.arcface_m,
        palm_teacher=palm_teacher,
        vein_teacher=vein_teacher,
        diffusion_steps=args.diffusion_steps,
        ddim_steps=args.ddim_steps,
        diffusion_base_channels=args.diffusion_base_channels,
        diffusion_time_dim=args.diffusion_time_dim,
        diffusion_dropout=args.diffusion_dropout,
        diffusion_stats_momentum=args.diffusion_stats_momentum,
    ).to(device)


def distillation_loss(student_logits, teacher_logits):
    return F.kl_div(
        F.log_softmax(student_logits, dim=1),
        F.softmax(teacher_logits.detach(), dim=1),
        reduction="batchmean",
    )


def missing_distillation_loss(output):
    if "teacher_logits_raw" not in output:
        return output["logits"].new_zeros(())
    return distillation_loss(output["fusion_logits_raw"], output["teacher_logits_raw"])


def scenario_outputs(model, encoded, recovery, labels):
    return {
        scenario: model.forward_from_encoded(encoded, recovery, labels=labels, scenario=scenario)
        for scenario in SCENARIOS
    }


def missing_classification_loss(outputs, labels, ce):
    return 0.5 * (
        ce(outputs[PALMPRINT_MISSING]["fusion_logits"], labels)
        + ce(outputs[PALMVEIN_MISSING]["fusion_logits"], labels)
    )


def missing_distillation_losses(outputs):
    return 0.5 * (
        missing_distillation_loss(outputs[PALMPRINT_MISSING])
        + missing_distillation_loss(outputs[PALMVEIN_MISSING])
    )


def recovery_identity_loss(model, recovery, labels, ce):
    losses = []
    if model.palm_teacher is not None:
        losses.append(ce(model.palm_teacher(recovery["generated_palm"], labels), labels))
    if model.vein_teacher is not None:
        losses.append(ce(model.vein_teacher(recovery["generated_vein"], labels), labels))
    return sum(losses) / len(losses) if losses else torch.zeros((), device=labels.device)


def orthogonality_loss(shared, specific):
    shared = F.normalize(shared, dim=1)
    specific = F.normalize(specific, dim=1)
    return (shared * specific).sum(dim=1).abs().mean()


def part_norm_balance_loss(shared, specific):
    shared_norm = shared.norm(dim=1).clamp_min(1e-6)
    specific_norm = specific.norm(dim=1).clamp_min(1e-6)
    return (shared_norm.log() - specific_norm.log()).abs().mean()


def paired_alignment_losses(model, palm, vein, labels, ce, args):
    encoded = model.encode_modalities(palm, vein)
    if model.palm_teacher is None or model.vein_teacher is None:
        raise ValueError("Paired alignment requires both pretrained ArcFace teachers")

    palm_logits = model.palm_teacher(encoded["f_palm"], labels)
    vein_logits = model.vein_teacher(encoded["f_vein"], labels)
    classification = 0.5 * (ce(palm_logits, labels) + ce(vein_logits, labels))
    palm_shared_logits = model.classifier(
        torch.cat([encoded["palm_shared"], torch.zeros_like(encoded["palm_specific"])], dim=1),
        labels,
    )
    vein_shared_logits = model.classifier(
        torch.cat([encoded["vein_shared"], torch.zeros_like(encoded["vein_specific"])], dim=1),
        labels,
    )
    shared_classification = 0.5 * (
        ce(palm_shared_logits, labels) + ce(vein_shared_logits, labels)
    )
    shared = cosine_alignment_loss(
        encoded["palm_shared"],
        encoded["vein_shared"],
        detach_target=False,
    )
    orthogonal = 0.5 * (
        orthogonality_loss(encoded["palm_shared"], encoded["palm_specific"])
        + orthogonality_loss(encoded["vein_shared"], encoded["vein_specific"])
    )
    norm_balance = 0.5 * (
        part_norm_balance_loss(encoded["palm_shared"], encoded["palm_specific"])
        + part_norm_balance_loss(encoded["vein_shared"], encoded["vein_specific"])
    )
    specific_cosine = F.cosine_similarity(
        encoded["palm_specific"],
        encoded["vein_specific"],
        dim=1,
    )
    specific = F.relu(specific_cosine - args.specific_margin).mean()
    loss = (
        classification
        + args.lambda_shared_cls * shared_classification
        + args.lambda_shared * shared
        + args.lambda_orthogonal * orthogonal
        + args.lambda_norm_balance * norm_balance
        + args.lambda_specific * specific
    )
    return {
        "loss": loss,
        "cls": classification,
        "shared_cls": shared_classification,
        "shared": shared,
        "orthogonal": orthogonal,
        "norm_balance": norm_balance,
        "specific": specific,
        "accuracy_logits": {
            "palm": palm_logits,
            "vein": vein_logits,
            "palm_shared": palm_shared_logits,
            "vein_shared": vein_shared_logits,
        },
    }


def complete_fusion_losses(model, palm, vein, labels, ce, args):
    encoded = model.encode_modalities(palm, vein)
    output = model.forward_from_encoded(encoded, {}, labels=labels, scenario=COMPLETE)
    classification = ce(output["fusion_logits"], labels)
    return {
        "loss": classification,
        "complete_cls": classification,
        "accuracy_logits": {COMPLETE: output["fusion_logits"]},
    }


def diffusion_losses(model, palm, vein, labels, ce, args):
    encoded = model.encode_modalities(palm, vein)
    loss = model.diffusion_loss(encoded)
    return {"loss": loss, "diffusion": loss}


def sampled_recovery_losses(model, palm, vein, labels, ce, args):
    encoded = model.encode_modalities(palm, vein)
    recovery = model.recover_modalities(encoded)
    reconstruction = 0.5 * (
        F.smooth_l1_loss(recovery["generated_palm_map"], encoded["palm_map"])
        + F.smooth_l1_loss(recovery["generated_vein_map"], encoded["vein_map"])
    )
    alignment = 0.5 * (
        cosine_alignment_loss(recovery["generated_palm"], encoded["f_palm"])
        + cosine_alignment_loss(recovery["generated_vein"], encoded["f_vein"])
    )
    identity = recovery_identity_loss(model, recovery, labels, ce)
    loss = (
        args.lambda_sample_rec * reconstruction
        + args.lambda_sample_cos * alignment
        + args.lambda_sample_id * identity
    )
    return {
        "loss": loss,
        "sample_rec": reconstruction,
        "sample_cos": alignment,
        "sample_id": identity,
    }


def sampled_fusion_losses(model, palm, vein, labels, ce, args):
    encoded = model.encode_modalities(palm, vein)
    recovery = model.recover_modalities(encoded)
    outputs = scenario_outputs(model, encoded, recovery, labels)
    cls_loss = missing_classification_loss(outputs, labels, ce)
    scenario_alignment = 0.5 * (
        cosine_alignment_loss(outputs[PALMPRINT_MISSING]["z"], outputs[COMPLETE]["z"])
        + cosine_alignment_loss(outputs[PALMVEIN_MISSING]["z"], outputs[COMPLETE]["z"])
    )
    distill = missing_distillation_losses(outputs)
    loss = (
        cls_loss
        + args.lambda_scenario * scenario_alignment
        + args.lambda_distill * distill
    )
    return {
        "loss": loss,
        "cls": cls_loss,
        "scenario": scenario_alignment,
        "distill": distill,
        "outputs": outputs,
        "accuracy_logits": {scenario: outputs[scenario]["fusion_logits"] for scenario in SCENARIOS},
    }


def configure_recovery_training(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in (model.p2v, model.v2p):
        for param in module.parameters():
            param.requires_grad = True


def configure_alignment_training(model):
    for param in model.parameters():
        param.requires_grad = False
    for encoder in (model.palm_encoder, model.vein_encoder):
        for module in (encoder.shared_head, encoder.specific_head):
            for param in module.parameters():
                param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True


def configure_complete_fusion_training(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in (
        model.fusion.cross_attention,
        model.fusion.channel_attention,
        model.fusion.project,
        model.classifier,
    ):
        for param in module.parameters():
            param.requires_grad = True


def configure_fusion_stage(model):
    for param in model.parameters():
        param.requires_grad = False
    model.fusion.available_fusion.reset_residual_scale()
    for param in model.fusion.available_fusion.parameters():
        param.requires_grad = True


def load_model_state(model, checkpoint_path, device, num_classes, expected_stage, args):
    checkpoint = safe_torch_load(checkpoint_path, device)
    if checkpoint.get("num_classes") != num_classes:
        raise ValueError("Missing-model checkpoint class count does not match train_list")
    if checkpoint.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError(
            f"Checkpoint architecture must be {ARCHITECTURE_VERSION!r}; "
            "reuse the encoder checkpoints and retrain the missing-model stages"
        )
    if checkpoint.get("training_stage") != expected_stage:
        raise ValueError(
            f"Expected a {expected_stage!r} checkpoint, got {checkpoint.get('training_stage')!r}: {checkpoint_path}"
        )
    checkpoint_args = checkpoint.get("args", {})
    mismatches = [
        key
        for key in CHECKPOINT_CONFIG_KEYS
        if checkpoint_args.get(key) != getattr(args, key, None)
    ]
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={checkpoint_args.get(key)!r}, current={getattr(args, key, None)!r}"
            for key in mismatches
        )
        raise ValueError(f"Checkpoint configuration mismatch ({details})")
    load_missing_model_state(model, checkpoint["model"])
    return checkpoint


def train_epoch(model, loader, optimizer, ce, device, args, loss_fn, loss_names, description, eval_diffusion):
    model.train()
    if eval_diffusion:
        model.p2v.eval()
        model.v2p.eval()
    sums = {name: 0.0 for name in loss_names}
    total = 0

    for palm, vein, labels, _ in tqdm(loader, desc=description, dynamic_ncols=True):
        palm = palm.to(device, non_blocking=True)
        vein = vein.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        losses = loss_fn(model, palm, vein, labels, ce, args)
        loss = losses["loss"]
        if not torch.isfinite(loss):
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
        optimizer.step()

        batch_size = labels.size(0)
        total += batch_size
        for name in loss_names:
            sums[name] += losses[name].item() * batch_size
        for name, logits in losses.get("accuracy_logits", {}).items():
            key = f"acc_{name}"
            sums[key] = sums.get(key, 0.0) + recognition_rate(logits, labels) * batch_size
    if total == 0:
        raise RuntimeError(f"No finite batches were produced during {description}")
    return {key: value / total for key, value in sums.items()}


def train_stage(model, loader, device, args, num_classes, stage, save_path):
    if stage == ALIGNMENT_STAGE:
        configure_alignment_training(model)
        epochs = args.alignment_epochs
        base_lr = args.alignment_lr
        min_lr = args.alignment_min_lr
        warmup_epochs = args.alignment_warmup_epochs
        loss_fn = paired_alignment_losses
        loss_names = (
            "loss",
            "cls",
            "shared_cls",
            "shared",
            "orthogonal",
            "norm_balance",
            "specific",
        )
    elif stage == COMPLETE_FUSION_STAGE:
        configure_complete_fusion_training(model)
        epochs = args.complete_fusion_epochs
        base_lr = args.complete_fusion_lr
        min_lr = args.complete_fusion_min_lr
        warmup_epochs = args.complete_fusion_warmup_epochs
        loss_fn = complete_fusion_losses
        loss_names = ("loss", "complete_cls")
    elif stage == FUSION_STAGE:
        configure_fusion_stage(model)
        epochs = args.fusion_epochs
        base_lr = args.fusion_lr
        min_lr = args.fusion_min_lr
        warmup_epochs = args.fusion_warmup_epochs
        loss_fn = sampled_fusion_losses
        loss_names = ("loss", "cls", "scenario", "distill")
    elif stage == RECOVERY_STAGE:
        configure_recovery_training(model)
        epochs = args.recovery_epochs
        base_lr = args.recovery_lr
        min_lr = args.recovery_min_lr
        warmup_epochs = args.recovery_warmup_epochs
        loss_fn = sampled_recovery_losses
        loss_names = ("loss", "sample_rec", "sample_cos", "sample_id")
    else:
        configure_recovery_training(model)
        epochs = args.epochs
        base_lr = args.lr
        min_lr = args.min_lr
        warmup_epochs = args.warmup_epochs
        loss_fn = diffusion_losses
        loss_names = ("loss", "diffusion")

    optimizer = make_optimizer(model, base_lr, args.wd)
    ce = nn.CrossEntropyLoss()
    best = float("inf")
    print(f"[Info] start {stage} stage: epochs={epochs}, trainable_params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(1, epochs + 1):
        lr = cosine_annealing_lr(base_lr, min_lr, epochs, warmup_epochs, epoch)
        set_optimizer_lr(optimizer, lr)
        stats = train_epoch(
            model,
            loader,
            optimizer,
            ce,
            device,
            args,
            loss_fn,
            loss_names,
            f"Train {stage}",
            eval_diffusion=stage in (RECOVERY_STAGE, FUSION_STAGE),
        )
        losses = " ".join(f"{name}={stats[name]:.4f}" for name in loss_names)
        accuracies = "".join(
            f" {name}={stats[name]:.4f}" for name in sorted(stats) if name.startswith("acc_")
        )
        print(f"[{stage} Epoch {epoch}] {losses}{accuracies} lr={lr:.6g}")
        if stats["loss"] < best:
            best = stats["loss"]
            save_checkpoint(
                save_path,
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "args": vars(args),
                    "num_classes": num_classes,
                    "best_loss": best,
                    "training_stage": stage,
                    "architecture_version": ARCHITECTURE_VERSION,
                },
            )
            print(f"[Info] saved {save_path} by {stage}_train_loss")


def train(args):
    device = resolve_device(args.device, require_available=True)
    set_random_seed(args.seed)
    set_input_fingerprints(args)
    num_classes = infer_num_classes(args.train_list)
    train_loader = make_loader(args.train_list, args, train=True)
    model = make_model(args, num_classes, device)
    stage_paths = {
        ALIGNMENT_STAGE: args.alignment_ckpt,
        COMPLETE_FUSION_STAGE: args.complete_fusion_ckpt,
        DIFFUSION_STAGE: args.diffusion_ckpt,
        RECOVERY_STAGE: args.recovery_ckpt,
        FUSION_STAGE: args.save_path,
    }
    stages = STAGE_ORDER if args.stage == "all" else (args.stage,)
    for stage in stages:
        stage_index = STAGE_ORDER.index(stage)
        if stage_index > 0:
            previous_stage = STAGE_ORDER[stage_index - 1]
            load_model_state(
                model,
                stage_paths[previous_stage],
                device,
                num_classes,
                expected_stage=previous_stage,
                args=args,
            )
        train_stage(model, train_loader, device, args, num_classes, stage, stage_paths[stage])


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Train missing-modality recognizer")
    parser.add_argument("--train_list", default="data_txt/tongji/ssfd_train_full.txt")
    parser.add_argument("--palm_ckpt", default="outputs/encoders/palm_best.pth")
    parser.add_argument("--vein_ckpt", default="outputs/encoders/vein_best.pth")
    parser.add_argument("--allow_legacy_encoder_ckpt", action="store_true")
    parser.add_argument("--alignment_ckpt", default="outputs/missing_model/alignment_best.pth")
    parser.add_argument("--complete_fusion_ckpt", default="outputs/missing_model/complete_fusion_best.pth")
    parser.add_argument("--diffusion_ckpt", default="outputs/missing_model/diffusion_best.pth")
    parser.add_argument("--recovery_ckpt", default="outputs/missing_model/recovery_best.pth")
    parser.add_argument("--save_path", default="outputs/missing_model/best.pth")
    parser.add_argument("--stage", choices=["all", *STAGE_ORDER], default="all")
    parser.add_argument("--alignment_epochs", type=int, default=30)
    parser.add_argument("--complete_fusion_epochs", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--recovery_epochs", type=int, default=30)
    parser.add_argument("--fusion_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--embedding_size", type=int, default=256)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--channel_reduction", type=int, default=4)
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--ddim_steps", type=int, default=5)
    parser.add_argument("--diffusion_base_channels", type=int, default=64)
    parser.add_argument("--diffusion_time_dim", type=int, default=128)
    parser.add_argument("--diffusion_dropout", type=float, default=0.0)
    parser.add_argument("--diffusion_stats_momentum", type=float, default=0.99)
    parser.add_argument("--lambda_sample_rec", type=float, default=1.0)
    parser.add_argument("--lambda_sample_cos", type=float, default=1.0)
    parser.add_argument("--lambda_sample_id", type=float, default=0.5)
    parser.add_argument("--lambda_shared_cls", type=float, default=1.0)
    parser.add_argument("--lambda_shared", type=float, default=1.0)
    parser.add_argument("--lambda_orthogonal", type=float, default=0.1)
    parser.add_argument("--lambda_norm_balance", type=float, default=0.1)
    parser.add_argument("--lambda_specific", type=float, default=0.1)
    parser.add_argument("--specific_margin", type=float, default=0.0)
    parser.add_argument("--lambda_distill", type=float, default=0.1)
    parser.add_argument("--lambda_scenario", type=float, default=0.5)
    parser.add_argument("--alignment_lr", type=float, default=1e-4)
    parser.add_argument("--alignment_min_lr", type=float, default=1e-6)
    parser.add_argument("--complete_fusion_lr", type=float, default=1e-4)
    parser.add_argument("--complete_fusion_min_lr", type=float, default=1e-6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--recovery_lr", type=float, default=1e-5)
    parser.add_argument("--recovery_min_lr", type=float, default=1e-6)
    parser.add_argument("--fusion_lr", type=float, default=1e-4)
    parser.add_argument("--fusion_min_lr", type=float, default=1e-6)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--arcface_s", type=float, default=32.0)
    parser.add_argument("--arcface_m", type=float, default=0.25)
    parser.add_argument("--alignment_warmup_epochs", type=int, default=2)
    parser.add_argument("--complete_fusion_warmup_epochs", type=int, default=2)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--recovery_warmup_epochs", type=int, default=2)
    parser.add_argument("--fusion_warmup_epochs", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    args = parser.parse_args(argv)
    if not -1.0 <= args.specific_margin <= 1.0:
        parser.error("--specific_margin must be in [-1, 1]")
    return args


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
