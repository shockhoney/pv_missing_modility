import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models.missing_model import (
    MissingModalityRecognizer,
    cosine_alignment_loss,
)
from utils.checkpoint import load_arcface_from_checkpoint, load_encoder_from_checkpoint, save_checkpoint
from utils.checkpoint_io import safe_torch_load
from utils.datasets_txt import MissingPairTxtDataset, infer_num_classes
from utils.evaluation import recognition_rate
from utils.preprocess import build_palm_transform, build_vein_transform
from utils.runtime import build_data_loader, cosine_annealing_lr, resolve_device, set_optimizer_lr, set_random_seed
from utils.scenarios import COMPLETE, PALMPRINT_MISSING, PALMVEIN_MISSING, SSFD_SCENARIOS


SCENARIOS = SSFD_SCENARIOS
DIFFUSION_STAGE = "diffusion"
RECOVERY_STAGE = "recovery"
FUSION_STAGE = "fusion"


def make_loader(list_path, args, train=False):
    dataset = MissingPairTxtDataset(
        list_path,
        build_palm_transform(args.input_size, train=train),
        build_vein_transform(args.input_size, train=train),
    )
    return build_data_loader(dataset, args.batch_size, args.num_workers, train=train)


def make_optimizer(model, lr, weight_decay):
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def make_model(args, num_classes, device):
    palm_encoder = load_encoder_from_checkpoint(args.palm_ckpt, "palm", args.embedding_size, device)
    vein_encoder = load_encoder_from_checkpoint(args.vein_ckpt, "vein", args.embedding_size, device)
    palm_teacher = load_arcface_from_checkpoint(args.palm_ckpt, args.embedding_size, device)
    vein_teacher = load_arcface_from_checkpoint(args.vein_ckpt, args.embedding_size, device)
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
        gate_init=args.missing_gate_init,
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


def classification_loss(outputs, labels, ce):
    return sum(ce(outputs[scenario]["logits"], labels) for scenario in SCENARIOS) / len(SCENARIOS)


def anchor_loss(model, complete_output, labels, ce):
    return 0.5 * (
        ce(model.classifier(complete_output["f_palm"], labels), labels)
        + ce(model.classifier(complete_output["f_vein"], labels), labels)
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
    cls_loss = classification_loss(outputs, labels, ce)
    anchor = anchor_loss(model, outputs[COMPLETE], labels, ce)
    scenario_alignment = 0.5 * (
        cosine_alignment_loss(outputs[PALMPRINT_MISSING]["z"], outputs[COMPLETE]["z"])
        + cosine_alignment_loss(outputs[PALMVEIN_MISSING]["z"], outputs[COMPLETE]["z"])
    )
    distill = missing_distillation_losses(outputs)
    loss = (
        cls_loss
        + args.lambda_anchor * anchor
        + args.lambda_scenario * scenario_alignment
        + args.lambda_distill * distill
    )
    return {
        "loss": loss,
        "cls": cls_loss,
        "scenario": scenario_alignment,
        "anchor": anchor,
        "distill": distill,
        "outputs": outputs,
    }


def configure_recovery_training(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in (model.p2v, model.v2p):
        for param in module.parameters():
            param.requires_grad = True


def configure_fusion_stage(model):
    for param in model.parameters():
        param.requires_grad = False
    model.fusion.available_fusion.reset_residual_scale()
    for module in (model.fusion.available_fusion, model.classifier):
        for param in module.parameters():
            param.requires_grad = True
    model.palm_missing_gate.requires_grad = True
    model.vein_missing_gate.requires_grad = True


def load_model_state(model, checkpoint_path, device, num_classes):
    checkpoint = safe_torch_load(checkpoint_path, device)
    if checkpoint.get("num_classes") != num_classes:
        raise ValueError("Missing-model checkpoint class count does not match train_list")
    model.load_state_dict(checkpoint["model"])
    return checkpoint


def train_epoch(model, loader, optimizer, ce, device, args, loss_fn, loss_names, description, eval_diffusion):
    model.train()
    if eval_diffusion:
        model.p2v.eval()
        model.v2p.eval()
    sums = {name: 0.0 for name in loss_names}
    sums.update({f"acc_{scenario}": 0.0 for scenario in SCENARIOS})
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
        if "outputs" in losses:
            for scenario in SCENARIOS:
                sums[f"acc_{scenario}"] += recognition_rate(losses["outputs"][scenario]["logits"], labels) * batch_size
    if total == 0:
        raise RuntimeError(f"No finite batches were produced during {description}")
    return {key: value / total for key, value in sums.items()}


def train_stage(model, loader, device, args, num_classes, stage, save_path):
    if stage == FUSION_STAGE:
        configure_fusion_stage(model)
        epochs = args.fusion_epochs
        base_lr = args.fusion_lr
        min_lr = args.fusion_min_lr
        warmup_epochs = args.fusion_warmup_epochs
        loss_fn = sampled_fusion_losses
        loss_names = ("loss", "cls", "scenario", "anchor", "distill")
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
        accuracies = ""
        if stage == FUSION_STAGE:
            accuracies = (
                f" acc_c={stats['acc_complete']:.4f} acc_pm={stats['acc_palmprint_missing']:.4f}"
                f" acc_vm={stats['acc_palmvein_missing']:.4f}"
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
                },
            )
            print(f"[Info] saved {save_path} by {stage}_train_loss")


def train(args):
    device = resolve_device(args.device, require_available=True)
    set_random_seed(args.seed)
    num_classes = infer_num_classes(args.train_list)
    train_loader = make_loader(args.train_list, args, train=True)
    model = make_model(args, num_classes, device)

    if args.stage in ("all", DIFFUSION_STAGE):
        train_stage(model, train_loader, device, args, num_classes, DIFFUSION_STAGE, args.diffusion_ckpt)
        if args.stage == DIFFUSION_STAGE:
            return

    if args.stage in ("all", RECOVERY_STAGE):
        load_model_state(model, args.diffusion_ckpt, device, num_classes)
        train_stage(model, train_loader, device, args, num_classes, RECOVERY_STAGE, args.recovery_ckpt)
        if args.stage == RECOVERY_STAGE:
            return

    load_model_state(model, args.recovery_ckpt, device, num_classes)
    train_stage(model, train_loader, device, args, num_classes, FUSION_STAGE, args.save_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Train missing-modality recognizer")
    parser.add_argument("--train_list", default="data_txt/tongji/ssfd_train_full.txt")
    parser.add_argument("--palm_ckpt", default="outputs/encoders/palm_best.pth")
    parser.add_argument("--vein_ckpt", default="outputs/encoders/vein_best.pth")
    parser.add_argument("--diffusion_ckpt", default="outputs/missing_model/diffusion_best.pth")
    parser.add_argument("--recovery_ckpt", default="outputs/missing_model/recovery_best.pth")
    parser.add_argument("--save_path", default="outputs/missing_model/best.pth")
    parser.add_argument("--stage", choices=["all", DIFFUSION_STAGE, RECOVERY_STAGE, FUSION_STAGE], default="all")
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
    parser.add_argument("--lambda_anchor", type=float, default=1.0)
    parser.add_argument("--lambda_distill", type=float, default=0.1)
    parser.add_argument("--lambda_scenario", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--recovery_lr", type=float, default=1e-5)
    parser.add_argument("--recovery_min_lr", type=float, default=1e-6)
    parser.add_argument("--fusion_lr", type=float, default=1e-4)
    parser.add_argument("--fusion_min_lr", type=float, default=1e-6)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--arcface_s", type=float, default=32.0)
    parser.add_argument("--arcface_m", type=float, default=0.25)
    parser.add_argument("--missing_gate_init", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--recovery_warmup_epochs", type=int, default=2)
    parser.add_argument("--fusion_warmup_epochs", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    return parser.parse_args(argv)


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
