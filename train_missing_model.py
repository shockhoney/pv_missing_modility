import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.missing_model import (
    MissingModalityRecognizer,
    consistency_loss,
    shared_alignment_loss,
    transformation_loss,
)
from utils.checkpoint import load_arcface_from_checkpoint, load_encoder_from_checkpoint
from utils.datasets_txt import MissingPairTxtDataset, infer_num_classes
from utils.evaluation import recognition_rate
from utils.preprocess import build_palm_transform, build_vein_transform


SCENARIOS = ("complete", "palmprint_missing", "palmvein_missing")


def make_loader(list_path, args, train=False):
    dataset = MissingPairTxtDataset(
        list_path,
        build_palm_transform(args.input_size, train=train),
        build_vein_transform(args.input_size, train=train),
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def epoch_lr(args, epoch):
    if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
        return args.lr * epoch / args.warmup_epochs
    span = max(1, args.epochs - args.warmup_epochs)
    step = max(0, epoch - args.warmup_epochs)
    scale = 0.5 * (1.0 + math.cos(math.pi * min(step, span) / span))
    return args.min_lr + (args.lr - args.min_lr) * scale


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def make_optimizer(model, args):
    params = [param for param in model.parameters() if param.requires_grad]
    return torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd)


def make_model(args, num_classes, device):
    palm_encoder = load_encoder_from_checkpoint(args.palm_ckpt, "palm", args.input_size, args.embedding_size, device)
    vein_encoder = load_encoder_from_checkpoint(args.vein_ckpt, "vein", args.input_size, args.embedding_size, device)
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
        cmft_hidden=args.cmft_hidden,
        heads=args.attn_heads,
        reduction=args.channel_reduction,
        arcface_s=args.arcface_s,
        arcface_m=args.arcface_m,
        palm_teacher=palm_teacher,
        vein_teacher=vein_teacher,
        gate_init=args.missing_gate_init,
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


def batch_losses(model, palm, vein, labels, ce, args):
    outputs = {scenario: model(palm, vein, labels=labels, scenario=scenario) for scenario in SCENARIOS}
    cls_loss = sum(ce(outputs[scenario]["logits"], labels) for scenario in SCENARIOS) / len(SCENARIOS)
    trans_loss = 0.5 * (
        transformation_loss(outputs["complete"]["hat_vein_specific"], outputs["complete"]["vein_specific"])
        + transformation_loss(outputs["complete"]["hat_palm_specific"], outputs["complete"]["palm_specific"])
    )
    shared_loss = shared_alignment_loss(outputs["complete"]["palm_shared"], outputs["complete"]["vein_shared"])
    anchor_loss = 0.5 * (
        ce(model.classifier(outputs["complete"]["f_palm"], labels), labels)
        + ce(model.classifier(outputs["complete"]["f_vein"], labels), labels)
    )
    avail_loss = 0.5 * (
        consistency_loss(outputs["palmprint_missing"]["z"], outputs["complete"]["f_vein"])
        + consistency_loss(outputs["palmvein_missing"]["z"], outputs["complete"]["f_palm"])
    )
    distill_loss = 0.5 * (
        missing_distillation_loss(outputs["palmprint_missing"])
        + missing_distillation_loss(outputs["palmvein_missing"])
    )
    loss = (
        cls_loss
        + args.lambda_anchor * anchor_loss
        + args.lambda_shared * shared_loss
        + args.lambda_trans * trans_loss
        + args.lambda_avail * avail_loss
        + args.lambda_distill * distill_loss
    )
    return loss, cls_loss, shared_loss, trans_loss, anchor_loss, avail_loss, distill_loss, outputs


def train_epoch(model, loader, optimizer, ce, device, args):
    model.train()
    sums = {"loss": 0.0, "cls": 0.0, "shared": 0.0, "trans": 0.0, "anchor": 0.0, "avail": 0.0, "distill": 0.0}
    sums.update({f"acc_{scenario}": 0.0 for scenario in SCENARIOS})
    total = 0

    for palm, vein, labels, _ in tqdm(loader, desc="Train missing", dynamic_ncols=True):
        palm = palm.to(device, non_blocking=True)
        vein = vein.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        loss, cls_loss, shared_loss, trans_loss, anchor_loss, avail_loss, distill_loss, outputs = batch_losses(
            model, palm, vein, labels, ce, args
        )
        if not torch.isfinite(loss):
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
        optimizer.step()

        batch_size = labels.size(0)
        total += batch_size
        sums["loss"] += loss.item() * batch_size
        sums["cls"] += cls_loss.item() * batch_size
        sums["shared"] += shared_loss.item() * batch_size
        sums["trans"] += trans_loss.item() * batch_size
        sums["anchor"] += anchor_loss.item() * batch_size
        sums["avail"] += avail_loss.item() * batch_size
        sums["distill"] += distill_loss.item() * batch_size
        for scenario in SCENARIOS:
            sums[f"acc_{scenario}"] += recognition_rate(outputs[scenario]["logits"], labels) * batch_size
    return {key: value / max(total, 1) for key, value in sums.items()}


def save_checkpoint(path, epoch, model, args, num_classes, best_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "args": vars(args),
            "num_classes": num_classes,
            "best_loss": best_loss,
        },
        path,
    )


def get_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Check the PyTorch CUDA build and NVIDIA driver.")
    return torch.device(name)


def train(args):
    device = get_device(args.device)
    num_classes = infer_num_classes(args.train_list)
    train_loader = make_loader(args.train_list, args, train=True)
    model = make_model(args, num_classes, device)
    optimizer = make_optimizer(model, args)
    ce = nn.CrossEntropyLoss()
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        lr = epoch_lr(args, epoch)
        set_lr(optimizer, lr)
        stats = train_epoch(model, train_loader, optimizer, ce, device, args)
        print(
            f"[Epoch {epoch}] loss={stats['loss']:.4f} cls={stats['cls']:.4f} "
            f"shared={stats['shared']:.4f} trans={stats['trans']:.4f} "
            f"anchor={stats['anchor']:.4f} avail={stats['avail']:.4f} distill={stats['distill']:.4f} "
            f"acc_c={stats['acc_complete']:.4f} acc_pm={stats['acc_palmprint_missing']:.4f} "
            f"acc_vm={stats['acc_palmvein_missing']:.4f} lr={lr:.6g}"
        )
        if stats["loss"] < best:
            best = stats["loss"]
            save_checkpoint(args.save_path, epoch, model, args, num_classes, best)
            print(f"[Info] saved {args.save_path} by train_loss")


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Train missing-modality recognizer")
    parser.add_argument("--train_list", default="data_txt/cumt/ssfd_train_full.txt")
    parser.add_argument("--palm_ckpt", default="outputs/encoders/palm_best.pth")
    parser.add_argument("--vein_ckpt", default="outputs/encoders/vein_best.pth")
    parser.add_argument("--save_path", default="outputs/missing_model/best.pth")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--embedding_size", type=int, default=256)
    parser.add_argument("--cmft_hidden", type=int, default=1024)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--channel_reduction", type=int, default=4)
    parser.add_argument("--lambda_shared", type=float, default=0.05)
    parser.add_argument("--lambda_trans", type=float, default=0.1)
    parser.add_argument("--lambda_anchor", type=float, default=1.0)
    parser.add_argument("--lambda_avail", type=float, default=1.0)
    parser.add_argument("--lambda_distill", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--arcface_s", type=float, default=32.0)
    parser.add_argument("--arcface_m", type=float, default=0.25)
    parser.add_argument("--missing_gate_init", type=float, default=-8.0)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    return parser.parse_args(argv)


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
