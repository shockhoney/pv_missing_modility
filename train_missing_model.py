import argparse
import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.missing_model import MissingModalityRecognizer, consistency_loss, recovery_loss
from utils.checkpoint import load_encoder_from_checkpoint
from utils.datasets_txt import MissingPairTxtDataset, infer_num_classes
from utils.evaluation import recognition_rate
from utils.preprocess import build_palm_transform, build_vein_transform


SCENARIOS = ("full", "palm_only", "vein_only")


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


def make_model(args, num_classes, device):
    palm_encoder = load_encoder_from_checkpoint(args.palm_ckpt, "palm", args.input_size, args.embedding_size, device)
    vein_encoder = load_encoder_from_checkpoint(args.vein_ckpt, "vein", args.input_size, args.embedding_size, device)
    return MissingModalityRecognizer(
        palm_encoder,
        vein_encoder,
        num_classes,
        dim=args.embedding_size,
        hidden=args.restore_hidden,
        heads=args.attn_heads,
        reduction=args.channel_reduction,
        arcface_s=args.arcface_s,
        arcface_m=args.arcface_m,
        freeze_encoders=not args.train_encoders,
    ).to(device)


def batch_losses(model, palm, vein, labels, ce, args):
    outputs = {scenario: model(palm, vein, labels=labels, scenario=scenario) for scenario in SCENARIOS}
    cls_loss = sum(ce(outputs[scenario]["logits"], labels) for scenario in SCENARIOS) / len(SCENARIOS)
    rec_loss = 0.5 * (
        recovery_loss(outputs["full"]["hat_vein"], outputs["full"]["f_vein"])
        + recovery_loss(outputs["full"]["hat_palm"], outputs["full"]["f_palm"])
    )
    cons_loss = 0.5 * (
        consistency_loss(outputs["palm_only"]["z"], outputs["full"]["z"])
        + consistency_loss(outputs["vein_only"]["z"], outputs["full"]["z"])
    )
    loss = cls_loss + args.lambda_rec * rec_loss + args.lambda_cons * cons_loss
    return loss, cls_loss, rec_loss, cons_loss, outputs


def train_epoch(model, loader, optimizer, ce, device, args):
    model.train()
    sums = {"loss": 0.0, "cls": 0.0, "rec": 0.0, "cons": 0.0, "acc": 0.0}
    total = 0
    for palm, vein, labels, _ in tqdm(loader, desc="Train missing", dynamic_ncols=True):
        palm = palm.to(device, non_blocking=True)
        vein = vein.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        loss, cls_loss, rec_loss, cons_loss, outputs = batch_losses(model, palm, vein, labels, ce, args)
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
        sums["rec"] += rec_loss.item() * batch_size
        sums["cons"] += cons_loss.item() * batch_size
        sums["acc"] += recognition_rate(outputs["full"]["logits"], labels) * batch_size
    return {key: value / max(total, 1) for key, value in sums.items()}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    correct = {scenario: 0 for scenario in SCENARIOS}
    total = 0
    for palm, vein, labels, _ in tqdm(loader, desc="Validate missing", dynamic_ncols=True, leave=False):
        palm = palm.to(device, non_blocking=True)
        vein = vein.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        for scenario in SCENARIOS:
            logits = model(palm, vein, scenario=scenario)["logits"]
            correct[scenario] += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    metrics = {scenario: correct[scenario] / max(total, 1) for scenario in SCENARIOS}
    metrics["avg"] = sum(metrics.values()) / len(SCENARIOS)
    return metrics


def save_checkpoint(path, epoch, model, args, num_classes, best_acc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "args": vars(args),
            "num_classes": num_classes,
            "best_acc": best_acc,
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
    val_loader = make_loader(args.val_list, args)
    model = make_model(args, num_classes, device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    ce = nn.CrossEntropyLoss()
    best, bad_epochs = -float("inf"), 0

    for epoch in range(1, args.epochs + 1):
        lr = epoch_lr(args, epoch)
        set_lr(optimizer, lr)
        train_stats = train_epoch(model, train_loader, optimizer, ce, device, args)
        val = validate(model, val_loader, device)
        print(
            f"[Epoch {epoch}] loss={train_stats['loss']:.4f} cls={train_stats['cls']:.4f} "
            f"rec={train_stats['rec']:.4f} cons={train_stats['cons']:.4f} "
            f"train_acc={train_stats['acc']:.4f} val_full={val['full']:.4f} "
            f"val_palm_only={val['palm_only']:.4f} val_vein_only={val['vein_only']:.4f} "
            f"val_avg={val['avg']:.4f} lr={lr:.6g}"
        )
        if val["avg"] > best + args.min_delta:
            best, bad_epochs = val["avg"], 0
            save_checkpoint(args.save_path, epoch, model, args, num_classes, best)
            print(f"[Info] saved {args.save_path} by val_avg")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Train missing-modality recognizer")
    parser.add_argument("--train_list", default="data_txt/polyu/closed_train_full.txt")
    parser.add_argument("--val_list", default="data_txt/polyu/closed_val_full.txt")
    parser.add_argument("--palm_ckpt", default="outputs/encoders/palm_best.pth")
    parser.add_argument("--vein_ckpt", default="outputs/encoders/vein_best.pth")
    parser.add_argument("--save_path", default="outputs/missing_model/best.pth")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--embedding_size", type=int, default=256)
    parser.add_argument("--restore_hidden", type=int, default=512)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--channel_reduction", type=int, default=4)
    parser.add_argument("--lambda_rec", type=float, default=1.0)
    parser.add_argument("--lambda_cons", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--arcface_s", type=float, default=32.0)
    parser.add_argument("--arcface_m", type=float, default=0.25)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--train_encoders", action="store_true")
    return parser.parse_args(argv)


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
