import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.backbones import build_encoder
from utils.datasets_txt import SingleModalityFromPairDataset
from utils.head import ArcFace
from utils.metrics import compute_eer, tar_at_far
from utils.preprocess import build_palm_transform, build_vein_transform


def get_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Check the PyTorch CUDA build and NVIDIA driver.")
    device = torch.device(name)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"[Info] using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("[Info] using CPU")
    return device


def infer_num_classes(list_path):
    labels = set()
    with open(list_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 3:
                labels.add(int(parts[2]))
    return len(labels)


def modality_transform(modality, img_size, train=False):
    return build_palm_transform(img_size, train=train) if modality == "palm" else build_vein_transform(img_size, train=train)


def make_loader(args, list_path, train=False):
    dataset = SingleModalityFromPairDataset(
        list_path,
        args.modality,
        modality_transform(args.modality, args.input_size, train=train),
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def make_encoder(args):
    pretrained = args.palm_pretrained if args.modality == "palm" else args.vein_pretrained
    return build_encoder(
        args.modality,
        input_channel=3,
        input_size=args.input_size,
        embedding_size=args.embedding_size,
        pretrained_path=pretrained,
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


def pair_scores(feats, labels):
    sim = feats @ feats.T
    i, j = np.triu_indices(labels.shape[0], k=1)
    return sim[i, j].astype(np.float32), (labels[i] == labels[j]).astype(np.int32)


def metrics_from_features(feats, labels):
    feats = np.asarray(feats, dtype=np.float32)
    labels = np.asarray(labels)
    scores, pair_labels = pair_scores(feats, labels)
    valid = np.isfinite(scores)
    scores, pair_labels = scores[valid], pair_labels[valid]
    if scores.size == 0 or np.unique(pair_labels).size < 2:
        return {"eer": float("inf"), "tar_1e4": 0.0, "tar_1e5": 0.0}
    tar_1e4 = tar_at_far(scores, pair_labels, 1e-4, is_similarity=True)
    tar_1e5 = tar_at_far(scores, pair_labels, 1e-5, is_similarity=True)
    return {
        "eer": float(compute_eer(scores, pair_labels, is_similarity=True)),
        "tar_1e4": float(tar_1e4["TAR"]),
        "tar_1e5": float(tar_1e5["TAR"]),
    }


@torch.no_grad()
def validate(encoder, loader, device, modality):
    encoder.eval()
    feats, labels = [], []
    for images, y in tqdm(loader, desc=f"Validate {modality}", dynamic_ncols=True, leave=False):
        emb = encoder(images.to(device, non_blocking=True))
        feats.append(F.normalize(emb, dim=1).cpu().numpy())
        labels.append(y.numpy())
    return metrics_from_features(np.vstack(feats), np.concatenate(labels))


def metric_improved(current, best, metric, min_delta):
    return current < best - min_delta if metric == "eer" else current > best + min_delta


def initial_best(metric):
    return float("inf") if metric == "eer" else -float("inf")


def save_checkpoint(path, epoch, encoder, classifier, args, num_classes):
    torch.save(
        {
            "epoch": epoch,
            "modality": args.modality,
            "encoder": encoder.state_dict(),
            "classifier": classifier.state_dict(),
            "args": vars(args),
            "num_classes": num_classes,
        },
        path,
    )


def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = get_device(args.device)
    num_classes = infer_num_classes(args.train_full_list)
    train_loader = make_loader(args, args.train_full_list, train=True)
    val_loader = make_loader(args, args.val_full_list)

    encoder = make_encoder(args).to(device)
    head = ArcFace(args.embedding_size, num_classes, args.arcface_s, args.arcface_m).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    ce = nn.CrossEntropyLoss()
    best, bad_epochs = initial_best(args.select_metric), 0
    best_path = os.path.join(args.save_dir, f"{args.modality}_best.pth")

    for epoch in range(1, args.epochs + 1):
        lr = epoch_lr(args, epoch)
        set_lr(optimizer, lr)
        encoder.train()
        head.train()
        loss_sum = correct = total = 0

        for images, labels in tqdm(train_loader, desc=f"{args.modality} epoch {epoch}", dynamic_ncols=True):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = head(encoder(images), labels)
            loss = ce(logits, labels)

            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()

            batch_size = labels.size(0)
            total += batch_size
            loss_sum += loss.item() * batch_size
            correct += (logits.argmax(1) == labels).sum().item()

        val = validate(encoder, val_loader, device, args.modality)
        print(
            f"[Epoch {epoch}] {args.modality} loss={loss_sum / max(total, 1):.4f} "
            f"acc={correct / max(total, 1):.4f} val_eer={val['eer'] * 100:.3f}% "
            f"tar1e4={val['tar_1e4']:.4f} tar1e5={val['tar_1e5']:.4f} lr={lr:.6g}"
        )

        if metric_improved(val[args.select_metric], best, args.select_metric, args.min_delta):
            best, bad_epochs = val[args.select_metric], 0
            save_checkpoint(best_path, epoch, encoder, head, args, num_classes)
            print(f"[Info] saved {best_path} by {args.select_metric}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Train palm/vein baseline encoder")
    parser.add_argument("--modality", choices=["palm", "vein"], default="palm")
    parser.add_argument("--train_full_list", default="data_txt/polyu/train_full.txt")
    parser.add_argument("--val_full_list", default="data_txt/polyu/val_full.txt")
    parser.add_argument("--save_dir", default="outputs/encoders")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--embedding_size", type=int, default=256)
    parser.add_argument("--palm_pretrained", default="pretrained/resnet50_imagenet1k_v2.pth")
    parser.add_argument("--vein_pretrained", default="pretrained/convnextv2_tiny_22k_224_ema.pt")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--arcface_s", type=float, default=32.0)
    parser.add_argument("--arcface_m", type=float, default=0.25)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--select_metric", choices=["eer", "tar_1e4", "tar_1e5"], default="tar_1e4")
    return parser.parse_args(argv)


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
