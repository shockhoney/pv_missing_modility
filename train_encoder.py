import argparse
import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.backbones import build_encoder
from utils.datasets_txt import SingleModalityFromPairDataset, infer_num_classes
from utils.head import ArcFace
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
    num_classes = infer_num_classes(args.train_list)
    train_loader = make_loader(args, args.train_list, train=True)

    encoder = make_encoder(args).to(device)
    head = ArcFace(args.embedding_size, num_classes, args.arcface_s, args.arcface_m).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    ce = nn.CrossEntropyLoss()
    best = float("inf")
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

        train_loss = loss_sum / max(total, 1)
        print(
            f"[Epoch {epoch}] {args.modality} loss={train_loss:.4f} "
            f"acc={correct / max(total, 1):.4f} lr={lr:.6g}"
        )

        if train_loss < best:
            best = train_loss
            save_checkpoint(best_path, epoch, encoder, head, args, num_classes)
            print(f"[Info] saved {best_path} by train_loss")


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Train palm/vein baseline encoder")
    parser.add_argument("--modality", choices=["palm", "vein"], default="palm")
    parser.add_argument("--train_list", default="data_txt/cumt/ssfd_train_full.txt")
    parser.add_argument("--save_dir", default="outputs/encoders")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--embedding_size", type=int, default=256)
    parser.add_argument("--palm_pretrained", default="pretrained/resnet18_imagenet1k_v1.pth")
    parser.add_argument("--vein_pretrained", default="pretrained/resnet18_imagenet1k_v1.pth")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--arcface_s", type=float, default=32.0)
    parser.add_argument("--arcface_m", type=float, default=0.25)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    return parser.parse_args(argv)


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
