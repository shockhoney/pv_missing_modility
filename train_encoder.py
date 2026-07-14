import argparse
import os
import sys

import torch
import torch.nn as nn
from tqdm import tqdm

from models.backbones import build_encoder
from utils.checkpoint import save_checkpoint
from utils.datasets_txt import SingleModalityFromPairDataset, infer_num_classes
from utils.evaluation import count_correct_predictions
from utils.head import ArcFace
from utils.preprocess import build_modality_transform
from utils.runtime import build_data_loader, cosine_annealing_lr, resolve_device, set_optimizer_lr


VEIN_DEFAULTS = {
    "epochs": 100,
    "lr": 1e-3,
    "backbone_lr": 3e-4,
    "wd": 5e-4,
    "arcface_m": 0.15,
    "warmup_epochs": 5,
    "label_smoothing": 0.05,
}


def make_loader(args, list_path, train=False):
    dataset = SingleModalityFromPairDataset(
        list_path,
        args.modality,
        build_modality_transform(args.modality, args.input_size, train=train),
    )
    return build_data_loader(dataset, args.batch_size, args.num_workers, train=train)


def make_encoder(args):
    pretrained = args.palm_pretrained if args.modality == "palm" else args.vein_pretrained
    return build_encoder(
        args.modality,
        input_channel=3,
        embedding_size=args.embedding_size,
        pretrained_path=pretrained,
    )


def make_optimizer(encoder, head, args):
    if args.backbone_lr is None:
        return torch.optim.SGD(
            list(encoder.parameters()) + list(head.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.wd,
        )

    backbone_params = list(encoder.backbone.parameters())
    backbone_ids = {id(param) for param in backbone_params}
    other_params = [param for param in encoder.parameters() if id(param) not in backbone_ids]
    return torch.optim.SGD(
        [
            {"params": backbone_params, "lr": args.backbone_lr, "lr_scale": args.backbone_lr / args.lr},
            {"params": other_params + list(head.parameters()), "lr": args.lr},
        ],
        momentum=0.9,
        weight_decay=args.wd,
    )


def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = resolve_device(args.device, require_available=True, announce=True)
    num_classes = infer_num_classes(args.train_list)
    train_loader = make_loader(args, args.train_list, train=True)

    encoder = make_encoder(args).to(device)
    head = ArcFace(args.embedding_size, num_classes, args.arcface_s, args.arcface_m).to(device)
    optimizer = make_optimizer(encoder, head, args)
    params = [param for group in optimizer.param_groups for param in group["params"]]
    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    best = float("inf")
    best_path = os.path.join(args.save_dir, f"{args.modality}_best.pth")

    for epoch in range(1, args.epochs + 1):
        lr = cosine_annealing_lr(args.lr, args.min_lr, args.epochs, args.warmup_epochs, epoch)
        set_optimizer_lr(optimizer, lr)
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
            correct += count_correct_predictions(logits, labels)

        train_loss = loss_sum / max(total, 1)
        print(
            f"[Epoch {epoch}] {args.modality} loss={train_loss:.4f} "
            f"acc={correct / max(total, 1):.4f} lr={lr:.6g}"
        )

        if train_loss < best:
            best = train_loss
            save_checkpoint(
                best_path,
                {
                    "epoch": epoch,
                    "modality": args.modality,
                    "encoder": encoder.state_dict(),
                    "classifier": head.state_dict(),
                    "args": vars(args),
                    "num_classes": num_classes,
                },
            )
            print(f"[Info] saved {best_path} by train_loss")


def parse_args(argv=None):
    tokens = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser("Train palm/vein baseline encoder")
    parser.add_argument("--modality", choices=["palm", "vein"], default="palm")
    parser.add_argument("--train_list", default="data_txt/tongji/ssfd_train_full.txt")
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
    parser.add_argument("--backbone_lr", type=float, default=None)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--arcface_s", type=float, default=32.0)
    parser.add_argument("--arcface_m", type=float, default=0.25)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    args = parser.parse_args(argv)
    if args.modality == "vein":
        for name, value in VEIN_DEFAULTS.items():
            option = f"--{name}"
            if not any(token == option or token.startswith(f"{option}=") for token in tokens):
                setattr(args, name, value)
    return args


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
