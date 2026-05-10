import argparse
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from models.stage1_mobileFacenet import MobileFaceNet
from utils.datasets_txt import SingleModalityFromPairDataset
from utils.head import ArcFace
from utils.metrics import compute_eer, tar_at_far


def get_transforms(img_size: int, strong: bool):
    ops = [transforms.Resize((img_size, img_size))]
    if strong:
        ops.extend(
            [
                transforms.RandomRotation(10),
                transforms.RandomAffine(0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )
    ops.extend(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    return transforms.Compose(ops)


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def infer_num_classes(list_path: str) -> int:
    labels = set()
    with open(list_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 3:
                labels.add(int(parts[2]))
    return len(labels)


def build_loader(
    list_path: str,
    modality: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    strong: bool,
):
    dataset = SingleModalityFromPairDataset(
        list_path,
        modality=modality,
        transform=get_transforms(img_size, strong=strong),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=strong,
        drop_last=strong,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_pair_scores(feats: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sim = feats @ feats.T
    i, j = np.triu_indices(labels.shape[0], k=1)
    scores = sim[i, j].astype(np.float32)
    pair_labels = (labels[i] == labels[j]).astype(np.int32)
    return scores, pair_labels


def tar_value(tar_ret) -> float:
    if isinstance(tar_ret, dict):
        return float(tar_ret["TAR"])
    return float(tar_ret)


@torch.no_grad()
def validate(encoder: nn.Module, loader: DataLoader, device):
    encoder.eval()
    feats, labels = [], []

    for images, batch_labels in tqdm(loader, desc="Validate encoder", dynamic_ncols=True, leave=False):
        images = images.to(device, non_blocking=True)
        embeddings = F.normalize(encoder(images), dim=1)
        feats.append(embeddings.cpu().numpy())
        labels.append(batch_labels.numpy())

    feats = np.vstack(feats)
    labels = np.concatenate(labels, axis=0)
    scores, pair_labels = build_pair_scores(feats, labels)
    return {
        "eer": float(compute_eer(scores, pair_labels, is_similarity=True)),
        "tar_1e4": tar_value(tar_at_far(scores, pair_labels, 1e-4, is_similarity=True)),
        "tar_1e5": tar_value(tar_at_far(scores, pair_labels, 1e-5, is_similarity=True)),
    }


def save_checkpoint(path: str, epoch: int, encoder, classifier, args, num_classes: int):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = infer_num_classes(args.train_full_list)

    train_loader = build_loader(
        args.train_full_list,
        args.modality,
        args.input_size,
        args.batch_size,
        args.num_workers,
        strong=True,
    )
    val_loader = build_loader(
        args.val_full_list,
        args.modality,
        args.input_size,
        args.batch_size,
        args.num_workers,
        strong=False,
    )

    encoder = MobileFaceNet(input_channel=3, input_size=args.input_size, embedding_size=args.embedding_size).to(device)
    classifier = ArcFace(args.embedding_size, num_classes, s=args.arcface_s, m=args.arcface_m).to(device)
    optimizer = torch.optim.SGD(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    early_stop = EarlyStopping(args.patience, args.min_delta)
    ce = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, f"{args.modality}_runs"))

    best_eer = float("inf")
    best_path = os.path.join(args.save_dir, f"{args.modality}_best.pth")

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        classifier.train()
        loss_sum = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"{args.modality} encoder epoch {epoch}", dynamic_ncols=True):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            feats = encoder(images)
            logits = classifier(feats, labels)
            loss = ce(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(classifier.parameters()), max_norm=5.0)
            optimizer.step()

            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

        scheduler.step()
        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)
        val_metrics = validate(encoder, val_loader, device)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("val/eer", val_metrics["eer"], epoch)
        writer.add_scalar("val/tar_1e4", val_metrics["tar_1e4"], epoch)
        writer.add_scalar("val/tar_1e5", val_metrics["tar_1e5"], epoch)

        print(
            f"[Epoch {epoch}] modality={args.modality} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_eer={val_metrics['eer'] * 100:.3f}% "
            f"val_tar@1e-4={val_metrics['tar_1e4'] * 100:.2f}% "
            f"val_tar@1e-5={val_metrics['tar_1e5'] * 100:.2f}%"
        )

        if val_metrics["eer"] < best_eer:
            best_eer = val_metrics["eer"]
            save_checkpoint(best_path, epoch, encoder, classifier, args, num_classes)
            print(f"[Info] saved best {args.modality} encoder checkpoint to {best_path}")

        if early_stop.step(val_metrics["eer"]):
            print(f"[Info] early stopping at epoch {epoch}; best_val_eer={best_eer * 100:.3f}%")
            break

    writer.close()


def main():
    parser = argparse.ArgumentParser("Train a single MobileFaceNet encoder")
    parser.add_argument("--modality", type=str, choices=["palm", "vein"], required=True)
    parser.add_argument("--train_full_list", type=str, default="data_txt/polyu_train_full.txt")
    parser.add_argument("--val_full_list", type=str, default="data_txt/polyu_val_full.txt")
    parser.add_argument("--save_dir", type=str, default="outputs_dymo/encoders")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--embedding_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--arcface_s", type=float, default=32.0)
    parser.add_argument("--arcface_m", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--min_delta", type=float, default=0.001)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
