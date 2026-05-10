import argparse
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from models.stage1_mobileFacenet import MobileFaceNet
from models.stage2 import Stage2Fusion
from utils.datasets_txt import MissingPairTxtDataset
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
    def __init__(self, patience: int = 100, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = None
        self.counter = 0

    def step(self, value: float) -> bool:
        if self.best_value is None or value < self.best_value - self.min_delta:
            self.best_value = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def infer_num_classes(list_path: str) -> int:
    labels = set()
    with open(list_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 3:
                labels.add(int(parts[2]))
    return len(labels)


def build_loader(list_path: str, img_size: int, batch_size: int, num_workers: int, strong: bool):
    tf = get_transforms(img_size, strong=strong)
    dataset = MissingPairTxtDataset(
        list_path,
        transform_palm=tf,
        transform_vein=tf,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=strong,
        drop_last=strong,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def safe_torch_load(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_encoder_checkpoint(module: nn.Module, checkpoint_path: str, device):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint_path}")
    ckpt = safe_torch_load(checkpoint_path, device)
    state_dict = ckpt.get("encoder", ckpt.get("model", ckpt))
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    print(f"[Info] loaded encoder weights from {checkpoint_path}")
    if missing:
        print(f"[Info] missing encoder keys: {len(missing)}")
    if unexpected:
        print(f"[Info] unexpected encoder keys: {len(unexpected)}")


def set_trainable(module: nn.Module, trainable: bool):
    for param in module.parameters():
        param.requires_grad = trainable


def build_pair_scores(feats: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sim = feats @ feats.T
    i, j = np.triu_indices(labels.shape[0], k=1)
    scores = sim[i, j].astype(np.float32)
    pair_labels = (labels[i] == labels[j]).astype(np.int32)
    return scores, pair_labels


def get_tar_value(tar_ret) -> float:
    if isinstance(tar_ret, dict):
        return float(tar_ret.get("TAR", tar_ret.get("tar", 0.0)))
    return float(tar_ret)


@torch.no_grad()
def validate(palm_encoder, vein_encoder, fusion_model, loader, device) -> Dict[str, float]:
    palm_encoder.eval()
    vein_encoder.eval()
    fusion_model.eval()

    feats, labels_all = [], []
    for palm, vein, labels, _ in tqdm(loader, desc="Validate teacher fusion", dynamic_ncols=True, leave=False):
        palm = palm.to(device, non_blocking=True)
        vein = vein.to(device, non_blocking=True)
        palm_feat = palm_encoder(palm)
        vein_feat = vein_encoder(vein)
        fused_feat = F.normalize(fusion_model(palm_feat, vein_feat), dim=1)
        feats.append(fused_feat.cpu().numpy())
        labels_all.append(labels.numpy())

    feats = np.vstack(feats)
    labels_np = np.concatenate(labels_all, axis=0)
    scores, pair_labels = build_pair_scores(feats, labels_np)
    return {
        "eer": float(compute_eer(scores, pair_labels, is_similarity=True)),
        "tar_1e4": get_tar_value(tar_at_far(scores, pair_labels, 1e-4, is_similarity=True)),
        "tar_1e5": get_tar_value(tar_at_far(scores, pair_labels, 1e-5, is_similarity=True)),
    }


def save_checkpoint(path, epoch, palm_encoder, vein_encoder, fusion_model, classifier, args, num_classes):
    torch.save(
        {
            "epoch": epoch,
            "cnn_palm": palm_encoder.state_dict(),
            "cnn_vein": vein_encoder.state_dict(),
            "fusion": fusion_model.state_dict(),
            "classifier": classifier.state_dict(),
            "args": vars(args),
            "num_classes": num_classes,
        },
        path,
    )


def build_optimizer(palm_encoder, vein_encoder, fusion_model, classifier, args):
    param_groups = [
        {"params": fusion_model.parameters(), "lr": args.lr},
        {"params": classifier.parameters(), "lr": args.lr},
    ]
    if not args.freeze_encoders:
        param_groups.extend(
            [
                {"params": palm_encoder.parameters(), "lr": args.encoder_lr},
                {"params": vein_encoder.parameters(), "lr": args.encoder_lr},
            ]
        )
    return torch.optim.Adam(param_groups, weight_decay=args.wd)


def main():
    parser = argparse.ArgumentParser("Train full-modality teacher fusion baseline")
    parser.add_argument("--train_full_list", type=str, default="data_txt/polyu_train_full.txt")
    parser.add_argument("--val_full_list", type=str, default="data_txt/polyu_val_full.txt")
    parser.add_argument("--palm_ckpt", type=str, default="outputs_dymo/encoders/palm_best.pth")
    parser.add_argument("--vein_ckpt", type=str, default="outputs_dymo/encoders/vein_best.pth")
    parser.add_argument("--save_dir", type=str, default="outputs_dymo/full_fusion")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--encoder_dim", type=int, default=256)
    parser.add_argument("--fusion_dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder_lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--arcface_s", type=float, default=30.0)
    parser.add_argument("--arcface_m", type=float, default=0.20)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--freeze_encoders", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "runs"))
    num_classes = infer_num_classes(args.train_full_list)

    train_loader = build_loader(args.train_full_list, args.input_size, args.batch_size, args.num_workers, strong=True)
    val_loader = build_loader(args.val_full_list, args.input_size, args.batch_size, args.num_workers, strong=False)

    palm_encoder = MobileFaceNet(input_channel=3, input_size=args.input_size, embedding_size=args.encoder_dim).to(device)
    vein_encoder = MobileFaceNet(input_channel=3, input_size=args.input_size, embedding_size=args.encoder_dim).to(device)
    fusion_model = Stage2Fusion(
        in_dim_global=args.encoder_dim,
        out_dim_final=args.fusion_dim,
        final_l2norm=True,
    ).to(device)
    classifier = ArcFace(args.fusion_dim, num_classes, s=args.arcface_s, m=args.arcface_m).to(device)

    load_encoder_checkpoint(palm_encoder, args.palm_ckpt, device)
    load_encoder_checkpoint(vein_encoder, args.vein_ckpt, device)
    set_trainable(palm_encoder, trainable=not args.freeze_encoders)
    set_trainable(vein_encoder, trainable=not args.freeze_encoders)
    if args.freeze_encoders:
        print("[Info] encoders frozen for teacher fusion training.")
    else:
        print(f"[Info] encoders will be fine-tuned with lr={args.encoder_lr}.")

    optimizer = build_optimizer(palm_encoder, vein_encoder, fusion_model, classifier, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()
    early_stop = EarlyStopping(args.patience, args.min_delta)
    best_eer = float("inf")
    best_path = os.path.join(args.save_dir, "stage2_best.pth")

    for epoch in range(1, args.epochs + 1):
        palm_encoder.train(not args.freeze_encoders)
        vein_encoder.train(not args.freeze_encoders)
        fusion_model.train()
        classifier.train()

        running_loss = 0.0
        running_acc = 0.0
        sample_count = 0

        for palm, vein, labels, _ in tqdm(train_loader, desc=f"Teacher fusion epoch {epoch}", dynamic_ncols=True):
            palm = palm.to(device, non_blocking=True)
            vein = vein.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            palm_feat = palm_encoder(palm)
            vein_feat = vein_encoder(vein)
            fused_feat = fusion_model(palm_feat, vein_feat)
            logits = classifier(fused_feat, labels)
            loss = ce(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            if not args.freeze_encoders:
                torch.nn.utils.clip_grad_norm_(palm_encoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(vein_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()

            batch_size = labels.size(0)
            sample_count += batch_size
            running_loss += loss.item() * batch_size
            running_acc += (logits.argmax(dim=1) == labels).float().mean().item() * batch_size

        scheduler.step()
        train_loss = running_loss / max(sample_count, 1)
        train_acc = running_acc / max(sample_count, 1)
        val_metrics = validate(palm_encoder, vein_encoder, fusion_model, val_loader, device)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("val/eer", val_metrics["eer"], epoch)
        writer.add_scalar("val/tar_1e4", val_metrics["tar_1e4"], epoch)
        writer.add_scalar("val/tar_1e5", val_metrics["tar_1e5"], epoch)

        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_eer={val_metrics['eer'] * 100:.3f}% "
            f"val_tar@1e-4={val_metrics['tar_1e4'] * 100:.2f}% "
            f"val_tar@1e-5={val_metrics['tar_1e5'] * 100:.2f}%"
        )

        if val_metrics["eer"] < best_eer:
            best_eer = val_metrics["eer"]
            save_checkpoint(best_path, epoch, palm_encoder, vein_encoder, fusion_model, classifier, args, num_classes)
            print(f"[Info] saved best teacher fusion checkpoint to {best_path}")

        if early_stop.step(val_metrics["eer"]):
            print(f"[Info] early stopping at epoch {epoch}; best_val_eer={best_eer * 100:.3f}%")
            break

    writer.close()


if __name__ == "__main__":
    main()
