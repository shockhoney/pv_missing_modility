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
from utils.datasets_txt import MissingPairTxtDataset, SingleModalityFromPairDataset
from utils.metrics import compute_eer, tar_at_far


def get_transforms(img_size: int, modality: str, strong: bool):
    ops = [transforms.Resize((img_size, img_size))]
    if strong:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
        if modality == "palm":
            ops.extend(
                [
                    transforms.RandomRotation(8),
                    transforms.RandomAffine(0, translate=(0.08, 0.08)),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15),
                ]
            )
        else:
            ops.extend(
                [
                    transforms.RandomRotation(4),
                    transforms.RandomAffine(0, translate=(0.04, 0.04)),
                    transforms.ColorJitter(brightness=0.08, contrast=0.08),
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


def infer_num_classes(list_path: str) -> int:
    labels = set()
    with open(list_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 3:
                labels.add(int(parts[2]))
    return len(labels)


def build_loader(list_path: str, modality: str, img_size: int, batch_size: int, num_workers: int, strong: bool):
    dataset = SingleModalityFromPairDataset(
        list_path,
        modality=modality,
        transform=get_transforms(img_size, modality=modality, strong=strong),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=strong,
        drop_last=strong,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_joint_loader(list_path: str, img_size: int, batch_size: int, num_workers: int, strong: bool):
    dataset = MissingPairTxtDataset(
        list_path,
        transform_palm=get_transforms(img_size, modality="palm", strong=strong),
        transform_vein=get_transforms(img_size, modality="vein", strong=strong),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=strong,
        drop_last=strong,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_pair_scores(feats: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sim = feats @ feats.T
    i, j = np.triu_indices(labels.shape[0], k=1)
    scores = sim[i, j].astype(np.float32)
    pair_labels = (labels[i] == labels[j]).astype(np.int32)
    return scores, pair_labels


def build_cross_pair_scores(query_feats: np.ndarray, gallery_feats: np.ndarray, query_labels: np.ndarray, gallery_labels: np.ndarray):
    sim = query_feats @ gallery_feats.T
    scores = sim.reshape(-1).astype(np.float32)
    pair_labels = (query_labels[:, None] == gallery_labels[None, :]).reshape(-1).astype(np.int32)
    return scores, pair_labels


def get_tar_value(tar_ret):
    if isinstance(tar_ret, dict):
        return float(tar_ret.get("TAR", tar_ret.get("tar", 0.0)))
    return float(tar_ret)


def safe_verification_metrics(feats: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    scores, pair_labels = build_pair_scores(feats, labels)
    if (pair_labels == 1).sum() == 0 or (pair_labels == 0).sum() == 0:
        return {"eer": float("nan"), "tar_1e4": float("nan"), "tar_1e5": float("nan")}
    return {
        "eer": float(compute_eer(scores, pair_labels, is_similarity=True)),
        "tar_1e4": get_tar_value(tar_at_far(scores, pair_labels, 1e-4, is_similarity=True)),
        "tar_1e5": get_tar_value(tar_at_far(scores, pair_labels, 1e-5, is_similarity=True)),
    }


def safe_cross_metrics(query_feats: np.ndarray, gallery_feats: np.ndarray, query_labels: np.ndarray, gallery_labels: np.ndarray):
    scores, pair_labels = build_cross_pair_scores(query_feats, gallery_feats, query_labels, gallery_labels)
    if (pair_labels == 1).sum() == 0 or (pair_labels == 0).sum() == 0:
        return {"eer": float("nan"), "tar_1e4": float("nan"), "tar_1e5": float("nan")}
    return {
        "eer": float(compute_eer(scores, pair_labels, is_similarity=True)),
        "tar_1e4": get_tar_value(tar_at_far(scores, pair_labels, 1e-4, is_similarity=True)),
        "tar_1e5": get_tar_value(tar_at_far(scores, pair_labels, 1e-5, is_similarity=True)),
    }


def cross_modal_supcon_loss(palm_feat: torch.Tensor, vein_feat: torch.Tensor, labels: torch.Tensor, temperature: float):
    labels = labels.view(-1, 1)
    positive = torch.eq(labels, labels.t()).float()
    positive_count = positive.sum(dim=1).clamp_min(1.0)

    logits_pv = palm_feat @ vein_feat.t() / temperature
    logits_vp = vein_feat @ palm_feat.t() / temperature
    log_prob_pv = logits_pv - torch.logsumexp(logits_pv, dim=1, keepdim=True)
    log_prob_vp = logits_vp - torch.logsumexp(logits_vp, dim=1, keepdim=True)

    loss_pv = -((positive * log_prob_pv).sum(dim=1) / positive_count).mean()
    loss_vp = -((positive * log_prob_vp).sum(dim=1) / positive_count).mean()
    return 0.5 * (loss_pv + loss_vp)


def alignment_loss(palm_feat: torch.Tensor, vein_feat: torch.Tensor):
    return 1.0 - F.cosine_similarity(palm_feat, vein_feat, dim=1).mean()


@torch.no_grad()
def validate(model, classifier, loader, device):
    model.eval()
    classifier.eval()
    ce = nn.CrossEntropyLoss()

    loss_meter = 0.0
    acc_meter = 0.0
    sample_count = 0
    all_feats = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Validate encoder", dynamic_ncols=True, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        feats = model(images)
        logits = classifier(F.normalize(feats, dim=1))
        loss = ce(logits, labels)

        batch_size = labels.size(0)
        sample_count += batch_size
        loss_meter += loss.item() * batch_size
        acc_meter += (logits.argmax(dim=1) == labels).float().mean().item() * batch_size

        all_feats.append(F.normalize(feats, dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    feats = np.vstack(all_feats)
    labels = np.concatenate(all_labels, axis=0)
    scores, pair_labels = build_pair_scores(feats, labels)
    eer = compute_eer(scores, pair_labels, is_similarity=True)
    tar_1e4 = get_tar_value(tar_at_far(scores, pair_labels, 1e-4, is_similarity=True))
    tar_1e5 = get_tar_value(tar_at_far(scores, pair_labels, 1e-5, is_similarity=True))

    return {
        "loss": loss_meter / max(sample_count, 1),
        "acc": acc_meter / max(sample_count, 1),
        "eer": float(eer),
        "tar_1e4": tar_1e4,
        "tar_1e5": tar_1e5,
    }


@torch.no_grad()
def validate_joint(palm_encoder, vein_encoder, classifier, loader, device):
    palm_encoder.eval()
    vein_encoder.eval()
    classifier.eval()
    ce = nn.CrossEntropyLoss()

    loss_meter = 0.0
    palm_acc_meter = 0.0
    vein_acc_meter = 0.0
    sample_count = 0
    palm_feats = []
    vein_feats = []
    fused_feats = []
    all_labels = []

    for palm, vein, labels, _ in tqdm(loader, desc="Validate joint encoders", dynamic_ncols=True, leave=False):
        palm = palm.to(device, non_blocking=True)
        vein = vein.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        palm_feat = F.normalize(palm_encoder(palm), dim=1)
        vein_feat = F.normalize(vein_encoder(vein), dim=1)
        fused_feat = F.normalize(palm_feat + vein_feat, dim=1)
        palm_logits = classifier(palm_feat)
        vein_logits = classifier(vein_feat)
        loss = 0.5 * (ce(palm_logits, labels) + ce(vein_logits, labels))

        batch_size = labels.size(0)
        sample_count += batch_size
        loss_meter += loss.item() * batch_size
        palm_acc_meter += (palm_logits.argmax(dim=1) == labels).float().mean().item() * batch_size
        vein_acc_meter += (vein_logits.argmax(dim=1) == labels).float().mean().item() * batch_size

        palm_feats.append(palm_feat.cpu().numpy())
        vein_feats.append(vein_feat.cpu().numpy())
        fused_feats.append(fused_feat.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    palm_feats = np.vstack(palm_feats)
    vein_feats = np.vstack(vein_feats)
    fused_feats = np.vstack(fused_feats)
    labels_np = np.concatenate(all_labels, axis=0)
    palm_metrics = safe_verification_metrics(palm_feats, labels_np)
    vein_metrics = safe_verification_metrics(vein_feats, labels_np)
    fused_metrics = safe_verification_metrics(fused_feats, labels_np)
    cross_metrics = safe_cross_metrics(palm_feats, vein_feats, labels_np, labels_np)

    return {
        "loss": loss_meter / max(sample_count, 1),
        "palm_acc": palm_acc_meter / max(sample_count, 1),
        "vein_acc": vein_acc_meter / max(sample_count, 1),
        "palm": palm_metrics,
        "vein": vein_metrics,
        "fused": fused_metrics,
        "cross": cross_metrics,
    }


def save_single_checkpoint(path, epoch, modality, encoder, classifier, args, num_classes):
    torch.save(
        {
            "epoch": epoch,
            "modality": modality,
            "encoder": encoder.state_dict(),
            "classifier": classifier.state_dict(),
            "args": vars(args),
            "num_classes": num_classes,
        },
        path,
    )


def save_joint_checkpoints(save_dir, epoch, palm_encoder, vein_encoder, classifier, args, num_classes):
    joint_payload = {
        "epoch": epoch,
        "modality": "joint",
        "palm_encoder": palm_encoder.state_dict(),
        "vein_encoder": vein_encoder.state_dict(),
        "classifier": classifier.state_dict(),
        "args": vars(args),
        "num_classes": num_classes,
    }
    torch.save(joint_payload, os.path.join(save_dir, "joint_best.pth"))

    for modality, encoder in [("palm", palm_encoder), ("vein", vein_encoder)]:
        torch.save(
            {
                "epoch": epoch,
                "modality": modality,
                "joint_training": True,
                "encoder": encoder.state_dict(),
                "classifier": classifier.state_dict(),
                "args": vars(args),
                "num_classes": num_classes,
            },
            os.path.join(save_dir, f"{modality}_best.pth"),
        )


def train_single(args, device, writer, num_classes):
    train_loader = build_loader(
        args.train_full_list,
        modality=args.modality,
        img_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        strong=True,
    )
    val_loader = build_loader(
        args.val_full_list,
        modality=args.modality,
        img_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        strong=False,
    )

    encoder = MobileFaceNet(input_channel=3, input_size=args.input_size, embedding_size=args.embedding_size).to(device)
    classifier = nn.Linear(args.embedding_size, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        weight_decay=args.wd,
    )
    ce = nn.CrossEntropyLoss()

    best_score = float("-inf")
    best_path = os.path.join(args.save_dir, f"{args.modality}_best.pth")

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        classifier.train()
        running_loss = 0.0
        running_acc = 0.0
        sample_count = 0

        for images, labels in tqdm(train_loader, desc=f"{args.modality} encoder epoch {epoch}", dynamic_ncols=True):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            feats = F.normalize(encoder(images), dim=1)
            logits = classifier(feats)
            loss = ce(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            sample_count += batch_size
            running_loss += loss.item() * batch_size
            running_acc += (logits.argmax(dim=1) == labels).float().mean().item() * batch_size

        train_loss = running_loss / max(sample_count, 1)
        train_acc = running_acc / max(sample_count, 1)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)

        val_metrics = validate(encoder, classifier, val_loader, device)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/acc", val_metrics["acc"], epoch)
        writer.add_scalar("val/eer", val_metrics["eer"], epoch)
        writer.add_scalar("val/tar_1e4", val_metrics["tar_1e4"], epoch)
        writer.add_scalar("val/tar_1e5", val_metrics["tar_1e5"], epoch)

        print(
            f"[Epoch {epoch}] modality={args.modality} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_acc={val_metrics['acc']:.4f} val_eer={val_metrics['eer'] * 100:.3f}% "
            f"val_tar@1e-4={val_metrics['tar_1e4'] * 100:.2f}% "
            f"val_tar@1e-5={val_metrics['tar_1e5'] * 100:.2f}%"
        )

        score = -val_metrics["eer"] + 0.2 * val_metrics["tar_1e4"]
        if score > best_score:
            best_score = score
            save_single_checkpoint(best_path, epoch, args.modality, encoder, classifier, args, num_classes)
            print(f"[Info] saved best {args.modality} encoder checkpoint to {best_path}")


def train_joint(args, device, writer, num_classes):
    train_loader = build_joint_loader(
        args.train_full_list,
        img_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        strong=True,
    )
    val_loader = build_joint_loader(
        args.val_full_list,
        img_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        strong=False,
    )

    palm_encoder = MobileFaceNet(input_channel=3, input_size=args.input_size, embedding_size=args.embedding_size).to(device)
    vein_encoder = MobileFaceNet(input_channel=3, input_size=args.input_size, embedding_size=args.embedding_size).to(device)
    classifier = nn.Linear(args.embedding_size, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        list(palm_encoder.parameters()) + list(vein_encoder.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        weight_decay=args.wd,
    )
    ce = nn.CrossEntropyLoss()

    best_score = float("-inf")

    for epoch in range(1, args.epochs + 1):
        palm_encoder.train()
        vein_encoder.train()
        classifier.train()
        running = {
            "loss": 0.0,
            "cls": 0.0,
            "cm_supcon": 0.0,
            "align": 0.0,
            "palm_acc": 0.0,
            "vein_acc": 0.0,
        }
        sample_count = 0

        for palm, vein, labels, _ in tqdm(train_loader, desc=f"joint encoder epoch {epoch}", dynamic_ncols=True):
            palm = palm.to(device, non_blocking=True)
            vein = vein.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            palm_feat = F.normalize(palm_encoder(palm), dim=1)
            vein_feat = F.normalize(vein_encoder(vein), dim=1)
            palm_logits = classifier(palm_feat)
            vein_logits = classifier(vein_feat)

            loss_cls = 0.5 * (ce(palm_logits, labels) + ce(vein_logits, labels))
            loss_cm = cross_modal_supcon_loss(palm_feat, vein_feat, labels, args.supcon_temperature)
            loss_align = alignment_loss(palm_feat, vein_feat)
            loss = loss_cls + args.lambda_cm * loss_cm + args.lambda_align * loss_align

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            sample_count += batch_size
            running["loss"] += loss.item() * batch_size
            running["cls"] += loss_cls.item() * batch_size
            running["cm_supcon"] += loss_cm.item() * batch_size
            running["align"] += loss_align.item() * batch_size
            running["palm_acc"] += (palm_logits.argmax(dim=1) == labels).float().mean().item() * batch_size
            running["vein_acc"] += (vein_logits.argmax(dim=1) == labels).float().mean().item() * batch_size

        for key, value in running.items():
            writer.add_scalar(f"train/{key}", value / max(sample_count, 1), epoch)

        val_metrics = validate_joint(palm_encoder, vein_encoder, classifier, val_loader, device)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/palm_acc", val_metrics["palm_acc"], epoch)
        writer.add_scalar("val/vein_acc", val_metrics["vein_acc"], epoch)
        for name in ["palm", "vein", "fused", "cross"]:
            writer.add_scalar(f"val/{name}_eer", val_metrics[name]["eer"], epoch)
            writer.add_scalar(f"val/{name}_tar_1e4", val_metrics[name]["tar_1e4"], epoch)
            writer.add_scalar(f"val/{name}_tar_1e5", val_metrics[name]["tar_1e5"], epoch)

        mean_eer = float(
            np.nanmean(
                [
                    val_metrics["palm"]["eer"],
                    val_metrics["vein"]["eer"],
                    val_metrics["fused"]["eer"],
                    val_metrics["cross"]["eer"],
                ]
            )
        )
        mean_tar_1e4 = float(
            np.nanmean(
                [
                    val_metrics["palm"]["tar_1e4"],
                    val_metrics["vein"]["tar_1e4"],
                    val_metrics["fused"]["tar_1e4"],
                    val_metrics["cross"]["tar_1e4"],
                ]
            )
        )
        score = -mean_eer + 0.2 * mean_tar_1e4

        print(
            f"[Epoch {epoch}] modality=joint "
            f"train_loss={running['loss'] / max(sample_count, 1):.4f} "
            f"train_cls={running['cls'] / max(sample_count, 1):.4f} "
            f"train_cm_supcon={running['cm_supcon'] / max(sample_count, 1):.4f} "
            f"train_align={running['align'] / max(sample_count, 1):.4f} "
            f"val_palm_eer={val_metrics['palm']['eer'] * 100:.3f}% "
            f"val_vein_eer={val_metrics['vein']['eer'] * 100:.3f}% "
            f"val_fused_eer={val_metrics['fused']['eer'] * 100:.3f}% "
            f"val_cross_eer={val_metrics['cross']['eer'] * 100:.3f}% "
            f"val_cross_tar@1e-4={val_metrics['cross']['tar_1e4'] * 100:.2f}%"
        )

        if score > best_score:
            best_score = score
            save_joint_checkpoints(args.save_dir, epoch, palm_encoder, vein_encoder, classifier, args, num_classes)
            print(f"[Info] saved joint, palm, and vein encoder checkpoints to {args.save_dir}")


def main():
    parser = argparse.ArgumentParser("Train MobileFaceNet encoders for palmprint-palmvein recognition")
    parser.add_argument("--modality", type=str, choices=["palm", "vein", "joint"], required=True)
    parser.add_argument("--train_full_list", type=str, default="data_txt/polyu_train_full.txt")
    parser.add_argument("--val_full_list", type=str, default="data_txt/polyu_val_full.txt")
    parser.add_argument("--save_dir", type=str, default="outputs_dymo/encoders")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--embedding_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--lambda_cm", type=float, default=0.2)
    parser.add_argument("--lambda_align", type=float, default=0.1)
    parser.add_argument("--supcon_temperature", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, f"{args.modality}_runs"))
    num_classes = infer_num_classes(args.train_full_list)

    if args.modality == "joint":
        train_joint(args, device, writer, num_classes)
    else:
        train_single(args, device, writer, num_classes)

    writer.close()


if __name__ == "__main__":
    main()
