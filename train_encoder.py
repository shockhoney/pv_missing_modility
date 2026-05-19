import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.backbones import build_encoder
from utils.augmentations import StarMix, UAAAffineAugmenter, optimize_uaa_params
from utils.datasets_txt import MissingPairTxtDataset, SingleModalityFromPairDataset
from utils.head import ArcFace
from utils.metrics import compute_eer, tar_at_far


def get_transforms(img_size, strong):
    ops = [transforms.Resize((img_size, img_size))]
    if strong:
        ops += [transforms.RandomHorizontalFlip(0.5), transforms.ColorJitter(0.15, 0.15)]
    ops += [
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
    return transforms.Compose(ops)


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


def make_loader(args, list_path, modality=None, strong=False):
    tf = get_transforms(args.input_size, strong)
    dataset = (
        MissingPairTxtDataset(list_path, tf, tf)
        if modality is None
        else SingleModalityFromPairDataset(list_path, modality, tf)
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=strong,
        drop_last=strong,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def make_encoder(modality, args):
    return build_encoder(modality, input_channel=3, input_size=args.input_size, embedding_size=args.embedding_size)


def pair_scores(feats, labels):
    sim = feats @ feats.T
    i, j = np.triu_indices(labels.shape[0], k=1)
    return sim[i, j].astype(np.float32), (labels[i] == labels[j]).astype(np.int32)


def metrics_from_features(feats, labels):
    feats = np.asarray(feats, dtype=np.float32)
    labels = np.asarray(labels)
    valid = np.isfinite(feats).all(axis=1)
    if not valid.all():
        print(f"[Warn] dropped {int((~valid).sum())} non-finite validation embeddings")
        feats, labels = feats[valid], labels[valid]
    if feats.shape[0] < 2:
        return {"eer": float("inf"), "tar_1e4": 0.0, "tar_1e5": 0.0}

    scores, pair_labels = pair_scores(feats, labels)
    valid = np.isfinite(scores)
    if not valid.all():
        scores, pair_labels = scores[valid], pair_labels[valid]
    if scores.size == 0 or np.unique(pair_labels).size < 2:
        return {"eer": float("inf"), "tar_1e4": 0.0, "tar_1e5": 0.0}

    tar_1e4 = tar_at_far(scores, pair_labels, 1e-4, is_similarity=True)
    tar_1e5 = tar_at_far(scores, pair_labels, 1e-5, is_similarity=True)
    return {
        "eer": float(compute_eer(scores, pair_labels, is_similarity=True)),
        "tar_1e4": float(tar_1e4["TAR"] if isinstance(tar_1e4, dict) else tar_1e4),
        "tar_1e5": float(tar_1e5["TAR"] if isinstance(tar_1e5, dict) else tar_1e5),
    }


@torch.no_grad()
def validate_single(encoder, loader, device, desc):
    encoder.eval()
    feats, labels = [], []
    for images, y in tqdm(loader, desc=desc, dynamic_ncols=True, leave=False):
        feats.append(F.normalize(encoder(images.to(device, non_blocking=True)), dim=1).cpu().numpy())
        labels.append(y.numpy())
    return metrics_from_features(np.vstack(feats), np.concatenate(labels))


@torch.no_grad()
def validate_joint(palm_encoder, vein_encoder, loader, device):
    palm_encoder.eval()
    vein_encoder.eval()
    palm_feats, vein_feats, labels = [], [], []
    for palm, vein, y, _ in tqdm(loader, desc="Validate joint", dynamic_ncols=True, leave=False):
        palm = palm.to(device, non_blocking=True)
        vein = vein.to(device, non_blocking=True)
        palm_feats.append(F.normalize(palm_encoder(palm), dim=1).cpu().numpy())
        vein_feats.append(F.normalize(vein_encoder(vein), dim=1).cpu().numpy())
        labels.append(y.numpy())

    palm_feats = np.vstack(palm_feats)
    vein_feats = np.vstack(vein_feats)
    labels = np.concatenate(labels)
    joint = palm_feats + vein_feats
    joint /= np.maximum(np.linalg.norm(joint, axis=1, keepdims=True), 1e-12)
    return {
        "palm": metrics_from_features(palm_feats, labels),
        "vein": metrics_from_features(vein_feats, labels),
        "joint": metrics_from_features(joint, labels),
    }


def save_checkpoint(path, epoch, modality, encoder, classifier, args, num_classes):
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


def save_joint(path, epoch, palm_encoder, vein_encoder, heads, args, num_classes):
    torch.save(
        {
            "epoch": epoch,
            "modality": "joint",
            "encoder": {"palm": palm_encoder.state_dict(), "vein": vein_encoder.state_dict()},
            "classifier": {name: head.state_dict() for name, head in heads.items() if head is not None},
            "args": vars(args),
            "num_classes": num_classes,
        },
        path,
    )


def apply_uaa(images, labels, encoder, classifier, augmenter, args, prev_params):
    if not args.use_uaa:
        return images, labels, prev_params
    was_training = encoder.training, classifier.training
    encoder.eval()
    classifier.eval()
    try:
        aug, selected_labels, params = optimize_uaa_params(
            encoder,
            classifier,
            images,
            labels,
            augmenter,
            steps=args.uaa_steps,
            step_size=args.uaa_step_size,
            beta=args.uaa_beta,
            prev_params=prev_params,
            gamma=args.uaa_gamma,
        )
    finally:
        encoder.train(was_training[0])
        classifier.train(was_training[1])
    return aug.detach(), selected_labels.detach(), params.detach()


def arcface_loss(head, feat, labels, ce):
    logits = head(feat, labels)
    return logits, ce(logits, labels)


def optimizer_step(loss, optimizer, params):
    if not torch.isfinite(loss):
        optimizer.zero_grad(set_to_none=True)
        return False
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(params, 5.0)
    if not torch.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        return False
    optimizer.step()
    return True


def modality_loss(modality, encoder, head, images, labels, ce, args, augmenter=None, mixer=None, prev_params=None):
    if modality == "palm":
        images, labels, prev_params = apply_uaa(images, labels, encoder, head, augmenter, args, prev_params)
        logits, loss = arcface_loss(head, encoder(images), labels, ce)
        return logits, loss, labels, prev_params

    if args.use_starmix:
        mixed, labels_a, labels_b, lam = mixer(images, labels)
        feat = encoder(mixed)
        logits_a = head(feat, labels_a)
        logits_b = head(feat, labels_b)
        loss = lam * ce(logits_a, labels_a) + (1.0 - lam) * ce(logits_b, labels_b)
        return logits_a, loss, labels_a, prev_params

    logits, loss = arcface_loss(head, encoder(images), labels, ce)
    return logits, loss, labels, prev_params


def train_single(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = get_device(args.device)
    num_classes = infer_num_classes(args.train_full_list)
    train_loader = make_loader(args, args.train_full_list, modality=args.modality, strong=True)
    val_loader = make_loader(args, args.val_full_list, modality=args.modality)

    encoder = make_encoder(args.modality, args).to(device)
    head = ArcFace(args.embedding_size, num_classes, args.arcface_s, args.arcface_m).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    ce = nn.CrossEntropyLoss()
    augmenter = UAAAffineAugmenter() if args.modality == "palm" else None
    mixer = StarMix() if args.modality == "vein" else None
    best_eer, bad_epochs, prev_uaa = float("inf"), 0, None
    best_path = os.path.join(args.save_dir, f"{args.modality}_best.pth")

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        head.train()
        loss_sum = correct = total = 0
        for images, labels in tqdm(train_loader, desc=f"{args.modality} epoch {epoch}", dynamic_ncols=True):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, loss, used_labels, prev_uaa = modality_loss(
                args.modality, encoder, head, images, labels, ce, args, augmenter, mixer, prev_uaa
            )

            if not optimizer_step(loss, optimizer, params):
                print("[Warn] skipped non-finite training step")
                continue

            total += used_labels.size(0)
            loss_sum += loss.item() * used_labels.size(0)
            correct += (logits.argmax(1) == used_labels).sum().item()

        scheduler.step()
        val = validate_single(encoder, val_loader, device, f"Validate {args.modality}")
        print(
            f"[Epoch {epoch}] {args.modality} loss={loss_sum / max(total, 1):.4f} "
            f"acc={correct / max(total, 1):.4f} val_eer={val['eer'] * 100:.3f}%"
        )
        improved = val["eer"] < best_eer - args.min_delta
        if improved:
            best_eer, bad_epochs = val["eer"], 0
            save_checkpoint(best_path, epoch, args.modality, encoder, head, args, num_classes)
            print(f"[Info] saved {best_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break


def train_joint(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = get_device(args.device)
    num_classes = infer_num_classes(args.train_full_list)
    train_loader = make_loader(args, args.train_full_list, strong=True)
    val_loader = make_loader(args, args.val_full_list)

    palm_encoder = make_encoder("palm", args).to(device)
    vein_encoder = make_encoder("vein", args).to(device)
    heads = {
        "palm": ArcFace(args.embedding_size, num_classes, args.arcface_s, args.arcface_m).to(device),
        "vein": ArcFace(args.embedding_size, num_classes, args.arcface_s, args.arcface_m).to(device),
        "joint": ArcFace(args.embedding_size, num_classes, args.arcface_s, args.arcface_m).to(device)
        if args.lambda_joint > 0
        else None,
    }
    encoders = {"palm": palm_encoder, "vein": vein_encoder}
    modules = [palm_encoder, vein_encoder, heads["palm"], heads["vein"]] + ([heads["joint"]] if heads["joint"] else [])
    params = [p for module in modules for p in module.parameters()]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    ce = nn.CrossEntropyLoss()
    augmenter, mixer = UAAAffineAugmenter(), StarMix()
    best = {"palm": float("inf"), "vein": float("inf"), "joint": float("inf")}
    bad_epochs, prev_uaa = 0, None

    for epoch in range(1, args.epochs + 1):
        for module in modules:
            module.train()
        loss_sum = 0.0
        for palm, vein, labels, _ in tqdm(train_loader, desc=f"joint epoch {epoch}", dynamic_ncols=True):
            palm = palm.to(device, non_blocking=True)
            vein = vein.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            palm_logits, loss_palm, _, prev_uaa = modality_loss(
                "palm", palm_encoder, heads["palm"], palm, labels, ce, args, augmenter, None, prev_uaa
            )
            vein_logits, loss_vein, _, _ = modality_loss(
                "vein", vein_encoder, heads["vein"], vein, labels, ce, args, None, mixer
            )
            palm_feat = palm_encoder(palm)
            vein_feat = vein_encoder(vein)
            loss = loss_palm + loss_vein + args.lambda_align * (1.0 - F.cosine_similarity(palm_feat, vein_feat).mean())

            if heads["joint"] is not None:
                _, loss_joint = arcface_loss(heads["joint"], F.normalize(palm_feat + vein_feat, dim=1), labels, ce)
                loss = loss + args.lambda_joint * loss_joint

            if not optimizer_step(loss, optimizer, params):
                print("[Warn] skipped non-finite training step")
                continue
            loss_sum += loss.item() * labels.size(0)

        scheduler.step()
        val = validate_joint(palm_encoder, vein_encoder, val_loader, device)
        print(
            f"[Epoch {epoch}] joint loss={loss_sum / max(len(train_loader.dataset), 1):.4f} "
            f"palm_eer={val['palm']['eer'] * 100:.3f}% "
            f"vein_eer={val['vein']['eer'] * 100:.3f}% "
            f"joint_eer={val['joint']['eer'] * 100:.3f}%"
        )

        joint_improved = val["joint"]["eer"] < best["joint"] - args.min_delta
        for name in ("palm", "vein"):
            if val[name]["eer"] < best[name] - args.min_delta:
                best[name] = val[name]["eer"]
                save_checkpoint(os.path.join(args.save_dir, f"{name}_best.pth"), epoch, name, encoders[name], heads[name], args, num_classes)
        if joint_improved:
            best["joint"] = val["joint"]["eer"]
            save_joint(os.path.join(args.save_dir, "joint_best.pth"), epoch, palm_encoder, vein_encoder, heads, args, num_classes)

        bad_epochs = 0 if joint_improved else bad_epochs + 1
        if bad_epochs >= args.patience:
            break


def parse_args():
    parser = argparse.ArgumentParser("Train palm/vein encoders")
    parser.add_argument("--modality", choices=["joint", "palm", "vein"], default="joint")
    parser.add_argument("--train_full_list", default="data_txt/polyu/train_full.txt")
    parser.add_argument("--val_full_list", default="data_txt/polyu/val_full.txt")
    parser.add_argument("--save_dir", default="outputs/encoders")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--embedding_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--arcface_s", type=float, default=32.0)
    parser.add_argument("--arcface_m", type=float, default=0.25)
    parser.add_argument("--lambda_align", type=float, default=1.0)
    parser.add_argument("--lambda_joint", type=float, default=0.0)
    parser.add_argument("--use_uaa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_starmix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--uaa_steps", type=int, default=1)
    parser.add_argument("--uaa_step_size", type=float, default=0.1)
    parser.add_argument("--uaa_beta", type=float, default=0.5)
    parser.add_argument("--uaa_gamma", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--min_delta", type=float, default=0.001)
    return parser.parse_args()


def main():
    args = parse_args()
    train_joint(args) if args.modality == "joint" else train_single(args)


if __name__ == "__main__":
    main()
