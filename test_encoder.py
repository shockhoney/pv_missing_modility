import argparse
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.backbones import build_encoder
from utils.datasets_txt import MissingPairTxtDataset
from utils.metrics import compute_eer, far_frr_acc_at_threshold, roc_auc, tar_at_far

FARS = (1e-5, 1e-4, 1e-3)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def safe_torch_load(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_encoder(ckpt_path: str, input_size: int, encoder_dim: int, device):
    ckpt = safe_torch_load(ckpt_path, device)
    modality = ckpt.get("modality", "palm")
    encoder = build_encoder(modality, input_channel=3, input_size=input_size, embedding_size=encoder_dim).to(device)
    encoder.load_state_dict(ckpt.get("encoder", ckpt.get("model", ckpt)), strict=False)
    encoder.eval()
    return encoder


def load_joint_encoders(args, device):
    if os.path.exists(args.joint_ckpt):
        ckpt = safe_torch_load(args.joint_ckpt, device)
        encoders = ckpt.get("encoder", {})
        if ckpt.get("modality") == "joint" and {"palm", "vein"} <= set(encoders):
            palm = build_encoder("palm", input_channel=3, input_size=args.input_size, embedding_size=args.encoder_dim).to(device)
            vein = build_encoder("vein", input_channel=3, input_size=args.input_size, embedding_size=args.encoder_dim).to(device)
            palm.load_state_dict(encoders["palm"], strict=False)
            vein.load_state_dict(encoders["vein"], strict=False)
            palm.eval()
            vein.eval()
            return palm, vein
    return (
        load_encoder(args.palm_ckpt, args.input_size, args.encoder_dim, device),
        load_encoder(args.vein_ckpt, args.input_size, args.encoder_dim, device),
    )


def build_loader(protocol_list: str, split_name: str, img_size: int, batch_size: int, num_workers: int):
    dataset = MissingPairTxtDataset(
        protocol_list,
        transform_palm=get_transforms(img_size),
        transform_vein=get_transforms(img_size),
        split_filter=split_name,
    )
    if len(dataset) == 0:
        return None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_pair_scores(feats: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sim = feats @ feats.T
    i, j = np.triu_indices(labels.shape[0], k=1)
    scores = sim[i, j].astype(np.float32)
    pair_labels = (labels[i] == labels[j]).astype(np.int32)
    return scores, pair_labels


def weighted_fusion(palm_feats: np.ndarray, vein_feats: np.ndarray, alpha: float) -> np.ndarray:
    fused = alpha * palm_feats + (1.0 - alpha) * vein_feats
    return fused / np.maximum(np.linalg.norm(fused, axis=1, keepdims=True), 1e-12)


def encode_selected(encoder, images, selected, device):
    if not bool(selected.any()):
        return None
    images = images[selected].to(device, non_blocking=True)
    return F.normalize(encoder(images), dim=1)


@torch.no_grad()
def extract_features(encoder, loader, modality: str, device):
    feats, labels = [], []
    modality_idx = 0 if modality == "palm" else 1

    for palm_img, vein_img, batch_labels, mask in tqdm(loader, desc=f"Extract {modality}", dynamic_ncols=True, leave=False):
        available = mask[:, modality_idx] > 0.5
        if available.sum() == 0:
            continue

        images = palm_img if modality == "palm" else vein_img
        embeddings = encode_selected(encoder, images, available, device)
        feats.append(embeddings.cpu().numpy())
        labels.append(batch_labels[available].numpy())

    if not feats:
        return None, None
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def eval_metrics(feats: np.ndarray, labels: np.ndarray, name: str):
    scores, pair_labels = build_pair_scores(feats, labels)
    positives = int((pair_labels == 1).sum())
    negatives = int((pair_labels == 0).sum())
    print(f"\n===== {name} =====")
    print(f"Samples: {len(labels)}, positive_pairs={positives}, negative_pairs={negatives}")
    if (pair_labels == 1).sum() == 0 or (pair_labels == 0).sum() == 0:
        print("insufficient positive or negative pairs")
        return

    eer, thr = compute_eer(scores, pair_labels, is_similarity=True, return_threshold=True)
    _, _, _, auc_val = roc_auc(scores, pair_labels, is_similarity=True)
    thr_stats = far_frr_acc_at_threshold(scores, pair_labels, thr, is_similarity=True)
    print(f"AUC : {auc_val:.4f}")
    print(f"EER : {eer * 100:.3f}% (threshold = {thr:.4f})")
    print(
        f"ACC@EER_thr = {thr_stats['ACC']:.4f}, "
        f"FAR={thr_stats['FAR']:.4f}, FRR={thr_stats['FRR']:.4f}"
    )
    print("TAR @ FAR:")
    for far in FARS:
        info = tar_at_far(scores, pair_labels, far, is_similarity=True)
        print(f"  FAR={far:.1e}: TAR={info['TAR']:.4f}, thr={info['threshold']:.4f}")


def evaluate_encoder(modality: str, ckpt_path: str, args, device):
    encoder = load_encoder(ckpt_path, args.input_size, args.encoder_dim, device)
    split_names = ["full", "random_missing"]
    split_names.insert(1, "palm_only" if modality == "palm" else "vein_only")

    for split_name in split_names:
        loader = build_loader(args.protocol_list, split_name, args.input_size, args.batch_size, args.num_workers)
        if loader is None:
            continue
        feats, labels = extract_features(encoder, loader, modality, device)
        if feats is not None:
            eval_metrics(feats, labels, f"{modality.capitalize()} Encoder - {split_name}")


@torch.no_grad()
def extract_joint_features(palm_encoder, vein_encoder, loader, device):
    palm_feats, vein_feats, labels = [], [], []
    for palm_img, vein_img, batch_labels, mask in tqdm(loader, desc="Extract joint", dynamic_ncols=True, leave=False):
        available = (mask[:, 0] > 0.5) & (mask[:, 1] > 0.5)
        if available.sum() == 0:
            continue
        palm_feats.append(encode_selected(palm_encoder, palm_img, available, device).cpu().numpy())
        vein_feats.append(encode_selected(vein_encoder, vein_img, available, device).cpu().numpy())
        labels.append(batch_labels[available].numpy())
    if not palm_feats:
        return None, None, None
    return np.concatenate(palm_feats, axis=0), np.concatenate(vein_feats, axis=0), np.concatenate(labels, axis=0)


@torch.no_grad()
def extract_missing_aware_features(palm_encoder, vein_encoder, loader, alpha: float, device):
    feats, labels = [], []
    for palm_img, vein_img, batch_labels, mask in tqdm(loader, desc="Extract missing-aware", dynamic_ncols=True, leave=False):
        has_palm = mask[:, 0] > 0.5
        has_vein = mask[:, 1] > 0.5
        groups = ((has_palm & has_vein, "both"), (has_palm & ~has_vein, "palm"), (~has_palm & has_vein, "vein"))
        for selected, source in groups:
            if not bool(selected.any()):
                continue
            if source == "both":
                palm_feat = encode_selected(palm_encoder, palm_img, selected, device)
                vein_feat = encode_selected(vein_encoder, vein_img, selected, device)
                feat = F.normalize(alpha * palm_feat + (1.0 - alpha) * vein_feat, dim=1)
            elif source == "palm":
                feat = encode_selected(palm_encoder, palm_img, selected, device)
            else:
                feat = encode_selected(vein_encoder, vein_img, selected, device)
            feats.append(feat.cpu().numpy())
            labels.append(batch_labels[selected].numpy())
    if not feats:
        return None, None
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def evaluate_joint(args, device):
    palm_encoder, vein_encoder = load_joint_encoders(args, device)
    for split_name in ["full", "random_missing"]:
        loader = build_loader(args.protocol_list, split_name, args.input_size, args.batch_size, args.num_workers)
        if loader is None:
            continue
        if split_name == "full":
            palm_feats, vein_feats, labels = extract_joint_features(palm_encoder, vein_encoder, loader, device)
            if palm_feats is None:
                continue
            eval_metrics(weighted_fusion(palm_feats, vein_feats, 0.5), labels, f"Joint Encoder - {split_name} alpha=0.50")
        else:
            feats, labels = extract_missing_aware_features(palm_encoder, vein_encoder, loader, 0.5, device)
            if feats is not None:
                eval_metrics(feats, labels, f"Joint Encoder - {split_name} missing-aware")


def main():
    parser = argparse.ArgumentParser("Evaluate Hetero-MMRNet encoders")
    parser.add_argument("--protocol_list", type=str, default="data_txt/polyu/test_missing_protocol.txt")
    parser.add_argument("--modality", type=str, choices=["palm", "vein", "joint", "all"], default="all")
    parser.add_argument("--palm_ckpt", type=str, default="outputs/encoders/palm_best.pth")
    parser.add_argument("--vein_ckpt", type=str, default="outputs/encoders/vein_best.pth")
    parser.add_argument("--joint_ckpt", type=str, default="outputs/encoders/joint_best.pth")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--encoder_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.modality in {"palm", "all"}:
        evaluate_encoder("palm", args.palm_ckpt, args, device)
    if args.modality in {"vein", "all"}:
        evaluate_encoder("vein", args.vein_ckpt, args, device)
    if args.modality in {"joint", "all"}:
        evaluate_joint(args, device)


if __name__ == "__main__":
    main()
