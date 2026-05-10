import argparse
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.stage1_mobileFacenet import MobileFaceNet
from models.stage2 import Stage2Fusion
from utils.datasets_txt import MissingPairTxtDataset
from utils.metrics import compute_eer, far_frr_acc_at_threshold, roc_auc, tar_at_far


def get_transforms(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )


def safe_torch_load(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_loader(list_path: str, input_size: int, batch_size: int, num_workers: int, split_filter: str | None):
    tf = get_transforms(input_size)
    dataset = MissingPairTxtDataset(
        list_path,
        transform_palm=tf,
        transform_vein=tf,
        split_filter=split_filter,
    )
    if len(dataset) == 0 and split_filter is not None:
        dataset = MissingPairTxtDataset(
            list_path,
            transform_palm=tf,
            transform_vein=tf,
            split_filter=None,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_pair_scores(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features)
    labels = np.asarray(labels)
    sim = features @ features.T
    i, j = np.triu_indices(labels.shape[0], k=1)
    scores = sim[i, j].astype(np.float32)
    pair_labels = (labels[i] == labels[j]).astype(np.int32)
    return scores, pair_labels


def eval_with_metrics(scores: np.ndarray, pair_labels: np.ndarray, name: str):
    eer, thr = compute_eer(scores, pair_labels, is_similarity=True, return_threshold=True)
    _, _, _, auc_val = roc_auc(scores, pair_labels, is_similarity=True)
    thr_stats = far_frr_acc_at_threshold(scores, pair_labels, thr, is_similarity=True)
    fars = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    tar_info = {far: tar_at_far(scores, pair_labels, far, is_similarity=True) for far in fars}

    print(f"\n===== {name} =====")
    print(f"AUC : {auc_val:.4f}")
    print(f"EER : {eer * 100:.3f}% (threshold = {thr:.4f})")
    print(
        f"ACC@EER_thr = {thr_stats['ACC']:.4f}, "
        f"FAR={thr_stats['FAR']:.4f}, FRR={thr_stats['FRR']:.4f}"
    )
    print("TAR @ FAR:")
    for far, info in tar_info.items():
        print(f"  FAR={far:.1e}: TAR={info['TAR']:.4f}, thr={info['threshold']:.4f}")


@torch.no_grad()
def extract_fusion_features(
    palm_encoder: nn.Module,
    vein_encoder: nn.Module,
    fusion_model: nn.Module,
    loader: DataLoader,
    device,
):
    palm_encoder.eval()
    vein_encoder.eval()
    fusion_model.eval()
    feats, labels_all = [], []

    for palm, vein, labels, mask in tqdm(loader, desc="Extract teacher fusion", dynamic_ncols=True):
        available = (mask[:, 0] > 0.5) & (mask[:, 1] > 0.5)
        if available.sum() == 0:
            continue
        palm = palm[available].to(device)
        vein = vein[available].to(device)
        labels = labels[available]

        palm_feat = palm_encoder(palm)
        vein_feat = vein_encoder(vein)
        fused_feat = F.normalize(fusion_model(palm_feat, vein_feat), dim=1)
        feats.append(fused_feat.cpu().numpy())
        labels_all.append(labels.numpy())

    if not feats:
        return None, None
    return np.concatenate(feats, axis=0), np.concatenate(labels_all, axis=0)


def main():
    parser = argparse.ArgumentParser("Evaluate full-modality teacher fusion baseline")
    parser.add_argument("--protocol_list", type=str, default="data_txt/polyu_test_missing_protocol.txt")
    parser.add_argument("--checkpoint", type=str, default="outputs_dymo/full_fusion/stage2_best.pth")
    parser.add_argument("--split_filter", type=str, default="full")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--encoder_dim", type=int, default=256)
    parser.add_argument("--fusion_dim", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = safe_torch_load(args.checkpoint, device)
    ckpt_args: Dict = ckpt.get("args", {})
    encoder_dim = int(ckpt_args.get("encoder_dim", args.encoder_dim))
    fusion_dim = int(ckpt_args.get("fusion_dim", args.fusion_dim))

    palm_encoder = MobileFaceNet(input_channel=3, input_size=args.input_size, embedding_size=encoder_dim).to(device)
    vein_encoder = MobileFaceNet(input_channel=3, input_size=args.input_size, embedding_size=encoder_dim).to(device)
    fusion_model = Stage2Fusion(
        in_dim_global=encoder_dim,
        out_dim_final=fusion_dim,
        final_l2norm=True,
    ).to(device)

    palm_encoder.load_state_dict(ckpt["cnn_palm"], strict=True)
    vein_encoder.load_state_dict(ckpt["cnn_vein"], strict=True)
    fusion_model.load_state_dict(ckpt["fusion"], strict=True)

    loader = build_loader(args.protocol_list, args.input_size, args.batch_size, args.num_workers, args.split_filter)
    feats, labels = extract_fusion_features(palm_encoder, vein_encoder, fusion_model, loader, device)
    if feats is None:
        raise RuntimeError("No full-modality samples found for evaluation.")
    scores, pair_labels = build_pair_scores(feats, labels)
    eval_with_metrics(scores, pair_labels, name=f"Teacher Fusion - {args.split_filter or 'all'}")


if __name__ == "__main__":
    main()
