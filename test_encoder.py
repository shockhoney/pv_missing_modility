import argparse
import csv
import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.backbones import build_encoder
from utils.checkpoint import safe_torch_load
from utils.datasets_txt import SingleModalityFromPairDataset
from utils.evaluation import eer_from_embeddings, recognition_rate
from utils.head import ArcFace
from utils.preprocess import build_palm_transform, build_vein_transform


def load_model(ckpt_path: str, modality: str, input_size: int, encoder_dim: int, device):
    ckpt = safe_torch_load(ckpt_path, device)
    encoder = build_encoder(modality, input_channel=3, input_size=input_size, embedding_size=encoder_dim).to(device)
    encoder.load_state_dict(ckpt.get("encoder", ckpt.get("model", ckpt)), strict=False)
    encoder.eval()

    classifier_state = ckpt.get("classifier")
    if classifier_state is None:
        raise ValueError("Checkpoint does not contain a classifier head for closed-set recognition.")
    num_classes = ckpt.get("num_classes", classifier_state["weight"].shape[0])
    ckpt_args = ckpt.get("args", {})
    classifier = ArcFace(
        encoder_dim,
        num_classes,
        ckpt_args.get("arcface_s", 32.0),
        ckpt_args.get("arcface_m", 0.25),
    ).to(device)
    classifier.load_state_dict(classifier_state)
    classifier.eval()
    return encoder, classifier


def build_loader(protocol_list: str, modality: str, split_name: str, img_size: int, batch_size: int, num_workers: int):
    transform = build_palm_transform(img_size) if modality == "palm" else build_vein_transform(img_size)
    dataset = SingleModalityFromPairDataset(
        protocol_list,
        modality=modality,
        transform=transform,
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


@torch.no_grad()
def extract_logits(encoder, classifier, loader, modality: str, device):
    logits, embeddings, labels = [], [], []
    for images, batch_labels in tqdm(loader, desc=f"Extract {modality}", dynamic_ncols=True, leave=False):
        images = images.to(device, non_blocking=True)
        features = encoder(images)
        embeddings.append(features.cpu().numpy())
        logits.append(classifier(features).cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(logits, axis=0), np.concatenate(embeddings, axis=0), np.concatenate(labels, axis=0)


def eval_metrics(logits: np.ndarray, embeddings: np.ndarray, labels: np.ndarray, name: str):
    print(f"\n===== {name} =====")
    print(f"Samples: {len(labels)}")
    logits = torch.as_tensor(logits)
    labels = torch.as_tensor(labels)
    print(f"Recognition Rate (%): {recognition_rate(logits, labels) * 100:.2f}")
    print(f"EER (%): {eer_from_embeddings(embeddings, labels) * 100:.2f}")


def error_analysis(preds: np.ndarray, labels: np.ndarray, samples, modality: str):
    rows = []
    counts = Counter()
    path_key = f"{modality}_path"
    for idx, (pred, label) in enumerate(zip(preds.tolist(), labels.tolist())):
        if pred == label:
            continue
        sample = samples[idx] if idx < len(samples) else {}
        counts[label] += 1
        rows.append(
            {
                "index": idx,
                "path": sample.get(path_key, ""),
                "true_label": label,
                "pred_label": pred,
            }
        )

    total = len(rows)
    top_label, top_count = counts.most_common(1)[0] if counts else (-1, 0)
    summary = {
        "num_errors": total,
        "num_error_classes": len(counts),
        "top_error_label": top_label,
        "top_error_count": top_count,
        "top_error_ratio": top_count / max(total, 1),
        "is_concentrated": top_count >= 3 or (top_count >= 2 and top_count / max(total, 1) >= 0.2),
        "class_counts": counts,
    }
    return rows, summary


def write_error_csv(path: str, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "path", "true_label", "pred_label"])
        writer.writeheader()
        writer.writerows(rows)


def print_error_summary(summary, top_k: int):
    print(
        f"Errors: {summary['num_errors']}, Error classes: {summary['num_error_classes']}, "
        f"Top class: {summary['top_error_label']} ({summary['top_error_count']}, "
        f"{summary['top_error_ratio'] * 100:.2f}%)"
    )
    state = "concentrated" if summary["is_concentrated"] else "scattered"
    print(f"Error distribution: {state}")
    for label, count in summary["class_counts"].most_common(top_k):
        print(f"  label {label}: {count}")


def evaluate_encoder(modality: str, ckpt_path: str, args, device):
    encoder, classifier = load_model(ckpt_path, modality, args.input_size, args.encoder_dim, device)
    split_names = ["complete", "palmvein_missing" if modality == "palm" else "palmprint_missing"]
    for split_name in split_names:
        loader = build_loader(args.protocol_list, modality, split_name, args.input_size, args.batch_size, args.num_workers)
        if loader is None:
            continue
        logits, embeddings, labels = extract_logits(encoder, classifier, loader, modality, device)
        eval_metrics(logits, embeddings, labels, f"{modality.capitalize()} Encoder - {split_name}")
        if args.error_csv and split_name != "complete":
            preds = logits.argmax(axis=1)
            rows, summary = error_analysis(preds, labels, loader.dataset.samples, modality)
            write_error_csv(args.error_csv, rows)
            print_error_summary(summary, args.error_top_k)
            print(f"Error CSV: {args.error_csv}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Evaluate single-modality encoders")
    parser.add_argument("--protocol_list", type=str, default="data_txt/cumt/ssfd_test_protocol.txt")
    parser.add_argument("--modality", type=str, choices=["palm", "vein"], default="palm")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--encoder_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--error_csv", type=str, default=None)
    parser.add_argument("--error_top_k", type=int, default=10)
    args = parser.parse_args(argv)
    if args.ckpt is None:
        args.ckpt = f"outputs/encoders/{args.modality}_best.pth"
    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_encoder(args.modality, args.ckpt, args, device)


if __name__ == "__main__":
    main()
