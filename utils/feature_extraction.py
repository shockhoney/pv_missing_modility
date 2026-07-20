"""Frozen-encoder feature extraction helpers for recovery training and evaluation."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.datasets_txt import MissingPairTxtDataset, SingleModalityFromPairDataset
from utils.preprocess import build_modality_transform, build_palm_transform, build_vein_transform


def paired_feature_loader(list_path, split, input_size, batch_size, num_workers):
    dataset = MissingPairTxtDataset(
        list_path,
        build_palm_transform(input_size),
        build_vein_transform(input_size),
        split_filter=split,
    )
    if not dataset.samples:
        raise ValueError(f"No samples found in {list_path!r} for split={split!r}")
    for sample in dataset.samples:
        if not (sample["palm_exists"] and sample["vein_exists"]):
            raise ValueError("Paired feature extraction requires complete samples")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def single_feature_loader(list_path, modality, split, input_size, batch_size, num_workers):
    dataset = SingleModalityFromPairDataset(
        list_path,
        modality,
        build_modality_transform(modality, input_size),
        split_filter=split,
    )
    if not dataset:
        raise ValueError(f"No {modality} samples found for split={split!r}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.inference_mode()
def extract_paired_features(palm_encoder, vein_encoder, loader, device, description):
    values = {"palm": [], "vein": [], "labels": []}
    for palm, vein, labels, _ in tqdm(loader, desc=description, dynamic_ncols=True):
        values["palm"].append(palm_encoder(palm.to(device, non_blocking=True)).cpu())
        values["vein"].append(vein_encoder(vein.to(device, non_blocking=True)).cpu())
        values["labels"].append(labels)
    return {key: torch.cat(items) for key, items in values.items()}


@torch.inference_mode()
def extract_single_features(encoder, loader, device, description):
    features, labels = [], []
    for images, batch_labels in tqdm(loader, desc=description, dynamic_ncols=True):
        features.append(encoder(images.to(device, non_blocking=True)).cpu())
        labels.append(batch_labels)
    return torch.cat(features), torch.cat(labels)
