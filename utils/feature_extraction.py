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


FEATURE_CACHE_VERSION = "frozen_encoder_pair_features_v1"
PREPROCESS_VERSION = "modality_transforms_224_v1"


def _paired_sample_order(loader):
    return [
        (
            sample["palm_path"],
            sample["vein_path"],
            int(sample["label"]),
        )
        for sample in loader.dataset.samples
    ]


def load_or_extract_paired_feature_cache(
    cache_path,
    list_path,
    split,
    palm_encoder,
    vein_encoder,
    palm_ckpt,
    vein_ckpt,
    device,
    input_size,
    embedding_size,
    batch_size,
    num_workers,
    description,
    force=False,
):
    """Load a fingerprinted frozen-encoder cache or extract it once."""

    from utils.checkpoint_io import (
        file_sha256,
        safe_torch_load,
        save_checkpoint,
    )

    loader = paired_feature_loader(
        list_path,
        split,
        input_size,
        batch_size,
        num_workers,
    )
    metadata = {
        "cache_version": FEATURE_CACHE_VERSION,
        "preprocess_version": PREPROCESS_VERSION,
        "protocol_sha256": file_sha256(list_path),
        "palm_encoder_sha256": file_sha256(palm_ckpt),
        "vein_encoder_sha256": file_sha256(vein_ckpt),
        "input_size": int(input_size),
        "embedding_size": int(embedding_size),
        "split": split,
        "sample_order": _paired_sample_order(loader),
    }
    if not force and cache_path:
        try:
            payload = safe_torch_load(cache_path, "cpu")
        except FileNotFoundError:
            payload = None
        if payload is not None and payload.get("metadata") == metadata:
            features = payload.get("features")
            if (
                isinstance(features, dict)
                and set(features) == {"palm", "vein", "labels"}
                and features["palm"].shape == features["vein"].shape
                and features["palm"].shape[1] == embedding_size
                and features["labels"].numel() == features["palm"].shape[0]
                and all(torch.isfinite(value).all() for value in features.values())
            ):
                print(f"[Cache] loaded {cache_path}")
                return features, metadata

    features = extract_paired_features(
        palm_encoder,
        vein_encoder,
        loader,
        device,
        description,
    )
    if features["palm"].shape != features["vein"].shape:
        raise ValueError("Paired feature cache modalities have different shapes")
    if features["palm"].shape[1] != embedding_size:
        raise ValueError("Extracted feature dimension differs from embedding_size")
    if not all(torch.isfinite(value).all() for value in features.values()):
        raise ValueError("Feature cache contains non-finite values")
    if cache_path:
        save_checkpoint(
            cache_path,
            {
                "metadata": metadata,
                "features": features,
            },
        )
        print(f"[Cache] saved {cache_path}")
    return features, metadata
