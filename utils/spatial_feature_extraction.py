"""Frozen embedding and spatial-map extraction for SpecFormer recovery."""

from __future__ import annotations

import torch
from tqdm import tqdm

from utils.feature_extraction import PREPROCESS_VERSION, paired_feature_loader


SPATIAL_FEATURE_CACHE_VERSION = "frozen_encoder_pair_spatial_features_v1"


def _paired_sample_order(loader):
    return [
        (sample["palm_path"], sample["vein_path"], int(sample["label"]))
        for sample in loader.dataset.samples
    ]


@torch.inference_mode()
def extract_paired_spatial_features(palm_encoder, vein_encoder, loader, device, description):
    """Extract pooled embeddings and final 7x7 feature maps in one pass."""

    values = {
        "palm": [],
        "vein": [],
        "palm_spatial": [],
        "vein_spatial": [],
        "labels": [],
    }
    for palm, vein, labels, _ in tqdm(loader, desc=description, dynamic_ncols=True):
        palm_spatial = palm_encoder(palm.to(device, non_blocking=True), return_spatial=True)
        vein_spatial = vein_encoder(vein.to(device, non_blocking=True), return_spatial=True)
        values["palm"].append(palm_encoder.embedding_from_features(palm_spatial).cpu())
        values["vein"].append(vein_encoder.embedding_from_features(vein_spatial).cpu())
        values["palm_spatial"].append(palm_spatial.half().cpu())
        values["vein_spatial"].append(vein_spatial.half().cpu())
        values["labels"].append(labels)
    return {key: torch.cat(items) for key, items in values.items()}


@torch.inference_mode()
def extract_single_spatial_features(encoder, loader, device, description):
    embeddings, spatial, labels = [], [], []
    for images, batch_labels in tqdm(loader, desc=description, dynamic_ncols=True):
        maps = encoder(images.to(device, non_blocking=True), return_spatial=True)
        embeddings.append(encoder.embedding_from_features(maps).cpu())
        spatial.append(maps.half().cpu())
        labels.append(batch_labels)
    return torch.cat(embeddings), torch.cat(spatial), torch.cat(labels)


def load_or_extract_paired_spatial_cache(
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
    """Load or create a fingerprinted embedding + fp16 spatial-map cache."""

    from utils.checkpoint_io import file_sha256, safe_torch_load, save_checkpoint

    loader = paired_feature_loader(
        list_path, split, input_size, batch_size, num_workers
    )
    metadata = {
        "cache_version": SPATIAL_FEATURE_CACHE_VERSION,
        "preprocess_version": PREPROCESS_VERSION,
        "protocol_sha256": file_sha256(list_path),
        "palm_encoder_sha256": file_sha256(palm_ckpt),
        "vein_encoder_sha256": file_sha256(vein_ckpt),
        "input_size": int(input_size),
        "embedding_size": int(embedding_size),
        "split": split,
        "sample_order": _paired_sample_order(loader),
    }
    expected_keys = {
        "palm", "vein", "palm_spatial", "vein_spatial", "labels"
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
                and set(features) == expected_keys
                and features["palm"].shape == features["vein"].shape
                and features["palm"].shape[1] == embedding_size
                and features["palm_spatial"].shape == features["vein_spatial"].shape
                and features["palm_spatial"].shape[:2]
                == (features["palm"].shape[0], embedding_size)
                and features["labels"].numel() == features["palm"].shape[0]
                and all(torch.isfinite(value).all() for value in features.values())
            ):
                print(f"[Cache] loaded {cache_path}")
                return features, metadata

    features = extract_paired_spatial_features(
        palm_encoder, vein_encoder, loader, device, description
    )
    if features["palm"].shape != features["vein"].shape:
        raise ValueError("Paired spatial cache modalities have different embedding shapes")
    if features["palm_spatial"].shape != features["vein_spatial"].shape:
        raise ValueError("Paired spatial cache modalities have different map shapes")
    if not all(torch.isfinite(value).all() for value in features.values()):
        raise ValueError("Spatial feature cache contains non-finite values")
    if cache_path:
        save_checkpoint(cache_path, {"metadata": metadata, "features": features})
        print(f"[Cache] saved {cache_path}")
    return features, metadata
