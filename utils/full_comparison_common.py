"""Shared image-level data, evaluation, and audit utilities for full baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset

from utils.comparison_metrics import identity_templates, metrics_from_representations
from utils.datasets_txt import MissingPairTxtDataset
from utils.preprocess import (
    build_paired_geometry_transform,
    build_palm_transform,
    build_vein_transform,
)
from utils.scenarios import PALMPRINT_MISSING, PALMVEIN_MISSING


SCENARIOS = (PALMPRINT_MISSING, PALMVEIN_MISSING)


class RemappedLabels(Dataset):
    """Map sparse protocol labels to contiguous training classifier indices."""

    def __init__(self, dataset: Dataset, label_ids: list[int]) -> None:
        self.dataset = dataset
        self.mapping = {int(label): index for index, label in enumerate(label_ids)}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        palm, vein, label, mask = self.dataset[index]
        if int(label) not in self.mapping:
            raise ValueError(f"Unexpected training label {label}")
        return palm, vein, self.mapping[int(label)], mask


def labels_in_protocol(path: str | Path) -> list[int]:
    labels: set[int] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 3:
                labels.add(int(parts[2]))
    if not labels:
        raise ValueError(f"No labels found in {path}")
    return sorted(labels)


def paired_dataset(
    list_path: str | Path,
    *,
    input_size: int,
    train: bool,
    split_filter: str | None = None,
    remap_labels: list[int] | None = None,
) -> Dataset:
    dataset = MissingPairTxtDataset(
        str(list_path),
        transform_palm=build_palm_transform(input_size, train=train, geometric=False),
        transform_vein=build_vein_transform(input_size, train=train, geometric=False),
        split_filter=split_filter,
        paired_transform=(build_paired_geometry_transform(input_size) if train else None),
    )
    if remap_labels is not None:
        return RemappedLabels(dataset, remap_labels)
    return dataset


def paired_loader(
    list_path: str | Path,
    *,
    input_size: int,
    batch_size: int,
    num_workers: int,
    train: bool,
    split_filter: str | None = None,
    remap_labels: list[int] | None = None,
    seed: int = 42,
) -> DataLoader:
    dataset = paired_dataset(
        list_path,
        input_size=input_size,
        train=train,
        split_filter=split_filter,
        remap_labels=remap_labels,
    )
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        # Recreating workers each epoch plus checkpointing ``generator`` makes
        # epoch-boundary resume reproduce worker/Python augmentation seeds.
        persistent_workers=False,
        generator=generator,
    )


@torch.inference_mode()
def collect_representations(
    loader: DataLoader,
    representation_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    representations, labels = [], []
    for palm, vein, batch_labels, masks in loader:
        palm = palm.to(device, non_blocking=True)
        vein = vein.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).bool()
        output = representation_fn(palm, vein, masks)
        if output.ndim != 2 or output.size(0) != palm.size(0):
            raise ValueError("Representation callback must return [B, D]")
        if not torch.isfinite(output).all():
            raise FloatingPointError("Evaluation representation contains NaN or Inf")
        representations.append(output.float().cpu())
        labels.append(batch_labels.long().cpu())
    if not representations:
        raise ValueError("Evaluation loader is empty")
    return torch.cat(representations), torch.cat(labels)


def evaluate_gallery_probe(
    gallery_loader: DataLoader,
    scenario_loaders: dict[str, DataLoader],
    representation_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    device: torch.device,
) -> dict[str, dict]:
    gallery, gallery_labels = collect_representations(
        gallery_loader, representation_fn, device
    )
    templates, template_labels = identity_templates(gallery, gallery_labels)
    results = {}
    for scenario in SCENARIOS:
        probes, probe_labels = collect_representations(
            scenario_loaders[scenario], representation_fn, device
        )
        results[scenario] = {
            "fused": metrics_from_representations(
                probes, probe_labels, templates, template_labels
            )
        }
    return results


def atomic_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def metric_rank(results: dict[str, dict]) -> tuple[float, ...]:
    """Rank checkpoints with the repository's locked two-scenario rule.

    Selection must not privilege whichever missing-modality scenario happens
    to be iterated first.  The four components intentionally match
    ``utils.comparison_metrics.validation_rank`` so full reproductions and the
    original controlled comparison use the same validation-only criterion.
    """

    fused = [results[scenario]["fused"] for scenario in SCENARIOS]
    return (
        sum(float(item["eer"]) for item in fused) / len(fused),
        max(float(item["eer"]) for item in fused),
        -sum(float(item["tar_at_far"][1e-4]) for item in fused) / len(fused),
        -sum(float(item["tar_at_far"][1e-3]) for item in fused) / len(fused),
    )


__all__ = [
    "RemappedLabels",
    "SCENARIOS",
    "atomic_json",
    "collect_representations",
    "evaluate_gallery_probe",
    "labels_in_protocol",
    "metric_rank",
    "paired_dataset",
    "paired_loader",
]
