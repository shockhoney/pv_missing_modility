import argparse
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
NA_TOKEN = "NA"
PROTOCOL_FILES = {
    "train_full": "train_full.txt",
    "val_full": "val_full.txt",
    "val_missing_fixed": "val_missing_fixed.txt",
    "test_missing_protocol": "test_missing_protocol.txt",
}


def _path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path.replace("\\", "/"))


def _rel(path: str) -> str:
    return os.path.relpath(path, PROJECT_ROOT).replace("\\", "/")


def _images(path: str) -> List[str]:
    return sorted(name for name in os.listdir(path) if name.lower().endswith(IMAGE_EXTS))


def _write(path: str, lines: List[str]) -> None:
    path = _path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(lines)


def _collect_pairs(root_dir: str, palm_dir_name: str, vein_dir_name: str) -> Dict[str, List[Tuple[str, str]]]:
    palm_root = _path(os.path.join(root_dir, palm_dir_name))
    vein_root = _path(os.path.join(root_dir, vein_dir_name))
    if not os.path.isdir(palm_root):
        raise FileNotFoundError(f"Palm directory not found: {palm_root}")
    if not os.path.isdir(vein_root):
        raise FileNotFoundError(f"Vein directory not found: {vein_root}")

    pairs = defaultdict(list)
    for subject in sorted(os.listdir(palm_root)):
        palm_dir = os.path.join(palm_root, subject)
        vein_dir = os.path.join(vein_root, subject)
        if not os.path.isdir(palm_dir) or not os.path.isdir(vein_dir):
            continue
        for name in _images(palm_dir):
            vein_path = os.path.join(vein_dir, name)
            if os.path.isfile(vein_path):
                class_id = f"{subject}_{name.split('_')[0]}"
                pairs[class_id].append((_rel(os.path.join(palm_dir, name)), _rel(vein_path)))
    return pairs


def _split_classes(class_ids: List[str], train_ratio: float, val_ratio: float, seed: int):
    rng = random.Random(seed)
    class_ids = list(class_ids)
    rng.shuffle(class_ids)
    n_train = max(1, int(len(class_ids) * train_ratio))
    n_val = max(1, int(len(class_ids) * val_ratio))
    if n_train + n_val >= len(class_ids):
        n_val = max(1, len(class_ids) - n_train - 1)
    train = sorted(class_ids[:n_train])
    val = sorted(class_ids[n_train:n_train + n_val])
    test = sorted(class_ids[n_train + n_val:]) or val
    return train, val, test


def _line(palm: str, vein: str, label: int, palm_exists: int, vein_exists: int, split: str) -> str:
    return f"{palm if palm_exists else NA_TOKEN} {vein if vein_exists else NA_TOKEN} {label} {palm_exists} {vein_exists} {split}\n"


def _full_protocol(pairs, class_ids, split):
    labels = {class_id: idx for idx, class_id in enumerate(class_ids)}
    return [
        _line(palm, vein, labels[class_id], 1, 1, split)
        for class_id in class_ids
        for palm, vein in sorted(pairs[class_id])
    ]


def _missing_protocol(pairs, class_ids, seed):
    labels = {class_id: idx for idx, class_id in enumerate(class_ids)}
    rng = random.Random(seed)
    lines = []
    for class_id in class_ids:
        for idx, (palm, vein) in enumerate(sorted(pairs[class_id])):
            palm_only = (idx + rng.randint(0, 1)) % 2 == 0
            split = "palm_only" if palm_only else "vein_only"
            lines.append(_line(palm, vein, labels[class_id], int(palm_only), int(not palm_only), split))
    return lines


def _test_protocol(pairs, class_ids, seed):
    labels = {class_id: idx for idx, class_id in enumerate(class_ids)}
    rng = random.Random(seed)
    lines = []
    for class_id in class_ids:
        for palm, vein in sorted(pairs[class_id]):
            label = labels[class_id]
            random_palm = rng.random() < 0.5
            lines += [
                _line(palm, vein, label, 1, 1, "full"),
                _line(palm, vein, label, 1, 0, "palm_only"),
                _line(palm, vein, label, 0, 1, "vein_only"),
                _line(palm, vein, label, int(random_palm), int(not random_palm), "random_missing"),
            ]
    return lines


def build_missing_protocols(
    root_dir: str,
    output_dir: str = "data_txt",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    palm_dir_name: str = "Red",
    vein_dir_name: str = "NIR",
):
    pairs = _collect_pairs(root_dir, palm_dir_name, vein_dir_name)
    class_ids = sorted(pairs)
    if not class_ids:
        raise RuntimeError(f"No paired samples found under: {root_dir}")

    train_ids, val_ids, test_ids = _split_classes(class_ids, train_ratio, val_ratio, seed)
    payload = {
        "train_full": _full_protocol(pairs, train_ids, "train"),
        "val_full": _full_protocol(pairs, val_ids, "val"),
        "val_missing_fixed": _missing_protocol(pairs, val_ids, seed + 7),
        "test_missing_protocol": _test_protocol(pairs, test_ids, seed + 13),
    }
    files = {key: os.path.join(output_dir, name) for key, name in PROTOCOL_FILES.items()}
    for key, lines in payload.items():
        _write(files[key], lines)

    return {
        "num_classes_total": len(class_ids),
        "num_classes_train": len(train_ids),
        "num_classes_val": len(val_ids),
        "num_classes_test": len(test_ids),
        "num_train_pairs": len(payload["train_full"]),
        "num_val_pairs": len(payload["val_full"]),
        "num_val_missing_pairs": len(payload["val_missing_fixed"]),
        "num_test_protocol_pairs": len(payload["test_missing_protocol"]),
        "files": {key: _path(path) for key, path in files.items()},
    }


def _read_protocol(list_file: str, split_filter: Optional[str] = None):
    samples = []
    with open(_path(list_file), "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            palm, vein, label, palm_exists, vein_exists, split = parts[:6]
            if split_filter is None or split == split_filter:
                samples.append({
                    "palm_path": palm,
                    "vein_path": vein,
                    "label": int(label),
                    "palm_exists": int(palm_exists),
                    "vein_exists": int(vein_exists),
                })
    return samples


def _load(path: str, exists: int, mode: str):
    return Image.open(_path(path)).convert(mode) if exists and path != NA_TOKEN else Image.new(mode, (224, 224), 0)


class MissingPairTxtDataset:
    def __init__(self, list_file: str, transform_palm=None, transform_vein=None, split_filter: Optional[str] = None):
        self.samples = _read_protocol(list_file, split_filter)
        self.transform_palm = transform_palm
        self.transform_vein = transform_vein

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        palm = _load(sample["palm_path"], sample["palm_exists"], "RGB")
        vein = _load(sample["vein_path"], sample["vein_exists"], "L")
        if self.transform_palm:
            palm = self.transform_palm(palm)
        if self.transform_vein:
            vein = self.transform_vein(vein)
        mask = torch.tensor([sample["palm_exists"], sample["vein_exists"]], dtype=torch.float32)
        return palm, vein, sample["label"], mask


class SingleModalityFromPairDataset:
    def __init__(self, list_file: str, modality: str, transform=None, split_filter: Optional[str] = None):
        if modality not in {"palm", "vein"}:
            raise ValueError(f"Unsupported modality: {modality}")
        self.modality = modality
        self.transform = transform
        self.samples = [
            sample for sample in _read_protocol(list_file, split_filter)
            if sample[f"{modality}_exists"] and sample[f"{modality}_path"] != NA_TOKEN
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        mode = "RGB" if self.modality == "palm" else "L"
        image = Image.open(_path(sample[f"{self.modality}_path"])).convert(mode)
        if self.transform:
            image = self.transform(image)
        return image, sample["label"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build missing-modality protocol files.")
    parser.add_argument("--root_dir", default="data")
    parser.add_argument("--output_dir", default="data_txt")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--palm_dir_name", default="Red")
    parser.add_argument("--vein_dir_name", default="NIR")
    return parser.parse_args()


def main():
    summary = build_missing_protocols(**vars(parse_args()))
    print("Missing-modality protocols generated.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
