import argparse
import hashlib
import os
import random
from typing import List, Optional, Tuple

import torch
from PIL import Image

from utils.scenarios import COMPLETE, PALMPRINT_MISSING, PALMVEIN_MISSING


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
NA_TOKEN = "NA"
SSFD_PROTOCOL_FILES = {
    "ssfd_train_full": "ssfd_train_full.txt",
    "ssfd_gallery_full": "ssfd_gallery_full.txt",
    "ssfd_test_protocol": "ssfd_test_protocol.txt",
}
DATASET_CONFIGS = {
    "tongji": {
        "root_dir": "data/tongji",
        "palm_dir": "palm_session1",
        "vein_dir": "vein_session1",
        "gallery_count": 8,
        "probe_count": 2,
        "class_count": 600,
        "session_split": False,
    },
    "cumt": {
        "root_dir": "data/CUMT",
        "palm_dir": "palmprint",
        "vein_dir": "palmvein",
        "gallery_count": 8,
        "probe_count": 2,
        "class_count": 290,
        "session_split": True,
    },
    "polyu": {
        "root_dir": "data/PolyU",
        "palm_dir": "Green",
        "vein_dir": "NIR",
        "gallery_count": 10,
        "probe_count": 2,
        "class_count": 500,
        "session_split": True,
    },
    "casia": {
        "root_dir": "data/CASIA",
        "palm_dir": "vi",
        "vein_dir": "ir",
        "gallery_count": 4,
        "probe_count": 2,
        "class_count": 200,
        "session_split": True,
    },
}


def _path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path.replace("\\", "/"))


def _rel(path: str) -> str:
    return os.path.relpath(path, PROJECT_ROOT).replace("\\", "/")


def _images(path: str) -> List[str]:
    return sorted(name for name in os.listdir(path) if name.lower().endswith(IMAGE_EXTS))


def _sample_key(name: str) -> str:
    stem = os.path.splitext(name)[0]
    return stem[:-4] if stem.endswith("_roi") else stem


def _write(path: str, lines: List[str]) -> None:
    path = _path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(lines)


def infer_num_classes(list_path: str) -> int:
    labels = set()
    with open(_path(list_path), "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 3:
                labels.add(int(parts[2]))
    return len(labels)


def _paired_samples(palm_dir: str, vein_dir: str) -> List[Tuple[str, str, str]]:
    palm = {_sample_key(name): os.path.join(palm_dir, name) for name in _images(palm_dir)}
    vein = {_sample_key(name): os.path.join(vein_dir, name) for name in _images(vein_dir)}
    return [(key, _rel(palm[key]), _rel(vein[key])) for key in sorted(palm.keys() & vein.keys())]


def _identity_rng(seed: int, dataset: str, class_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{dataset}:{class_id}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _session_key(dataset: str, sample_key: str) -> str:
    if dataset == "polyu":
        return sample_key.split("_", 1)[0]
    index = int(sample_key.rsplit("_", 1)[-1])
    group_size = 5 if dataset == "cumt" else 3
    offset = 0 if dataset == "cumt" else 1
    return str((index - offset) // group_size)


def _split_gallery_probe(samples, dataset: str, class_id: str, seed: int, config):
    gallery_count = config["gallery_count"]
    probe_count = config["probe_count"]
    needed = gallery_count + probe_count
    if len(samples) != needed:
        raise RuntimeError(f"Class {class_id} has {len(samples)} paired samples; expected exactly {needed}")

    rng = _identity_rng(seed, dataset, class_id)
    if not config["session_split"]:
        shuffled = list(samples)
        rng.shuffle(shuffled)
        return {"gallery": sorted(shuffled[:gallery_count]), "probe": sorted(shuffled[gallery_count:])}

    groups = {}
    for sample in samples:
        groups.setdefault(_session_key(dataset, sample[0]), []).append(sample)
    if len(groups) != probe_count:
        raise RuntimeError(f"Class {class_id} has {len(groups)} session groups; expected {probe_count}")

    gallery, probe = [], []
    for group in groups.values():
        rng.shuffle(group)
        probe.append(group[0])
        gallery.extend(group[1:])
    if len(gallery) != gallery_count:
        raise RuntimeError(f"Class {class_id} produced {len(gallery)} gallery pairs; expected {gallery_count}")
    return {"gallery": sorted(gallery), "probe": sorted(probe)}


def _collect_pairs(root_dir: str, dataset: str):
    config = DATASET_CONFIGS[dataset]
    palm_root = _path(os.path.join(root_dir, config["palm_dir"]))
    vein_root = _path(os.path.join(root_dir, config["vein_dir"]))
    if not os.path.isdir(palm_root):
        raise FileNotFoundError(f"Palm directory not found: {palm_root}")
    if not os.path.isdir(vein_root):
        raise FileNotFoundError(f"Vein directory not found: {vein_root}")

    pairs = {}
    for class_id in sorted(set(os.listdir(palm_root)) & set(os.listdir(vein_root))):
        palm_dir = os.path.join(palm_root, class_id)
        vein_dir = os.path.join(vein_root, class_id)
        if not os.path.isdir(palm_dir) or not os.path.isdir(vein_dir):
            continue
        samples = _paired_samples(palm_dir, vein_dir)
        expected = config["gallery_count"] + config["probe_count"]
        if len(samples) != expected:
            raise RuntimeError(f"Class {class_id} has {len(samples)} paired samples; expected exactly {expected}")
        pairs[class_id] = samples
    return pairs


def _split_identity_ids(class_ids, dataset: str, seed: int, config):
    train_fraction = config["gallery_count"] / (config["gallery_count"] + config["probe_count"])
    num_train = round(len(class_ids) * train_fraction)
    shuffled = list(class_ids)
    _identity_rng(seed, dataset, "identity-split").shuffle(shuffled)
    return sorted(shuffled[:num_train]), sorted(shuffled[num_train:])


def _line(palm: str, vein: str, label: int, palm_exists: int, vein_exists: int, split: str) -> str:
    return f"{palm if palm_exists else NA_TOKEN} {vein if vein_exists else NA_TOKEN} {label} {palm_exists} {vein_exists} {split}\n"


def _full_protocol(pairs, class_ids, split):
    labels = {class_id: idx for idx, class_id in enumerate(class_ids)}
    return [
        _line(palm, vein, labels[class_id], 1, 1, split)
        for class_id in class_ids
        for _, palm, vein in pairs[class_id]
    ]


def _ssfd_test_protocol(pairs, class_ids):
    labels = {class_id: idx for idx, class_id in enumerate(class_ids)}
    lines = []
    for class_id in class_ids:
        for _, palm, vein in pairs[class_id]:
            label = labels[class_id]
            lines += [
                _line(palm, vein, label, 1, 1, COMPLETE),
                _line(palm, vein, label, 0, 1, PALMPRINT_MISSING),
                _line(palm, vein, label, 1, 0, PALMVEIN_MISSING),
            ]
    return lines


def build_ssfd_protocols(root_dir: str, output_dir: str, dataset: str, seed: int = 2026):
    dataset = dataset.lower()
    config = DATASET_CONFIGS[dataset]
    pairs = _collect_pairs(root_dir, dataset)
    class_ids = sorted(pairs)
    if not class_ids:
        raise RuntimeError(f"No paired samples found under: {root_dir}")
    expected_classes = config["class_count"]
    if len(class_ids) != expected_classes:
        raise RuntimeError(f"{dataset} has {len(class_ids)} paired classes; expected {expected_classes}")

    train_ids, test_ids = _split_identity_ids(class_ids, dataset, seed, config)
    test_pairs = {
        class_id: _split_gallery_probe(pairs[class_id], dataset, class_id, seed, config)
        for class_id in test_ids
    }
    gallery_pairs = {class_id: test_pairs[class_id]["gallery"] for class_id in test_ids}
    probe_pairs = {class_id: test_pairs[class_id]["probe"] for class_id in test_ids}
    payload = {
        "ssfd_train_full": _full_protocol(pairs, train_ids, "train"),
        "ssfd_gallery_full": _full_protocol(gallery_pairs, test_ids, "gallery"),
        "ssfd_test_protocol": _ssfd_test_protocol(probe_pairs, test_ids),
    }
    files = {key: os.path.join(output_dir, name) for key, name in SSFD_PROTOCOL_FILES.items()}
    for key, lines in payload.items():
        _write(files[key], lines)

    return {
        "dataset": dataset.lower(),
        "seed": seed,
        "num_classes_total": len(class_ids),
        "num_train_identities": len(train_ids),
        "num_test_identities": len(test_ids),
        "num_train_pairs": len(payload["ssfd_train_full"]),
        "num_gallery_pairs": len(payload["ssfd_gallery_full"]),
        "num_probe_pairs": len(payload["ssfd_test_protocol"]) // 3,
        "num_test_protocol_pairs": len(payload["ssfd_test_protocol"]),
        "files": {key: _path(path) for key, path in files.items()},
    }


def build_all_protocols(output_root: str = "data_txt", seed: int = 2026):
    return {
        dataset: build_ssfd_protocols(
            config["root_dir"],
            os.path.join(output_root, dataset),
            dataset,
            seed,
        )
        for dataset, config in DATASET_CONFIGS.items()
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build paired closed-set missing-modality protocol files.")
    parser.add_argument("--dataset", choices=[*DATASET_CONFIGS, "all"], default="all")
    parser.add_argument("--root_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.dataset == "all":
        summaries = build_all_protocols(args.output_dir or "data_txt", args.seed)
    else:
        config = DATASET_CONFIGS[args.dataset]
        summaries = {
            args.dataset: build_ssfd_protocols(
                args.root_dir or config["root_dir"],
                args.output_dir or f"data_txt/{args.dataset}",
                args.dataset,
                args.seed,
            )
        }
    for dataset, summary in summaries.items():
        print(f"[{dataset.upper()}]")
        for key, value in summary.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
