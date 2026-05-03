import os
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
NA_TOKEN = "NA"


def _as_project_rel(path: str) -> str:
    return os.path.relpath(path, PROJECT_ROOT).replace("\\", "/")


def _as_abs_path(path: str) -> str:
    return os.path.join(PROJECT_ROOT, path.replace("\\", "/"))


def _blank_rgb() -> Image.Image:
    return Image.new("RGB", (224, 224), color=0)


def _blank_gray() -> Image.Image:
    return Image.new("L", (224, 224), color=0)


def _write_lines(path: str, lines: Iterable[str]) -> None:
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def gen_polyu_list(root_dir, out_txt="polyu_list.txt", train_ratio=0.8, val_ratio=0.1, seed=42):
    random.seed(seed)

    all_pids = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])
    pid2label = {pid: idx for idx, pid in enumerate(all_pids)}

    lines = []

    for pid in all_pids:
        person_dir = os.path.join(root_dir, pid)
        imgs = sorted([
            f for f in os.listdir(person_dir)
            if f.lower().endswith(IMAGE_EXTS)
        ])
        if not imgs:
            continue

        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        for i, img_name in enumerate(imgs):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"

            img_path = os.path.relpath(os.path.join(person_dir, img_name), PROJECT_ROOT).replace("\\", "/")
            label = pid2label[pid]

            lines.append(f"{img_path} {label} {split}\n")

    _write_lines(out_txt, lines)


def phase2_list(root_dir, train_txt, val_txt, val_ratio=0.2, seed=42):
    ir_dir = os.path.join(root_dir, "ir")
    vi_dir = os.path.join(root_dir, "vi")

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    pairs = []

    for name in sorted(os.listdir(ir_dir)):
        ir_path = os.path.join(ir_dir, name)
        if not os.path.isfile(ir_path):
            continue

        ext = os.path.splitext(name)[1].lower()
        if ext not in exts:
            continue

        vi_path = os.path.join(vi_dir, name)
        if not os.path.exists(vi_path):
            continue

        person_str = name.split("_")[0]
        label = int(person_str) - 1
        pairs.append(f"{_as_project_rel(ir_path)} {_as_project_rel(vi_path)} {label}\n")

    rng = random.Random(seed)
    rng.shuffle(pairs)

    n_total = len(pairs)
    n_val = max(1, int(n_total * val_ratio))
    train_pairs = pairs[n_val:]
    val_pairs = pairs[:n_val]

    _write_lines(train_txt, train_pairs)
    _write_lines(val_txt, val_pairs)


def _collect_polyu_pairs(
    root_dir: str,
    palm_dir_name: str = "Red",
    vein_dir_name: str = "NIR",
) -> Dict[str, List[Tuple[str, str]]]:
    palm_root = os.path.join(root_dir, palm_dir_name)
    vein_root = os.path.join(root_dir, vein_dir_name)

    if not os.path.isdir(palm_root):
        raise FileNotFoundError(f"Palm directory not found: {palm_root}")
    if not os.path.isdir(vein_root):
        raise FileNotFoundError(f"Vein directory not found: {vein_root}")

    pairs_by_class: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for subject in sorted(os.listdir(palm_root)):
        palm_subject_dir = os.path.join(palm_root, subject)
        vein_subject_dir = os.path.join(vein_root, subject)
        if not os.path.isdir(palm_subject_dir) or not os.path.isdir(vein_subject_dir):
            continue

        palm_files = sorted([
            f for f in os.listdir(palm_subject_dir)
            if f.lower().endswith(IMAGE_EXTS)
        ])

        for file_name in palm_files:
            palm_path = os.path.join(palm_subject_dir, file_name)
            vein_path = os.path.join(vein_subject_dir, file_name)
            if not os.path.isfile(vein_path):
                continue

            palm_instance = file_name.split("_")[0]
            class_id = f"{subject}_{palm_instance}"
            pairs_by_class[class_id].append((_as_project_rel(palm_path), _as_project_rel(vein_path)))

    return pairs_by_class


def _split_classes(class_ids: List[str], train_ratio: float, val_ratio: float, seed: int):
    rng = random.Random(seed)
    class_ids = list(class_ids)
    rng.shuffle(class_ids)

    total = len(class_ids)
    n_train = max(1, int(total * train_ratio))
    n_val = max(1, int(total * val_ratio))
    if n_train + n_val >= total:
        n_val = max(1, total - n_train - 1)
    train_ids = sorted(class_ids[:n_train])
    val_ids = sorted(class_ids[n_train:n_train + n_val])
    test_ids = sorted(class_ids[n_train + n_val:])
    if not test_ids:
        test_ids = val_ids
    return train_ids, val_ids, test_ids


def _format_missing_line(
    palm_path: str,
    vein_path: str,
    label: int,
    palm_exists: int,
    vein_exists: int,
    split_name: str,
) -> str:
    palm_value = palm_path if palm_exists else NA_TOKEN
    vein_value = vein_path if vein_exists else NA_TOKEN
    return f"{palm_value} {vein_value} {label} {palm_exists} {vein_exists} {split_name}\n"


def _build_full_lines(
    pairs_by_class: Dict[str, List[Tuple[str, str]]],
    class_ids: List[str],
    split_name: str,
) -> List[str]:
    label_map = {class_id: idx for idx, class_id in enumerate(class_ids)}
    lines: List[str] = []
    for class_id in class_ids:
        label = label_map[class_id]
        for palm_path, vein_path in sorted(pairs_by_class[class_id]):
            lines.append(_format_missing_line(palm_path, vein_path, label, 1, 1, split_name))
    return lines


def _build_missing_fixed_lines(
    pairs_by_class: Dict[str, List[Tuple[str, str]]],
    class_ids: List[str],
    seed: int,
) -> List[str]:
    label_map = {class_id: idx for idx, class_id in enumerate(class_ids)}
    rng = random.Random(seed)
    lines: List[str] = []

    for class_id in class_ids:
        label = label_map[class_id]
        class_pairs = sorted(pairs_by_class[class_id])
        for idx, (palm_path, vein_path) in enumerate(class_pairs):
            mode = "palm_only" if (idx + rng.randint(0, 1)) % 2 == 0 else "vein_only"
            if mode == "palm_only":
                lines.append(_format_missing_line(palm_path, vein_path, label, 1, 0, mode))
            else:
                lines.append(_format_missing_line(palm_path, vein_path, label, 0, 1, mode))
    return lines


def _build_test_protocol_lines(
    pairs_by_class: Dict[str, List[Tuple[str, str]]],
    class_ids: List[str],
    seed: int,
) -> List[str]:
    label_map = {class_id: idx for idx, class_id in enumerate(class_ids)}
    rng = random.Random(seed)
    lines: List[str] = []

    for class_id in class_ids:
        label = label_map[class_id]
        for palm_path, vein_path in sorted(pairs_by_class[class_id]):
            lines.append(_format_missing_line(palm_path, vein_path, label, 1, 1, "full"))
            lines.append(_format_missing_line(palm_path, vein_path, label, 1, 0, "palm_only"))
            lines.append(_format_missing_line(palm_path, vein_path, label, 0, 1, "vein_only"))
            if rng.random() < 0.5:
                lines.append(_format_missing_line(palm_path, vein_path, label, 1, 0, "random_missing"))
            else:
                lines.append(_format_missing_line(palm_path, vein_path, label, 0, 1, "random_missing"))
    return lines


def build_polyu_missing_protocols(
    root_dir: str,
    output_dir: str = "data_txt",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    palm_dir_name: str = "Red",
    vein_dir_name: str = "NIR",
):
    pairs_by_class = _collect_polyu_pairs(root_dir, palm_dir_name=palm_dir_name, vein_dir_name=vein_dir_name)
    class_ids = sorted(pairs_by_class.keys())
    if not class_ids:
        raise RuntimeError(f"No paired PolyU samples found under: {root_dir}")

    train_ids, val_ids, test_ids = _split_classes(class_ids, train_ratio, val_ratio, seed)
    output_dir_abs = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(output_dir_abs, exist_ok=True)

    train_full = _build_full_lines(pairs_by_class, train_ids, "train")
    val_full = _build_full_lines(pairs_by_class, val_ids, "val")
    val_missing = _build_missing_fixed_lines(pairs_by_class, val_ids, seed + 7)
    test_protocol = _build_test_protocol_lines(pairs_by_class, test_ids, seed + 13)

    output_files = {
        "train_full": os.path.join(output_dir_abs, "polyu_train_full.txt"),
        "val_full": os.path.join(output_dir_abs, "polyu_val_full.txt"),
        "val_missing_fixed": os.path.join(output_dir_abs, "polyu_val_missing_fixed.txt"),
        "test_missing_protocol": os.path.join(output_dir_abs, "polyu_test_missing_protocol.txt"),
    }

    _write_lines(output_files["train_full"], train_full)
    _write_lines(output_files["val_full"], val_full)
    _write_lines(output_files["val_missing_fixed"], val_missing)
    _write_lines(output_files["test_missing_protocol"], test_protocol)

    return {
        "num_classes_total": len(class_ids),
        "num_classes_train": len(train_ids),
        "num_classes_val": len(val_ids),
        "num_classes_test": len(test_ids),
        "num_train_pairs": len(train_full),
        "num_val_pairs": len(val_full),
        "num_val_missing_pairs": len(val_missing),
        "num_test_protocol_pairs": len(test_protocol),
        "files": output_files,
    }


class TxtImageDataset:
    def __init__(self, list_file, split="train", transform=None):
        self.samples = []
        self.transform = transform

        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 3:
                    continue

                img_path, label_str, split_str = parts
                if split_str != split:
                    continue

                label = int(label_str)
                self.samples.append((_as_abs_path(img_path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class PairTxtDataset:
    def __init__(self, list_file, transform_palm=None, transform_vein=None):
        self.samples = []
        self.transform_palm = transform_palm
        self.transform_vein = transform_vein

        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                ir_path, vi_path, label_str = parts[:3]

                palm_path = _as_abs_path(vi_path)
                vein_path = _as_abs_path(ir_path)
                label = int(label_str)
                self.samples.append((palm_path, vein_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        palm_path, vein_path, label = self.samples[idx]
        palm_img = Image.open(palm_path).convert("RGB")
        vein_img = Image.open(vein_path).convert("L")

        if self.transform_palm:
            palm_img = self.transform_palm(palm_img)
        if self.transform_vein:
            vein_img = self.transform_vein(vein_img)
        return palm_img, vein_img, label


class MissingPairTxtDataset:
    def __init__(
        self,
        list_file: str,
        transform_palm=None,
        transform_vein=None,
        split_filter: Optional[str] = None,
    ):
        self.samples = []
        self.transform_palm = transform_palm
        self.transform_vein = transform_vein
        self.split_filter = split_filter

        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue

                palm_path, vein_path, label_str, palm_exists_str, vein_exists_str, split_name = parts[:6]
                if split_filter is not None and split_name != split_filter:
                    continue

                self.samples.append({
                    "palm_path": palm_path,
                    "vein_path": vein_path,
                    "label": int(label_str),
                    "palm_exists": int(palm_exists_str),
                    "vein_exists": int(vein_exists_str),
                    "split": split_name,
                })

    def __len__(self):
        return len(self.samples)

    def _load_palm(self, sample):
        if sample["palm_exists"] and sample["palm_path"] != NA_TOKEN:
            img = Image.open(_as_abs_path(sample["palm_path"])).convert("RGB")
        else:
            img = _blank_rgb()
        if self.transform_palm:
            img = self.transform_palm(img)
        return img

    def _load_vein(self, sample):
        if sample["vein_exists"] and sample["vein_path"] != NA_TOKEN:
            img = Image.open(_as_abs_path(sample["vein_path"])).convert("L")
        else:
            img = _blank_gray()
        if self.transform_vein:
            img = self.transform_vein(img)
        return img

    def __getitem__(self, idx):
        import torch

        sample = self.samples[idx]
        palm_img = self._load_palm(sample)
        vein_img = self._load_vein(sample)
        mask = torch.tensor([sample["palm_exists"], sample["vein_exists"]], dtype=torch.float32)
        return palm_img, vein_img, sample["label"], mask


class SingleModalityFromPairDataset:
    def __init__(
        self,
        list_file: str,
        modality: str,
        transform=None,
        split_filter: Optional[str] = None,
    ):
        if modality not in {"palm", "vein"}:
            raise ValueError(f"Unsupported modality: {modality}")

        self.modality = modality
        self.transform = transform
        self.samples = []

        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue

                palm_path, vein_path, label_str, palm_exists_str, vein_exists_str, split_name = parts[:6]
                if split_filter is not None and split_name != split_filter:
                    continue

                sample = {
                    "label": int(label_str),
                    "split": split_name,
                }
                if modality == "palm":
                    if int(palm_exists_str) != 1 or palm_path == NA_TOKEN:
                        continue
                    sample["path"] = palm_path
                else:
                    if int(vein_exists_str) != 1 or vein_path == NA_TOKEN:
                        continue
                    sample["path"] = vein_path
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = _as_abs_path(sample["path"])
        if self.modality == "palm":
            image = Image.open(img_path).convert("RGB")
        else:
            image = Image.open(img_path).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        return image, sample["label"]


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
