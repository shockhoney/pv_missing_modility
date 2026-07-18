import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
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


def _collect_pairs(root_dir: str, palm_subdir: str, vein_subdir: str, expected: int):
    palm_root = _path(os.path.join(root_dir, palm_subdir))
    vein_root = _path(os.path.join(root_dir, vein_subdir))
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
        if len(samples) != expected:
            raise RuntimeError(f"Class {class_id} has {len(samples)} paired samples; expected exactly {expected}")
        pairs[class_id] = samples
    return pairs


def _cross_session_gallery_probe(gallery_samples, probe_samples, dataset, class_id, seed, config):
    gallery_split = _split_gallery_probe(gallery_samples, dataset, class_id, seed, config)
    probe_split = _split_gallery_probe(probe_samples, dataset, class_id, seed, config)
    return {"gallery": gallery_split["gallery"], "probe": probe_split["probe"]}


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




TONGJI_PROJECT_ROOT = Path(__file__).resolve().parents[1]
TONGJI_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
TONGJI_PROTOCOL_NAME = "tongji_session1_id_disjoint_closed_set_v1"
TONGJI_IDENTITY_COUNTS = {"train": 432, "val": 48, "test": 120}
TONGJI_SAMPLES_PER_IDENTITY = 10
TONGJI_GALLERY_PER_IDENTITY = 8
TONGJI_PROBE_PER_IDENTITY = 2
TONGJI_PROTOCOL_FILES = {
    "train": "ssfd_train_full.txt",
    "val_gallery": "ssfd_val_gallery_full.txt",
    "val_protocol": "ssfd_val_protocol.txt",
    "test_gallery": "ssfd_gallery_full.txt",
    "test_protocol": "ssfd_test_protocol.txt",
}
TONGJI_IDENTITY_FILES = {
    "train": "train_identities.txt",
    "val": "val_identities.txt",
    "test": "test_identities.txt",
}


def _tongji_resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (TONGJI_PROJECT_ROOT / path).resolve()


def _tongji_stored_path(path):
    path = path.resolve()
    try:
        return path.relative_to(TONGJI_PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _tongji_sample_key(path):
    stem = path.stem
    return stem[:-4] if stem.endswith("_roi") else stem


def _tongji_image_map(directory):
    images = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in TONGJI_IMAGE_EXTENSIONS
    ]
    mapping = {}
    for path in sorted(images):
        key = _tongji_sample_key(path)
        if key in mapping:
            raise RuntimeError(f"Duplicate sample key {key!r} in {directory}")
        mapping[key] = path
    return mapping


def _collect_tongji_session1_pairs(root_dir):
    root = _tongji_resolve(root_dir)
    palm_root = root / "palm_session1"
    vein_root = root / "vein_session1"
    if not palm_root.is_dir() or not vein_root.is_dir():
        raise FileNotFoundError(
            f"Tongji Session1 directories not found: {palm_root} and {vein_root}"
        )

    palm_ids = {path.name for path in palm_root.iterdir() if path.is_dir()}
    vein_ids = {path.name for path in vein_root.iterdir() if path.is_dir()}
    if palm_ids != vein_ids:
        missing_palm = sorted(vein_ids - palm_ids)
        missing_vein = sorted(palm_ids - vein_ids)
        raise RuntimeError(
            "Tongji Session1 modality identity sets differ: "
            f"missing palm={missing_palm[:5]}, missing vein={missing_vein[:5]}"
        )
    if len(palm_ids) != sum(TONGJI_IDENTITY_COUNTS.values()):
        raise RuntimeError(f"Tongji Session1 has {len(palm_ids)} identities; expected 600")

    pairs = {}
    for identity in sorted(palm_ids):
        palm = _tongji_image_map(palm_root / identity)
        vein = _tongji_image_map(vein_root / identity)
        if set(palm) != set(vein):
            raise RuntimeError(f"Tongji Session1 sample keys differ between modalities for {identity}")
        if len(palm) != TONGJI_SAMPLES_PER_IDENTITY:
            raise RuntimeError(
                f"Tongji Session1 identity {identity} has {len(palm)} pairs; expected 10"
            )
        pairs[identity] = [
            (key, _tongji_stored_path(palm[key]), _tongji_stored_path(vein[key]))
            for key in sorted(palm)
        ]
    return pairs


def _tongji_rng(seed, scope):
    digest = hashlib.sha256(f"{seed}:tongji:{scope}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _split_tongji_identities(identities, seed):
    shuffled = list(identities)
    _tongji_rng(seed, "session1-id-disjoint:identity-split").shuffle(shuffled)
    train_end = TONGJI_IDENTITY_COUNTS["train"]
    val_end = train_end + TONGJI_IDENTITY_COUNTS["val"]
    return {
        "train": sorted(shuffled[:train_end]),
        "val": sorted(shuffled[train_end:val_end]),
        "test": sorted(shuffled[val_end:]),
    }


def _split_tongji_gallery_probe(samples, identity, seed):
    shuffled = list(samples)
    _tongji_rng(seed, f"session1-id-disjoint:sample-split:{identity}").shuffle(shuffled)
    return (
        sorted(shuffled[:TONGJI_GALLERY_PER_IDENTITY]),
        sorted(shuffled[TONGJI_GALLERY_PER_IDENTITY:]),
    )


def _tongji_line(palm, vein, label, palm_exists, vein_exists, split):
    return (
        f"{palm if palm_exists else NA_TOKEN} "
        f"{vein if vein_exists else NA_TOKEN} "
        f"{label} {palm_exists} {vein_exists} {split}\n"
    )


def _tongji_complete_lines(pairs, identities, split):
    labels = {identity: label for label, identity in enumerate(identities)}
    return [
        _tongji_line(palm, vein, labels[identity], 1, 1, split)
        for identity in identities
        for _, palm, vein in pairs[identity]
    ]


def _tongji_probe_lines(pairs, identities):
    labels = {identity: label for label, identity in enumerate(identities)}
    lines = []
    for identity in identities:
        for _, palm, vein in pairs[identity]:
            label = labels[identity]
            lines.extend(
                [
                    _tongji_line(palm, vein, label, 1, 1, COMPLETE),
                    _tongji_line(palm, vein, label, 0, 1, PALMPRINT_MISSING),
                    _tongji_line(palm, vein, label, 1, 0, PALMVEIN_MISSING),
                ]
            )
    return lines


def _tongji_write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(lines), encoding="utf-8")
    temporary.replace(path)


def _tongji_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_tongji_identities(path):
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(values) != len(set(values)):
        raise RuntimeError(f"Duplicate identities in {path}")
    return values


def _read_tongji_records(path):
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 6:
            raise RuntimeError(f"{path}:{line_number} has {len(fields)} fields; expected 6")
        palm, vein, label, palm_exists, vein_exists, split = fields
        records.append(
            {
                "palm": palm,
                "vein": vein,
                "label": int(label),
                "palm_exists": int(palm_exists),
                "vein_exists": int(vein_exists),
                "split": split,
            }
        )
    return records


def _tongji_identity_and_key(record, palm_root, vein_root):
    available = []
    for modality, exists, root in (
        ("palm", record["palm_exists"], palm_root),
        ("vein", record["vein_exists"], vein_root),
    ):
        value = record[modality]
        if exists not in {0, 1}:
            raise RuntimeError(f"Invalid {modality} mask: {exists}")
        if exists == 0:
            if value != NA_TOKEN:
                raise RuntimeError(f"Missing {modality} must use {NA_TOKEN}")
            continue
        if value == NA_TOKEN:
            raise RuntimeError(f"Available {modality} cannot use {NA_TOKEN}")
        path = _tongji_resolve(value)
        if not path.is_file():
            raise FileNotFoundError(f"Protocol image does not exist: {path}")
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"Protocol path is outside Tongji Session1 {modality}: {path}"
            ) from error
        if len(relative.parts) != 2:
            raise RuntimeError(f"Unexpected Tongji Session1 path layout: {path}")
        available.append((relative.parts[0], _tongji_sample_key(path)))
    if not available:
        raise RuntimeError("A protocol row cannot have both modalities missing")
    if len(set(available)) != 1:
        raise RuntimeError(f"Palm and vein row pairing differs: {available}")
    return available[0]


def _validate_tongji_partition(
    records,
    identities,
    expected_per_identity,
    allowed_splits,
    palm_root,
    vein_root,
):
    identity_set = set(identities)
    expected_label_by_identity = {
        identity: label for label, identity in enumerate(identities)
    }
    expected_masks = {
        "train": (1, 1),
        "gallery": (1, 1),
        COMPLETE: (1, 1),
        PALMPRINT_MISSING: (0, 1),
        PALMVEIN_MISSING: (1, 0),
    }
    label_by_identity = {}
    samples = defaultdict(set)
    scenarios = defaultdict(list)
    for record in records:
        if record["split"] not in allowed_splits:
            raise RuntimeError(f"Unexpected protocol split/scenario: {record['split']}")
        actual_masks = (record["palm_exists"], record["vein_exists"])
        if actual_masks != expected_masks[record["split"]]:
            raise RuntimeError(
                f"Scenario {record['split']} has invalid modality masks {actual_masks}"
            )
        identity, key = _tongji_identity_and_key(record, palm_root, vein_root)
        if identity not in identity_set:
            raise RuntimeError(f"Identity {identity} occurs in the wrong partition")
        previous = label_by_identity.setdefault(identity, record["label"])
        if previous != record["label"]:
            raise RuntimeError(f"Identity {identity} has inconsistent labels")
        samples[identity].add(key)
        scenarios[(identity, key)].append(record["split"])

    if label_by_identity != expected_label_by_identity:
        raise RuntimeError("Partition identity-to-label mapping is incorrect")
    for identity in identities:
        if len(samples[identity]) != expected_per_identity:
            raise RuntimeError(
                f"Identity {identity} has {len(samples[identity])} logical samples; "
                f"expected {expected_per_identity}"
            )
    return samples, scenarios


def validate_tongji_session1_protocol(
    root_dir="data/tongji", output_dir="data_txt/tongji"
):
    root = _tongji_resolve(root_dir)
    output = _tongji_resolve(output_dir)
    palm_root = (root / "palm_session1").resolve()
    vein_root = (root / "vein_session1").resolve()
    identities = {
        name: _read_tongji_identities(output / filename)
        for name, filename in TONGJI_IDENTITY_FILES.items()
    }
    for name, expected in TONGJI_IDENTITY_COUNTS.items():
        if len(identities[name]) != expected:
            raise RuntimeError(
                f"{name} has {len(identities[name])} identities; expected {expected}"
            )
    if set(identities["train"]) & set(identities["val"]):
        raise RuntimeError("Train and validation identities overlap")
    if set(identities["train"]) & set(identities["test"]):
        raise RuntimeError("Train and test identities overlap")
    if set(identities["val"]) & set(identities["test"]):
        raise RuntimeError("Validation and test identities overlap")
    if len(set().union(*map(set, identities.values()))) != 600:
        raise RuntimeError("Train/validation/test identity union does not contain 600 identities")

    records = {
        name: _read_tongji_records(output / filename)
        for name, filename in TONGJI_PROTOCOL_FILES.items()
    }
    expected_rows = {
        "train": 4320,
        "val_gallery": 384,
        "val_protocol": 288,
        "test_gallery": 960,
        "test_protocol": 720,
    }
    for name, expected in expected_rows.items():
        if len(records[name]) != expected:
            raise RuntimeError(f"{name} has {len(records[name])} rows; expected {expected}")

    train_samples, train_scenarios = _validate_tongji_partition(
        records["train"], identities["train"], 10, {"train"}, palm_root, vein_root
    )
    val_gallery, val_gallery_scenarios = _validate_tongji_partition(
        records["val_gallery"],
        identities["val"],
        8,
        {"gallery"},
        palm_root,
        vein_root,
    )
    val_probe, val_probe_scenarios = _validate_tongji_partition(
        records["val_protocol"],
        identities["val"],
        2,
        {COMPLETE, PALMPRINT_MISSING, PALMVEIN_MISSING},
        palm_root,
        vein_root,
    )
    test_gallery, test_gallery_scenarios = _validate_tongji_partition(
        records["test_gallery"],
        identities["test"],
        8,
        {"gallery"},
        palm_root,
        vein_root,
    )
    test_probe, test_probe_scenarios = _validate_tongji_partition(
        records["test_protocol"],
        identities["test"],
        2,
        {COMPLETE, PALMPRINT_MISSING, PALMVEIN_MISSING},
        palm_root,
        vein_root,
    )

    for scenario_map in (
        train_scenarios,
        val_gallery_scenarios,
        test_gallery_scenarios,
    ):
        if any(len(values) != 1 for values in scenario_map.values()):
            raise RuntimeError("A train or Gallery sample occurs more than once")
    required_scenarios = Counter([COMPLETE, PALMPRINT_MISSING, PALMVEIN_MISSING])
    for scenario_map in (val_probe_scenarios, test_probe_scenarios):
        if any(Counter(values) != required_scenarios for values in scenario_map.values()):
            raise RuntimeError(
                "Each Probe pair must have exactly the three missing-modality scenarios"
            )
    for gallery, probe, split_name in (
        (val_gallery, val_probe, "validation"),
        (test_gallery, test_probe, "test"),
    ):
        if any(gallery[identity] & probe[identity] for identity in gallery):
            raise RuntimeError(f"{split_name} Gallery and Probe samples overlap")
    if sum(map(len, train_samples.values())) != 4320:
        raise RuntimeError("Training logical sample count is incorrect")

    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol") != TONGJI_PROTOCOL_NAME:
            raise RuntimeError("Protocol manifest name is incorrect")
        for filename, expected_hash in manifest.get("sha256", {}).items():
            if _tongji_sha256(output / filename) != expected_hash:
                raise RuntimeError(f"Protocol hash mismatch: {filename}")

    return {
        "protocol": TONGJI_PROTOCOL_NAME,
        "session": "session1_only",
        "num_classes_total": 600,
        "num_train_identities": 432,
        "num_val_identities": 48,
        "num_test_identities": 120,
        "num_train_pairs": 4320,
        "num_val_gallery_pairs": 384,
        "num_val_probe_pairs": 96,
        "num_test_gallery_pairs": 960,
        "num_test_probe_pairs": 240,
        "validation": "passed",
    }


def build_tongji_session1_protocol(
    root_dir="data/tongji", output_dir="data_txt/tongji", seed=2026
):
    pairs = _collect_tongji_session1_pairs(root_dir)
    identities = _split_tongji_identities(sorted(pairs), seed)
    validation_gallery, validation_probe = {}, {}
    test_gallery, test_probe = {}, {}
    for identity in identities["val"]:
        validation_gallery[identity], validation_probe[identity] = _split_tongji_gallery_probe(
            pairs[identity], identity, seed
        )
    for identity in identities["test"]:
        test_gallery[identity], test_probe[identity] = _split_tongji_gallery_probe(
            pairs[identity], identity, seed
        )

    payload = {
        "train": _tongji_complete_lines(pairs, identities["train"], "train"),
        "val_gallery": _tongji_complete_lines(
            validation_gallery, identities["val"], "gallery"
        ),
        "val_protocol": _tongji_probe_lines(validation_probe, identities["val"]),
        "test_gallery": _tongji_complete_lines(test_gallery, identities["test"], "gallery"),
        "test_protocol": _tongji_probe_lines(test_probe, identities["test"]),
    }
    output = _tongji_resolve(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, filename in TONGJI_PROTOCOL_FILES.items():
        _tongji_write_lines(output / filename, payload[name])
    for name, filename in TONGJI_IDENTITY_FILES.items():
        _tongji_write_lines(
            output / filename,
            [f"{identity}\n" for identity in identities[name]],
        )

    generated_files = [*TONGJI_PROTOCOL_FILES.values(), *TONGJI_IDENTITY_FILES.values()]
    manifest = {
        "protocol": TONGJI_PROTOCOL_NAME,
        "dataset": "tongji",
        "session": "session1_only",
        "seed": seed,
        "identity_counts": TONGJI_IDENTITY_COUNTS,
        "samples_per_identity": {
            "train": 10,
            "validation_gallery": 8,
            "validation_probe": 2,
            "test_gallery": 8,
            "test_probe": 2,
        },
        "sha256": {
            filename: _tongji_sha256(output / filename) for filename in generated_files
        },
    }
    _tongji_write_lines(
        output / "manifest.json",
        [json.dumps(manifest, indent=2, sort_keys=True) + "\n"],
    )
    summary = validate_tongji_session1_protocol(root_dir, output_dir)
    summary.update(
        {
            "dataset": "tongji",
            "seed": seed,
            "files": {
                name: str(output / filename)
                for name, filename in TONGJI_PROTOCOL_FILES.items()
            },
            "manifest": str(output / "manifest.json"),
        }
    )
    return summary


def build_ssfd_protocols(root_dir: str, output_dir: str, dataset: str, seed: int = 2026):
    dataset = dataset.lower()
    if dataset == "tongji":
        return build_tongji_session1_protocol(root_dir, output_dir, seed)

    config = DATASET_CONFIGS[dataset]
    expected_pairs = config["gallery_count"] + config["probe_count"]
    pairs = _collect_pairs(root_dir, config["palm_dir"], config["vein_dir"], expected_pairs)
    class_ids = sorted(pairs)
    if not class_ids:
        raise RuntimeError(f"No paired samples found under: {root_dir}")
    expected_classes = config["class_count"]
    if len(class_ids) != expected_classes:
        raise RuntimeError(f"{dataset} has {len(class_ids)} paired classes; expected {expected_classes}")

    train_ids, test_ids = _split_identity_ids(class_ids, dataset, seed, config)
    if "probe_palm_dir" in config:
        probe_source = _collect_pairs(
            root_dir,
            config["probe_palm_dir"],
            config["probe_vein_dir"],
            expected_pairs,
        )
        if set(probe_source) != set(pairs):
            raise RuntimeError(f"{dataset} Gallery and Probe sessions contain different identity sets")
        test_pairs = {
            class_id: _cross_session_gallery_probe(
                pairs[class_id], probe_source[class_id], dataset, class_id, seed, config
            )
            for class_id in test_ids
        }
    else:
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


class CrossModalRecoveryDataset:
    """Return real images and target-preprocessed cross-modal recovery inputs.

    A vein image is transformed with the palm pipeline before vein-to-palm
    recovery, while a palm image is transformed with the vein pipeline before
    palm-to-vein recovery. Missing paths remain black tensors and are excluded
    according to the availability mask by the caller.
    """

    def __init__(
        self,
        list_file: Optional[str],
        transform_palm,
        transform_vein,
        split_filter: Optional[str] = None,
        samples=None,
    ):
        if samples is not None and list_file is not None:
            raise ValueError("Pass either list_file or samples, not both")
        self.samples = (
            list(samples)
            if samples is not None
            else _read_protocol(list_file, split_filter)
        )
        self.transform_palm = transform_palm
        self.transform_vein = transform_vein

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        palm_image = _load(sample["palm_path"], sample["palm_exists"], "RGB")
        vein_image = _load(sample["vein_path"], sample["vein_exists"], "RGB")
        palm = self.transform_palm(palm_image)
        vein = self.transform_vein(vein_image)
        vein_as_palm = self.transform_palm(vein_image)
        palm_as_vein = self.transform_vein(palm_image)
        return {
            "palm": palm,
            "vein": vein,
            "vein_as_palm": vein_as_palm,
            "palm_as_vein": palm_as_vein,
            "label": sample["label"],
            "mask": torch.tensor(
                [sample["palm_exists"], sample["vein_exists"]],
                dtype=torch.float32,
            ),
        }


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
