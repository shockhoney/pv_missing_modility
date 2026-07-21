import hashlib
import os
from collections.abc import Mapping
from typing import Any

import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def tensor_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Pretrained checkpoint must be a state dict or contain one")

    for key in ("state_dict", "model", "encoder"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            checkpoint = value
            break

    state = {str(key): value for key, value in checkpoint.items() if torch.is_tensor(value)}
    if not state:
        raise TypeError("No tensor state dict found in pretrained checkpoint")
    return state


def remove_state_prefix(key: str, prefixes=("module.", "model.", "backbone.")) -> str:
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def save_checkpoint(path, payload):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(payload, path)


def save_checkpoint_atomic(path, payload):
    """Write a checkpoint completely before atomically replacing its target."""

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp-{os.getpid()}"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
