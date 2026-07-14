import math
import random

import numpy as np
import torch
from torch.utils.data import DataLoader


def set_random_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def resolve_device(name=None, require_available=False, announce=False):
    requested = name or ("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        if require_available:
            raise RuntimeError("CUDA is not available. Check the PyTorch CUDA build and NVIDIA driver.")
        requested = "cpu"

    device = torch.device(requested)
    if announce:
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            print(f"[Info] using GPU: {torch.cuda.get_device_name(device)}")
        else:
            print("[Info] using CPU")
    return device


def cosine_annealing_lr(base_lr, min_lr, epochs, warmup_epochs, epoch):
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    span = max(1, epochs - warmup_epochs)
    step = max(0, epoch - warmup_epochs)
    scale = 0.5 * (1.0 + math.cos(math.pi * min(step, span) / span))
    return min_lr + (base_lr - min_lr) * scale


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr * group.get("lr_scale", 1.0)


def build_data_loader(dataset, batch_size, num_workers, train=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
