"""Resumable image-level DMRNet experiment for identity-disjoint validation.

This is the full end-to-end counterpart of the spatial-feature adapter. The
existing palmprint and palm-vein ResNet-18 checkpoints initialize two trainable
encoders; their final ``[B, 256, 7, 7]`` maps are passed to
:class:`DMRNetAdapter`. Each paired item is assigned exactly one uniformly
random non-empty modality combination per optimization step.
"""

from __future__ import annotations

import copy
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import torch
import torch.nn as nn

from models.backbones import build_encoder
from models.comparisons.dmrnet import DMRNetAdapter
from utils.checkpoint_io import file_sha256, safe_torch_load, save_checkpoint_atomic
from utils.full_comparison_common import (
    SCENARIOS,
    evaluate_gallery_probe,
    labels_in_protocol,
    metric_rank,
    paired_loader,
)
from utils.runtime import resolve_device, set_random_seed


ARCHITECTURE_VERSION: Final = "dmrnet_image_dual_resnet18_spatial_dmr_v1"
OFFICIAL_COMMIT: Final = "8e6c81f1f0dc9f009dcc216e7f8633f13520a993"
WARMUP_EPOCHS: Final = 5
LR_MILESTONES: Final = (16, 33, 50)
MOMENTUM: Final = 0.9
WEIGHT_DECAY: Final = 5e-4


class DMRNetImageModel(nn.Module):
    """Two trainable ResNet-18 encoders followed by spatial DMRNet."""

    def __init__(
        self,
        *,
        embedding_dim: int = 256,
        num_classes: int = 432,
        alpha: float = 1e-3,
        beta: float = 0.5,
        warmup_epochs: int = WARMUP_EPOCHS,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.palm_encoder = build_encoder(
            "palm", input_channel=3, embedding_size=self.embedding_dim
        )
        self.vein_encoder = build_encoder(
            "vein", input_channel=3, embedding_size=self.embedding_dim
        )
        self.dmrnet = DMRNetAdapter(
            input_channels=self.embedding_dim,
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
            alpha=alpha,
            beta=beta,
            hcr_top_v=2,
            hcr_warmup_epochs=warmup_epochs,
        )

    @staticmethod
    def _validate_images(
        palm: torch.Tensor, vein: torch.Tensor
    ) -> None:
        if palm.ndim != 4 or palm.size(1) != 3:
            raise ValueError("palm must have shape [B, 3, H, W]")
        if vein.ndim != 4 or vein.size(1) != 3:
            raise ValueError("vein must have shape [B, 3, H, W]")
        if palm.shape != vein.shape or palm.device != vein.device:
            raise ValueError("Palm and vein images must share shape and device")
        if not palm.is_floating_point() or not vein.is_floating_point():
            raise TypeError("Palm and vein images must be floating point")

    def feature_maps(
        self, palm: torch.Tensor, vein: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode both images without pooling, retaining end-to-end gradients."""

        self._validate_images(palm, vein)
        return (
            self.palm_encoder.forward_features(palm),
            self.vein_encoder.forward_features(vein),
        )

    def training_step(
        self,
        palm: torch.Tensor,
        vein: torch.Tensor,
        labels: torch.Tensor,
        *,
        epoch: int,
        generator: torch.Generator | None = None,
    ) -> dict:
        """Encode a paired batch once and run one random-combination DMR step."""

        palm_map, vein_map = self.feature_maps(palm, vein)
        return self.dmrnet.training_step(
            palm_map,
            vein_map,
            labels,
            epoch=epoch,
            generator=generator,
        )

    def forward(
        self,
        palm: torch.Tensor,
        vein: torch.Tensor,
        palm_present: bool | torch.Tensor = True,
        vein_present: bool | torch.Tensor = True,
        *,
        labels: torch.Tensor | None = None,
        epoch: int | None = None,
        sample: bool | None = None,
        update_hcr_stats: bool = True,
    ) -> dict:
        """Explicit-mask compatibility path; training should use ``training_step``."""

        palm_map, vein_map = self.feature_maps(palm, vein)
        return self.dmrnet(
            palm_map,
            vein_map,
            palm_present,
            vein_present,
            labels=labels,
            epoch=epoch,
            sample=sample,
            update_hcr_stats=update_hcr_stats,
        )

    def representation(
        self,
        palm: torch.Tensor,
        vein: torch.Tensor,
        palm_present: bool | torch.Tensor = True,
        vein_present: bool | torch.Tensor = True,
    ) -> torch.Tensor:
        """Return the normalized pooled mean ``mu`` used for matching."""

        palm_map, vein_map = self.feature_maps(palm, vein)
        return self.dmrnet.representation(
            palm_map,
            vein_map,
            palm_present,
            vein_present,
            sample=False,
        )


def _load_encoder_initialization(
    encoder: nn.Module, checkpoint_path: str | Path, modality: str
) -> None:
    """Strictly initialize one full project encoder from its existing checkpoint."""

    checkpoint = safe_torch_load(checkpoint_path, "cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Invalid encoder checkpoint: {checkpoint_path}")
    saved_modality = checkpoint.get("modality")
    if saved_modality is not None and str(saved_modality).lower() != modality:
        raise ValueError(
            f"Encoder modality mismatch in {checkpoint_path}: {saved_modality!r}"
        )
    state = checkpoint.get("encoder")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"Checkpoint has no encoder state: {checkpoint_path}")
    encoder.load_state_dict(state, strict=True)


def _initialize_encoders(
    model: DMRNetImageModel,
    palm_checkpoint: str | Path,
    vein_checkpoint: str | Path,
) -> None:
    _load_encoder_initialization(model.palm_encoder, palm_checkpoint, "palm")
    _load_encoder_initialization(model.vein_encoder, vein_checkpoint, "vein")


def _protocol_identity_set(path: str | Path) -> set[str]:
    """Read physical identity directory names, independent of remapped labels."""

    identities: set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.split()
            if len(parts) < 3:
                continue
            row_identities = {
                Path(value).parent.name for value in parts[:2] if value != "NA"
            }
            if len(row_identities) != 1:
                raise ValueError(
                    f"Inconsistent paired identity at {path}:{line_number}: "
                    f"{sorted(row_identities)}"
                )
            identities.update(row_identities)
    if not identities:
        raise ValueError(f"No physical identities found in {path}")
    return identities


def _validate_identity_disjoint_protocol(args) -> None:
    train_identities = _protocol_identity_set(args.train_list)
    gallery_identities = _protocol_identity_set(args.val_gallery_list)
    probe_identities = _protocol_identity_set(args.val_protocol_list)
    if gallery_identities != probe_identities:
        raise ValueError("Validation gallery and probe identity sets differ")
    overlap = train_identities & gallery_identities
    if overlap:
        preview = sorted(overlap)[:5]
        raise ValueError(
            f"Training and validation identities overlap: {preview}"
        )


def _loaders(args, label_ids: list[int]):
    _validate_identity_disjoint_protocol(args)
    train_loader = paired_loader(
        args.train_list,
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=True,
        remap_labels=label_ids,
    )
    gallery_loader = paired_loader(
        args.val_gallery_list,
        input_size=args.input_size,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        train=False,
    )
    probes = {
        scenario: paired_loader(
            args.val_protocol_list,
            input_size=args.input_size,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            train=False,
            split_filter=scenario,
        )
        for scenario in SCENARIOS
    }
    return train_loader, gallery_loader, probes


def representation_callback(model: DMRNetImageModel):
    def callback(
        palm: torch.Tensor, vein: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        return model.representation(palm, vein, masks[:, 0], masks[:, 1])

    return callback


_representation_callback = representation_callback

def _learning_rate(epoch: int, base: float) -> float:
    """Official five-epoch warm-up and 1/6, 2/6, 3/6 step schedule."""

    if epoch <= WARMUP_EPOCHS:
        return float(base) * epoch / WARMUP_EPOCHS
    decay_count = sum(epoch >= milestone for milestone in LR_MILESTONES)
    return float(base) * (0.1**decay_count)


def _set_lr(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def _averages(totals: dict[str, float], batches: int) -> dict[str, float]:
    return {name: value / max(1, batches) for name, value in totals.items()}


def _train_epoch(
    model: DMRNetImageModel,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    epoch: int,
    combination_generator: torch.Generator,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = defaultdict(float)
    batches = 0
    amp = device.type == "cuda"
    for palm, vein, labels, _ in loader:
        palm = palm.to(device, non_blocking=True)
        vein = vein.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp
        ):
            output = model.training_step(
                palm,
                vein,
                labels,
                epoch=epoch,
                generator=combination_generator,
            )
            losses = output["loss_dict"]
            loss = losses["total"]
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite DMRNet loss at epoch {epoch}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        for name in ("task", "kl", "hcr", "total"):
            totals[name] += float(losses[name].detach())
        batches += 1
    if batches == 0:
        raise ValueError("DMRNet training loader is empty")
    if epoch == WARMUP_EPOCHS and not bool(
        torch.all(model.dmrnet.combination_variance_element_count[1:] > 0)
    ):
        raise RuntimeError(
            "Warm-up ended before every non-empty modality combination was observed"
        )
    return _averages(totals, batches)


def _fingerprints(args) -> dict[str, str]:
    return {
        "train_list_sha256": file_sha256(args.train_list),
        "validation_gallery_sha256": file_sha256(args.val_gallery_list),
        "validation_protocol_sha256": file_sha256(args.val_protocol_list),
        "palm_encoder_sha256": file_sha256(args.palm_ckpt),
        "vein_encoder_sha256": file_sha256(args.vein_ckpt),
    }


def _loader_generator(loader) -> torch.Generator | None:
    generator = getattr(loader, "generator", None)
    if generator is None:
        generator = getattr(getattr(loader, "sampler", None), "generator", None)
    return generator


def _progress_payload(
    model: DMRNetImageModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    history: list[dict],
    best: dict,
    args,
    label_ids: list[int],
    fingerprints: dict[str, str],
    combination_generator: torch.Generator,
    train_loader,
    early_stopping: dict | None = None,
) -> dict:
    loader_generator = _loader_generator(train_loader)
    stopping = copy.deepcopy(early_stopping)
    return {
        "architecture_version": ARCHITECTURE_VERSION,
        "official_commit": OFFICIAL_COMMIT,
        "method": "dmrnet",
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "history": copy.deepcopy(history),
        "best": copy.deepcopy(best),
        "args": copy.deepcopy(vars(args)),
        "label_ids": list(label_ids),
        "fingerprints": dict(fingerprints),
        "early_stopping": stopping,
        "stop_reason": stopping.get("stop_reason") if stopping else None,
        "actual_epochs": (
            copy.deepcopy(stopping.get("actual_epochs"))
            if stopping
            else {"main": int(epoch)}
        ),
        "combination_rng_state": combination_generator.get_state(),
        "loader_rng_state": (
            loader_generator.get_state() if loader_generator is not None else None
        ),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def _restore_progress(
    payload: dict,
    model: DMRNetImageModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    combination_generator: torch.Generator,
    train_loader,
    *,
    label_ids: list[int],
    fingerprints: dict[str, str],
) -> tuple[int, list[dict], dict]:
    if payload.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("Incompatible DMRNet progress checkpoint")
    if payload.get("method") != "dmrnet":
        raise ValueError("Progress checkpoint method is not DMRNet")
    if payload.get("official_commit") != OFFICIAL_COMMIT:
        raise ValueError("Progress checkpoint official-source revision differs")
    if payload.get("fingerprints") != fingerprints:
        raise ValueError("DMRNet progress data or encoder fingerprints differ")
    if list(payload.get("label_ids", [])) != label_ids:
        raise ValueError("DMRNet progress training labels differ")

    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload.get("scaler", {}))
    combination_generator.set_state(payload["combination_rng_state"])
    loader_generator = _loader_generator(train_loader)
    loader_state = payload.get("loader_rng_state")
    if loader_generator is not None and loader_state is not None:
        loader_generator.set_state(loader_state)
    torch.set_rng_state(payload["cpu_rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_state"):
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    return (
        int(payload["epoch"]) + 1,
        list(payload.get("history", [])),
        dict(payload.get("best", {})),
    )


def _best_payload(
    model: DMRNetImageModel,
    epoch: int,
    validation: dict,
    history: list[dict],
    args,
    label_ids: list[int],
    fingerprints: dict[str, str],
) -> dict:
    return {
        "architecture_version": ARCHITECTURE_VERSION,
        "official_commit": OFFICIAL_COMMIT,
        "method": "dmrnet",
        "model": model.state_dict(),
        "best_epoch": int(epoch),
        "validation": copy.deepcopy(validation),
        "args": copy.deepcopy(vars(args)),
        "label_ids": list(label_ids),
        "fingerprints": dict(fingerprints),
        "history": copy.deepcopy(history),
        "hard_combination_codes": model.dmrnet.hard_combination_codes().cpu(),
        "variance_by_combination": model.dmrnet.variance_by_combination().cpu(),
    }


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[DMRNetImageModel, dict]:
    """Strictly restore the self-contained best checkpoint for inference."""

    target_device = torch.device(device)
    payload = safe_torch_load(checkpoint_path, "cpu")
    if not isinstance(payload, dict):
        raise TypeError("Invalid DMRNet best checkpoint")
    if payload.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("Incompatible DMRNet best checkpoint")
    if payload.get("official_commit") != OFFICIAL_COMMIT:
        raise ValueError("Best checkpoint official-source revision differs")
    if payload.get("method") != "dmrnet":
        raise ValueError("Best checkpoint method is not DMRNet")
    saved_args = payload.get("args")
    label_ids = payload.get("label_ids")
    state = payload.get("model")
    if not isinstance(saved_args, Mapping):
        raise ValueError("Best checkpoint has no training arguments")
    if not isinstance(label_ids, list) or not label_ids:
        raise ValueError("Best checkpoint has no training label IDs")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Best checkpoint has no model state")
    model = DMRNetImageModel(
        embedding_dim=int(saved_args["embedding_size"]),
        num_classes=len(label_ids),
        alpha=float(saved_args.get("alpha", 1e-3)),
        beta=float(saved_args.get("beta", 0.5)),
        warmup_epochs=WARMUP_EPOCHS,
    )
    model.load_state_dict(state, strict=True)
    model.to(target_device).eval()
    return model, payload


def train(args) -> Path:
    """Train or resume the full Tongji DMRNet validation-selection run."""

    device = resolve_device(args.device, require_available=True, announce=True)
    set_random_seed(args.seed)
    _validate_identity_disjoint_protocol(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = output_dir / "last.pth", output_dir / "best.pth"

    label_ids = labels_in_protocol(args.train_list)
    train_loader, gallery_loader, probes = _loaders(args, label_ids)
    fingerprints = _fingerprints(args)
    model = DMRNetImageModel(
        embedding_dim=args.embedding_size,
        num_classes=len(label_ids),
        alpha=float(getattr(args, "alpha", 1e-3)),
        beta=float(getattr(args, "beta", 0.5)),
        warmup_epochs=WARMUP_EPOCHS,
    ).to(device)
    _initialize_encoders(model, args.palm_ckpt, args.vein_ckpt)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    combination_generator = torch.Generator().manual_seed(args.seed + 1701)
    history: list[dict] = []
    best = {"rank": None, "epoch": None, "validation": None}
    patience = int(getattr(args, "early_stopping_patience", 12))
    configured_minimum = getattr(args, "min_epochs", 6)
    minimum_epochs = max(
        WARMUP_EPOCHS,
        6,
        int(6 if configured_minimum is None else configured_minimum),
    )
    if patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    non_improving_validations = 0
    early_stopping = None
    start_epoch = 1

    if last_path.is_file():
        payload = safe_torch_load(last_path, "cpu")
        start_epoch, history, best = _restore_progress(
            payload,
            model,
            optimizer,
            scaler,
            combination_generator,
            train_loader,
            label_ids=label_ids,
            fingerprints=fingerprints,
        )
        early_stopping = payload.get("early_stopping")
        if isinstance(early_stopping, dict):
            non_improving_validations = int(
                early_stopping.get("non_improving_validations", 0)
            )
        elif best.get("epoch") is not None:
            non_improving_validations = sum(
                1
                for record in history
                if record.get("validation") is not None
                and int(record.get("epoch", 0)) > int(best["epoch"])
            )
        print(f"[Resume] DMRNet epoch={start_epoch}", flush=True)

    already_stopped = (
        (
            isinstance(early_stopping, dict)
            and bool(early_stopping.get("stopped"))
        )
        or (
            start_epoch - 1 >= minimum_epochs
            and non_improving_validations >= patience
        )
    )
    if already_stopped:
        print("[DMRNet early-stop] progress already satisfies stopping rule", flush=True)
        if not best_path.is_file():
            raise RuntimeError("Stopped DMRNet progress has no best checkpoint")
        return best_path

    callback = representation_callback(model)
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        learning_rate = _learning_rate(epoch, args.learning_rate)
        _set_lr(optimizer, learning_rate)
        losses = _train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            epoch,
            combination_generator,
        )
        model.eval()
        validation = evaluate_gallery_probe(
            gallery_loader, probes, callback, device
        )
        rank = metric_rank(validation)
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "losses": losses,
            "hard_combination_codes": (
                model.dmrnet.hard_combination_codes().cpu().tolist()
            ),
            "variance_by_combination": (
                model.dmrnet.variance_by_combination().cpu().tolist()
            ),
            "validation": validation,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        improved = best.get("rank") is None or rank < tuple(best["rank"])
        if improved:
            best = {"rank": rank, "epoch": epoch, "validation": validation}
            non_improving_validations = 0
            save_checkpoint_atomic(
                best_path,
                _best_payload(
                    model,
                    epoch,
                    validation,
                    history,
                    args,
                    label_ids,
                    fingerprints,
                ),
            )
        else:
            non_improving_validations += 1
        should_stop = (
            epoch >= minimum_epochs
            and non_improving_validations >= patience
        )
        stop_reason = None
        if should_stop:
            stop_reason = (
                f"metric_rank did not strictly improve for {patience} "
                f"consecutive validation checks after best epoch {best['epoch']}"
            )
        early_stopping = {
            "enabled": True,
            "patience": patience,
            "min_epochs": minimum_epochs,
            "non_improving_validations": non_improving_validations,
            "last_improved": bool(improved),
            "stopped": bool(should_stop),
            "stop_reason": stop_reason,
            "actual_epochs": {"main": epoch},
        }
        record["early_stopping"] = copy.deepcopy(early_stopping)
        save_checkpoint_atomic(
            last_path,
            _progress_payload(
                model,
                optimizer,
                scaler,
                epoch,
                history,
                best,
                args,
                label_ids,
                fingerprints,
                combination_generator,
                train_loader,
                early_stopping,
            ),
        )
        eers = " ".join(
            f"{name}={validation[name]['fused']['eer']*100:.3f}%"
            for name in SCENARIOS
        )
        hard = record["hard_combination_codes"]
        print(
            f"[DMRNet {epoch:03d}/{args.epochs:03d}] "
            f"loss={losses['total']:.4f} lr={learning_rate:.6g} "
            f"hard={hard} {eers} time={record['elapsed_seconds']:.1f}s",
            flush=True,
        )
        if should_stop:
            print(
                f"[DMRNet early-stop] epoch={epoch} reason={stop_reason}",
                flush=True,
            )
            break
    return best_path


__all__ = [
    "ARCHITECTURE_VERSION",
    "DMRNetImageModel",
    "LR_MILESTONES",
    "OFFICIAL_COMMIT",
    "WARMUP_EPOCHS",
    "_initialize_encoders",
    "_learning_rate",
    "_representation_callback",
    "load_checkpoint_model",
    "representation_callback",
    "_restore_progress",
    "_train_epoch",
    "_validate_identity_disjoint_protocol",
    "train",
]
