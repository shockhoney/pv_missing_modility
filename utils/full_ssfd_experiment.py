"""Single-stage, resumable image-level reproduction of SSFD-Net on Tongji.

The paper's Stage-A VGG16 teacher pretraining is replaced by the project's
existing per-modality ResNet-18 checkpoints (``outputs/encoders/*_best.pth``,
trained by ``train_encoder.py`` on the same 432 Tongji identities). Their
frozen ArcFace identity heads serve as Dp/Dv for equations (8) and (10); the
encoder backbones are fine-tuned inside the SSFD shared/specific branches.

The main stage trains :class:`models.comparisons.ssfd.ResNetSSFDNet` with the
paper's equations (5)--(11) and selects the best checkpoint with the locked
validation protocol.

The module deliberately owns no dataset or evaluation implementation. It uses
``full_comparison_common.paired_loader`` and ``evaluate_gallery_probe`` so the
full baselines share the same identity-disjoint protocol and metric code.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.comparisons.ssfd import (
    ARCHITECTURE_VERSION as MODEL_ARCHITECTURE_VERSION,
    ResNetSSFDNet,
    ResNetSharedSpecificEncoder,
)
from utils.checkpoint_io import file_sha256, safe_torch_load, save_checkpoint_atomic
from utils.full_comparison_common import (
    evaluate_gallery_probe,
    labels_in_protocol,
    metric_rank,
    paired_loader,
)
from utils.head import ArcFace
from utils.runtime import set_random_seed
from utils.scenarios import PALMPRINT_MISSING, PALMVEIN_MISSING


EXPERIMENT_VERSION = "full_ssfd_resnet_tongji_v2"
METHOD = "ssfd"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RESUME_INVARIANTS = (
    "input_size",
    "batch_size",
    "eval_batch_size",
    "num_workers",
    "learning_rate",
    "weight_decay",
    "feature_dim",
    "embedding_size",
    "cmft_hidden_dim",
    "dropout",
    "triplet_margin",
    "palm_checkpoint",
    "vein_checkpoint",
    "palm_arcface_s",
    "palm_arcface_m",
    "vein_arcface_s",
    "vein_arcface_m",
    "share_cmft_weights",
    "eval_every",
    "seed",
    "max_steps_per_epoch",
)


@dataclass(slots=True)
class SSFDFullConfig:
    """Paper settings plus explicit controls required for resumable execution.

    ``feature_dim`` is one shared/specific part width; the underlying project
    encoder has ``embedding_size = 2 * feature_dim``.
    """

    train_list: str = "data_txt/tongji/ssfd_train_full.txt"
    val_gallery_list: str = "data_txt/tongji/ssfd_val_gallery_full.txt"
    val_protocol_list: str = "data_txt/tongji/ssfd_val_protocol.txt"
    output_dir: str = "outputs/gipssr/full_comparisons/tongji/seed_42/ssfd"
    input_size: int = 224
    batch_size: int = 32
    eval_batch_size: int = 32
    num_workers: int = 4
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.005
    feature_dim: int = 128
    embedding_size: int = 256
    cmft_hidden_dim: int = 512
    dropout: float = 0.5
    triplet_margin: float = 0.1  # The paper leaves m unspecified.
    palm_checkpoint: str = "outputs/encoders/palm_best.pth"
    vein_checkpoint: str = "outputs/encoders/vein_best.pth"
    palm_arcface_s: float = 32.0
    palm_arcface_m: float = 0.25
    vein_arcface_s: float = 32.0
    vein_arcface_m: float = 0.15
    share_cmft_weights: bool = True
    eval_every: int = 1
    seed: int = 42
    resume: bool = True
    max_steps_per_epoch: int | None = None
    early_stopping_patience: int = 12
    min_epochs: int = 6

    def validate(self) -> None:
        if self.input_size < 32:
            raise ValueError("input_size must be at least 32 for ResNet-18")
        for name in ("batch_size", "eval_batch_size", "epochs"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0 or self.eval_every < 0:
            raise ValueError("num_workers and eval_every must be non-negative")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.feature_dim <= 0 or self.cmft_hidden_dim <= 0:
            raise ValueError("feature_dim and cmft_hidden_dim must be positive")
        if self.embedding_size != 2 * self.feature_dim:
            raise ValueError("embedding_size must equal 2 * feature_dim")
        for value in (
            self.palm_arcface_s, self.vein_arcface_s,
            self.palm_arcface_m, self.vein_arcface_m,
        ):
            if not isinstance(value, (int, float)) or not value > 0:
                raise ValueError("arcface scale/margin values must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.triplet_margin < 0.0:
            raise ValueError("triplet_margin must be non-negative")
        if self.max_steps_per_epoch is not None and self.max_steps_per_epoch <= 0:
            raise ValueError("max_steps_per_epoch must be positive when set")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if self.min_epochs <= 0:
            raise ValueError("min_epochs must be positive")


def _device(value: torch.device | str) -> torch.device:
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _resolve_checkpoint(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def _config_payload(config: SSFDFullConfig) -> dict:
    payload = asdict(config)
    for key in ("palm_checkpoint", "vein_checkpoint"):
        payload[key] = str(_resolve_checkpoint(payload[key]))
    return payload


def _protocol_fingerprints(config: SSFDFullConfig) -> dict[str, str]:
    return {
        "train_list_sha256": file_sha256(config.train_list),
        "val_gallery_list_sha256": file_sha256(config.val_gallery_list),
        "val_protocol_list_sha256": file_sha256(config.val_protocol_list),
        "palm_checkpoint_sha256": file_sha256(
            _resolve_checkpoint(config.palm_checkpoint)
        ),
        "vein_checkpoint_sha256": file_sha256(
            _resolve_checkpoint(config.vein_checkpoint)
        ),
    }


def _checkpoint_path(config: SSFDFullConfig, name: str) -> Path:
    return Path(config.output_dir) / f"{name}.pth"


def _numpy_state_payload() -> dict:
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "name": name,
        "keys": torch.from_numpy(keys.copy()),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _capture_rng(loader: DataLoader) -> dict:
    payload = {
        "python": random.getstate(),
        "numpy": _numpy_state_payload(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda"] = torch.cuda.get_rng_state_all()
    generator = getattr(loader, "generator", None)
    if generator is not None:
        payload["loader_generator"] = generator.get_state()
    return payload


def _restore_rng(payload: dict | None, loader: DataLoader) -> None:
    if not payload:
        return
    random.setstate(payload["python"])
    numpy_state = payload["numpy"]
    np.random.set_state(
        (
            numpy_state["name"],
            numpy_state["keys"].cpu().numpy().astype(np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(payload["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in payload:
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda"]])
    generator = getattr(loader, "generator", None)
    if generator is not None and "loader_generator" in payload:
        generator.set_state(payload["loader_generator"].cpu())


def _validate_resume_checkpoint(
    checkpoint: dict,
    *,
    stage: str,
    config: SSFDFullConfig,
    label_ids: list[int],
) -> None:
    expected = {
        "experiment_version": EXPERIMENT_VERSION,
        "model_architecture_version": MODEL_ARCHITECTURE_VERSION,
        "method": METHOD,
        "stage": stage,
        "label_ids": label_ids,
        "fingerprints": _protocol_fingerprints(config),
    }
    differences = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    saved_config = checkpoint.get("config")
    current_config = _config_payload(config)
    if not isinstance(saved_config, dict):
        differences["config"] = (saved_config, "mapping")
    else:
        for key in _RESUME_INVARIANTS:
            if saved_config.get(key) != current_config.get(key):
                differences[f"config.{key}"] = (
                    saved_config.get(key), current_config.get(key)
                )
    if differences:
        raise ValueError(f"Incompatible SSFD resume checkpoint: {differences}")


def _train_loader(config: SSFDFullConfig, label_ids: list[int]) -> DataLoader:
    loader = paired_loader(
        config.train_list,
        input_size=config.input_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train=True,
        remap_labels=label_ids,
    )
    if loader.generator is not None:
        loader.generator.manual_seed(config.seed)
    return loader


def build_validation_loaders(
    config: SSFDFullConfig,
) -> tuple[DataLoader, dict[str, DataLoader]]:
    gallery = paired_loader(
        config.val_gallery_list,
        input_size=config.input_size,
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        train=False,
    )
    scenarios = {
        PALMPRINT_MISSING: paired_loader(
            config.val_protocol_list,
            input_size=config.input_size,
            batch_size=config.eval_batch_size,
            num_workers=config.num_workers,
            train=False,
            split_filter=PALMPRINT_MISSING,
        ),
        PALMVEIN_MISSING: paired_loader(
            config.val_protocol_list,
            input_size=config.input_size,
            batch_size=config.eval_batch_size,
            num_workers=config.num_workers,
            train=False,
            split_filter=PALMVEIN_MISSING,
        ),
    }
    return gallery, scenarios


def load_frozen_recognizer(
    config: SSFDFullConfig,
    *,
    modality: str,
    num_classes: int,
    device: torch.device | str,
) -> ArcFace:
    """Load a frozen per-modality ArcFace identity head (Dp or Dv).

    The head was trained by ``train_encoder.py`` on the same 432 Tongji
    identities; its cosine logits replace the paper's VGG16 Dp/Dv outputs in
    equations (8) and (10).
    """
    if modality not in {"palm", "vein"}:
        raise ValueError("modality must be palm or vein")
    checkpoint_path = _resolve_checkpoint(
        config.palm_checkpoint if modality == "palm" else config.vein_checkpoint
    )
    checkpoint = safe_torch_load(checkpoint_path, "cpu")
    saved_modality = str(checkpoint.get("modality", "")).lower()
    if saved_modality and saved_modality != modality:
        raise ValueError(
            f"Encoder modality mismatch in {checkpoint_path}: {saved_modality!r}"
        )
    saved_classes = int(checkpoint.get("num_classes", 0))
    if saved_classes != num_classes:
        raise ValueError(
            f"{modality} recognizer has {saved_classes} classes, "
            f"protocol requires {num_classes}"
        )
    state = checkpoint.get("classifier")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint has no classifier state: {checkpoint_path}")
    saved_args = checkpoint.get("args") or {}
    saved_s = float(saved_args.get("arcface_s", config.palm_arcface_s if modality == "palm" else config.vein_arcface_s))
    saved_m = float(saved_args.get("arcface_m", config.palm_arcface_m if modality == "palm" else config.vein_arcface_m))
    configured_s = config.palm_arcface_s if modality == "palm" else config.vein_arcface_s
    configured_m = config.palm_arcface_m if modality == "palm" else config.vein_arcface_m
    if saved_s != configured_s or saved_m != configured_m:
        raise ValueError(
            f"{modality} recognizer ArcFace (s={saved_s}, m={saved_m}) "
            f"does not match config (s={configured_s}, m={configured_m})"
        )
    in_features = int(saved_args.get("embedding_size", config.embedding_size))
    if in_features != config.embedding_size:
        raise ValueError(
            f"{modality} recognizer expects {in_features} features, "
            f"config uses {config.embedding_size}"
        )
    head = ArcFace(in_features, num_classes, configured_s, configured_m)
    head.load_state_dict(state, strict=True)
    head.eval().requires_grad_(False)
    return head.to(_device(device))


def load_shared_specific_encoder(
    config: SSFDFullConfig,
    *,
    modality: str,
    device: torch.device | str,
) -> ResNetSharedSpecificEncoder:
    """Build and strictly initialize one shared/specific ResNet-18 encoder."""
    if modality not in {"palm", "vein"}:
        raise ValueError("modality must be palm or vein")
    checkpoint_path = _resolve_checkpoint(
        config.palm_checkpoint if modality == "palm" else config.vein_checkpoint
    )
    checkpoint = safe_torch_load(checkpoint_path, "cpu")
    saved_modality = str(checkpoint.get("modality", "")).lower()
    if saved_modality and saved_modality != modality:
        raise ValueError(
            f"Encoder modality mismatch in {checkpoint_path}: {saved_modality!r}"
        )
    state = checkpoint.get("encoder")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint has no encoder state: {checkpoint_path}")
    encoder = ResNetSharedSpecificEncoder(
        config.embedding_size,
        use_se=(modality == "vein"),
    ).to(_device(device))
    encoder.load_encoder_state(state, strict=True)
    return encoder


def build_ssfd_model(
    config: SSFDFullConfig,
    *,
    num_classes: int,
    device: torch.device | str,
    palm_classifier: ArcFace | None = None,
    vein_classifier: ArcFace | None = None,
    label_ids: list[int] | None = None,
) -> ResNetSSFDNet:
    """Build the SSFD model with checkpoint-frozen Dp/Dv and pretrained encoders."""
    config.validate()
    target_device = _device(device)
    if label_ids is not None and len(label_ids) != num_classes:
        raise ValueError("num_classes must match len(label_ids)")
    palm_classifier = palm_classifier or load_frozen_recognizer(
        config,
        modality="palm",
        num_classes=num_classes,
        device=target_device,
    )
    vein_classifier = vein_classifier or load_frozen_recognizer(
        config,
        modality="vein",
        num_classes=num_classes,
        device=target_device,
    )
    palm_encoder = load_shared_specific_encoder(
        config, modality="palm", device=target_device
    )
    vein_encoder = load_shared_specific_encoder(
        config, modality="vein", device=target_device
    )
    return ResNetSSFDNet(
        num_classes=num_classes,
        embedding_size=config.embedding_size,
        cmft_hidden_dim=config.cmft_hidden_dim,
        dropout=config.dropout,
        triplet_margin=config.triplet_margin,
        palm_classifier=palm_classifier,
        vein_classifier=vein_classifier,
        palm_encoder=palm_encoder,
        vein_encoder=vein_encoder,
        share_cmft_weights=config.share_cmft_weights,
    ).to(target_device)


def reconstruct_model(
    config: SSFDFullConfig,
    *,
    num_classes: int,
    device: torch.device | str,
) -> ResNetSSFDNet:
    """Recreate the model topology without touching encoder checkpoint files.

    Every parameter is overwritten by ``load_state_dict`` immediately
    afterwards, so fresh initialization is sufficient for evaluation loading.
    """
    palm_head = ArcFace(
        config.embedding_size, num_classes,
        config.palm_arcface_s, config.palm_arcface_m,
    )
    vein_head = ArcFace(
        config.embedding_size, num_classes,
        config.vein_arcface_s, config.vein_arcface_m,
    )
    return ResNetSSFDNet(
        num_classes=num_classes,
        embedding_size=config.embedding_size,
        cmft_hidden_dim=config.cmft_hidden_dim,
        dropout=config.dropout,
        triplet_margin=config.triplet_margin,
        palm_classifier=palm_head,
        vein_classifier=vein_head,
        palm_encoder=ResNetSharedSpecificEncoder(config.embedding_size, use_se=False),
        vein_encoder=ResNetSharedSpecificEncoder(config.embedding_size, use_se=True),
        share_cmft_weights=config.share_cmft_weights,
    ).to(_device(device))


def make_representation_callback(
    model: ResNetSSFDNet,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return the common gallery/probe callback with per-sample mask routing."""

    def representation(
        palm: torch.Tensor,
        vein: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        if masks.ndim != 2 or masks.shape != (palm.size(0), 2):
            raise ValueError("masks must have shape [batch, 2]")
        if palm.size(0) != vein.size(0):
            raise ValueError("palm and vein batch sizes must match")
        present = masks.to(device=palm.device, dtype=torch.bool)
        palm_present, vein_present = present[:, 0], present[:, 1]
        if torch.any(~palm_present & ~vein_present):
            raise ValueError("SSFD cannot represent a sample with both modalities absent")
        output = palm.new_empty((palm.size(0), model.representation_dim))
        complete = palm_present & vein_present
        palm_only = palm_present & ~vein_present
        vein_only = ~palm_present & vein_present
        if torch.any(complete):
            indices = complete.nonzero(as_tuple=False).flatten()
            output.index_copy_(
                0,
                indices,
                model.complete_representation(palm[indices], vein[indices]),
            )
        if torch.any(palm_only):
            indices = palm_only.nonzero(as_tuple=False).flatten()
            output.index_copy_(
                0,
                indices,
                model.missing_representation(palm[indices], "palm"),
            )
        if torch.any(vein_only):
            indices = vein_only.nonzero(as_tuple=False).flatten()
            output.index_copy_(
                0,
                indices,
                model.missing_representation(vein[indices], "vein"),
            )
        return output

    return representation


def train_ssfd_epoch(
    model: ResNetSSFDNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str,
    max_steps: int | None = None,
) -> dict[str, float]:
    target_device = _device(device)
    model.train()
    totals: dict[str, float] = {}
    steps = 0
    for palm, vein, labels, masks in loader:
        if not torch.all(masks.bool().all(dim=1)):
            raise ValueError("SSFD main stage requires complete paired training data")
        palm = palm.to(target_device, non_blocking=True)
        vein = vein.to(target_device, non_blocking=True)
        labels = labels.to(target_device, dtype=torch.long, non_blocking=True)
        losses = model.loss_dict(palm, vein, labels)
        loss = losses["total"]
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite SSFD loss: {loss}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    if steps == 0:
        raise ValueError("SSFD Stage-B training loader produced no batches")
    return {**{name: value / steps for name, value in totals.items()}, "steps": float(steps)}


def _progress_payload(
    *,
    model: ResNetSSFDNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict],
    best_rank: tuple[float, ...] | None,
    config: SSFDFullConfig,
    label_ids: list[int],
    loader: DataLoader,
    early_stopping: dict | None = None,
) -> dict:
    stopping = copy.deepcopy(early_stopping)
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "model_architecture_version": MODEL_ARCHITECTURE_VERSION,
        "method": METHOD,
        "stage": "ssfd_full",
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history,
        "best_rank": list(best_rank) if best_rank is not None else None,
        "label_ids": label_ids,
        "config": _config_payload(config),
        "fingerprints": _protocol_fingerprints(config),
        "rng": _capture_rng(loader),
        "early_stopping": stopping,
        "stop_reason": stopping.get("stop_reason") if stopping else None,
        "actual_epochs": (
            copy.deepcopy(stopping.get("actual_epochs"))
            if stopping
            else {"main": int(epoch)}
        ),
    }


def train_stage(
    config: SSFDFullConfig,
    *,
    model: ResNetSSFDNet,
    label_ids: list[int],
    loader: DataLoader,
    device: torch.device | str,
    gallery_loader: DataLoader | None = None,
    scenario_loaders: dict[str, DataLoader] | None = None,
) -> tuple[ResNetSSFDNet, list[dict]]:
    """Run/resume the SSFD main stage and checkpoint every completed epoch."""
    target_device = _device(device)
    model.to(target_device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(
        parameters,
        lr=config.learning_rate,
        momentum=0.9,
        weight_decay=config.weight_decay,
    )
    last_path = _checkpoint_path(config, "last")
    best_path = _checkpoint_path(config, "best")
    start_epoch = 0
    history: list[dict] = []
    best_rank: tuple[float, ...] | None = None
    patience = int(getattr(config, "early_stopping_patience", 12))
    minimum_epochs = max(6, int(getattr(config, "min_epochs", 6)))
    non_improving_validations = 0
    early_stopping = None
    if patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if config.resume and last_path.is_file():
        checkpoint = safe_torch_load(last_path, target_device)
        _validate_resume_checkpoint(
            checkpoint,
            stage="ssfd_full",
            config=config,
            label_ids=label_ids,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        history = list(checkpoint.get("history", []))
        saved_rank = checkpoint.get("best_rank")
        best_rank = tuple(saved_rank) if saved_rank is not None else None
        early_stopping = checkpoint.get("early_stopping")
        if isinstance(early_stopping, dict):
            non_improving_validations = int(
                early_stopping.get("non_improving_validations", 0)
            )
        elif best_rank is not None:
            matching_epochs = [
                int(record["epoch"])
                for record in history
                if record.get("validation") is not None
                and metric_rank(record["validation"]) == best_rank
            ]
            if matching_epochs:
                last_best_epoch = max(matching_epochs)
                non_improving_validations = sum(
                    1
                    for record in history
                    if record.get("validation") is not None
                    and int(record.get("epoch", 0)) > last_best_epoch
                )
        _restore_rng(checkpoint.get("rng"), loader)
    if start_epoch > config.epochs:
        raise ValueError("Main-stage checkpoint epoch exceeds configured epochs")

    if isinstance(early_stopping, dict) and early_stopping.get("stopped"):
        print(
            "[SSFD early-stop] checkpoint already stopped: "
            f"{early_stopping.get('stop_reason')}",
            flush=True,
        )
        if not best_path.is_file():
            raise RuntimeError("Stopped SSFD progress has no best checkpoint")
        return model, history

    for epoch in range(start_epoch + 1, config.epochs + 1):
        started = time.monotonic()
        metrics = train_ssfd_epoch(
            model,
            loader,
            optimizer,
            device=target_device,
            max_steps=config.max_steps_per_epoch,
        )
        validation = None
        improved: bool | None = None
        should_evaluate = config.eval_every > 0 and epoch % config.eval_every == 0
        if should_evaluate:
            if gallery_loader is None or scenario_loaders is None:
                raise ValueError("Validation loaders are required when eval_every > 0")
            model.eval()
            validation = evaluate_gallery_probe(
                gallery_loader,
                scenario_loaders,
                make_representation_callback(model),
                target_device,
            )
            rank = metric_rank(validation)
            improved = best_rank is None or rank < best_rank
            if improved:
                best_rank = rank
                non_improving_validations = 0
                save_checkpoint_atomic(
                    str(best_path),
                    {
                        "experiment_version": EXPERIMENT_VERSION,
                        "model_architecture_version": MODEL_ARCHITECTURE_VERSION,
                        "method": METHOD,
                        "stage": "ssfd_full_best",
                        "epoch": epoch,
                        "best_epoch": epoch,
                        "model": model.state_dict(),
                        "validation": validation,
                        "rank": list(rank),
                        "label_ids": label_ids,
                        "config": _config_payload(config),
                        "fingerprints": _protocol_fingerprints(config),
                    },
                )
            else:
                non_improving_validations += 1
        should_stop = (
            validation is not None
            and epoch >= minimum_epochs
            and non_improving_validations >= patience
        )
        stop_reason = None
        if should_stop:
            stop_reason = (
                f"metric_rank did not strictly improve for {patience} "
                "consecutive validation checks"
            )
        early_stopping = {
            "enabled": True,
            "patience": patience,
            "min_epochs": minimum_epochs,
            "non_improving_validations": non_improving_validations,
            "last_improved": improved,
            "stopped": bool(should_stop),
            "stop_reason": stop_reason,
            "actual_epochs": {
                "main": epoch,
            },
        }
        record = {
            "epoch": epoch,
            "train": metrics,
            "validation": validation,
            "early_stopping": copy.deepcopy(early_stopping),
            "seconds": time.monotonic() - started,
        }
        history.append(record)
        validation_text = ""
        if validation is not None:
            validation_text = (
                f" pm_eer={validation[PALMPRINT_MISSING]['fused']['eer']:.6f}"
                f" vm_eer={validation[PALMVEIN_MISSING]['fused']['eer']:.6f}"
            )
        print(
            f"[SSFD main] epoch={epoch}/{config.epochs} "
            f"total={metrics['total']:.6f} cls={metrics['classification']:.6f} "
            f"tri={metrics['triplet']:.6f} trans={metrics['transformation']:.6f} "
            f"inter={metrics['inter_consistency']:.6f} "
            f"intra={metrics['intra_consistency']:.6f} "
            f"steps={int(metrics['steps'])}{validation_text} "
            f"seconds={record['seconds']:.1f}",
            flush=True,
        )
        save_checkpoint_atomic(
            str(last_path),
            _progress_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                history=history,
                best_rank=best_rank,
                config=config,
                label_ids=label_ids,
                loader=loader,
                early_stopping=early_stopping,
            ),
        )
        if should_stop:
            print(
                f"[SSFD early-stop] epoch={epoch} reason={stop_reason}",
                flush=True,
            )
            break
    return model, history


def train_ssfd(
    config: SSFDFullConfig,
    *,
    device: torch.device | str = "cuda",
) -> dict:
    """Run the single main stage and return the model plus auditable history.

    The paper's Stage-A Dp/Dv pretraining is intentionally replaced by the
    project's frozen per-modality ArcFace heads; only the SSFD main stage is
    trained.
    """
    config.validate()
    target_device = _device(device)
    set_random_seed(config.seed)
    label_ids = labels_in_protocol(config.train_list)
    num_classes = len(label_ids)

    model = build_ssfd_model(
        config,
        num_classes=num_classes,
        device=target_device,
        label_ids=label_ids,
    )
    training_loader = _train_loader(config, label_ids)
    gallery_loader = scenario_loaders = None
    if config.eval_every > 0:
        gallery_loader, scenario_loaders = build_validation_loaders(config)
    model, history = train_stage(
        config,
        model=model,
        label_ids=label_ids,
        loader=training_loader,
        device=target_device,
        gallery_loader=gallery_loader,
        scenario_loaders=scenario_loaders,
    )
    return {
        "model": model,
        "label_ids": label_ids,
        "history": history,
        "representation_callback": make_representation_callback(model),
    }


def load_trained_ssfd(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cuda",
) -> tuple[ResNetSSFDNet, SSFDFullConfig, list[int]]:
    """Reconstruct an evaluation model from one self-contained checkpoint file.

    No encoder checkpoint files or network downloads are needed: the frozen
    ArcFace Dp/Dv heads, both trainable encoders, CMFT, and the fusion
    classifier all live in ``model.state_dict``.
    """
    target_device = _device(device)
    checkpoint = safe_torch_load(str(checkpoint_path), target_device)
    expected = {
        "experiment_version": EXPERIMENT_VERSION,
        "model_architecture_version": MODEL_ARCHITECTURE_VERSION,
        "method": METHOD,
    }
    differences = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if checkpoint.get("stage") not in {"ssfd_full", "ssfd_full_best"}:
        differences["stage"] = (
            checkpoint.get("stage"), "ssfd_full or ssfd_full_best"
        )
    if differences:
        raise ValueError(f"Unsupported SSFD evaluation checkpoint: {differences}")
    saved_config = checkpoint.get("config")
    label_ids = checkpoint.get("label_ids")
    if not isinstance(saved_config, dict) or not isinstance(label_ids, list) or not label_ids:
        raise ValueError("SSFD checkpoint is missing config or label_ids")
    config = SSFDFullConfig(**saved_config)
    config.validate()
    num_classes = len(label_ids)
    model = reconstruct_model(config, num_classes=num_classes, device=target_device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, config, [int(label) for label in label_ids]


# Generic callbacks used by the full-comparison runner.
def build(
    config: SSFDFullConfig,
    *,
    device: torch.device | str = "cuda",
) -> ResNetSSFDNet:
    label_ids = labels_in_protocol(config.train_list)
    return build_ssfd_model(
        config,
        num_classes=len(label_ids),
        device=device,
        label_ids=label_ids,
    )


def train(
    config: SSFDFullConfig,
    *,
    device: torch.device | str = "cuda",
) -> dict:
    return train_ssfd(config, device=device)


def representation_callback(
    model: ResNetSSFDNet,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    return make_representation_callback(model)


__all__ = [
    "EXPERIMENT_VERSION",
    "METHOD",
    "SSFDFullConfig",
    "build",
    "build_ssfd_model",
    "build_validation_loaders",
    "load_frozen_recognizer",
    "load_shared_specific_encoder",
    "load_trained_ssfd",
    "make_representation_callback",
    "reconstruct_model",
    "representation_callback",
    "train",
    "train_ssfd",
    "train_ssfd_epoch",
    "train_stage",
]
