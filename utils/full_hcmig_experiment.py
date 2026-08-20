"""Resumable two-stage image-level HCMIG experiment.

Stage 1 trains the hierarchical bidirectional image generators and PatchGAN
discriminators.  Stage 2 freezes every generation module and trains the two
pretrained ResNet-18 encoders plus MDSFF on paired identity labels.
Validation always uses the end-to-end complete/missing path and the
identity-disjoint gallery/probe protocol.
"""

from __future__ import annotations

import copy
import random
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import torch

from models.comparisons.hcmig import HCMIGAdapter
from utils.checkpoint_io import (
    file_sha256,
    safe_torch_load,
    save_checkpoint_atomic,
)
from utils.full_comparison_common import (
    SCENARIOS,
    evaluate_gallery_probe,
    labels_in_protocol,
    metric_rank,
    paired_loader,
)
from utils.runtime import resolve_device, set_random_seed


ARCHITECTURE_VERSION = "hcmig_tifs2025_resnet_mdsff_v2"
PAPER_LEARNING_RATE = 1e-4
PAPER_WEIGHT_DECAY = 5e-3
PAPER_BATCH_SIZE = 4
PAPER_DROPOUT = 0.5
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_normalized_to_domain(images: torch.Tensor) -> torch.Tensor:
    """Convert common-loader tensors to the generator's [-1, 1] domain."""

    if images.ndim != 4 or images.size(1) != 3:
        raise ValueError("images must have shape [batch, 3, height, width]")
    mean = images.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return ((images * std + mean).clamp(0.0, 1.0) * 2.0 - 1.0)


def _averages(totals: dict[str, float], batches: int) -> dict[str, float]:
    return {
        name: value / max(batches, 1)
        for name, value in totals.items()
    }


def _finite(loss: torch.Tensor, description: str) -> None:
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite {description}")


def _set_requires_grad(
    parameters: Iterable[torch.nn.Parameter], enabled: bool
) -> None:
    for parameter in parameters:
        parameter.requires_grad_(enabled)


def _clip(
    parameters: list[torch.nn.Parameter],
    scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    gradient_clip: float,
) -> None:
    if gradient_clip > 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)


def _batch_size(args) -> int:
    value = int(
        getattr(
            args,
            "batch_size",
            getattr(args, "hcmig_micro_batch", PAPER_BATCH_SIZE),
        )
    )
    if value <= 0:
        raise ValueError("HCMIG batch size must be positive")
    return value


def _micro_batch_size(args) -> int:
    value = int(getattr(args, "hcmig_micro_batch", _batch_size(args)))
    if value <= 0:
        raise ValueError("HCMIG micro-batch size must be positive")
    return value


def _eval_batch_size(args) -> int:
    value = int(
        getattr(
            args,
            "eval_batch_size",
            getattr(args, "hcmig_eval_batch", PAPER_BATCH_SIZE),
        )
    )
    if value <= 0:
        raise ValueError("HCMIG evaluation batch size must be positive")
    return value


def _output_dir(args) -> Path:
    value = getattr(args, "output_dir", None)
    if value is None:
        value = getattr(args, "save_dir", None)
    if value is None:
        raise ValueError("args.output_dir or args.save_dir is required")
    return Path(value)


def _paper_configuration(args) -> dict:
    return {
        "generator_epochs": int(
            getattr(args, "generator_epochs", getattr(args, "epochs", 1))
        ),
        "recognition_epochs": int(getattr(args, "epochs", 1)),
        "learning_rate": float(
            getattr(args, "hcmig_learning_rate", PAPER_LEARNING_RATE)
        ),
        "weight_decay": float(
            getattr(args, "hcmig_weight_decay", PAPER_WEIGHT_DECAY)
        ),
        "batch_size": _batch_size(args),
        "micro_batch_size": _micro_batch_size(args),
        "eval_batch_size": _eval_batch_size(args),
        "dropout": float(
            getattr(args, "hcmig_dropout", PAPER_DROPOUT)
        ),
        "generator_optimizer": str(
            getattr(args, "hcmig_generator_optimizer", "sgd")
        ).lower(),
        "discriminator_optimizer": str(
            getattr(args, "hcmig_discriminator_optimizer", "sgd")
        ).lower(),
        "recognition_optimizer": "sgd",
        "input_size": int(getattr(args, "input_size", 224)),
        "embedding_size": int(getattr(args, "embedding_size", 256)),
        "stochastic_eval": bool(
            getattr(args, "hcmig_stochastic_eval", True)
        ),
        "base_channels": int(
            getattr(args, "hcmig_base_channels", 64)
        ),
        "fft_radius_ratio": float(
            getattr(args, "hcmig_fft_radius_ratio", 0.1)
        ),
        "recognition_embedding_dim": getattr(
            args, "hcmig_recognition_embedding_dim", None
        ),
        "recognition_hidden_dim": getattr(
            args, "hcmig_recognition_hidden_dim", None
        ),
        "palm_checkpoint": str(
            getattr(args, "palm_ckpt", "outputs/encoders/palm_best.pth")
        ),
        "vein_checkpoint": str(
            getattr(args, "vein_ckpt", "outputs/encoders/vein_best.pth")
        ),
    }


def _validate_configuration(configuration: dict) -> None:
    for name in ("generator_epochs", "recognition_epochs"):
        if configuration[name] < 0:
            raise ValueError(f"{name} must be non-negative")
    for name in ("learning_rate", "weight_decay"):
        if configuration[name] < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if configuration["input_size"] < 32:
        raise ValueError("HCMIG input size must be at least 32")
    if configuration["input_size"] % 4:
        raise ValueError("HCMIG input size must be divisible by 4")
    if not 0.0 <= configuration["dropout"] < 1.0:
        raise ValueError("HCMIG dropout must be in [0, 1)")
    if configuration["base_channels"] <= 0:
        raise ValueError("HCMIG base_channels must be positive")
    if configuration["embedding_size"] <= 0 or configuration["embedding_size"] % 2:
        raise ValueError("HCMIG embedding_size must be positive and even")
    if not 0.0 < configuration["fft_radius_ratio"] < 0.5:
        raise ValueError("HCMIG fft_radius_ratio must be in (0, 0.5)")
    for key in ("generator_optimizer", "discriminator_optimizer"):
        if configuration[key] not in {"sgd", "adam"}:
            raise ValueError(f"{key} must be 'sgd' or 'adam'")


def _optional_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    source = Path(path)
    return file_sha256(source) if source.is_file() else None


def _fingerprints(args, configuration: dict) -> dict:
    source_path = getattr(args, "hcmig_source_path", "HCMIG.pdf")
    palm_fingerprint = _optional_sha256(configuration["palm_checkpoint"])
    vein_fingerprint = _optional_sha256(configuration["vein_checkpoint"])
    return {
        "train_list_sha256": file_sha256(args.train_list),
        "validation_gallery_sha256": file_sha256(
            args.val_gallery_list
        ),
        "validation_protocol_sha256": file_sha256(
            args.val_protocol_list
        ),
        "hcmig_pdf_sha256": _optional_sha256(source_path),
        "encoder_initialization": "project_resnet18_tongji",
        "palm_encoder_sha256": palm_fingerprint,
        "vein_encoder_sha256": vein_fingerprint,
    }


def _load_encoder_initialization(
    encoder, checkpoint_path: str | Path, modality: str
) -> None:
    """Strictly initialize one ResNet-18 encoder from its project checkpoint."""

    checkpoint = safe_torch_load(checkpoint_path, "cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Invalid encoder checkpoint: {checkpoint_path}")
    saved_modality = checkpoint.get("modality")
    if saved_modality is not None and str(saved_modality).lower() != modality:
        raise ValueError(
            f"Encoder modality mismatch in {checkpoint_path}: {saved_modality!r}"
        )
    state = checkpoint.get("encoder")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint has no encoder state: {checkpoint_path}")
    encoder.load_state_dict(state, strict=True)


def _initialize_encoders(
    model: HCMIGAdapter,
    configuration: dict,
) -> None:
    _load_encoder_initialization(
        model.palm_encoder, configuration["palm_checkpoint"], "palm"
    )
    _load_encoder_initialization(
        model.vein_encoder, configuration["vein_checkpoint"], "vein"
    )


def _optimizer(
    name: str,
    parameters: Iterable[torch.nn.Parameter],
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    parameters = list(parameters)
    if not parameters:
        raise ValueError("Cannot construct an optimizer without parameters")
    if name == "sgd":
        # The paper states SGD but does not report momentum, so none is added.
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if name == "adam":
        # Optional CycleGAN-compatible discriminator/generator fallback.
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            betas=(0.5, 0.999),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def _loaders(args, label_ids: list[int], configuration: dict):
    train_loader = paired_loader(
        args.train_list,
        input_size=configuration["input_size"],
        batch_size=configuration["batch_size"],
        num_workers=int(getattr(args, "num_workers", 0)),
        train=True,
        remap_labels=label_ids,
    )
    gallery_loader = paired_loader(
        args.val_gallery_list,
        input_size=configuration["input_size"],
        batch_size=configuration["eval_batch_size"],
        num_workers=int(getattr(args, "num_workers", 0)),
        train=False,
    )
    probes = {
        scenario: paired_loader(
            args.val_protocol_list,
            input_size=configuration["input_size"],
            batch_size=configuration["eval_batch_size"],
            num_workers=int(getattr(args, "num_workers", 0)),
            train=False,
            split_filter=scenario,
        )
        for scenario in SCENARIOS
    }
    return train_loader, gallery_loader, probes


def train_generation_epoch(
    model: HCMIGAdapter,
    loader,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    *,
    micro_batch_size: int,
    gradient_clip: float,
) -> dict[str, float]:
    """Train one alternating discriminator/generator epoch."""

    model.train()
    model.set_training_stage("generation")
    generator_parameters = list(model.generator_parameters())
    discriminator_parameters = list(model.discriminator_parameters())
    totals: dict[str, float] = defaultdict(float)
    batches = 0
    amp = device.type == "cuda"

    for palm, vein, _, _ in loader:
        palm_domain = imagenet_normalized_to_domain(
            palm.to(device, non_blocking=True)
        )
        vein_domain = imagenet_normalized_to_domain(
            vein.to(device, non_blocking=True)
        )
        batch_size = palm_domain.size(0)

        discriminator_optimizer.zero_grad(set_to_none=True)
        discriminator_values: dict[str, float] = defaultdict(float)
        for start in range(0, batch_size, micro_batch_size):
            stop = min(start + micro_batch_size, batch_size)
            fraction = (stop - start) / batch_size
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                losses = model.discriminator_loss_dict(
                    palm_domain[start:stop],
                    vein_domain[start:stop],
                )
                scaled_loss = losses["total"] * fraction
            _finite(scaled_loss, "HCMIG discriminator loss")
            scaler.scale(scaled_loss).backward()
            for name, value in losses.items():
                discriminator_values[name] += (
                    fraction * float(value.detach())
                )
        _clip(
            discriminator_parameters,
            scaler,
            discriminator_optimizer,
            gradient_clip,
        )
        scaler.step(discriminator_optimizer)
        scaler.update()

        _set_requires_grad(discriminator_parameters, False)
        generator_optimizer.zero_grad(set_to_none=True)
        generator_values: dict[str, float] = defaultdict(float)
        try:
            for start in range(0, batch_size, micro_batch_size):
                stop = min(start + micro_batch_size, batch_size)
                fraction = (stop - start) / batch_size
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp,
                ):
                    losses = model.generator_loss_dict(
                        palm_domain[start:stop],
                        vein_domain[start:stop],
                    )
                    scaled_loss = losses["total"] * fraction
                _finite(scaled_loss, "HCMIG generator loss")
                scaler.scale(scaled_loss).backward()
                for name, value in losses.items():
                    generator_values[name] += (
                        fraction * float(value.detach())
                    )
            _clip(
                generator_parameters,
                scaler,
                generator_optimizer,
                gradient_clip,
            )
            scaler.step(generator_optimizer)
            scaler.update()
        finally:
            _set_requires_grad(discriminator_parameters, True)

        for name, value in generator_values.items():
            totals[f"generator_{name}"] += value
        for name, value in discriminator_values.items():
            totals[f"discriminator_{name}"] += value
        batches += 1

    if batches == 0:
        raise ValueError(
            "HCMIG generation loader is empty; reduce the batch size"
        )
    return _averages(totals, batches)


def train_recognition_epoch(
    model: HCMIGAdapter,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    *,
    gradient_clip: float,
) -> dict[str, float]:
    """Train one MDSFF epoch on complete paired training samples."""

    model.train()
    model.set_training_stage("recognition")
    for module in model.generation_modules():
        module.eval()
    parameters = list(model.recognition_parameters())
    totals: dict[str, float] = defaultdict(float)
    batches = 0
    amp = device.type == "cuda"

    for palm, vein, labels, _ in loader:
        palm_domain = imagenet_normalized_to_domain(
            palm.to(device, non_blocking=True)
        )
        vein_domain = imagenet_normalized_to_domain(
            vein.to(device, non_blocking=True)
        )
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp,
        ):
            losses = model.recognition_loss_dict(
                palm_domain,
                vein_domain,
                labels,
                stochastic=True,
            )
            loss = losses["total"]
        _finite(loss, "HCMIG recognition loss")
        scaler.scale(loss).backward()
        _clip(parameters, scaler, optimizer, gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        for name, value in losses.items():
            totals[name] += float(value.detach())
        batches += 1

    if batches == 0:
        raise ValueError(
            "HCMIG recognition loader is empty; reduce the batch size"
        )
    return _averages(totals, batches)


def representation_callback(
    model: HCMIGAdapter,
    *,
    device: torch.device,
    stochastic: bool,
    seed: int,
):
    """Build a mask-aware complete/missing HCMIG representation callback."""

    sampling_generator = None
    if stochastic:
        sampling_generator = torch.Generator(device=device)
        sampling_generator.manual_seed(int(seed))

    def callback(
        palm: torch.Tensor,
        vein: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        if masks.ndim != 2 or masks.shape != (palm.size(0), 2):
            raise ValueError("masks must have shape [batch, 2]")
        palm_domain = imagenet_normalized_to_domain(palm)
        vein_domain = imagenet_normalized_to_domain(vein)
        palm_present = masks[:, 0].bool()
        vein_present = masks[:, 1].bool()
        if (~(palm_present | vein_present)).any():
            raise ValueError("HCMIG cannot recognize a sample with no modality")

        embeddings = palm.new_empty(
            palm.size(0), model.mdsff.embedding_dim
        )
        groups = (
            (palm_present & vein_present, "complete"),
            (palm_present & ~vein_present, "palm"),
            (~palm_present & vein_present, "vein"),
        )
        for selected, scenario in groups:
            if not selected.any():
                continue
            if scenario == "complete":
                output = model.recognition_forward(
                    palm_domain[selected],
                    vein_domain[selected],
                    stochastic=stochastic,
                    generator=sampling_generator,
                )
            elif scenario == "palm":
                output = model.recognize(
                    palm_domain=palm_domain[selected],
                    stochastic=stochastic,
                    generator=sampling_generator,
                )
            else:
                output = model.recognize(
                    vein_domain=vein_domain[selected],
                    stochastic=stochastic,
                    generator=sampling_generator,
                )
            embeddings[selected] = output["normalized_embedding"]
        return embeddings

    return callback


def evaluate(
    model: HCMIGAdapter,
    gallery_loader,
    probes,
    device: torch.device,
    *,
    stochastic: bool,
    seed: int,
) -> dict[str, dict]:
    """Evaluate with a freshly seeded sampler for epoch-comparable metrics."""

    model.eval()
    callback = representation_callback(
        model,
        device=device,
        stochastic=stochastic,
        seed=seed,
    )
    return evaluate_gallery_probe(
        gallery_loader,
        probes,
        callback,
        device,
    )


def _optimizer_states(optimizers: dict[str, torch.optim.Optimizer]) -> dict:
    return {
        name: optimizer.state_dict()
        for name, optimizer in optimizers.items()
    }


def _progress_payload(
    model: HCMIGAdapter,
    *,
    stage: str,
    epoch: int,
    histories: dict,
    optimizers: dict[str, torch.optim.Optimizer],
    best: dict,
    scaler: torch.amp.GradScaler,
    train_loader,
    configuration: dict,
    label_ids: list[int],
    fingerprints: dict,
    early_stopping: dict | None = None,
) -> dict:
    stopping = copy.deepcopy(early_stopping)
    return {
        "architecture_version": ARCHITECTURE_VERSION,
        "method": "hcmig",
        "stage": stage,
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizers": _optimizer_states(optimizers),
        "histories": copy.deepcopy(histories),
        "best": copy.deepcopy(best),
        "scaler": scaler.state_dict(),
        "configuration": copy.deepcopy(configuration),
        "label_ids": list(label_ids),
        "fingerprints": copy.deepcopy(fingerprints),
        "early_stopping": stopping,
        "stop_reason": (
            stopping.get("stop_reason") if isinstance(stopping, dict) else None
        ),
        "actual_epochs": (
            copy.deepcopy(stopping.get("actual_epochs"))
            if isinstance(stopping, dict)
            else {
                "generation": len(histories.get("generation", [])),
                "recognition": len(histories.get("recognition", [])),
            }
        ),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else []
        ),
        "python_rng_state": random.getstate(),
        "loader_rng_state": train_loader.generator.get_state(),
    }


def _restore_rng(payload: dict, train_loader, device: torch.device) -> None:
    torch.set_rng_state(payload["cpu_rng_state"])
    if device.type == "cuda" and payload.get("cuda_rng_state"):
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    if payload.get("python_rng_state") is not None:
        random.setstate(payload["python_rng_state"])
    if payload.get("loader_rng_state") is not None:
        train_loader.generator.set_state(payload["loader_rng_state"])


def _args_payload(args) -> dict:
    try:
        return copy.deepcopy(vars(args))
    except TypeError:
        if isinstance(args, SimpleNamespace):
            return copy.deepcopy(args.__dict__)
        raise


def _build_model(configuration: dict, num_classes: int) -> HCMIGAdapter:
    return HCMIGAdapter(
        base_channels=configuration["base_channels"],
        fft_radius_ratio=configuration["fft_radius_ratio"],
        num_classes=num_classes,
        recognition_embedding_size=configuration["embedding_size"],
        recognition_embedding_dim=configuration[
            "recognition_embedding_dim"
        ],
        recognition_hidden_dim=configuration[
            "recognition_hidden_dim"
        ],
        recognition_dropout=configuration["dropout"],
    )


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: torch.device | str,
) -> tuple[HCMIGAdapter, dict]:
    """Load a standalone best checkpoint for complete/missing inference."""

    payload = safe_torch_load(checkpoint_path, "cpu")
    if payload.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("Incompatible HCMIG checkpoint")
    configuration = payload.get("configuration")
    label_ids = payload.get("label_ids")
    if not isinstance(configuration, dict) or not isinstance(label_ids, list):
        raise ValueError("HCMIG checkpoint lacks configuration or label_ids")
    model = _build_model(configuration, len(label_ids))
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model, payload


def train(args) -> Path:
    """Train or resume both HCMIG stages and return the best checkpoint."""

    device = resolve_device(
        getattr(args, "device", None),
        require_available=True,
        announce=True,
    )
    set_random_seed(int(getattr(args, "seed", 42)))
    configuration = _paper_configuration(args)
    _validate_configuration(configuration)
    fingerprints = _fingerprints(args, configuration)
    output_dir = _output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pth"
    best_path = output_dir / "best.pth"

    label_ids = labels_in_protocol(args.train_list)
    train_loader, gallery_loader, probes = _loaders(
        args, label_ids, configuration
    )
    model = _build_model(configuration, len(label_ids)).to(device)
    _initialize_encoders(model, configuration)

    learning_rate = configuration["learning_rate"]
    weight_decay = configuration["weight_decay"]
    generator_optimizer = _optimizer(
        configuration["generator_optimizer"],
        model.generator_parameters(),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    discriminator_optimizer = _optimizer(
        configuration["discriminator_optimizer"],
        model.discriminator_parameters(),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    recognition_optimizer = _optimizer(
        "sgd",
        model.recognition_parameters(),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda"
    )
    histories: dict[str, list] = {
        "generation": [],
        "recognition": [],
    }
    best = {
        "rank": None,
        "epoch": None,
        "validation": None,
    }
    patience = int(getattr(args, "early_stopping_patience", 12))
    configured_minimum = getattr(args, "min_epochs", 6)
    minimum_epochs = max(6, int(6 if configured_minimum is None else configured_minimum))
    if patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    non_improving_validations = 0
    early_stopping = None
    stage, start_epoch = "generation", 1

    if last_path.is_file():
        payload = safe_torch_load(last_path, "cpu")
        if payload.get("architecture_version") != ARCHITECTURE_VERSION:
            raise ValueError("Incompatible HCMIG progress checkpoint")
        if payload.get("label_ids") != label_ids:
            raise ValueError("HCMIG progress checkpoint label set differs")
        if payload.get("configuration") != configuration:
            raise ValueError(
                "HCMIG progress checkpoint configuration differs"
            )
        if payload.get("fingerprints") != fingerprints:
            raise ValueError(
                "HCMIG progress checkpoint fingerprints differ"
            )
        model.load_state_dict(payload["model"], strict=True)
        histories = payload["histories"]
        best = payload["best"]
        early_stopping = payload.get("early_stopping")
        if isinstance(early_stopping, dict):
            non_improving_validations = int(
                early_stopping.get("non_improving_validations", 0)
            )
        elif best.get("epoch") is not None:
            non_improving_validations = sum(
                1
                for record in histories.get("recognition", [])
                if record.get("validation") is not None
                and int(record.get("epoch", 0)) > int(best["epoch"])
            )
        stage = str(payload["stage"])
        start_epoch = int(payload["epoch"]) + 1
        target_optimizers = (
            {
                "generator": generator_optimizer,
                "discriminator": discriminator_optimizer,
            }
            if stage == "generation"
            else {"recognition": recognition_optimizer}
        )
        for name, optimizer in target_optimizers.items():
            optimizer.load_state_dict(payload["optimizers"][name])
        scaler.load_state_dict(payload.get("scaler", {}))
        _restore_rng(payload, train_loader, device)
        print(
            f"[Resume] HCMIG stage={stage} epoch={start_epoch}",
            flush=True,
        )

    gradient_clip = float(getattr(args, "gradient_clip", 0.0))
    generation_stop = configuration["generator_epochs"]
    generation_cap = getattr(args, "hcmig_generation_epoch_cap", None)
    if generation_cap is not None:
        capped = min(generation_stop, int(generation_cap))
        if capped < generation_stop:
            print(
                f"[HCMIG generation-cap] capping generation stage at "
                f"epoch {capped} (configured {generation_stop})",
                flush=True,
            )
        generation_stop = capped
    if stage == "generation":
        generation_optimizers = {
            "generator": generator_optimizer,
            "discriminator": discriminator_optimizer,
        }
        for epoch in range(start_epoch, generation_stop + 1):
            started = time.time()
            losses = train_generation_epoch(
                model,
                train_loader,
                generator_optimizer,
                discriminator_optimizer,
                device,
                scaler,
                micro_batch_size=configuration["micro_batch_size"],
                gradient_clip=gradient_clip,
            )
            record = {
                "epoch": epoch,
                "losses": losses,
                "elapsed_seconds": time.time() - started,
            }
            histories["generation"].append(record)
            save_checkpoint_atomic(
                last_path,
                _progress_payload(
                    model,
                    stage="generation",
                    epoch=epoch,
                    histories=histories,
                    optimizers=generation_optimizers,
                    best=best,
                    scaler=scaler,
                    train_loader=train_loader,
                    configuration=configuration,
                    label_ids=label_ids,
                    fingerprints=fingerprints,
                ),
            )
            print(
                f"[HCMIG generation "
                f"{epoch:03d}/{generation_stop:03d}] "
                f"G={losses['generator_total']:.4f} "
                f"D={losses['discriminator_total']:.4f} "
                f"time={record['elapsed_seconds']:.1f}s",
                flush=True,
            )

        stage, start_epoch = "recognition", 1
        model.set_training_stage("recognition")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        save_checkpoint_atomic(
            last_path,
            _progress_payload(
                model,
                stage="recognition",
                epoch=0,
                histories=histories,
                optimizers={"recognition": recognition_optimizer},
                best=best,
                scaler=scaler,
                train_loader=train_loader,
                configuration=configuration,
                label_ids=label_ids,
                fingerprints=fingerprints,
            ),
        )

    model.set_training_stage("recognition")
    eval_every = max(1, int(getattr(args, "eval_every", 1)))
    if (
        stage == "recognition"
        and (
            (
                isinstance(early_stopping, dict)
                and bool(early_stopping.get("stopped"))
            )
            or (
                start_epoch - 1 >= minimum_epochs
                and non_improving_validations >= patience
            )
        )
    ):
        print("[HCMIG early-stop] progress already satisfies stopping rule", flush=True)
        if not best_path.is_file():
            raise RuntimeError("Stopped HCMIG progress has no best checkpoint")
        return best_path
    for epoch in range(
        start_epoch, configuration["recognition_epochs"] + 1
    ):
        started = time.time()
        losses = train_recognition_epoch(
            model,
            train_loader,
            recognition_optimizer,
            device,
            scaler,
            gradient_clip=gradient_clip,
        )
        should_evaluate = (
            epoch % eval_every == 0
            or epoch == configuration["recognition_epochs"]
        )
        validation = None
        if should_evaluate:
            validation = evaluate(
                model,
                gallery_loader,
                probes,
                device,
                stochastic=configuration["stochastic_eval"],
                seed=int(getattr(args, "seed", 42)) + 1009,
            )
        record = {
            "epoch": epoch,
            "losses": losses,
            "validation": validation,
            "elapsed_seconds": time.time() - started,
        }
        histories["recognition"].append(record)

        if validation is not None:
            rank = metric_rank(validation)
            improved = best["rank"] is None or rank < tuple(best["rank"])
            if improved:
                best = {
                    "rank": rank,
                    "epoch": epoch,
                    "validation": validation,
                }
                non_improving_validations = 0
                save_checkpoint_atomic(
                    best_path,
                    {
                        "architecture_version": ARCHITECTURE_VERSION,
                        "method": "hcmig",
                        "training_stage": (
                            "identity_validation_selection"
                        ),
                        "model": model.state_dict(),
                        "best_epoch": epoch,
                        "validation": validation,
                        "args": _args_payload(args),
                        "configuration": copy.deepcopy(configuration),
                        "label_ids": label_ids,
                        "fingerprints": copy.deepcopy(fingerprints),
                        "histories": copy.deepcopy(histories),
                        "optimizer_configuration": {
                            "generator": configuration[
                                "generator_optimizer"
                            ],
                            "discriminator": configuration[
                                "discriminator_optimizer"
                            ],
                            "recognition": "sgd",
                        },
                    },
                )
            else:
                non_improving_validations += 1
        else:
            improved = False
        should_stop = (
            validation is not None
            and epoch >= minimum_epochs
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
            "last_improved": bool(improved) if validation is not None else None,
            "stopped": bool(should_stop),
            "stop_reason": stop_reason,
            "actual_epochs": {
                "generation": len(histories["generation"]),
                "recognition": epoch,
            },
        }
        record["early_stopping"] = copy.deepcopy(early_stopping)

        save_checkpoint_atomic(
            last_path,
            _progress_payload(
                model,
                stage="recognition",
                epoch=epoch,
                histories=histories,
                optimizers={"recognition": recognition_optimizer},
                best=best,
                scaler=scaler,
                train_loader=train_loader,
                configuration=configuration,
                label_ids=label_ids,
                fingerprints=fingerprints,
                early_stopping=early_stopping,
            ),
        )
        eers = ""
        if validation is not None:
            eers = " ".join(
                f"{name}="
                f"{validation[name]['fused']['eer'] * 100:.3f}%"
                for name in SCENARIOS
            )
        print(
            f"[HCMIG recognition "
            f"{epoch:03d}/{configuration['recognition_epochs']:03d}] "
            f"loss={losses['total']:.4f} {eers} "
            f"time={record['elapsed_seconds']:.1f}s",
            flush=True,
        )
        if should_stop:
            print(
                f"[HCMIG early-stop] epoch={epoch} reason={stop_reason}",
                flush=True,
            )
            break

    if not best_path.is_file():
        raise RuntimeError(
            "No HCMIG best checkpoint was produced; "
            "recognition_epochs must include at least one validation"
        )
    return best_path


__all__ = [
    "ARCHITECTURE_VERSION",
    "evaluate",
    "imagenet_normalized_to_domain",
    "load_checkpoint_model",
    "representation_callback",
    "train",
    "train_generation_epoch",
    "train_recognition_epoch",
]
