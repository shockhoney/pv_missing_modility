"""Resumable teacher/deployment image-level MMANet experiment."""

from __future__ import annotations

import copy
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import torch

from models.comparisons.mmanet_full import (
    ARCHITECTURE_VERSION,
    OFFICIAL_COMMIT,
    MMANetImageModel,
)
from utils.checkpoint_io import safe_torch_load, save_checkpoint_atomic
from utils.full_comparison_common import (
    SCENARIOS,
    evaluate_gallery_probe,
    labels_in_protocol,
    metric_rank,
    paired_loader,
)
from utils.runtime import resolve_device, set_random_seed


def _load_stem(stem, checkpoint_path: str) -> None:
    source = safe_torch_load(checkpoint_path, "cpu")["encoder"]
    translated = {}
    prefixes = {
        "backbone.conv1.": "network.0.",
        "backbone.bn1.": "network.1.",
        "backbone.layer1.": "network.4.",
        "backbone.layer2.": "network.5.",
    }
    for key, value in source.items():
        for old, new in prefixes.items():
            if key.startswith(old):
                target_key = new + key[len(old) :]
                # Project encoders use torchvision's 7x7 conv1, whereas the
                # released MMANet classification backbone uses 3x3. Preserve
                # its architecture and transfer the aligned central kernel.
                if (
                    target_key == "network.0.weight"
                    and value.ndim == 4
                    and tuple(value.shape[-2:]) == (7, 7)
                ):
                    value = value[:, :, 2:5, 2:5]
                translated[target_key] = value
                break
    missing, unexpected = stem.load_state_dict(translated, strict=False)
    allowed_missing = lambda key: (
        key.endswith("num_batches_tracked") or key.startswith("se_layer.")
    )
    if unexpected or any(not allowed_missing(key) for key in missing):
        raise ValueError(
            f"Could not initialize MMANet stem from {checkpoint_path}: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _initialize_network_stems(model, palm_checkpoint: str, vein_checkpoint: str) -> None:
    for network in (model.teacher, model.deployment):
        _load_stem(network.palm_stem, palm_checkpoint)
        _load_stem(network.vein_stem, vein_checkpoint)


def _lr(epoch: int, base: float, warmup: int = 5) -> float:
    if epoch <= warmup:
        return base * epoch / warmup
    factor = 1.0
    for milestone in (16, 33, 50):
        if epoch >= milestone:
            factor *= 0.1
    return base * factor


def _set_lr(optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def _loaders(args, label_ids):
    train_loader = paired_loader(
        args.train_list,
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=True,
        remap_labels=label_ids,
        seed=args.seed,
    )
    audit_loader = paired_loader(
        args.train_list,
        input_size=args.input_size,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        train=False,
        remap_labels=label_ids,
        seed=args.seed,
    )
    gallery_loader = paired_loader(
        args.val_gallery_list,
        input_size=args.input_size,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        train=False,
        seed=args.seed,
    )
    probes = {
        scenario: paired_loader(
            args.val_protocol_list,
            input_size=args.input_size,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            train=False,
            split_filter=scenario,
            seed=args.seed,
        )
        for scenario in SCENARIOS
    }
    return train_loader, audit_loader, gallery_loader, probes


def _average(totals, batches):
    return {name: value / max(1, batches) for name, value in totals.items()}


def _teacher_epoch(model, loader, optimizer, device, scaler, gradient_clip):
    model.teacher.train()
    totals = defaultdict(float)
    batches = 0
    amp = scaler.is_enabled()
    for palm, vein, labels, _ in loader:
        palm, vein, labels = (
            palm.to(device, non_blocking=True),
            vein.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output = model.teacher_loss(palm, vein, labels)
            loss = output["total"]
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite MMANet teacher loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.teacher.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        totals["total"] += float(loss.detach())
        batches += 1
    return _average(totals, batches)


def _deployment_epoch(model, loader, optimizer, device, scaler, epoch, gradient_clip):
    model.teacher.eval()
    model.deployment.train()
    model.regularizer.train()
    totals = defaultdict(float)
    batches = 0
    amp = scaler.is_enabled()
    for palm, vein, labels, _ in loader:
        palm, vein, labels = (
            palm.to(device, non_blocking=True),
            vein.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )
        codes = torch.randint(1, 4, (labels.numel(),), device=device)
        palm_present = (codes & 1).bool()
        vein_present = (codes & 2).bool()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            losses = model.deployment_loss(
                palm,
                vein,
                labels,
                palm_present=palm_present,
                vein_present=vein_present,
                epoch=epoch,
            )
            loss = losses["total"]
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite MMANet deployment loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        parameters = list(model.deployment.parameters()) + list(model.regularizer.parameters())
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        for name in ("task", "mad", "mar", "total"):
            totals[name] += float(losses[name].detach())
        batches += 1
    return _average(totals, batches)


@torch.inference_mode()
def _mar_histograms(model, loader, device):
    model.deployment.eval()
    histograms = torch.zeros(3, model.num_classes, dtype=torch.long, device=device)
    for palm, vein, _, _ in loader:
        palm = palm.to(device, non_blocking=True)
        vein = vein.to(device, non_blocking=True)
        outputs = (
            model.deployment(palm, vein, True, False),
            model.deployment(palm, vein, False, True),
            model.deployment(palm, vein, True, True),
        )
        for index, output in enumerate(outputs):
            predictions = output["logits"].argmax(dim=1)
            histograms[index] += torch.bincount(
                predictions, minlength=model.num_classes
            )
    return tuple(histograms)


def representation_callback(model: MMANetImageModel):
    def callback(palm, vein, masks):
        return model.representation(palm, vein, masks[:, 0], masks[:, 1])

    return callback


_representation_callback = representation_callback


def _progress(
    model,
    stage,
    epoch,
    histories,
    optimizer,
    best,
    scaler,
    loader,
    early_stopping=None,
):
    stopping = copy.deepcopy(early_stopping)
    return {
        "architecture_version": ARCHITECTURE_VERSION,
        "stage": stage,
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "histories": copy.deepcopy(histories),
        "best": copy.deepcopy(best),
        "early_stopping": stopping,
        "stop_reason": (
            stopping.get("stop_reason") if isinstance(stopping, dict) else None
        ),
        "actual_epochs": (
            copy.deepcopy(stopping.get("actual_epochs"))
            if isinstance(stopping, dict)
            else {
                "teacher": len(histories.get("teacher", [])),
                "deployment": len(histories.get("deployment", [])),
            }
        ),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "scaler": scaler.state_dict(),
        "loader_rng_state": loader.generator.get_state(),
    }


def train(args) -> Path:
    device = resolve_device(args.device, require_available=True, announce=True)
    set_random_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = output_dir / "last.pth", output_dir / "best.pth"
    label_ids = labels_in_protocol(args.train_list)
    train_loader, audit_loader, gallery_loader, probes = _loaders(args, label_ids)
    model = MMANetImageModel(
        num_classes=len(label_ids), mad_weight=30.0, mar_weight=0.5, warmup_epochs=5
    ).to(device)
    _initialize_network_stems(model, args.palm_ckpt, args.vein_ckpt)
    teacher_optimizer = torch.optim.SGD(
        model.teacher.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    deployment_parameters = list(model.deployment.parameters()) + list(model.regularizer.parameters())
    deployment_optimizer = torch.optim.SGD(
        deployment_parameters,
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    histories = {"teacher": [], "deployment": []}
    best = {"rank": None, "epoch": None, "validation": None}
    patience = int(getattr(args, "early_stopping_patience", 12))
    minimum_epochs = max(6, int(getattr(args, "min_epochs", 6)))
    if patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    non_improving_validations = 0
    early_stopping = None
    stage, start_epoch = "teacher", 1
    resume_scaler_state = None
    if last_path.is_file():
        payload = safe_torch_load(last_path, "cpu")
        if payload.get("architecture_version") != ARCHITECTURE_VERSION:
            raise ValueError("Incompatible MMANet progress checkpoint")
        model.load_state_dict(payload["model"], strict=True)
        stage, start_epoch = payload["stage"], int(payload["epoch"]) + 1
        histories, best = payload["histories"], payload["best"]
        early_stopping = payload.get("early_stopping")
        if isinstance(early_stopping, dict):
            non_improving_validations = int(
                early_stopping.get("non_improving_validations", 0)
            )
        elif best.get("epoch") is not None:
            non_improving_validations = sum(
                1
                for record in histories.get("deployment", [])
                if record.get("validation") is not None
                and int(record.get("epoch", 0)) > int(best["epoch"])
            )
        optimizer = teacher_optimizer if stage == "teacher" else deployment_optimizer
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["cpu_rng_state"])
        if device.type == "cuda" and payload.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        if payload.get("loader_rng_state") is not None:
            train_loader.generator.set_state(payload["loader_rng_state"])
        resume_scaler_state = payload.get("scaler")
        print(f"[Resume] MMANet stage={stage} epoch={start_epoch}", flush=True)

    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and bool(args.amp)
    )
    if resume_scaler_state:
        scaler.load_state_dict(resume_scaler_state)
    if stage == "teacher":
        for epoch in range(start_epoch, args.teacher_epochs + 1):
            started = time.time()
            learning_rate = _lr(epoch, args.learning_rate)
            _set_lr(teacher_optimizer, learning_rate)
            losses = _teacher_epoch(
                model, train_loader, teacher_optimizer, device, scaler, args.gradient_clip
            )
            histories["teacher"].append(
                {"epoch": epoch, "learning_rate": learning_rate, "losses": losses}
            )
            save_checkpoint_atomic(
                last_path,
                _progress(
                    model, "teacher", epoch, histories, teacher_optimizer, best, scaler, train_loader
                ),
            )
            print(
                f"[MMANet teacher {epoch:03d}/{args.teacher_epochs:03d}] "
                f"loss={losses['total']:.4f} time={time.time()-started:.1f}s",
                flush=True,
            )
        stage, start_epoch = "deployment", 1
        model.freeze_teacher()
        save_checkpoint_atomic(
            last_path,
            _progress(
                model, "deployment", 0, histories, deployment_optimizer, best, scaler, train_loader
            ),
        )

    model.freeze_teacher()
    if (
        stage == "deployment"
        and isinstance(early_stopping, dict)
        and early_stopping.get("stopped")
    ):
        print(
            "[MMANet early-stop] checkpoint already stopped: "
            f"{early_stopping.get('stop_reason')}",
            flush=True,
        )
        if not best_path.is_file():
            raise RuntimeError("Stopped MMANet progress has no best checkpoint")
        return best_path
    callback = representation_callback(model)
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        learning_rate = _lr(epoch, args.learning_rate)
        _set_lr(deployment_optimizer, learning_rate)
        losses = _deployment_epoch(
            model, train_loader, deployment_optimizer, device, scaler, epoch, args.gradient_clip
        )
        if epoch <= model.warmup_epochs and not bool(model.mar_observed[epoch - 1]):
            palm_hist, vein_hist, complete_hist = _mar_histograms(model, audit_loader, device)
            model.record_mar_epoch(epoch, palm_hist, vein_hist, complete_hist)
        model.eval()
        validation = evaluate_gallery_probe(gallery_loader, probes, callback, device)
        rank = metric_rank(validation)
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "losses": losses,
            "weak_modality": model.weak_modality,
            "validation": validation,
            "elapsed_seconds": time.time() - started,
        }
        histories["deployment"].append(record)
        improved = best["rank"] is None or rank < tuple(best["rank"])
        if improved:
            best = {"rank": rank, "epoch": epoch, "validation": validation}
            non_improving_validations = 0
            save_checkpoint_atomic(
                best_path,
                {
                    "architecture_version": ARCHITECTURE_VERSION,
                    "official_commit": OFFICIAL_COMMIT,
                    "method": "mmanet",
                    "model": model.state_dict(),
                    "best_epoch": epoch,
                    "validation": validation,
                    "args": vars(args),
                    "label_ids": label_ids,
                    "histories": histories,
                },
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
            "actual_epochs": {
                "teacher": len(histories["teacher"]),
                "deployment": epoch,
            },
        }
        record["early_stopping"] = copy.deepcopy(early_stopping)
        save_checkpoint_atomic(
            last_path,
            _progress(
                model,
                "deployment",
                epoch,
                histories,
                deployment_optimizer,
                best,
                scaler,
                train_loader,
                early_stopping,
            ),
        )
        eers = " ".join(
            f"{name}={validation[name]['fused']['eer']*100:.3f}%" for name in SCENARIOS
        )
        print(
            f"[MMANet deployment {epoch:03d}/{args.epochs:03d}] "
            f"loss={losses['total']:.4f} weak={model.weak_modality} {eers} "
            f"time={record['elapsed_seconds']:.1f}s",
            flush=True,
        )
        if should_stop:
            print(
                f"[MMANet early-stop] epoch={epoch} reason={stop_reason}",
                flush=True,
            )
            break
    return best_path


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[MMANetImageModel, dict]:
    """Strictly reconstruct teacher, deployment and MAR state for inference."""

    payload = safe_torch_load(checkpoint_path, "cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("Invalid MMANet checkpoint")
    if payload.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("Incompatible MMANet checkpoint")
    if payload.get("method") != "mmanet":
        raise ValueError("Checkpoint method is not MMANet")
    label_ids = payload.get("label_ids")
    state = payload.get("model")
    if not isinstance(label_ids, list) or not label_ids:
        raise ValueError("MMANet checkpoint lacks training label IDs")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("MMANet checkpoint lacks model state")
    model = MMANetImageModel(
        num_classes=len(label_ids),
        mad_weight=30.0,
        mar_weight=0.5,
        warmup_epochs=5,
    )
    model.load_state_dict(state, strict=True)
    model.freeze_teacher()
    model.to(torch.device(device)).eval()
    return model, dict(payload)


__all__ = [
    "load_checkpoint_model",
    "representation_callback",
    "train",
]
