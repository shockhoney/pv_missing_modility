"""Resumable two-stage image-level SimMLM experiment."""

from __future__ import annotations

import copy
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import torch

from models.comparisons.simmlm_full import (
    ARCHITECTURE_VERSION,
    OFFICIAL_COMMIT,
    SimMLMImageModel,
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


def _averages(totals: dict[str, float], count: int) -> dict[str, float]:
    return {name: value / max(count, 1) for name, value in totals.items()}


def _load_expert_initialization(expert, checkpoint_path: str, num_classes: int) -> None:
    checkpoint = safe_torch_load(checkpoint_path, "cpu")
    if int(checkpoint.get("num_classes", -1)) != int(num_classes):
        raise ValueError(f"Expert class count mismatch in {checkpoint_path}")
    expert.encoder.load_state_dict(checkpoint["encoder"], strict=True)
    weight = checkpoint.get("classifier", {}).get("weight")
    if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != (
        num_classes,
        expert.classifier.in_features,
    ):
        raise ValueError(f"Invalid ArcFace initialization in {checkpoint_path}")
    with torch.no_grad():
        expert.classifier.weight.copy_(weight)
        expert.classifier.bias.zero_()


def _loaders(args, label_ids: list[int]):
    train_loader = paired_loader(
        args.train_list,
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=True,
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
    return train_loader, gallery_loader, probes


def representation_callback(model: SimMLMImageModel):
    def callback(palm: torch.Tensor, vein: torch.Tensor, masks: torch.Tensor):
        return model.representation(palm, vein, masks[:, 0], masks[:, 1])

    return callback


_representation_callback = representation_callback


def _train_experts_epoch(model, loader, optimizers, device, scaler, gradient_clip):
    model.train()
    totals: dict[str, float] = defaultdict(float)
    batches = 0
    amp = scaler.is_enabled()
    for palm, vein, labels, _ in loader:
        palm = palm.to(device, non_blocking=True)
        vein = vein.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            palm_loss = model.expert_loss("palm", palm, labels)["total"]
            vein_loss = model.expert_loss("vein", vein, labels)["total"]
            loss = palm_loss + vein_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite SimMLM expert loss")
        scaler.scale(loss).backward()
        for optimizer in optimizers:
            scaler.unscale_(optimizer)
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        for optimizer in optimizers:
            scaler.step(optimizer)
        scaler.update()
        totals["palm_expert"] += float(palm_loss.detach())
        totals["vein_expert"] += float(vein_loss.detach())
        totals["total"] += float(loss.detach())
        batches += 1
    return _averages(totals, batches)


def _train_cooperative_epoch(model, loader, optimizer, device, scaler, gradient_clip):
    model.train()
    totals: dict[str, float] = defaultdict(float)
    batches = 0
    amp = scaler.is_enabled()
    for palm, vein, labels, _ in loader:
        palm = palm.to(device, non_blocking=True)
        vein = vein.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        fewer_is_palm = torch.rand(labels.numel(), device=device) < 0.5
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            losses = model.cooperative_loss(
                palm, vein, labels, fewer_is_palm=fewer_is_palm
            )
            loss = losses["total"]
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite SimMLM cooperative loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        for name in ("task_more", "task_fewer", "mofe", "total"):
            totals[name] += float(losses[name].detach())
        batches += 1
    return _averages(totals, batches)


def _progress_payload(
    model,
    stage,
    epoch,
    histories,
    optimizers,
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
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "histories": copy.deepcopy(histories),
        "best": copy.deepcopy(best),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "scaler": scaler.state_dict(),
        "loader_rng_state": loader.generator.get_state(),
        "early_stopping": stopping,
        "stop_reason": (
            stopping.get("stop_reason") if isinstance(stopping, dict) else None
        ),
        "actual_epochs": (
            copy.deepcopy(stopping.get("actual_epochs"))
            if isinstance(stopping, dict)
            else {
                "expert": len(histories.get("expert", [])),
                "cooperative": len(histories.get("cooperative", [])),
            }
        ),
    }


def train(args) -> Path:
    device = resolve_device(args.device, require_available=True, announce=True)
    set_random_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pth"
    best_path = output_dir / "best.pth"
    label_ids = labels_in_protocol(args.train_list)
    train_loader, gallery_loader, probes = _loaders(args, label_ids)
    model = SimMLMImageModel(
        embedding_dim=args.embedding_size,
        num_classes=len(label_ids),
        ranking_weight=0.1,
    ).to(device)
    histories: dict[str, list] = {"expert": [], "cooperative": []}
    best = {"rank": None, "epoch": None, "validation": None}
    patience = int(getattr(args, "early_stopping_patience", 12))
    configured_minimum = getattr(args, "min_epochs", 6)
    minimum_epochs = max(6, int(6 if configured_minimum is None else configured_minimum))
    if patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    non_improving_validations = 0
    early_stopping = None

    expert_optimizers = [
        torch.optim.Adam(model.palm_expert.parameters(), lr=args.expert_lr),
        torch.optim.Adam(model.vein_expert.parameters(), lr=args.expert_lr),
    ]
    cooperative_optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    stage, start_epoch = "expert", 1
    resume_scaler_state = None
    if last_path.is_file():
        progress = safe_torch_load(last_path, "cpu")
        if progress.get("architecture_version") != ARCHITECTURE_VERSION:
            raise ValueError("Incompatible SimMLM progress checkpoint")
        model.load_state_dict(progress["model"], strict=True)
        histories = progress["histories"]
        best = progress["best"]
        early_stopping = progress.get("early_stopping")
        if isinstance(early_stopping, dict):
            non_improving_validations = int(
                early_stopping.get("non_improving_validations", 0)
            )
        elif best.get("epoch") is not None:
            non_improving_validations = sum(
                1
                for record in histories.get("cooperative", [])
                if record.get("validation") is not None
                and int(record.get("epoch", 0)) > int(best["epoch"])
            )
        stage = progress["stage"]
        start_epoch = int(progress["epoch"]) + 1
        target_optimizers = (
            expert_optimizers if stage == "expert" else [cooperative_optimizer]
        )
        for optimizer, state in zip(target_optimizers, progress["optimizers"]):
            optimizer.load_state_dict(state)
        torch.set_rng_state(progress["cpu_rng_state"])
        if device.type == "cuda" and progress.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(progress["cuda_rng_state"])
        if progress.get("loader_rng_state") is not None:
            train_loader.generator.set_state(progress["loader_rng_state"])
        resume_scaler_state = progress.get("scaler")
        print(f"[Resume] SimMLM stage={stage} epoch={start_epoch}", flush=True)
    else:
        _load_expert_initialization(model.palm_expert, args.palm_ckpt, len(label_ids))
        _load_expert_initialization(model.vein_expert, args.vein_ckpt, len(label_ids))

    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and bool(args.amp)
    )
    if resume_scaler_state:
        scaler.load_state_dict(resume_scaler_state)
    if stage == "expert":
        for epoch in range(start_epoch, args.expert_epochs + 1):
            started = time.time()
            losses = _train_experts_epoch(
                model, train_loader, expert_optimizers, device, scaler, args.gradient_clip
            )
            histories["expert"].append({"epoch": epoch, "losses": losses})
            save_checkpoint_atomic(
                last_path,
                _progress_payload(
                    model, "expert", epoch, histories, expert_optimizers, best, scaler, train_loader
                ),
            )
            print(
                f"[SimMLM expert {epoch:03d}/{args.expert_epochs:03d}] "
                f"loss={losses['total']:.4f} time={time.time()-started:.1f}s",
                flush=True,
            )
        stage, start_epoch = "cooperative", 1
        save_checkpoint_atomic(
            last_path,
            _progress_payload(
                model, "cooperative", 0, histories, [cooperative_optimizer], best, scaler, train_loader
            ),
        )

    callback = representation_callback(model)
    already_stopped = (
        stage == "cooperative"
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
    )
    if already_stopped:
        if not isinstance(early_stopping, dict) or not early_stopping.get("stopped"):
            stop_epoch = start_epoch - 1
            early_stopping = {
                "enabled": True,
                "patience": patience,
                "min_epochs": minimum_epochs,
                "non_improving_validations": non_improving_validations,
                "last_improved": False,
                "stopped": True,
                "stop_reason": (
                    f"metric_rank did not strictly improve for {patience} "
                    f"consecutive validation checks after best epoch {best['epoch']}"
                ),
                "actual_epochs": {
                    "expert": len(histories["expert"]),
                    "cooperative": stop_epoch,
                },
            }
            save_checkpoint_atomic(
                last_path,
                _progress_payload(
                    model, "cooperative", stop_epoch, histories,
                    [cooperative_optimizer], best, scaler, train_loader,
                    early_stopping,
                ),
            )
        print(
            f"[SimMLM early-stop] checkpoint already satisfies stopping rule: "
            f"{early_stopping.get('stop_reason')}",
            flush=True,
        )
        if not best_path.is_file():
            raise RuntimeError("Stopped SimMLM progress has no best checkpoint")
        return best_path
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        losses = _train_cooperative_epoch(
            model, train_loader, cooperative_optimizer, device, scaler, args.gradient_clip
        )
        model.eval()
        validation = evaluate_gallery_probe(
            gallery_loader, probes, callback, device
        )
        rank = metric_rank(validation)
        record = {
            "epoch": epoch,
            "losses": losses,
            "validation": validation,
            "elapsed_seconds": time.time() - started,
        }
        histories["cooperative"].append(record)
        improved = best["rank"] is None or rank < tuple(best["rank"])
        if improved:
            best = {"rank": rank, "epoch": epoch, "validation": validation}
            non_improving_validations = 0
            save_checkpoint_atomic(
                best_path,
                {
                    "architecture_version": ARCHITECTURE_VERSION,
                    "official_commit": OFFICIAL_COMMIT,
                    "method": "simmlm",
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
                "expert": len(histories["expert"]),
                "cooperative": epoch,
            },
        }
        record["early_stopping"] = copy.deepcopy(early_stopping)
        save_checkpoint_atomic(
            last_path,
            _progress_payload(
                model,
                "cooperative",
                epoch,
                histories,
                [cooperative_optimizer],
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
            f"[SimMLM cooperative {epoch:03d}/{args.epochs:03d}] "
            f"loss={losses['total']:.4f} {eers} time={record['elapsed_seconds']:.1f}s",
            flush=True,
        )
        if should_stop:
            print(
                f"[SimMLM early-stop] epoch={epoch} reason={stop_reason}",
                flush=True,
            )
            break
    return best_path


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[SimMLMImageModel, dict]:
    """Strictly reconstruct the complete DMoME system for inference."""

    payload = safe_torch_load(checkpoint_path, "cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("Invalid SimMLM checkpoint")
    if payload.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("Incompatible SimMLM checkpoint")
    if payload.get("method") != "simmlm":
        raise ValueError("Checkpoint method is not SimMLM")
    saved_args = payload.get("args")
    label_ids = payload.get("label_ids")
    state = payload.get("model")
    if not isinstance(saved_args, Mapping):
        raise ValueError("SimMLM checkpoint lacks training arguments")
    if not isinstance(label_ids, list) or not label_ids:
        raise ValueError("SimMLM checkpoint lacks training label IDs")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("SimMLM checkpoint lacks model state")
    model = SimMLMImageModel(
        embedding_dim=int(saved_args["embedding_size"]),
        num_classes=len(label_ids),
        ranking_weight=0.1,
    )
    model.load_state_dict(state, strict=True)
    model.to(torch.device(device)).eval()
    return model, dict(payload)


__all__ = [
    "load_checkpoint_model",
    "representation_callback",
    "train",
]
