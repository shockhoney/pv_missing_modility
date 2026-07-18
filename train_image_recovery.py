import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models.image_recovery import ARCHITECTURE_VERSION, BidirectionalImageRecovery
from utils.checkpoint import load_encoder_teacher_from_checkpoint
from utils.checkpoint_io import file_sha256, save_checkpoint
from utils.datasets_txt import (
    DATASET_CONFIGS,
    CrossModalRecoveryDataset,
)
from utils.evaluation import gallery_probe_scores
from utils.preprocess import build_palm_transform, build_vein_transform
from utils.runtime import (
    build_data_loader,
    cosine_annealing_lr,
    resolve_device,
    set_optimizer_lr,
    set_random_seed,
)
from utils.scenarios import COMPLETE, PALMPRINT_MISSING, PALMVEIN_MISSING


def freeze(module):
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


def make_datasets(args):
    palm_transform = build_palm_transform(args.input_size)
    vein_transform = build_vein_transform(args.input_size)
    training = CrossModalRecoveryDataset(
        args.train_list, palm_transform, vein_transform
    )
    gallery = CrossModalRecoveryDataset(
        args.val_gallery_list, palm_transform, vein_transform
    )
    probe = CrossModalRecoveryDataset(
        args.val_protocol_list, palm_transform, vein_transform, split_filter=COMPLETE
    )
    validation_labels = sorted({sample["label"] for sample in gallery.samples})
    probe_labels = {sample["label"] for sample in probe.samples}
    if set(validation_labels) != probe_labels:
        raise ValueError("Recovery validation Gallery and Probe identity labels differ")
    print(
        f"[Info] recovery identities: train={len(set(s['label'] for s in training.samples))}, "
        f"validation={len(validation_labels)}"
    )
    return training, gallery, probe, sorted(validation_labels)


def symmetric_contrastive_loss(query, target, temperature):
    query = F.normalize(query, dim=1)
    target = F.normalize(target, dim=1)
    logits = query @ target.t() / temperature
    diagonal = torch.arange(query.size(0), device=query.device)
    return 0.5 * (
        F.cross_entropy(logits, diagonal) + F.cross_entropy(logits.t(), diagonal)
    )


def recovery_loss(
    model,
    palm_encoder,
    vein_encoder,
    palm_teacher,
    vein_teacher,
    batch,
    device,
    args,
):
    palm = batch["palm"].to(device, non_blocking=True)
    vein = batch["vein"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    with torch.no_grad():
        target_palm_embedding = palm_encoder(palm)
        target_vein_embedding = vein_encoder(vein)

    generated = model(
        batch["vein_as_palm"].to(device, non_blocking=True),
        batch["palm_as_vein"].to(device, non_blocking=True),
    )
    generated_palm_embedding = palm_encoder(generated["generated_palm"])
    generated_vein_embedding = vein_encoder(generated["generated_vein"])
    pixel = 0.5 * (
        F.smooth_l1_loss(generated["generated_palm"], palm)
        + F.smooth_l1_loss(generated["generated_vein"], vein)
    )
    cosine = 0.5 * (
        (1.0 - F.cosine_similarity(
            generated_palm_embedding, target_palm_embedding, dim=1
        )).mean()
        + (1.0 - F.cosine_similarity(
            generated_vein_embedding, target_vein_embedding, dim=1
        )).mean()
    )
    contrastive = 0.5 * (
        symmetric_contrastive_loss(
            generated_palm_embedding, target_palm_embedding.detach(), args.temperature
        )
        + symmetric_contrastive_loss(
            generated_vein_embedding, target_vein_embedding.detach(), args.temperature
        )
    )
    identity = 0.5 * (
        F.cross_entropy(palm_teacher(generated_palm_embedding, labels), labels)
        + F.cross_entropy(vein_teacher(generated_vein_embedding, labels), labels)
    )
    loss = (
        args.lambda_pixel * pixel
        + args.lambda_cosine * cosine
        + args.lambda_contrastive * contrastive
        + args.lambda_identity * identity
    )
    return {
        "loss": loss,
        "pixel": pixel,
        "cosine": cosine,
        "contrastive": contrastive,
        "identity": identity,
    }


@torch.inference_mode()
def extract(model, palm_encoder, vein_encoder, loader, device, description):
    values = {key: [] for key in ("palm", "vein", "generated_palm", "generated_vein")}
    labels = []
    for batch in tqdm(loader, desc=description, dynamic_ncols=True, leave=False):
        palm = batch["palm"].to(device, non_blocking=True)
        vein = batch["vein"].to(device, non_blocking=True)
        generated = model(
            batch["vein_as_palm"].to(device, non_blocking=True),
            batch["palm_as_vein"].to(device, non_blocking=True),
        )
        values["palm"].append(palm_encoder(palm).cpu())
        values["vein"].append(vein_encoder(vein).cpu())
        values["generated_palm"].append(palm_encoder(generated["generated_palm"]).cpu())
        values["generated_vein"].append(vein_encoder(generated["generated_vein"]).cpu())
        labels.append(batch["label"])
    return {key: torch.cat(items) for key, items in values.items()}, torch.cat(labels)


def top1(scores, candidate_labels, probe_labels):
    return float(
        candidate_labels[scores.argmax(1)].eq(probe_labels).float().mean().item()
    )


def select_weight(available_scores, recovery_scores, candidate_labels, probe_labels, grid):
    best_weight, best_accuracy = 0.0, -1.0
    for weight in grid:
        accuracy = top1(
            available_scores + weight * recovery_scores, candidate_labels, probe_labels
        )
        if accuracy > best_accuracy:
            best_weight, best_accuracy = weight, accuracy
    return best_weight, best_accuracy


@torch.inference_mode()
def validate(model, palm_encoder, vein_encoder, gallery_loader, probe_loader, device, args):
    model.eval()
    gallery, gallery_labels = extract(
        model, palm_encoder, vein_encoder, gallery_loader, device, "Recovery validation gallery"
    )
    probes, probe_labels = extract(
        model, palm_encoder, vein_encoder, probe_loader, device, "Recovery validation probes"
    )
    palm_available, candidate_labels = gallery_probe_scores(
        gallery["vein"], gallery_labels, probes["vein"]
    )
    palm_recovery, _ = gallery_probe_scores(
        gallery["generated_palm"], gallery_labels, probes["generated_palm"]
    )
    vein_available, _ = gallery_probe_scores(
        gallery["palm"], gallery_labels, probes["palm"]
    )
    vein_recovery, _ = gallery_probe_scores(
        gallery["generated_vein"], gallery_labels, probes["generated_vein"]
    )
    grid = [step * args.alpha_step for step in range(round(args.alpha_max / args.alpha_step) + 1)]
    palm_weight, palm_fused = select_weight(
        palm_available, palm_recovery, candidate_labels, probe_labels, grid
    )
    vein_weight, vein_fused = select_weight(
        vein_available, vein_recovery, candidate_labels, probe_labels, grid
    )
    return {
        "palm_available": top1(palm_available, candidate_labels, probe_labels),
        "palm_recovery": top1(palm_recovery, candidate_labels, probe_labels),
        "palm_fused": palm_fused,
        "palm_alpha": palm_weight,
        "vein_available": top1(vein_available, candidate_labels, probe_labels),
        "vein_recovery": top1(vein_recovery, candidate_labels, probe_labels),
        "vein_fused": vein_fused,
        "vein_alpha": vein_weight,
    }


def train(args):
    device = resolve_device(args.device, require_available=True, announce=True)
    set_random_seed(args.seed)
    training, gallery, probe, validation_labels = make_datasets(args)
    train_loader = build_data_loader(
        training, args.batch_size, args.num_workers, train=True
    )
    gallery_loader = build_data_loader(gallery, args.batch_size, args.num_workers)
    probe_loader = build_data_loader(probe, args.batch_size, args.num_workers)

    palm_encoder, palm_teacher, _ = load_encoder_teacher_from_checkpoint(
        args.palm_ckpt, "palm", args.embedding_size, device
    )
    vein_encoder, vein_teacher, _ = load_encoder_teacher_from_checkpoint(
        args.vein_ckpt, "vein", args.embedding_size, device
    )
    for module in (palm_encoder, vein_encoder, palm_teacher, vein_teacher):
        freeze(module)
    model = BidirectionalImageRecovery(
        channels=args.recovery_channels, blocks=args.recovery_blocks
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        lr = cosine_annealing_lr(args.lr, args.min_lr, args.epochs, args.warmup_epochs, epoch)
        set_optimizer_lr(optimizer, lr)
        sums = {key: 0.0 for key in ("loss", "pixel", "cosine", "contrastive", "identity")}
        total = 0
        for batch in tqdm(train_loader, desc=f"Train image recovery {epoch}", dynamic_ncols=True):
            losses = recovery_loss(
                model,
                palm_encoder,
                vein_encoder,
                palm_teacher,
                vein_teacher,
                batch,
                device,
                args,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            batch_size = batch["label"].numel()
            total += batch_size
            for key in sums:
                sums[key] += float(losses[key].detach().item()) * batch_size
        validation = validate(
            model, palm_encoder, vein_encoder, gallery_loader, probe_loader, device, args
        )
        stats = " ".join(f"{key}={value / total:.4f}" for key, value in sums.items())
        val_text = " ".join(f"{key}={value:.4f}" for key, value in validation.items())
        print(f"[Epoch {epoch}] {stats} lr={lr:.6g} | {val_text}")
        score = min(validation["palm_fused"], validation["vein_fused"])
        if score > best:
            best = score
            save_checkpoint(
                args.save_path,
                {
                    "architecture_version": ARCHITECTURE_VERSION,
                    "training_stage": "image_recovery",
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "args": vars(args),
                    "validation": validation,
                    "validation_labels": validation_labels,
                    "best_selection_score": best,
                    "fusion_weights": {
                        PALMPRINT_MISSING: validation["palm_alpha"],
                        PALMVEIN_MISSING: validation["vein_alpha"],
                    },
                    "palm_encoder_sha256": file_sha256(args.palm_ckpt),
                    "vein_encoder_sha256": file_sha256(args.vein_ckpt),
                },
            )
            print(f"[Info] saved {args.save_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Train image-domain missing-modality recovery")
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), default="tongji")
    parser.add_argument("--train_list", default="data_txt/tongji/ssfd_train_full.txt")
    parser.add_argument(
        "--val_gallery_list", default="data_txt/tongji/ssfd_val_gallery_full.txt"
    )
    parser.add_argument(
        "--val_protocol_list", default="data_txt/tongji/ssfd_val_protocol.txt"
    )
    parser.add_argument("--palm_ckpt", default="outputs/encoders/palm_best.pth")
    parser.add_argument("--vein_ckpt", default="outputs/encoders/vein_best.pth")
    parser.add_argument("--save_path", default="outputs/image_recovery/best.pth")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--embedding_size", type=int, default=256)
    parser.add_argument("--recovery_channels", type=int, default=32)
    parser.add_argument("--recovery_blocks", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda_pixel", type=float, default=0.25)
    parser.add_argument("--lambda_cosine", type=float, default=2.0)
    parser.add_argument("--lambda_contrastive", type=float, default=0.5)
    parser.add_argument("--lambda_identity", type=float, default=0.02)
    parser.add_argument("--alpha_step", type=float, default=0.05)
    parser.add_argument("--alpha_max", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.alpha_step <= 0 or args.alpha_max < 0:
        parser.error("Fusion-weight search bounds must be non-negative")
    return args


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
