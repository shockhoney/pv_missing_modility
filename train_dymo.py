import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from models.dymo import PalmVeinDynamicTransformer
from utils.datasets_txt import MissingPairTxtDataset
from utils.metrics import compute_eer, tar_at_far
from utils.prototype_loss import PrototypeLoss


def get_transforms(img_size: int, strong: bool):
    ops = [transforms.Resize((img_size, img_size))]
    if strong:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(8),
                transforms.RandomAffine(0, translate=(0.08, 0.08)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        )
    ops.extend(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    return transforms.Compose(ops)


def infer_num_classes(list_path: str) -> int:
    labels = set()
    with open(list_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 3:
                labels.add(int(parts[2]))
    return len(labels)


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_pair_scores(feats: np.ndarray, labels: np.ndarray):
    feats = feats.astype(np.float32)
    labels = labels.astype(np.int64)
    sim = feats @ feats.T
    i, j = np.triu_indices(labels.shape[0], k=1)
    scores = sim[i, j].astype(np.float32)
    pair_labels = (labels[i] == labels[j]).astype(np.int32)
    return scores, pair_labels


def get_tar_value(tar_ret):
    if isinstance(tar_ret, dict):
        return float(tar_ret.get("TAR", tar_ret.get("tar", 0.0)))
    return float(tar_ret)


def build_loader(list_path: str, input_size: int, batch_size: int, num_workers: int, strong: bool, split_filter=None):
    tf = get_transforms(input_size, strong=strong)
    dataset = MissingPairTxtDataset(
        list_path,
        transform_palm=tf,
        transform_vein=tf,
        split_filter=split_filter,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=strong,
        drop_last=strong,
        num_workers=num_workers,
        pin_memory=True,
    )


def load_encoder_checkpoint(module: nn.Module, checkpoint_path: str, device):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint_path}")
    ckpt = safe_torch_load(checkpoint_path, device)
    state_dict = ckpt.get("encoder", ckpt.get("model", ckpt))
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    print(f"[Info] loaded encoder weights from {checkpoint_path}")
    if missing:
        print(f"[Info] missing encoder keys: {len(missing)}")
    if unexpected:
        print(f"[Info] unexpected encoder keys: {len(unexpected)}")


def load_recoverer_checkpoint(model: PalmVeinDynamicTransformer, checkpoint_path: str, device):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Recoverer checkpoint not found: {checkpoint_path}")
    ckpt = safe_torch_load(checkpoint_path, device)
    state_dict = ckpt.get("model", ckpt)

    prefixes = ("vein_from_palm.", "palm_from_vein.")
    current_state = model.state_dict()
    recoverer_state = {
        k: v
        for k, v in state_dict.items()
        if k.startswith(prefixes) and k in current_state and current_state[k].shape == v.shape
    }
    current_state.update(recoverer_state)
    model.load_state_dict(current_state, strict=False)
    skipped = [k for k, v in state_dict.items() if k.startswith(prefixes) and (k not in recoverer_state)]
    print(f"[Info] loaded recoverer weights from {checkpoint_path}")
    if skipped:
        print(f"[Info] skipped incompatible recoverer keys: {len(skipped)}")


def build_optimizer(model, args):
    encoder_params = list(model.cnn_palm.parameters()) + list(model.cnn_vein.parameters())
    recoverer_params = list(model.vein_from_palm.parameters()) + list(model.palm_from_vein.parameters())
    other_ids = {id(param) for param in encoder_params + recoverer_params}
    other_params = [param for param in model.parameters() if id(param) not in other_ids]

    param_groups = [
        {"params": encoder_params, "lr": args.encoder_lr},
        {"params": other_params, "lr": args.transformer_lr},
    ]
    if args.train_recoverers:
        param_groups.append({"params": recoverer_params, "lr": args.recoverer_lr})
    return torch.optim.AdamW(param_groups, weight_decay=args.wd)


def set_recoverers_trainable(model: PalmVeinDynamicTransformer, trainable: bool):
    for module in [model.vein_from_palm, model.palm_from_vein]:
        for param in module.parameters():
            param.requires_grad = trainable


def mode_masks(batch_size: int, device) -> Dict[str, torch.Tensor]:
    return {
        "full": torch.zeros(batch_size, 2, dtype=torch.bool, device=device),
        "palm_only": torch.tensor([[False, True]], dtype=torch.bool, device=device).expand(batch_size, -1),
        "vein_only": torch.tensor([[True, False]], dtype=torch.bool, device=device).expand(batch_size, -1),
    }


def forward_training_states(model: PalmVeinDynamicTransformer, palm, vein, labels):
    encoded = model.encode_modalities(palm, vein)
    masks = mode_masks(palm.size(0), palm.device)

    states = {
        "full": model.forward_from_encoded(
            encoded,
            masks["full"],
            use_recovered_mask=torch.zeros_like(masks["full"]),
            labels=labels,
        ),
        "palm_only_real": model.forward_from_encoded(
            encoded,
            masks["palm_only"],
            use_recovered_mask=torch.zeros_like(masks["palm_only"]),
            labels=labels,
        ),
        "vein_only_real": model.forward_from_encoded(
            encoded,
            masks["vein_only"],
            use_recovered_mask=torch.zeros_like(masks["vein_only"]),
            labels=labels,
        ),
        "palm_only_rec": model.forward_from_encoded(
            encoded,
            masks["palm_only"],
            use_recovered_mask=masks["palm_only"],
            labels=labels,
        ),
        "vein_only_rec": model.forward_from_encoded(
            encoded,
            masks["vein_only"],
            use_recovered_mask=masks["vein_only"],
            labels=labels,
        ),
    }
    return encoded, states


def compute_proto_update(labels: torch.Tensor, feat: torch.Tensor, num_classes: int):
    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    class_sum = one_hot.t() @ feat
    class_count = one_hot.sum(dim=0, keepdim=True).t()
    return class_sum, class_count


def compute_quality(embedding: torch.Tensor, prototypes: torch.Tensor):
    scores = F.normalize(embedding, dim=1) @ F.normalize(prototypes, dim=1).t()
    return scores.max(dim=1).values


def cosine_distance_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    if student.size(1) != teacher.size(1):
        raise ValueError(
            "Distillation requires projection_dim to match encoder_dim. "
            f"Got student dim={student.size(1)} and teacher dim={teacher.size(1)}."
        )
    return 1.0 - F.cosine_similarity(student, teacher, dim=1).mean()


def compute_distillation_losses(encoded: Dict[str, torch.Tensor], states: Dict[str, Dict[str, torch.Tensor]]):
    teacher_palm = F.normalize(encoded["palm_global"].detach(), dim=1)
    teacher_vein = F.normalize(encoded["vein_global"].detach(), dim=1)
    teacher_full = F.normalize(teacher_palm + teacher_vein, dim=1)

    loss_distill_full = cosine_distance_loss(states["full"]["embedding"], teacher_full)
    loss_distill_real = cosine_distance_loss(states["palm_only_real"]["embedding"], teacher_palm)
    loss_distill_real = loss_distill_real + cosine_distance_loss(states["vein_only_real"]["embedding"], teacher_vein)

    k_teacher = teacher_full @ teacher_full.t()
    k_student = states["full"]["embedding"] @ states["full"]["embedding"].t()
    loss_rel = F.mse_loss(k_student, k_teacher)
    return loss_distill_full, loss_distill_real, loss_rel


@torch.no_grad()
def extract_protocol_features(model, loader, device, protocol_mode: str, use_recovered: bool):
    model.eval()
    feats: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    for palm, vein, target, mask in tqdm(
        loader, desc=f"Val-{protocol_mode}-{'rec' if use_recovered else 'real'}", dynamic_ncols=True, leave=False
    ):
        palm = palm.to(device, non_blocking=True)
        vein = vein.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        missing_mask = (1.0 - mask.to(device, non_blocking=True)).bool()
        recovered_mask = missing_mask.clone() if use_recovered else torch.zeros_like(missing_mask)
        if protocol_mode == "full":
            missing_mask = torch.zeros_like(missing_mask)
            recovered_mask = torch.zeros_like(missing_mask)

        output = model(palm, vein, missing_mask=missing_mask, use_recovered_mask=recovered_mask)
        feats.append(output["embedding"].cpu().numpy())
        labels.append(target.cpu().numpy())

    feats = np.vstack(feats)
    labels = np.concatenate(labels, axis=0)
    return feats, labels


@torch.no_grad()
def evaluate_verification(model, loaders: Dict[str, DataLoader], device, far_list: List[float], use_recovered: bool):
    results = {}
    for split_name, loader in loaders.items():
        feats, labels = extract_protocol_features(
            model,
            loader,
            device,
            protocol_mode=split_name,
            use_recovered=use_recovered,
        )
        scores, pair_labels = build_pair_scores(feats, labels)
        if (pair_labels == 1).sum() == 0 or (pair_labels == 0).sum() == 0:
            results[split_name] = {"eer": float("nan"), "tar_list": [(far, float("nan")) for far in far_list]}
            continue
        eer = compute_eer(scores, pair_labels, is_similarity=True)
        tar_list = []
        for far in far_list:
            tar_list.append((far, get_tar_value(tar_at_far(scores, pair_labels, far, is_similarity=True))))
        results[split_name] = {"eer": float(eer), "tar_list": tar_list}
    return results


def format_tar_list(tar_list):
    return " ".join([f"TAR@FAR={far:.0e}:{tar * 100:.2f}%" for far, tar in tar_list])


def main():
    parser = argparse.ArgumentParser("Train DyMo backbone for palmprint-palmvein missing-modality recognition")
    parser.add_argument("--train_full_list", type=str, default="data_txt/polyu_train_full.txt")
    parser.add_argument("--val_full_list", type=str, default="data_txt/polyu_val_full.txt")
    parser.add_argument("--val_missing_list", type=str, default="data_txt/polyu_val_missing_fixed.txt")
    parser.add_argument("--palm_ckpt", type=str, default="outputs_dymo/encoders/palm_best.pth")
    parser.add_argument("--vein_ckpt", type=str, default="outputs_dymo/encoders/vein_best.pth")
    parser.add_argument("--recoverer_ckpt", type=str, default="outputs_dymo/recoverer/recoverer_best.pth")
    parser.add_argument("--save_dir", type=str, default="outputs_dymo/dymo")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--encoder_dim", type=int, default=256)
    parser.add_argument("--token_grid", type=int, default=4)
    parser.add_argument("--transformer_dim", type=int, default=256)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--transformer_lr", type=float, default=3e-4)
    parser.add_argument("--encoder_lr", type=float, default=3e-5)
    parser.add_argument("--recoverer_lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--lambda_full_cls", type=float, default=1.0)
    parser.add_argument("--lambda_partial_cls", type=float, default=1.0)
    parser.add_argument("--lambda_pt", type=float, default=1.0)
    parser.add_argument("--lambda_cons", type=float, default=None)
    parser.add_argument("--lambda_cons_real", type=float, default=0.2)
    parser.add_argument("--lambda_cons_rec", type=float, default=0.5)
    parser.add_argument("--lambda_distill_full", type=float, default=0.5)
    parser.add_argument("--lambda_distill_real", type=float, default=0.1)
    parser.add_argument("--lambda_state_real", type=float, default=0.2)
    parser.add_argument("--lambda_state_rec", type=float, default=0.1)
    parser.add_argument("--lambda_rel", type=float, default=0.1)
    parser.add_argument("--full_baseline_eer", type=float, default=None)
    parser.add_argument("--pt_warmup_epochs", type=int, default=3)
    parser.add_argument("--prototype_temperature", type=float, default=0.1)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--train_recoverers", action="store_true")
    parser.add_argument("--far_list", type=float, nargs="+", default=[1e-4, 1e-5])
    parser.add_argument("--arcface_s", type=float, default=64.0)
    parser.add_argument("--arcface_m", type=float, default=0.5)
    args = parser.parse_args()
    if args.lambda_cons is not None:
        args.lambda_cons_real = args.lambda_cons
        args.lambda_cons_rec = args.lambda_cons

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "runs"))
    num_classes = infer_num_classes(args.train_full_list)

    train_loader = build_loader(
        args.train_full_list,
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        strong=True,
    )
    val_loaders = {
        "full": build_loader(
            args.val_full_list,
            input_size=args.input_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            strong=False,
        ),
        "palm_only": build_loader(
            args.val_missing_list,
            input_size=args.input_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            strong=False,
            split_filter="palm_only",
        ),
        "vein_only": build_loader(
            args.val_missing_list,
            input_size=args.input_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            strong=False,
            split_filter="vein_only",
        ),
        "mixed_missing": build_loader(
            args.val_missing_list,
            input_size=args.input_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            strong=False,
        ),
    }

    model = PalmVeinDynamicTransformer(
        num_classes=num_classes,
        input_size=args.input_size,
        encoder_dim=args.encoder_dim,
        token_grid=args.token_grid,
        transformer_dim=args.transformer_dim,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        projection_dim=args.projection_dim,
        arcface_s=args.arcface_s,
        arcface_m=args.arcface_m,
    ).to(device)
    load_encoder_checkpoint(model.cnn_palm, args.palm_ckpt, device)
    load_encoder_checkpoint(model.cnn_vein, args.vein_ckpt, device)
    load_recoverer_checkpoint(model, args.recoverer_ckpt, device)
    if not args.train_recoverers:
        set_recoverers_trainable(model, trainable=False)
        print("[Info] recoverers frozen for DyMo training.")

    optimizer = build_optimizer(model, args)
    ce = nn.CrossEntropyLoss()
    criterion_pt = PrototypeLoss(
        temperature=args.prototype_temperature,
        metric_name="cosine_similarity",
    ).to(device)

    best_score = float("-inf")
    best_path = os.path.join(args.save_dir, "dymo_best.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        class_sum = torch.zeros(num_classes, args.projection_dim, device=device)
        class_count = torch.zeros(num_classes, 1, device=device)
        running = {
            "loss": 0.0,
            "cls": 0.0,
            "full_cls": 0.0,
            "partial_cls": 0.0,
            "pt": 0.0,
            "cons_real": 0.0,
            "cons_rec": 0.0,
            "distill_full": 0.0,
            "distill_real": 0.0,
            "state_real": 0.0,
            "state_rec": 0.0,
            "rel": 0.0,
            "before_q": 0.0,
            "after_q": 0.0,
            "accepted": 0.0,
        }
        sample_count = 0

        for palm, vein, labels, _ in tqdm(train_loader, desc=f"DyMo epoch {epoch}", dynamic_ncols=True):
            palm = palm.to(device, non_blocking=True)
            vein = vein.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            encoded, states = forward_training_states(model, palm, vein, labels)
            labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()

            loss_full_cls = ce(states["full"]["logits"], labels)
            loss_partial_real_cls = 0.5 * (
                ce(states["palm_only_real"]["logits"], labels) + ce(states["vein_only_real"]["logits"], labels)
            )
            loss_partial_rec_cls = 0.5 * (
                ce(states["palm_only_rec"]["logits"], labels) + ce(states["vein_only_rec"]["logits"], labels)
            )
            loss_partial_cls = 0.5 * (loss_partial_real_cls + loss_partial_rec_cls)
            loss_cls = (loss_full_cls + 2.0 * loss_partial_real_cls + 2.0 * loss_partial_rec_cls) / 5.0

            pt_losses = []
            for key in ["full", "palm_only_real", "vein_only_real", "palm_only_rec", "vein_only_rec"]:
                pt_losses.append(criterion_pt(labels_one_hot, model.prototypes.detach(), states[key]["embedding"]))
                batch_sum, batch_count = compute_proto_update(labels, states[key]["embedding"].detach(), num_classes)
                class_sum += batch_sum
                class_count += batch_count
            loss_pt = sum(pt_losses) / len(pt_losses)

            loss_cons_real = 1.0 - F.cosine_similarity(
                states["palm_only_real"]["embedding"], states["full"]["embedding"].detach(), dim=1
            ).mean()
            loss_cons_real = loss_cons_real + 1.0 - F.cosine_similarity(
                states["vein_only_real"]["embedding"], states["full"]["embedding"].detach(), dim=1
            ).mean()
            loss_cons_rec = 1.0 - F.cosine_similarity(
                states["palm_only_rec"]["embedding"], states["full"]["embedding"].detach(), dim=1
            ).mean()
            loss_cons_rec = loss_cons_rec + 1.0 - F.cosine_similarity(
                states["vein_only_rec"]["embedding"], states["full"]["embedding"].detach(), dim=1
            ).mean()
            loss_distill_full, loss_distill_real, loss_rel = compute_distillation_losses(encoded, states)
            loss_state_real = 1.0 - F.cosine_similarity(
                states["palm_only_real"]["embedding"], states["vein_only_real"]["embedding"], dim=1
            ).mean()
            loss_state_rec = 1.0 - F.cosine_similarity(
                states["palm_only_rec"]["embedding"], states["vein_only_rec"]["embedding"], dim=1
            ).mean()

            loss = args.lambda_full_cls * (loss_full_cls / 5.0)
            loss = loss + args.lambda_partial_cls * ((2.0 * loss_partial_real_cls + 2.0 * loss_partial_rec_cls) / 5.0)
            if epoch >= args.pt_warmup_epochs:
                loss = loss + args.lambda_pt * loss_pt
            loss = loss + args.lambda_cons_real * loss_cons_real
            loss = loss + args.lambda_cons_rec * loss_cons_rec
            loss = loss + args.lambda_distill_full * loss_distill_full
            loss = loss + args.lambda_distill_real * loss_distill_real
            loss = loss + args.lambda_state_real * loss_state_real
            loss = loss + args.lambda_state_rec * loss_state_rec
            loss = loss + args.lambda_rel * loss_rel

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                before_quality = 0.5 * (
                    compute_quality(states["palm_only_real"]["embedding"], model.prototypes)
                    + compute_quality(states["vein_only_real"]["embedding"], model.prototypes)
                )
                after_quality = 0.5 * (
                    compute_quality(states["palm_only_rec"]["embedding"], model.prototypes)
                    + compute_quality(states["vein_only_rec"]["embedding"], model.prototypes)
                )
                accepted = (after_quality > before_quality).float()

            batch_size = labels.size(0)
            sample_count += batch_size
            running["loss"] += loss.item() * batch_size
            running["cls"] += loss_cls.item() * batch_size
            running["full_cls"] += loss_full_cls.item() * batch_size
            running["partial_cls"] += loss_partial_cls.item() * batch_size
            running["pt"] += loss_pt.item() * batch_size
            running["cons_real"] += loss_cons_real.item() * batch_size
            running["cons_rec"] += loss_cons_rec.item() * batch_size
            running["distill_full"] += loss_distill_full.item() * batch_size
            running["distill_real"] += loss_distill_real.item() * batch_size
            running["state_real"] += loss_state_real.item() * batch_size
            running["state_rec"] += loss_state_rec.item() * batch_size
            running["rel"] += loss_rel.item() * batch_size
            running["before_q"] += before_quality.mean().item() * batch_size
            running["after_q"] += after_quality.mean().item() * batch_size
            running["accepted"] += accepted.mean().item() * batch_size

        valid = class_count.squeeze(1) > 0
        updated_prototypes = model.prototypes.detach().clone()
        updated_prototypes[valid] = F.normalize(class_sum[valid] / class_count[valid], dim=1)
        model.prototypes.data.copy_(updated_prototypes)

        train_loss = running["loss"] / max(sample_count, 1)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/classification", running["cls"] / max(sample_count, 1), epoch)
        writer.add_scalar("train/full_cls", running["full_cls"] / max(sample_count, 1), epoch)
        writer.add_scalar("train/partial_cls", running["partial_cls"] / max(sample_count, 1), epoch)
        writer.add_scalar("train/prototype", running["pt"] / max(sample_count, 1), epoch)
        writer.add_scalar("train/consistency_real", running["cons_real"] / max(sample_count, 1), epoch)
        writer.add_scalar("train/consistency_rec", running["cons_rec"] / max(sample_count, 1), epoch)
        writer.add_scalar("train/distill_full", running["distill_full"] / max(sample_count, 1), epoch)
        writer.add_scalar("train/distill_real", running["distill_real"] / max(sample_count, 1), epoch)
        writer.add_scalar("train/state_real", running["state_real"] / max(sample_count, 1), epoch)
        writer.add_scalar("train/state_rec", running["state_rec"] / max(sample_count, 1), epoch)
        writer.add_scalar("train/distill_rel", running["rel"] / max(sample_count, 1), epoch)
        writer.add_scalar("train/before_quality", running["before_q"] / max(sample_count, 1), epoch)
        writer.add_scalar("train/after_quality", running["after_q"] / max(sample_count, 1), epoch)
        writer.add_scalar("train/accepted_ratio_proxy", running["accepted"] / max(sample_count, 1), epoch)

        if epoch % args.eval_every != 0 and epoch != args.epochs:
            print(f"[Epoch {epoch}] train_loss={train_loss:.4f}")
            continue

        val_real = evaluate_verification(model, val_loaders, device, far_list=args.far_list, use_recovered=False)
        val_rec = evaluate_verification(model, val_loaders, device, far_list=args.far_list, use_recovered=True)

        full_eer = val_rec["full"]["eer"]
        palm_eer = val_rec["palm_only"]["eer"]
        vein_eer = val_rec["vein_only"]["eer"]
        mixed_eer = val_rec["mixed_missing"]["eer"]
        mean_eer_rec = float(np.nanmean([full_eer, palm_eer, vein_eer, mixed_eer]))
        full_penalty = 0.0
        if args.full_baseline_eer is not None:
            full_penalty = 0.5 * max(0.0, full_eer - args.full_baseline_eer)
        score = -mean_eer_rec - full_penalty

        for split_name in ["full", "palm_only", "vein_only", "mixed_missing"]:
            metrics_real = val_real[split_name]
            metrics_rec = val_rec[split_name]
            writer.add_scalar(f"val_real/{split_name}_eer", metrics_real["eer"], epoch)
            writer.add_scalar(f"val_rec/{split_name}_eer", metrics_rec["eer"], epoch)
            if metrics_real["tar_list"]:
                writer.add_scalar(f"val_real/{split_name}_tar_far_1e4", metrics_real["tar_list"][0][1], epoch)
            if metrics_rec["tar_list"]:
                writer.add_scalar(f"val_rec/{split_name}_tar_far_1e4", metrics_rec["tar_list"][0][1], epoch)
            print(
                f"[Val:{split_name}] real_only EER={metrics_real['eer'] * 100:.3f}% {format_tar_list(metrics_real['tar_list'])} | "
                f"+recovered EER={metrics_rec['eer'] * 100:.3f}% {format_tar_list(metrics_rec['tar_list'])}"
            )

        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f} "
            f"mean_val_eer_rec={mean_eer_rec * 100:.3f}% "
            f"full_penalty={full_penalty:.4f} "
            f"proxy_before_q={running['before_q'] / max(sample_count, 1):.4f} "
            f"proxy_after_q={running['after_q'] / max(sample_count, 1):.4f} "
            f"proxy_accept={running['accepted'] / max(sample_count, 1):.4f}"
        )

        if score > best_score:
            best_score = score
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "args": vars(args),
                    "num_classes": num_classes,
                },
                best_path,
            )
            print(f"[Info] saved best DyMo checkpoint to {best_path}")

    writer.close()


if __name__ == "__main__":
    main()
