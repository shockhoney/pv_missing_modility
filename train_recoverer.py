import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from models.dymo import PalmVeinDynamicTransformer
from utils.datasets_txt import MissingPairTxtDataset


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


def build_loader(list_path: str, img_size: int, batch_size: int, num_workers: int, strong: bool, split_filter=None):
    tf = get_transforms(img_size, strong=strong)
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


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


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


def set_encoder_trainable(model: PalmVeinDynamicTransformer, trainable: bool):
    for encoder in [model.cnn_palm, model.cnn_vein]:
        for param in encoder.parameters():
            param.requires_grad = trainable


def build_optimizer(model, palm_head, vein_head, args):
    encoder_params = list(model.cnn_palm.parameters()) + list(model.cnn_vein.parameters())
    recoverer_params = (
        list(model.vein_from_palm.parameters())
        + list(model.palm_from_vein.parameters())
        + list(palm_head.parameters())
        + list(vein_head.parameters())
    )
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.encoder_lr},
            {"params": recoverer_params, "lr": args.lr},
        ],
        weight_decay=args.wd,
    )


def token_cosine_loss(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(src.flatten(1), target.flatten(1), dim=1).mean()


def compute_recovery_quality(encoded):
    rec_vein = encoded["recovered_vein_global"]
    rec_palm = encoded["recovered_palm_global"]
    rec_vein_tokens = encoded["recovered_vein_tokens"]
    rec_palm_tokens = encoded["recovered_palm_tokens"]

    with torch.no_grad():
        vein_global = encoded["vein_global"].detach()
        palm_global = encoded["palm_global"].detach()
        vein_tokens = encoded["vein_tokens"].detach()
        palm_tokens = encoded["palm_tokens"].detach()

        global_cos_vein = F.cosine_similarity(rec_vein, vein_global, dim=1)
        global_cos_palm = F.cosine_similarity(rec_palm, palm_global, dim=1)
        token_cos_vein = F.cosine_similarity(rec_vein_tokens.flatten(1), vein_tokens.flatten(1), dim=1)
        token_cos_palm = F.cosine_similarity(rec_palm_tokens.flatten(1), palm_tokens.flatten(1), dim=1)
        target_vein_conf = 0.5 * ((global_cos_vein + 1.0) / 2.0) + 0.5 * ((token_cos_vein + 1.0) / 2.0)
        target_palm_conf = 0.5 * ((global_cos_palm + 1.0) / 2.0) + 0.5 * ((token_cos_palm + 1.0) / 2.0)

    return {
        "global_cos": 0.5 * (global_cos_vein.mean() + global_cos_palm.mean()),
        "token_cos": 0.5 * (token_cos_vein.mean() + token_cos_palm.mean()),
        "target_vein_conf": target_vein_conf,
        "target_palm_conf": target_palm_conf,
    }


def compute_losses(encoded, logits_palm, logits_vein, logits_rec_palm, logits_rec_vein, labels, ce):
    rec_vein = encoded["recovered_vein_global"]
    rec_palm = encoded["recovered_palm_global"]
    rec_vein_tokens = encoded["recovered_vein_tokens"]
    rec_palm_tokens = encoded["recovered_palm_tokens"]
    vein_global = encoded["vein_global"].detach()
    palm_global = encoded["palm_global"].detach()
    vein_tokens = encoded["vein_tokens"].detach()
    palm_tokens = encoded["palm_tokens"].detach()
    quality = compute_recovery_quality(encoded)

    loss_id_real = ce(logits_palm, labels) + ce(logits_vein, labels)
    loss_id_rec = ce(logits_rec_vein, labels) + ce(logits_rec_palm, labels)
    loss_l2 = F.mse_loss(rec_vein, vein_global) + F.mse_loss(rec_palm, palm_global)
    loss_cos = (1.0 - F.cosine_similarity(rec_vein, vein_global, dim=1)).mean()
    loss_cos = loss_cos + (1.0 - F.cosine_similarity(rec_palm, palm_global, dim=1)).mean()
    loss_token_l2 = (
        F.mse_loss(rec_vein_tokens, vein_tokens)
        + F.mse_loss(rec_palm_tokens, palm_tokens)
    )
    loss_token_cos = (
        token_cosine_loss(rec_vein_tokens, vein_tokens)
        + token_cosine_loss(rec_palm_tokens, palm_tokens)
    )
    loss_conf = (
        F.mse_loss(encoded["recovered_vein_confidence"], quality["target_vein_conf"])
        + F.mse_loss(encoded["recovered_palm_confidence"], quality["target_palm_conf"])
    )
    total = loss_id_real + loss_id_rec + loss_l2 + loss_cos + loss_token_l2 + loss_token_cos + loss_conf
    return {
        "loss_id_real": loss_id_real,
        "loss_id_rec": loss_id_rec,
        "loss_l2": loss_l2,
        "loss_cos": loss_cos,
        "loss_token_l2": loss_token_l2,
        "loss_token_cos": loss_token_cos,
        "loss_conf": loss_conf,
        "global_cos": quality["global_cos"],
        "token_cos": quality["token_cos"],
        "loss_total_unweighted": total,
    }


@torch.no_grad()
def validate(model, palm_head, vein_head, loader, device, args):
    model.eval()
    palm_head.eval()
    vein_head.eval()

    loss_meter = 0.0
    global_cos_meter = 0.0
    token_cos_meter = 0.0
    conf_meter = 0.0
    acc_real_meter = 0.0
    acc_rec_meter = 0.0
    sample_count = 0
    ce = nn.CrossEntropyLoss()

    for palm, vein, labels, _ in tqdm(loader, desc="Validate recoverer", dynamic_ncols=True, leave=False):
        palm = palm.to(device, non_blocking=True)
        vein = vein.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        encoded = model.encode_modalities(palm, vein)
        logits_palm = palm_head(F.normalize(encoded["palm_global"], dim=1))
        logits_vein = vein_head(F.normalize(encoded["vein_global"], dim=1))
        logits_rec_palm = palm_head(F.normalize(encoded["recovered_palm_global"], dim=1))
        logits_rec_vein = vein_head(F.normalize(encoded["recovered_vein_global"], dim=1))

        losses = compute_losses(encoded, logits_palm, logits_vein, logits_rec_palm, logits_rec_vein, labels, ce)
        loss = (
            args.lambda_id_real * losses["loss_id_real"]
            + args.lambda_id_rec * losses["loss_id_rec"]
            + args.lambda_l2 * losses["loss_l2"]
            + args.lambda_cos * losses["loss_cos"]
            + args.lambda_token_l2 * losses["loss_token_l2"]
            + args.lambda_token_cos * losses["loss_token_cos"]
            + args.lambda_conf * losses["loss_conf"]
        )

        acc_real = ((logits_palm.argmax(dim=1) == labels).float().mean() + (logits_vein.argmax(dim=1) == labels).float().mean()) / 2.0
        acc_rec = (
            (logits_rec_palm.argmax(dim=1) == labels).float().mean()
            + (logits_rec_vein.argmax(dim=1) == labels).float().mean()
        ) / 2.0

        batch_size = labels.size(0)
        sample_count += batch_size
        loss_meter += loss.item() * batch_size
        global_cos_meter += losses["global_cos"].item() * batch_size
        token_cos_meter += losses["token_cos"].item() * batch_size
        conf_meter += losses["loss_conf"].item() * batch_size
        acc_real_meter += acc_real.item() * batch_size
        acc_rec_meter += acc_rec.item() * batch_size

    return {
        "loss": loss_meter / max(sample_count, 1),
        "global_cos": global_cos_meter / max(sample_count, 1),
        "token_cos": token_cos_meter / max(sample_count, 1),
        "conf_mse": conf_meter / max(sample_count, 1),
        "acc_real": acc_real_meter / max(sample_count, 1),
        "acc_rec": acc_rec_meter / max(sample_count, 1),
    }


def main():
    parser = argparse.ArgumentParser("Train feature-level cross-modal recoverers for DyMo")
    parser.add_argument("--train_full_list", type=str, default="data_txt/polyu_train_full.txt")
    parser.add_argument("--val_full_list", type=str, default="data_txt/polyu_val_full.txt")
    parser.add_argument("--palm_ckpt", type=str, default="outputs_dymo/encoders/palm_best.pth")
    parser.add_argument("--vein_ckpt", type=str, default="outputs_dymo/encoders/vein_best.pth")
    parser.add_argument("--save_dir", type=str, default="outputs_dymo/recoverer")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--freeze_encoder_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--encoder_dim", type=int, default=256)
    parser.add_argument("--token_grid", type=int, default=4)
    parser.add_argument("--transformer_dim", type=int, default=256)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder_lr", type=float, default=3e-5)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--lambda_id_real", type=float, default=1.0)
    parser.add_argument("--lambda_id_rec", type=float, default=0.5)
    parser.add_argument("--lambda_l2", type=float, default=1.0)
    parser.add_argument("--lambda_cos", type=float, default=1.0)
    parser.add_argument("--lambda_token_l2", type=float, default=2.0)
    parser.add_argument("--lambda_token_cos", type=float, default=1.0)
    parser.add_argument("--lambda_conf", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "runs"))
    num_classes = infer_num_classes(args.train_full_list)

    train_loader = build_loader(
        args.train_full_list,
        img_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        strong=True,
    )
    val_loader = build_loader(
        args.val_full_list,
        img_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        strong=False,
    )

    model = PalmVeinDynamicTransformer(
        num_classes=num_classes,
        input_size=args.input_size,
        encoder_dim=args.encoder_dim,
        token_grid=args.token_grid,
        transformer_dim=args.transformer_dim,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        projection_dim=args.projection_dim,
    ).to(device)
    palm_head = nn.Linear(args.encoder_dim, num_classes).to(device)
    vein_head = nn.Linear(args.encoder_dim, num_classes).to(device)

    load_encoder_checkpoint(model.cnn_palm, args.palm_ckpt, device)
    load_encoder_checkpoint(model.cnn_vein, args.vein_ckpt, device)
    set_encoder_trainable(model, trainable=False)
    optimizer = build_optimizer(model, palm_head, vein_head, args)
    ce = nn.CrossEntropyLoss()

    best_score = float("-inf")
    best_path = os.path.join(args.save_dir, "recoverer_best.pth")
    encoders_unfrozen = False

    for epoch in range(1, args.epochs + 1):
        if (not encoders_unfrozen) and epoch > args.freeze_encoder_epochs:
            set_encoder_trainable(model, trainable=True)
            encoders_unfrozen = True
            print(f"[Info] unfroze encoders at epoch {epoch}; encoder lr={args.encoder_lr}")

        model.train()
        palm_head.train()
        vein_head.train()

        running_loss = 0.0
        running_global_cos = 0.0
        running_token_cos = 0.0
        running_conf = 0.0
        running_acc_real = 0.0
        running_acc_rec = 0.0
        sample_count = 0

        for palm, vein, labels, _ in tqdm(train_loader, desc=f"Recoverer epoch {epoch}", dynamic_ncols=True):
            palm = palm.to(device, non_blocking=True)
            vein = vein.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            encoded = model.encode_modalities(palm, vein)
            logits_palm = palm_head(F.normalize(encoded["palm_global"], dim=1))
            logits_vein = vein_head(F.normalize(encoded["vein_global"], dim=1))
            logits_rec_palm = palm_head(F.normalize(encoded["recovered_palm_global"], dim=1))
            logits_rec_vein = vein_head(F.normalize(encoded["recovered_vein_global"], dim=1))

            losses = compute_losses(encoded, logits_palm, logits_vein, logits_rec_palm, logits_rec_vein, labels, ce)
            loss = (
                args.lambda_id_real * losses["loss_id_real"]
                + args.lambda_id_rec * losses["loss_id_rec"]
                + args.lambda_l2 * losses["loss_l2"]
                + args.lambda_cos * losses["loss_cos"]
                + args.lambda_token_l2 * losses["loss_token_l2"]
                + args.lambda_token_cos * losses["loss_token_cos"]
                + args.lambda_conf * losses["loss_conf"]
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            sample_count += batch_size
            running_loss += loss.item() * batch_size

            running_global_cos += losses["global_cos"].item() * batch_size
            running_token_cos += losses["token_cos"].item() * batch_size
            running_conf += losses["loss_conf"].item() * batch_size

            acc_real = ((logits_palm.argmax(dim=1) == labels).float().mean() + (logits_vein.argmax(dim=1) == labels).float().mean()) / 2.0
            acc_rec = (
                (logits_rec_palm.argmax(dim=1) == labels).float().mean()
                + (logits_rec_vein.argmax(dim=1) == labels).float().mean()
            ) / 2.0
            running_acc_real += acc_real.item() * batch_size
            running_acc_rec += acc_rec.item() * batch_size

        train_loss = running_loss / max(sample_count, 1)
        train_global_cos = running_global_cos / max(sample_count, 1)
        train_token_cos = running_token_cos / max(sample_count, 1)
        train_conf = running_conf / max(sample_count, 1)
        train_acc_real = running_acc_real / max(sample_count, 1)
        train_acc_rec = running_acc_rec / max(sample_count, 1)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/global_cos", train_global_cos, epoch)
        writer.add_scalar("train/token_cos", train_token_cos, epoch)
        writer.add_scalar("train/conf_mse", train_conf, epoch)
        writer.add_scalar("train/acc_real", train_acc_real, epoch)
        writer.add_scalar("train/acc_rec", train_acc_rec, epoch)

        val_metrics = validate(model, palm_head, vein_head, val_loader, device, args)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/global_cos", val_metrics["global_cos"], epoch)
        writer.add_scalar("val/token_cos", val_metrics["token_cos"], epoch)
        writer.add_scalar("val/conf_mse", val_metrics["conf_mse"], epoch)
        writer.add_scalar("val/acc_real", val_metrics["acc_real"], epoch)
        writer.add_scalar("val/acc_rec", val_metrics["acc_rec"], epoch)

        score = (
            0.4 * val_metrics["global_cos"]
            + 0.4 * val_metrics["token_cos"]
            + 0.2 * val_metrics["acc_rec"]
        )
        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f} "
            f"train_global_cos={train_global_cos:.4f} train_token_cos={train_token_cos:.4f} "
            f"train_conf_mse={train_conf:.4f} "
            f"train_acc_real={train_acc_real:.4f} train_acc_rec={train_acc_rec:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_global_cos={val_metrics['global_cos']:.4f} val_token_cos={val_metrics['token_cos']:.4f} "
            f"val_conf_mse={val_metrics['conf_mse']:.4f} "
            f"val_acc_real={val_metrics['acc_real']:.4f} val_acc_rec={val_metrics['acc_rec']:.4f}"
        )

        if score > best_score:
            best_score = score
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "palm_head": palm_head.state_dict(),
                    "vein_head": vein_head.state_dict(),
                    "args": vars(args),
                    "num_classes": num_classes,
                },
                best_path,
            )
            print(f"[Info] saved best recoverer checkpoint to {best_path}")

    writer.close()


if __name__ == "__main__":
    main()
