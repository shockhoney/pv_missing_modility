import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.dymo import PalmVeinDynamicTransformer, PalmVeinDyMoSelector
from models.stage1_mobileFacenet import MobileFaceNet
from utils.datasets_txt import MissingPairTxtDataset
from utils.metrics import compute_eer, far_frr_acc_at_threshold, roc_auc, tar_at_far


def get_transforms(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )


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


def build_pair_scores(features, labels):
    features = np.asarray(features)
    labels = np.asarray(labels)
    sim = features @ features.T
    i, j = np.triu_indices(labels.shape[0], k=1)
    scores = sim[i, j]
    pair_labels = (labels[i] == labels[j]).astype(int)
    return scores, pair_labels


def eval_with_metrics(scores, pair_labels, name):
    eer, thr = compute_eer(scores, pair_labels, is_similarity=True, return_threshold=True)
    _, _, _, auc_val = roc_auc(scores, pair_labels, is_similarity=True)
    thr_stats = far_frr_acc_at_threshold(scores, pair_labels, thr, is_similarity=True)
    fars = [1e-5, 1e-4, 1e-3]
    tar_info = {far: tar_at_far(scores, pair_labels, far, is_similarity=True) for far in fars}

    print(f"\n===== {name} =====")
    print(f"AUC : {auc_val:.4f}")
    print(f"EER : {eer * 100:.3f}% (threshold = {thr:.4f})")
    print(
        f"ACC@EER_thr = {thr_stats['ACC']:.4f}, "
        f"FAR={thr_stats['FAR']:.4f}, FRR={thr_stats['FRR']:.4f}"
    )
    print("TAR @ FAR:")
    for far, info in tar_info.items():
        print(f"  FAR={far:.1e}: TAR={info['TAR']:.4f}, thr={info['threshold']:.4f}")


def compute_eer_from_features(features, labels):
    scores, pair_labels = build_pair_scores(features, labels)
    if (pair_labels == 1).sum() == 0 or (pair_labels == 0).sum() == 0:
        return float("nan")
    return float(compute_eer(scores, pair_labels, is_similarity=True))


def build_loaders(protocol_list, input_size, batch_size, num_workers):
    tf_test = get_transforms(input_size)
    loaders: Dict[str, DataLoader] = {}
    for split_name in ["full", "palm_only", "vein_only", "random_missing"]:
        dataset = MissingPairTxtDataset(
            protocol_list,
            transform_palm=tf_test,
            transform_vein=tf_test,
            split_filter=split_name,
        )
        if len(dataset) == 0:
            continue
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders


def load_encoder_checkpoint(module: MobileFaceNet, checkpoint_path: str, device):
    ckpt = safe_torch_load(checkpoint_path, device)
    state_dict = ckpt.get("encoder", ckpt.get("model", ckpt))
    module.load_state_dict(state_dict, strict=False)


@torch.no_grad()
def extract_single_modality_features(encoder, loader, modality: str, device, require_available_only: bool = True):
    encoder.eval()
    feats, labels = [], []

    for palm_img, vein_img, labs, mask in tqdm(loader, desc=f"Extract {modality}", dynamic_ncols=True, leave=False):
        mask = mask.to(device)
        available_mask = mask[:, 0] > 0.5 if modality == "palm" else mask[:, 1] > 0.5
        if require_available_only and available_mask.sum() == 0:
            continue

        images = palm_img if modality == "palm" else vein_img
        images = images.to(device)
        labs = labs.to(device)

        if require_available_only:
            images = images[available_mask]
            labs = labs[available_mask]
            if labs.numel() == 0:
                continue

        embedding = F.normalize(encoder(images), dim=1)
        feats.append(embedding.cpu().numpy())
        labels.append(labs.cpu().numpy())

    if not feats:
        return None, None
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


@torch.no_grad()
def extract_dymo_real_only(model, loader, device):
    model.eval()
    feats, labels = [], []

    for palm_img, vein_img, labs, mask in tqdm(loader, desc="Extract DyMo real", dynamic_ncols=True, leave=False):
        palm_img = palm_img.to(device)
        vein_img = vein_img.to(device)
        labs = labs.to(device)
        missing_mask = (1.0 - mask.to(device)).bool()
        output = model(
            palm_img,
            vein_img,
            missing_mask=missing_mask,
            use_recovered_mask=torch.zeros_like(missing_mask),
        )
        feats.append(output["embedding"].cpu().numpy())
        labels.append(labs.cpu().numpy())

    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


@torch.no_grad()
def extract_dymo_selected(
    selector,
    loader,
    stats,
    device,
    selection_mode: str,
    temperature: float,
    selection_tau,
    quality_mode: str,
):
    selector.eval()
    feats, labels = [], []
    selected_ratio = []

    for palm_img, vein_img, labs, mask in tqdm(loader, desc="Extract DyMo selected", dynamic_ncols=True, leave=False):
        palm_img = palm_img.to(device)
        vein_img = vein_img.to(device)
        labs = labs.to(device)
        missing_mask = (1.0 - mask.to(device)).bool()

        output = selector(
            palm_img,
            vein_img,
            missing_mask=missing_mask,
            stats=stats,
            selection_mode=selection_mode,
            temperature=temperature,
            selection_tau=selection_tau,
            quality_mode=quality_mode,
            return_details=False,
        )
        feats.append(output["embedding"].cpu().numpy())
        labels.append(labs.cpu().numpy())
        selected_ratio.append(output["selected_recovered_mask"].any(dim=1).float().cpu().numpy())

    return (
        np.concatenate(feats, axis=0),
        np.concatenate(labels, axis=0),
        float(np.concatenate(selected_ratio, axis=0).mean()) if selected_ratio else 0.0,
    )


@torch.no_grad()
def collect_dymo_selection_candidates(selector, loader, stats, device, selection_mode: str, temperature: float, quality_mode: str):
    selector.eval()
    before_feats, after_feats, labels = [], [], []
    rewards, has_candidate, missing_indices = [], [], []
    reward_key = "open_reward" if selection_mode == "open" else "ics_reward"

    for palm_img, vein_img, labs, mask in tqdm(loader, desc="Collect DyMo candidates", dynamic_ncols=True, leave=False):
        palm_img = palm_img.to(device)
        vein_img = vein_img.to(device)
        labs = labs.to(device)
        missing_mask = (1.0 - mask.to(device)).bool()

        output = selector(
            palm_img,
            vein_img,
            missing_mask=missing_mask,
            stats=stats,
            selection_mode=selection_mode,
            temperature=temperature,
            selection_tau=0.0,
            quality_mode=quality_mode,
            return_details=True,
        )
        before_feats.append(output["before"]["embedding"].cpu().numpy())
        after_feats.append(output["after"]["embedding"].cpu().numpy())
        labels.append(labs.cpu().numpy())
        rewards.append(output["rewards"][reward_key].cpu().numpy())
        has_candidate.append(missing_mask.any(dim=1).cpu().numpy())
        missing_indices.append(missing_mask.float().argmax(dim=1).long().cpu().numpy())

    return {
        "before": np.concatenate(before_feats, axis=0),
        "after": np.concatenate(after_feats, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "rewards": np.concatenate(rewards, axis=0),
        "has_candidate": np.concatenate(has_candidate, axis=0).astype(bool),
        "missing_index": np.concatenate(missing_indices, axis=0).astype(np.int64),
    }


def select_candidate_features(candidates, selection_tau):
    rewards = candidates["rewards"]
    tau = np.asarray(selection_tau, dtype=np.float32)
    if tau.ndim == 0:
        thresholds = np.full_like(rewards, float(tau), dtype=np.float32)
    elif tau.size == 2:
        thresholds = tau.reshape(-1)[candidates["missing_index"]]
    elif tau.size == rewards.size:
        thresholds = tau.reshape(rewards.shape)
    else:
        raise ValueError("selection_tau must be scalar, length-2, or one threshold per sample.")

    use_after = (rewards > thresholds) & candidates["has_candidate"]
    features = candidates["before"].copy()
    features[use_after] = candidates["after"][use_after]
    return features, candidates["labels"], float(use_after.mean()) if use_after.size else 0.0


def search_best_tau(selector, loader, stats, device, selection_mode: str, temperature: float, quality_mode: str, tau_grid):
    candidates = collect_dymo_selection_candidates(
        selector,
        loader,
        stats=stats,
        device=device,
        selection_mode=selection_mode,
        temperature=temperature,
        quality_mode=quality_mode,
    )
    before_eer = compute_eer_from_features(candidates["before"], candidates["labels"])

    all_after = candidates["before"].copy()
    all_after[candidates["has_candidate"]] = candidates["after"][candidates["has_candidate"]]
    after_eer = compute_eer_from_features(all_after, candidates["labels"])

    best = {
        "tau": 0.0,
        "eer": float("inf"),
        "accepted_ratio": 0.0,
        "before_eer": before_eer,
        "after_eer": after_eer,
    }
    for tau in tau_grid:
        features, labels, accepted_ratio = select_candidate_features(candidates, float(tau))
        eer = compute_eer_from_features(features, labels)
        if np.isnan(eer):
            continue
        if eer < best["eer"]:
            best = {
                "tau": float(tau),
                "eer": float(eer),
                "accepted_ratio": accepted_ratio,
                "before_eer": before_eer,
                "after_eer": after_eer,
            }
    return best


def main():
    parser = argparse.ArgumentParser("Evaluate DyMo checkpoints on missing-modality protocols")
    parser.add_argument("--train_full_list", type=str, default="data_txt/polyu_train_full.txt")
    parser.add_argument("--protocol_list", type=str, default="data_txt/polyu_test_missing_protocol.txt")
    parser.add_argument("--val_protocol_list", type=str, default="data_txt/polyu_val_missing_fixed.txt")
    parser.add_argument("--checkpoint", type=str, default="outputs_dymo/dymo/dymo_best.pth")
    parser.add_argument("--stats_path", type=str, default="outputs_dymo/dymo/gaussian/subset_gaussian.pt")
    parser.add_argument("--palm_ckpt", type=str, default="outputs_dymo/encoders/palm_best.pth")
    parser.add_argument("--vein_ckpt", type=str, default="outputs_dymo/encoders/vein_best.pth")
    parser.add_argument("--selection_mode", type=str, default="open", choices=["open", "ics"])
    parser.add_argument("--selection_temperature", type=float, default=0.1)
    parser.add_argument("--selection_tau", type=float, default=0.0)
    parser.add_argument("--quality_mode", type=str, default="log_prob", choices=["log_prob", "probability"])
    parser.add_argument("--search_tau", action="store_true")
    parser.add_argument("--tau_min", type=float, default=-0.2)
    parser.add_argument("--tau_max", type=float, default=0.5)
    parser.add_argument("--tau_steps", type=int, default=71)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--encoder_dim", type=int, default=256)
    parser.add_argument("--token_grid", type=int, default=4)
    parser.add_argument("--transformer_dim", type=int, default=256)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--projection_dim", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = infer_num_classes(args.train_full_list)

    palm_encoder = MobileFaceNet(input_channel=3, input_size=args.input_size, embedding_size=args.encoder_dim).to(device)
    vein_encoder = MobileFaceNet(input_channel=3, input_size=args.input_size, embedding_size=args.encoder_dim).to(device)
    load_encoder_checkpoint(palm_encoder, args.palm_ckpt, device)
    load_encoder_checkpoint(vein_encoder, args.vein_ckpt, device)

    ckpt = safe_torch_load(args.checkpoint, device)
    ckpt_args = ckpt.get("args", {})
    projection_dim = args.projection_dim
    if projection_dim is None:
        projection_dim = int(ckpt_args.get("projection_dim", args.encoder_dim))

    model = PalmVeinDynamicTransformer(
        num_classes=num_classes,
        input_size=args.input_size,
        encoder_dim=args.encoder_dim,
        token_grid=args.token_grid,
        transformer_dim=args.transformer_dim,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        projection_dim=projection_dim,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    stats = safe_torch_load(args.stats_path, device)
    selector = PalmVeinDyMoSelector(model).to(device)

    loaders = build_loaders(args.protocol_list, args.input_size, args.batch_size, args.num_workers)
    selection_tau = args.selection_tau

    if args.search_tau:
        print("\n########## Selector Tau Search on Validation ##########")
        tau_grid = np.linspace(args.tau_min, args.tau_max, args.tau_steps)
        val_loaders = build_loaders(args.val_protocol_list, args.input_size, args.batch_size, args.num_workers)
        tau_by_missing = [args.selection_tau, args.selection_tau]
        for split_name in ["palm_only", "vein_only"]:
            if split_name not in val_loaders:
                continue
            result = search_best_tau(
                selector,
                val_loaders[split_name],
                stats=stats,
                device=device,
                selection_mode=args.selection_mode,
                temperature=args.selection_temperature,
                quality_mode=args.quality_mode,
                tau_grid=tau_grid,
            )
            if split_name == "palm_only":
                tau_by_missing[1] = result["tau"]
            else:
                tau_by_missing[0] = result["tau"]
            print(
                f"{split_name}: best_tau={result['tau']:.4f} "
                f"val_eer_before={result['before_eer'] * 100:.3f}% "
                f"val_eer_after={result['after_eer'] * 100:.3f}% "
                f"val_eer_selected={result['eer'] * 100:.3f}% "
                f"accepted_ratio={result['accepted_ratio']:.4f}"
            )
        selection_tau = tau_by_missing
        print(
            f"Using selector tau by missing modality: "
            f"missing_palm={tau_by_missing[0]:.4f}, missing_vein={tau_by_missing[1]:.4f}"
        )

    print("\n########## Single-Modality Baseline ##########")
    for split_name, loader in loaders.items():
        if split_name in {"full", "palm_only", "random_missing"}:
            feats, labels = extract_single_modality_features(palm_encoder, loader, modality="palm", device=device)
            if feats is not None:
                scores, pair_labels = build_pair_scores(feats, labels)
                eval_with_metrics(scores, pair_labels, name=f"Palm Encoder - {split_name}")
        if split_name in {"full", "vein_only", "random_missing"}:
            feats, labels = extract_single_modality_features(vein_encoder, loader, modality="vein", device=device)
            if feats is not None:
                scores, pair_labels = build_pair_scores(feats, labels)
                eval_with_metrics(scores, pair_labels, name=f"Vein Encoder - {split_name}")

    print("\n########## DyMo Without Recovered Selection ##########")
    for split_name, loader in loaders.items():
        feats, labels = extract_dymo_real_only(model, loader, device=device)
        scores, pair_labels = build_pair_scores(feats, labels)
        eval_with_metrics(scores, pair_labels, name=f"DyMo Real-Only - {split_name}")

    print("\n########## DyMo With Recovered Selection ##########")
    for split_name, loader in loaders.items():
        feats, labels, selected_ratio = extract_dymo_selected(
            selector,
            loader,
            stats=stats,
            device=device,
            selection_mode=args.selection_mode,
            temperature=args.selection_temperature,
            selection_tau=selection_tau,
            quality_mode=args.quality_mode,
        )
        scores, pair_labels = build_pair_scores(feats, labels)
        eval_with_metrics(scores, pair_labels, name=f"DyMo Selected - {split_name}")
        print(f"Recovered modality accepted ratio: {selected_ratio:.4f}")


if __name__ == "__main__":
    main()
