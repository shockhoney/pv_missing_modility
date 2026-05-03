import argparse
import os
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.dymo import PalmVeinDynamicTransformer
from utils.datasets_txt import MissingPairTxtDataset
from utils.dymo_stats import DEFAULT_SUBSET_MASKS, build_subset2id, compute_subset_statistics


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


@torch.no_grad()
def collect_subset_embeddings(
    model: PalmVeinDynamicTransformer,
    loader: DataLoader,
    device,
    subset_mask: Tuple[bool, bool],
):
    all_feat = []
    all_labels = []
    missing_mask = torch.tensor([subset_mask], dtype=torch.bool, device=device)

    for palm, vein, labels, _ in tqdm(loader, desc=f"Subset-{subset_mask}", dynamic_ncols=True, leave=False):
        palm = palm.to(device, non_blocking=True)
        vein = vein.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        mask = missing_mask.expand(labels.size(0), -1)
        output = model(palm, vein, missing_mask=mask, use_recovered_mask=torch.zeros_like(mask))
        all_feat.append(output["embedding"].cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_feat, dim=0), torch.cat(all_labels, dim=0)


def main():
    parser = argparse.ArgumentParser("Prepare DyMo subset Gaussian statistics")
    parser.add_argument("--train_full_list", type=str, default="data_txt/polyu_train_full.txt")
    parser.add_argument("--checkpoint", type=str, default="outputs_dymo/dymo/dymo_best.pth")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
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

    if not args.output_path:
        output_dir = os.path.join(os.path.dirname(args.checkpoint), "gaussian")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "subset_gaussian.pt")
    else:
        output_path = args.output_path
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    tf = get_transforms(args.input_size)
    dataset = MissingPairTxtDataset(args.train_full_list, transform_palm=tf, transform_vein=tf)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

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
    model.eval()

    subset2id = build_subset2id(DEFAULT_SUBSET_MASKS)
    class_prototypes = []
    overall_centroids = []
    class_dist_std = []
    overall_dist_std = []

    for subset_mask in DEFAULT_SUBSET_MASKS:
        feats, labels = collect_subset_embeddings(model, loader, device, subset_mask)
        stats = compute_subset_statistics(feats, labels, num_classes=num_classes)
        class_prototypes.append(stats["class_prototypes"])
        overall_centroids.append(stats["overall_centroid"])
        class_dist_std.append(stats["class_dist_std"])
        overall_dist_std.append(stats["overall_dist_std"])

    payload: Dict = {
        "subset_masks": DEFAULT_SUBSET_MASKS,
        "subset2id": subset2id,
        "class_prototypes": torch.stack(class_prototypes, dim=0),
        "overall_centroids": torch.stack(overall_centroids, dim=0),
        "class_dist_std": torch.stack(class_dist_std, dim=0),
        "overall_dist_std": torch.stack(overall_dist_std, dim=0),
    }
    torch.save(payload, output_path)
    print(f"[Info] saved DyMo stats to {output_path}")


if __name__ == "__main__":
    main()
