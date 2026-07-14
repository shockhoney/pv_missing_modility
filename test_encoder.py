import argparse

import torch
from tqdm import tqdm

from utils.checkpoint import load_encoder_from_checkpoint
from utils.datasets_txt import SingleModalityFromPairDataset
from utils.evaluation import format_gallery_probe_metrics, gallery_probe_metrics
from utils.preprocess import build_modality_transform
from utils.runtime import build_data_loader, resolve_device
from utils.scenarios import PALMPRINT_MISSING, PALMVEIN_MISSING


def build_loader(protocol_list: str, modality: str, split_name: str, img_size: int, batch_size: int, num_workers: int):
    transform = build_modality_transform(modality, img_size)
    dataset = SingleModalityFromPairDataset(
        protocol_list,
        modality=modality,
        transform=transform,
        split_filter=split_name,
    )
    if len(dataset) == 0:
        return None
    return build_data_loader(dataset, batch_size, num_workers)


@torch.no_grad()
def extract_embeddings(encoder, loader, modality: str, device):
    embeddings, labels = [], []
    for images, batch_labels in tqdm(loader, desc=f"Extract {modality}", dynamic_ncols=True, leave=False):
        images = images.to(device, non_blocking=True)
        features = encoder(images)
        embeddings.append(features.cpu())
        labels.append(batch_labels)
    return torch.cat(embeddings), torch.cat(labels)


def evaluate_encoder(modality: str, ckpt_path: str, args, device):
    encoder = load_encoder_from_checkpoint(ckpt_path, modality, args.encoder_dim, device)
    gallery_loader = build_loader(
        args.gallery_list,
        modality,
        None,
        args.input_size,
        args.batch_size,
        args.num_workers,
    )
    gallery_embeddings, gallery_labels = extract_embeddings(encoder, gallery_loader, modality, device)
    split_name = PALMVEIN_MISSING if modality == "palm" else PALMPRINT_MISSING
    loader = build_loader(args.protocol_list, modality, split_name, args.input_size, args.batch_size, args.num_workers)
    embeddings, labels = extract_embeddings(encoder, loader, modality, device)
    metrics = gallery_probe_metrics(
        gallery_embeddings,
        gallery_labels,
        embeddings,
        labels,
        topk=args.top_k,
        far_points=args.far_points,
    )
    print(f"\n===== {modality.capitalize()} Encoder =====")
    print("\n".join(format_gallery_probe_metrics(metrics)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Evaluate single-modality encoders")
    parser.add_argument("--gallery_list", default="data_txt/tongji/ssfd_train_full.txt")
    parser.add_argument("--protocol_list", default="data_txt/tongji/ssfd_test_protocol.txt")
    parser.add_argument("--modality", type=str, choices=["palm", "vein"], default="palm")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--encoder_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--far_points", type=float, nargs="+", default=[1e-4, 1e-5])
    args = parser.parse_args(argv)
    if args.ckpt is None:
        args.ckpt = f"outputs/encoders/{args.modality}_best.pth"
    return args


def main():
    args = parse_args()
    device = resolve_device(args.device)
    evaluate_encoder(args.modality, args.ckpt, args, device)


if __name__ == "__main__":
    main()
