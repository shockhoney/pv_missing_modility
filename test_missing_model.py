import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.backbones import build_encoder
from models.missing_model import MissingModalityRecognizer
from utils.checkpoint import safe_torch_load
from utils.datasets_txt import MissingPairTxtDataset
from utils.head import ArcFace
from utils.preprocess import build_palm_transform, build_vein_transform


SPLITS = ("complete", "palmprint_missing", "palmvein_missing")


def build_model(ckpt, device):
    ckpt_args = ckpt.get("args", {})
    dim = ckpt_args.get("embedding_size", 256)
    input_size = ckpt_args.get("input_size", 224)
    num_classes = ckpt["num_classes"]
    state = ckpt["model"]
    palm_encoder = build_encoder("palm", input_channel=3, input_size=input_size, embedding_size=dim).to(device)
    vein_encoder = build_encoder("vein", input_channel=3, input_size=input_size, embedding_size=dim).to(device)
    palm_teacher = None
    vein_teacher = None
    if any(key.startswith("palm_teacher.") for key in state):
        palm_teacher = ArcFace(
            dim,
            num_classes,
            ckpt_args.get("palm_teacher_s", ckpt_args.get("arcface_s", 32.0)),
            ckpt_args.get("palm_teacher_m", ckpt_args.get("arcface_m", 0.25)),
        ).to(device)
    if any(key.startswith("vein_teacher.") for key in state):
        vein_teacher = ArcFace(
            dim,
            num_classes,
            ckpt_args.get("vein_teacher_s", ckpt_args.get("arcface_s", 32.0)),
            ckpt_args.get("vein_teacher_m", ckpt_args.get("arcface_m", 0.25)),
        ).to(device)
    model = MissingModalityRecognizer(
        palm_encoder,
        vein_encoder,
        num_classes,
        dim=dim,
        cmft_hidden=ckpt_args.get("cmft_hidden", 1024),
        heads=ckpt_args.get("attn_heads", 4),
        reduction=ckpt_args.get("channel_reduction", 4),
        arcface_s=ckpt_args.get("arcface_s", 32.0),
        arcface_m=ckpt_args.get("arcface_m", 0.25),
        palm_teacher=palm_teacher,
        vein_teacher=vein_teacher,
        gate_init=ckpt_args.get("missing_gate_init", -8.0),
    ).to(device)
    state = {
        key: value
        for key, value in state.items()
        if not key.startswith(("palm_teacher_encoder.", "vein_teacher_encoder."))
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"palm_missing_gate", "vein_missing_gate"}
    if unexpected or any(key not in allowed_missing for key in missing):
        raise RuntimeError(f"Invalid missing-model checkpoint. Missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, input_size


def load_model(ckpt_path, device):
    return build_model(safe_torch_load(ckpt_path, device), device)


def build_loader(protocol_list, split_name, img_size, batch_size, num_workers):
    dataset = MissingPairTxtDataset(
        protocol_list,
        build_palm_transform(img_size),
        build_vein_transform(img_size),
        split_filter=split_name,
    )
    if len(dataset) == 0:
        return None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate_split(model, loader, split_name, device):
    correct = total = 0
    for palm, vein, labels, mask in tqdm(loader, desc=f"Evaluate {split_name}", dynamic_ncols=True, leave=False):
        palm = palm.to(device, non_blocking=True)
        vein = vein.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits = model(palm, vein, mask=mask)["logits"]
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1), total


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model, img_size = load_model(args.ckpt, device)
    for split_name in SPLITS:
        loader = build_loader(args.protocol_list, split_name, img_size, args.batch_size, args.num_workers)
        if loader is None:
            continue
        acc, total = evaluate_split(model, loader, split_name, device)
        print(f"{split_name}: Samples={total}, Recognition Rate (%): {acc * 100:.2f}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Evaluate missing-modality recognizer")
    parser.add_argument("--protocol_list", default="data_txt/cumt/ssfd_test_protocol.txt")
    parser.add_argument("--ckpt", default="outputs/missing_model/best.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args(argv)


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
