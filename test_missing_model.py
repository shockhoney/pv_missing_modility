import argparse

import torch
from tqdm import tqdm

from models.backbones import build_encoder
from models.missing_model import MissingModalityRecognizer
from utils.checkpoint import safe_torch_load
from utils.datasets_txt import MissingPairTxtDataset
from utils.evaluation import format_gallery_probe_metrics, gallery_probe_metrics
from utils.head import ArcFace
from utils.preprocess import build_palm_transform, build_vein_transform
from utils.runtime import build_data_loader, resolve_device, set_random_seed
from utils.scenarios import PALMPRINT_MISSING, PALMVEIN_MISSING, SSFD_SCENARIOS


SPLITS = SSFD_SCENARIOS
DIAGNOSTICS = {
    PALMPRINT_MISSING: (
        ("available_only_vs_complete_gallery", "projected_f_vein", "z"),
        ("recovered_palm_vs_palm_gallery", "generated_palm", "f_palm"),
    ),
    PALMVEIN_MISSING: (
        ("available_only_vs_complete_gallery", "projected_f_palm", "z"),
        ("recovered_vein_vs_vein_gallery", "generated_vein", "f_vein"),
    ),
}
PROJECTED_FEATURES = {
    "projected_f_palm": "f_palm",
    "projected_f_vein": "f_vein",
}


def build_model(ckpt, device):
    ckpt_args = ckpt.get("args", {})
    dim = ckpt_args.get("embedding_size", 256)
    input_size = ckpt_args.get("input_size", 224)
    num_classes = ckpt["num_classes"]
    state = ckpt["model"]
    if any(key.startswith(("p2v.net.", "v2p.net.")) for key in state):
        raise ValueError("This checkpoint uses the removed MLP recovery model; train a diffusion checkpoint first")
    palm_encoder = build_encoder("palm", input_channel=3, embedding_size=dim).to(device)
    vein_encoder = build_encoder("vein", input_channel=3, embedding_size=dim).to(device)
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
        heads=ckpt_args.get("attn_heads", 4),
        reduction=ckpt_args.get("channel_reduction", 4),
        arcface_s=ckpt_args.get("arcface_s", 32.0),
        arcface_m=ckpt_args.get("arcface_m", 0.25),
        palm_teacher=palm_teacher,
        vein_teacher=vein_teacher,
        gate_init=ckpt_args.get("missing_gate_init", 0.0),
        diffusion_steps=ckpt_args.get("diffusion_steps", 100),
        ddim_steps=ckpt_args.get("ddim_steps", 20),
        diffusion_base_channels=ckpt_args.get("diffusion_base_channels", 64),
        diffusion_time_dim=ckpt_args.get("diffusion_time_dim", 128),
        diffusion_dropout=ckpt_args.get("diffusion_dropout", 0.0),
        diffusion_stats_momentum=ckpt_args.get("diffusion_stats_momentum", 0.99),
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
    return build_data_loader(dataset, batch_size, num_workers)


@torch.no_grad()
def extract_embeddings(model, loader, description, device, keys=("z",)):
    embeddings = {key: [] for key in keys}
    labels = []
    for palm, vein, batch_labels, mask in tqdm(loader, desc=description, dynamic_ncols=True, leave=False):
        palm = palm.to(device, non_blocking=True)
        vein = vein.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        output = model(palm, vein, mask=mask)
        for key in keys:
            source_key = PROJECTED_FEATURES.get(key)
            value = model.fusion.available_only(output[source_key]) if source_key else output[key]
            embeddings[key].append(value.cpu())
        labels.append(batch_labels)
    return {key: torch.cat(values) for key, values in embeddings.items()}, torch.cat(labels)


def print_metrics(split_name, metrics):
    print(f"\n[{split_name}]")
    print("\n".join(format_gallery_probe_metrics(metrics)))


def evaluate_embeddings(gallery, gallery_labels, probes, probe_labels, args):
    return gallery_probe_metrics(
        gallery,
        gallery_labels,
        probes,
        probe_labels,
        topk=args.top_k,
        far_points=args.far_points,
    )


def evaluate(args):
    device = resolve_device(args.device)
    set_random_seed(args.seed)
    model, img_size = load_model(args.ckpt, device)
    gallery_loader = build_loader(args.gallery_list, None, img_size, args.batch_size, args.num_workers)
    gallery_keys = ("z", "f_palm", "f_vein") if args.diagnostics else ("z",)
    gallery_embeddings, gallery_labels = extract_embeddings(
        model, gallery_loader, "Build gallery", device, gallery_keys
    )
    if args.diagnostics:
        scale = model.fusion.available_fusion.residual_scale.detach().cpu()
        print(f"[diagnostic] residual_scale={scale.item():.6f}, effective_scale={torch.tanh(scale).item():.6f}")

    for split_name in SPLITS:
        loader = build_loader(args.protocol_list, split_name, img_size, args.batch_size, args.num_workers)
        if loader is None:
            continue
        diagnostic_specs = DIAGNOSTICS.get(split_name, ()) if args.diagnostics else ()
        probe_keys = tuple(dict.fromkeys(("z", *(spec[1] for spec in diagnostic_specs))))
        probe_embeddings, probe_labels = extract_embeddings(
            model, loader, f"Evaluate {split_name}", device, probe_keys
        )
        print_metrics(
            split_name,
            evaluate_embeddings(
                gallery_embeddings["z"], gallery_labels, probe_embeddings["z"], probe_labels, args
            ),
        )
        for diagnostic_name, probe_key, gallery_key in diagnostic_specs:
            print_metrics(
                f"{split_name}/{diagnostic_name}",
                evaluate_embeddings(
                    gallery_embeddings[gallery_key],
                    gallery_labels,
                    probe_embeddings[probe_key],
                    probe_labels,
                    args,
                ),
            )


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Evaluate missing-modality recognizer")
    parser.add_argument("--gallery_list", default="data_txt/tongji/ssfd_gallery_full.txt")
    parser.add_argument("--protocol_list", default="data_txt/tongji/ssfd_test_protocol.txt")
    parser.add_argument("--ckpt", default="outputs/missing_model/best.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--far_points", type=float, nargs="+", default=[1e-4, 1e-5])
    return parser.parse_args(argv)


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
