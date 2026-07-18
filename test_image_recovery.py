import argparse

import torch
from tqdm import tqdm

from models.image_recovery import (
    ARCHITECTURE_VERSION,
    BidirectionalImageRecovery,
    load_image_recovery_state,
)
from utils.checkpoint import load_encoder_from_checkpoint
from utils.checkpoint_io import file_sha256, safe_torch_load
from utils.datasets_txt import CrossModalRecoveryDataset
from utils.evaluation import (
    format_gallery_probe_metrics,
    gallery_probe_scores,
    score_matrix_metrics,
)
from utils.preprocess import build_palm_transform, build_vein_transform
from utils.runtime import build_data_loader, resolve_device, set_random_seed
from utils.scenarios import PALMPRINT_MISSING, PALMVEIN_MISSING


SCENARIOS = (PALMPRINT_MISSING, PALMVEIN_MISSING)


def build_loader(list_path, split, args):
    dataset = CrossModalRecoveryDataset(
        list_path,
        build_palm_transform(args.input_size),
        build_vein_transform(args.input_size),
        split_filter=split,
    )
    if not dataset:
        return None
    return build_data_loader(dataset, args.batch_size, args.num_workers)


@torch.inference_mode()
def extract_embeddings(recovery, palm_encoder, vein_encoder, loader, device, description):
    values = {key: [] for key in ("palm", "vein", "generated_palm", "generated_vein")}
    labels = []
    for batch in tqdm(loader, desc=description, dynamic_ncols=True, leave=False):
        palm = batch["palm"].to(device, non_blocking=True)
        vein = batch["vein"].to(device, non_blocking=True)
        generated = recovery(
            batch["vein_as_palm"].to(device, non_blocking=True),
            batch["palm_as_vein"].to(device, non_blocking=True),
        )
        values["palm"].append(palm_encoder(palm).cpu())
        values["vein"].append(vein_encoder(vein).cpu())
        values["generated_palm"].append(
            palm_encoder(generated["generated_palm"]).cpu()
        )
        values["generated_vein"].append(
            vein_encoder(generated["generated_vein"]).cpu()
        )
        labels.append(batch["label"])
    return {key: torch.cat(items) for key, items in values.items()}, torch.cat(labels)


def domain_scores(gallery, gallery_labels, probes, probe_key, gallery_key):
    return gallery_probe_scores(gallery[gallery_key], gallery_labels, probes[probe_key])


def metrics(scores, candidate_labels, probe_labels, args):
    return score_matrix_metrics(
        scores,
        candidate_labels,
        probe_labels,
        topk=args.top_k,
        far_points=args.far_points,
    )


def print_metrics(name, result):
    print(f"\n[{name}]")
    print("\n".join(format_gallery_probe_metrics(result)))


def identity_roll(embeddings, labels, shift):
    """Circularly assign every identity's recovered probes to another identity."""
    unique = labels.unique(sorted=True)
    output = torch.empty_like(embeddings)
    for destination_index, destination_label in enumerate(unique):
        source_label = unique[(destination_index + shift) % unique.numel()]
        destination = torch.nonzero(labels == destination_label, as_tuple=False).flatten()
        source = torch.nonzero(labels == source_label, as_tuple=False).flatten()
        if destination.numel() != source.numel():
            raise ValueError("Identity-level shuffling requires equal probe counts per identity")
        output[destination] = embeddings[source]
    return output


def evaluate_scenario(
    scenario,
    gallery,
    gallery_labels,
    probes,
    probe_labels,
    complete_probes,
    complete_labels,
    alpha,
    args,
):
    if not torch.equal(probe_labels, complete_labels):
        raise ValueError("Missing and complete probe rows must be aligned for oracle diagnostics")

    if scenario == PALMPRINT_MISSING:
        available_key = "vein"
        recovery_key = "generated_palm"
        oracle_key = "palm"
    elif scenario == PALMVEIN_MISSING:
        available_key = "palm"
        recovery_key = "generated_vein"
        oracle_key = "vein"
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    available_scores, candidate_labels = domain_scores(
        gallery, gallery_labels, probes, available_key, available_key
    )
    recovery_scores, recovery_labels = domain_scores(
        gallery, gallery_labels, probes, recovery_key, recovery_key
    )
    oracle_scores, oracle_labels = domain_scores(
        gallery, gallery_labels, complete_probes, oracle_key, oracle_key
    )
    if not torch.equal(candidate_labels, recovery_labels) or not torch.equal(
        candidate_labels, oracle_labels
    ):
        raise ValueError("Gallery candidate order differs between score domains")

    fused_scores = available_scores + alpha * recovery_scores
    print_metrics(f"{scenario}/available_only", metrics(
        available_scores, candidate_labels, probe_labels, args
    ))
    print_metrics(f"{scenario}/recovery_domain_only", metrics(
        recovery_scores, candidate_labels, probe_labels, args
    ))
    print_metrics(f"{scenario}/fused(alpha={alpha:g})", metrics(
        fused_scores, candidate_labels, probe_labels, args
    ))
    print_metrics(f"{scenario}/oracle_real_missing", metrics(
        available_scores + alpha * oracle_scores, candidate_labels, probe_labels, args
    ))

    shuffled_top1 = []
    max_shuffles = min(args.shuffle_trials, int(probe_labels.unique().numel()) - 1)
    for shift in range(1, max_shuffles + 1):
        shuffled = identity_roll(probes[recovery_key], probe_labels, shift)
        shuffled_scores, shuffled_labels = gallery_probe_scores(
            gallery[recovery_key], gallery_labels, shuffled
        )
        shuffled_result = metrics(
            available_scores + alpha * shuffled_scores,
            shuffled_labels,
            probe_labels,
            args,
        )
        shuffled_top1.append(shuffled_result["topk"][1])
    if shuffled_top1:
        shuffled_tensor = torch.tensor(shuffled_top1)
        print(
            f"[{scenario}/shuffled_recovery] trials={len(shuffled_top1)}, "
            f"Top-1 mean={shuffled_tensor.mean().item() * 100:.2f}%, "
            f"std={shuffled_tensor.std(unbiased=False).item() * 100:.2f}%, "
            f"range=[{shuffled_tensor.min().item() * 100:.2f}%, "
            f"{shuffled_tensor.max().item() * 100:.2f}%]"
        )


def evaluate(args):
    device = resolve_device(args.device)
    set_random_seed(args.seed)
    checkpoint = safe_torch_load(args.ckpt, device)
    checkpoint_args = checkpoint.get("args", {})
    saved_version = checkpoint.get("architecture_version")
    if saved_version not in (None, ARCHITECTURE_VERSION):
        raise ValueError(
            f"Unsupported recovery architecture {saved_version!r}; expected {ARCHITECTURE_VERSION!r}"
        )
    for modality, path in (("palm", args.palm_ckpt), ("vein", args.vein_ckpt)):
        expected = checkpoint.get(f"{modality}_encoder_sha256")
        if expected is not None:
            actual = file_sha256(path)
            if actual != expected:
                raise ValueError(
                    f"{modality} encoder checkpoint differs from the recovery checkpoint teacher: "
                    f"expected SHA-256 {expected}, got {actual}"
                )

    channels = checkpoint_args.get("recovery_channels", 32)
    blocks = checkpoint_args.get("recovery_blocks", 3)
    recovery = BidirectionalImageRecovery(channels=channels, blocks=blocks).to(device)
    load_image_recovery_state(recovery, checkpoint)
    recovery.eval()

    palm_encoder = load_encoder_from_checkpoint(
        args.palm_ckpt, "palm", args.embedding_size, device
    )
    vein_encoder = load_encoder_from_checkpoint(
        args.vein_ckpt, "vein", args.embedding_size, device
    )

    saved_weights = checkpoint.get("fusion_weights", {})
    palm_alpha = args.palm_missing_alpha
    vein_alpha = args.vein_missing_alpha
    if palm_alpha is None:
        palm_alpha = float(saved_weights.get(PALMPRINT_MISSING, 0.15))
    if vein_alpha is None:
        vein_alpha = float(saved_weights.get(PALMVEIN_MISSING, 0.15))
    print(
        f"[Info] checkpoint_epoch={checkpoint.get('epoch', 'unknown')} "
        f"fusion_weights=({PALMPRINT_MISSING}:{palm_alpha:g}, "
        f"{PALMVEIN_MISSING}:{vein_alpha:g})"
    )

    gallery_loader = build_loader(args.gallery_list, None, args)
    complete_loader = build_loader(args.protocol_list, "complete", args)
    gallery, gallery_labels = extract_embeddings(
        recovery, palm_encoder, vein_encoder, gallery_loader, device, "Build recovery gallery"
    )
    complete, complete_labels = extract_embeddings(
        recovery, palm_encoder, vein_encoder, complete_loader, device, "Build oracle probes"
    )
    for scenario, alpha in (
        (PALMPRINT_MISSING, palm_alpha),
        (PALMVEIN_MISSING, vein_alpha),
    ):
        loader = build_loader(args.protocol_list, scenario, args)
        probes, probe_labels = extract_embeddings(
            recovery, palm_encoder, vein_encoder, loader, device, f"Evaluate {scenario}"
        )
        evaluate_scenario(
            scenario,
            gallery,
            gallery_labels,
            probes,
            probe_labels,
            complete,
            complete_labels,
            alpha,
            args,
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Evaluate recovery-domain missing-modality recognition")
    parser.add_argument("--gallery_list", default="data_txt/tongji/ssfd_gallery_full.txt")
    parser.add_argument("--protocol_list", default="data_txt/tongji/ssfd_test_protocol.txt")
    parser.add_argument("--ckpt", default="outputs/image_recovery/best.pth")
    parser.add_argument("--palm_ckpt", default="outputs/encoders/palm_best.pth")
    parser.add_argument("--vein_ckpt", default="outputs/encoders/vein_best.pth")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--embedding_size", type=int, default=256)
    parser.add_argument("--recovery_channels", type=int, default=32)
    parser.add_argument("--recovery_blocks", type=int, default=3)
    parser.add_argument("--palm_missing_alpha", type=float, default=None)
    parser.add_argument("--vein_missing_alpha", type=float, default=None)
    parser.add_argument("--shuffle_trials", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--far_points", type=float, nargs="+", default=[1e-3, 1e-4])
    return parser.parse_args(argv)


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
