"""Evaluate GIPSSR-Net under deterministic partial missing-modality rates.

The protocol follows SSFD-Net's definition: for a target missing rate, only a
subset of the otherwise complete test probes loses the target modality.  The
remaining probes keep both modalities.  Training is therefore not repeated for
each rate; the retained fixed-full-train checkpoints are evaluated directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from models.gipssr import ARCHITECTURE_VERSION, GIPSSRNet
from utils.checkpoint import load_encoder_from_checkpoint
from utils.checkpoint_io import file_sha256, safe_torch_load
from utils.comparison_protocols import DATASETS, get_protocol_spec
from utils.evaluation import score_matrix_metrics
from utils.gipssr_feature_extraction import load_or_extract_paired_spatial_cache
from utils.runtime import resolve_device, set_random_seed
from utils.scenarios import PALMPRINT_MISSING, PALMVEIN_MISSING


EXPERIMENT_VERSION = "gipssr_partial_missing_rates_seed42_v2"
MASK_VERSION = "shared_nested_probe_mask_sha256_seed_v1"
RATES = (0.10, 0.20, 0.50, 0.80, 1.00)
SEED = 42
SCENARIOS = (PALMPRINT_MISSING, PALMVEIN_MISSING)
METRICS = ("eer", "top1", "tar_1e-3", "tar_1e-4")
CHECKPOINT_ROOT = Path("outputs/gipssr/ablations/checkpoints")
REFERENCE_RESULT_ROOT = Path("outputs/gipssr/ablations/results")
DEFAULT_OUTPUT_ROOT = Path("outputs/gipssr/missing_rate_experiments")
DEFAULT_CACHE_ROOT = Path("outputs/gipssr/cache/missing_rate_experiments")


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def rate_key(rate: float) -> str:
    return f"{int(round(rate * 100))}"


def half_up_missing_count(total: int, rate: float) -> int:
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be in [0, 1]")
    return min(total, int(math.floor(total * rate + 0.5)))


def _stable_mask_seed(dataset: str, seed: int) -> int:
    digest = hashlib.sha256(
        f"{MASK_VERSION}:{dataset}:{int(seed)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def mask_sha256(mask: torch.Tensor) -> str:
    if mask.ndim != 1 or mask.dtype != torch.bool:
        raise ValueError("mask must be a one-dimensional bool tensor")
    return hashlib.sha256(bytes(mask.to(torch.uint8).tolist())).hexdigest()


def nested_missing_masks(
    total: int,
    rates: tuple[float, ...] = RATES,
    *,
    dataset: str,
    seed: int,
    sample_order: list | tuple | None = None,
) -> dict[str, dict]:
    if tuple(sorted(set(rates))) != tuple(rates):
        raise ValueError("rates must be unique and sorted")
    if sample_order is None:
        sample_order = list(range(total))
    if len(sample_order) != total:
        raise ValueError("sample_order length must equal total")
    salt = _stable_mask_seed(dataset, seed)
    ranked = []
    for index, sample in enumerate(sample_order):
        canonical = json.dumps(sample, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(
            f"{MASK_VERSION}:{salt}:{canonical}".encode("utf-8")
        ).digest()
        ranked.append((digest, index))
    order = [index for _, index in sorted(ranked)]
    order_sha256 = hashlib.sha256(
        json.dumps(order, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    masks: dict[str, dict] = {}
    previous = torch.zeros(total, dtype=torch.bool)
    for rate in rates:
        count = half_up_missing_count(total, rate)
        mask = torch.zeros(total, dtype=torch.bool)
        mask[order[:count]] = True
        if torch.any(previous & ~mask):
            raise RuntimeError("Missing-rate masks are not nested")
        key = rate_key(rate)
        masks[key] = {
            "requested_rate": float(rate),
            "missing_count": count,
            "total_probe_count": total,
            "actual_rate": count / total,
            "selection_order_sha256": order_sha256,
            "mask_sha256": mask_sha256(mask),
            "missing_indices": torch.nonzero(mask, as_tuple=False).flatten().tolist(),
            "mask": mask,
        }
        previous = mask
    return masks


def cohort_normalize_scores(scores: torch.Tensor, target_std: float = 0.05) -> torch.Tensor:
    """Match the row-wise calibration used by CUEF's final fused scores."""

    if scores.ndim != 2 or scores.size(1) < 2:
        raise ValueError("scores must have shape [probes, identities>=2]")
    centered = scores - scores.mean(dim=1, keepdim=True)
    spread = centered.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-4)
    return float(target_std) * centered / spread


def complete_multimodal_scores(
    palm_direction_scores: torch.Tensor,
    vein_direction_scores: torch.Tensor,
) -> torch.Tensor:
    """Symmetrically fuse the two deployed directional inference paths."""

    if palm_direction_scores.shape != vein_direction_scores.shape:
        raise ValueError("Directional score matrices must have equal shape")
    return cohort_normalize_scores(
        0.5 * (palm_direction_scores + vein_direction_scores)
    )


def mix_score_rows(
    complete_scores: torch.Tensor,
    missing_scores: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if complete_scores.shape != missing_scores.shape:
        raise ValueError("Complete and missing scores must have equal shape")
    if mask.shape != (complete_scores.size(0),) or mask.dtype != torch.bool:
        raise ValueError("Mask shape or dtype is incompatible with scores")
    return torch.where(mask[:, None], missing_scores, complete_scores)


def checkpoint_path(dataset: str, seed: int) -> Path:
    return CHECKPOINT_ROOT / dataset / f"seed_{seed}" / "full" / "best.pth"


def reference_result_path(dataset: str, seed: int) -> Path:
    return REFERENCE_RESULT_ROOT / dataset / f"seed_{seed}" / "full.json"


def verify_file(path: str | Path, expected_sha256: str, label: str) -> None:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected_sha256}")


def load_full_model(checkpoint: dict, device: torch.device) -> GIPSSRNet:
    if checkpoint.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("Partial-rate evaluation requires the final GIPSSR-Net architecture")
    if checkpoint.get("training_stage") != "fixed_full_train":
        raise ValueError("Partial-rate evaluation requires a fixed-full-train checkpoint")
    saved_args = checkpoint.get("args", {})
    ablation = saved_args.get("ablation", checkpoint.get("ablation", "full"))
    if ablation != "full":
        raise ValueError(f"Expected full model checkpoint, got ablation={ablation!r}")
    config = checkpoint["configuration"]
    model = GIPSSRNet(
        input_dim=int(config["input_dim"]),
        shared_dim=int(config["shared_dim"]),
        specific_dim=int(config["specific_dim"]),
        min_recovery_weight=float(saved_args.get("min_recovery_weight", 0.15)),
        max_recovery_weight=float(config["max_recovery_weight"]),
        retrieval_dropout=float(saved_args.get("retrieval_dropout", 0.10)),
        branch_floor=float(saved_args.get("branch_floor", 0.0)),
        transformer_layers=int(config["transformer_layers"]),
        transformer_heads=int(config["transformer_heads"]),
        dropout=float(config["dropout"]),
        topk_candidates=int(saved_args.get("topk_candidates", 5)),
        role_queries=int(saved_args.get("role_queries", 4)),
        candidate_dropout=float(saved_args.get("candidate_dropout", 0.20)),
        max_refinement=float(saved_args.get("max_refinement", 0.25)),
        ablation="full",
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def _to_device(values: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved = {
        key: value.to(device=device, non_blocking=True)
        for key, value in values.items()
    }
    moved["labels"] = moved["labels"].long()
    return moved


def load_test_features(dataset: str, args, device: torch.device) -> tuple[dict, dict, dict]:
    spec = get_protocol_spec(dataset)
    verify_file(spec.test_gallery_list, spec.test_gallery_sha256, "test gallery")
    verify_file(spec.test_protocol_list, spec.test_protocol_sha256, "test protocol")
    verify_file(spec.palm_checkpoint, spec.palm_checkpoint_sha256, "palm encoder")
    verify_file(spec.vein_checkpoint, spec.vein_checkpoint_sha256, "vein encoder")

    palm_encoder = load_encoder_from_checkpoint(
        spec.palm_checkpoint, "palm", args.embedding_size, device
    )
    vein_encoder = load_encoder_from_checkpoint(
        spec.vein_checkpoint, "vein", args.embedding_size, device
    )
    cache_dir = args.cache_root / dataset
    gallery, gallery_metadata = load_or_extract_paired_spatial_cache(
        str(cache_dir / "test_gallery_spatial.pt"),
        spec.test_gallery_list,
        None,
        palm_encoder,
        vein_encoder,
        spec.palm_checkpoint,
        spec.vein_checkpoint,
        device,
        args.input_size,
        args.embedding_size,
        args.extract_batch_size,
        args.num_workers,
        f"Extract {dataset} complete test gallery",
        force=args.force_recache,
    )
    probes, probe_metadata = load_or_extract_paired_spatial_cache(
        str(cache_dir / "test_probe_spatial.pt"),
        spec.test_protocol_list,
        "complete",
        palm_encoder,
        vein_encoder,
        spec.palm_checkpoint,
        spec.vein_checkpoint,
        device,
        args.input_size,
        args.embedding_size,
        args.extract_batch_size,
        args.num_workers,
        f"Extract {dataset} complete test probes",
        force=args.force_recache,
    )
    del palm_encoder, vein_encoder
    if probes["labels"].numel() != spec.test_probes_per_scenario:
        raise ValueError(
            f"{dataset} probe count {probes['labels'].numel()} != "
            f"{spec.test_probes_per_scenario}"
        )
    if gallery["labels"].unique().numel() != spec.test_gallery_identities:
        raise ValueError("Test gallery identity count differs from locked protocol")
    if set(gallery_metadata) != set(probe_metadata):
        raise RuntimeError("Gallery/probe cache metadata schemas differ")
    return _to_device(gallery, device), _to_device(probes, device), {
        "gallery": gallery_metadata,
        "probes": probe_metadata,
    }


def basic_metrics(metrics: dict) -> dict[str, float]:
    topk = metrics["topk"]
    tar = metrics["tar_at_far"]
    return {
        "eer": float(metrics["eer"]),
        "top1": float(topk[1] if 1 in topk else topk["1"]),
        "tar_1e-3": float(tar[1e-3] if 1e-3 in tar else tar["0.001"]),
        "tar_1e-4": float(tar[1e-4] if 1e-4 in tar else tar["0.0001"]),
    }


def evaluate_scores(
    scores: torch.Tensor,
    candidate_labels: torch.Tensor,
    probe_labels: torch.Tensor,
) -> dict:
    return score_matrix_metrics(
        scores.detach().float().cpu(),
        candidate_labels.detach().cpu(),
        probe_labels.detach().cpu(),
        topk=(1, 5),
        far_points=(1e-3, 1e-4),
        warn_far_resolution=False,
    )


def _json_metric(block: dict, name: str) -> float:
    if name == "eer":
        return float(block["eer"])
    if name == "top1":
        return float(block["topk"].get("1", block["topk"].get(1)))
    far = "0.001" if name == "tar_1e-3" else "0.0001"
    return float(block["tar_at_far"].get(far, block["tar_at_far"].get(float(far))))


def verify_against_retained_100_percent(
    dataset: str,
    seed: int,
    checkpoint_file: Path,
    results: dict,
) -> str:
    path = reference_result_path(dataset, seed)
    reference = json.loads(path.read_text(encoding="utf-8"))
    checkpoint_hash = file_sha256(checkpoint_file)
    if reference.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Retained result/checkpoint SHA-256 mismatch")
    spec = get_protocol_spec(dataset)
    if reference.get("gallery_protocol_sha256") != spec.test_gallery_sha256:
        raise ValueError("Retained result gallery protocol mismatch")
    if reference.get("probe_protocol_sha256") != spec.test_protocol_sha256:
        raise ValueError("Retained result probe protocol mismatch")
    for scenario in SCENARIOS:
        expected = reference["results"][scenario]["fused"]
        actual = results[scenario]["100"]
        for metric in METRICS:
            difference = abs(_json_metric(expected, metric) - basic_metrics(actual)[metric])
            if difference > 1e-7:
                raise ValueError(
                    f"{dataset}/seed{seed}/{scenario}/100% {metric} differs "
                    f"from retained result by {difference:g}"
                )
    return file_sha256(path)


@torch.inference_mode()
def evaluate_seed(
    dataset: str,
    seed: int,
    gallery: dict[str, torch.Tensor],
    probes: dict[str, torch.Tensor],
    masks: dict[str, dict],
    device: torch.device,
    mask_seed: int,
) -> dict:
    spec = get_protocol_spec(dataset)
    path = checkpoint_path(dataset, seed)
    checkpoint = safe_torch_load(path, device)
    fingerprints = checkpoint.get("fingerprints", {})
    if fingerprints.get("palm_encoder_sha256") != spec.palm_checkpoint_sha256:
        raise ValueError("Checkpoint palm encoder fingerprint mismatch")
    if fingerprints.get("vein_encoder_sha256") != spec.vein_checkpoint_sha256:
        raise ValueError("Checkpoint vein encoder fingerprint mismatch")
    set_random_seed(seed)
    model = load_full_model(checkpoint, device)
    memory = model.build_gallery_memory(gallery)
    palm_output = model.recover_with_gallery(
        probes["palm"], probes["palm_spatial"], "palm", memory
    )
    vein_output = model.recover_with_gallery(
        probes["vein"], probes["vein_spatial"], "vein", memory
    )
    if not torch.equal(palm_output["candidate_labels"], vein_output["candidate_labels"]):
        raise RuntimeError("Directional gallery label orders differ")
    candidate_labels = palm_output["candidate_labels"]
    palm_scores = palm_output["fused_scores"]
    vein_scores = vein_output["fused_scores"]
    complete_scores = complete_multimodal_scores(palm_scores, vein_scores)
    directional = {
        PALMPRINT_MISSING: vein_scores,
        PALMVEIN_MISSING: palm_scores,
    }
    results: dict[str, dict[str, dict]] = {scenario: {} for scenario in SCENARIOS}
    for scenario in SCENARIOS:
        for key, mask_info in masks.items():
            mixed = mix_score_rows(
                complete_scores,
                directional[scenario],
                mask_info["mask"].to(device),
            )
            results[scenario][key] = evaluate_scores(
                mixed, candidate_labels, probes["labels"]
            )
    complete_metrics = evaluate_scores(
        complete_scores, candidate_labels, probes["labels"]
    )
    reference_sha = verify_against_retained_100_percent(
        dataset, seed, path, results
    )
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "dataset": dataset,
        "protocol_version": spec.protocol_version,
        "model_seed": seed,
        "mask_seed": int(mask_seed),
        "architecture_version": checkpoint["architecture_version"],
        "training_stage": checkpoint["training_stage"],
        "best_epoch": int(checkpoint["best_epoch"]),
        "checkpoint": str(path),
        "checkpoint_sha256": file_sha256(path),
        "reference_100_percent_result": str(reference_result_path(dataset, seed)),
        "reference_100_percent_result_sha256": reference_sha,
        "gallery_protocol_sha256": spec.test_gallery_sha256,
        "probe_protocol_sha256": spec.test_protocol_sha256,
        "palm_encoder_sha256": spec.palm_checkpoint_sha256,
        "vein_encoder_sha256": spec.vein_checkpoint_sha256,
        "complete_score_rule": (
            "row-wise CUEF cohort normalization of the equal-weight mean of "
            "palm-available and vein-available GIPSSR fused score matrices"
        ),
        "shared_mask_across_missing_directions": True,
        "complete_0_percent": complete_metrics,
        "results": results,
    }


def build_summary(payloads: dict[str, dict], mask_payloads: dict) -> dict:
    summary = {
        "experiment_version": EXPERIMENT_VERSION,
        "datasets": {},
        "rates": list(RATES),
        "model_seed": SEED,
        "mask_seed": SEED,
    }
    for dataset, payload in payloads.items():
        block = {
            "probe_count": get_protocol_spec(dataset).test_probes_per_scenario,
            "masks": mask_payloads[dataset],
            "complete_0_percent": basic_metrics(payload["complete_0_percent"]),
            "results": {scenario: {} for scenario in SCENARIOS},
        }
        for scenario in SCENARIOS:
            for key in (rate_key(rate) for rate in RATES):
                block["results"][scenario][key] = basic_metrics(
                    payload["results"][scenario][key]
                )
        summary["datasets"][dataset] = block
    return summary


def _format(summary: dict, metric: str) -> str:
    precision = 4 if metric == "eer" else 2
    return f"{summary[metric] * 100:.{precision}f}"


def render_report(summary: dict) -> str:
    lines = [
        "# GIPSSR-Net 不同缺失率测试结果",
        "",
        (
            "协议：按照 SSFD-Net 的定义，仅在测试 probe 中随机选择目标比例的样本令指定模态缺失；"
            "其余 probe 保持双模态完整。每个数据集复用相同的完整配对训练 checkpoint，不按缺失率重训。"
        ),
        "",
        (
            "随机性：模型与测试缺失 mask 均固定为 seed=42。"
            "两个缺失方向共享同一组 probe，且 10% ⊂ 20% ⊂ 50% ⊂ 80% ⊂ 100%。"
        ),
        "",
        (
            "完整 probe 同时使用真实掌纹与掌静脉，分别运行两个既有 GIPSSR 单方向推理路径，"
            "将两张 fused score 等权平均后施加逐 probe cohort normalization；缺失 probe "
            "使用对应单方向 fused score。随后按原 probe 顺序拼接整张分数矩阵统一计算指标。"
        ),
        "",
    ]
    labels = {
        PALMPRINT_MISSING: "掌纹缺失（仅掌静脉可用）",
        PALMVEIN_MISSING: "掌静脉缺失（仅掌纹可用）",
    }
    for dataset in DATASETS:
        block = summary["datasets"][dataset]
        lines.extend([f"## {dataset.upper()}", ""])
        mask_rows = []
        for rate in RATES:
            key = rate_key(rate)
            mask = block["masks"][key]
            mask_rows.append(
                f"{key}%={mask['missing_count']}/{mask['total_probe_count']} "
                f"({mask['actual_rate'] * 100:.2f}%)"
            )
        lines.extend(["实际缺失数量：" + "；".join(mask_rows) + "。", ""])
        for scenario in SCENARIOS:
            lines.extend(
                [
                    f"### {labels[scenario]}",
                    "",
                    "| 缺失率 | EER ↓ (%) | Top-1 ↑ (%) | TAR@FAR=1e-3 ↑ (%) | TAR@FAR=1e-4 ↑ (%) |",
                    "| ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for rate in RATES:
                key = rate_key(rate)
                result = block["results"][scenario][key]
                lines.append(
                    f"| {key}% | {_format(result, 'eer')} | {_format(result, 'top1')} | "
                    f"{_format(result, 'tar_1e-3')} | {_format(result, 'tar_1e-4')} |"
                )
            lines.append("")
    lines.extend(
        [
            "## 协议说明",
            "",
            "- 缺失率是缺失目标模态的 probe 数量占全部测试 probe 数量的比例；gallery 始终完整。",
            "- CUMT 的样本数不能精确整除 10%、20%、80%，采用 `floor(N×r+0.5)` 四舍五入，并报告实际比例。",
            "- PolyU 不属于 SSFD-Net 原论文的三个数据集之一；本实验仅沿用其测试时随机缺失定义。",
            "- 100% 行由本脚本重新计算，并与仓库保留的正式 100% 测试结果逐指标核对通过。",
            "- CUMT 每个方向仅有 6,612 个 impostor scores，TAR@FAR=1e-4 实际对应 FAR=0 经验点。",
            "",
        ]
    )
    return "\n".join(lines)


def public_mask_payload(masks: dict[str, dict]) -> dict:
    return {
        key: {name: value for name, value in block.items() if name != "mask"}
        for key, block in masks.items()
    }


def run(args) -> dict:
    device = resolve_device(args.device, require_available=True, announce=True)
    set_random_seed(SEED)
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    payloads: dict[str, dict] = {}
    mask_payloads: dict[str, dict] = {}
    for dataset in selected:
        print(f"\n===== {dataset.upper()} partial missing-rate evaluation =====", flush=True)
        gallery, probes, cache_metadata = load_test_features(dataset, args, device)
        masks = nested_missing_masks(
            probes["labels"].numel(),
            dataset=dataset,
            seed=SEED,
            sample_order=cache_metadata["probes"]["sample_order"],
        )
        mask_payloads[dataset] = public_mask_payload(masks)
        print(f"[Evaluate] {dataset} seed={SEED}", flush=True)
        payload = evaluate_seed(dataset, SEED, gallery, probes, masks, device, SEED)
        payload["cache_metadata"] = cache_metadata
        payload["masks"] = public_mask_payload(masks)
        payloads[dataset] = payload
        atomic_write_json(
            args.output_root / dataset / f"seed_{SEED}" / "results.json",
            payload,
        )
        del gallery, probes
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if tuple(selected) != DATASETS:
        print("[Info] subset run complete; cross-dataset summary not written", flush=True)
        return payloads
    summary = build_summary(payloads, mask_payloads)
    summary_path = args.output_root / "summary.json"
    report_path = args.output_root / "report.md"
    atomic_write_json(summary_path, summary)
    report_path.write_text(render_report(summary), encoding="utf-8")
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "summary": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "report": str(report_path),
        "report_sha256": file_sha256(report_path),
        "datasets": list(DATASETS),
        "model_seed": SEED,
        "mask_seed": SEED,
        "rates": list(RATES),
    }
    atomic_write_json(args.output_root / "run_manifest.json", manifest)
    print(f"[Done] wrote {summary_path} and {report_path}", flush=True)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Evaluate deterministic partial missing rates")
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--embedding_size", type=int, default=256)
    parser.add_argument("--extract_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--force_recache", action="store_true")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache_root", type=Path, default=DEFAULT_CACHE_ROOT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
