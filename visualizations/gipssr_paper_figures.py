"""Create publication-ready GIPSSR-Net result visualizations.

Figure 1 follows the feature-distribution visualizations used by ShaSpec
(CVPR 2023), DiCMoR (ICCV 2023), and Missing Modality Prediction (ECCV 2024).
Figure 2 follows biometric score-distribution and confidence visualizations
used by PIC-Score (CVPRW 2023), FUME (CVPR 2025), and QME (ICCV 2025).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, spearmanr
from sklearn.manifold import TSNE

from models.gipssr import ARCHITECTURE_VERSION, GIPSSRNet
from utils.checkpoint import load_encoder_from_checkpoint
from utils.checkpoint_io import file_sha256, safe_torch_load
from utils.evaluation import score_matrix_metrics
from utils.feature_extraction import paired_feature_loader
from utils.gipssr_feature_extraction import load_or_extract_paired_spatial_cache
from utils.runtime import resolve_device, set_random_seed
from utils.scenarios import COMPLETE

DATASETS = {
    "polyu": {
        "gallery": ROOT / "data_txt/polyu/ssfd_gallery_full.txt",
        "protocol": ROOT / "data_txt/polyu/ssfd_test_protocol.txt",
        "palm_ckpt": ROOT / "outputs/encoders/identity_8_2/polyu/palm_best.pth",
        "vein_ckpt": ROOT / "outputs/encoders/identity_8_2/polyu/vein_best.pth",
    },
    "cumt": {
        "gallery": ROOT / "data_txt/cumt/ssfd_gallery_full.txt",
        "protocol": ROOT / "data_txt/cumt/ssfd_test_protocol.txt",
        "palm_ckpt": ROOT / "outputs/encoders/identity_8_2/cumt/palm_best.pth",
        "vein_ckpt": ROOT / "outputs/encoders/identity_8_2/cumt/vein_best.pth",
    },
    "tongji": {
        "gallery": ROOT / "data_txt/tongji/ssfd_gallery_full.txt",
        "protocol": ROOT / "data_txt/tongji/ssfd_test_protocol.txt",
        "palm_ckpt": ROOT / "outputs/encoders/palm_best.pth",
        "vein_ckpt": ROOT / "outputs/encoders/vein_best.pth",
    },
}

DISPLAY_NAMES = {"polyu": "PolyU", "cumt": "CUMT", "tongji": "Tongji"}

INK = "#17202A"
MUTED = "#667085"
GRID = "#D9DEE7"
LIGHT_GRID = "#EEF1F5"
IMPOSTOR = "#3B6FB6"
GENUINE = "#E58B35"
RECOVERY_GREEN = "#4D9B75"
IDENTITY_COLORS = ("#3B6FB6", "#E58B35", "#5A9A63", "#8B67B2", "#C65A7A")
BRANCH_COLORS = ("#3B6FB6", "#D9A62E", "#758C48", "#8B67B2")
BRANCH_HATCHES = ("///", "...", "xxx", "\\\\\\")
BRANCH_NAMES = ("Available", "Shared-Same", "Shared-Cross", "Recovered")


@dataclass
class DatasetFeatures:
    gallery: dict[str, torch.Tensor]
    probes: dict[str, torch.Tensor]


def configure_plotting(font_family: str = "DejaVu Sans") -> None:
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "figure.titlesize": 12.0,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def checkpoint_path(dataset: str, seed: int) -> Path:
    return (
        ROOT
        / "outputs/gipssr/ablations/checkpoints"
        / dataset
        / f"seed_{seed}"
        / "full/best.pth"
    )


def cpu_dict(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().float().cpu() for key, value in values.items()}


def to_device(values: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    output = {key: value.to(device) for key, value in values.items()}
    output["labels"] = output["labels"].long()
    return output


def build_model(path: Path, device: torch.device) -> tuple[GIPSSRNet, dict[str, Any]]:
    checkpoint = safe_torch_load(path, device)
    if checkpoint.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError(f"{path} is not a final GIPSSR-Net checkpoint")
    saved_args = checkpoint.get("args", {})
    config = checkpoint["configuration"]
    model = GIPSSRNet(
        input_dim=int(config["input_dim"]),
        shared_dim=int(config["shared_dim"]),
        specific_dim=int(config["specific_dim"]),
        transformer_layers=int(config["transformer_layers"]),
        transformer_heads=int(config["transformer_heads"]),
        dropout=float(config["dropout"]),
        max_recovery_weight=float(config["max_recovery_weight"]),
        min_recovery_weight=float(saved_args.get("min_recovery_weight", 0.15)),
        retrieval_dropout=float(saved_args.get("retrieval_dropout", 0.10)),
        branch_floor=float(saved_args.get("branch_floor", 0.0)),
        topk_candidates=int(saved_args.get("topk_candidates", 5)),
        role_queries=int(saved_args.get("role_queries", 4)),
        candidate_dropout=float(saved_args.get("candidate_dropout", 0.20)),
        max_refinement=float(saved_args.get("max_refinement", 0.25)),
        ablation=str(saved_args.get("ablation", checkpoint.get("ablation", "full"))),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint

def verify_encoder_binding(
    checkpoint: dict[str, Any], palm_ckpt: Path, vein_ckpt: Path
) -> None:
    fingerprints = checkpoint.get("fingerprints", {})
    for name, path in (("palm", palm_ckpt), ("vein", vein_ckpt)):
        expected = fingerprints.get(f"{name}_encoder_sha256")
        actual = file_sha256(path)
        if expected is not None and actual != expected:
            raise ValueError(
                f"{name} encoder {path} does not match the recovery checkpoint"
            )


def load_dataset_features(
    dataset: str,
    palm_encoder,
    vein_encoder,
    cache_dir: Path,
    device: torch.device,
    input_size: int,
    embedding_size: int,
    batch_size: int,
    num_workers: int,
) -> DatasetFeatures:
    config = DATASETS[dataset]
    palm_ckpt = Path(config["palm_ckpt"])
    vein_ckpt = Path(config["vein_ckpt"])
    gallery, _ = load_or_extract_paired_spatial_cache(
        str(cache_dir / f"{dataset}_gallery.pth"),
        str(config["gallery"]),
        None,
        palm_encoder,
        vein_encoder,
        str(palm_ckpt),
        str(vein_ckpt),
        device,
        input_size,
        embedding_size,
        batch_size,
        num_workers,
        f"Extract {dataset.upper()} gallery features",
    )
    probes, _ = load_or_extract_paired_spatial_cache(
        str(cache_dir / f"{dataset}_complete_probes.pth"),
        str(config["protocol"]),
        COMPLETE,
        palm_encoder,
        vein_encoder,
        str(palm_ckpt),
        str(vein_ckpt),
        device,
        input_size,
        embedding_size,
        batch_size,
        num_workers,
        f"Extract {dataset.upper()} paired test probes",
    )
    if not torch.equal(gallery["labels"].long().unique(sorted=True), probes["labels"].long().unique(sorted=True)):
        raise ValueError(f"{dataset}: gallery and probe identity sets differ")
    return DatasetFeatures(gallery=gallery, probes=probes)


@torch.inference_mode()
def capture_direction(
    model: GIPSSRNet,
    memory: dict[str, torch.Tensor],
    probes: dict[str, torch.Tensor],
    available: str,
) -> dict[str, torch.Tensor]:
    coarse: dict[str, torch.Tensor] = {}

    def capture_coarse(_module, _inputs, output):
        coarse["mean"] = output[0].detach()

    hook = model.recovery_decoder.register_forward_hook(capture_coarse)
    try:
        output = model.recover_with_gallery(
            probes[available],
            probes[f"{available}_spatial"],
            available,
            memory,
        )
    finally:
        hook.remove()
    if "mean" not in coarse:
        raise RuntimeError("Stage-1 recovery hook did not capture a coarse embedding")
    target = "vein" if available == "palm" else "palm"
    probe_labels = probes["labels"].long()
    target_indices = torch.searchsorted(memory["labels"], probe_labels)
    if not torch.equal(memory["labels"][target_indices], probe_labels):
        raise ValueError("Probe identity is absent from the gallery memory")
    keys = (
        "mean",
        "log_variance",
        "base_branch_scores",
        "recovered_scores",
        "fused_scores",
        "branch_weights",
        "fusion_uncertainty",
        "fusion_conflict",
        "posterior_entropy",
        "recovery_weight",
        "candidate_labels",
    )
    captured = {key: output[key].detach().float().cpu() for key in keys}
    captured.update(
        {
            "coarse_mean": coarse["mean"].float().cpu(),
            "target_embedding": memory[f"{target}_embedding"][target_indices]
            .detach()
            .float()
            .cpu(),
            "probe_labels": probe_labels.cpu(),
        }
    )
    return captured


@torch.inference_mode()
def run_model(
    model: GIPSSRNet,
    features: DatasetFeatures,
    device: torch.device,
    include_encodings: bool,
) -> dict[str, Any]:
    gallery = to_device(features.gallery, device)
    probes = to_device(features.probes, device)
    memory = model.build_gallery_memory(gallery)
    result: dict[str, Any] = {
        "palm_available": capture_direction(model, memory, probes, "palm"),
        "vein_available": capture_direction(model, memory, probes, "vein"),
    }
    if include_encodings:
        result["encodings"] = {
            "palm_embedding": memory["palm_embedding"].detach().float().cpu(),
            "vein_embedding": memory["vein_embedding"].detach().float().cpu(),
            "palm_shared": memory["palm_shared"].detach().float().cpu(),
            "vein_shared": memory["vein_shared"].detach().float().cpu(),
            "labels": memory["labels"].long().cpu(),
        }
    del gallery, probes, memory
    torch.cuda.empty_cache()
    return result


def selected_identity_mask(labels: np.ndarray, count: int = 5) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(labels)
    if unique.size < count:
        raise ValueError(f"Need at least {count} identities for the feature visualization")
    ranks = np.linspace(0, unique.size - 1, count, dtype=int)
    selected = unique[ranks]
    return np.isin(labels, selected), selected


def joint_tsne(parts: list[np.ndarray], seed: int = 42) -> list[np.ndarray]:
    lengths = [part.shape[0] for part in parts]
    matrix = np.concatenate(parts, axis=0)
    perplexity = min(15.0, max(5.0, (matrix.shape[0] - 1) / 3.0))
    coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        max_iter=2000,
        random_state=seed,
    ).fit_transform(matrix)
    coordinates = (coordinates - coordinates.mean(axis=0, keepdims=True)) / (
        coordinates.std(axis=0, keepdims=True) + 1e-8
    )
    split = np.cumsum(lengths)[:-1]
    return list(np.split(coordinates, split))


def style_embedding_axis(ax: plt.Axes, panel: str, title: str) -> None:
    ax.set_title(f"{panel} {title}", loc="left", fontweight="bold", pad=5)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(x=0.06, y=0.18)
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.8)


def plot_embedding_context(ax: plt.Axes, parts: list[np.ndarray]) -> None:
    for coords, marker in zip(parts, ("o", "^")):
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=7,
            marker=marker,
            color="#DFE3E8",
            alpha=0.55,
            linewidths=0,
            zorder=0,

        )
def plot_identity_pair(
    ax: plt.Axes,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    labels: np.ndarray,
    selected: np.ndarray,
    marker_a: str,
    marker_b: str,
) -> None:
    color_map = {int(identity): IDENTITY_COLORS[index] for index, identity in enumerate(selected)}
    for index, identity in enumerate(labels):
        color = color_map[int(identity)]
        ax.plot(
            [coords_a[index, 0], coords_b[index, 0]],
            [coords_a[index, 1], coords_b[index, 1]],
            color="#C9CED8",
            linewidth=0.65,
            alpha=0.65,
            zorder=1,
        )
        ax.scatter(
            coords_a[index, 0],
            coords_a[index, 1],
            s=35,
            marker=marker_a,
            facecolor=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.scatter(
            coords_b[index, 0],
            coords_b[index, 1],
            s=42,
            marker=marker_b,
            facecolor="white",
            edgecolor=color,
            linewidth=1.2,
            zorder=3,
        )


def plot_recovery_triplet(
    ax: plt.Axes,
    target: np.ndarray,
    coarse: np.ndarray,
    final: np.ndarray,
    labels: np.ndarray,
    selected: np.ndarray,
) -> None:
    color_map = {int(identity): IDENTITY_COLORS[index] for index, identity in enumerate(selected)}
    for index, identity in enumerate(labels):
        color = color_map[int(identity)]
        ax.plot(
            [coarse[index, 0], final[index, 0], target[index, 0]],
            [coarse[index, 1], final[index, 1], target[index, 1]],
            color="#C9CED8",
            linewidth=0.7,
            alpha=0.72,
            zorder=1,
        )
        ax.scatter(
            target[index, 0],
            target[index, 1],
            s=45,
            marker="o",
            facecolor="white",
            edgecolor=color,
            linewidth=1.25,
            zorder=4,
        )
        ax.scatter(
            coarse[index, 0],
            coarse[index, 1],
            s=38,
            marker="x",
            color=color,
            linewidth=1.25,
            zorder=3,
        )
        ax.scatter(
            final[index, 0],
            final[index, 1],
            s=34,
            marker="D",
            facecolor=color,
            edgecolor=INK,
            linewidth=0.45,
            zorder=5,
        )


def cosine_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float(), b.float(), dim=1).mean().item())


def create_figure_one(
    result: dict[str, Any], output_dir: Path, dataset: str
) -> dict[str, Any]:
    enc = result["encodings"]
    template_labels_all = enc["labels"].numpy()
    template_mask, selected = selected_identity_mask(template_labels_all, count=5)
    template_labels = template_labels_all[template_mask]

    raw_coords_all = joint_tsne(
        [
            enc["palm_embedding"].numpy(),
            enc["vein_embedding"].numpy(),
        ]
    )
    shared_coords_all = joint_tsne(
        [
            enc["palm_shared"].numpy(),
            enc["vein_shared"].numpy(),
        ]
    )
    raw_coords = [coords[template_mask] for coords in raw_coords_all]
    shared_coords = [coords[template_mask] for coords in shared_coords_all]

    palm_missing = result["vein_available"]
    vein_missing = result["palm_available"]
    palm_probe_labels = palm_missing["probe_labels"].numpy()
    vein_probe_labels = vein_missing["probe_labels"].numpy()
    palm_probe_mask = np.isin(palm_probe_labels, selected)
    vein_probe_mask = np.isin(vein_probe_labels, selected)
    palm_recovery_all = joint_tsne(
        [
            palm_missing["target_embedding"].numpy(),
            palm_missing["coarse_mean"].numpy(),
            palm_missing["mean"].numpy(),
        ]
    )
    vein_recovery_all = joint_tsne(
        [
            vein_missing["target_embedding"].numpy(),
            vein_missing["coarse_mean"].numpy(),
            vein_missing["mean"].numpy(),
        ]
    )
    palm_recovery_coords = [coords[palm_probe_mask] for coords in palm_recovery_all]
    vein_recovery_coords = [coords[vein_probe_mask] for coords in vein_recovery_all]

    raw_similarity = cosine_mean(enc["palm_embedding"], enc["vein_embedding"])
    shared_similarity = cosine_mean(enc["palm_shared"], enc["vein_shared"])
    palm_coarse = cosine_mean(palm_missing["coarse_mean"], palm_missing["target_embedding"])
    palm_final = cosine_mean(palm_missing["mean"], palm_missing["target_embedding"])
    vein_coarse = cosine_mean(vein_missing["coarse_mean"], vein_missing["target_embedding"])
    vein_final = cosine_mean(vein_missing["mean"], vein_missing["target_embedding"])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    fig.suptitle("Cross-modal alignment and missing-modality feature recovery", y=0.985, fontweight="bold")
    fig.text(
        0.5,
        0.947,
        f"{DISPLAY_NAMES[dataset]} · seed 42 · t-SNE fitted on all {template_labels_all.size} identities · five highlighted",
        ha="center",
        va="center",
        fontsize=7.5,
        color=MUTED,
    )

    style_embedding_axis(axes[0, 0], "(a)", "Frozen-encoder gallery templates")
    plot_embedding_context(axes[0, 0], raw_coords_all)
    plot_identity_pair(
        axes[0, 0],
        raw_coords[0],
        raw_coords[1],
        template_labels,
        selected,
        "o",
        "^",
    )
    axes[0, 0].text(
        0.03,
        0.04,
        f"Mean template cosine = {raw_similarity:.3f}",
        transform=axes[0, 0].transAxes,
        fontsize=7.2,
        color=MUTED,
    )

    style_embedding_axis(axes[0, 1], "(b)", "IGDCA shared embeddings")
    plot_embedding_context(axes[0, 1], shared_coords_all)
    plot_identity_pair(
        axes[0, 1],
        shared_coords[0],
        shared_coords[1],
        template_labels,
        selected,
        "o",
        "^",
    )
    axes[0, 1].text(
        0.03,
        0.04,
        f"Mean template cosine = {shared_similarity:.3f}",
        transform=axes[0, 1].transAxes,
        fontsize=7.2,
        color=MUTED,
    )

    style_embedding_axis(axes[1, 0], "(c)", "Palmprint missing")
    plot_recovery_triplet(
        axes[1, 0],
        palm_recovery_coords[0],
        palm_recovery_coords[1],
        palm_recovery_coords[2],
        palm_probe_labels[palm_probe_mask],
        selected,
    )
    axes[1, 0].text(
        0.03,
        0.04,
        f"Cosine to gallery template: coarse {palm_coarse:.3f} → final {palm_final:.3f}",
        transform=axes[1, 0].transAxes,
        fontsize=7.2,
        color=MUTED,
    )

    style_embedding_axis(axes[1, 1], "(d)", "Palm vein missing")
    plot_recovery_triplet(
        axes[1, 1],
        vein_recovery_coords[0],
        vein_recovery_coords[1],
        vein_recovery_coords[2],
        vein_probe_labels[vein_probe_mask],
        selected,
    )
    axes[1, 1].text(
        0.03,
        0.04,
        f"Cosine to gallery template: coarse {vein_coarse:.3f} → final {vein_final:.3f}",
        transform=axes[1, 1].transAxes,
        fontsize=7.2,
        color=MUTED,
    )

    identity_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=IDENTITY_COLORS[index],
            markeredgecolor="white",
            markersize=6,
            label=f"ID {int(identity)}",
        )
        for index, identity in enumerate(selected)
    ]
    status_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=INK, markeredgecolor="white", markersize=6, label="Palmprint"),
        Line2D([0], [0], marker="^", linestyle="", markerfacecolor="white", markeredgecolor=INK, markersize=6, label="Palm vein"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=INK, markersize=6, label="Gallery target"),
        Line2D([0], [0], marker="x", linestyle="", color=INK, markersize=6, label="Coarse"),
        Line2D([0], [0], marker="D", linestyle="", markerfacecolor=INK, markeredgecolor=INK, markersize=5, label="Final"),
    ]
    first_legend = fig.legend(
        handles=identity_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=5,
        frameon=False,
        title="Identity color",
        title_fontsize=7.2,
        handletextpad=0.35,
        columnspacing=1.1,
    )
    fig.add_artist(first_legend)
    fig.legend(
        handles=status_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.066),
        ncol=5,
        frameon=False,
        title="Marker meaning",
        title_fontsize=7.2,
        handletextpad=0.35,
        columnspacing=1.05,
    )
    fig.subplots_adjust(top=0.91, bottom=0.16, left=0.07, right=0.985, hspace=0.28, wspace=0.16)
    save_figure(fig, output_dir / "figure1_feature_alignment_recovery")
    plt.close(fig)

    return {
        "dataset": dataset,
        "seed": 42,
        "gallery_identity_count": int(template_labels_all.size),
        "probe_count": int(palm_probe_labels.size),
        "selected_identities": [int(value) for value in selected],
        "tsne_fit_observations": {
            "template_alignment": int(2 * template_labels_all.size),
            "recovery_per_direction": int(3 * palm_probe_labels.size),
        },
        "raw_template_cosine_mean": raw_similarity,
        "shared_template_cosine_mean": shared_similarity,
        "palmprint_missing": {
            "coarse_gallery_template_cosine_mean": palm_coarse,
            "final_gallery_template_cosine_mean": palm_final,
        },
        "palmvein_missing": {
            "coarse_gallery_template_cosine_mean": vein_coarse,
            "final_gallery_template_cosine_mean": vein_final,
        },
    }


def genuine_impostor(scores: torch.Tensor, labels: torch.Tensor, candidate_labels: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    matches = labels[:, None].eq(candidate_labels[None, :])
    if not torch.all(matches.sum(dim=1) == 1):
        raise ValueError("Each probe must have exactly one genuine gallery identity")
    genuine = scores[matches].float().numpy()
    impostor = scores[~matches].float().numpy()
    return genuine, impostor


def impostor_normalize(genuine: np.ndarray, impostor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = float(impostor.mean())
    std = float(impostor.std())
    return (genuine - mean) / max(std, 1e-8), (impostor - mean) / max(std, 1e-8)


def kde_summary(
    result: dict[str, torch.Tensor],
    score_key: str,
    score_index: int | None,
    grid: np.ndarray,
) -> dict[str, Any]:
    scores = result[score_key]
    if score_index is not None:
        scores = scores[:, :, score_index]
    genuine, impostor = genuine_impostor(
        scores,
        result["probe_labels"],
        result["candidate_labels"],
    )
    genuine, impostor = impostor_normalize(genuine, impostor)
    genuine_density = gaussian_kde(genuine)(grid)
    impostor_density = gaussian_kde(impostor)(grid)
    metrics = score_matrix_metrics(
        scores,
        candidate_labels=result["candidate_labels"],
        probe_labels=result["probe_labels"],
        topk=(1, 5),
        far_points=(1e-3,),
    )
    return {
        "genuine_density": genuine_density,
        "impostor_density": impostor_density,
        "threshold": float(np.quantile(impostor, 0.999, method="higher")),
        "eer": float(metrics["eer"]),
        "overlap": float(
            np.trapezoid(np.minimum(genuine_density, impostor_density), grid)
        ),
    }


def score_grid(result: dict[str, torch.Tensor]) -> np.ndarray:
    values = []
    for key, index in (("base_branch_scores", 0), ("fused_scores", None)):
        scores = result[key]
        if index is not None:
            scores = scores[:, :, index]
        genuine, impostor = genuine_impostor(
            scores,
            result["probe_labels"],
            result["candidate_labels"],
        )
        genuine, impostor = impostor_normalize(genuine, impostor)
        values.extend((genuine, impostor))
    combined = np.concatenate(values)
    lower, upper = np.quantile(combined, [0.001, 0.999])
    padding = 0.06 * (upper - lower)
    return np.linspace(lower - padding, upper + padding, 500)


def plot_score_density(ax: plt.Axes, grid: np.ndarray, summary: dict[str, Any], panel: str, title: str) -> None:
    g = summary["genuine_density"]
    i = summary["impostor_density"]
    ax.fill_between(grid, 0, i, color=IMPOSTOR, alpha=0.17)
    ax.fill_between(grid, 0, g, color=GENUINE, alpha=0.17)
    ax.fill_between(grid, 0, np.minimum(g, i), color="#8A8F98", alpha=0.16, hatch="////", edgecolor="#8A8F98", linewidth=0.0)
    ax.plot(grid, i, color=IMPOSTOR, linewidth=1.7, label="Impostor")
    ax.plot(grid, g, color=GENUINE, linewidth=1.7, linestyle="--", label="Genuine")
    threshold = summary["threshold"]
    ax.axvline(threshold, color=INK, linestyle=":", linewidth=1.25)
    ax.text(
        threshold,
        ax.get_ylim()[1] * 0.68,
        "FAR=10⁻³",
        rotation=90,
        va="top",
        ha="right",
        fontsize=6.8,
        color=INK,
    )
    ax.set_title(f"{panel} {title}", loc="left", fontweight="bold", pad=5)
    ax.set_xlabel("Impostor-normalized score")
    ax.set_ylabel("Density")
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.text(
        0.98,
        0.94,
        f"EER {summary['eer']*100:.3f}%\nOverlap {summary['overlap']:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color=MUTED,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )


def bootstrap_mean_interval(values: np.ndarray, rng: np.random.Generator, repeats: int = 1000) -> tuple[float, float]:
    if values.size == 1:
        value = float(values[0])
        return value, value
    indices = rng.integers(0, values.size, size=(repeats, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def quantile_groups(values: np.ndarray, count: int = 5) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    groups = np.empty(values.size, dtype=int)
    for group, indices in enumerate(np.array_split(order, count)):
        groups[indices] = group
    return groups


def create_figure_two(
    result: dict[str, torch.Tensor], output_dir: Path, dataset: str
) -> dict[str, Any]:
    grid = score_grid(result)
    available_summary = kde_summary(result, "base_branch_scores", 0, grid)
    fused_summary = kde_summary(result, "fused_scores", None, grid)

    uncertainty = result["log_variance"].exp().numpy()
    recovery_error = (
        1.0
        - F.cosine_similarity(
            result["mean"],
            result["target_embedding"],
            dim=1,
        )
    ).numpy()
    weights = result["branch_weights"].numpy()
    groups = quantile_groups(uncertainty, count=5)
    log_uncertainty = np.log10(np.clip(uncertainty, 1e-8, None))
    rho, p_value = spearmanr(log_uncertainty, recovery_error)

    rng = np.random.default_rng(42)
    bin_x = []
    bin_y = []
    bin_low = []
    bin_high = []
    bin_weights = []
    bin_counts = []
    for group in range(5):
        keep = groups == group
        bin_counts.append(int(keep.sum()))
        bin_x.append(float(log_uncertainty[keep].mean()))
        bin_y.append(float(recovery_error[keep].mean()))
        low, high = bootstrap_mean_interval(recovery_error[keep], rng)
        bin_low.append(low)
        bin_high.append(high)
        bin_weights.append(weights[keep].mean(axis=0))
    bin_x = np.asarray(bin_x)
    bin_y = np.asarray(bin_y)
    bin_low = np.asarray(bin_low)
    bin_high = np.asarray(bin_high)
    bin_weights = np.stack(bin_weights)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.35))
    fig.suptitle("CUEF score separation and uncertainty-aware weighting", y=0.985, fontweight="bold")
    fig.text(
        0.5,
        0.947,
        f"{DISPLAY_NAMES[dataset]} · palm available → palm vein missing · seed 42 full checkpoint · n={uncertainty.size} probes",
        ha="center",
        va="center",
        fontsize=7.5,
        color=MUTED,
    )

    plot_score_density(axes[0, 0], grid, available_summary, "(a)", "Available-modality scores")
    plot_score_density(axes[0, 1], grid, fused_summary, "(b)", "Final CUEF scores")
    density_handles = [
        Line2D([0], [0], color=IMPOSTOR, linewidth=1.7, label="Impostor"),
        Line2D([0], [0], color=GENUINE, linewidth=1.7, linestyle="--", label="Genuine"),
        Line2D([0], [0], color=INK, linewidth=1.25, linestyle=":", label="FAR=10⁻³ threshold"),
    ]
    axes[0, 1].legend(handles=density_handles, loc="upper center", bbox_to_anchor=(-0.10, 1.17), ncol=3, frameon=False)

    ax = axes[1, 0]
    ax.scatter(log_uncertainty, recovery_error, s=14, color="#A9AFB9", alpha=0.48, edgecolor="none", zorder=1)
    yerr = np.vstack([bin_y - bin_low, bin_high - bin_y])
    ax.errorbar(
        bin_x,
        bin_y,
        yerr=yerr,
        color=IMPOSTOR,
        marker="o",
        markerfacecolor="white",
        markeredgecolor=IMPOSTOR,
        markeredgewidth=1.1,
        markersize=5.0,
        linewidth=1.6,
        capsize=2.5,
        zorder=3,
    )
    ax.set_title("(c) Recovery uncertainty calibration", loc="left", fontweight="bold", pad=5)
    ax.set_xlabel("log₁₀ predicted recovery variance")
    ax.set_ylabel("1 − cosine(recovered, gallery template)")
    ax.grid(color=LIGHT_GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.text(
        0.04,
        0.94,
        f"Spearman ρ = {rho:.3f}\np = {p_value:.2g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.1,
        color=MUTED,
    )

    ax = axes[1, 1]
    x = np.arange(5)
    bottom = np.zeros(5)
    for index, (name, color, hatch) in enumerate(zip(BRANCH_NAMES, BRANCH_COLORS, BRANCH_HATCHES)):
        ax.bar(
            x,
            bin_weights[:, index],
            bottom=bottom,
            width=0.72,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            hatch=hatch,
            label=name,
        )
        if index == 3:
            for column, value in enumerate(bin_weights[:, index]):
                ax.text(
                    column,
                    bottom[column] + value / 2.0,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if value >= 0.12 else INK,
                    fontweight="bold",
                )
        bottom += bin_weights[:, index]
    ax.set_title("(d) Branch weights across uncertainty", loc="left", fontweight="bold", pad=5)
    ax.set_xlabel("Predicted recovery uncertainty (low → high)")
    ax.set_ylabel("Mean branch weight")
    ax.set_xticks(x, [f"Q{index}" for index in range(1, 6)])
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    branch_handles, branch_labels = ax.get_legend_handles_labels()
    fig.legend(
        branch_handles,
        branch_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.2,
    )

    fig.subplots_adjust(top=0.89, bottom=0.15, left=0.085, right=0.985, hspace=0.34, wspace=0.25)
    save_figure(fig, output_dir / "figure2_cuef_diagnostics")
    plt.close(fig)

    return {
        "dataset": dataset,
        "direction": "palm_available_palmvein_missing",
        "seed": 42,
        "probe_count": int(uncertainty.size),
        "gallery_identity_count": int(result["candidate_labels"].numel()),
        "score_counts": {
            "genuine": int(uncertainty.size),
            "impostor": int(
                uncertainty.size * (result["candidate_labels"].numel() - 1)
            ),
        },
        "uncertainty_scatter_observations": int(uncertainty.size),
        "available_score": {
            "eer": available_summary["eer"],
            "distribution_overlap": available_summary["overlap"],
        },
        "fused_score": {
            "eer": fused_summary["eer"],
            "distribution_overlap": fused_summary["overlap"],
        },
        "uncertainty_recovery_error_spearman": {
            "rho": float(rho),
            "p_value": float(p_value),
        },
        "uncertainty_quintiles": [
            {
                "quintile": index + 1,
                "probe_count": bin_counts[index],
                "mean_log10_variance": float(bin_x[index]),
                "mean_recovery_error": float(bin_y[index]),
                "recovery_error_ci95": [float(bin_low[index]), float(bin_high[index])],
                "mean_branch_weights": {
                    name: float(bin_weights[index, branch])
                    for branch, name in enumerate(BRANCH_NAMES)
                },
            }
            for index in range(5)
        ],
    }


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Render GIPSSR-Net paper visualizations")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dataset", choices=tuple(DATASETS), default=None)
    parser.add_argument("--font_family", default=None)
    parser.add_argument("--output_dir", default=str(ROOT / "outputs/gipssr/figures"))
    parser.add_argument("--cache_dir", default=str(ROOT / "outputs/gipssr/figures/cache"))
    parser.add_argument("--extract_batch_size", type=int, default=128)
    parser.add_argument("--memory_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--embedding_size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figure1_dataset = args.dataset or "polyu"
    figure2_dataset = args.dataset or "cumt"
    font_family = args.font_family or ("DejaVu Serif" if args.dataset else "DejaVu Sans")
    configure_plotting(font_family)
    set_random_seed(42)
    device = resolve_device(args.device, require_available=True, announce=True)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    figure1_palm_ckpt = Path(DATASETS[figure1_dataset]["palm_ckpt"])
    figure1_vein_ckpt = Path(DATASETS[figure1_dataset]["vein_ckpt"])
    figure1_model, figure1_checkpoint = build_model(checkpoint_path(figure1_dataset, 42), device)
    verify_encoder_binding(figure1_checkpoint, figure1_palm_ckpt, figure1_vein_ckpt)
    palm_encoder = load_encoder_from_checkpoint(
        str(figure1_palm_ckpt), "palm", args.embedding_size, device
    )
    vein_encoder = load_encoder_from_checkpoint(
        str(figure1_vein_ckpt), "vein", args.embedding_size, device
    )

    figure1_features = load_dataset_features(
        figure1_dataset,
        palm_encoder,
        vein_encoder,
        cache_dir,
        device,
        args.input_size,
        args.embedding_size,
        args.extract_batch_size,
        args.num_workers,
    )
    figure1_result = run_model(figure1_model, figure1_features, device, include_encodings=True)
    figure_one_metrics = create_figure_one(figure1_result, output_dir, figure1_dataset)
    del figure1_result, figure1_model, figure1_features
    torch.cuda.empty_cache()

    figure2_palm_ckpt = Path(DATASETS[figure2_dataset]["palm_ckpt"])
    figure2_vein_ckpt = Path(DATASETS[figure2_dataset]["vein_ckpt"])
    palm_encoder = load_encoder_from_checkpoint(
        str(figure2_palm_ckpt), "palm", args.embedding_size, device
    )
    vein_encoder = load_encoder_from_checkpoint(
        str(figure2_vein_ckpt), "vein", args.embedding_size, device
    )

    figure2_features = load_dataset_features(
        figure2_dataset,
        palm_encoder,
        vein_encoder,
        cache_dir,
        device,
        args.input_size,
        args.embedding_size,
        args.extract_batch_size,
        args.num_workers,
    )
    figure2_checkpoint_path = checkpoint_path(figure2_dataset, 42)
    figure2_model, figure2_checkpoint = build_model(figure2_checkpoint_path, device)
    verify_encoder_binding(
        figure2_checkpoint, figure2_palm_ckpt, figure2_vein_ckpt
    )
    figure2_result = run_model(
        figure2_model, figure2_features, device, include_encodings=False
    )
    figure_two_metrics = create_figure_two(
        figure2_result["palm_available"], output_dir, figure2_dataset
    )
    del figure2_result, figure2_model
    torch.cuda.empty_cache()

    metadata = {
        "figure1": figure_one_metrics,
        "figure2": figure_two_metrics,
        "sources": {
            "figure1": {
                "dataset": figure1_dataset,
                "palm_encoder_sha256": file_sha256(figure1_palm_ckpt),
                "vein_encoder_sha256": file_sha256(figure1_vein_ckpt),
                "checkpoint_sha256": file_sha256(checkpoint_path(figure1_dataset, 42)),
                "gallery_sha256": file_sha256(DATASETS[figure1_dataset]["gallery"]),
                "protocol_sha256": file_sha256(DATASETS[figure1_dataset]["protocol"]),
            },
            "figure2": {
                "dataset": figure2_dataset,
                "palm_encoder_sha256": file_sha256(figure2_palm_ckpt),
                "vein_encoder_sha256": file_sha256(figure2_vein_ckpt),
                "checkpoint_sha256": file_sha256(figure2_checkpoint_path),
                "gallery_sha256": file_sha256(DATASETS[figure2_dataset]["gallery"]),
                "protocol_sha256": file_sha256(DATASETS[figure2_dataset]["protocol"]),
            },
        },
        "rendering": {
            "matplotlib": matplotlib.__version__,
            "font_family": font_family,
            "tsne_random_seed": 42,
            "tsne_init": "pca",
            "tsne_max_iter": 2000,
        },
    }
    with (output_dir / "visualization_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(f"[Done] figures and metrics saved to {output_dir}")


if __name__ == "__main__":
    main()
