"""Generate publication figures for shared-identity feature recovery.

The script reuses the frozen encoders, fitted projector, validation-selected
fusion weights, and unchanged Tongji test protocol used by
test_shared_feature_recovery.py. It never fits or tunes a model on test data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter, PercentFormatter
import numpy as np
import torch
import torch.nn.functional as F

from models.shared_feature_recovery import (
    ARCHITECTURE_VERSION,
    RegularizedSharedIdentityProjector,
)
from utils.checkpoint import load_encoder_from_checkpoint
from utils.checkpoint_io import file_sha256, safe_torch_load
from utils.evaluation import gallery_probe_scores, score_matrix_metrics
from utils.feature_extraction import extract_paired_features, paired_feature_loader
from utils.runtime import resolve_device, set_random_seed
from utils.scenarios import PALMPRINT_MISSING, PALMVEIN_MISSING


COLORS = {
    "blue": "#2F6B9A",
    "blue_light": "#A8C7DF",
    "orange": "#D88932",
    "orange_light": "#F1C997",
    "ink": "#263238",
    "gray": "#6F7B83",
    "gray_light": "#D9DEE2",
    "grid": "#E6EAED",
    "white": "#FFFFFF",
}
METHOD_STYLES = {
    "available": {
        "label": "Available only",
        "color": COLORS["gray"],
        "linestyle": "--",
        "marker": "s",
    },
    "shared": {
        "label": "Shared identity only",
        "color": COLORS["orange"],
        "linestyle": ":",
        "marker": "^",
    },
    "fused": {
        "label": "Score fusion",
        "color": COLORS["blue"],
        "linestyle": "-",
        "marker": "o",
    },
}
SCENARIO_TITLES = {
    PALMPRINT_MISSING: "Palmprint missing · vein available",
    PALMVEIN_MISSING: "Palm-vein missing · palmprint available",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": COLORS["ink"],
            "axes.linewidth": 0.8,
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "legend.frameon": False,
            "figure.facecolor": COLORS["white"],
            "axes.facecolor": COLORS["white"],
            "savefig.facecolor": COLORS["white"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def verify_encoder_fingerprints(checkpoint, args) -> None:
    for name, path in (
        ("palm_encoder", args.palm_ckpt),
        ("vein_encoder", args.vein_ckpt),
    ):
        expected = checkpoint.get(f"{name}_sha256")
        if expected is not None and file_sha256(path) != expected:
            raise ValueError(
                f"{name} checkpoint fingerprint differs from recovery fitting"
            )


def verification_curve(scores, candidate_labels, probe_labels):
    scores = torch.as_tensor(scores, dtype=torch.float64)
    candidate_labels = torch.as_tensor(
        candidate_labels, dtype=torch.long, device=scores.device
    )
    probe_labels = torch.as_tensor(
        probe_labels, dtype=torch.long, device=scores.device
    )
    genuine_mask = probe_labels[:, None].eq(candidate_labels[None, :])
    genuine = scores[genuine_mask]
    impostor = scores[~genuine_mask]
    values = torch.cat([genuine, impostor])
    is_genuine = torch.cat(
        [
            torch.ones_like(genuine, dtype=torch.bool),
            torch.zeros_like(impostor, dtype=torch.bool),
        ]
    )
    order = values.argsort(descending=True)
    values = values[order]
    is_genuine = is_genuine[order]
    group_ends = torch.cat(
        [
            torch.nonzero(values[1:] != values[:-1], as_tuple=False).flatten(),
            values.new_tensor([values.numel() - 1], dtype=torch.long),
        ]
    )
    true_accepts = is_genuine.cumsum(0)[group_ends].double()
    false_accepts = (~is_genuine).cumsum(0)[group_ends].double()
    tar = torch.cat([values.new_zeros(1), true_accepts / genuine.numel()])
    far = torch.cat([values.new_zeros(1), false_accepts / impostor.numel()])
    return far.cpu().numpy(), tar.cpu().numpy()


def hard_negative_margin(scores, candidate_labels, probe_labels):
    scores = torch.as_tensor(scores, dtype=torch.float32)
    candidate_labels = torch.as_tensor(
        candidate_labels, dtype=torch.long, device=scores.device
    )
    probe_labels = torch.as_tensor(
        probe_labels, dtype=torch.long, device=scores.device
    )
    genuine_mask = probe_labels[:, None].eq(candidate_labels[None, :])
    genuine = scores[genuine_mask]
    hardest_negative = scores.masked_fill(genuine_mask, -torch.inf).max(dim=1).values
    return (genuine - hardest_negative).cpu().numpy()


def metric_record(scores, candidate_labels, probe_labels):
    metrics = score_matrix_metrics(
        scores,
        candidate_labels,
        probe_labels,
        topk=(1, 5),
        far_points=(1e-3, 1e-4),
        warn_far_resolution=False,
    )
    return {
        "eer": metrics["eer"],
        "tar_at_far": {
            "1e-3": metrics["tar_at_far"][1e-3],
            "1e-4": metrics["tar_at_far"][1e-4],
        },
        "top1": metrics["topk"][1],
        "top5": metrics["topk"][5],
        "far_count_resolution": metrics["far_count_resolution"],
        "minimum_nonzero_far": metrics["minimum_nonzero_far"],
        "num_gallery_identities": metrics["num_gallery_identities"],
        "num_probes": metrics["num_probes"],
        "num_genuine_scores": metrics["num_genuine_scores"],
        "num_impostor_scores": metrics["num_impostor_scores"],
    }


def scenario_scores(
    scenario,
    projector,
    gallery,
    probes,
    weights,
    dimensions,
):
    if scenario == PALMVEIN_MISSING:
        available, target = "palm", "vein"
    elif scenario == PALMPRINT_MISSING:
        available, target = "vein", "palm"
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    available_scores, candidate_labels = gallery_probe_scores(
        gallery[available],
        gallery["labels"],
        probes[available],
    )
    gallery_available_shared = projector.transform(
        gallery[available], available, dimensions
    )
    gallery_target_shared = projector.transform(gallery[target], target, dimensions)
    probe_shared = projector.transform(probes[available], available, dimensions)
    same_scores, same_labels = gallery_probe_scores(
        gallery_available_shared,
        gallery["labels"],
        probe_shared,
    )
    cross_scores, cross_labels = gallery_probe_scores(
        gallery_target_shared,
        gallery["labels"],
        probe_shared,
    )
    if not torch.equal(candidate_labels, same_labels) or not torch.equal(
        candidate_labels, cross_labels
    ):
        raise ValueError("Candidate label orders differ across feature domains")

    cross_weight = float(weights["cross_gallery_weight"])
    alpha = float(weights["alpha"])
    shared_scores = (
        (1.0 - cross_weight) * same_scores + cross_weight * cross_scores
    )
    fused_scores = (available_scores + alpha * shared_scores) / (1.0 + alpha)
    return {
        "available_modality": available,
        "target_modality": target,
        "alpha": alpha,
        "cross_gallery_weight": cross_weight,
        "candidate_labels": candidate_labels.cpu(),
        "probe_labels": probes["labels"].cpu(),
        "available": available_scores.cpu(),
        "shared": shared_scores.cpu(),
        "fused": fused_scores.cpu(),
    }


def extract_figure_data(args):
    device = resolve_device(args.device, require_available=True, announce=True)
    set_random_seed(args.seed)
    checkpoint = safe_torch_load(args.ckpt, device)
    if checkpoint.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("Checkpoint is not a shared identity recovery model")
    verify_encoder_fingerprints(checkpoint, args)

    saved_args = checkpoint.get("args", {})
    input_size = int(saved_args.get("input_size", args.input_size))
    embedding_size = int(
        saved_args.get("embedding_size", args.embedding_size)
    )
    configuration = checkpoint["configuration"]
    dimensions = int(configuration["dimensions"])
    projector = RegularizedSharedIdentityProjector(
        embedding_size,
        unit_input=bool(configuration["unit_input"]),
    ).to(device)
    projector.load_state_dict(checkpoint["model"])
    projector.eval()

    palm_encoder = load_encoder_from_checkpoint(
        args.palm_ckpt, "palm", embedding_size, device
    )
    vein_encoder = load_encoder_from_checkpoint(
        args.vein_ckpt, "vein", embedding_size, device
    )
    gallery = extract_paired_features(
        palm_encoder,
        vein_encoder,
        paired_feature_loader(
            args.gallery_list,
            None,
            input_size,
            args.extract_batch_size,
            args.num_workers,
        ),
        device,
        "Extract visualization Gallery",
    )
    probes = extract_paired_features(
        palm_encoder,
        vein_encoder,
        paired_feature_loader(
            args.protocol_list,
            "complete",
            input_size,
            args.extract_batch_size,
            args.num_workers,
        ),
        device,
        "Extract visualization Probes",
    )
    gallery = {
        key: value.to(device) if key != "labels" else value.to(device)
        for key, value in gallery.items()
    }
    probes = {
        key: value.to(device) if key != "labels" else value.to(device)
        for key, value in probes.items()
    }

    scenarios = {}
    for scenario in (PALMPRINT_MISSING, PALMVEIN_MISSING):
        scenarios[scenario] = scenario_scores(
            scenario,
            projector,
            gallery,
            probes,
            checkpoint["fusion_weights"][scenario],
            dimensions,
        )

    template_labels = gallery["labels"].unique(sorted=True)
    palm_shared = projector.transform(gallery["palm"], "palm", dimensions)
    vein_shared = projector.transform(gallery["vein"], "vein", dimensions)
    palm_templates = torch.stack(
        [
            palm_shared[gallery["labels"] == label].mean(dim=0)
            for label in template_labels
        ]
    )
    vein_templates = torch.stack(
        [
            vein_shared[gallery["labels"] == label].mean(dim=0)
            for label in template_labels
        ]
    )

    return {
        "checkpoint": checkpoint,
        "dimensions": dimensions,
        "scenarios": scenarios,
        "shared_gallery": {
            "labels": template_labels.cpu(),
            "palm": palm_templates.cpu(),
            "vein": vein_templates.cpu(),
        },
    }


def save_figure(fig, output_dir, stem, dpi):
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def add_figure_header(fig, title, subtitle):
    fig.suptitle(title, fontsize=15, fontweight="semibold", y=0.985)
    fig.text(
        0.5,
        0.925,
        subtitle,
        ha="center",
        va="top",
        fontsize=9.2,
        color=COLORS["gray"],
    )


def plot_low_far_roc(data, output_dir, dpi):
    scenarios = data["scenarios"]
    curves = {}
    global_min_tar = 1.0
    min_nonzero_far = None
    for scenario, values in scenarios.items():
        curves[scenario] = {}
        for method in ("available", "shared", "fused"):
            far, tar = verification_curve(
                values[method],
                values["candidate_labels"],
                values["probe_labels"],
            )
            curves[scenario][method] = (far, tar)
            positive = far > 0
            if positive.any():
                minimum = float(far[positive].min())
                min_nonzero_far = (
                    minimum
                    if min_nonzero_far is None
                    else min(min_nonzero_far, minimum)
                )
                visible = positive & (far <= 1e-2)
                if visible.any():
                    global_min_tar = min(global_min_tar, float(tar[visible].min()))

    y_min = max(0.45, np.floor((global_min_tar - 0.025) * 20) / 20)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), sharey=True)
    add_figure_header(
        fig,
        "Low-FAR verification performance",
        (
            "Tongji Session 1 test · 120 identities · 240 probes per scenario · "
            "28,560 impostor scores · empirical stair-step curves"
        ),
    )
    for panel, (ax, scenario) in enumerate(zip(axes, scenarios)):
        values = scenarios[scenario]
        panel_metrics = {}
        for method in ("available", "shared", "fused"):
            far, tar = curves[scenario][method]
            visible = (far > 0) & (far <= 1e-2)
            metrics = metric_record(
                values[method],
                values["candidate_labels"],
                values["probe_labels"],
            )
            panel_metrics[method] = metrics
            style = METHOD_STYLES[method]
            ax.step(
                far[visible],
                tar[visible],
                where="post",
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.0 if method == "fused" else 1.7,
                label=style["label"],
                zorder=3 if method == "fused" else 2,
            )
            for target_far in (1e-4, 1e-3):
                eligible = far <= target_far + 1e-12
                operating_tar = float(tar[eligible].max())
                ax.plot(
                    target_far,
                    operating_tar,
                    marker=style["marker"],
                    markersize=4.8,
                    markerfacecolor=(
                        style["color"]
                        if method == "fused"
                        else COLORS["white"]
                    ),
                    markeredgecolor=style["color"],
                    markeredgewidth=0.9,
                    linestyle="none",
                    zorder=5,
                )
        for target_far in (1e-4, 1e-3):
            ax.axvline(
                target_far,
                color=COLORS["gray_light"],
                linewidth=0.8,
                zorder=0,
            )
        ax.set_xscale("log")
        ax.set_xlim(min_nonzero_far * 0.92, 1e-2)
        ax.set_ylim(y_min, 1.005)
        ax.set_title(SCENARIO_TITLES[scenario], pad=9)
        ax.set_xlabel("False accept rate (log scale)")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
        eer_summary = (
            f"EER · available {panel_metrics['available']['eer'] * 100:.2f}%\n"
            f"shared {panel_metrics['shared']['eer'] * 100:.2f}% · "
            f"fusion {panel_metrics['fused']['eer'] * 100:.2f}%"
        )
        ax.text(
            0.98,
            0.04,
            eer_summary,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.4,
            linespacing=1.3,
            bbox={
                "facecolor": COLORS["white"],
                "edgecolor": COLORS["gray_light"],
                "boxstyle": "round,pad=0.3",
                "alpha": 0.94,
            },
        )
        ax.text(
            0.01,
            0.02,
            f"({chr(97 + panel)})",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="semibold",
        )
    axes[0].set_ylabel("True accept rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        handlelength=2.8,
        columnspacing=1.8,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.22, wspace=0.13)
    return save_figure(fig, output_dir, "figure2_low_far_roc", dpi)


def plot_hard_negative_margins(data, output_dir, dpi):
    scenarios = data["scenarios"]
    margins = {}
    all_values = []
    for scenario, values in scenarios.items():
        baseline = hard_negative_margin(
            values["available"],
            values["candidate_labels"],
            values["probe_labels"],
        )
        fused = hard_negative_margin(
            values["fused"],
            values["candidate_labels"],
            values["probe_labels"],
        )
        margins[scenario] = (baseline, fused)
        all_values.extend((baseline, fused))
    combined = np.concatenate(all_values)
    low, high = float(combined.min()), float(combined.max())
    padding = max(0.02, (high - low) * 0.06)
    limits = (low - padding, high + padding)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0), sharex=True, sharey=True)
    add_figure_header(
        fig,
        "Probe-wise hard-negative identity margins",
        (
            "Each point is one test probe · margin = genuine score − highest impostor score · "
            "positive values indicate correct rank-1 separation"
        ),
    )
    for panel, (ax, scenario) in enumerate(zip(axes, scenarios)):
        baseline, fused = margins[scenario]
        resolved = (baseline <= 0) & (fused > 0)
        fusion_error = fused <= 0
        both_correct = (baseline > 0) & (fused > 0)
        new_error = (baseline > 0) & (fused <= 0)

        ax.scatter(
            baseline[both_correct],
            fused[both_correct],
            s=24,
            facecolors=COLORS["white"],
            edgecolors=COLORS["blue"],
            linewidths=0.9,
            alpha=0.72,
            zorder=2,
        )
        ax.scatter(
            baseline[resolved],
            fused[resolved],
            s=38,
            color=COLORS["orange"],
            edgecolors=COLORS["ink"],
            linewidths=0.35,
            marker="o",
            zorder=4,
        )
        ax.scatter(
            baseline[fusion_error],
            fused[fusion_error],
            s=34,
            color=COLORS["ink"],
            marker="x",
            linewidths=1.1,
            zorder=5,
        )
        ax.plot(
            limits,
            limits,
            color=COLORS["gray"],
            linewidth=1.0,
            linestyle="--",
            zorder=1,
        )
        ax.axhline(0, color=COLORS["gray_light"], linewidth=0.9, zorder=0)
        ax.axvline(0, color=COLORS["gray_light"], linewidth=0.9, zorder=0)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(SCENARIO_TITLES[scenario], pad=9)
        ax.set_xlabel("Available-only margin")
        ax.grid(color=COLORS["grid"], linewidth=0.7)
        delta = fused - baseline
        summary = (
            f"Baseline errors: {(baseline <= 0).sum()} / {baseline.size}\n"
            f"Fusion errors: {(fused <= 0).sum()} / {fused.size}\n"
            f"Resolved: {resolved.sum()}   New: {new_error.sum()}\n"
            f"Median Δ margin: {np.median(delta):+.3f}"
        )
        ax.text(
            0.04,
            0.96,
            summary,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.6,
            linespacing=1.35,
            bbox={
                "facecolor": COLORS["white"],
                "edgecolor": COLORS["gray_light"],
                "boxstyle": "round,pad=0.35",
                "alpha": 0.94,
            },
        )
        ax.text(
            0.98,
            0.03,
            "Above diagonal = larger margin",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.2,
            color=COLORS["gray"],
        )
        ax.text(
            0.01,
            0.02,
            f"({chr(97 + panel)})",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="semibold",
        )
    axes[0].set_ylabel("Fusion margin")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["white"],
            markeredgecolor=COLORS["blue"],
            label="Correct before and after",
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["orange"],
            markeredgecolor=COLORS["ink"],
            label="Resolved by fusion",
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color=COLORS["ink"],
            linestyle="none",
            label="Fusion error",
            markersize=6,
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        columnspacing=2.0,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.19, wspace=0.12)
    return save_figure(fig, output_dir, "figure3_hard_negative_margin", dpi), margins


def shared_space_analysis(shared_gallery):
    labels = shared_gallery["labels"]
    palm = F.normalize(shared_gallery["palm"].float(), dim=1)
    vein = F.normalize(shared_gallery["vein"].float(), dim=1)
    similarities = palm @ vein.t()
    same = similarities.diag().numpy()
    negative_mask = torch.eye(similarities.size(0), dtype=torch.bool)
    hardest = similarities.masked_fill(negative_mask, -torch.inf).max(dim=1).values.numpy()

    combined = torch.cat([palm, vein], dim=0)
    centered = combined - combined.mean(dim=0, keepdim=True)
    _, singular_values, right_vectors = torch.linalg.svd(
        centered, full_matrices=False
    )
    coordinates = centered @ right_vectors[:2].t()
    explained = singular_values.square()
    explained = (explained[:2] / explained.sum()).numpy()
    return {
        "labels": labels.numpy(),
        "same_similarity": same,
        "hardest_negative_similarity": hardest,
        "pca_coordinates": coordinates.numpy(),
        "pca_explained": explained,
    }


def plot_direct_cross_modal_diagnostic(data, output_dir, dpi, pca_identities):
    analysis = shared_space_analysis(data["shared_gallery"])
    same = analysis["same_similarity"]
    hardest = analysis["hardest_negative_similarity"]
    coordinates = analysis["pca_coordinates"]
    labels = analysis["labels"]
    explained = analysis["pca_explained"]
    num_identities = labels.size
    shown = min(pca_identities, num_identities)
    selected = np.linspace(0, num_identities - 1, shown, dtype=int)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.8, 4.9),
        gridspec_kw={"width_ratios": [0.82, 1.35]},
    )
    add_figure_header(
        fig,
        "Cross-modal structure in the shared identity space",
        (
            f"Tongji Session 1 test Gallery · {num_identities} unseen identities · "
            f"{data['dimensions']}-D shared features · PCA panel shows {shown} uniformly sampled identities"
        ),
    )

    ax = axes[0]
    groups = [same, hardest]
    positions = [1, 2]
    box = ax.boxplot(
        groups,
        positions=positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        tick_labels=["Paired\nsame identity", "Hardest\ncross identity"],
        medianprops={"color": COLORS["ink"], "linewidth": 1.5},
        whiskerprops={"color": COLORS["gray"], "linewidth": 1.0},
        capprops={"color": COLORS["gray"], "linewidth": 1.0},
        boxprops={"edgecolor": COLORS["ink"], "linewidth": 0.8},
    )
    box["boxes"][0].set_facecolor(COLORS["blue_light"])
    box["boxes"][1].set_facecolor(COLORS["orange_light"])
    rng = np.random.default_rng(2026)
    for position, values, color in zip(
        positions, groups, (COLORS["blue"], COLORS["orange"])
    ):
        jitter = rng.uniform(-0.13, 0.13, size=values.size)
        ax.scatter(
            position + jitter,
            values,
            s=11,
            color=color,
            alpha=0.42,
            linewidths=0,
            zorder=2,
        )
    alignment_rate = float((same > hardest).mean())
    ax.text(
        0.04,
        0.96,
        (
            f"Paired > hardest negative: {alignment_rate * 100:.1f}%\n"
            f"Median paired similarity: {np.median(same):.3f}\n"
            f"Median hardest negative: {np.median(hardest):.3f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.7,
        linespacing=1.35,
        bbox={
            "facecolor": COLORS["white"],
            "edgecolor": COLORS["gray_light"],
            "boxstyle": "round,pad=0.35",
            "alpha": 0.94,
        },
    )
    ax.set_title("Cross-modal similarity separation", pad=9)
    ax.set_ylabel("Cosine similarity")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.text(
        0.01,
        0.02,
        "(a)",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="semibold",
    )

    ax = axes[1]
    palm_coordinates = coordinates[:num_identities]
    vein_coordinates = coordinates[num_identities:]
    for index in selected:
        ax.plot(
            [palm_coordinates[index, 0], vein_coordinates[index, 0]],
            [palm_coordinates[index, 1], vein_coordinates[index, 1]],
            color=COLORS["gray_light"],
            linewidth=0.8,
            alpha=0.9,
            zorder=1,
        )
    ax.scatter(
        palm_coordinates[selected, 0],
        palm_coordinates[selected, 1],
        s=30,
        color=COLORS["blue"],
        marker="o",
        edgecolors=COLORS["white"],
        linewidths=0.5,
        label="Palmprint template",
        zorder=3,
    )
    ax.scatter(
        vein_coordinates[selected, 0],
        vein_coordinates[selected, 1],
        s=34,
        color=COLORS["orange"],
        marker="^",
        edgecolors=COLORS["white"],
        linewidths=0.5,
        label="Palm-vein template",
        zorder=3,
    )
    ax.set_title("PCA of paired identity templates", pad=9)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.grid(color=COLORS["grid"], linewidth=0.7)
    ax.legend(loc="best")
    ax.text(
        0.98,
        0.03,
        "Gray line connects the same identity",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.2,
        color=COLORS["gray"],
    )
    ax.text(
        0.01,
        0.02,
        "(b)",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="semibold",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.13, wspace=0.22)
    paths = save_figure(
        fig, output_dir, "diagnostic_direct_cross_modal_alignment", dpi
    )
    return paths, analysis


def canonical_shared_space_analysis(data):
    gallery = data["shared_gallery"]
    palm = gallery["palm"].float()
    vein = gallery["vein"].float()
    palm_centered = palm - palm.mean(dim=0, keepdim=True)
    vein_centered = vein - vein.mean(dim=0, keepdim=True)
    denominator = (
        palm_centered.square().sum(dim=0).sqrt()
        * vein_centered.square().sum(dim=0).sqrt()
    ).clamp_min(1e-12)
    test_component_correlations = (
        (palm_centered * vein_centered).sum(dim=0) / denominator
    ).numpy()
    train_correlations = (
        data["checkpoint"]["model"]["canonical_correlations"][
            : data["dimensions"]
        ]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    projected_margins = {}
    for scenario, values in data["scenarios"].items():
        projected_margins[scenario] = {
            "available": hard_negative_margin(
                values["available"],
                values["candidate_labels"],
                values["probe_labels"],
            ),
            "shared": hard_negative_margin(
                values["shared"],
                values["candidate_labels"],
                values["probe_labels"],
            ),
        }

    direct = shared_space_analysis(gallery)
    return {
        "labels": gallery["labels"].numpy(),
        "palm_templates": palm.numpy(),
        "vein_templates": vein.numpy(),
        "train_canonical_correlations": train_correlations,
        "test_component_correlations": test_component_correlations,
        "projected_margins": projected_margins,
        "direct_same_similarity": direct["same_similarity"],
        "direct_hardest_negative_similarity": direct[
            "hardest_negative_similarity"
        ],
    }


def rolling_average(values, window=11):
    values = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return values.copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def plot_canonical_shared_identity_space(data, output_dir, dpi):
    analysis = canonical_shared_space_analysis(data)
    train_correlations = analysis["train_canonical_correlations"]
    test_correlations = analysis["test_component_correlations"]
    ranks = np.arange(1, train_correlations.size + 1)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.8, 4.9),
        gridspec_kw={"width_ratios": [1.18, 1.0]},
    )
    add_figure_header(
        fig,
        "Cross-modal canonical structure and shared-space discrimination",
        (
            "CCA fitted on 432 training identities · component correspondence and "
            "identity margins evaluated on 120 disjoint Tongji test identities"
        ),
    )

    ax = axes[0]
    ax.plot(
        ranks,
        train_correlations,
        color=COLORS["blue"],
        linewidth=1.8,
        label="Fitted training correlation",
        zorder=3,
    )
    ax.scatter(
        ranks,
        test_correlations,
        s=10,
        color=COLORS["orange"],
        alpha=0.30,
        linewidths=0,
        label="Test Gallery component correlation",
        zorder=2,
    )
    ax.plot(
        ranks,
        rolling_average(test_correlations),
        color=COLORS["orange"],
        linewidth=1.6,
        linestyle="--",
        label="Test correlation · 11-component mean",
        zorder=4,
    )
    ax.axhline(0, color=COLORS["gray_light"], linewidth=0.9, zorder=0)
    ax.set_xlim(1, train_correlations.size)
    lower = min(-0.1, float(test_correlations.min()) - 0.04)
    ax.set_ylim(lower, 0.96)
    ax.set_title("Canonical correlation by component", pad=9)
    ax.set_xlabel("Canonical component rank")
    ax.set_ylabel("Palmprint–palm-vein correlation")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.legend(loc="upper right", fontsize=8.2)
    ax.text(
        0.97,
        0.54,
        (
            f"First 10 components\n"
            f"train mean: {train_correlations[:10].mean():.3f}\n"
            f"test mean: {test_correlations[:10].mean():.3f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        linespacing=1.35,
        bbox={
            "facecolor": COLORS["white"],
            "edgecolor": COLORS["gray_light"],
            "boxstyle": "round,pad=0.35",
            "alpha": 0.94,
        },
    )
    ax.text(
        0.01,
        0.02,
        "(a)",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="semibold",
    )

    ax = axes[1]
    order = [
        (PALMPRINT_MISSING, "available"),
        (PALMPRINT_MISSING, "shared"),
        (PALMVEIN_MISSING, "available"),
        (PALMVEIN_MISSING, "shared"),
    ]
    groups = [
        analysis["projected_margins"][scenario][representation]
        for scenario, representation in order
    ]
    positions = [1, 2, 4, 5]
    boxes = ax.boxplot(
        groups,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        tick_labels=[
            "Vein\navailable",
            "Vein\nshared",
            "Palmprint\navailable",
            "Palmprint\nshared",
        ],
        medianprops={"color": COLORS["ink"], "linewidth": 1.5},
        whiskerprops={"color": COLORS["gray"], "linewidth": 1.0},
        capprops={"color": COLORS["gray"], "linewidth": 1.0},
        boxprops={"edgecolor": COLORS["ink"], "linewidth": 0.8},
    )
    for index, patch in enumerate(boxes["boxes"]):
        patch.set_facecolor(
            COLORS["gray_light"] if index % 2 == 0 else COLORS["blue_light"]
        )
    rng = np.random.default_rng(2026)
    for index, (position, values) in enumerate(zip(positions, groups)):
        jitter = rng.uniform(-0.17, 0.17, size=values.size)
        ax.scatter(
            position + jitter,
            values,
            s=8,
            color=COLORS["gray"] if index % 2 == 0 else COLORS["blue"],
            alpha=0.18,
            linewidths=0,
            zorder=2,
        )
    ax.axhline(0, color=COLORS["ink"], linewidth=0.9, linestyle="--")
    ax.axvline(3, color=COLORS["gray_light"], linewidth=0.9)
    ax.set_title("Identity margins after shared projection", pad=9)
    ax.set_ylabel("Genuine score − highest impostor score")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    for x_position, scenario, modality in (
        (0.25, PALMPRINT_MISSING, "Vein"),
        (0.75, PALMVEIN_MISSING, "Palmprint"),
    ):
        available = analysis["projected_margins"][scenario]["available"]
        shared = analysis["projected_margins"][scenario]["shared"]
        ax.text(
            x_position,
            0.97,
            (
                f"{modality} Top-1 errors\n"
                f"{(available <= 0).sum()} → {(shared <= 0).sum()} of {shared.size}"
            ),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
            linespacing=1.3,
            bbox={
                "facecolor": COLORS["white"],
                "edgecolor": COLORS["gray_light"],
                "boxstyle": "round,pad=0.32",
                "alpha": 0.94,
            },
        )
    ax.text(
        0.01,
        0.02,
        "(b)",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="semibold",
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.80,
        bottom=0.14,
        wspace=0.19,
    )
    paths = save_figure(fig, output_dir, "figure4_shared_identity_space", dpi)
    return paths, analysis


def save_source_data(data, margins, shared_analysis, output_dir):
    arrays = {
        "shared_gallery_labels": shared_analysis["labels"],
        "shared_palm_templates": shared_analysis["palm_templates"],
        "shared_vein_templates": shared_analysis["vein_templates"],
        "train_canonical_correlations": shared_analysis[
            "train_canonical_correlations"
        ],
        "test_component_correlations": shared_analysis[
            "test_component_correlations"
        ],
        "direct_same_similarity": shared_analysis["direct_same_similarity"],
        "direct_hardest_negative_similarity": shared_analysis[
            "direct_hardest_negative_similarity"
        ],
    }
    for scenario, values in data["scenarios"].items():
        prefix = scenario.replace("-", "_")
        arrays[f"{prefix}_candidate_labels"] = values["candidate_labels"].numpy()
        arrays[f"{prefix}_probe_labels"] = values["probe_labels"].numpy()
        arrays[f"{prefix}_available_scores"] = values["available"].numpy()
        arrays[f"{prefix}_shared_scores"] = values["shared"].numpy()
        arrays[f"{prefix}_fused_scores"] = values["fused"].numpy()
        arrays[f"{prefix}_available_margin"] = margins[scenario][0]
        arrays[f"{prefix}_fused_margin"] = margins[scenario][1]
        arrays[f"{prefix}_shared_projection_margin"] = shared_analysis[
            "projected_margins"
        ][scenario]["shared"]
    path = output_dir / "figure_source_data.npz"
    np.savez_compressed(path, **arrays)
    return path


def build_manifest(args, data, margins, shared_analysis, figure_paths, data_path):
    scenario_records = {}
    for scenario, values in data["scenarios"].items():
        scenario_records[scenario] = {
            "available_modality": values["available_modality"],
            "target_modality": values["target_modality"],
            "alpha": values["alpha"],
            "cross_gallery_weight": values["cross_gallery_weight"],
            "metrics": {
                method: metric_record(
                    values[method],
                    values["candidate_labels"],
                    values["probe_labels"],
                )
                for method in ("available", "shared", "fused")
            },
            "margin": {
                "available_errors": int((margins[scenario][0] <= 0).sum()),
                "fused_errors": int((margins[scenario][1] <= 0).sum()),
                "resolved_errors": int(
                    (
                        (margins[scenario][0] <= 0)
                        & (margins[scenario][1] > 0)
                    ).sum()
                ),
                "new_errors": int(
                    (
                        (margins[scenario][0] > 0)
                        & (margins[scenario][1] <= 0)
                    ).sum()
                ),
                "median_delta": float(
                    np.median(margins[scenario][1] - margins[scenario][0])
                ),
            },
            "shared_projection_margin": {
                "available_errors": int(
                    (
                        shared_analysis["projected_margins"][scenario][
                            "available"
                        ]
                        <= 0
                    ).sum()
                ),
                "shared_errors": int(
                    (
                        shared_analysis["projected_margins"][scenario]["shared"]
                        <= 0
                    ).sum()
                ),
            },
        }
    same = shared_analysis["direct_same_similarity"]
    hardest = shared_analysis["direct_hardest_negative_similarity"]
    train_correlations = shared_analysis["train_canonical_correlations"]
    test_correlations = shared_analysis["test_component_correlations"]
    return {
        "method": "regularized shared-identity feature recovery",
        "architecture_version": ARCHITECTURE_VERSION,
        "shared_dimensions": data["dimensions"],
        "test_data_used_for_fitting_or_tuning": False,
        "source_fingerprints": {
            "checkpoint": file_sha256(args.ckpt),
            "palm_encoder": file_sha256(args.palm_ckpt),
            "vein_encoder": file_sha256(args.vein_ckpt),
            "gallery_protocol": file_sha256(args.gallery_list),
            "probe_protocol": file_sha256(args.protocol_list),
            "figure_source_data": file_sha256(data_path),
        },
        "figure_contracts": {
            "figure2_low_far_roc": (
                "Empirical verification curves; FAR is logarithmic and begins at "
                "the minimum supported positive FAR."
            ),
            "figure3_hard_negative_margin": (
                "One point per probe; axes compare genuine-minus-maximum-impostor "
                "margins before and after fusion."
            ),
            "figure4_shared_identity_space": (
                "Fitted training canonical correlations, corresponding unseen-test "
                "Gallery component correlations, and hard-negative identity margins "
                "before and after the shared projection."
            ),
        },
        "scenarios": scenario_records,
        "shared_space": {
            "num_identities": int(same.size),
            "mean_train_canonical_correlation": float(train_correlations.mean()),
            "mean_test_component_correlation": float(test_correlations.mean()),
            "positive_test_component_fraction": float(
                (test_correlations > 0).mean()
            ),
            "first_10_train_correlation_mean": float(
                train_correlations[:10].mean()
            ),
            "first_10_test_correlation_mean": float(
                test_correlations[:10].mean()
            ),
            "direct_cross_modal_diagnostic": {
                "paired_exceeds_hardest_negative": float(
                    (same > hardest).mean()
                ),
                "median_paired_similarity": float(np.median(same)),
                "median_hardest_negative_similarity": float(
                    np.median(hardest)
                ),
                "used_for_fitting_or_fusion_selection": False,
            },
        },
        "figures": {
            path.name: file_sha256(path)
            for path in figure_paths
        },
    }


def generate(args):
    if args.dpi < 150:
        raise ValueError("--dpi must be at least 150 for publication export")
    if args.include_alignment_diagnostic and args.pca_identities < 2:
        raise ValueError("--pca_identities must be at least 2")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    data = extract_figure_data(args)
    figure_paths = []
    figure_paths.extend(plot_low_far_roc(data, output_dir, args.dpi))
    margin_paths, margins = plot_hard_negative_margins(
        data, output_dir, args.dpi
    )
    figure_paths.extend(margin_paths)
    shared_paths, shared_analysis = plot_canonical_shared_identity_space(
        data,
        output_dir,
        args.dpi,
    )
    figure_paths.extend(shared_paths)
    if args.include_alignment_diagnostic:
        diagnostic_paths, _ = plot_direct_cross_modal_diagnostic(
            data,
            output_dir,
            args.dpi,
            args.pca_identities,
        )
        figure_paths.extend(diagnostic_paths)
    data_path = save_source_data(data, margins, shared_analysis, output_dir)
    manifest = build_manifest(
        args,
        data,
        margins,
        shared_analysis,
        figure_paths,
        data_path,
    )
    manifest_path = output_dir / "figure_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f"[Info] saved source data: {data_path}")
    print(f"[Info] saved manifest: {manifest_path}")
    for path in figure_paths:
        print(f"[Info] saved figure: {path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        "Generate shared feature recovery publication figures"
    )
    parser.add_argument(
        "--gallery_list",
        default="data_txt/tongji/ssfd_gallery_full.txt",
    )
    parser.add_argument(
        "--protocol_list",
        default="data_txt/tongji/ssfd_test_protocol.txt",
    )
    parser.add_argument(
        "--ckpt",
        default="outputs/shared_feature_recovery/best.pth",
    )
    parser.add_argument(
        "--palm_ckpt",
        default="outputs/encoders/palm_best.pth",
    )
    parser.add_argument(
        "--vein_ckpt",
        default="outputs/encoders/vein_best.pth",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/shared_feature_recovery/figures",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--embedding_size", type=int, default=256)
    parser.add_argument("--extract_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--pca_identities", type=int, default=40)
    parser.add_argument(
        "--include_alignment_diagnostic",
        action="store_true",
        help="Also export the direct cross-modal template-alignment diagnostic",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    generate(parse_args())

