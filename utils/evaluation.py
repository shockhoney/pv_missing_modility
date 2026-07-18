import math
import warnings
from numbers import Integral, Real

import torch
import torch.nn.functional as F


def _as_label_vector(values, name, device=None, allow_empty=False):
    try:
        raw = torch.as_tensor(values, device=device)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a one-dimensional integer label vector") from error
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not allow_empty and raw.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if raw.dtype == torch.bool or raw.is_complex():
        raise ValueError(f"{name} must contain integer labels")
    if raw.is_floating_point():
        if not torch.isfinite(raw).all():
            raise ValueError(f"{name} must contain only finite labels")
        if not torch.equal(raw, raw.round()):
            raise ValueError(f"{name} must contain integer-valued labels")
    return raw.to(dtype=torch.long)


def _as_score_matrix(values, name="scores", device=None):
    try:
        scores = torch.as_tensor(values, dtype=torch.float32, device=device)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric two-dimensional matrix") from error
    if scores.ndim != 2 or scores.size(0) == 0 or scores.size(1) == 0:
        raise ValueError(f"{name} must be a non-empty [num_probes, num_candidates] matrix")
    if not torch.isfinite(scores).all():
        raise ValueError(f"{name} must contain only finite values")
    return scores


def _validate_topk(topk):
    try:
        values = tuple(topk)
    except TypeError as error:
        raise ValueError("topk must be an iterable of positive integers") from error
    validated = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
            raise ValueError("Top-k values must be positive integers")
        validated.append(int(value))
    validated = tuple(sorted(set(validated)))
    if not validated:
        raise ValueError("At least one positive Top-k value is required")
    return validated


def _validate_far_points(far_points):
    try:
        values = tuple(far_points)
    except TypeError as error:
        raise ValueError("far_points must be an iterable of finite values in [0, 1]") from error
    validated = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("FAR operating points must be finite real values in [0, 1]")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("FAR operating points must be finite real values in [0, 1]")
        if value not in validated:
            validated.append(value)
    if not validated:
        raise ValueError("At least one FAR operating point is required")
    return tuple(validated)


def _validate_score_matrix_inputs(scores, candidate_labels, probe_labels):
    scores = _as_score_matrix(scores)
    candidate_labels = _as_label_vector(
        candidate_labels, "candidate_labels", device=scores.device
    )
    probe_labels = _as_label_vector(probe_labels, "probe_labels", device=scores.device)
    if scores.size(0) != probe_labels.numel():
        raise ValueError("Scores rows must match the number of probe labels")
    if scores.size(1) != candidate_labels.numel():
        raise ValueError("Candidate labels must match the score-matrix columns")
    if candidate_labels.unique().numel() != candidate_labels.numel():
        raise ValueError("Candidate labels must be unique")
    if candidate_labels.numel() < 2:
        raise ValueError("At least two candidate identities are required for verification metrics")
    genuine_counts = probe_labels[:, None].eq(candidate_labels[None, :]).sum(dim=1)
    if not torch.all(genuine_counts == 1):
        raise ValueError("Every probe identity must occur exactly once in the candidate labels")
    return scores, candidate_labels, probe_labels


def count_correct_predictions(logits, labels):
    logits = torch.as_tensor(logits)
    labels = torch.as_tensor(labels)
    if labels.numel() == 0:
        return 0
    return int((logits.argmax(1) == labels).sum().item())


def recognition_rate(logits, labels):
    logits = torch.as_tensor(logits)
    labels = torch.as_tensor(labels)
    if labels.numel() == 0:
        return 0.0
    return count_correct_predictions(logits, labels) / labels.numel()


def build_gallery_templates(embeddings, labels):
    embeddings = _as_score_matrix(embeddings, "gallery_embeddings")
    labels = _as_label_vector(labels, "gallery_labels", device=embeddings.device)
    if embeddings.size(0) != labels.numel():
        raise ValueError("Gallery embeddings must have one label per sample")
    gallery_labels = labels.unique(sorted=True)
    templates = torch.stack([embeddings[labels == label].mean(dim=0) for label in gallery_labels])
    if torch.any(templates.norm(dim=1) <= 0):
        raise ValueError("Every gallery identity template must have non-zero norm")
    return F.normalize(templates, dim=1), gallery_labels


def gallery_probe_scores(gallery_embeddings, gallery_labels, probe_embeddings):
    templates, template_labels = build_gallery_templates(gallery_embeddings, gallery_labels)
    probes = _as_score_matrix(probe_embeddings, "probe_embeddings", device=templates.device)
    if probes.size(1) != templates.size(1):
        raise ValueError("Probe and gallery embedding dimensions must match")
    if torch.any(probes.norm(dim=1) <= 0):
        raise ValueError("Every probe embedding must have non-zero norm")
    return F.normalize(probes, dim=1) @ templates.t(), template_labels


def _verification_curve(genuine_scores, impostor_scores):
    genuine = torch.as_tensor(genuine_scores, dtype=torch.float64).flatten()
    impostor = torch.as_tensor(impostor_scores, dtype=torch.float64).flatten()
    if genuine.numel() == 0 or impostor.numel() == 0:
        raise ValueError("Both genuine and impostor scores are required")

    scores = torch.cat([genuine, impostor])
    genuine_mask = torch.cat(
        [torch.ones_like(genuine, dtype=torch.bool), torch.zeros_like(impostor, dtype=torch.bool)]
    )
    order = scores.argsort(descending=True)
    scores = scores[order]
    genuine_mask = genuine_mask[order]
    group_ends = torch.cat([
        torch.nonzero(scores[1:] != scores[:-1], as_tuple=False).flatten(),
        scores.new_tensor([scores.numel() - 1], dtype=torch.long),
    ])
    true_accepts = genuine_mask.cumsum(0)[group_ends].double()
    false_accepts = (~genuine_mask).cumsum(0)[group_ends].double()
    tar = torch.cat([genuine.new_zeros(1), true_accepts / genuine.numel()])
    far = torch.cat([impostor.new_zeros(1), false_accepts / impostor.numel()])
    return far, tar


def _eer_from_curve(far, tar):
    difference = far - (1.0 - tar)
    upper = int(torch.nonzero(difference >= 0, as_tuple=False)[0].item())
    if upper == 0 or difference[upper] == 0:
        return float(far[upper].item())
    lower = upper - 1
    weight = float((-difference[lower] / (difference[upper] - difference[lower])).item())
    return float((far[lower] + weight * (far[upper] - far[lower])).item())


def _tar_from_curve(far, tar, target_far):
    if not 0.0 <= target_far <= 1.0:
        raise ValueError("target_far must be in [0, 1]")
    valid = far <= target_far + 1e-12
    return float(tar[valid].max().item())


def gallery_probe_metrics(
    gallery_embeddings,
    gallery_labels,
    probe_embeddings,
    probe_labels,
    topk=(1, 5),
    far_points=(1e-3, 1e-4),
):
    scores, template_labels = gallery_probe_scores(gallery_embeddings, gallery_labels, probe_embeddings)
    return score_matrix_metrics(
        scores,
        template_labels,
        probe_labels,
        topk=topk,
        far_points=far_points,
    )


def score_matrix_metrics(
    scores,
    candidate_labels,
    probe_labels,
    topk=(1, 5),
    far_points=(1e-3, 1e-4),
    warn_far_resolution=True,
):
    """Evaluate a precomputed probe-by-identity score matrix.

    This is useful when the final score combines several embedding domains and
    therefore cannot be represented by one concatenated cosine embedding.
    """
    scores, template_labels, probe_labels = _validate_score_matrix_inputs(
        scores, candidate_labels, probe_labels
    )
    if not isinstance(warn_far_resolution, bool):
        raise ValueError("warn_far_resolution must be a bool")

    genuine_mask = probe_labels[:, None].eq(template_labels[None, :])
    genuine_scores = scores[genuine_mask]
    impostor_scores = scores[~genuine_mask]
    far, tar = _verification_curve(genuine_scores, impostor_scores)

    far_points = _validate_far_points(far_points)
    far_count_resolution = 1.0 / impostor_scores.numel()
    minimum_nonzero_far = float(far[far > 0].min().item())
    for point in far_points:
        if warn_far_resolution and point + 1e-12 < minimum_nonzero_far:
            warnings.warn(
                f"FAR={point:g} is below the minimum positive empirical FAR {minimum_nonzero_far:g} "
                f"(count resolution {far_count_resolution:g}); "
                "the returned TAR is measured at FAR=0.",
                RuntimeWarning,
                stacklevel=2,
            )

    topk = _validate_topk(topk)
    max_k = min(max(topk), template_labels.numel())
    ranked_labels = template_labels[scores.topk(max_k, dim=1).indices]
    topk_accuracy = {
        k: float(ranked_labels[:, : min(k, max_k)].eq(probe_labels[:, None]).any(dim=1).float().mean().item())
        for k in topk
    }
    return {
        "eer": _eer_from_curve(far, tar),
        "tar_at_far": {float(point): _tar_from_curve(far, tar, point) for point in far_points},
        "topk": topk_accuracy,
        "far_count_resolution": far_count_resolution,
        "minimum_nonzero_far": minimum_nonzero_far,
        "num_gallery_identities": int(template_labels.numel()),
        "num_probes": int(probe_labels.numel()),
        "num_genuine_scores": int(genuine_scores.numel()),
        "num_impostor_scores": int(impostor_scores.numel()),
    }


def format_gallery_probe_metrics(metrics):
    lines = [
        (
            f"Gallery identities={metrics['num_gallery_identities']}, Probes={metrics['num_probes']}, "
            f"Genuine={metrics['num_genuine_scores']}, Impostor={metrics['num_impostor_scores']}, "
            f"FAR count resolution={metrics['far_count_resolution']:g}, "
            f"minimum positive FAR={metrics['minimum_nonzero_far']:g}"
        ),
        f"EER (%): {metrics['eer'] * 100:.2f}",
    ]
    for far, tar in metrics["tar_at_far"].items():
        unresolved = (
            " [FAR=0 empirical point]"
            if far + 1e-12 < metrics["minimum_nonzero_far"]
            else ""
        )
        lines.append(f"TAR@FAR={far:g} (%): {tar * 100:.2f}{unresolved}")
    lines.append(
        "Top-k Accuracy: "
        + ", ".join(f"Top-{k}={value * 100:.2f}%" for k, value in metrics["topk"].items())
    )
    return lines
