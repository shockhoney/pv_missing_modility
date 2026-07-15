import warnings

import torch
import torch.nn.functional as F


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
    embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if embeddings.ndim != 2 or embeddings.size(0) != labels.numel():
        raise ValueError("Gallery embeddings must be [N, D] with one label per sample")
    gallery_labels = labels.unique(sorted=True)
    if gallery_labels.numel() == 0:
        raise ValueError("Gallery is empty")
    templates = torch.stack([embeddings[labels == label].mean(dim=0) for label in gallery_labels])
    return F.normalize(templates, dim=1), gallery_labels


def gallery_probe_scores(gallery_embeddings, gallery_labels, probe_embeddings):
    templates, template_labels = build_gallery_templates(gallery_embeddings, gallery_labels)
    probes = F.normalize(torch.as_tensor(probe_embeddings, dtype=torch.float32), dim=1)
    if probes.ndim != 2 or probes.size(1) != templates.size(1):
        raise ValueError("Probe and gallery embedding dimensions must match")
    return probes @ templates.t(), template_labels


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
    probe_labels = torch.as_tensor(probe_labels, dtype=torch.long)
    if scores.size(0) != probe_labels.numel():
        raise ValueError("Probe embeddings and labels must have the same length")

    genuine_mask = probe_labels[:, None].eq(template_labels[None, :])
    if not torch.all(genuine_mask.sum(dim=1) == 1):
        raise ValueError("Every probe identity must occur exactly once in the gallery templates")
    genuine_scores = scores[genuine_mask]
    impostor_scores = scores[~genuine_mask]
    far, tar = _verification_curve(genuine_scores, impostor_scores)

    far_points = tuple(float(point) for point in far_points)
    for point in far_points:
        if not 0.0 <= point <= 1.0:
            raise ValueError("FAR operating points must be in [0, 1]")
    far_count_resolution = 1.0 / impostor_scores.numel()
    minimum_nonzero_far = float(far[far > 0].min().item())
    for point in far_points:
        if point + 1e-12 < minimum_nonzero_far:
            warnings.warn(
                f"FAR={point:g} is below the minimum positive empirical FAR {minimum_nonzero_far:g} "
                f"(count resolution {far_count_resolution:g}); "
                "the returned TAR is measured at FAR=0.",
                RuntimeWarning,
                stacklevel=2,
            )

    topk = tuple(sorted(set(int(k) for k in topk if int(k) > 0)))
    if not topk:
        raise ValueError("At least one positive Top-k value is required")
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
