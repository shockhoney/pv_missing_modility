from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F


COSINE_EPS = 1e-6
DEFAULT_SUBSET_MASKS: Tuple[Tuple[bool, bool], ...] = (
    (False, False),
    (False, True),
    (True, False),
)


def subset_name(mask: Tuple[bool, bool]) -> str:
    if mask == (False, False):
        return "full"
    if mask == (False, True):
        return "palm_only"
    if mask == (True, False):
        return "vein_only"
    return "".join(["1" if item else "0" for item in mask])


def build_subset2id(subset_masks: Iterable[Tuple[bool, bool]]) -> Dict[Tuple[bool, bool], int]:
    subset_masks = list(subset_masks)
    return {tuple(bool(v) for v in mask): idx for idx, mask in enumerate(subset_masks)}


def gaussian_tail_probability(distance: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    std = std.clamp_min(COSINE_EPS)
    distribution = torch.distributions.Normal(
        torch.zeros_like(distance, dtype=distance.dtype, device=distance.device),
        std,
    )
    prob = 2.0 * (1.0 - distribution.cdf(distance.abs()))
    return prob.clamp_min(COSINE_EPS)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(a, b, dim=-1)


def _extract_subset_stats(stats: Dict, mask: Tuple[bool, bool]):
    subset2id = stats["subset2id"]
    subset_id = subset2id[tuple(bool(v) for v in mask)]
    return (
        stats["class_prototypes"][subset_id],
        stats["overall_centroids"][subset_id],
        stats["class_dist_std"][subset_id],
        stats["overall_dist_std"][subset_id],
    )


def score_embeddings(
    feat: torch.Tensor,
    masks: torch.Tensor,
    stats: Dict,
    recovery_confidence: torch.Tensor | None = None,
    temperature: float = 0.1,
    quality_mode: str = "log_prob",
) -> Dict[str, torch.Tensor]:
    """
    Score embeddings for DyMo selection.

    Returns:
        pred_class, ics_quality, ics_gaussian, open_quality, open_gaussian
    """
    device = feat.device
    batch_size = feat.size(0)

    pred_class = torch.zeros(batch_size, dtype=torch.long, device=device)
    ics_quality = torch.zeros(batch_size, dtype=feat.dtype, device=device)
    ics_gaussian = torch.zeros(batch_size, dtype=feat.dtype, device=device)
    open_quality = torch.zeros(batch_size, dtype=feat.dtype, device=device)
    open_gaussian = torch.zeros(batch_size, dtype=feat.dtype, device=device)

    if recovery_confidence is None:
        recovery_confidence = torch.ones(batch_size, dtype=feat.dtype, device=device)
    else:
        recovery_confidence = recovery_confidence.to(device=device, dtype=feat.dtype).clamp(0.0, 1.0)

    for idx in range(batch_size):
        mask_tuple = tuple(bool(v) for v in masks[idx].tolist())
        class_prototypes, overall_centroid, class_std, overall_std = _extract_subset_stats(stats, mask_tuple)
        class_prototypes = class_prototypes.to(device=device, dtype=feat.dtype)
        overall_centroid = overall_centroid.to(device=device, dtype=feat.dtype)
        class_std = class_std.to(device=device, dtype=feat.dtype)
        overall_std = overall_std.to(device=device, dtype=feat.dtype)

        scores = torch.mv(class_prototypes, feat[idx]) / temperature
        if quality_mode == "log_prob":
            quality_scores = torch.log_softmax(scores, dim=0)
        elif quality_mode == "probability":
            quality_scores = torch.softmax(scores, dim=0)
        else:
            raise ValueError(f"Unsupported selector quality_mode: {quality_mode}")
        quality_cls, pred = torch.max(quality_scores, dim=0)

        dist_cls = cosine_distance(class_prototypes[pred].unsqueeze(0), feat[idx].unsqueeze(0)).squeeze(0)
        dist_all = cosine_distance(overall_centroid.unsqueeze(0), feat[idx].unsqueeze(0)).squeeze(0)

        gaussian_cls = gaussian_tail_probability(dist_cls, class_std[pred])
        gaussian_all = gaussian_tail_probability(dist_all, overall_std)

        pred_class[idx] = pred
        ics_quality[idx] = quality_cls
        ics_gaussian[idx] = gaussian_cls
        open_quality[idx] = quality_cls
        open_gaussian[idx] = gaussian_all * recovery_confidence[idx]

    return {
        "pred_class": pred_class,
        "ics_quality": ics_quality,
        "ics_gaussian": ics_gaussian,
        "open_quality": open_quality,
        "open_gaussian": open_gaussian,
        "quality_mode": quality_mode,
    }


def calibrated_reward(
    before_quality: torch.Tensor,
    before_gaussian: torch.Tensor,
    after_quality: torch.Tensor,
    after_gaussian: torch.Tensor,
    quality_mode: str = "log_prob",
) -> Tuple[torch.Tensor, torch.Tensor]:
    calibration = torch.ones_like(before_quality)
    calibration = torch.where(
        after_gaussian < before_gaussian,
        after_gaussian / (before_gaussian + COSINE_EPS),
        calibration,
    )
    if quality_mode == "log_prob":
        reward = after_quality - before_quality + torch.log(calibration.clamp_min(COSINE_EPS))
    elif quality_mode == "probability":
        reward = after_quality * calibration - before_quality
    else:
        raise ValueError(f"Unsupported selector quality_mode: {quality_mode}")
    return reward, calibration


def compute_selection_rewards(
    before_scores: Dict[str, torch.Tensor],
    after_scores: Dict[str, torch.Tensor],
    quality_mode: str | None = None,
) -> Dict[str, torch.Tensor]:
    if quality_mode is None:
        quality_mode = after_scores.get("quality_mode", before_scores.get("quality_mode", "log_prob"))
    open_reward, open_calibration = calibrated_reward(
        before_scores["open_quality"],
        before_scores["open_gaussian"],
        after_scores["open_quality"],
        after_scores["open_gaussian"],
        quality_mode=quality_mode,
    )
    ics_reward, ics_calibration = calibrated_reward(
        before_scores["ics_quality"],
        before_scores["ics_gaussian"],
        after_scores["ics_quality"],
        after_scores["ics_gaussian"],
        quality_mode=quality_mode,
    )
    return {
        "open_reward": open_reward,
        "open_calibration": open_calibration,
        "ics_reward": ics_reward,
        "ics_calibration": ics_calibration,
    }


def compute_subset_statistics(
    feats: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Dict[str, torch.Tensor]:
    feats = F.normalize(feats, dim=1)
    labels = labels.long()
    feat_dim = feats.size(1)

    class_prototypes = torch.zeros(num_classes, feat_dim, dtype=feats.dtype)
    class_dist_std = torch.full((num_classes,), 1.0, dtype=feats.dtype)

    overall_centroid = F.normalize(feats.mean(dim=0, keepdim=True), dim=1).squeeze(0)
    overall_distance = cosine_distance(feats, overall_centroid.unsqueeze(0).expand_as(feats))
    overall_dist_std = overall_distance.std(unbiased=False).clamp_min(1e-3)

    for class_id in range(num_classes):
        loc = labels == class_id
        if loc.any():
            class_feats = feats[loc]
            proto = F.normalize(class_feats.mean(dim=0, keepdim=True), dim=1).squeeze(0)
            class_prototypes[class_id] = proto
            distances = cosine_distance(class_feats, proto.unsqueeze(0).expand_as(class_feats))
            class_dist_std[class_id] = distances.std(unbiased=False).clamp_min(1e-3)
        else:
            class_prototypes[class_id] = overall_centroid
            class_dist_std[class_id] = overall_dist_std

    return {
        "class_prototypes": F.normalize(class_prototypes, dim=1),
        "overall_centroid": overall_centroid,
        "class_dist_std": class_dist_std,
        "overall_dist_std": overall_dist_std,
    }
