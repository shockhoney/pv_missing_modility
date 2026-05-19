import numpy as np
from sklearn.metrics import roc_curve, auc

def _ensure_numpy(x):
    return np.asarray(x)

def _safe_threshold(thresholds, idx):
    if np.isfinite(thresholds[idx]):
        return thresholds[idx]
    finite = np.where(np.isfinite(thresholds))[0]
    if finite.size == 0:
        return 0.0
    return thresholds[finite[np.argmin(np.abs(finite - idx))]]

def compute_confusion_from_scores(scores, labels, threshold, is_similarity=True):

    scores = _ensure_numpy(scores)
    labels = _ensure_numpy(labels).astype(int)
    if is_similarity:
        preds = (scores >= threshold).astype(int)
    else:
        preds = (scores <= threshold).astype(int)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    return tp, fp, tn, fn

def far_frr_acc_at_threshold(scores, labels, threshold, is_similarity=True):
    tp, fp, tn, fn = compute_confusion_from_scores(scores, labels, threshold, is_similarity)
    # definitions:
    # FAR = FP / (FP + TN)
    # FRR = FN / (FN + TP)
    # ACC = (TP + TN) / total
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    acc = (tp + tn) / (tp + fp + tn + fn)
    return {'FAR':far, 'FRR':frr, 'ACC':acc}

def compute_eer(scores, labels, is_similarity=True, return_threshold=False):

    scores = _ensure_numpy(scores)
    labels = _ensure_numpy(labels).astype(int)
    # roc_curve assumes labels: positive label = 1, scores higher -> positive
    if not is_similarity:
        # invert distances to similarity by negative sign
        scores_ = -scores
    else:
        scores_ = scores

    fpr, tpr, thresholds = roc_curve(labels, scores_, pos_label=1)
    fnr = 1 - tpr
    # find point where abs(fpr - fnr) minimal
    abs_diffs = np.abs(fpr - fnr)
    idxE = np.nanargmin(abs_diffs)
    eer = (fpr[idxE] + fnr[idxE]) / 2.0  # approximate
    # better: linear interpolation between nearest points around crossing
    # find where fpr - fnr changes sign
    diffs = fpr - fnr
    # if sign change exists, interpolate
    sign_changes = np.where(np.sign(diffs[:-1]) != np.sign(diffs[1:]))[0]
    if sign_changes.size > 0:
        i = sign_changes[0]
        # linear interpolation for threshold where fpr == fnr
        x = [diffs[i], diffs[i+1]]
        y = [fpr[i], fpr[i+1]]
        # interpolation fraction
        if (x[1] - x[0]) != 0:
            frac = -x[0] / (x[1] - x[0])
        else:
            frac = 0.0
        eer_interp = y[0] + frac * (y[1] - y[0])
        eer = eer_interp
        if np.isfinite(thresholds[i]) and np.isfinite(thresholds[i + 1]):
            thr = thresholds[i] + frac * (thresholds[i + 1] - thresholds[i])
        else:
            thr = _safe_threshold(thresholds, i + 1)
    else:
        thr = _safe_threshold(thresholds, idxE)
    if return_threshold:
        # if we inverted scores earlier, thr corresponds to scores_
        if not is_similarity:
            thr = -thr
        return eer, thr
    return eer

def roc_auc(scores, labels, is_similarity=True):
    scores = _ensure_numpy(scores)
    labels = _ensure_numpy(labels).astype(int)
    if not is_similarity:
        scores_ = -scores
    else:
        scores_ = scores
    fpr, tpr, thresholds = roc_curve(labels, scores_, pos_label=1)
    return fpr, tpr, thresholds, auc(fpr, tpr)

def tar_at_far(scores, labels, target_far, is_similarity=True):

    scores = _ensure_numpy(scores)
    labels = _ensure_numpy(labels).astype(int)
    # convert for roc_curve if needed
    scores_ = -scores if not is_similarity else scores
    fpr, tpr, thresholds = roc_curve(labels, scores_, pos_label=1)
    # fpr == FAR, tpr == TAR
    # find indices where fpr <= target_far, choose max tpr among them
    idxs = np.where(fpr <= target_far)[0]
    if idxs.size > 0:
        idx = idxs[-1]  # largest threshold with fpr <= target_far
        return {'TAR': float(tpr[idx]), 'FAR': float(fpr[idx]), 'threshold': float(thresholds[idx])}
    # else FAR never <= target_far -> return smallest FAR (closest)
    idx = np.argmin(np.abs(fpr - target_far))
    return {'TAR': float(tpr[idx]), 'FAR': float(fpr[idx]), 'threshold': float(thresholds[idx])}

