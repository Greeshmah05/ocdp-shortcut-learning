"""
Evaluation metrics for shortcut learning and group robustness benchmarks.
Complies with evaluation protocol in .agents/rules/evaluation-protocol.md.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


# Canonical Waterbirds training group counts (used for weighted average accuracy)
WATERBIRDS_TRAIN_GROUP_PROPORTIONS = {
    0: 3498 / 4795,
    1: 184 / 4795,
    2: 56 / 4795,
    3: 1057 / 4795
}


def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray
) -> Dict[str, Any]:
    """
    Computes overall accuracy, per-group accuracy (G0..G3),
    mean group accuracy (MGA), worst-group accuracy (WGA),
    worst-class accuracy (WCA), and re-weighted average accuracy.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)

    total_samples = len(y_true)
    overall_acc = float(np.mean(y_true == y_pred))

    # Per-group accuracy
    group_accs = {}
    group_counts = {}
    for g in range(4):
        mask = (groups == g)
        count = int(np.sum(mask))
        group_counts[f"G{g}_count"] = count
        if count > 0:
            group_accs[f"G{g}_acc"] = float(np.mean(y_pred[mask] == y_true[mask]))
        else:
            group_accs[f"G{g}_acc"] = 0.0

    valid_group_accs = [group_accs[f"G{g}_acc"] for g in range(4) if group_counts[f"G{g}_count"] > 0]
    mean_group_acc = float(np.mean(valid_group_accs)) if valid_group_accs else 0.0
    worst_group_acc = float(np.min(valid_group_accs)) if valid_group_accs else 0.0

    # Per-class accuracy (for Tier 3 selection)
    class_accs = {}
    for c in range(2):
        mask = (y_true == c)
        if np.sum(mask) > 0:
            class_accs[f"C{c}_acc"] = float(np.mean(y_pred[mask] == y_true[mask]))
        else:
            class_accs[f"C{c}_acc"] = 0.0
    worst_class_acc = float(min(class_accs.values())) if class_accs else 0.0

    # Weighted average accuracy (Group DRO convention)
    weighted_acc = float(sum(
        group_accs[f"G{g}_acc"] * WATERBIRDS_TRAIN_GROUP_PROPORTIONS[g]
        for g in range(4)
    ))

    return {
        "overall_acc": overall_acc,
        "mean_group_acc": mean_group_acc,
        "worst_group_acc": worst_group_acc,
        "worst_class_acc": worst_class_acc,
        "weighted_avg_acc": weighted_acc,
        **group_accs,
        **group_counts,
        **class_accs,
        "total_samples": total_samples
    }


def compute_calibration_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15
) -> Dict[str, float]:
    """
    Computes Expected Calibration Error (ECE) and Brier Score.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)

    # ECE with equal-width binning
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    # Brier score (multi-class or binary)
    if probs.shape[1] == 2:
        brier = float(brier_score_loss(y_true, probs[:, 1]))
    else:
        one_hot = np.eye(probs.shape[1])[y_true]
        brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    return {
        "ece": float(ece),
        "brier_score": brier
    }


def compute_minority_detection_metrics(
    shortcut_scores: np.ndarray,
    groups: np.ndarray
) -> Dict[str, float]:
    """
    Evaluates how effectively the shortcut score detects minority-group images (G1, G2).
    In Waterbirds, G1 (landbird on water) and G2 (waterbird on land) are bias-conflicting.
    A lower context confidence in the true label s_i = p_ctx(y_i|x_i) indicates bias conflict.
    Minority detector signal: 1.0 - s_i.
    """
    groups = np.asarray(groups)
    shortcut_scores = np.asarray(shortcut_scores)
    
    # Binary label: 1 if minority (G1 or G2), 0 if majority (G0 or G3)
    is_minority = ((groups == 1) | (groups == 2)).astype(int)
    anomaly_signal = 1.0 - shortcut_scores

    try:
        auroc = float(roc_auc_score(is_minority, anomaly_signal))
    except Exception:
        auroc = 0.5

    try:
        auprc = float(average_precision_score(is_minority, anomaly_signal))
    except Exception:
        auprc = 0.0

    return {
        "minority_detection_auroc": auroc,
        "minority_detection_auprc": auprc
    }
