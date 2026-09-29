"""
Canonical semantic segmentation metrics module.

Provides a single, mathematically verified scoring authority across training,
evaluation, linear probing, and embedded hardware validation.

Primary Metric:
    - Foreground IoU (iou_foreground): TP / (TP + FP + FN + eps)
      Represents actual path traversability IoU, uninflated by background dominance.

Secondary Metrics:
    - Background IoU (iou_background): TN / (TN + FP + FN + eps)
    - 2-Class Macro Mean IoU (miou_macro): 0.5 * (iou_foreground + iou_background)
    - Precision: TP / (TP + FP + eps)
    - Recall: TP / (TP + FN + eps)
    - F1 / Dice: 2*TP / (2*TP + FP + FN + eps)
    - Pixel Accuracy: (TP + TN) / (TP + TN + FP + FN + eps)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Sequence, Union

EPS = 1e-7


def compute_confusion_counts(
    y_true: Union[np.ndarray, Sequence],
    y_pred: Union[np.ndarray, Sequence],
    threshold: float = 0.5,
    per_image: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute TP, FP, FN, TN counts from ground truth binary masks and predictions.

    Args:
        y_true: Ground truth mask array (N, H, W) or (N, H, W, 1), binary or float.
        y_pred: Predicted probabilities or binary masks of matching shape.
        threshold: Decision threshold for binarizing y_pred.
        per_image: If True, returns counts per image (shape: (N,)).
                   If False, returns scalar counts summed over the dataset.

    Returns:
        (tp, fp, fn, tn) as float64 (scalars if per_image=False, arrays of shape (N,) if True).
    """
    y_t = np.asarray(y_true, dtype=np.float32)
    y_p = np.asarray(y_pred, dtype=np.float32)

    # Ensure at least 3D: (N, H, W)
    if y_t.ndim == 2:
        y_t = np.expand_dims(y_t, axis=0)
    if y_p.ndim == 2:
        y_p = np.expand_dims(y_p, axis=0)

    # Squeeze trailing channel dimension if present: (N, H, W, 1) -> (N, H, W)
    if y_t.ndim == 4 and y_t.shape[-1] == 1:
        y_t = np.squeeze(y_t, axis=-1)
    if y_p.ndim == 4 and y_p.shape[-1] == 1:
        y_p = np.squeeze(y_p, axis=-1)

    mask_bin = y_t > 0.5
    pred_bin = y_p > threshold

    # Axis over spatial dimensions (height, width)
    axis = tuple(range(1, mask_bin.ndim))

    tp_i = np.sum(mask_bin & pred_bin, axis=axis).astype(np.float64)
    fp_i = np.sum((~mask_bin) & pred_bin, axis=axis).astype(np.float64)
    fn_i = np.sum(mask_bin & (~pred_bin), axis=axis).astype(np.float64)
    tn_i = np.sum((~mask_bin) & (~pred_bin), axis=axis).astype(np.float64)

    if per_image:
        return tp_i, fp_i, fn_i, tn_i
    return float(tp_i.sum()), float(fp_i.sum()), float(fn_i.sum()), float(tn_i.sum())


def compute_metrics_from_counts(
    tp: Union[float, np.ndarray],
    fp: Union[float, np.ndarray],
    fn: Union[float, np.ndarray],
    tn: Union[float, np.ndarray],
    eps: float = EPS,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute standardized segmentation metrics from confusion counts.

    Args:
        tp, fp, fn, tn: Pixel confusion counts (scalars or arrays).
        eps: Epsilon smoothing term to prevent division by zero.

    Returns:
        Dict mapping metric names to values.
    """
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)
    tn = np.asarray(tn, dtype=np.float64)

    total_pixels = tp + fp + fn + tn + eps
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    dice = f1
    iou_fg = tp / (tp + fp + fn + eps)
    iou_bg = tn / (tn + fp + fn + eps)
    miou_macro = 0.5 * (iou_fg + iou_bg)
    accuracy = (tp + tn) / total_pixels

    def _scalar_or_array(val):
        if val.ndim == 0:
            return float(val)
        return val

    return {
        "iou_foreground": _scalar_or_array(iou_fg),
        "iou_background": _scalar_or_array(iou_bg),
        "miou_macro": _scalar_or_array(miou_macro),
        "precision": _scalar_or_array(precision),
        "recall": _scalar_or_array(recall),
        "f1": _scalar_or_array(f1),
        "dice": _scalar_or_array(dice),
        "accuracy": _scalar_or_array(accuracy),
    }


def threshold_sweep(
    y_true: Union[np.ndarray, Sequence],
    y_pred_prob: Union[np.ndarray, Sequence],
    thresholds: Optional[Sequence[float]] = None,
) -> Dict[str, list]:
    """
    Perform a threshold sweep to generate Precision-Recall curves and find the optimal threshold.

    Args:
        y_true: Ground truth binary masks.
        y_pred_prob: Predicted probability maps in [0, 1].
        thresholds: List of thresholds to evaluate. Defaults to 19 points in [0.05, 0.95].

    Returns:
        Dict containing threshold sweep lists and optimal threshold indices.
    """
    if thresholds is None:
        thresholds = [float(t) for t in np.round(np.linspace(0.05, 0.95, 19), 3)]

    sweep = {
        "thresholds": list(thresholds),
        "iou_foreground": [],
        "iou_background": [],
        "miou_macro": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    best_iou_fg = -1.0
    best_thresh_fg = 0.5

    for t in thresholds:
        tp, fp, fn, tn = compute_confusion_counts(y_true, y_pred_prob, threshold=t, per_image=False)
        m = compute_metrics_from_counts(tp, fp, fn, tn)
        for k in ("iou_foreground", "iou_background", "miou_macro", "precision", "recall", "f1"):
            sweep[k].append(m[k])
        if m["iou_foreground"] > best_iou_fg:
            best_iou_fg = m["iou_foreground"]
            best_thresh_fg = t

    sweep["optimal_threshold_iou_fg"] = float(best_thresh_fg)
    sweep["optimal_iou_fg"] = float(best_iou_fg)
    return sweep


def evaluate_segmentation(
    y_true: Union[np.ndarray, Sequence],
    y_pred_prob: Union[np.ndarray, Sequence],
    threshold: float = 0.5,
    compute_sweep: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of segmentation predictions.

    Reports:
        1. 'global': Metrics computed on pooled dataset pixels (primary).
        2. 'per_image': Mean and std across per-image metrics (secondary).
        3. 'sweep': Complete threshold sweep and optimal threshold.

    Args:
        y_true: Ground truth binary masks.
        y_pred_prob: Predicted probability maps in [0, 1].
        threshold: Operational decision threshold.
        compute_sweep: Whether to compute threshold sweep.

    Returns:
        Structured dictionary of evaluation results.
    """
    # 1. Global (pooled) metrics
    tp_g, fp_g, fn_g, tn_g = compute_confusion_counts(y_true, y_pred_prob, threshold=threshold, per_image=False)
    global_metrics = compute_metrics_from_counts(tp_g, fp_g, fn_g, tn_g)
    global_metrics["threshold"] = float(threshold)

    # 2. Per-image metrics
    tp_i, fp_i, fn_i, tn_i = compute_confusion_counts(y_true, y_pred_prob, threshold=threshold, per_image=True)
    per_img_metrics = compute_metrics_from_counts(tp_i, fp_i, fn_i, tn_i)
    n_images = len(tp_i)

    per_image_summary = {
        "n_images": n_images,
        "iou_foreground_mean": float(np.mean(per_img_metrics["iou_foreground"])),
        "iou_foreground_std": float(np.std(per_img_metrics["iou_foreground"])),
        "iou_background_mean": float(np.mean(per_img_metrics["iou_background"])),
        "iou_background_std": float(np.std(per_img_metrics["iou_background"])),
        "miou_macro_mean": float(np.mean(per_img_metrics["miou_macro"])),
        "miou_macro_std": float(np.std(per_img_metrics["miou_macro"])),
        "precision_mean": float(np.mean(per_img_metrics["precision"])),
        "precision_std": float(np.std(per_img_metrics["precision"])),
        "recall_mean": float(np.mean(per_img_metrics["recall"])),
        "recall_std": float(np.std(per_img_metrics["recall"])),
        "f1_mean": float(np.mean(per_img_metrics["f1"])),
        "f1_std": float(np.std(per_img_metrics["f1"])),
        "accuracy_mean": float(np.mean(per_img_metrics["accuracy"])),
        "accuracy_std": float(np.std(per_img_metrics["accuracy"])),
    }

    results = {
        "global": global_metrics,
        "per_image": per_image_summary,
    }

    if compute_sweep:
        results["sweep"] = threshold_sweep(y_true, y_pred_prob)

    return results
