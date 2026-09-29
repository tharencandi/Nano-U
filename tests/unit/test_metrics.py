"""Unit tests for src/utils/metrics.py (BinaryIoU) and src/metrics.py (canonical segmentation metrics)."""

import numpy as np
import pytest
import tensorflow as tf

from src.utils.metrics import BinaryIoU
from src.metrics import (
    compute_confusion_counts,
    compute_metrics_from_counts,
    threshold_sweep,
    evaluate_segmentation,
)


def _make_iou(threshold=0.5):
    return BinaryIoU(threshold=threshold)


# ── Legacy BinaryIoU correctness ──────────────────────────────────────────────

def test_binary_iou_perfect():
    """Mix of fg/bg, perfect prediction → mIoU == 1.0."""
    m = _make_iou()
    rng = np.random.default_rng(0)
    y = tf.constant((rng.random((1, 8, 8, 1)) > 0.5).astype(np.float32))
    m.update_state(y, y)
    assert abs(m.result().numpy() - 1.0) < 1e-5


def test_binary_iou_all_foreground():
    """All pixels foreground, perfect pred → fg IoU=1, bg IoU=0 → mIoU=0.5."""
    m = _make_iou()
    y = tf.ones((1, 8, 8, 1))
    m.update_state(y, y)
    assert abs(m.result().numpy() - 0.5) < 1e-5


def test_binary_iou_no_overlap():
    """Pred all foreground, label all background → IoU 0 for both classes → ~0."""
    m = _make_iou()
    y_true = tf.zeros((1, 4, 4, 1))
    y_pred = tf.ones((1, 4, 4, 1))
    m.update_state(y_true, y_pred)
    assert m.result().numpy() < 0.1


def test_binary_iou_half_overlap():
    m = _make_iou()
    y_true = tf.constant([[[[1.], [1.], [0.], [0.]]]])  # shape (1,1,4,1)
    y_pred = tf.constant([[[[1.], [0.], [1.], [0.]]]])
    m.update_state(y_true, y_pred)
    result = m.result().numpy()
    assert 0.0 < result < 1.0


# ── Canonical src.metrics Tests ───────────────────────────────────────────────

def test_confusion_counts_perfect_binary():
    """Test confusion counts on known synthetic grid."""
    # 2 images, 2x2 pixels = 8 pixels total
    # Image 0: 2 TP, 2 TN (perfect)
    # Image 1: 1 TP, 1 FP, 1 FN, 1 TN
    y_true = np.array([
        [[1, 1], [0, 0]],
        [[1, 0], [1, 0]],
    ])
    y_pred = np.array([
        [[1, 1], [0, 0]],
        [[1, 1], [0, 0]],
    ])

    # Global counts
    tp, fp, fn, tn = compute_confusion_counts(y_true, y_pred, threshold=0.5, per_image=False)
    assert tp == 3.0  # 2 in img0, 1 in img1
    assert fp == 1.0  # 0 in img0, 1 in img1
    assert fn == 1.0  # 0 in img0, 1 in img1
    assert tn == 3.0  # 2 in img0, 1 in img1
    assert tp + fp + fn + tn == 8.0

    # Per-image counts
    tp_i, fp_i, fn_i, tn_i = compute_confusion_counts(y_true, y_pred, threshold=0.5, per_image=True)
    assert np.array_equal(tp_i, [2.0, 1.0])
    assert np.array_equal(fp_i, [0.0, 1.0])
    assert np.array_equal(fn_i, [0.0, 1.0])
    assert np.array_equal(tn_i, [2.0, 1.0])


def test_metrics_from_counts_mathematical_precision():
    """Verify exact formula values for known counts."""
    # TP=30, FP=10, FN=10, TN=50
    # total = 100
    # iou_fg = 30 / (30 + 10 + 10) = 30/50 = 0.60
    # iou_bg = 50 / (50 + 10 + 10) = 50/70 = 0.7142857
    # miou_macro = 0.5 * (0.60 + 0.7142857) = 0.65714285
    # precision = 30 / (30 + 10) = 0.75
    # recall = 30 / (30 + 10) = 0.75
    # f1 / dice = 60 / (60 + 10 + 10) = 60/80 = 0.75
    # accuracy = 80 / 100 = 0.80
    m = compute_metrics_from_counts(tp=30, fp=10, fn=10, tn=50)
    assert abs(m["iou_foreground"] - 0.60) < 1e-5
    assert abs(m["iou_background"] - (50.0 / 70.0)) < 1e-5
    assert abs(m["miou_macro"] - 0.5 * (0.60 + 50.0 / 70.0)) < 1e-5
    assert abs(m["precision"] - 0.75) < 1e-5
    assert abs(m["recall"] - 0.75) < 1e-5
    assert abs(m["f1"] - 0.75) < 1e-5
    assert abs(m["accuracy"] - 0.80) < 1e-5


def test_foreground_vs_2class_inflation():
    """Demonstrate why background dominance inflates 2-class mIoU over foreground IoU."""
    # Agricultural scenario: 80% background, 20% path
    # TN=75, FP=5, FN=5, TP=15 (Total 100 pixels)
    # Path (foreground) IoU = 15 / (15 + 5 + 5) = 15/25 = 0.60
    # Background IoU = 75 / (75 + 5 + 5) = 75/85 = 0.8824
    # 2-class mIoU = 0.5 * (0.60 + 0.8824) = 0.7412 (+14.1 pp inflation!)
    m = compute_metrics_from_counts(tp=15, fp=5, fn=5, tn=75)
    assert abs(m["iou_foreground"] - 0.60) < 1e-5
    assert m["miou_macro"] > m["iou_foreground"] + 0.14


def test_threshold_sweep_monotonic_properties():
    """Verify threshold sweep on predicted probabilities."""
    y_true = np.array([[[1.0, 1.0], [0.0, 0.0]]])
    # Predictions strictly ordered
    y_pred = np.array([[[0.9, 0.6], [0.4, 0.1]]])

    sweep = threshold_sweep(y_true, y_pred, thresholds=[0.2, 0.5, 0.8])
    # As threshold increases, precision should increase or stay same, recall should decrease or stay same
    precisions = sweep["precision"]
    recalls = sweep["recall"]
    assert precisions[0] <= precisions[-1]
    assert recalls[0] >= recalls[-1]
    assert "optimal_threshold_iou_fg" in sweep


def test_evaluate_segmentation_full_pipeline():
    """Test the complete high-level evaluate_segmentation function."""
    rng = np.random.default_rng(42)
    masks = (rng.random((10, 60, 80, 1)) > 0.6).astype(np.float32)
    probs = np.clip(masks + rng.normal(0, 0.2, masks.shape), 0.0, 1.0)

    res = evaluate_segmentation(masks, probs, threshold=0.5, compute_sweep=True)
    assert "global" in res
    assert "per_image" in res
    assert "sweep" in res

    # Global metrics present and within [0, 1]
    for key in ("iou_foreground", "iou_background", "miou_macro", "precision", "recall", "f1"):
        assert 0.0 <= res["global"][key] <= 1.0
        assert 0.0 <= res["per_image"][f"{key}_mean"] <= 1.0
