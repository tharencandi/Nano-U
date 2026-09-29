#!/usr/bin/env python3
"""
Automated 4-Fold Representation Probing & Feature Emergence Aggregator.

Executes linear diagnostic probing across all 4 Leave-One-Scene-Out (LOSO)
cross-validation folds of TinyAgri, computes per-layer emergence statistics
(mean ± std IoU_fg and mIoU), and generates publication-grade cross-validation
plots for the Master thesis manuscript (Ballan Lab crossover).
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.probing.evaluator import run_probing_suite, STAGE_LABELS


def main():
    print("=" * 75)
    print("  4-FOLD CROSS-VALIDATION LAYER PROBING SUITE (NANO-U)")
    print("=" * 75)

    base_results_dir = os.path.join(PROJECT_ROOT, "results", "probing")
    os.makedirs(base_results_dir, exist_ok=True)

    fold_reports = {}
    stages = [
        "enc1b",
        "enc2b",
        "enc3b",
        "bottleneck_b",
        "dec1_conv_b",
        "dec2_conv_b",
        "dec3_conv_b",
    ]

    for fold_idx in range(4):
        model_path = os.path.join(PROJECT_ROOT, "models", f"TinyAgri_fold{fold_idx}", "nano_u.tflite")
        config_path = os.path.join(PROJECT_ROOT, "config", f"TinyAgri_fold{fold_idx}.yaml")
        fold_out_dir = os.path.join(base_results_dir, f"fold_{fold_idx}")
        os.makedirs(fold_out_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  EXECUTING PROBING ON FOLD {fold_idx}")
        print(f"  Model:  {model_path}")
        print(f"  Config: {config_path}")
        print(f"  Output: {fold_out_dir}")
        print(f"{'='*60}\n")

        res = run_probing_suite(
            model_path=model_path,
            config_path=config_path,
            epochs=20,
            batch_size=16,
            results_dir=fold_out_dir,
            export_to_thesis=False,
        )
        fold_reports[f"fold_{fold_idx}"] = res

    # Aggregate Layer-wise Metrics across 4 Folds
    print("\n" + "=" * 75)
    print("  AGGREGATING 4-FOLD PROBING METRICS ACROSS SCENES")
    print("=" * 75)

    agg_metrics = {s: {"miou": [], "iou_fg": [], "precision": [], "recall": [], "dice": []} for s in stages}
    agg_metrics["full_model"] = {"miou": [], "iou_fg": [], "precision": [], "recall": [], "dice": []}

    for fold_name, report in fold_reports.items():
        pm = report["probe_metrics"]
        for s in stages:
            agg_metrics[s]["miou"].append(pm[s]["miou"])
            agg_metrics[s]["iou_fg"].append(pm[s]["iou_foreground"])
            agg_metrics[s]["precision"].append(pm[s]["precision"])
            agg_metrics[s]["recall"].append(pm[s]["recall"])
            agg_metrics[s]["dice"].append(pm[s]["dice"])
        agg_metrics["full_model"]["miou"].append(pm["full_model"]["miou"])
        agg_metrics["full_model"]["iou_fg"].append(pm["full_model"]["iou_foreground"])
        agg_metrics["full_model"]["precision"].append(pm["full_model"]["precision"])
        agg_metrics["full_model"]["recall"].append(pm["full_model"]["recall"])
        agg_metrics["full_model"]["dice"].append(pm["full_model"]["dice"])

    summary_table = {}
    print(f"\n{'Stage':<20} | {'Shape':<12} | {'Params':<6} | {'Mean IoU_fg':<16} | {'Mean mIoU':<16} | {'Mean Recall':<12}")
    print("-" * 92)

    stage_param_counts = {
        "enc1b": 4 + 1,
        "enc2b": 8 + 1,
        "enc3b": 16 + 1,
        "bottleneck_b": 16 + 1,
        "dec1_conv_b": 16 + 1,
        "dec2_conv_b": 8 + 1,
        "dec3_conv_b": 4 + 1,
    }

    for s in stages:
        label = STAGE_LABELS.get(s, s)
        n_params = stage_param_counts[s]
        shape_str = f"{pm[s]['shape'][0]}x{pm[s]['shape'][1]}x{pm[s]['shape'][2]}"
        m_iou_fg = np.mean(agg_metrics[s]["iou_fg"])
        s_iou_fg = np.std(agg_metrics[s]["iou_fg"])
        m_miou = np.mean(agg_metrics[s]["miou"])
        s_miou = np.std(agg_metrics[s]["miou"])
        m_rec = np.mean(agg_metrics[s]["recall"])
        s_rec = np.std(agg_metrics[s]["recall"])

        summary_table[s] = {
            "label": label,
            "shape": shape_str,
            "params": n_params,
            "iou_fg_mean": float(m_iou_fg),
            "iou_fg_std": float(s_iou_fg),
            "miou_mean": float(m_miou),
            "miou_std": float(s_miou),
            "recall_mean": float(m_rec),
            "recall_std": float(s_rec),
        }
        print(f"{label:<20} | {shape_str:<12} | {n_params:<6} | {m_iou_fg:.4f} ± {s_iou_fg:.4f} | {m_miou:.4f} ± {s_miou:.4f} | {m_rec:.4f} ± {s_rec:.4f}")

    # Full Model
    fm_iou_fg = np.mean(agg_metrics["full_model"]["iou_fg"])
    fs_iou_fg = np.std(agg_metrics["full_model"]["iou_fg"])
    fm_miou = np.mean(agg_metrics["full_model"]["miou"])
    fs_miou = np.std(agg_metrics["full_model"]["miou"])
    fm_rec = np.mean(agg_metrics["full_model"]["recall"])
    fs_rec = np.std(agg_metrics["full_model"]["recall"])
    print("-" * 92)
    print(f"{'Full Nano-U Model':<20} | {'60x80x1':<12} | {'16337':<6} | {fm_iou_fg:.4f} ± {fs_iou_fg:.4f} | {fm_miou:.4f} ± {fs_miou:.4f} | {fm_rec:.4f} ± {fs_rec:.4f}")

    summary_table["full_model"] = {
        "label": "Full Deployed Nano-U",
        "shape": "60x80x1",
        "params": 16337,
        "iou_fg_mean": float(fm_iou_fg),
        "iou_fg_std": float(fs_iou_fg),
        "miou_mean": float(fm_miou),
        "miou_std": float(fs_miou),
        "recall_mean": float(fm_rec),
        "recall_std": float(fs_rec),
    }

    # Save Aggregate JSON
    summary_path = os.path.join(base_results_dir, "cv_4folds_probing_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_table, f, indent=2)
    print(f"\nSaved cross-validation summary JSON: {summary_path}")

    # Plot Publication-Grade 4-Fold Feature Emergence Curve
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x_indices = np.arange(len(stages))
    stage_names = [STAGE_LABELS.get(s, s) for s in stages]

    mean_iou_fg = [summary_table[s]["iou_fg_mean"] for s in stages]
    std_iou_fg = [summary_table[s]["iou_fg_std"] for s in stages]

    mean_miou = [summary_table[s]["miou_mean"] for s in stages]
    std_miou = [summary_table[s]["miou_std"] for s in stages]

    # Foreground IoU Curve
    ax.plot(x_indices, mean_iou_fg, marker="o", color="#1f77b4", lw=2.5, ms=8, label="Probe IoU$_{fg}$ (Mean)")
    ax.fill_between(
        x_indices,
        np.array(mean_iou_fg) - np.array(std_iou_fg),
        np.array(mean_iou_fg) + np.array(std_iou_fg),
        color="#1f77b4",
        alpha=0.15,
        label="±1 Std Dev (Across 4 Scenes)",
    )

    # mIoU Curve
    ax.plot(x_indices, mean_miou, marker="s", color="#2ca02c", lw=2.0, ms=7, ls="--", label="Probe mIoU (Mean)")

    # Baseline Horizontal Lines
    ax.axhline(fm_iou_fg, color="#d62728", ls=":", lw=2.0, label=f"Full Model IoU$_{{fg}}$ Baseline ({fm_iou_fg:.3f})")

    ax.set_xticks(x_indices)
    ax.set_xticklabels(stage_names, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Segmentation Score", fontsize=12)
    ax.set_xlabel("Architecture Hierarchy (Layer Depth)", fontsize=12)
    ax.set_title("Nano-U Feature Emergence Curve (4-Fold Cross-Validation)", fontsize=13, fontweight="bold")
    ax.set_ylim(0.2, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    out_curve = os.path.join(base_results_dir, "cv_feature_emergence_curve.png")
    plt.savefig(out_curve, dpi=200)
    plt.close(fig)
    print(f"Saved aggregated emergence plot: {out_curve}")

    # Export to Thesis Manuscript Images
    thesis_img_dir = os.path.abspath(
        os.path.join(PROJECT_ROOT, "..", "thesis-manuscript", "images", "probing")
    )
    os.makedirs(thesis_img_dir, exist_ok=True)
    shutil.copy(out_curve, os.path.join(thesis_img_dir, "cv_feature_emergence_curve.png"))
    print(f"Exported plot to thesis manuscript: {thesis_img_dir}/cv_feature_emergence_curve.png")


if __name__ == "__main__":
    main()
