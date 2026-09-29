"""
End-to-end Representation Probing Evaluator for Nano-U.

Trains linear diagnostic probes across model depths, records semantic metrics,
computes Centered Kernel Alignment (CKA), and generates publication-grade
visualization panels for the thesis manuscript.
"""

import os
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import Dict, Any, Optional, List
from .extractor import LayerActivationExtractor
from .probes import PixelLinearProbe, compute_linear_cka
from src.data import make_dataset, sorted_by_frame
from src.utils.config import load_config
from src.evaluate import compute_segmentation_metrics, run_inference


STAGE_LABELS = {
    "enc1b": "Enc 1 (60x80x4)",
    "enc2b": "Enc 2 (30x40x8)",
    "enc3b": "Enc 3 (15x20x16)",
    "bottleneck_b": "Bottleneck (5x10x16)",
    "dec1_conv_b": "Dec 1 (15x20x16)",
    "dec2_conv_b": "Dec 2 (30x40x8)",
    "dec3_conv_b": "Dec 3 (60x80x4)",
}


def load_dataset_split(config: Dict[str, Any], split: str, batch_size: int = 16):
    """Load images and masks for a specified split."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(root_dir, p)

    split_cfg = config["data"]["paths"]["processed"][split]
    img_dir = resolve(split_cfg["img"])
    mask_dir = resolve(split_cfg["mask"])

    imgs = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir)) if f.endswith(".png")]
    masks = [os.path.join(mask_dir, f) for f in sorted(os.listdir(mask_dir)) if f.endswith(".png")]

    imgs = sorted_by_frame(imgs)
    masks = sorted_by_frame(masks)

    mean = config["data"]["normalization"]["mean"]
    std = config["data"]["normalization"]["std"]
    input_shape = config.get("data", {}).get("input_shape", [60, 80, 3])

    ds = make_dataset(
        imgs,
        masks,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        target_size=(input_shape[0], input_shape[1]),
        mean=mean,
        std=std,
    )
    return ds, imgs, masks


def run_probing_suite(
    model_path: str,
    config_path: str,
    epochs: int = 20,
    batch_size: int = 16,
    results_dir: Optional[str] = None,
    export_to_thesis: bool = True,
) -> Dict[str, Any]:
    """Execute complete representation probing pipeline."""
    config = load_config(config_path)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    if results_dir is None:
        results_dir = os.path.join(root_dir, "results", "probing")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("  STAGE 2: REPRESENTATION PROBING (THE BALLAN CROSSOVER)")
    print(f"  Model:   {model_path}")
    print(f"  Config:  {config_path}")
    print(f"  Epochs:  {epochs} per layer probe")
    print("=" * 70)

    # 1. Initialize Activation Extractor
    extractor = LayerActivationExtractor(model_path)
    stages = list(extractor.tensor_registry.keys())

    # 2. Extract activations for train and test splits
    print("\n[Step 1/4] Extracting activations across splits...")
    train_ds, train_imgs, _ = load_dataset_split(config, "train", batch_size=batch_size)
    test_ds, test_imgs, _ = load_dataset_split(config, "test", batch_size=batch_size)

    print(f"  Extracting train split ({len(train_imgs)} images)...")
    X_train, Y_train, imgs_train = extractor.extract_dataset(train_ds, target_stages=stages)
    print(f"  Extracting test split ({len(test_imgs)} images)...")
    X_test, Y_test, imgs_test = extractor.extract_dataset(test_ds, target_stages=stages)

    # 3. Train diagnostic linear probes
    print(f"\n[Step 2/4] Training linear diagnostic probes ({epochs} epochs each)...")
    probe_metrics = {}
    test_predictions = {}
    trained_probes = {}

    for stage in stages:
        stage_title = STAGE_LABELS.get(stage, stage)
        feat_train = X_train[stage]
        feat_test = X_test[stage]
        input_shape = feat_train.shape[1:]

        print(f"\n--- Training Probe for Stage: {stage_title} (shape: {input_shape}) ---")
        probe = PixelLinearProbe(stage, input_shape=input_shape, target_shape=(60, 80))
        history = probe.train(
            feat_train,
            Y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )
        final_loss = history["loss"][-1]
        print(f"  Probe converged: initial loss={history['loss'][0]:.4f} -> final loss={final_loss:.4f}")

        # Evaluate on test set
        probs = probe.predict_probs(feat_test, batch_size=batch_size)
        metrics = compute_segmentation_metrics(probs, Y_test, operating_threshold=0.5)

        probe_metrics[stage] = {
            "label": stage_title,
            "shape": list(input_shape),
            "final_train_loss": float(final_loss),
            "iou_foreground": float(metrics.get("iou_foreground", metrics.get("dice", 0.0))),
            "iou_background": float(metrics.get("iou_background", 0.0)),
            "miou": float(metrics["miou"]),
            "dice": float(metrics["dice"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
        }
        test_predictions[stage] = probs
        trained_probes[stage] = probe
        print(f"  Test Score: IoU_fg={probe_metrics[stage]['iou_foreground']:.4f} | mIoU={metrics['miou']:.4f} | Dice={metrics['dice']:.4f} | Prec={metrics['precision']:.4f} | Rec={metrics['recall']:.4f}")

    # Baseline Full Model Score
    print("\n--- Baseline Full Model Inference on Test Set ---")
    base_probs, base_masks, _ = run_inference(model_path, test_ds)
    base_metrics = compute_segmentation_metrics(base_probs, base_masks, operating_threshold=0.5)
    probe_metrics["full_model"] = {
        "label": "Full Deployed Nano-U",
        "iou_foreground": float(base_metrics.get("iou_foreground", base_metrics.get("dice", 0.0))),
        "iou_background": float(base_metrics.get("iou_background", 0.0)),
        "miou": float(base_metrics["miou"]),
        "dice": float(base_metrics["dice"]),
        "precision": float(base_metrics["precision"]),
        "recall": float(base_metrics["recall"]),
        "f1": float(base_metrics["f1"]),
    }
    print(f"  Full Model: IoU_fg={probe_metrics['full_model']['iou_foreground']:.4f} | mIoU={base_metrics['miou']:.4f} | Dice={base_metrics['dice']:.4f}")

    # 4. Compute Centered Kernel Alignment (CKA) across layers
    print("\n[Step 3/4] Computing Centered Kernel Alignment (CKA) matrix...")
    n_stages = len(stages)
    cka_matrix = np.zeros((n_stages, n_stages), dtype=np.float32)

    for i, s1 in enumerate(stages):
        for j, s2 in enumerate(stages):
            if j < i:
                cka_matrix[i, j] = cka_matrix[j, i]
            elif i == j:
                cka_matrix[i, j] = 1.0
            else:
                cka_matrix[i, j] = compute_linear_cka(X_test[s1], X_test[s2])

    print("  CKA matrix computed successfully.")

    # 5. Generate Publication-Quality Visualizations
    print("\n[Step 4/4] Generating publication plots...")
    
    # Plot 1: Feature Emergence Curve (mIoU vs Depth)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x_indices = np.arange(n_stages)
    stage_names = [STAGE_LABELS.get(s, s) for s in stages]
    mious = [probe_metrics[s]["miou"] for s in stages]
    dices = [probe_metrics[s]["dice"] for s in stages]

    ax1.plot(x_indices, mious, marker="o", color="#1f77b4", lw=2.5, ms=8, label="Probe mIoU")
    ax1.plot(x_indices, dices, marker="s", color="#2ca02c", lw=2, ms=7, ls="--", label="Probe Dice")
    ax1.axhline(base_metrics["miou"], color="#d62728", ls=":", lw=2, label=f"Full Model Baseline (mIoU={base_metrics['miou']:.3f})")
    
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(stage_names, rotation=25, ha="right", fontsize=10)
    ax1.set_ylabel("Metric Score", fontsize=12)
    ax1.set_xlabel("Architecture Hierarchy (Depth)", fontsize=12)
    ax1.set_title("Nano-U Feature Emergence Curve: Linear Decodability across Depths", fontsize=13, fontweight="bold")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    curve_path = os.path.join(results_dir, "feature_emergence_curve.png")
    plt.savefig(curve_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {curve_path}")

    # Plot 2: CKA Similarity Matrix Heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cka_matrix, cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(n_stages))
    ax.set_yticks(np.arange(n_stages))
    ax.set_xticklabels(stage_names, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(stage_names, fontsize=9)
    ax.set_title("Representational Similarity Matrix (Linear CKA)", fontsize=13, fontweight="bold")
    
    # Annotate matrix cells
    for i in range(n_stages):
        for j in range(n_stages):
            val = cka_matrix[i, j]
            text_color = "white" if val < 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    cka_path = os.path.join(results_dir, "cka_similarity_matrix.png")
    plt.savefig(cka_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {cka_path}")

    # Plot 3: Qualitative Progression (Sample Visualizations)
    sample_idx = min(5, len(Y_test) - 1)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    mean_arr = np.array(config["data"]["normalization"]["mean"], dtype=np.float32)
    std_arr = np.array(config["data"]["normalization"]["std"], dtype=np.float32)
    img_sample = np.clip(imgs_test[sample_idx] * std_arr + mean_arr, 0.0, 1.0)
    gt_sample = Y_test[sample_idx, :, :, 0]

    # Row 1: Input, GT, Enc1, Bottleneck
    axes[0, 0].imshow(img_sample); axes[0, 0].set_title("Input Frame"); axes[0, 0].axis("off")
    axes[0, 1].imshow(gt_sample, cmap="gray"); axes[0, 1].set_title("Ground Truth Path"); axes[0, 1].axis("off")
    axes[0, 2].imshow(test_predictions["enc1b"][sample_idx, :, :, 0], cmap="viridis", vmin=0, vmax=1)
    axes[0, 2].set_title(f"Probe: {STAGE_LABELS['enc1b']}\n(mIoU={probe_metrics['enc1b']['miou']:.3f})"); axes[0, 2].axis("off")
    axes[0, 3].imshow(test_predictions["bottleneck_b"][sample_idx, :, :, 0], cmap="viridis", vmin=0, vmax=1)
    axes[0, 3].set_title(f"Probe: {STAGE_LABELS['bottleneck_b']}\n(mIoU={probe_metrics['bottleneck_b']['miou']:.3f})"); axes[0, 3].axis("off")

    # Row 2: Dec1, Dec2, Dec3, Full Model
    axes[1, 0].imshow(test_predictions["dec1_conv_b"][sample_idx, :, :, 0], cmap="viridis", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Probe: {STAGE_LABELS['dec1_conv_b']}\n(mIoU={probe_metrics['dec1_conv_b']['miou']:.3f})"); axes[1, 0].axis("off")
    axes[1, 1].imshow(test_predictions["dec2_conv_b"][sample_idx, :, :, 0], cmap="viridis", vmin=0, vmax=1)
    axes[1, 1].set_title(f"Probe: {STAGE_LABELS['dec2_conv_b']}\n(mIoU={probe_metrics['dec2_conv_b']['miou']:.3f})"); axes[1, 1].axis("off")
    axes[1, 2].imshow(test_predictions["dec3_conv_b"][sample_idx, :, :, 0], cmap="viridis", vmin=0, vmax=1)
    axes[1, 2].set_title(f"Probe: {STAGE_LABELS['dec3_conv_b']}\n(mIoU={probe_metrics['dec3_conv_b']['miou']:.3f})"); axes[1, 2].axis("off")
    axes[1, 3].imshow(base_probs[sample_idx, :, :, 0], cmap="viridis", vmin=0, vmax=1)
    axes[1, 3].set_title(f"Full Deployed Model\n(mIoU={base_metrics['miou']:.3f})"); axes[1, 3].axis("off")

    plt.suptitle("Layer-Wise Semantic Perception Emergence in Nano-U", fontsize=14, fontweight="bold")
    plt.tight_layout()
    samples_path = os.path.join(results_dir, "sample_probe_predictions.png")
    plt.savefig(samples_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {samples_path}")

    # Save JSON summary
    summary_data = {
        "model_path": model_path,
        "config_path": config_path,
        "epochs": epochs,
        "stages": stages,
        "probe_metrics": probe_metrics,
        "cka_matrix": cka_matrix.tolist(),
    }
    json_path = os.path.join(results_dir, "probing_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  Saved JSON report: {json_path}")

    # 6. Export to Thesis Manuscript Images
    if export_to_thesis:
        thesis_img_dir = os.path.abspath(
            os.path.join(root_dir, "..", "thesis-manuscript", "images", "probing")
        )
        os.makedirs(thesis_img_dir, exist_ok=True)
        shutil.copy(curve_path, os.path.join(thesis_img_dir, "feature_emergence_curve.png"))
        shutil.copy(cka_path, os.path.join(thesis_img_dir, "cka_similarity_matrix.png"))
        shutil.copy(samples_path, os.path.join(thesis_img_dir, "sample_probe_predictions.png"))
        print(f"  Exported plots to thesis manuscript: {thesis_img_dir}")

    print("\n" + "=" * 70)
    print("  PROBING EXPERIMENT COMPLETE")
    print("=" * 70)
    return summary_data
