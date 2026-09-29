#!/usr/bin/env python3
"""
Generate 4-fold Leave-One-Scene-Out Cross-Validation manifests and symlink datasets.
Zero-leakage partition:
- In each fold, 1 full scene is completely held out for testing.
- The remaining 3 scenes provide training and validation frames.
- In each training scene:
    * The last ~15% contiguous frames form the validation set.
    * A 20-frame temporal buffer gap preceding the validation set is excluded from training.
    * The remaining earlier frames form the training set.

Creates:
  1. JSON manifests in data/manifests/cv_fold_{0..3}/
  2. Standard dataset folder structure with symlinks in data/cv_splits/fold_{0..3}/:
     - train/img, train/mask
     - val/img, val/mask
     - test/img, test/mask
  3. Ready-to-run YAML configs in config/TinyAgri_fold{0..3}.yaml
"""

import os
import sys
import json
import re
import yaml
from pathlib import Path
from typing import List, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "TinyAgri_full"
MANIFEST_ROOT = REPO_ROOT / "data" / "manifests"
SPLITS_ROOT = REPO_ROOT / "data" / "cv_splits"
CONFIG_ROOT = REPO_ROOT / "config"

SCENES = [
    "Crops_scene1",
    "Crops_scene2",
    "Tomatoes_scene1",
    "Tomatoes_scene2",
]


def get_frame_num(path: Path) -> int:
    m = re.search(r"frame(\d+)\.png", path.name)
    return int(m.group(1)) if m else 0


def load_scene_pairs(scene_name: str) -> List[Dict[str, str]]:
    scene_dir = DATA_ROOT / scene_name
    img_files = sorted(list((scene_dir / "img").glob("*.png")), key=get_frame_num)
    mask_files = sorted(list((scene_dir / "mask").glob("*.png")), key=get_frame_num)

    assert len(img_files) == len(mask_files), f"Count mismatch in {scene_name}"
    pairs = []
    for img_p, mask_p in zip(img_files, mask_files):
        assert img_p.name == mask_p.name, f"Name mismatch: {img_p.name} != {mask_p.name}"
        pairs.append({
            "img": str(img_p.resolve()),
            "mask": str(mask_p.resolve()),
            "filename": img_p.name,
            "scene": scene_name,
            "frame": get_frame_num(img_p),
        })
    return pairs


def split_training_scene(pairs: List[Dict[str, str]], val_ratio: float = 0.15, buffer_gap: int = 20):
    n = len(pairs)
    n_val = max(1, round(n * val_ratio))
    val_start = n - n_val
    train_end = max(1, val_start - buffer_gap)

    train_pairs = pairs[:train_end]
    buffer_pairs = pairs[train_end:val_start]
    val_pairs = pairs[val_start:]

    return train_pairs, val_pairs, buffer_pairs


def make_symlink_dir(pairs: List[Dict[str, str]], img_target_dir: Path, mask_target_dir: Path):
    img_target_dir.mkdir(parents=True, exist_ok=True)
    mask_target_dir.mkdir(parents=True, exist_ok=True)
    for p in pairs:
        src_img = Path(p["img"])
        src_mask = Path(p["mask"])
        dst_img = img_target_dir / p["filename"]
        dst_mask = mask_target_dir / p["filename"]
        if not dst_img.exists():
            dst_img.symlink_to(src_img)
        if not dst_mask.exists():
            dst_mask.symlink_to(src_mask)


def create_fold_yaml(fold_idx: int, test_scene: str):
    base_yaml_path = CONFIG_ROOT / "TinyAgri_config.yaml"
    if not base_yaml_path.exists():
        base_yaml_path = CONFIG_ROOT / "config.yaml"
    with open(base_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    fold_split_dir = f"data/cv_splits/fold_{fold_idx}"
    cfg["data"]["paths"]["processed"]["root"] = fold_split_dir
    cfg["data"]["paths"]["processed"]["train"]["img"] = f"{fold_split_dir}/train/img"
    cfg["data"]["paths"]["processed"]["train"]["mask"] = f"{fold_split_dir}/train/mask"
    cfg["data"]["paths"]["processed"]["val"]["img"] = f"{fold_split_dir}/val/img"
    cfg["data"]["paths"]["processed"]["val"]["mask"] = f"{fold_split_dir}/val/mask"
    cfg["data"]["paths"]["processed"]["test"]["img"] = f"{fold_split_dir}/test/img"
    cfg["data"]["paths"]["processed"]["test"]["mask"] = f"{fold_split_dir}/test/mask"

    cfg["data"]["paths"]["models_dir"] = f"models/TinyAgri_fold{fold_idx}"
    cfg["data"]["paths"]["results_dir"] = f"results/TinyAgri_fold{fold_idx}"
    cfg["training"]["bu_net"]["models_dir"] = f"models/TinyAgri_fold{fold_idx}"
    cfg["training"]["nano_u"]["models_dir"] = f"models/TinyAgri_fold{fold_idx}"
    cfg["training"]["nano_u"]["distillation"]["teacher_weights"] = f"models/TinyAgri_fold{fold_idx}/bu_net.h5"

    # Cross-scene CV adaptation: the paper's patience=10 was tuned for
    # single-scene temporal splits where val_loss drops immediately. In
    # leave-one-scene-out CV, BU-Net needs ~15 epochs for BatchNorm stats
    # and deep representations to align across 3 heterogeneous scenes
    # before cross-scene val_loss starts improving. patience=30 and an
    # earlier LR decay (plateau_patience=5) let the optimizer reach the
    # breakout point that Fold 3 demonstrated at epoch 12-15.
    cfg["training"]["bu_net"]["patience"] = 30
    cfg["training"]["bu_net"]["lr_plateau_patience"] = 5

    out_yaml_path = CONFIG_ROOT / f"TinyAgri_fold{fold_idx}.yaml"
    with open(out_yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  Created config: {out_yaml_path.relative_to(REPO_ROOT)}")


def main():
    print("=" * 70)
    print("PREPARING 4-FOLD LEAVE-ONE-SCENE-OUT (LOSO) CV SPLITS & CONFIGS")
    print("=" * 70)

    scene_data = {s: load_scene_pairs(s) for s in SCENES}
    for s, p in scene_data.items():
        print(f"Loaded {s}: {len(p)} frames (frame range {p[0]['frame']} .. {p[-1]['frame']})")

    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    SPLITS_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {}

    for fold_idx, test_scene in enumerate(SCENES):
        manifest_dir = MANIFEST_ROOT / f"cv_fold_{fold_idx}"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        fold_split_dir = SPLITS_ROOT / f"fold_{fold_idx}"
        fold_split_dir.mkdir(parents=True, exist_ok=True)

        train_scenes = [s for s in SCENES if s != test_scene]

        train_pairs_all = []
        val_pairs_all = []
        buffer_counts = {}

        for ts in train_scenes:
            tr, va, bu = split_training_scene(scene_data[ts], val_ratio=0.15, buffer_gap=20)
            train_pairs_all.extend(tr)
            val_pairs_all.extend(va)
            buffer_counts[ts] = len(bu)

        test_pairs_all = list(scene_data[test_scene])

        # 1. Write manifest files
        with open(manifest_dir / "train.json", "w") as f:
            json.dump(train_pairs_all, f, indent=2)
        with open(manifest_dir / "val.json", "w") as f:
            json.dump(val_pairs_all, f, indent=2)
        with open(manifest_dir / "test.json", "w") as f:
            json.dump(test_pairs_all, f, indent=2)

        # 2. Build symlink directories for seamless compatibility
        make_symlink_dir(train_pairs_all, fold_split_dir / "train" / "img", fold_split_dir / "train" / "mask")
        make_symlink_dir(val_pairs_all, fold_split_dir / "val" / "img", fold_split_dir / "val" / "mask")
        make_symlink_dir(test_pairs_all, fold_split_dir / "test" / "img", fold_split_dir / "test" / "mask")

        # 3. Create fold-specific YAML config
        create_fold_yaml(fold_idx, test_scene)

        summary[f"fold_{fold_idx}"] = {
            "test_scene": test_scene,
            "train_scenes": train_scenes,
            "n_train": len(train_pairs_all),
            "n_val": len(val_pairs_all),
            "n_test": len(test_pairs_all),
            "n_buffer_dropped": sum(buffer_counts.values()),
            "config_file": f"config/TinyAgri_fold{fold_idx}.yaml",
        }

        print(f"\n--- Fold {fold_idx} ---")
        print(f"  Test Scene (Held-Out) : {test_scene:<18} ({len(test_pairs_all)} frames)")
        print(f"  Training Scenes       : {', '.join(train_scenes)}")
        print(f"  Train Frames          : {len(train_pairs_all):4d}")
        print(f"  Validation Frames     : {len(val_pairs_all):4d}")
        print(f"  Buffer Gap Dropped    : {sum(buffer_counts.values()):4d} (20 frames/scene)")
        print(f"  Symlink Dataset Dir   : {fold_split_dir.relative_to(REPO_ROOT)}")

    with open(MANIFEST_ROOT / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("4-FOLD CV SPLITS & CONFIGS READY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
