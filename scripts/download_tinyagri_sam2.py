#!/usr/bin/env python3
"""
Parallel downloader for full SAM2-annotated TinyAgri dataset from Hugging Face.
Uses ThreadPoolExecutor(max_workers=16) for high-throughput downloading.
Repo: federico-pizz/TinyAgri

Scenes:
  - Crops/scene1:    274 frames (prefix: d6_s1_frame<N>.png)
  - Crops/scene2:    380 frames (prefix: d6_s2_frame<N>.png)
  - Tomatoes/scene1: 692 frames (prefix: d5_s1_frame<N>.png)
  - Tomatoes/scene2: 464 frames (prefix: d5_s2_frame<N>.png)
Total: 1,810 frames (images + masks)
"""

import os
import sys
import re
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "federico-pizz/TinyAgri"
DEST_ROOT = Path("/home/tharen/MASTERS/thesis/Nano-U/data/TinyAgri_full")

SCENES = {
    "Crops/scene1": ("Crops_scene1", "d6_s1"),
    "Crops/scene2": ("Crops_scene2", "d6_s2"),
    "Tomatoes/scene1": ("Tomatoes_scene1", "d5_s1"),
    "Tomatoes/scene2": ("Tomatoes_scene2", "d5_s2"),
}

EXPECTED_COUNTS = {
    "Crops/scene1": 274,
    "Crops/scene2": 380,
    "Tomatoes/scene1": 692,
    "Tomatoes/scene2": 464,
}


def download_single_pair(item):
    repo_id, mf, raw_hf_path, dest_mask, dest_img = item

    # Download mask if needed
    if not dest_mask.exists():
        cached_mask = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=mf,
        )
        shutil.copyfile(cached_mask, dest_mask)

    # Download image if needed
    if not dest_img.exists():
        cached_img = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=raw_hf_path,
        )
        shutil.copyfile(cached_img, dest_img)

    return dest_mask.name


def main():
    print("=" * 65)
    print("PARALLEL DOWNLOAD: FULL SAM2 TINYAGRI DATASET")
    print(f"Target: {DEST_ROOT} (Concurrency: 16 threads)")
    print("=" * 65)

    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    all_files = api.list_repo_files(REPO_ID, repo_type="dataset")

    for hf_scene, (local_scene, prefix) in SCENES.items():
        scene_dir = DEST_ROOT / local_scene
        img_dir = scene_dir / "img"
        mask_dir = scene_dir / "mask"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        # Check existing files
        existing_imgs = len(list(img_dir.glob("*.png")))
        existing_masks = len(list(mask_dir.glob("*.png")))
        expected = EXPECTED_COUNTS[hf_scene]

        if existing_imgs == expected and existing_masks == expected:
            print(f"\n{local_scene}: Already complete ({existing_imgs} pairs verified). Skipping.")
            continue

        raw_files = [f for f in all_files if f.startswith(f"{hf_scene}/")]
        raw_frame_map = {}
        for rf in raw_files:
            m = re.search(r"frame(\d+)\.png", rf)
            if m:
                raw_frame_map[int(m.group(1))] = rf

        sam2_files = [f for f in all_files if f.startswith(f"SAM2-masks/{hf_scene}/")]
        print(f"\nProcessing {hf_scene} -> {local_scene} ({len(sam2_files)} frames, expected {expected})...")

        download_tasks = []
        for mf in sam2_files:
            m = re.search(r"mask(\d+)\.png", mf)
            if not m:
                continue
            frame_num = int(m.group(1))
            if frame_num not in raw_frame_map:
                continue

            raw_hf_path = raw_frame_map[frame_num]
            canonical_name = f"{prefix}_frame{frame_num}.png"
            dest_mask = mask_dir / canonical_name
            dest_img = img_dir / canonical_name

            if not (dest_mask.exists() and dest_img.exists()):
                download_tasks.append((REPO_ID, mf, raw_hf_path, dest_mask, dest_img))

        print(f"  {len(download_tasks)} frame pairs need downloading. Launching parallel pool...")
        completed = 0
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(download_single_pair, t) for t in download_tasks]
            for fut in as_completed(futures):
                fut.result()
                completed += 1
                if completed % 50 == 0 or completed == len(download_tasks):
                    print(f"    Progress: {completed}/{len(download_tasks)} pairs ({completed/len(download_tasks)*100:.1f}%)")

        final_imgs = len(list(img_dir.glob("*.png")))
        final_masks = len(list(mask_dir.glob("*.png")))
        print(f"  Verified {local_scene}: {final_imgs} images, {final_masks} masks.")
        assert final_imgs == expected, f"Mismatch in {local_scene} images: {final_imgs} != {expected}"
        assert final_masks == expected, f"Mismatch in {local_scene} masks: {final_masks} != {expected}"

    print("\n" + "=" * 65)
    print("DATASET VERIFICATION SUMMARY")
    print("=" * 65)
    total_imgs = 0
    total_masks = 0
    for hf_scene, (local_scene, _) in SCENES.items():
        scene_dir = DEST_ROOT / local_scene
        n_imgs = len(list((scene_dir / "img").glob("*.png")))
        n_masks = len(list((scene_dir / "mask").glob("*.png")))
        total_imgs += n_imgs
        total_masks += n_masks
        print(f"  {local_scene:<18}: {n_imgs:4d} images | {n_masks:4d} masks")
    print("-" * 65)
    print(f"  {'TOTAL':<18}: {total_imgs:4d} images | {total_masks:4d} masks")
    print("=" * 65)
    print("All 1,810 SAM2 frames successfully downloaded and verified!")


if __name__ == "__main__":
    main()
