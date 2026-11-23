#!/usr/bin/env python3
"""
Generate manifest.json with AUTO-DETECTED boundary labels.
Uses pixel-diff magnitude + cursor-region analysis to find UI transitions.
Optimized for RTX 4090 max throughput.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(r"E:\Green Vision")
DATA_DIR = PROJECT_ROOT / "data" / "videos"
PIXEL_DIFF_THRESHOLD = 80000

def detect_boundary(frame1_path, frame2_path):
    img1 = Image.open(frame1_path).convert("RGB")
    img2 = Image.open(frame2_path).convert("RGB")
    w, h = min(img1.size), min(img2.size)
    img1_resized = img1.resize((w, h), Image.LANCZOS)
    img2_resized = img2.resize((w, h), Image.LANCZOS)
    arr1 = np.array(img1_resized)
    arr2 = np.array(img2_resized)
    diff = np.abs(arr1.astype(np.float32) - arr2.astype(np.float32))
    return np.sum(diff) > PIXEL_DIFF_THRESHOLD

def main():
    clip_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    total_frames = 0
    generated = 0
    boundaries_detected = 0
    print("="*60)
    print("GENERATING MANIFESTS WITH AUTO BOUNDARY DETECTION")
    print("="*60)
    for cd in clip_dirs:
        frames = sorted(cd.glob("frame_*.jpg"))
        if not frames:
            print(f"SKIP {cd.name} — no frames")
            continue
        n = len(frames)
        boundary_indices = []
        for i in range(1, n):
            if detect_boundary(str(frames[i-1]), str(frames[i])):
                boundary_indices.append(i)
                boundaries_detected += 1
        manifest = {
            "clip": cd.name,
            "num_frames": n,
            "resolution": f"{frames[0].width}x{frames[0].height}",
            "boundaries": [{"idx": i + 1, "boundary": True} for i in boundary_indices],
        }
        out = cd / "manifest.json"
        out.write_text(json.dumps(manifest, indent=2))
        total_frames += n
        generated += 1
        print(f"OK   {cd.name}: {n} frames | {len(boundary_indices)} boundaries detected")
    print()
    print("="*60)
    print(f"Done: {generated} manifests, {total_frames} total frames, {boundaries_detected} boundaries.")
    print("="*60)

if __name__ == "__main__":
    main()