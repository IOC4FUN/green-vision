#!/usr/bin/env python3
"""
Generate manifest.json for each clip directory under data/videos/.
The pretraining pipeline requires these to build datasets.
Run once after collection; idempotent (overwrites existing).
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(r"E:\Green Vision")
DATA_DIR = PROJECT_ROOT / "data" / "videos"

def main():
    clip_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    total_frames = 0
    generated = 0

    for cd in clip_dirs:
        frames = sorted(cd.glob("frame_*.jpg"))
        if not frames:
            print(f"SKIP {cd.name} — no frames")
            continue

        n = len(frames)
        # Detect resolution from first frame
        res_str = "unknown"
        try:
            from PIL import Image
            img = Image.open(frames[0])
            w, h = img.size
            res_str = f"{w}x{h}"
        except Exception:
            pass

        manifest = {
            "clip": cd.name,
            "num_frames": n,
            "resolution": res_str,
            "frames": [{"idx": i + 1, "boundary": False} for i in range(n)],
        }

        out = cd / "manifest.json"
        out.write_text(json.dumps(manifest, indent=2))
        total_frames += n
        generated += 1
        print(f"OK   {cd.name}: {n} frames ({res_str})")

    print(f"\nDone: {generated} manifests, {total_frames} total frames.")

if __name__ == "__main__":
    main()
