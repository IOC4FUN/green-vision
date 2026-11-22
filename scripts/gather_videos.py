#!/usr/bin/env python3
"""
Autonomous YouTube video gatherer for Green Vision.
Finds Geoplanetical gameplay videos on YouTube, downloads them at low
resolution, extracts UI-relevant keyframes, deletes the source video to
save disk space, and reports a summary.

Usage (from project root):
    .venv\\Scripts\\python scripts/gather_videos.py [--max-videos N] [--fps F] [--min-duration S]

Defaults are tuned for storage efficiency:
  • 480p download  (~50-150 MB per 20-min video vs ~500 MB at 1080p)
  • 1 fps extraction (enough to catch every UI transition in a strategy game)
  • Source video deleted after successful frame extraction
  • JPEG quality 5 (good fidelity, ~60 % of q2 file size)
"""
import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Make src/ importable when run from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gather_videos")

# ── yt-dlp search + metadata ────────────────────────────────────────────────
def _find_ytdlp() -> str:
    """Locate yt-dlp binary in the project venv."""
    candidates = [
        PROJECT_ROOT / ".venv" / "Scripts" / "yt-dlp.exe",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    import shutil as _sh
    return _sh.which("yt-dlp") or "yt-dlp"


def search_youtube(query: str, max_results: int = 30) -> list[dict]:
    """Use yt-dlp's built-in YouTube search; returns list of video info dicts."""
    ytdlp = _find_ytdlp()
    cmd = [
        ytdlp,
        "--flat-playlist",
        "--print", "%(id)s|%(title)s|%(duration)s|%(view_count)s",
        f"ytsearch{max_results}:{query}",
    ]
    log.info("Searching YouTube: %s (top %d)", query, max_results)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        log.error("yt-dlp search failed: %s", proc.stderr[:400])
        return []
    results = []
    for line in proc.stdout.strip().splitlines():
        parts = line.split("|", 3)
        if len(parts) < 4:
            continue
        vid_id, title, dur_str, views_str = parts[0], parts[1], parts[2], parts[3]
        try:
            duration = float(dur_str) if dur_str and dur_str != "NA" else 0.0
            views = int(views_str) if views_str and views_str != "NA" else 0
        except ValueError:
            duration, views = 0.0, 0
        results.append({
            "id": vid_id,
            "title": title,
            "duration_s": duration,
            "views": views,
            "url": f"https://www.youtube.com/watch?v={vid_id}",
        })
    log.info("Found %d candidate videos", len(results))
    return results

# ── Filter ───────────────────────────────────────────────────────────────────
def filter_videos(videos: list[dict], min_duration: float = 300.0) -> list[dict]:
    """Keep only Geoplanetical gameplay clips long enough to be useful."""
    filtered = []
    for v in videos:
        t = v["title"].lower()
        # Must mention the game (or close variants)
        if not any(kw in t for kw in ("geoplanetical", "geo-planeticial", "polytale")):
            continue
        # Skip trailers / reviews that are too short
        if v["duration_s"] < min_duration:
            continue
        # Prefer longer gameplay over short clips
        filtered.append(v)
    # Sort by duration descending – longest first gives most frames per download
    filtered.sort(key=lambda x: x["duration_s"], reverse=True)
    log.info("After filtering: %d usable videos (min %.0f s)", len(filtered), min_duration)
    return filtered

# ── Download + extract one video ─────────────────────────────────────────────
def process_video(
    url: str,
    title: str,
    output_dir: Path,
    fps: float = 1.0,
    target_res: int = 480,
    jpeg_q: int = 5,
) -> dict:
    """
    Download → extract frames → delete source video.
    Returns a summary dict.
    """
    import re as _re
    slug = _re.sub(r'[^\w-]+', '_', title)[:60] or "untitled"
    clip_dir = output_dir / slug
    dl_dir = output_dir / "downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)
    # 1. Download at target resolution
    out_tmpl = str(dl_dir / f"{slug}.%(ext)s")
    ytdlp = _find_ytdlp()
    fmt = f"best[height<={target_res}][ext=mp4]/best[height<={target_res}]/worst[ext=mp4]/worst"
    cmd = [
        ytdlp,
        "--no-playlist",
        "-f", fmt,
        "-o", out_tmpl,
        url,
    ]
    log.info("  ↓ Downloading '%s' …", title[:70])
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        err = proc.stderr[-300:] if proc.stderr else "unknown error"
        log.warning("  ✗ Download failed: %s", err.replace('\n', ' '))
        return {"title": title, "ok": False, "error": "download_failed"}

    # Find the downloaded file (yt-dlp names it from our -o template)
    video_path = None
    for ext in (".mp4", ".webm", ".mkv", ".flv"):
        candidate = dl_dir / f"{slug}{ext}"
        if candidate.exists():
            video_path = candidate
            break
    if video_path is None or not video_path.exists():
        files = sorted(dl_dir.glob(f"{slug}.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        video_path = files[0] if files else None
    if video_path is None:
        log.warning("  ✗ Could not locate downloaded file")
        return {"title": title, "ok": False, "error": "file_not_found"}

    dl_size_mb = video_path.stat().st_size / 1e6
    dl_time = time.perf_counter() - t0
    log.info("  ↓ Done in %.0f s (%.1f MB)", dl_time, dl_size_mb)

    # 2. Extract frames via ffmpeg (low fps, compressed JPEGs)
    interval = 1.0 / fps
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"select='not(mod(t\\,{interval:.4f}))',setpts=N/{fps:g}/TB",
        "-vsync", "vfr",
        "-q:v", str(jpeg_q),
        str(clip_dir / "frame_%05d.jpg"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        log.warning("  ✗ Frame extraction failed: %s", proc.stderr[-200:])
        return {"title": title, "ok": False, "error": "extract_failed"}

    frame_files = sorted(clip_dir.glob("frame_*.jpg"))
    n_frames = len(frame_files)
    frames_mb = sum(f.stat().st_size for f in frame_files) / 1e6
    log.info("  ✓ Extracted %d frames (%.1f MB total)", n_frames, frames_mb)

    # 3. Detect action boundaries (simple pixel-diff heuristic)
    import numpy as np
    from PIL import Image
    prev_gray = None
    boundaries = 0
    boundary_indices = []
    for idx, fp in enumerate(frame_files):
        img = Image.open(fp).convert("L")
        small = np.array(img.resize((64, 64)), dtype=np.float32) / 255.0
        if prev_gray is not None:
            delta = float(np.mean(np.abs(small - prev_gray)))
            if delta >= 0.05:
                boundaries += 1
                boundary_indices.append(idx)
        prev_gray = small

    # 4. Write manifest
    manifest = {
        "source": url,
        "title": title,
        "duration_s": round(video_path.stat().st_size / 1e6 * fps, 1),  # approx
        "fps_requested": fps,
        "num_frames": n_frames,
        "action_boundaries": boundaries,
        "boundary_indices": boundary_indices,
        "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "download_mb": round(dl_size_mb, 1),
        "frames_mb": round(frames_mb, 1),
        "jpeg_quality": jpeg_q,
        "frames": [
            {"t": i * interval, "idx": i, "path": str(fp.resolve()),
             "w": Image.open(fp).size[0], "h": Image.open(fp).size[1],
             "boundary": i in set(boundary_indices)}
            for i, fp in enumerate(frame_files)
        ],
    }
    (clip_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # 5. Delete source video to reclaim space
    if not keep_source:
        video_path.unlink(missing_ok=True)
        log.info("  🗑 Deleted source video (%.1f MB reclaimed)", dl_size_mb)

    return {
        "title": title, "ok": True, "url": url,
        "frames": n_frames, "boundaries": boundaries,
        "target_res": target_res,
        "download_mb": round(dl_size_mb, 1), "frames_mb": round(frames_mb, 1),
        "time_s": round(time.perf_counter() - t0, 1),
    }

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Gather Geoplanetical gameplay videos from YouTube")
    ap.add_argument("--max-videos", type=int, default=0, help="Max videos to download (0 = unlimited, default)")
    ap.add_argument("--fps", type=float, default=1.0, help="Frame extraction rate (default 1 fps)")
    ap.add_argument("--res", type=int, default=480, help="Target resolution height (default 480; use 720 or 1080 for higher quality)")
    ap.add_argument("--min-duration", type=float, default=300.0, help="Min video duration in seconds (default 300)")
    ap.add_argument("--output-dir", type=str, default=None, help="Output dir (default data/videos)")
    ap.add_argument("--keep-source", action="store_true", help="Keep downloaded .mp4 files after extraction")
    ap.add_argument("--search-queries", nargs="+",
                    default=["Geoplanetical gameplay walkthrough", "Geoplanetical let's play full game",
                             "Geoplanetical tutorial beginner guide"],
                    help="YouTube search queries")
    args = ap.parse_args()

    global keep_source
    keep_source = args.keep_source

    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "data" / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Disk space guard
    usage = shutil.disk_usage(str(out_dir))
    free_gb = usage.free / 1024**3
    log.info("Disk: %.1f GB free of %.1f GB total", free_gb, usage.total / 1024**3)
    MIN_FREE_GB = 20.0
    if free_gb < MIN_FREE_GB:
        log.error("Less than %.0f GB free – aborting to avoid filling the drive.", MIN_FREE_GB)
        sys.exit(1)

    # Search all queries and merge results (dedup by video ID)
    all_videos: dict[str, dict] = {}
    for q in args.search_queries:
        results = search_youtube(q, max_results=20)
        for v in results:
            if v["id"] not in all_videos:
                all_videos[v["id"]] = v
    candidates = list(all_videos.values())
    filtered = filter_videos(candidates, min_duration=args.min_duration)

    if not filtered:
        log.warning("No suitable Geoplanetical videos found. The game is very new (EA Oct 28 2026).")
        log.info("Try again in a few days as more content is uploaded.")
        return

    # max_videos=0 means unlimited
    if args.max_videos > 0:
        selected = filtered[:args.max_videos]
    else:
        selected = filtered
    log.info("\n═══ Processing %d videos (%s) ═══\n", len(selected), "unlimited" if args.max_videos == 0 else f"capped at {args.max_videos}")

    summaries = []
    for i, vid in enumerate(selected, 1):
        log.info("[%d/%d] %s (%.0f s, %d views)", i, len(selected), vid["title"][:60], vid["duration_s"], vid["views"])
        result = process_video(
            url=vid["url"],
            title=vid["title"],
            output_dir=out_dir,
            fps=args.fps,
            target_res=args.res,
            jpeg_q=5,
        )
        summaries.append(result)
        # Small pause between downloads to be polite
        time.sleep(2)

    # ── Summary report ────────────────────────────────────────────────
    ok = [s for s in summaries if s.get("ok")]
    fail = [s for s in summaries if not s.get("ok")]
    total_frames = sum(s.get("frames", 0) for s in ok)
    total_boundaries = sum(s.get("boundaries", 0) for s in ok)
    total_dl_mb = sum(s.get("download_mb", 0) for s in ok)
    total_frames_mb = sum(s.get("frames_mb", 0) for s in ok)

    print(f"\n{'═'*60}")
    print(f"  GATHER COMPLETE")
    print(f"{'═'*60}")
    print(f"  Videos processed : {len(ok)} ok / {len(fail)} failed")
    print(f"  Total frames     : {total_frames:,}")
    print(f"  Action boundaries: {total_boundaries:,}")
    print(f"  Downloaded (del): {total_dl_mb:.0f} MB")
    print(f"  Frames on disk   : {total_frames_mb:.1f} MB")
    print(f"  Net savings      : {max(0, total_dl_mb - total_frames_mb):.0f} MB per video cycle")
    if fail:
        print(f"\n  Failed videos:")
        for f in fail:
            print(f"    ✗ {f['title'][:50]} – {f.get('error','?')}")
    print(f"\n  Output dir: {out_dir}")
    print(f"{'═'*60}\n")

if __name__ == "__main__":
    main()
