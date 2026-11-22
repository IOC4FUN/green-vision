#!/usr/bin/env python3
"""
Comprehensive video collection for Green Vision.
Downloads ALL useful Geoplanetical gameplay videos from YouTube,
extracts frames at 1 fps, creates manifests per clip.
Targets full-session streams + guides covering every menu/screen.

Usage:
    .venv\\Scripts\\python scripts/collect_all_videos.py [--max-videos N] [--res RES]
Defaults: unlimited videos, 480p target resolution.
"""
import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("collect_all")

# ── Locate tools ─────────────────────────────────────────────────────────────
def find_tool(name: str) -> str:
    """Find a tool in venv Scripts, WinGet Links, WinGet Packages, or system PATH."""
    candidates = [
        PROJECT_ROOT / ".venv" / "Scripts" / f"{name}.exe",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    # Check winget links
    winget_links = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links"
    w = winget_links / f"{name}.exe"
    if w.is_file():
        return str(w)
    # Check winget packages (deep search for ffmpeg etc.)
    winget_pkgs = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    if winget_pkgs.exists():
        for exe in winget_pkgs.rglob(f"{name}.exe"):
            return str(exe)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"Cannot locate {name}")

YTDLP = None
FFMPEG = None

def init_tools():
    global YTDLP, FFMPEG
    # Use python -m yt_dlp (pip install) — the .exe wrapper lacks EJS scripts
    YTDLP = [sys.executable, "-m", "yt_dlp"]
    FFMPEG = find_tool("ffmpeg")
    log.info("yt-dlp: %s (via python -m)", sys.executable)
    log.info("ffmpeg: %s", FFMPEG)

# ── Video list (curated for maximum UI coverage) ─────────────────────────────
# Format: (video_id, title_hint, expected_duration_s, priority)
# Priority 1 = must-have (full gameplay), 2 = useful (guides/tutorials), 3 = bonus
VIDEO_TARGETS = [
    # ── MENU & START COVERAGE (user priority: start menu → game entry → all screens) ──
    ("jsEUZN6hge4", "EVERY Menu Explained - Taking Over Germany LP1 Tutorial", 2478, 1),
    ("p8WPz-i1Oqo", "How to play and get off to a good start - Tutorial EN", 4851, 1),
    ("gJQvleUeX-s", "Geoplanetical full gameplay from start", 1550, 1),
    # ── LONG SESSIONS (most frames per download) ──
    ("3zWSgTAX4D4", "Can I eliminate Americas debt - Geoplanetical Live 8hr", 28832, 1),
    ("qHZDa8DfVy4", "GEOPLANETICAL Pre-Release Livestream 3.5hr", 13165, 1),
    ("xMCdl87-Eno", "GEOPLANETICAL UNITED STATES GAMEPLAY", 8526, 1),
    ("mcdVoApNMjs", "Geoplanetical Ep 2", 8138, 1),
    ("OPo16hyYCOg", "Is This The Superpower Game Weve Been Waiting For", 5582, 1),
    ("DpDdoNyUYLc", "GEOPLANETICAL GAMEPLAY BRAZIL", 5496, 1),
    ("g0jc84L8jSM", "Geoplanetical unlabeled full session", 4597, 1),
    # ── MEDIUM EPISODES (variety of scenarios & decisions) ──
    ("5yi8eihAf2Q", "Geoplanetical S1Ep006", 3017, 1),
    ("JLsY2ge65_s", "Geoplanetical S1Ep005", 2197, 1),
    ("MIDsBijjx3I", "Geoplanetical S1Ep012", 1918, 1),
    ("jhe_Xp4oiSI", "Geoplanetical S1Ep1", 1819, 1),
    ("7xWoSQuFsAA", "Geoplanetical S1Ep008", 1746, 1),
    ("HG4pbEEUgVw", "Geoplanetical S1Ep009", 1507, 1),
    # ── SHORT / SPECIFIC (military, UI demos) ──
    ("OR9MwWdaDjY", "Military Invasion and Annexation Showcase", 878, 2),
    ("H1DEd_Oem78", "This Game Makes HoI4 Look SIMPLE", 1113, 2),
    ("oRIZU3L1JJU", "This New Map Game Lets You Do Almost ANYTHING", 1090, 2),
    # ── TUTORIALS (menu walkthroughs) ──
    ("fNT1PrarfMY", "No-BS Beginners Guide", 540, 2),
]

# ── Download + extract one video ─────────────────────────────────────────────
def process_video(vid_id: str, title_hint: str, output_dir: Path, target_res: int = 480) -> dict:
    slug = title_hint.replace(" ", "_")[:60].replace("'", "")
    clip_dir = output_dir / slug
    dl_dir = output_dir / "downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://www.youtube.com/watch?v={vid_id}"
    out_tmpl = str(dl_dir / f"{slug}.%(ext)s")

    # Pick format: prefer mp4 at target res or below; fallback to worst available
    fmt = f"best[height<={target_res}][ext=mp4]/best[height<={target_res}]/worst[ext=mp4]/worst"

    # Find node for yt-dlp JS runtime (required since yt-dlp 2025+)
    node_candidates = [
        Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links" / "node.exe",
        Path("C:/Program Files/nodejs/node.exe"),
    ]
    node_path = None
    for nc in node_candidates:
        if nc.is_file():
            node_path = str(nc)
            break

    js_runtime_arg = []
    if node_path:
        js_runtime_arg = ["--js-runtimes", f"node:{node_path}"]

    cmd = [
        *YTDLP,
        *js_runtime_arg,
        "--no-playlist",
        "-f", fmt,
        "-o", out_tmpl,
        "--merge-output-format", "mp4",
        "--retries", "3",
        "--fragment-retries", "5",
        url,
    ]

    log.info("  ↓ Downloading '%s' (%s) …", title_hint[:60], vid_id)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        err = (proc.stderr or "unknown")[-400:].replace('\n', ' ')
        log.warning("  ✗ Download failed: %s", err)
        return {"title": title_hint, "ok": False, "error": "download_failed", "detail": err}

    # Locate downloaded file
    video_path = None
    for ext in (".mp4", ".webm", ".mkv", ".flv"):
        candidate = dl_dir / f"{slug}{ext}"
        if candidate.exists():
            video_path = candidate
            break
    if video_path is None:
        files = sorted(dl_dir.glob(f"{slug}.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        video_path = files[0] if files else None
    if video_path is None or not video_path.exists():
        log.warning("  ✗ Could not locate downloaded file for %s", slug)
        return {"title": title_hint, "id": vid_id, "ok": False, "error": "file_not_found"}

    dl_size_mb = video_path.stat().st_size / 1e6
    dl_time = time.perf_counter() - t0
    log.info("  ↓ Done in %.0fs (%.1f MB)", dl_time, dl_size_mb)

    # Extract frames at 1 fps using ffmpeg
    cmd = [
        FFMPEG, "-y",
        "-i", str(video_path),
        "-vf", "select='not(mod(t\\,1))',setpts=N/1/TB",
        "-fps_mode", "vfr",
        "-q:v", "5",
        str(clip_dir / "frame_%05d.jpg"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        err = (proc.stderr or "")[-300:].replace('\n', ' ')
        log.warning("  ✗ Frame extraction failed: %s", err)
        return {"title": title_hint, "id": vid_id, "ok": False, "error": "extract_failed", "detail": err}

    frame_files = sorted(clip_dir.glob("frame_*.jpg"))
    n_frames = len(frame_files)
    frames_mb = sum(f.stat().st_size for f in frame_files) / 1e6
    log.info("  ✓ Extracted %d frames (%.1f MB)", n_frames, frames_mb)

    # Get video resolution from first frame
    import struct
    w, h = 0, 0
    if frame_files:
        try:
            with open(frame_files[0], 'rb') as fh:
                data = fh.read()
            # JPEG SOF marker parsing
            i = 2
            while i < len(data):
                if data[i] == 0xFF and data[i+1] in (0xC0, 0xC1, 0xC2, 0xC3):
                    h = struct.unpack('>H', data[i+5:i+7])[0]
                    w = struct.unpack('>H', data[i+7:i+9])[0]
                    break
                elif data[i] == 0xFF:
                    i += 2 + struct.unpack('>H', data[i+2:i+4])[0]
                else:
                    i += 1
        except Exception:
            pass

    # Write manifest
    manifest = {
        "source": url,
        "video_id": vid_id,
        "title": title_hint,
        "num_frames": n_frames,
        "resolution": f"{w}x{h}" if w else "unknown",
        "fps_requested": 1.0,
        "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "download_mb": round(dl_size_mb, 1),
        "frames_mb": round(frames_mb, 1),
    }
    (clip_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Delete source video to save space
    video_path.unlink(missing_ok=True)
    log.info("  🗑 Deleted source (%.1f MB reclaimed)", dl_size_mb)

    return {
        "title": title_hint, "id": vid_id, "ok": True,
        "frames": n_frames, "res": f"{w}x{h}",
        "download_mb": round(dl_size_mb, 1), "frames_mb": round(frames_mb, 1),
        "time_s": round(time.perf_counter() - t0, 1),
    }

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Comprehensive Geoplanetical video collection")
    ap.add_argument("--max-videos", type=int, default=0, help="Max videos (0=all targets)")
    ap.add_argument("--res", type=int, default=1080, help="Target resolution height (default 1080)")
    ap.add_argument("--output-dir", type=str, default=None, help="Output dir (default data/videos)")
    args = ap.parse_args()

    init_tools()

    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "data" / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Disk guard
    usage = shutil.disk_usage(str(out_dir))
    free_gb = usage.free / 1024**3
    log.info("Disk: %.1f GB free of %.1f GB total", free_gb, usage.total / 1024**3)
    if free_gb < 5.0:
        log.error("Less than 5 GB free — aborting.")
        sys.exit(1)

    targets = VIDEO_TARGETS[:args.max_videos] if args.max_videos > 0 else VIDEO_TARGETS
    log.info("=== Collecting %d videos at %dp ===", len(targets), args.res)

    results = []
    for i, (vid_id, title_hint, dur_s, priority) in enumerate(targets):
        log.info("\n[%d/%d] Priority %d | ~%dm | %s",
                 i+1, len(targets), priority, dur_s//60, title_hint[:50])
        try:
            r = process_video(vid_id, title_hint, out_dir, target_res=args.res)
        except Exception as e:
            r = {"title": title_hint, "id": vid_id, "ok": False, "error": str(e)}
        results.append(r)
        status = "✓" if r.get("ok") else "✗"
        log.info("  %s %s → %s", status, title_hint[:40], 
                 f"{r['frames']} frames @ {r.get('res','?')}" if r.get("ok") else r.get("error","?"))

    # Summary
    ok_count = sum(1 for r in results if r.get("ok"))
    total_frames = sum(r.get("frames", 0) for r in results)
    total_mb = sum(r.get("frames_mb", 0) for r in results)
    log.info("\n" + "=" * 60)
    log.info("COLLECTION COMPLETE: %d/%d videos OK, %d total frames (%.1f MB)",
             ok_count, len(results), total_frames, total_mb)
    log.info("=" * 60)

    # Write collection manifest
    summary_path = out_dir / "collection_summary.json"
    summary_path.write_text(json.dumps({
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_videos_targeted": len(targets),
        "total_videos_ok": ok_count,
        "total_frames": total_frames,
        "total_frames_mb": round(total_mb, 1),
        "results": results,
    }, indent=2))
    log.info("Summary saved to %s", summary_path)

if __name__ == "__main__":
    main()
