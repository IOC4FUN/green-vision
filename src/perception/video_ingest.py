"""
YouTube / local-video ingestion for Green Vision.

Downloads screen-recorded tutorials or gameplay videos and extracts
keyframes so they can be used as:
  • Supervised-pretraining episodes  (state → action → next_state)
  • Visual grounding examples        (UI element screenshots + labels)
  • World-model training data       (transition tuples)

Pipeline (inspired by VideoAgentTrek, arXiv:2510.19488):
    1. yt-dlp downloads the video (or we accept a local file path).
    2. ffmpeg extracts frames at scene-change boundaries OR fixed FPS.
    3. Each frame is stored alongside metadata (timestamp, source URL,
       optional OCR text) in a structured directory layout.
    4. A lightweight *action-event detector* (heuristic v1: detect large
       pixel diffs between consecutive frames to flag "something happened")
       tags which frames are likely post-action states.

Usage:
    from src.perception.video_ingest import VideoIngestor

    vi = VideoIngestor(output_dir="data/videos")
    clips = vi.download_and_extract(
        url="https://youtube.com/watch?v=abc123",
        fps=2,                # extract every 0.5 s
        min_scene_delta=0.08, # ignore near-identical frames
    )
    for clip in clips:
        print(clip.timestamp_s, clip.frame_path)
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("perception.video_ingest")

# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class ExtractedFrame:
    """One extracted frame with its metadata."""
    timestamp_s: float          # seconds into the source video
    frame_index: int            # sequential index within this extraction run
    frame_path: str             # absolute path to the saved .png / .jpg
    width: int = 0
    height: int = 0
    is_action_boundary: bool = False   # True if a large visual change was detected
    ocr_text: str = ""

@dataclass
class VideoClip:
    """A single ingested video and all its extracted frames."""
    source_url: str             # YouTube URL or local file path
    title: str                  # video title (best-effort)
    duration_s: float           # total video length in seconds
    output_dir: str             # directory where frames were written
    manifest_path: str          # JSON manifest alongside the frames
    frames: list[ExtractedFrame] = field(default_factory=list)
    download_time_s: float = 0.0
    extract_time_s: float = 0.0

# ── Main class ──────────────────────────────────────────────────────────────

class VideoIngestor:
    """
    Downloads videos (YouTube via yt-dlp, or local files) and extracts
    keyframes for downstream training.

    Parameters
    ----------
    output_dir : base directory for storing downloaded videos + extracted frames.
    ffmpeg_bin : path to the ffmpeg executable (default: 'ffmpeg' on PATH).
    ytdlp_bin  : path to the yt-dlp executable (default: 'yt-dlp' on PATH).
    max_video_mb : refuse downloads larger than this (MB). 0 = no limit.
    """

    def __init__(
        self,
        output_dir: str | Path = "data/videos",
        ffmpeg_bin: str = "ffmpeg",
        ytdlp_bin: str = "yt-dlp",
        max_video_mb: int = 2048,
        keep_source_video: bool = False,
        min_free_gb: float = 5.0,
        jpeg_quality: int = 5,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ffmpeg = ffmpeg_bin
        self.ytdlp = ytdlp_bin
        self.max_video_bytes = max_video_mb * 1024 * 1024 if max_video_mb > 0 else 0
        self.keep_source_video = keep_source_video
        self.min_free_bytes = int(min_free_gb * 1024**3)
        self.jpeg_quality = jpeg_quality  # 1=best … 31=worst; 5 ≈ good UI fidelity at ~60 % size of q2
        # Verify binaries exist at construction time so we fail fast.
        for name, bin_path in (("ffmpeg", self.ffmpeg), ("yt-dlp", self.ytdlp)):
            if shutil.which(bin_path) is None and not Path(bin_path).is_file():
                logger.warning("%s not found on PATH – related features will be unavailable", name)

    # ── Storage guards ────────────────────────────────────────────────
    def _check_disk_space(self) -> bool:
        """Return True if enough free space remains; log a warning otherwise."""
        import shutil as _sh
        usage = _sh.disk_usage(str(self.output_dir))
        free_gb = usage.free / 1024**3
        if free_gb < self.min_free_gb:
            logger.warning(
                "Low disk space: %.1f GB free (< %.1f GB minimum). "
                "Consider running cleanup_old_clips() before ingesting more videos.",
                free_gb, self.min_free_gb,
            )
            return False
        return True

    def cleanup_old_clips(self, max_age_days: int = 7, keep_manifests: bool = True) -> int:
        """
        Remove extracted-frame directories older than *max_age_days*.
        Returns the number of clip dirs removed.
        If *keep_manifests* is True the manifest.json is preserved so the
        pretrainer can still reference it (frames are gone though).
        """
        import time as _time
        now = _time.time()
        cutoff = now - max_age_days * 86400
        removed = 0
        for child in sorted(self.output_dir.iterdir()):
            if not child.is_dir() or child.name == "downloads":
                continue
            mtime = child.stat().st_mtime
            if mtime < cutoff:
                # Delete frame files but optionally keep manifest
                for f in child.glob("frame_*.jpg"):
                    f.unlink(missing_ok=True)
                if not keep_manifests:
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    mp = child / "manifest.json"
                    if mp.exists():
                        mp.write_text(json.dumps({"note": "frames purged", "purged_at": _time.strftime("%Y-%m-%dT%H:%M:%S")}))
                removed += 1
                logger.info("Cleaned up old clip dir: %s", child.name)
        if removed:
            logger.info("cleanup_old_clips: removed frames from %d dirs (> %d days old)", removed, max_age_days)
        return removed

    # ── Public API ────────────────────────────────────────────────────────

    def download_and_extract(
        self,
        url: str,
        fps: float = 2.0,
        min_scene_delta: float = 0.05,
        quality: str = "best[height<=720]/best",
        local_only: bool = False,
    ) -> VideoClip:
        """
        Full pipeline: download → extract frames → detect action boundaries → write manifest.

        Parameters
        ----------
        url : YouTube URL or absolute path to a local video file.
        fps : target extraction rate (frames per second).
        min_scene_delta : minimum mean-absolute pixel change (0-1) between
                          consecutive frames to flag an action boundary.
        quality : yt-dlp format selector string.
        local_only : if True, skip the download step (url must be a local path).

        Returns
        -------
        VideoClip with all extracted frames populated.
        """
        t_start = time.perf_counter()
        is_local = Path(url).is_file() or local_only

        # 1 ── Download
        if is_local:
            video_path = Path(url).resolve()
            title = video_path.stem
            logger.info("Using local video: %s", video_path)
        else:
            video_path, title = self._download(url, quality=quality)

        # 2 ── Probe duration
        duration_s = self._probe_duration(video_path)
        logger.info(
            "Video ready: %s (%.1f s, %.1f MB)",
            title, duration_s, video_path.stat().st_size / 1e6,
        )

        # 3 ── Extract frames
        clip_dir = self.output_dir / f"{self._slug(title)}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        frames = self._extract_frames(
            video_path, clip_dir, fps=fps, min_scene_delta=min_scene_delta,
        )
        extract_time = time.perf_counter() - t_start

        # 4 ── Write manifest
        manifest = {
            "source": str(url),
            "title": title,
            "duration_s": round(duration_s, 2),
            "fps_requested": fps,
            "min_scene_delta": min_scene_delta,
            "num_frames": len(frames),
            "action_boundaries": sum(1 for f in frames if f.is_action_boundary),
            "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "frames": [
                {
                    "t": f.timestamp_s,
                    "idx": f.frame_index,
                    "path": f.frame_path,
                    "w": f.width,
                    "h": f.height,
                    "boundary": f.is_action_boundary,
                }
                for f in frames
            ],
        }
        manifest_path = clip_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        clip = VideoClip(
            source_url=str(url),
            title=title,
            duration_s=duration_s,
            output_dir=str(clip_dir),
            manifest_path=str(manifest_path),
            frames=frames,
            download_time_s=round(extract_time, 2),
            extract_time_s=round(extract_time, 2),
        )
        logger.info(
            "Ingested %d frames (%d boundaries) from '%s' in %.1f s",
            len(frames), manifest["action_boundaries"], title, extract_time,
        )
        return clip

    def load_manifest(self, clip_dir: str | Path) -> list[ExtractedFrame]:
        """Re-load previously extracted frames from a manifest file."""
        mp = Path(clip_dir) / "manifest.json"
        if not mp.exists():
            raise FileNotFoundError(f"No manifest at {mp}")
        data = json.loads(mp.read_text())
        return [
            ExtractedFrame(
                timestamp_s=f["t"],
                frame_index=f["idx"],
                frame_path=f["path"],
                width=f.get("w", 0),
                height=f.get("h", 0),
                is_action_boundary=f.get("boundary", False),
            )
            for f in data.get("frames", [])
        ]

    # ── Internal helpers ──────────────────────────────────────────────────

    def _download(self, url: str, quality: str) -> tuple[Path, str]:
        """Download via yt-dlp; returns (video_path, title)."""
        out_tmpl = str(self.output_dir / "downloads" / "%s.%(ext)s")
        cmd = [
            self.ytdlp,
            "--no-playlist",
            "-f", quality + "/best",
            "-o", out_tmpl,
            "--write-info-json",
            "--print-after-download", "%(title)s|%(filepath)s",
            url,
        ]
        logger.info("Downloading: %s", url)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(f"yt-dlp failed ({proc.returncode}): {proc.stderr[:500]}")

        elapsed = time.perf_counter() - t0
        # Parse the last line of stdout for title|path
        lines = [l.strip() for l in proc.stdout.strip().splitlines() if l.strip()]
        if not lines:
            raise RuntimeError("yt-dlp produced no output")
        parts = lines[-1].rsplit("|", 1)
        title = parts[0] if len(parts) == 2 else Path(url).stem
        video_path = Path(parts[1]) if len(parts) == 2 else self._find_latest_download()

        # Size guard
        if self.max_video_bytes and video_path.stat().st_size > self.max_video_bytes:
            video_path.unlink(missing_ok=True)
            raise ValueError(
                f"Video too large ({video_path.stat().st_size / 1e6:.0f} MB > "
                f"{self.max_video_mb} MB limit)"
            )
        logger.info("Downloaded '%s' in %.1f s (%.1f MB)",
                     title, elapsed, video_path.stat().st_size / 1e6)
        return video_path, title

    def _find_latest_download(self) -> Path:
        dl_dir = self.output_dir / "downloads"
        files = sorted(dl_dir.glob("*.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            raise RuntimeError("No downloaded file found")
        return files[0]

    @staticmethod
    def _slug(title: str, max_len: int = 60) -> str:
        import re
        s = re.sub(r'[^\w\-]+', '_', title).strip('_')
        return s[:max_len] or "untitled"

    def _probe_duration(self, video_path: Path) -> float:
        # Use ffprobe for cleaner duration readout
        ffprobe = self.ffmpeg.replace("ffmpeg", "ffprobe")
        try:
            proc = subprocess.run(
                [ffprobe, "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
                capture_output=True, text=True, timeout=30,
            )
            return float(proc.stdout.strip())
        except Exception:
            logger.warning("Could not probe duration; defaulting to 0")
            return 0.0

    def _extract_frames(
        self,
        video_path: Path,
        out_dir: Path,
        fps: float = 2.0,
        min_scene_delta: float = 0.05,
    ) -> list[ExtractedFrame]:
        """
        Extract frames at the given rate using ffmpeg, then flag action
        boundaries by comparing consecutive frame pixel statistics.
        """
        import numpy as np
        from PIL import Image

        interval = 1.0 / fps
        # ffmpeg filter: select every Nth frame based on time
        cmd = [
            self.ffmpeg, "-i", str(video_path),
            "-vf", f"select='not(mod(t\\,{interval:.4f}))',setpts=N/{fps:g}/TB",
            "-vsync", "vfr",
            "-q:v", "2",
            str(out_dir / "frame_%05d.jpg"),
        ]
        logger.info("Extracting frames @ %.1f fps …", fps)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg extract failed: {proc.stderr[:500]}")
        elapsed = time.perf_counter() - t0

        # Collect extracted files in order
        frame_files = sorted(out_dir.glob("frame_*.jpg"))
        if not frame_files:
            logger.warning("No frames extracted")
            return []

        logger.info("Extracted %d frames in %.1f s", len(frame_files), elapsed)

        # Detect action boundaries via mean-abs-diff on downsampled grayscale
        frames: list[ExtractedFrame] = []
        prev_gray: Optional[np.ndarray] = None
        for idx, fp in enumerate(frame_files):
            img = Image.open(fp).convert("L")
            w, h = img.size
            small = np.array(img.resize((64, 64)), dtype=np.float32) / 255.0

            boundary = False
            if prev_gray is not None:
                delta = float(np.mean(np.abs(small - prev_gray)))
                boundary = delta >= min_scene_delta
            prev_gray = small

            ts = idx * interval
            frames.append(ExtractedFrame(
                timestamp_s=round(ts, 3),
                frame_index=idx,
                frame_path=str(fp.resolve()),
                width=w,
                height=h,
                is_action_boundary=boundary,
            ))

        n_boundaries = sum(1 for f in frames if f.is_action_boundary)
        logger.info("Action boundaries detected: %d / %d", n_boundaries, len(frames))
        return frames
