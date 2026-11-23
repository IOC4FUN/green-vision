"""
OCR engine for Green Vision.

Extracts text from game screenshots to build a structured understanding of
the current UI state (menus open, values visible, notifications).

Uses Tesseract via pytesseract. Falls back gracefully if tesseract binary
is not installed yet.
"""
from __future__ import annotations
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("perception.ocr")

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None

# Precompiled patterns for common Geoplanetical UI elements
UI_PATTERNS: dict[str, re.Pattern] = {
    "budget_balance": re.compile(r"(?:Balance|State cash|Treasury)[:\s]*(-?\d+[.,]?\d*)\s*(?:M|B|bn)?", re.I),
    "approval_rating": re.compile(r"(?:Approval|Popularity|Support)[:\s]*(\d+\.?\d*)%", re.I),
    "date_display": re.compile(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s+(\d{4})"),
    "country_name": re.compile(r"^(?:Germany|United States|Japan|Brazil|France|UK|China|India|Russia|Canada|Australia)", re.M),
    "notification": re.compile(r"[A-Z][a-z]+ (?:has been|is now|will be|was passed|was vetoed|struck|declared)", re.I),
}


@dataclass
class OCRResult:
    raw_text: str
    extracted: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    processing_ms: float = 0.0


class OCREngine:
    """
    Wraps Tesseract OCR with game-specific post-processing.

    Usage:
        ocr = OCREngine(lang="eng")
        result = ocr.process(frame_array)   # numpy H×W×3
        print(result.extracted.get("budget_balance", "?"))
    """

    def __init__(self, lang: str = "eng", min_confidence: float = 40.0):
        self._lang = lang
        self._min_conf = min_confidence
        self._available = self._check_tesseract()
        if self._available and pytesseract is not None:
            # Point pytesseract at the resolved binary so it works even when
            # tesseract is not on the process PATH (e.g. fresh shell).
            exe = self._find_tesseract()
            if exe:
                pytesseract.pytesseract.tesseract_cmd = exe
        else:
            logger.warning(
                "Tesseract binary not found. Install via: winget install UB-Mannheim.TesseractOCR "
                "or download from https://github.com/UB-Mannheim/tesseract/wiki"
            )

    # Known install locations checked as fallback when tesseract is not on PATH.
    _TESSERACT_FALLBACKS = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]

    @classmethod
    def _find_tesseract(cls) -> Optional[str]:
        """Return the path to the tesseract binary, or None."""
        import shutil
        found = shutil.which("tesseract")
        if found:
            return found
        for p in cls._TESSERACT_FALLBACKS:
            if os.path.isfile(p):
                return p
        return None

    @classmethod
    def _check_tesseract(cls) -> bool:
        try:
            exe = cls._find_tesseract()
            if exe is None:
                return False
            subprocess.run([exe, "--version"], capture_output=True, timeout=5)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    @property
    def available(self) -> bool:
        return self._available and pytesseract is not None and Image is not None

    def process(self, frame: np.ndarray) -> OCRResult:
        """Run OCR on a BGR/RGB numpy array and extract game-relevant fields."""
        import time
        t0 = time.perf_counter()

        if not self.available:
            return OCRResult(raw_text="", processing_ms=(time.perf_counter()-t0)*1000)

        # Ensure RGB for PIL (mss returns BGRA/BGR on Windows)
        if frame.ndim == 3 and frame.shape[2] >= 3:
            bgr = frame[:, :, :3]
            rgb = bgr[:, :, ::-1].copy()
        elif frame.ndim == 2:
            rgb = np.stack([frame] * 3, axis=-1)
        else:
            rgb = frame
        pil_img = Image.fromarray(rgb.astype(np.uint8))

        try:
            config = f"--oem 3 --psm 6 -l {self._lang}"
            raw = pytesseract.image_to_string(pil_img, config=config).strip()
        except Exception as e:
            logger.error("OCR failed: %s", e)
            return OCRResult(raw_text="", processing_ms=(time.perf_counter()-t0)*1000)

        # Extract structured fields
        extracted: dict[str, str] = {}
        for key, pattern in UI_PATTERNS.items():
            m = pattern.search(raw)
            if m:
                extracted[key] = m.group(0).strip()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        confidence = min(len(extracted) / len(UI_PATTERNS), 1.0)

        return OCRResult(
            raw_text=raw,
            extracted=extracted,
            confidence=confidence,
            processing_ms=elapsed_ms,
        )
