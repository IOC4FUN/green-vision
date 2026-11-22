# perception/ — eyes

Turns pixels into data the brain can use.

## Files
- `capture.py` — `ScreenCapture`: background-thread grabber via `mss` (Windows GDI/DXGI backend – captures even when another window is on top). Ring buffer (`deque`, default 16) keeps recent frames for temporal context; resizes each frame to 224×224 RGB for the CNN. API: `start()`, `stop()`, `latest_frame()` (non-blocking), `recent_frames(n)`.
  - `Frame` dataclass: `timestamp`, `raw` (H×W×3 uint8 BGR), `resized` (224² RGB), `ocr_text`, `window_title`.
  - Needs `pip install mss`; PIL optional but used for LANCZOS resize.
- `ocr_engine.py` — `OCREngine`: pytesseract wrapper + game-specific post-processing. `UI_PATTERNS` dict of compiled regexes extracts budget_balance, approval_rating, date_display, country_name, notification from raw text. `process(frame)` → `OCRResult(raw_text, extracted, confidence, processing_ms)`. Degrades gracefully if tesseract binary missing (`available` flag).

## Dev notes
- Capture runs in its own daemon thread at `capture_fps` (default 5); never block the main loop on it.
- OCR is the slow path (~tens–hundreds ms). If you raise decision rate, consider running OCR off-thread or only when a menu is open.
- Add new UI fields by extending `UI_PATTERNS` here, then surface them into the state vector in `agent._update_state_vector`.
