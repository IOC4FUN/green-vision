@echo off
REM ─── Green Vision – One-time Windows Setup ────────────────────────
setlocal
set PROJECT="E:\Green Vision"
cd /d "%PROJECT%"

echo === Creating virtual environment ===
python -m venv .venv
if errorlevel 1 (
    echo [ERROR] Failed to create venv. Is Python 3.10+ installed?
    pause
    exit /b 1
)

echo === Installing dependencies ===
".venv\Scripts\pip.exe" install --upgrade pip
".venv\Scripts\pip.exe" install -r requirements.txt
if errorlevel 1 (
    echo [WARN] Some packages may have failed. Check output above.
)

echo === Checking Tesseract OCR ===
where tesseract >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Tesseract not in PATH. Install via: winget install UB-Mannheim.TesseractOCR
    echo        Then add C:\Program Files\Tesseract-OCR to your system PATH.
) else (
    echo [OK] Tesseract found.
)


echo.
echo === Setup complete! ===
echo Run the agent:     run_agent.bat
echo Run dashboard:     run_dashboard.bat
echo Run smoke test:    .venv\Scripts\python.exe scripts\smoke_test.py
pause
