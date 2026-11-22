@echo off
REM ====================================
REM GREEN VISION — FINAL PRETRAINING RUN
REM NO USER INTERACTION REQUIRED
REM ALL PATHS QUOTED TO HANDLE SPACES
REM ====================================
echo [START] Autonomous Pretraining on RTX 4090
cd /D "E:\Green Vision"

:: Step 1: Create venv at project root (absolute paths)
echo.
echo [1/5] Creating virtual environment...
C:\Users\Benny\AppData\Local\Programs\Python\Python312\python.exe -m venv ".venv-cuda"
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to create venv
    pause
    exit /b 1
)
echo ✓ Done

:: Step 2: Upgrade pip + install PyTorch CUDA via index URL
echo.
echo [2/5] Installing PyTorch with CUDA support...
C:\Users\Benny\AppData\Local\Programs\Python\Python312\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
if %ERRORLEVEL% neq 0 (
    echo WARNING: CUDA wheel failed, trying CPU fallback...
    C:\Users\Benny\AppData\Local\Programs\Python\Python312\python.exe -m pip install torch torchvision torchaudio --quiet
)
if %ERRORLEVEL% neq 0 (
    echo ERROR: PyTorch installation failed
    pause
    exit /b 1
)
echo ✓ Done

:: Step 3: Verify CUDA is available using venv Python directly
echo.
echo [3/5] Verifying CUDA installation...
"E:\Green Vision\.venv-cuda\Scripts\python.exe" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

set TORCH_CUDA=%errorlevel%
if "%TORCH_CUDA%"=="1" (
    echo ERROR: CUDA not detected — aborting
    pause
    exit /b 1
)
echo ✓ CUDA confirmed

:: Step 4: Install dependencies
echo.
echo [4/5] Installing project dependencies...
C:\Users\Benny\AppData\Local\Programs\Python\Python312\python.exe -m pip install opencv-python-headless pytesseract Pillow mss pyautogui psutil requests PyYAML python-dotenv rich gymnasium flask yt-dlp cryptography --quiet
if %ERRORLEVEL% neq 0 exit /b 1
echo ✓ Done

:: Step 5: Run pretraining
echo.
echo ====================================
echo STARTING PRETRAINING — STAGE A
echo ====================================
echo Target: 111,275 frames | Batch size: 8 | Epochs: 2
echo Estimated time: ~30-40 minutes on RTX 4090
echo.

"E:\Green Vision\.venv-cuda\Scripts\python.exe" "scripts/pretrain_from_video.py" --stage a --epochs-a 2 --batch-size 8

if %ERRORLEVEL% neq 0 (
    echo.
    echo ====================================
    echo PRETRAINING FAILED!
    echo ====================================
) else (
    echo.
    echo ====================================
    echo PRETRAINING COMPLETE!
    echo Checkpoint saved to: data\\models\
    echo ====================================
)

pause