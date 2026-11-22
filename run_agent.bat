@echo off
REM ─── Green Vision – Agent Launcher (Windows) ─────────────────────
setlocal
set VENV="E:\Green Vision\.venv\Scripts\python.exe"
set PROJECT="E:\Green Vision"

if not exist "%VENV%" (
    echo [ERROR] Virtual environment not found at %VENV%
    echo Run setup first: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo Starting Green Vision agent...
cd /d "%PROJECT%"

REM Launch with args passed through
"%VENV%" main.py %*

pause
