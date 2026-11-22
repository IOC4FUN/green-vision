@echo off
REM ─── Green Vision – Web Dashboard Launcher (Windows) ──────────────
setlocal
set VENV="E:\Green Vision\.venv\Scripts\python.exe"
set PROJECT="E:\Green Vision"

if not exist "%VENV%" (
    echo [ERROR] Virtual environment not found.
    pause
    exit /b 1
)

echo Starting Green Vision web dashboard on http://127.0.0.1:8090 ...
cd /d "%PROJECT%"
"%VENV%" main.py --web-dashboard --port 8090

pause
