"""
Green Vision — Session Boot File
=================================
Read this first in any new agent session to re-orient instantly.
Last updated: 2026-11-22
"""

# ── Project ────────────────────────────────────────────────────────────────
PROJECT_ROOT = r"E:\Green Vision"
GIT_REMOTE   = "https://github.com/IOC4FUN/green-vision.git"
BRANCH       = "master"
LAST_COMMIT  = "c72931bff"  # feat(brain): add curiosity module, curriculum manager & pretraining launchers

# ── Current Stage ──────────────────────────────────────────────────────────
STAGE = "A — CNN Reconstruction Pretraining"
STATUS = "READY TO LAUNCH"
NOTES = [
    "All code committed & pushed (curiosity, curriculum, launchers).",
    "Data collection complete: 111,275 frames from 16 video sources.",
    "No pending extractions or downloads.",
    "Next action: run scripts/run_pretrain_final.py on RTX 4090.",
]

# ── Key Files ──────────────────────────────────────────────────────────────
KEY_FILES = {
    "launcher":      "scripts/run_pretrain_final.py",       # Stage A entry point
    "launcher_bat":  "scripts/run_pretrain_final_v2.bat",   # double-click wrapper
    "orchestrator":  "scripts/run_training_master.py",      # two-stage pipeline controller
    "curiosity":     "src/brain/curiosity.py",               # ICM intrinsic-motivation module (used by agent)
    "curriculum":    "scripts/curriculum.py",               # difficulty-scheduled training
    "venv":          ".venv-cuda\\Scripts\\python.exe",     # torch 2.15 CUDA + deps
    "data_dir":      "data/videos/",                       # 111K extracted frames
    "readme":        "README.md",
    "changelog":     "CHANGELOG.md",
}

# ── Environment ────────────────────────────────────────────────────────────
ENV = {
    "os":            "Windows 10/11 x64",
    "gpu":           "NVIDIA RTX 4090 (24 GB VRAM)",
    "git":           r"C:\Program Files\Git\cmd\git.exe",   # v2.56.1.windows.1
    "python_venv":   ".venv-cuda",                        # torch, opencv, yt-dlp, pyautogui, flask, gymnasium
    "model_coding":  "Qwen 3.8 27B — use for code writing & analysis",
    "model_training":"Qwen 3.5 9B  — use during long training runs",
}

# ── Agent Workflow Rules ───────────────────────────────────────────────────
RULES = [
    "Start a NEW chat session each time you return (avoids context-window crashes).",
    "Read this file first: os.fs.read('E:/Green Vision/session_boot.py').",
    "For training runs: os.shell.run with background:true and timeoutMs=720000.",
    "Use full git path if os.git.* tools fail: 'C:\\Program Files\\Git\\cmd\\git.exe'.",
    "Throwaway scripts go in scripts/tmp/ (already .gitignored) — never pollute repo root.",
    "Pre-commit hook blocks secrets; post-commit auto-updates CHANGELOG.md.",
    "data/videos/ is .gitignored — do NOT stage it.",
]

# ── Known Issues / TODOs ───────────────────────────────────────────────────
KNOWN_ISSUES = [
    # "(none as of 2026-11-22)",
]
TODO = [
    "Run Stage A pretraining (CNN reconstruction on 111K frames).",
    "After Stage A completes → launch Stage B (policy fine-tuning via run_training_master.py).",
    "Monitor GPU memory during training; RTX 4090 has 24 GB VRAM.",
]

if __name__ == "__main__":
    print("=" * 60)
    print(f"Green Vision | {STAGE} | {STATUS}")
    print(f"Last commit: {LAST_COMMIT}")
    for n in NOTES:
        print(f"  • {n}")
    print(f"\nNext: python {KEY_FILES['launcher']}")
    print("=" * 60)
