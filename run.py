#!/usr/bin/env python3
"""MiniLLM launcher - start here.

    python3 run.py              menu
    python3 run.py 3            run level 3 (extra options pass through, e.g. --iters 500)
    python3 run.py chat         talk to your level 5 model
    python3 run.py info         your hardware, settings and disk use
    python3 run.py clean        free disk space (downloads, models, or everything)
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from minillm import hardware  # noqa: E402
from minillm.console import ask_yes, c, fix_console, human_size  # noqa: E402

LEVELS = {
    "0": ("level0_counting.py", "Counting model", "pure Python, counts which letter follows which (seconds)"),
    "1": ("level1_neural_bigram.py", "First neural network", "learns that table by gradient descent (~10 s)"),
    "2": ("level2_mlp.py", "Looks 8 letters back", "embeddings + a hidden layer (~15 s)"),
    "3": ("level3_tiny_gpt.py", "Tiny GPT (letters)", "a real transformer writing Shakespeare-like text"),
    "4": ("level4_story_gpt.py", "Story GPT (words)", "own tokenizer + a bigger GPT trained on stories"),
    "5": ("level5_chat.py", "Chat model", "fine-tunes level 4 to answer requests"),
}
VENV = ROOT / ".venv"
INSTALLER = "install.bat" if os.name == "nt" else "./install.sh"
LAUNCH = "py" if os.name == "nt" else "python3"


def venv_python():
    path = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(path) if path.exists() else None


def python_for(level):
    vp = venv_python()
    if vp:
        return vp
    if level != "0" and importlib.util.find_spec("torch") is None:
        sys.exit(f"PyTorch isn't installed yet. Run the installer first:  {INSTALLER}\n"
                 f"(Level 0 works without it:  {LAUNCH} run.py 0)")
    return sys.executable


def run_level(level, extra):
    script = ROOT / "levels" / LEVELS[level][0]
    proc = subprocess.Popen([python_for(level), str(script), *extra])
    while True:
        try:
            return proc.wait()
        except KeyboardInterrupt:
            pass  # the level itself handles Ctrl+C (and saves its progress)


def folder_size(path):
    total = 0
    for dirpath, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


FOLDERS = [("PyTorch environment", VENV), ("Downloaded data", ROOT / "data"), ("Trained models", ROOT / "out")]


def show_info():
    profile = hardware.load_profile()
    if profile:
        print(c("\nYour setup", "bold"))
        print(f"  Device     {profile.get('device_name')} ({profile.get('device')})")
        print(f"  Tier       {profile.get('tier')}  (detected: {profile.get('auto_tier')})")
        print(f"  PyTorch    {profile.get('torch')} on Python {profile.get('python')}")
        print(f"  Installed  {profile.get('created')}")
    else:
        print(c(f"\nNot installed yet - run {INSTALLER}", "yellow"))
    print(c("\nDisk use", "bold"))
    for label, path in FOLDERS:
        size = human_size(folder_size(path)) if path.exists() else "-"
        print(f"  {label + ' (' + path.name + '/)':<30} {size}")
    out = ROOT / "out"
    if out.exists():
        for model_dir in sorted(p for p in out.iterdir() if p.is_dir()):
            print(f"    {model_dir.name:<28} {human_size(folder_size(model_dir))}")
    print(f"  {'Free on this disk':<30} {human_size(shutil.disk_usage(ROOT).free)}\n")
    return 0


def clean(argv):
    choices = {
        "data": ("downloads and token files", [ROOT / "data"]),
        "models": ("trained models", [ROOT / "out"]),
        "all": ("downloads and trained models", [ROOT / "data", ROOT / "out"]),
        "everything": ("downloads, models AND PyTorch (run the installer again after)",
                       [ROOT / "data", ROOT / "out", VENV, hardware.PROFILE_PATH]),
    }
    sizes = {k: sum(folder_size(p) if p.is_dir() else (p.stat().st_size if p.exists() else 0) for p in paths)
             for k, (_, paths) in choices.items()}
    what = argv[0] if argv and argv[0] in choices else None
    if what is None:
        print(c("\nFree up disk space", "bold"))
        for i, key in enumerate(choices, 1):
            print(f"  {i}  {key:<11} {human_size(sizes[key]):>9}   {choices[key][0]}")
        try:
            pick = input("\nWhich one? (Enter = cancel) ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return 0
        keys = list(choices)
        what = keys[int(pick) - 1] if pick.isdigit() and 1 <= int(pick) <= len(keys) else (pick if pick in choices else None)
        if what is None:
            return 0
    if "--yes" not in argv and not ask_yes(f"Delete {choices[what][0]} ({human_size(sizes[what])})?", default=False):
        return 0
    for path in choices[what][1]:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()
    print(f"Freed {human_size(sizes[what])}.")
    return 0


def menu():
    profile = hardware.load_profile()
    print(c("\nMiniLLM - build a language model from scratch, from tiny to big\n", "bold"))
    if profile:
        print(f"Running on {profile.get('device_name')} - tier '{profile.get('tier')}'\n")
    else:
        print(c(f"PyTorch isn't installed yet: run {INSTALLER} (level 0 works without it)\n", "yellow"))
    for key, (_, title, blurb) in LEVELS.items():
        print(f"  {c(key, 'cyan')}      {title:<22} {c(blurb, 'dim')}")
    print(f"\n  {c('chat', 'cyan')}   talk to your level 5 model")
    print(f"  {c('info', 'cyan')}   hardware, settings and disk use")
    print(f"  {c('clean', 'cyan')}  free disk space")
    try:
        choice = input("\nPick one (Enter = quit): ").strip()
    except (EOFError, KeyboardInterrupt):
        return 0
    return dispatch(choice.split())


def dispatch(argv):
    if not argv:
        return 0
    cmd, extra = argv[0].lower(), argv[1:]
    if cmd.startswith("level"):
        cmd = cmd[5:]
    if cmd in LEVELS:
        return run_level(cmd, extra)
    if cmd == "chat":
        return run_level("5", ["--chat", *extra])
    if cmd == "info":
        return show_info()
    if cmd == "clean":
        return clean(extra)
    if cmd in ("help", "-h", "--help"):
        print(__doc__)
        return 0
    print(f"Unknown choice '{cmd}'.\n{__doc__}")
    return 2


if __name__ == "__main__":
    fix_console()
    sys.exit(dispatch(sys.argv[1:]) if len(sys.argv) > 1 else menu())
