"""Small helpers for terminal output. Standard library only."""
from __future__ import annotations

import os
import sys

_CODES = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33", "cyan": "36"}
_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and "NO_COLOR" not in os.environ
if _COLOR and os.name == "nt":
    os.system("")  # switches on colour support in the Windows console


def c(text, style):
    """Colour text when the terminal supports it."""
    return f"\033[{_CODES[style]}m{text}\033[0m" if _COLOR else str(text)


def fix_console():
    """Never crash on characters an old terminal can't show."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def ask_yes(question, default=True):
    hint = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"{question} {hint} ").strip().lower()
    except EOFError:
        return default
    if not answer:
        return default
    return answer in ("y", "yes")


def human_size(n_bytes):
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def fmt_time(seconds):
    seconds = int(max(0, seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds}s"
