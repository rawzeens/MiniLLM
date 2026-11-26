#!/bin/sh
# MiniLLM installer for macOS and Linux: finds a Python 3 and runs install.py,
# which detects your hardware and picks the right PyTorch.
#   ./install.sh            (add --dry-run, --cpu, --tier small, --yes ...)
cd "$(dirname "$0")" || exit 1
for p in python3.12 python3.13 python3.11 python3.10 python3 python; do
  if command -v "$p" >/dev/null 2>&1 && "$p" -c "import sys; sys.exit(sys.version_info < (3, 8))" 2>/dev/null; then
    exec "$p" install.py "$@"
  fi
done
echo "Python 3 wasn't found."
echo "  macOS:  install it from https://www.python.org/downloads/  (or: brew install python@3.12)"
echo "  Linux:  sudo apt install python3 python3-venv   (or your distro's equivalent)"
exit 1
