#!/usr/bin/env python3
"""MiniLLM installer: looks at your hardware and installs the PyTorch that fits it.

    python3 install.py              detect, show the plan, ask, install
    python3 install.py --dry-run    only show what was detected and what would happen
    python3 install.py --yes        don't ask questions
    python3 install.py --cpu        force the CPU-only PyTorch (smallest download)
    python3 install.py --tier small choose the model size yourself (tiny/small/medium/large)
    python3 install.py --no-venv    install into the current Python (for Google Colab)
    python3 install.py --reinstall  delete the .venv folder and start over

Needs only Python 3.8+ (standard library). Everything goes into the .venv folder inside
this project, so deleting the project folder removes it all.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from minillm import hardware, presets  # noqa: E402  (both use the standard library only)
from minillm.console import ask_yes, c, fix_console  # noqa: E402

VENV = ROOT / ".venv"
GOOD_PYTHONS = [(3, 12), (3, 13), (3, 11), (3, 10)]  # best first
PROBE = "import sys, platform; print(sys.version_info[0], sys.version_info[1], sys.version_info[2], platform.machine())"
VERIFY = r"""
import json, torch
r = {"version": torch.__version__, "device": "cpu", "name": "CPU"}
try:
    if torch.cuda.is_available():
        r["device"], r["name"] = "cuda", torch.cuda.get_device_name(0)
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        r["device"], r["name"] = "xpu", torch.xpu.get_device_name(0)
    elif torch.backends.mps.is_available():
        r["device"], r["name"] = "mps", "Apple GPU"
    x = torch.randn(512, 512, device=r["device"])
    (x @ x).sum().item()
except Exception as e:
    r["error"] = str(e)[:300]
print("MINILLM " + json.dumps(r))
"""
LAUNCH = "py" if os.name == "nt" else "python3"


def die(msg):
    print(c("\n" + msg, "red"))
    sys.exit(1)


def venv_python():
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def probe(exe):
    out = hardware.run([str(exe), "-c", PROBE]).split()
    if len(out) != 4:
        return None
    try:
        version = tuple(int(x) for x in out[:3])
    except ValueError:
        return None
    return {"exe": str(exe), "version": version, "machine": out[3].lower()}


def find_pythons(hw):
    """Every usable Python on this computer, best first."""
    found, seen = [], set()

    def add(exe):
        exe = (exe or "").strip()
        if not exe or not os.path.exists(exe):
            return
        key = os.path.realpath(exe)
        if key in seen or key.startswith(os.path.realpath(VENV)):
            return
        seen.add(key)
        info = probe(exe)
        if info:
            found.append(info)

    add(sys.executable)
    for v in ("3.12", "3.13", "3.11", "3.10"):
        add(shutil.which("python" + v))
    add(shutil.which("python3"))
    if os.name == "nt" and shutil.which("py"):
        for v in ("3.12", "3.13", "3.11", "3.10"):
            add(hardware.run(["py", "-" + v, "-c", "import sys; print(sys.executable)"]))
    uv = shutil.which("uv")
    if uv:
        for v in ("3.12", "3.13", "3.11", "3.10"):
            add(hardware.run([uv, "python", "find", "--system", v]))
    if hw["apple_silicon"]:  # an Intel-mode (Rosetta) Python can't use the Apple GPU
        found = [p for p in found if p["machine"] == "arm64"]
    found = [p for p in found if p["version"][:2] >= (3, 9)]
    found.sort(key=lambda p: GOOD_PYTHONS.index(p["version"][:2]) if p["version"][:2] in GOOD_PYTHONS else 10)
    return found, uv


def pick_python(hw, args):
    """Returns (how, info): how is 'python' (use info['exe']) or 'uv' (let uv fetch Python 3.12)."""
    if args.python:
        info = probe(args.python)
        if not info:
            die(f"Can't run {args.python}")
        return "python", info
    found, uv = find_pythons(hw)
    best = found[0] if found else None
    if best and best["version"][:2] in GOOD_PYTHONS:
        return "python", best
    if uv:
        return "uv", None
    if best:
        return "python", best  # 3.9 or 3.14+: works, PyTorch just picks a matching version
    if hw["apple_silicon"] and hw["rosetta"]:
        die("Your Python runs in Intel mode (Rosetta), which can't use the Apple GPU.\n"
            "Install the Apple Silicon Python from https://www.python.org/downloads/ (or: brew install python@3.12),\n"
            "then run this installer again.")
    die("No suitable Python found. Install Python 3.12 from https://www.python.org/downloads/ and try again.")


def describe_python(how, info):
    if how == "uv":
        return "3.12 (downloaded by uv, about 40 MB)"
    v = ".".join(map(str, info["version"]))
    note = "" if info["version"][:2] in GOOD_PYTHONS else c("  (works, but 3.12 is the safest choice)", "yellow")
    return f"{v}  {info['exe']}{note}"


def gpu_line(hw):
    acc = hw["accelerator"]
    if acc["type"] == "cuda":
        count = f"{acc['count']} x " if acc["count"] > 1 else ""
        return f"{count}{acc['name']}, {acc['memory_gb']:.0f} GB, CUDA driver {acc['cuda_driver'] or '?'}"
    if acc["type"] == "rocm":
        return f"{acc['name']}, {acc['memory_gb']:.0f} GB, ROCm {acc['rocm_version'] or '?'}"
    if acc["type"] == "mps":
        cores = f"{acc['gpu_cores']} cores, " if acc.get("gpu_cores") else ""
        return f"{acc['name']}, {cores}Metal (shares the {hw['ram_gb']:.0f} GB memory)"
    if acc["type"] == "xpu":
        return f"{acc['name']} (Intel XPU)"
    others = ", ".join(hw.get("other_gpus") or [])
    return "none PyTorch can use - the CPU will do the work" + (f"  (found: {others})" if others else "")


def show_hardware(hw):
    print(c("Your computer", "bold"))
    rows = [("System", f"{hw['os_name']} ({hw['arch']})"),
            ("CPU", f"{hw['cpu']}, {hw['cpu_cores']} cores"),
            ("Memory", f"{hw['ram_gb']:.0f} GB"),
            ("GPU", gpu_line(hw)),
            ("Free disk", f"{hw['disk_free_gb']:.1f} GB")]
    for label, value in rows:
        print(f"  {label:<10} {value}")
    if hw["rosetta"]:
        print(c("  Note: this Python runs in Intel mode (Rosetta); a native one will be used instead.", "yellow"))
    if hw["os"] == "Windows" and any("Radeon" in g for g in hw.get("other_gpus") or []):
        print(c("  Note: PyTorch can't use AMD GPUs on Windows yet, so training runs on the CPU.", "yellow"))


def pip(py, *args):
    return subprocess.call([str(py), "-m", "pip", *args, "--disable-pip-version-check"])


def verify_torch(py):
    for line in hardware.run([str(py), "-c", VERIFY], timeout=300).splitlines():
        if line.startswith("MINILLM "):
            return json.loads(line[len("MINILLM "):])
    return None


def works(result, expect):
    return bool(result) and not result.get("error") and (expect == "cpu" or result["device"] == expect)


def install_torch(py, plan, reinstall):
    expect = plan["attempts"][0]["expect"]
    have = verify_torch(py)
    if works(have, expect) and not reinstall:
        print(f"PyTorch {have['version']} is already installed and works on '{have['device']}' - nothing to download.")
        return have
    if have:
        print("Removing the PyTorch that's there now (it doesn't match your hardware)...")
        pip(py, "uninstall", "-y", "torch")
    for attempt in plan["attempts"]:
        print(c(f"\nInstalling PyTorch - {attempt['label']}", "bold"))
        cmd = ["install", "--no-cache-dir", *attempt["packages"]]
        if attempt.get("index"):
            cmd += ["--index-url", attempt["index"]]
        if pip(py, *cmd) == 0:
            result = verify_torch(py)
            if works(result, attempt["expect"]):
                return result
            reason = (result or {}).get("error") or "it can't reach the GPU (driver too old?)"
            print(c(f"  Installed, but {reason}. Trying the next option...", "yellow"))
            pip(py, "uninstall", "-y", "torch")
    die("Couldn't install a working PyTorch.\n"
        "Check your internet connection, or try:  " + LAUNCH + " install.py --cpu\n"
        "If it says no matching version, install Python 3.12 from python.org and run this again.")


def main():
    fix_console()
    ap = argparse.ArgumentParser(description="MiniLLM installer")
    ap.add_argument("--dry-run", action="store_true", help="only show what would happen")
    ap.add_argument("--yes", "-y", action="store_true", help="don't ask questions")
    ap.add_argument("--cpu", action="store_true", help="install the CPU-only PyTorch")
    ap.add_argument("--tier", choices=presets.TIERS, help="model size to use")
    ap.add_argument("--no-venv", action="store_true", help="install into this Python (Google Colab)")
    ap.add_argument("--python", help="use this Python for the .venv")
    ap.add_argument("--reinstall", action="store_true", help="delete .venv and install again")
    args = ap.parse_args()

    print(c("\nMiniLLM installer", "bold") + " - checking your hardware...\n")
    hw = hardware.detect()
    show_hardware(hw)

    plan = hardware.torch_plan(hw, force_cpu=args.cpu)
    auto_tier = "tiny" if args.cpu else hardware.choose_tier(hw)
    tier = args.tier or auto_tier

    if args.no_venv:
        how, py_info = "python", probe(sys.executable)
    else:
        how, py_info = pick_python(hw, args)
    venv_ok = VENV.exists() and venv_python().exists() and not args.reinstall
    torch_gb = 0.0 if (venv_ok or args.no_venv) else plan["size_gb"]

    free = hw["disk_free_gb"]
    need = torch_gb * 1.3 + presets.DISK_GB[tier]  # pip briefly needs extra room while unpacking
    while not args.tier and tier != "tiny" and free < need + 1.0:
        tier = presets.lower(tier)
        need = torch_gb * 1.3 + presets.DISK_GB[tier]

    print(c("\nThe plan", "bold"))
    print(f"  PyTorch    {plan['name']}" + (f", about {plan['size_gb']:.1f} GB" if torch_gb else " (already set up)"))
    if plan.get("note"):
        print(c(f"             {plan['note']}", "yellow"))
    if args.no_venv:
        print(f"  Python     this one ({sys.executable}), no .venv")
    else:
        print(f"  Python     {describe_python(how, py_info)}")
        print(f"             -> private environment in {VENV.name}/ (delete the folder to remove everything)")
    print(f"  Model size tier '{tier}' - {presets.TIER_INFO[tier]}"
          + (c(f" (lowered from '{auto_tier}' to fit your free disk)", "yellow") if tier != (args.tier or auto_tier) else ""))
    print(f"             level 3: {presets.approx_params(3, tier) / 1e6:.1f}M parameters, "
          f"level 4: {presets.approx_params(4, tier) / 1e6:.1f}M parameters (~{presets.TIER_TIME[tier]})")
    print(f"  Disk       about {need:.1f} GB in total, {free:.1f} GB free")
    if free < need + 0.5:
        print(c("  Warning: that's very tight. Free some space first, or use --cpu --tier tiny.", "red"))
    elif free < need + 3:
        print(c("  Your disk is nearly full. Downloads are kept small; 'python3 run.py clean' frees space later.",
                "yellow"))

    if args.dry_run:
        print("\n(dry run - nothing installed)")
        return
    if not args.yes and not ask_yes("\nInstall now?"):
        print("Cancelled.")
        return

    # 1. a private Python environment
    if args.no_venv:
        py = Path(sys.executable)
    else:
        if args.reinstall and VENV.exists():
            print("Deleting the old .venv...")
            shutil.rmtree(VENV)
        if not venv_python().exists():
            print(c("\nCreating the Python environment (.venv)", "bold"))
            if how == "uv":
                ok = subprocess.call([shutil.which("uv"), "venv", "--seed", "--python", "3.12", str(VENV)]) == 0
            else:
                ok = subprocess.call([py_info["exe"], "-m", "venv", str(VENV)]) == 0
                if not ok and shutil.which("uv"):
                    ok = subprocess.call([shutil.which("uv"), "venv", "--seed", "--python", py_info["exe"],
                                          str(VENV)]) == 0
            if not ok:
                die("Couldn't create the environment. On Ubuntu/Debian run:  sudo apt install python3-venv")
        py = venv_python()
        pip(py, "install", "--no-cache-dir", "--upgrade", "--quiet", "pip")

    # 2. PyTorch, 3. the two small helpers
    result = install_torch(py, plan, args.reinstall)
    print(c("\nInstalling numpy and certifi", "bold"))
    if pip(py, "install", "--no-cache-dir", "--quiet", plan.get("numpy", "numpy"), "certifi") != 0:
        die("Couldn't install numpy.")

    # 4. remember the result for the levels
    device = "cpu" if args.cpu else result["device"]
    acc = hw["accelerator"]
    if device == "mps":
        device_name = acc.get("name") or "Apple GPU"
    elif device == "cpu":
        device_name = hw["cpu"]
    else:
        device_name = result.get("name") or acc.get("name") or device
    python_version = hardware.run([str(py), "-c", "import platform; print(platform.python_version())"]).strip()
    hardware.save_profile({
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "tier": tier,
        "auto_tier": auto_tier,
        "device": device,
        "device_name": device_name,
        "torch": result["version"],
        "python": python_version,
        "venv": not args.no_venv,
        "hardware": hw,
    })

    print(c("\nAll set!", "green") + f"  {device_name} ({device}), tier '{tier}', PyTorch {result['version']}")
    print(f"Settings saved in {hardware.PROFILE_PATH.name}. Next:\n")
    print(f"  {LAUNCH} run.py          open the menu")
    print(f"  {LAUNCH} run.py 0        your first model (takes seconds)")
    print(f"  {LAUNCH} run.py info     check disk use any time\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
