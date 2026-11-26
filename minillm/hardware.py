"""Hardware detection for MiniLLM.

Standard library only, so the installer can use it before PyTorch is installed.
Works on Python 3.8 and newer, on macOS, Linux and Windows.

detect()       -> what this computer has (CPU, memory, GPU, free disk)
choose_tier()  -> which model sizes fit that hardware
torch_plan()   -> which PyTorch build to install for it
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROFILE_PATH = ROOT / "hardware.json"
TORCH_INDEX = "https://download.pytorch.org/whl/"

# PyTorch builds for NVIDIA cards: (lowest CUDA driver version that runs it, build tag), newest first.
CUDA_BUILDS = [
    ((13, 0), "cu130"), ((12, 9), "cu129"), ((12, 8), "cu128"), ((12, 6), "cu126"),
    ((12, 4), "cu124"), ((12, 1), "cu121"), ((11, 8), "cu118"),
]
# PyTorch builds for AMD cards (Linux only): (ROCm version, build tag), newest first.
ROCM_BUILDS = [((6, 4), "rocm6.4"), ((6, 3), "rocm6.3"), ((6, 2), "rocm6.2"), ((6, 1), "rocm6.1")]


def run(cmd, timeout=20):
    """Run a command and return what it printed, or '' if it's missing or failed."""
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return ""
    return res.stdout if res.returncode == 0 else ""


def version_tuple(text):
    nums = re.findall(r"\d+", text or "")
    return tuple(int(n) for n in nums[:2]) if nums else (0,)


def _float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


# ------------------------------------------------------------------ the basics

def os_name():
    system = platform.system()
    if system == "Darwin":
        return "macOS " + platform.mac_ver()[0]
    if system == "Windows":
        return "Windows " + platform.release()
    if system == "Linux":
        try:
            with open("/etc/os-release", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        return line.split("=", 1)[1].strip().strip('"')
        except OSError:
            pass
        return "Linux"
    return system or "unknown"


def is_apple_silicon():
    if platform.system() != "Darwin":
        return False
    return platform.machine() == "arm64" or run(["sysctl", "-n", "hw.optional.arm64"]).strip() == "1"


def is_rosetta():
    """True when this Python is an Intel build running on an Apple Silicon Mac."""
    return platform.system() == "Darwin" and run(["sysctl", "-n", "sysctl.proc_translated"]).strip() == "1"


def ram_gb():
    system = platform.system()
    try:
        if system == "Darwin":
            return int(run(["sysctl", "-n", "hw.memsize"]).strip()) / 1024 ** 3
        if system == "Linux":
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / 1024 ** 2
        if system == "Windows":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.ullTotalPhys / 1024 ** 3
    except Exception:
        pass
    return 0.0


def cpu_name():
    system = platform.system()
    if system == "Darwin":
        name = run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        if name:
            return name
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif system == "Windows":
        name = run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]).strip()
        if name:
            return name.splitlines()[0].strip()
    return platform.processor() or platform.machine() or "unknown CPU"


# ------------------------------------------------------------------ GPUs

def _nvidia():
    exe = shutil.which("nvidia-smi")
    if not exe and platform.system() == "Windows":
        guess = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        exe = str(guess) if guess.exists() else None
    if not exe:
        return None
    fmt = "--format=csv,noheader,nounits"
    out = run([exe, "--query-gpu=name,memory.total,compute_cap", fmt])
    with_cc = bool(out.strip())
    if not with_cc:  # older drivers don't know compute_cap
        out = run([exe, "--query-gpu=name,memory.total", fmt])
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        gpus.append({"name": parts[0], "memory_gb": _float(parts[1]) / 1024,
                     "cc": _float(parts[2]) if with_cc and len(parts) > 2 else 0.0})
    if not gpus:
        return None
    match = re.search(r"CUDA Version:\s*([\d.]+)", run([exe]))
    best = max(gpus, key=lambda g: g["memory_gb"])
    return {"type": "cuda", "vendor": "NVIDIA", "name": best["name"], "count": len(gpus),
            "memory_gb": round(best["memory_gb"], 1), "compute_capability": best["cc"],
            "cuda_driver": match.group(1) if match else ""}


def _amd():
    if platform.system() != "Linux":
        return None  # PyTorch only supports AMD GPUs on Linux (ROCm)
    exe = shutil.which("rocm-smi") or ("/opt/rocm/bin/rocm-smi" if os.path.exists("/opt/rocm/bin/rocm-smi") else None)
    if not exe:
        return None
    name, mem = "AMD GPU", 0.0
    try:
        info = json.loads(run([exe, "--showproductname", "--showmeminfo", "vram", "--json"]) or "{}")
        for card in info.values():
            if not isinstance(card, dict):
                continue
            for key, value in card.items():
                low = key.lower()
                if "vram total memory" in low:
                    mem = max(mem, _float(value) / 1024 ** 3)
                elif low == "card series" and name == "AMD GPU":
                    name = str(value)
    except (ValueError, TypeError, AttributeError):
        pass
    version = ""
    for path in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"):
        try:
            version = Path(path).read_text().strip().split("-")[0]
            break
        except OSError:
            pass
    return {"type": "rocm", "vendor": "AMD", "name": name, "count": 1,
            "memory_gb": round(mem, 1), "rocm_version": version}


def _apple(ram):
    if not is_apple_silicon():
        return None
    match = re.search(r"Total Number of Cores:\s*(\d+)", run(["system_profiler", "SPDisplaysDataType"], timeout=30))
    chip = cpu_name()
    return {"type": "mps", "vendor": "Apple", "name": chip + " GPU" if chip.startswith("Apple") else "Apple GPU",
            "count": 1, "memory_gb": round(ram, 1), "gpu_cores": int(match.group(1)) if match else 0,
            "shared_memory": True}


def gpu_names():
    """Every graphics chip the OS reports (used to explain why a GPU can't be used)."""
    system = platform.system()
    if system == "Windows":
        out = run(["powershell", "-NoProfile", "-Command",
                   "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }"])
        return [line.strip() for line in out.splitlines() if line.strip()]
    if system == "Linux":
        out = run(["lspci"])
        return [line.split(": ", 1)[-1] for line in out.splitlines() if re.search(r"VGA|3D|Display", line)]
    return []


def _intel(names):
    for name in names:
        if re.search(r"Intel.*(Arc|Data Center GPU)", name):
            return {"type": "xpu", "vendor": "Intel", "name": name, "count": 1, "memory_gb": 0.0}
    return None


def detect():
    ram = ram_gb()
    names = []
    acc = _nvidia() or _amd() or _apple(ram)
    if acc is None:
        names = gpu_names()
        acc = _intel(names) or {"type": "cpu", "vendor": "", "name": "", "count": 0, "memory_gb": 0.0}
    return {
        "os": platform.system(),
        "os_name": os_name(),
        "arch": platform.machine(),
        "apple_silicon": is_apple_silicon(),
        "rosetta": is_rosetta(),
        "cpu": cpu_name(),
        "cpu_cores": os.cpu_count() or 1,
        "ram_gb": round(ram, 1),
        "disk_free_gb": round(shutil.disk_usage(ROOT).free / 1024 ** 3, 1),
        "accelerator": acc,
        "other_gpus": names,
    }


# ------------------------------------------------------------------ decisions

def choose_tier(hw):
    """Pick the model size that trains in a reasonable time on this hardware."""
    acc = hw["accelerator"]
    kind, mem = acc["type"], acc.get("memory_gb") or 0
    if kind in ("cuda", "rocm"):  # by graphics memory (VRAM)
        if mem >= 20:
            return "large"
        if mem >= 10:
            return "medium"
        return "small" if mem >= 5 else "tiny"
    if kind == "mps":  # Apple GPUs train ~5-10x slower than NVIDIA ones, so only Max chips get medium
        if (acc.get("gpu_cores") or 0) >= 30 and mem >= 32:
            return "medium"
        return "small" if mem >= 8 else "tiny"
    if kind == "xpu":
        return "small"
    return "tiny"


def torch_plan(hw, force_cpu=False):
    """Which PyTorch to install. Each attempt is tried in order until one works on this machine."""
    acc, system = hw["accelerator"], hw["os"]
    cpu = {"label": "CPU-only build", "index": TORCH_INDEX + "cpu", "packages": ["torch"], "expect": "cpu"}

    if system == "Darwin":
        if hw["apple_silicon"]:
            return {"name": "PyTorch for macOS" + ("" if force_cpu else " with Apple GPU support (Metal)"),
                    "size_gb": 0.8,
                    "attempts": [{"label": "standard macOS build", "index": None, "packages": ["torch"],
                                  "expect": "cpu" if force_cpu else "mps"}]}
        return {"name": "PyTorch 2.2.2 (the last version made for Intel Macs)", "size_gb": 0.5, "numpy": "numpy<2",
                "attempts": [{"label": "Intel Mac build", "index": None,
                              "packages": ["torch==2.2.2", "numpy<2"], "expect": "cpu"}]}

    if force_cpu or acc["type"] == "cpu":
        return {"name": "PyTorch, CPU-only build (smallest download)", "size_gb": 0.8, "attempts": [cpu]}

    if acc["type"] == "cuda":
        driver = version_tuple(acc.get("cuda_driver"))
        blackwell = (acc.get("compute_capability") or 0) >= 12  # RTX 50xx needs CUDA 12.8+ builds
        tags = [tag for need, tag in CUDA_BUILDS if need <= driver and not (blackwell and need < (12, 8))]
        attempts = [{"label": f"NVIDIA CUDA {tag[2:-1]}.{tag[-1]} build", "index": TORCH_INDEX + tag,
                     "packages": ["torch"], "expect": "cuda"} for tag in tags[:3]]
        plan = {"name": "PyTorch for NVIDIA GPUs (CUDA)", "size_gb": 4.5, "attempts": attempts + [cpu]}
        if not attempts:
            plan.update(name="PyTorch, CPU-only build", size_gb=0.8,
                        note="Your NVIDIA driver is too old for current PyTorch. Update it to use the GPU.")
        return plan

    if acc["type"] == "rocm":
        have = version_tuple(acc.get("rocm_version"))
        tags = [tag for need, tag in ROCM_BUILDS if have == (0,) or need <= have]
        attempts = [{"label": f"AMD ROCm {tag[4:]} build", "index": TORCH_INDEX + tag,
                     "packages": ["torch"], "expect": "cuda"} for tag in tags[:3]]
        return {"name": "PyTorch for AMD GPUs (ROCm)", "size_gb": 6.0, "attempts": attempts + [cpu]}

    if acc["type"] == "xpu":
        return {"name": "PyTorch for Intel GPUs (XPU)", "size_gb": 3.0,
                "attempts": [{"label": "Intel XPU build", "index": TORCH_INDEX + "xpu",
                              "packages": ["torch"], "expect": "xpu"}, cpu]}

    return {"name": "PyTorch, CPU-only build", "size_gb": 0.8, "attempts": [cpu]}


# ------------------------------------------------------------------ the saved profile

def load_profile():
    try:
        return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_profile(profile):
    PROFILE_PATH.write_text(json.dumps(profile, indent=2), encoding="utf-8")
