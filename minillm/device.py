"""Pick the device (GPU or CPU) and number format when a level starts."""
from __future__ import annotations

import contextlib
import os
import re

import torch

from . import hardware


class Runtime:
    def __init__(self, device, amp_dtype, tier, name):
        self.device = device
        self.amp_dtype = amp_dtype  # None = normal 32-bit floats
        self.tier = tier
        self.name = name

    def autocast(self):
        """Mixed precision: half-size numbers where it's safe. ~2x faster on NVIDIA/AMD, ~15% on Apple."""
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def grad_scaler(self):
        if self.amp_dtype is not torch.float16:
            return None
        try:
            return torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler()

    def describe(self):
        fmt = {torch.bfloat16: "bf16", torch.float16: "fp16"}.get(self.amp_dtype, "fp32")
        return f"{self.name} ({self.device.type}, {fmt})"


def available(kind):
    if kind == "cuda":
        return torch.cuda.is_available()
    if kind == "mps":
        mps = getattr(torch.backends, "mps", None)
        return bool(mps and mps.is_available())
    if kind == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    return kind == "cpu"


def best_device():
    for kind in ("cuda", "xpu", "mps"):
        if available(kind):
            return kind
    return "cpu"


def _apple_bf16(chip_name):
    """bf16 runs ~15% faster on M2 and newer (M1 has no bf16 hardware). Test it before trusting it."""
    match = re.search(r"M(\d+)", chip_name or "")
    if not match or int(match.group(1)) < 2 or os.environ.get("MINILLM_FP32"):
        return None
    try:
        w = torch.randn(64, 64, device="mps", requires_grad=True)
        with torch.autocast(device_type="mps", dtype=torch.bfloat16):
            loss = (w @ w).float().sum()
        loss.backward()
        torch.mps.synchronize()
        return torch.bfloat16
    except Exception:
        return None


def setup(device=None, tier=None, seed=1337):
    profile = hardware.load_profile()
    want = device or os.environ.get("MINILLM_DEVICE") or profile.get("device") or best_device()
    if not available(want.split(":")[0]):
        fallback = best_device()
        print(f"Note: '{want}' isn't available here - using '{fallback}' instead.")
        want = fallback
    dev = torch.device(want)

    amp = None
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        name = torch.cuda.get_device_name(dev)
        if getattr(torch.version, "hip", None):  # AMD
            amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:  # NVIDIA: bf16 from the RTX 30xx / A100 generation on, fp16 before that
            amp = torch.bfloat16 if torch.cuda.get_device_capability(dev)[0] >= 8 else torch.float16
    elif dev.type == "mps":
        name = profile.get("hardware", {}).get("accelerator", {}).get("name") or "Apple GPU"
        amp = _apple_bf16(name)
    elif dev.type == "xpu":
        name = torch.xpu.get_device_name(0)
    else:
        name = profile.get("hardware", {}).get("cpu") or "CPU"

    if tier is None:
        tier = profile.get("tier")
    if tier is None:
        tier = hardware.choose_tier(hardware.detect()) if dev.type != "cpu" else "tiny"
        print(f"(No install profile found - using tier '{tier}'. Run the installer to save your settings.)")
    torch.manual_seed(seed)
    return Runtime(dev, amp, tier, name)
