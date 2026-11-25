"""Which compute device YOLO runs on - for live detection AND for training.

Measured on an Apple Silicon Mac (torch 2.14.1, ultralytics 8.4.171):
  * Ultralytics' own "automatic" choice is cuda-or-cpu: it never picks the
    Apple GPU (mps), so leaving `device` unset always meant CPU here.
  * live detection 640 px: cpu ~43 ms/frame, mps ~9 ms/frame (~4.6× faster)
  * training 640 px, 40 images, 15 epochs: cpu 81 s, mps 38 s (~2.1× faster),
    same mAP50.
  * "cuda" on a Mac raises "Invalid CUDA device" - on every frame, if it was
    passed straight through.

So "auto" here means: mps on Apple Silicon, cuda when torch sees an NVIDIA
GPU, else cpu - and a device this computer does not have falls back to cpu
with a note instead of failing.
"""
from __future__ import annotations

import platform
import sys

AUTO = "auto"


def _torch():
    import torch  # heavy: only imported when a device is actually needed
    return torch


def mps_available() -> bool:
    # Intel build (torch 2.2.2): mps reports available on Intel Macs with an
    # AMD GPU and under Rosetta, but torchvision 0.17 has no mps NMS, so every
    # detection raises NotImplementedError. mps is for Apple Silicon only.
    if not is_apple_silicon():
        return False
    try:
        return bool(_torch().backends.mps.is_available())
    except Exception:
        return False


def cuda_available() -> bool:
    try:
        return bool(_torch().cuda.is_available())
    except Exception:
        return False


def torch_has_cuda_build() -> bool:
    """False for a CPU-only PyTorch (the default pip wheel on Windows): such a
    build can never use an NVIDIA card, whatever the computer has."""
    try:
        torch = _torch()
        if torch.version.cuda is not None:
            return True
        # v9.0.0: a v6-style app froze the CPU torch/version.py into its
        # archive, so trust the CUDA library itself as well.
        from pathlib import Path
        return (Path(torch.__file__).parent / "lib" / "torch_cuda.dll").is_file()
    except Exception:
        return False


def is_apple_silicon() -> bool:
    """Cheap platform check (no torch import) for labelling UI choices."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _kind(device) -> str:
    d = str(device).strip().lower()
    if d in ("", "none", AUTO):
        return AUTO
    if d == "mps":
        return "mps"
    if d == "cpu":
        return "cpu"
    if d.startswith("cuda") or d.replace(",", "").isdigit():
        return "cuda"
    return "unknown"


def resolve(requested) -> tuple[str, str]:
    """(device to pass to ultralytics, note for the user - "" if none).

    requested: None / "auto" / "mps" / "cuda" / "cuda:1" / "0" / "cpu".
    """
    kind = _kind(requested)
    if kind == AUTO:
        if mps_available():
            return "mps", ""
        if cuda_available():
            return "0", ""
        if sys.platform == "win32":
            # An NVIDIA card is there but unused: tell the student (v9.0.0).
            try:
                from . import gpupack
                if gpupack.nvidia_info():
                    return resolve("cuda")
            except Exception:
                pass
        return "cpu", ""
    if kind == "cpu":
        return "cpu", ""
    if kind == "mps":
        if mps_available():
            return "mps", ""
        return "cpu", "⚠️ No mps GPU (Apple Silicon) here — using the CPU"
    if kind == "cuda":
        if cuda_available():
            d = str(requested).strip().lower()
            return (d.split(":", 1)[1] if d.startswith("cuda:") else "0" if d == "cuda" else d), ""
        if sys.platform == "win32":
            # v9.0.0: say WHY (no card / old driver / GPU Pack missing or
            # unpacked in the wrong folder / ...) instead of a generic line.
            try:
                from . import gpupack
                why = gpupack.diagnose()
                detail = why["lines"][-1] if why["lines"] else ""
                hint = "" if why.get("can_install") else ""
                return "cpu", f"⚠️ cuda not available — using the CPU · {detail}{hint}"
            except Exception:
                pass
        if sys.platform != "darwin" and not torch_has_cuda_build():
            return "cpu", "⚠️ This app has the CPU version of PyTorch and can't use an NVIDIA card (cuda) — using the CPU"
        return "cpu", "⚠️ No usable NVIDIA card (cuda) found (no card, or the driver is too old) — using the CPU"
    return "cpu", f"⚠️ Unknown device {requested!r} — using the CPU"


def describe(device) -> str:
    """Human name for a resolved device, e.g. for "Training on ..."."""
    kind = _kind(device)
    if kind == "mps":
        return "GPU Apple Silicon (mps)"
    if kind == "cuda":
        return f"GPU NVIDIA (cuda:{str(device).split(':')[-1]})" if str(device) != "cuda" else "GPU NVIDIA (cuda)"
    return "CPU"


# Choices for the Train page (value, label). The labels say whether THIS
# computer can use each one, without importing torch just to draw a menu.
def train_choices():
    mac = is_apple_silicon()
    return [
        (AUTO, "Automatic (recommended) — " + ("mps GPU on this computer" if mac else "GPU if there is one, otherwise CPU")),
        ("mps", "GPU / mps — Mac with Apple Silicon" + (" ✓" if mac else " ✗ not available on this computer")),
        ("cuda", "GPU / cuda — NVIDIA graphics card" + (" ✗ not available on this computer" if sys.platform == "darwin" else "")),
        ("cpu", "CPU — works on every computer, but slowest"),
    ]
