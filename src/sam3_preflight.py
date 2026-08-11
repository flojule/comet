#!/usr/bin/env python3
# sam3_preflight.py
"""Check that this machine can actually run the SAM 3 backend.

Run this before track_sam3.py — it fails in seconds with a specific reason,
instead of a stack trace forty minutes into a frame extraction.

    python src/sam3_preflight.py
"""
from __future__ import annotations

import importlib
import platform
import shutil
import sys

# (label, minimum) — from the upstream SAM 3 README.
MIN_PYTHON = (3, 12)
MIN_TORCH  = (2, 7)
MIN_CUDA   = (12, 6)

OK, WARN, FAIL = "  ok  ", " warn ", " FAIL "


def _line(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label}" + (f" — {detail}" if detail else ""))


def check_python() -> bool:
    v = sys.version_info
    got = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= MIN_PYTHON:
        _line(OK, "python", got)
        return True
    _line(FAIL, "python", f"{got}, SAM 3 needs ≥ {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")
    return False


def check_torch() -> bool:
    try:
        import torch
    except ImportError:
        _line(FAIL, "torch", "not installed — pip install -r requirements-sam3.txt")
        return False

    ver = torch.__version__.split("+")[0]
    parts = tuple(int(x) for x in ver.split(".")[:2])
    if parts >= MIN_TORCH:
        _line(OK, "torch", ver)
    else:
        _line(FAIL, "torch", f"{ver}, SAM 3 needs ≥ {MIN_TORCH[0]}.{MIN_TORCH[1]}")
        return False

    if not torch.cuda.is_available():
        _line(FAIL, "cuda", "torch.cuda.is_available() is False — SAM 3 video "
                            "tracking needs a GPU")
        return False

    cuda_ver = torch.version.cuda or "unknown"
    try:
        cparts = tuple(int(x) for x in cuda_ver.split(".")[:2])
        status = OK if cparts >= MIN_CUDA else WARN
    except ValueError:
        status = WARN
    _line(status, "cuda", f"torch built against {cuda_ver} "
                          f"(recommended ≥ {MIN_CUDA[0]}.{MIN_CUDA[1]})")

    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gb = p.total_memory / 1024 ** 3
        status = OK if gb >= 16 else WARN
        _line(status, f"gpu{i}", f"{p.name}, {gb:.1f} GB")
    return True


def check_sam3() -> bool:
    try:
        mod = importlib.import_module("sam3.model_builder")
    except ImportError as e:
        _line(FAIL, "sam3", f"{e} — clone facebookresearch/sam3 and `pip install -e .`")
        return False
    if not hasattr(mod, "build_sam3_video_predictor"):
        _line(FAIL, "sam3", "sam3.model_builder has no build_sam3_video_predictor — "
                            "the installed sam3 is too old or is a different package")
        return False
    _line(OK, "sam3", "build_sam3_video_predictor importable")
    return True


def check_pipeline_deps() -> bool:
    ok = True
    for mod in ("cv2", "numpy", "scipy"):
        try:
            __import__(mod)
            _line(OK, mod)
        except ImportError:
            _line(FAIL, mod, "pip install -r requirements.txt")
            ok = False
    _line(OK if shutil.which("ffmpeg") else WARN, "ffmpeg",
          "found" if shutil.which("ffmpeg") else "missing — only to_webm.py needs it")
    return ok


def main() -> int:
    print(f"Comet SAM 3 preflight — {platform.platform()}\n")
    results = [
        check_python(),
        check_pipeline_deps(),
        check_torch(),
        check_sam3(),
    ]
    print()
    if all(results):
        print("All checks passed — `python src/track_sam3.py` should run.")
        return 0
    print("Preflight failed. Fix the FAIL lines above, then re-run.")
    print("Note: SAM 3 checkpoints are gated on Hugging Face — request access at")
    print("      https://huggingface.co/facebook/sam3 and `huggingface-cli login`.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
