from __future__ import annotations

import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def load_extension():
    if not torch.cuda.is_available():
        raise RuntimeError("The FastWAM W4A8 CUTLASS backend requires CUDA.")
    root = Path(__file__).resolve().parents[4]
    cutlass = Path(os.environ.get("CUTLASS_PATH", root / "third_party" / "cutlass"))
    if not (cutlass / "include/cutlass/cutlass.h").is_file():
        raise RuntimeError(f"CUTLASS headers not found under {cutlass}.")
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    if shutil.which("ninja") is None and (Path(sys.executable).parent / "ninja").is_file():
        os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}"
    csrc = Path(__file__).resolve().parent / "csrc"
    return load(
        name="fastwam_steerquant_cutlass_w4a8_sm89_shape_v3",
        sources=[str(csrc / "binding.cpp"), str(csrc / "w4a8.cu")],
        extra_include_paths=[str(cutlass / "include")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-lineinfo"],
        with_cuda=True,
        verbose=os.environ.get("FASTWAM_W4A8_BUILD_VERBOSE", "0") == "1",
    )
