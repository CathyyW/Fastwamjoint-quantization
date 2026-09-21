from __future__ import annotations

import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def _cutlass_root() -> Path:
    candidates: list[Path] = []
    if os.environ.get("CUTLASS_PATH"):
        candidates.append(Path(os.environ["CUTLASS_PATH"]))
    package_root = Path(__file__).resolve().parents[4]
    candidates.append(package_root / "third_party" / "cutlass")
    for candidate in candidates:
        if (candidate / "include" / "cutlass" / "cutlass.h").is_file():
            return candidate
    raise RuntimeError("CUTLASS headers were not found; set CUTLASS_PATH to the pinned CUTLASS v4.4.2 checkout.")


def _fast_hadamard_root() -> Path:
    candidates: list[Path] = []
    if os.environ.get("FAST_HADAMARD_PATH"):
        candidates.append(Path(os.environ["FAST_HADAMARD_PATH"]))
    package_root = Path(__file__).resolve().parents[4]
    candidates.extend(
        (
            package_root / "third_party" / "fast-hadamard-transform",
        )
    )
    for candidate in candidates:
        if (candidate / "csrc" / "fast_hadamard_transform_common.h").is_file():
            return candidate
    raise RuntimeError(
        "Fast Hadamard Transform headers were not found; set FAST_HADAMARD_PATH "
        "to a Dao-AILab/fast-hadamard-transform checkout."
    )


@lru_cache(maxsize=1)
def load_extension():
    if not torch.cuda.is_available():
        raise RuntimeError("The native W4A4 backend requires CUDA.")
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    if shutil.which("ninja") is None:
        environment_bin = str(Path(sys.executable).parent)
        if (Path(environment_bin) / "ninja").is_file():
            os.environ["PATH"] = f"{environment_bin}:{os.environ.get('PATH', '')}"
    source_root = Path(__file__).resolve().parent / "csrc"
    return load(
        name="fastwam_steerquant_cutlass_w4a4_sm89_wam_rot_v4",
        sources=[str(source_root / "binding.cpp"), str(source_root / "w4a4.cu")],
        extra_include_paths=[
            str(_cutlass_root() / "include"),
            str(_fast_hadamard_root() / "csrc"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-lineinfo"],
        with_cuda=True,
        verbose=os.environ.get("FASTWAM_W4A4_BUILD_VERBOSE", "0") == "1",
    )
