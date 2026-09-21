#!/usr/bin/env bash
set -euo pipefail
STEERQUANT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STEERQUANT_PYTHON="${STEERQUANT_PYTHON:-python}"
"$STEERQUANT_PYTHON" "$STEERQUANT_ROOT/scripts/setup_dependencies.py"
export CUTLASS_PATH="$STEERQUANT_ROOT/third_party/cutlass"
export FAST_HADAMARD_PATH="$STEERQUANT_ROOT/third_party/fast-hadamard-transform"
export MAX_JOBS="${MAX_JOBS:-2}"
"$STEERQUANT_PYTHON" -c 'from fastwam_steerquant.kernels.w4a8.loader import load_extension; print(load_extension())'
"$STEERQUANT_PYTHON" -c 'from fastwam_steerquant.kernels.w4a4.loader import load_extension; print(load_extension())'
