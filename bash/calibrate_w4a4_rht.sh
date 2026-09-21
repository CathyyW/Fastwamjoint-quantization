#!/usr/bin/env bash
set -euo pipefail
STEERQUANT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${STEERQUANT_PYTHON:-python}" "$STEERQUANT_ROOT/scripts/calibrate.py" --activation-bits 4 --rotation rht "$@"
