#!/usr/bin/env bash
# Wrap the existing SERVER command; this script supplies no robot action command.
set -euo pipefail
if [ "$#" -eq 0 ]; then
  echo 'Usage: bash with_real_w4a8_runtime.sh python -u <existing-server.py> <server-options>' >&2
  exit 2
fi
STEERQUANT_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export FASTWAM_RUNTIME_PROFILE="$STEERQUANT_REPO/configs/real_robot_w4a8_runtime.json"
export FASTWAM_W4A8_TILE=64
unset FASTWAM_W4A8_EXPERIMENTAL_SMALL_TILE
export PYTHONPATH="$STEERQUANT_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$@"
