#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-e123}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
exec "$PYTHON_BIN" -B -m experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v1.run "$@"
