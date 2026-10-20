#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-e118}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
exec "$PYTHON_BIN" -B -m experiments.conditioned_vcg.E118_short_exploration_earliness_8x8_182k_v1.run "$@"
