#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-e127}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec timeout --signal=TERM --kill-after=10s 600 "${PYTHON_BIN:-$ROOT/.venv/bin/python}" -B -m experiments.conditioned_vcg.E127_same_state_ranking_diagnostic_191k_v1.run "$@"
