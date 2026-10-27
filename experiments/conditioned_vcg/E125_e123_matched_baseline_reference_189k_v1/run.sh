#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-e125}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec "${PYTHON_BIN:-$ROOT/.venv/bin/python}" -B -m experiments.conditioned_vcg.E125_e123_matched_baseline_reference_189k_v1.run "$@"
