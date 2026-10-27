#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/e139-matplotlib}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
exec .venv/bin/python -B -m experiments.conditioned_vcg.E139_grid_mixture_three_seeds_203k_v1.run "$@"
