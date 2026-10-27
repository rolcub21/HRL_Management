#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/e138-matplotlib}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
exec .venv/bin/python -B -m experiments.conditioned_vcg.E138_residual_grid_mixture_pilot_202k_v1.run "$@"
