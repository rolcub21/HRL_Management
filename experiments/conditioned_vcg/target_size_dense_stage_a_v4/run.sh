#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="${HRL_PROJECT_ROOT:-$(cd "$script_dir/../../.." && pwd)}"
staging_root="$(cd "$script_dir/../../.." && pwd)"
export PYTHONPATH="$staging_root:$project_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${TRAIN_THREADS:-1}"
export MKL_NUM_THREADS="${TRAIN_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${TRAIN_THREADS:-1}"

exec "$project_root/.venv/bin/python" -B -m experiments.conditioned_vcg.target_size_dense_stage_a_v4.run "$@"
