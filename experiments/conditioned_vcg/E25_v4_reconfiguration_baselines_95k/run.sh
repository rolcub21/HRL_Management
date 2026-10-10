#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python_bin="$repo_root/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  python_bin=python3
fi

command_name="${1:?use prepare, run-pilot, run-one, run-method, run-all, analyze, or analyze-partial}"
shift

if [[ "$command_name" == "run-all" ]]; then
  for method in \
    duration_aware_dynamic_pslap \
    duration_aware_pslap_ga_2009_rolling_capacity_aware_partial \
    kim2020_a3c_spatial_adapted__stochastic; do
    "$python_bin" -m experiments.conditioned_vcg.E25_v4_reconfiguration_baselines_95k.run \
      run-method --method "$method" "$@"
  done
else
  "$python_bin" -m experiments.conditioned_vcg.E25_v4_reconfiguration_baselines_95k.run \
    "$command_name" "$@"
fi
