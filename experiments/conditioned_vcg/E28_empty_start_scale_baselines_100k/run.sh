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

command_name="${1:?use prepare, run-pilot, freeze-confirmation, run-one, run-method, run-all, or analyze}"
shift
if [[ "$command_name" == "analyze" ]]; then
  "$python_bin" -m experiments.conditioned_vcg.E28_empty_start_scale_baselines_100k.analysis_fix \
    "$@"
else
  "$python_bin" -m experiments.conditioned_vcg.E28_empty_start_scale_baselines_100k.run \
    "$command_name" "$@"
fi
