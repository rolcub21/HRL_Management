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

command_name="${1:?use prepare, run-pilot, freeze-confirmation, run-one, run-all, or analyze}"
shift
"$python_bin" -m experiments.conditioned_vcg.E27_certified_dynamic_sustained_99k.run \
  "$command_name" "$@"
