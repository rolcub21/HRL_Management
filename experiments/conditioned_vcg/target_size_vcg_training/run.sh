#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../../.." && pwd)"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

if [[ $# -lt 1 ]]; then
  echo "usage: bash run.sh <self-check|prepare|train-stage-a|train-stage-b|freeze-test|run-one|run-panel|analyze> [options]" >&2
  exit 2
fi

case "$1" in
  self-check|prepare|train-stage-a|train-stage-b)
    exec "$project_root/.venv/bin/python" -B -m experiments.conditioned_vcg.target_size_vcg_training.run "$@"
    ;;
  freeze-test|run-one|run-panel|analyze)
    exec "$project_root/.venv/bin/python" -B -m experiments.conditioned_vcg.target_size_vcg_training.evaluate "$@"
    ;;
  *)
    echo "unknown command: $1" >&2
    exit 2
    ;;
esac
