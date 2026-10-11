#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
MODULE="experiments.conditioned_vcg.E25_scale_baseline_transfer_95k.run"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/vcg-conditioned-e25-scale-baselines-95k}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

command_name="${1:-}"
shift || true
cd "${ROOT}"

run_module() {
  "${PYTHON}" -u -m "${MODULE}" "$@" --output-dir "${OUTPUT_DIR}"
}

case "${command_name}" in
  prepare|run-pilot|analyze|analyze-partial)
    run_module "${command_name}" "$@"
    ;;
  run-one)
    run_module run-one --method "${1:?method}" --scenario "${2:?scenario}" \
      --seed "${3:?seed}" --replicate "${4:?replicate}"
    ;;
  run-dynamic)
    run_module run-method --method duration_aware_dynamic_pslap
    ;;
  run-ga)
    run_module run-method --method duration_aware_pslap_ga_2009_rolling_capacity_aware_partial
    ;;
  run-a3c)
    run_module run-method --method kim2020_a3c_spatial_adapted__stochastic
    ;;
  run-all)
    bash "$0" run-dynamic
    bash "$0" run-ga
    bash "$0" run-a3c
    ;;
  *)
    echo "usage: $0 {prepare|run-pilot|run-dynamic|run-ga|run-a3c|run-all|run-one METHOD SCENARIO SEED REPLICATE|analyze|analyze-partial}" >&2
    exit 2
    ;;
esac
