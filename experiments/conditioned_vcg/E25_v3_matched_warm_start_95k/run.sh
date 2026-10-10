#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/vcg-conditioned-e25-v3-matched-warm-start-95k}"
MODULE="experiments.conditioned_vcg.E25_v3_matched_warm_start_95k.run"

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
  prepare|run-pilot|run-all|analyze|analyze-partial)
    run_module "${command_name}" "$@"
    ;;
  run-method)
    run_module run-method --method "${1:?method}"
    ;;
  run-one)
    run_module run-one --method "${1:?method}" --scenario "${2:?scenario}" \
      --seed "${3:?seed}" --replicate "${4:?replicate}"
    ;;
  *)
    echo "usage: $0 {prepare|run-pilot|run-all|run-method METHOD|run-one METHOD SCENARIO SEED REPLICATE|analyze|analyze-partial}" >&2
    exit 2
    ;;
esac
