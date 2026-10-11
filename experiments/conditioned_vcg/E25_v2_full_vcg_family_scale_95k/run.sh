#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
MODULE="experiments.conditioned_vcg.E25_v2_full_vcg_family_scale_95k.run"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/vcg-conditioned-e25-v2-full-vcg-family-scale-95k-v2}"

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
  run-lambda)
    run_module run-lambda --preference-lambda "${1:?lambda}"
    ;;
  run-one)
    run_module run-one --preference-lambda "${1:?lambda}" \
      --scenario "${2:?scenario}" --seed "${3:?seed}"
    ;;
  *)
    echo "usage: $0 {prepare|run-pilot|run-all|run-lambda LAMBDA|run-one LAMBDA SCENARIO SEED|analyze|analyze-partial}" >&2
    exit 2
    ;;
esac
