#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/vcg-conditioned-e14-threshold-refinement-95k}"
MODULE="experiments.conditioned_vcg.E14_certification_scalability_95k.threshold_refinement"
COMMAND="${1:-}"
budgets=(32 64)
scenarios=(
  size_5x5_occ_low size_5x5_occ_medium size_5x5_occ_high
  size_8x8_occ_low size_8x8_occ_medium size_8x8_occ_high
  size_10x10_occ_low size_10x10_occ_medium size_10x10_occ_high
)
common=(--output-dir "${OUTPUT_DIR}")

run_one() {
  "${PYTHON_BIN}" -m "${MODULE}" run-one "${common[@]}" \
    --max-nodes "$1" --scenario "$2"
}

cd "${ROOT}"
case "${COMMAND}" in
  prepare)
    "${PYTHON_BIN}" -m "${MODULE}" prepare "${common[@]}"
    ;;
  run-one)
    run_one "${2:?max-nodes required}" "${3:?scenario required}"
    ;;
  run-all)
    for budget in "${budgets[@]}"; do
      for scenario in "${scenarios[@]}"; do
        run_one "${budget}" "${scenario}"
      done
    done
    "${PYTHON_BIN}" -m "${MODULE}" analyze "${common[@]}"
    ;;
  analyze)
    "${PYTHON_BIN}" -m "${MODULE}" analyze "${common[@]}" "${@:2}"
    ;;
  *)
    echo "usage: $0 {prepare|run-one MAX_NODES SCENARIO|run-all|analyze}" >&2
    exit 2
    ;;
esac
