#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results/vcg-unified-lambda-pair-seed14-85k-development}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

run_arm() {
  local variant="$1"
  local output_name="$2"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/train_vcg_unified.py" \
    --variant "${variant}" \
    --device "${DEVICE}" \
    --output-dir "${RESULT_ROOT}/${output_name}"
}

case "${1:-}" in
  prepare)
    "${PYTHON_BIN}" -c "import train_vcg_unified, compare_unified_vcg; print('unified VCG imports: OK')"
    ;;
  run-vcg)
    run_arm vcg vcg
    ;;
  run-constrained)
    run_arm vcg-handling-constraint vcg-handling-constraint
    ;;
  run)
    run_arm vcg vcg
    run_arm vcg-handling-constraint vcg-handling-constraint
    ;;
  compare)
    "${PYTHON_BIN}" "${PROJECT_ROOT}/compare_unified_vcg.py" \
      --vcg-dir "${RESULT_ROOT}/vcg" \
      --constrained-dir "${RESULT_ROOT}/vcg-handling-constraint" \
      --output "${RESULT_ROOT}/comparison.json"
    ;;
  *)
    echo "usage: $0 {prepare|run-vcg|run-constrained|run|compare}" >&2
    exit 2
    ;;
esac
