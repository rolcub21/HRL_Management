#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
RUNNER="${ROOT}/experiments/conditioned_vcg/E03_certification_ablation_90k/scale_stress.py"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/vcg-conditioned-e03-scale-stress-97k}"
COMMAND="${1:-}"

case "${COMMAND}" in
  prepare)
    "${PYTHON_BIN}" "${RUNNER}" prepare --output-dir "${OUTPUT_DIR}"
    ;;
  run-pilot)
    "${PYTHON_BIN}" "${RUNNER}" run-pilot --output-dir "${OUTPUT_DIR}"
    ;;
  run-pair)
    if [[ $# -ne 3 ]]; then
      echo "usage: $0 run-pair {stress_8x8_saturation_burst|stress_10x10_saturation_burst} {97100..97109}" >&2
      exit 2
    fi
    "${PYTHON_BIN}" "${RUNNER}" run-pair --output-dir "${OUTPUT_DIR}" --scenario "$2" --seed "$3"
    ;;
  run-scenario)
    if [[ $# -ne 2 ]]; then
      echo "usage: $0 run-scenario {stress_8x8_saturation_burst|stress_10x10_saturation_burst}" >&2
      exit 2
    fi
    "${PYTHON_BIN}" "${RUNNER}" run-scenario --output-dir "${OUTPUT_DIR}" --scenario "$2"
    ;;
  run-all)
    "${PYTHON_BIN}" "${RUNNER}" run-all --output-dir "${OUTPUT_DIR}"
    ;;
  analyze)
    "${PYTHON_BIN}" "${RUNNER}" analyze --output-dir "${OUTPUT_DIR}"
    ;;
  analyze-partial)
    "${PYTHON_BIN}" "${RUNNER}" analyze --output-dir "${OUTPUT_DIR}" --allow-partial
    ;;
  *)
    echo "usage: $0 {prepare|run-pilot|run-pair SCENARIO SEED|run-scenario SCENARIO|run-all|analyze|analyze-partial}" >&2
    exit 2
    ;;
esac
