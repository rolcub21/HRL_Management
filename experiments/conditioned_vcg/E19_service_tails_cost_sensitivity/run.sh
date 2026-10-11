#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/vcg-e19-matplotlib}"
OUTPUT="${OUTPUT:-${ROOT}/results/vcg-conditioned-e19-service-cost}"
PROGRAM="${ROOT}/experiments/conditioned_vcg/E19_service_tails_cost_sensitivity/analyze.py"

case "${1:-}" in
  prepare)
    "${PYTHON_BIN}" "${PROGRAM}" prepare --output "${OUTPUT}"
    ;;
  analyze)
    "${PYTHON_BIN}" "${PROGRAM}" analyze --output "${OUTPUT}"
    ;;
  run)
    "${PYTHON_BIN}" "${PROGRAM}" run --output "${OUTPUT}"
    ;;
  *)
    echo "usage: $0 {prepare|analyze|run}" >&2
    exit 2
    ;;
esac
