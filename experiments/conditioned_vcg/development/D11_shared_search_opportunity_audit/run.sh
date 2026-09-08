#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/vcg-d11-shared-search-opportunity-95k}"
PROGRAM="${ROOT}/experiments/conditioned_vcg/development/D11_shared_search_opportunity_audit/program.py"
AUDIT="${ROOT}/experiments/conditioned_vcg/development/D11_shared_search_opportunity_audit/audit.py"
COMMAND="${1:-}"
common=(--output-dir "${OUTPUT_DIR}")

case "${COMMAND}" in
  prepare)
    "${PYTHON_BIN}" "${PROGRAM}" prepare "${common[@]}"
    ;;
  sample)
    "${PYTHON_BIN}" "${PROGRAM}" sample "${common[@]}"
    ;;
  run)
    "${PYTHON_BIN}" "${PROGRAM}" prepare "${common[@]}"
    "${PYTHON_BIN}" "${AUDIT}" run "${common[@]}"
    "${PYTHON_BIN}" "${AUDIT}" analyze "${common[@]}" --allow-partial
    ;;
  analyze)
    "${PYTHON_BIN}" "${AUDIT}" analyze "${common[@]}" "${@:2}"
    ;;
  authenticate)
    "${PYTHON_BIN}" "${PROGRAM}" authenticate "${common[@]}"
    "${PYTHON_BIN}" "${AUDIT}" authenticate "${common[@]}"
    ;;
  *)
    echo "usage: $0 {prepare|sample|run|analyze|authenticate}" >&2
    exit 2
    ;;
esac
