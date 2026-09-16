#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/vcg-d12-relocation-family-integrated-confirmation-95k-v2}"
MODULE="experiments.conditioned_vcg.development.D12_relocation_family_certification.confirmation"
COMMAND="${1:-}"
common=(--output-dir "${OUTPUT_DIR}")

case "${COMMAND}" in
  authenticate-inputs)
    cd "${ROOT}"
    "${PYTHON_BIN}" -m "${MODULE}" authenticate-inputs
    ;;
  prepare)
    cd "${ROOT}"
    "${PYTHON_BIN}" -m "${MODULE}" prepare "${common[@]}"
    ;;
  run-medium)
    cd "${ROOT}"
    "${PYTHON_BIN}" -m "${MODULE}" run-medium "${common[@]}"
    "${PYTHON_BIN}" -m "${MODULE}" analyze "${common[@]}" --allow-partial
    ;;
  run-high)
    cd "${ROOT}"
    "${PYTHON_BIN}" -m "${MODULE}" run-high "${common[@]}"
    "${PYTHON_BIN}" -m "${MODULE}" analyze "${common[@]}" --allow-partial
    ;;
  run-both)
    cd "${ROOT}"
    "${PYTHON_BIN}" -m "${MODULE}" run-both "${common[@]}"
    "${PYTHON_BIN}" -m "${MODULE}" analyze "${common[@]}"
    ;;
  analyze)
    cd "${ROOT}"
    "${PYTHON_BIN}" -m "${MODULE}" analyze "${common[@]}" "${@:2}"
    ;;
  *)
    echo "usage: $0 {authenticate-inputs|prepare|run-medium|run-high|run-both|analyze}" >&2
    exit 2
    ;;
esac
