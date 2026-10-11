#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
OUTPUT="${OUTPUT:-${ROOT}/results/vcg-conditioned-e20-constant-type-penalty-92k}"
PROGRAM="${ROOT}/experiments/conditioned_vcg/E20_constant_type_penalty_92k/run.py"
DEVICE="${DEVICE:-cpu}"

case "${1:-}" in
  prepare)
    "${PYTHON_BIN}" "${PROGRAM}" calibrate --output "${OUTPUT}" --device "${DEVICE}"
    "${PYTHON_BIN}" "${PROGRAM}" prepare --output "${OUTPUT}"
    ;;
  run-pilot)
    "${PYTHON_BIN}" "${PROGRAM}" run-pilot --output "${OUTPUT}" --device "${DEVICE}"
    "${PYTHON_BIN}" "${PROGRAM}" analyze-pilot --output "${OUTPUT}"
    ;;
  run-confirmation)
    "${PYTHON_BIN}" "${PROGRAM}" run-confirmation --output "${OUTPUT}" --device "${DEVICE}"
    "${PYTHON_BIN}" "${PROGRAM}" analyze --output "${OUTPUT}"
    ;;
  analyze)
    "${PYTHON_BIN}" "${PROGRAM}" analyze --output "${OUTPUT}"
    ;;
  run-all)
    "${PYTHON_BIN}" "${PROGRAM}" calibrate --output "${OUTPUT}" --device "${DEVICE}"
    "${PYTHON_BIN}" "${PROGRAM}" prepare --output "${OUTPUT}"
    "${PYTHON_BIN}" "${PROGRAM}" run-confirmation --output "${OUTPUT}" --device "${DEVICE}"
    "${PYTHON_BIN}" "${PROGRAM}" analyze --output "${OUTPUT}"
    ;;
  *)
    echo "usage: $0 {prepare|run-pilot|run-confirmation|analyze|run-all}" >&2
    exit 2
    ;;
esac
