#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/vcg-conditioned-e14-certificate-reuse-95k}"
PROGRAM="${ROOT}/experiments/conditioned_vcg/E14_certification_scalability_95k/program.py"
CAPTURE="${ROOT}/experiments/conditioned_vcg/E14_certification_scalability_95k/capture.py"
REPLAY="${ROOT}/experiments/conditioned_vcg/E14_certification_scalability_95k/replay.py"
CONFIRM="${ROOT}/experiments/conditioned_vcg/E14_certification_scalability_95k/confirm.py"
COMMAND="${1:-}"
common=(--output-dir "${OUTPUT_DIR}")

prepare() {
  "${PYTHON_BIN}" "${PROGRAM}" prepare "${common[@]}"
}

case "${COMMAND}" in
  prepare)
    prepare
    ;;
  capture-medium|capture-high|capture-both)
    prepare
    "${PYTHON_BIN}" "${CAPTURE}" "${COMMAND}" "${common[@]}"
    ;;
  replay-medium|replay-high|replay-both)
    prepare
    "${PYTHON_BIN}" "${REPLAY}" "${COMMAND}" "${common[@]}" "${@:2}"
    ;;
  analyze-medium|analyze-high|analyze-both)
    "${PYTHON_BIN}" "${REPLAY}" "${COMMAND}" "${common[@]}"
    ;;
  confirm-timing-medium|confirm-path-medium|confirm-suffix-medium|confirm-medium|confirm-timing-high|confirm-path-high|confirm-suffix-high|confirm-high)
    prepare
    "${PYTHON_BIN}" "${CONFIRM}" "${COMMAND}" "${common[@]}"
    ;;
  analyze-online)
    "${PYTHON_BIN}" "${CONFIRM}" analyze "${common[@]}"
    ;;
  authenticate)
    "${PYTHON_BIN}" "${PROGRAM}" authenticate "${common[@]}"
    "${PYTHON_BIN}" "${CAPTURE}" authenticate "${common[@]}"
    ;;
  *)
    echo "usage: $0 {prepare|capture-medium|replay-medium|analyze-medium|capture-high|replay-high|analyze-high|capture-both|replay-both|analyze-both|confirm-timing-medium|confirm-path-medium|confirm-suffix-medium|confirm-medium|confirm-timing-high|confirm-path-high|confirm-suffix-high|confirm-high|analyze-online|authenticate}" >&2
    exit 2
    ;;
esac
