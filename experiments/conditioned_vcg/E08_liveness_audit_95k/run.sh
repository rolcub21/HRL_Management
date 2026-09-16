#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
OUTPUT="${OUTPUT:-${ROOT}/results/vcg-conditioned-e08-liveness-audit-95k}"
ANALYZE="${ROOT}/experiments/conditioned_vcg/E08_liveness_audit_95k/analyze_existing.py"
PILOT="${ROOT}/experiments/conditioned_vcg/E08_liveness_audit_95k/instrumented_pilot.py"
SUMMARY="${ROOT}/experiments/conditioned_vcg/E08_liveness_audit_95k/summarize_pilot.py"
COMMAND="${1:-}"

case "${COMMAND}" in
  analyze-existing)
    "${PYTHON_BIN}" "${ANALYZE}" --output "${OUTPUT}"
    ;;
  prepare)
    "${PYTHON_BIN}" "${ANALYZE}" --output "${OUTPUT}"
    "${PYTHON_BIN}" "${PILOT}" prepare --output "${OUTPUT}"
    ;;
  run-pilot)
    "${PYTHON_BIN}" "${PILOT}" run --output "${OUTPUT}"
    "${PYTHON_BIN}" "${SUMMARY}" --output "${OUTPUT}"
    ;;
  summarize-pilot)
    "${PYTHON_BIN}" "${SUMMARY}" --output "${OUTPUT}"
    ;;
  run-all)
    "${PYTHON_BIN}" "${ANALYZE}" --output "${OUTPUT}"
    "${PYTHON_BIN}" "${PILOT}" run --output "${OUTPUT}"
    "${PYTHON_BIN}" "${SUMMARY}" --output "${OUTPUT}"
    ;;
  status)
    "${PYTHON_BIN}" "${PILOT}" analyze --output "${OUTPUT}"
    ;;
  *)
    echo "usage: $0 {analyze-existing|prepare|run-pilot|summarize-pilot|run-all|status}" >&2
    exit 2
    ;;
esac
