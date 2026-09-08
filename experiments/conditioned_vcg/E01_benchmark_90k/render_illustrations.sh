#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/vcg-conditioned-final-comparison-90k-cpu-v3/illustrations}"
TRACE="${ROOT}/experiments/conditioned_vcg/E01_benchmark_90k/trace_current_method_illustrations.py"
RENDERER="${ROOT}/render_vcg_unified_policy_filmstrip.py"

"${PYTHON_BIN}" "${TRACE}" --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${RENDERER}" \
  --trace "${OUTPUT_DIR}/e1-current-storage-placement-trace.json" \
  --output-dir "${OUTPUT_DIR}" \
  --stem "e1-current-storage-placement-filmstrip" \
  --no-video

"${PYTHON_BIN}" "${RENDERER}" \
  --trace "${OUTPUT_DIR}/e1-current-rehandle-vs-delivery-trace.json" \
  --output-dir "${OUTPUT_DIR}" \
  --stem "e1-current-rehandle-vs-delivery-filmstrip" \
  --no-video
