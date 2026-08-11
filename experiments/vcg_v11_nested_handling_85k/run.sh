#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${PROJECT_ROOT}/run_vcg_v11_nested_handling_pilot.py"

COMMAND="${1:-run-all}"
export PYTHONPATH="${PROJECT_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

exec "${PROJECT_ROOT}/.venv/bin/python" -u "${RUNNER}" "${COMMAND}" \
  --project-root "${PROJECT_ROOT}" \
  --device cuda
