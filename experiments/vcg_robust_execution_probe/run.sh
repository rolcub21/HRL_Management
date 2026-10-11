#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
OUTPUT="${PROJECT_ROOT}/results/vcg-robust-execution-probe/robust-execution-report.json"

exec "${PYTHON}" "${PROJECT_ROOT}/vcg_robust_execution_probe.py" \
  --output "${OUTPUT}"
