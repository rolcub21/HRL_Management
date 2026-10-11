#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
OUTPUT="${PROJECT_ROOT}/results/vcg-recursive-robust-kernel-reduced-yard/recursive-robust-report.json"

exec "${PYTHON}" "${PROJECT_ROOT}/vcg_recursive_robust_kernel_experiment.py" \
  --output "${OUTPUT}"
