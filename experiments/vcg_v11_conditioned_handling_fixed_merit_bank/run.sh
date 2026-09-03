#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-cpu}"

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" diagnose_vcg_v11_conditioned_handling_fixed_bank.py \
  --device "${DEVICE}"
