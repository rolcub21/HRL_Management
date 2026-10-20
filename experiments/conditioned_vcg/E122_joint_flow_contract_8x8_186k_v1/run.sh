#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-e122}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
exec "$PYTHON_BIN" -B -m experiments.conditioned_vcg.E122_joint_flow_contract_8x8_186k_v1.run "$@"
