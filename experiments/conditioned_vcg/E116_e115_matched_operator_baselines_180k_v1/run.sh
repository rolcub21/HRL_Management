#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-e116}"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
exec "$PYTHON_BIN" -B -m experiments.conditioned_vcg.E116_e115_matched_operator_baselines_180k_v1.run "$@"

