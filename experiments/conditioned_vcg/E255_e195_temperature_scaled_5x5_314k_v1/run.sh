#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python -m experiments.conditioned_vcg.E255_e195_temperature_scaled_5x5_314k_v1.run "$@"
