#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/e136-matplotlib}"
exec .venv/bin/python -B -m experiments.conditioned_vcg.E136_grid_interface_gate_200k_v1.run "$@"
