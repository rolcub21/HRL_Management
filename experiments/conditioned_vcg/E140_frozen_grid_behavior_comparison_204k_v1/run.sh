#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/e140-matplotlib}"
exec .venv/bin/python -B -m experiments.conditioned_vcg.E140_frozen_grid_behavior_comparison_204k_v1.run "$@"
