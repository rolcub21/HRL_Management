#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/e137-matplotlib}"
exec .venv/bin/python -B -m experiments.conditioned_vcg.E137_grid_rollout_readiness_201k_v1.run "$@"
