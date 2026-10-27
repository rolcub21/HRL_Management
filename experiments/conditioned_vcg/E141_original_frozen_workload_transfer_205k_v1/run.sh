#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/e141-matplotlib}"
exec .venv/bin/python -B -m experiments.conditioned_vcg.E141_original_frozen_workload_transfer_205k_v1.run "$@"
