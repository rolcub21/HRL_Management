#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python -B -m \
  experiments.conditioned_vcg.E115_release_et_cost_vcg_8x8_179k_v1.run "$@"
