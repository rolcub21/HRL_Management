#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
exec .venv/bin/python -m \
  experiments.conditioned_vcg.E226_e215_seed0_guard_free_training_285k_v1.run "$@"

