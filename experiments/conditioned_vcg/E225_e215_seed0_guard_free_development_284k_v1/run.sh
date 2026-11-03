#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
exec .venv/bin/python -m \
  experiments.conditioned_vcg.E225_e215_seed0_guard_free_development_284k_v1.run "$@"

