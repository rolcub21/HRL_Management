#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python -B -m \
  experiments.conditioned_vcg.E107_queue_context_stage_a_8x8_171k_v1.run "$@"
