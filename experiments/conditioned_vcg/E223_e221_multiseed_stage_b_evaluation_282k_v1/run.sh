#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
exec .venv/bin/python -m \
  experiments.conditioned_vcg.E223_e221_multiseed_stage_b_evaluation_282k_v1.run "$@"

