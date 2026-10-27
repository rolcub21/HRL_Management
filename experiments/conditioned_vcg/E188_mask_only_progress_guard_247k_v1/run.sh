#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python -B -m \
  experiments.conditioned_vcg.E188_mask_only_progress_guard_247k_v1.run "$@"

