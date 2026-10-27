#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python -B -m \
  experiments.conditioned_vcg.E189_e188_roundoff_repair_248k_v1.run "$@"
