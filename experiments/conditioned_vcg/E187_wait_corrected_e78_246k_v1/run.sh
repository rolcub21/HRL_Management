#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python -B -m \
  experiments.conditioned_vcg.E187_wait_corrected_e78_246k_v1.run "$@"

