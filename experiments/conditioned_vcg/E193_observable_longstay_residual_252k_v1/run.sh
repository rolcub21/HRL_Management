#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
exec .venv/bin/python -m \
  experiments.conditioned_vcg.E193_observable_longstay_residual_252k_v1.run "$@"
