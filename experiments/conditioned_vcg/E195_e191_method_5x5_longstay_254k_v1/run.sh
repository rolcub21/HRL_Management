#!/usr/bin/env bash
set -euo pipefail

exec .venv/bin/python -B -m \
  experiments.conditioned_vcg.E195_e191_method_5x5_longstay_254k_v1.run "$@"
