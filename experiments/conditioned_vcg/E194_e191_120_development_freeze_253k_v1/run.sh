#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
exec .venv/bin/python -m \
  experiments.conditioned_vcg.E194_e191_120_development_freeze_253k_v1.run "$@"
