#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
exec .venv/bin/python -m \
  experiments.conditioned_vcg.E202_e191_multiseed_development_freeze_261k_v1.run "$@"
