#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
exec .venv/bin/python -B -m \
  experiments.conditioned_vcg.E243_e195_expanded_prefilled_curve_302k_v1.run "$@"
