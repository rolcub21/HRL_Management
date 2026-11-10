#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
exec .venv/bin/python -B -m \
  experiments.conditioned_vcg.E262_e255_e258_simple_dial_curve_5x5_321k_v1.run "$@"
