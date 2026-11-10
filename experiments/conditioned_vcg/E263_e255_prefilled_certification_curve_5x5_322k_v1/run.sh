#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
exec .venv/bin/python -B -m \
  experiments.conditioned_vcg.E263_e255_prefilled_certification_curve_5x5_322k_v1.run "$@"
