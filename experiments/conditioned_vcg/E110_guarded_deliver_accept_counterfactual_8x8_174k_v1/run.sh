#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/hrl-matplotlib}"

exec .venv/bin/python -B -m \
  experiments.conditioned_vcg.E110_guarded_deliver_accept_counterfactual_8x8_174k_v1.run "$@"
