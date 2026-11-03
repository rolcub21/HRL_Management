#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python -m \
  experiments.conditioned_vcg.E227_e218_seeds12_guard_free_training_286k_v1.run "$@"

