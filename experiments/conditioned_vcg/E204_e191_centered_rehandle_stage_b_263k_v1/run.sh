#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
exec .venv/bin/python -m experiments.conditioned_vcg.E204_e191_centered_rehandle_stage_b_263k_v1.run "$@"
