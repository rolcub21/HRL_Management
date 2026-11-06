#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
exec .venv/bin/python -m experiments.conditioned_vcg.E244_e215_scale_matched_10x10_stage_a_303k_v1.run "$@"

