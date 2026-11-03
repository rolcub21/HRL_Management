#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
exec .venv/bin/python -m experiments.conditioned_vcg.E222_e215_e218_fresh_matched_8x8_panel_281k_v1.run "$@"
