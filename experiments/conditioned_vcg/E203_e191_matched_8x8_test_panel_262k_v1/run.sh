#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
exec .venv/bin/python -m experiments.conditioned_vcg.E203_e191_matched_8x8_test_panel_262k_v1.run "$@"

