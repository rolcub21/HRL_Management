#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
exec .venv/bin/python -B -m experiments.conditioned_vcg.E164_fixed_admission_storage_clock_223k_v1.run "$@"
