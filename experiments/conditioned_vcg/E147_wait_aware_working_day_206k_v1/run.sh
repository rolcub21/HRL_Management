#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python -u -B -m experiments.conditioned_vcg.E147_wait_aware_working_day_206k_v1.run "$@"
