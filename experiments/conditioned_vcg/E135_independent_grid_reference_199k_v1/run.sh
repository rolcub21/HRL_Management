#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python -B -m experiments.conditioned_vcg.E135_independent_grid_reference_199k_v1.run "$@"
