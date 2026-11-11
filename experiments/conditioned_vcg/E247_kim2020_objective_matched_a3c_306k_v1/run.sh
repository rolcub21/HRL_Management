#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python -B -m experiments.conditioned_vcg.E247_kim2020_objective_matched_a3c_306k_v1.run "$@"
