#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$root"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec .venv/bin/python -B -m experiments.conditioned_vcg.E84_unsafe_legal_5x5_pilot_149k_v1.run "$@"
