#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
exec .venv/bin/python -m experiments.conditioned_vcg.E205_e204_paired_tradeoff_development_264k_v1.run "$@"
