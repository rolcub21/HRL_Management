#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
exec .venv/bin/python -m experiments.conditioned_vcg.E212_e204_heavy_band_mechanism_audit_271k_v1.run "$@"
