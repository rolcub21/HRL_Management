#!/usr/bin/env bash
set -euo pipefail
.venv/bin/python -m experiments.conditioned_vcg.E180_wait_cost_unmasked_pilot_239k_v1.run "$@"
