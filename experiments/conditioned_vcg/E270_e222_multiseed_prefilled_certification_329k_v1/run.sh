#!/usr/bin/env bash
set -euo pipefail

python -B -m experiments.conditioned_vcg.E270_e222_multiseed_prefilled_certification_329k_v1.run "$@"
