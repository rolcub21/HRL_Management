#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
exec .venv/bin/python -m experiments.conditioned_vcg.E213_post_placement_continuation_audit_272k_v1.run "$@"
