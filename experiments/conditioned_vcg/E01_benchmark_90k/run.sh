#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "${ROOT}/experiments/vcg_conditioned_final_comparison_90k/run.sh" "$@"
