#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
exec bash "${ROOT}/experiments/vcg_v11_conditioned_handling_seed1_convergence_continuation/run.sh" "$@"
