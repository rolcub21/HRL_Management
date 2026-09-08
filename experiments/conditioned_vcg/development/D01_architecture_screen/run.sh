#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
exec bash "${ROOT}/experiments/vcg_preference_conditioned_architecture_screen_85k/run.sh" "$@"
