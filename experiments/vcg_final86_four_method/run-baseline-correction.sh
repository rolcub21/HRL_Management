#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
command_name="${1:-run}"
exec "${project_root}/.venv/bin/python" \
  "${project_root}/run_vcg_final86_baseline_correction.py" \
  "${command_name}"
