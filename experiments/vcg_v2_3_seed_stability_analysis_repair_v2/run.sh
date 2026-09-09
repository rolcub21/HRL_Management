#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"
export PYTHONDONTWRITEBYTECODE=1

python_bin="${project_root}/.venv/bin/python"
"${python_bin}" vcg_v2_3_seed_stability_analysis_repair_v2.py prepare
"${python_bin}" vcg_v2_3_seed_stability_analysis_repair_v2.py analyze
"${python_bin}" vcg_v2_3_seed_stability_analysis_repair_v2.py validate
