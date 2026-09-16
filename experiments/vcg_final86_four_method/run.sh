#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${project_root}/.venv/bin/python"
runner="${project_root}/run_vcg_final86_four_method.py"

command_name="${1:-prepare}"
case "${command_name}" in
  prepare)
    exec "${python_bin}" "${runner}" prepare
    ;;
  open-panel)
    confirmation="${2:-}"
    exec "${python_bin}" "${runner}" open-panel --confirm "${confirmation}"
    ;;
  run)
    exec "${python_bin}" "${runner}" evaluate --method all --execute
    ;;
  run-method)
    method="${2:?usage: run.sh run-method METHOD}"
    exec "${python_bin}" "${runner}" evaluate --method "${method}" --execute
    ;;
  inspect)
    method="${2:-all}"
    exec "${python_bin}" "${runner}" evaluate --method "${method}"
    ;;
  analyze)
    exec "${python_bin}" "${runner}" analyze
    ;;
  status)
    exec "${python_bin}" "${runner}" status
    ;;
  *)
    echo "usage: run.sh {prepare|open-panel TOKEN|run|run-method METHOD|inspect [METHOD]|analyze|status}" >&2
    exit 2
    ;;
esac
