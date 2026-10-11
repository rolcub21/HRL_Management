#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${project_root}/.venv/bin/python"
program="${project_root}/vcg_v2_3_kim2020_supplement_evaluation.py"
training_root="${project_root}/results/vcg-v2-3-kim2020-supplement-85k"
output_root="${project_root}/results/vcg-v2-3-kim2020-supplement-evaluation-85k"

usage() {
  echo "usage: $0 {prepare|evaluate_seed 0|1|2|analyze|validate}" >&2
  exit 2
}

[[ $# -ge 1 ]] || usage
command_name="$1"
shift

common=(
  --project-root "${project_root}"
  --training-root "${training_root}"
  --output-root "${output_root}"
)

case "${command_name}" in
  prepare)
    [[ $# -eq 0 ]] || usage
    exec "${python_bin}" "${program}" prepare "${common[@]}"
    ;;
  evaluate_seed)
    [[ $# -eq 1 ]] || usage
    exec "${python_bin}" "${program}" evaluate-seed "${common[@]}" --seed "$1"
    ;;
  analyze)
    [[ $# -eq 0 ]] || usage
    exec "${python_bin}" "${program}" analyze "${common[@]}"
    ;;
  validate)
    [[ $# -eq 0 ]] || usage
    exec "${python_bin}" "${program}" validate "${common[@]}"
    ;;
  *) usage ;;
esac
