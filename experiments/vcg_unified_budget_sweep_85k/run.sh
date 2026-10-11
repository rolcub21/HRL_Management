#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results/vcg-unified-budget-sweep-seed14-300ep-85k-development}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

output_name() {
  case "$1" in
    lambda0) echo "lambda0" ;;
    budget10) echo "budget-10" ;;
    budget8) echo "budget-8" ;;
    budget5) echo "budget-5" ;;
    *) echo "unknown sweep arm: $1" >&2; exit 2 ;;
  esac
}

run_arm() {
  local arm="$1"
  local output_dir="${RESULT_ROOT}/$(output_name "${arm}")"
  if [[ -e "${output_dir}" ]]; then
    if "${PYTHON_BIN}" -c "import json,pathlib,sys; p=pathlib.Path(sys.argv[1])/'training-summary.json'; d=json.loads(p.read_text()) if p.is_file() else {}; raise SystemExit(0 if d.get('status') == 'complete' and d.get('sweep_arm',{}).get('name') == sys.argv[2] and d.get('fixed_training_horizon') == 300 else 1)" "${output_dir}" "${arm}"; then
      echo "${arm} is already complete; skipping"
      return
    fi
    echo "${arm} has a partial/non-complete directory; stop and inspect ${output_dir}" >&2
    exit 1
  fi
  "${PYTHON_BIN}" "${PROJECT_ROOT}/train_vcg_unified_budget_sweep.py" \
    --arm "${arm}" \
    --device "${DEVICE}" \
    --output-dir "${output_dir}"
}

compare_sweep() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/compare_vcg_unified_budget_sweep.py" \
    --root "${RESULT_ROOT}" \
    --output "${RESULT_ROOT}/budget-sweep-comparison.json"
}

case "${1:-}" in
  prepare)
    "${PYTHON_BIN}" -c "import json,train_vcg_unified_budget_sweep as s; print(json.dumps(s.plan(), indent=2))"
    ;;
  run-vcg)
    run_arm lambda0
    ;;
  run-budget)
    case "${2:-}" in
      10) run_arm budget10 ;;
      8) run_arm budget8 ;;
      5) run_arm budget5 ;;
      *) echo "budget must be 10, 8, or 5" >&2; exit 2 ;;
    esac
    ;;
  run-all)
    run_arm lambda0
    run_arm budget10
    run_arm budget8
    run_arm budget5
    compare_sweep
    ;;
  compare)
    compare_sweep
    ;;
  *)
    echo "usage: $0 {prepare|run-vcg|run-budget 10|8|5|run-all|compare}" >&2
    exit 2
    ;;
esac
