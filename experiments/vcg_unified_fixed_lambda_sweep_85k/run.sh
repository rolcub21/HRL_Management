#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results/vcg-unified-fixed-lambda-sweep-seed14-300ep-85k-development}"
LAMBDA_ZERO_DIR="${LAMBDA_ZERO_DIR:-${PROJECT_ROOT}/results/vcg-unified-budget-sweep-seed14-300ep-85k-development/lambda0}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

lambda_dir() {
  case "$1" in
    0.05) echo "lambda-0.05" ;;
    0.10|0.1) echo "lambda-0.10" ;;
    0.20|0.2) echo "lambda-0.20" ;;
    0.30|0.3) echo "lambda-0.30" ;;
    *) echo "lambda must be 0.05, 0.10, 0.20, or 0.30" >&2; exit 2 ;;
  esac
}

run_arm() {
  local fixed_lambda="$1"
  local output_dir="${RESULT_ROOT}/$(lambda_dir "${fixed_lambda}")"
  if [[ -e "${output_dir}" ]]; then
    if "${PYTHON_BIN}" -c "import json,math,pathlib,sys; p=pathlib.Path(sys.argv[1])/'training-summary.json'; d=json.loads(p.read_text()) if p.is_file() else {}; got=d.get('fixed_lambda_sweep_arm',{}).get('fixed_lambda_after_warmup'); raise SystemExit(0 if d.get('status') == 'complete' and got is not None and math.isclose(float(got),float(sys.argv[2]),rel_tol=0,abs_tol=1e-12) and d.get('fixed_training_horizon') == 300 else 1)" "${output_dir}" "${fixed_lambda}"; then
      echo "lambda=${fixed_lambda} is already complete; skipping"
      return
    fi
    echo "lambda=${fixed_lambda} has a partial/non-complete directory; inspect ${output_dir}" >&2
    exit 1
  fi
  "${PYTHON_BIN}" "${PROJECT_ROOT}/train_vcg_unified_fixed_lambda_sweep.py" \
    --fixed-lambda "${fixed_lambda}" \
    --device "${DEVICE}" \
    --output-dir "${output_dir}"
}

compare_sweep() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/compare_vcg_unified_fixed_lambda_sweep.py" \
    --root "${RESULT_ROOT}" \
    --lambda-zero-dir "${LAMBDA_ZERO_DIR}" \
    --output "${RESULT_ROOT}/fixed-lambda-comparison.json"
}

case "${1:-}" in
  prepare)
    if [[ ! -f "${LAMBDA_ZERO_DIR}/training-summary.json" ]]; then
      echo "completed lambda-zero parent is missing: ${LAMBDA_ZERO_DIR}" >&2
      exit 1
    fi
    "${PYTHON_BIN}" -c "import json,train_vcg_unified_fixed_lambda_sweep as s; print(json.dumps(s.plan(),indent=2))"
    ;;
  run-lambda)
    run_arm "${2:-}"
    ;;
  run-all)
    run_arm 0.05
    run_arm 0.10
    run_arm 0.20
    run_arm 0.30
    compare_sweep
    ;;
  compare)
    compare_sweep
    ;;
  *)
    echo "usage: $0 {prepare|run-lambda 0.05|0.10|0.20|0.30|run-all|compare}" >&2
    exit 2
    ;;
esac
