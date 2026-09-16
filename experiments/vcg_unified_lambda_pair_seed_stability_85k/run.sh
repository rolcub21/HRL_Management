#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results/vcg-unified-lambda-pair-seed-stability-85k}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

validate_seed() {
  case "$1" in
    15|16|17) ;;
    *) echo "model seed must be 15, 16, or 17" >&2; exit 2 ;;
  esac
}

run_arm() {
  local seed="$1"
  local variant="$2"
  local output_dir="${RESULT_ROOT}/seed-${seed}/${variant}"
  validate_seed "${seed}"
  if [[ -e "${output_dir}" ]]; then
    if "${PYTHON_BIN}" -c "import json, pathlib, sys; p=pathlib.Path(sys.argv[1])/'training-summary.json'; d=json.loads(p.read_text()) if p.is_file() else {}; raise SystemExit(0 if d.get('status') == 'complete' and d.get('variant') == sys.argv[2] else 1)" "${output_dir}" "${variant}"; then
      echo "seed ${seed} ${variant} is already complete; skipping"
      return
    fi
    echo "seed ${seed} ${variant} has a partial/non-complete directory; stop and inspect ${output_dir}" >&2
    exit 1
  fi
  "${PYTHON_BIN}" "${PROJECT_ROOT}/train_vcg_unified_seed_stability.py" \
    --model-seed "${seed}" \
    --variant "${variant}" \
    --device "${DEVICE}" \
    --output-dir "${output_dir}"
}

compare_seed() {
  local seed="$1"
  validate_seed "${seed}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/compare_unified_vcg.py" \
    --vcg-dir "${RESULT_ROOT}/seed-${seed}/vcg" \
    --constrained-dir "${RESULT_ROOT}/seed-${seed}/vcg-handling-constraint" \
    --output "${RESULT_ROOT}/seed-${seed}/comparison.json"
}

run_seed() {
  local seed="$1"
  run_arm "${seed}" vcg
  run_arm "${seed}" vcg-handling-constraint
  compare_seed "${seed}"
}

case "${1:-}" in
  prepare)
    "${PYTHON_BIN}" -c "import train_vcg_unified_seed_stability as s; s._validate_profiles(); print({k: v.to_dict() for k, v in s.SEED_PROFILES.items()})"
    ;;
  run-vcg)
    run_arm "${2:-}" vcg
    ;;
  run-constrained)
    run_arm "${2:-}" vcg-handling-constraint
    ;;
  compare-seed)
    compare_seed "${2:-}"
    ;;
  run-seed)
    run_seed "${2:-}"
    ;;
  run-all)
    for seed in 15 16 17; do
      run_seed "${seed}"
    done
    "${PYTHON_BIN}" "${PROJECT_ROOT}/compare_unified_vcg_seed_stability.py" \
      --root "${RESULT_ROOT}" \
      --output "${RESULT_ROOT}/stability-comparison.json"
    ;;
  compare-all)
    "${PYTHON_BIN}" "${PROJECT_ROOT}/compare_unified_vcg_seed_stability.py" \
      --root "${RESULT_ROOT}" \
      --output "${RESULT_ROOT}/stability-comparison.json"
    ;;
  *)
    echo "usage: $0 {prepare|run-vcg SEED|run-constrained SEED|compare-seed SEED|run-seed SEED|run-all|compare-all}" >&2
    exit 2
    ;;
esac
