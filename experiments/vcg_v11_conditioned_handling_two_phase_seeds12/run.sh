#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/vcg-v1-1-conditioned-handling-seeds12-two-phase-development}"

runner() {
  "$PYTHON_BIN" "$ROOT/run_vcg_v11_conditioned_handling_two_phase_seeds12.py" \
    "$@" \
    --project-root "$ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --device "$DEVICE"
}

run_seed() {
  local seed="$1"
  local seed_root="$OUTPUT_ROOT/seed-$seed"
  local resume=()
  if [[ -f "$seed_root/terminal.pth" ]]; then
    echo "Seed $seed fixed terminal already exists."
    return 0
  fi
  if [[ -f "$seed_root/latest.pth" ]]; then
    resume=(--resume-existing)
  fi
  runner "train-seed$seed" "${resume[@]}"
}

case "${1:-}" in
  prepare)
    runner prepare
    ;;
  run-seed1)
    runner prepare
    run_seed 1
    ;;
  run-seed2)
    runner prepare
    run_seed 2
    ;;
  analyze)
    runner analyze
    ;;
  run-all)
    runner prepare
    run_seed 1
    run_seed 2
    runner analyze
    ;;
  *)
    echo "Usage: $0 {prepare|run-seed1|run-seed2|analyze|run-all}" >&2
    exit 2
    ;;
esac
