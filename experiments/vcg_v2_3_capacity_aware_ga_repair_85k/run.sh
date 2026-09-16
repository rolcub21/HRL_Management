#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ai_diagnosis/HRL_Management"
RUN_MODE="${1:-prepare}"
shift || true

cd "$PROJECT_ROOT"

COMMON_ARGS=(
  --v2-3-source-dir results/vcg-constrained-v2-3-gamma1-ablation-seed10-200ep
  --v1-1-control-dir results/vcg-dense-v1-1-v2-2-panel-control-12instance
  --v2-2-source-dir results/vcg-constrained-v2-2-development-seed10-200ep
  --matched-source-dir results/vcg-v2-3-matched-baselines-85k
  --output-dir results/vcg-v2-3-capacity-aware-ga-repair-85k
  --device cpu
)

case "$RUN_MODE" in
  prepare)
    EXTRA_ARGS=()
    ;;
  run)
    # Safe both after `prepare` and from a fresh output directory: the runner
    # accepts only an exact matching contract/ledger set.
    EXTRA_ARGS=(--execute-repaired-ga --resume-existing)
    ;;
  resume)
    EXTRA_ARGS=(--execute-repaired-ga --resume-existing)
    ;;
  *)
    echo "usage: $0 {prepare|run|resume} [additional runner arguments]" >&2
    exit 2
    ;;
esac

PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -u compare_vcg_v2_3_capacity_aware_ga_repair.py \
  "${COMMON_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
