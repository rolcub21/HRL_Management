#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ai_diagnosis/HRL_Management"
COMMAND="${1:-prepare}"
shift || true

cd "$PROJECT_ROOT"

case "$COMMAND" in
  prepare)
    if (($# != 0)); then
      echo "usage: $0 prepare" >&2
      exit 2
    fi
    ;;
  run_seed)
    if (($# != 1)) || [[ "$1" != "11" && "$1" != "12" && "$1" != "13" ]]; then
      echo "usage: $0 run_seed {11|12|13}" >&2
      exit 2
    fi
    ;;
  analyze)
    if (($# != 0)); then
      echo "usage: $0 analyze" >&2
      exit 2
    fi
    ;;
  *)
    echo "usage: $0 {prepare|run_seed {11|12|13}|analyze}" >&2
    exit 2
    ;;
esac

PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  nice -n 10 .venv/bin/python -u vcg_v2_3_seed_stability.py "$COMMAND" "$@"
