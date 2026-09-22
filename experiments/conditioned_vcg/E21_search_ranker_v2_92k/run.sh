#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cpu}"
RUNNER="$ROOT/experiments/conditioned_vcg/E21_search_ranker_v2_92k/run.py"
RENDERER="$ROOT/experiments/conditioned_vcg/E21_search_ranker_v2_92k/render_paper_comparison.py"

case "${1:-}" in
  prepare)
    "$PYTHON_BIN" "$RUNNER" prepare
    ;;
  run-pilot)
    "$PYTHON_BIN" "$RUNNER" prepare
    "$PYTHON_BIN" "$RUNNER" run --instance-limit 5 --device "$DEVICE"
    "$PYTHON_BIN" "$RUNNER" analyze --allow-partial
    ;;
  run-all)
    "$PYTHON_BIN" "$RUNNER" prepare
    "$PYTHON_BIN" "$RUNNER" run --device "$DEVICE"
    "$PYTHON_BIN" "$RUNNER" analyze
    ;;
  analyze)
    "$PYTHON_BIN" "$RUNNER" analyze
    ;;
  figures)
    "$PYTHON_BIN" "$RENDERER"
    ;;
  *)
    echo "usage: $0 {prepare|run-pilot|run-all|analyze|figures}" >&2
    exit 2
    ;;
esac
