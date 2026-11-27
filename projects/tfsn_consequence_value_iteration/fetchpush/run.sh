#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
PYTHON="${PYTHON:-/home/ai_diagnosis/SG/.venv/bin/python}"
FETCH_VENDOR="/home/ai_diagnosis/SG/sn/third_party/gymnasium-robotics-fetch-v4"
export PYTHONPATH="${FETCH_VENDOR}:/home/ai_diagnosis/SG/sn:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-tfsn-fetchpush}"

case "${1:-}" in
  smoke)
    exec "${PYTHON}" -m projects.tfsn_consequence_value_iteration.fetchpush.point1_transfer smoke --device cpu
    ;;
  prepare-full)
    exec "${PYTHON}" -m projects.tfsn_consequence_value_iteration.fetchpush.point1_transfer prepare --profile full --device "${DEVICE:-cuda}"
    ;;
  evaluate-development)
    exec "${PYTHON}" -m projects.tfsn_consequence_value_iteration.fetchpush.point1_transfer evaluate --profile full --device "${DEVICE:-cuda}"
    ;;
  freeze-development-goals)
    exec "${PYTHON}" -m projects.tfsn_consequence_value_iteration.fetchpush.point1_transfer freeze-goals --profile full
    ;;
  audit)
    exec "${PYTHON}" -m projects.tfsn_consequence_value_iteration.fetchpush.point1_transfer audit
    ;;
  *)
    echo "usage: $0 {audit|smoke|freeze-development-goals|prepare-full|evaluate-development}" >&2
    exit 2
    ;;
esac
