#!/usr/bin/env bash
set -euo pipefail

experiment="experiments/conditioned_vcg/E270_e222_multiseed_prefilled_certification_329k_v1"
output="results/vcg-conditioned-e270-e222-multiseed-prefilled-certification-329k-v1"
gate_output="${output}-gate"
workers="${E270_PREPARE_WORKERS:-6}"

mkdir -p "${output}/logs" "${gate_output}"

MPLCONFIGDIR=/tmp/e270-gate-mpl \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash "${experiment}/run.sh" gate --device cuda 2>&1 | \
tee -a "${gate_output}/run.log"

MPLCONFIGDIR=/tmp/e270-prepare-mpl \
bash "${experiment}/run.sh" prepare --device cpu --workers "${workers}" 2>&1 | \
tee -a "${output}/logs/prepare.log"

pids=()
for seed in 0 1 2; do
  (
    MPLCONFIGDIR="/tmp/e270-seed${seed}-mpl" \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    bash "${experiment}/run.sh" evaluate-vcg \
      --model-seed "${seed}" --device cuda 2>&1 | \
    tee -a "${output}/logs/vcg-seed-${seed}.log"
  ) &
  pids+=("$!")
done

(
  MPLCONFIGDIR=/tmp/e270-dynamic-mpl \
  bash "${experiment}/run.sh" evaluate-dynamic --device cpu 2>&1 | \
  tee -a "${output}/logs/dynamic.log"
) &
pids+=("$!")

cleanup() {
  for pid in "${pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup INT TERM

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
trap - INT TERM

if (( failed != 0 )); then
  echo "E270 stopped because at least one evaluation shard failed." >&2
  exit 1
fi

MPLCONFIGDIR=/tmp/e270-analyze-mpl \
bash "${experiment}/run.sh" analyze --device cpu 2>&1 | \
tee -a "${output}/logs/analyze.log"
