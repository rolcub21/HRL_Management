#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p results/vcg-conditioned-e139-grid-mixture-three-seeds-203k-v1
exec 9>results/vcg-conditioned-e139-grid-mixture-three-seeds-203k-v1/night-queue.lock
flock -n 9 || { echo 'An E139 night queue already holds the lock.'; exit 1; }
trap 'exit 130' INT TERM
e139_script=experiments/conditioned_vcg/E139_grid_mixture_three_seeds_203k_v1/run.sh
e138_script=experiments/conditioned_vcg/E138_residual_grid_mixture_pilot_202k_v1/run.sh
# Wait, never start a second writer against the existing seed-0 checkpoint.
e139_waited=0
while pgrep -f '^[^ ]*python[^ ]* -B -m experiments.conditioned_vcg.E138_residual_grid_mixture_pilot_202k_v1.run train( |$)' >/dev/null; do
    if ((e139_waited >= 86400)); then echo 'Seed 0 still active after 24h; queue stopped.'; exit 1; fi
    echo 'E139 queue: waiting for active E138 seed 0; checking again in 60s.'
    sleep 60
    e139_waited=$((e139_waited+60))
done
e139_failures=0
run_or_record_failure() {
    if "$@"; then return 0; else
        e139_code=$?
        if ((e139_code == 130 || e139_code == 143)); then exit "$e139_code"; fi
        e139_failures=$((e139_failures+1))
        echo "E139 queue: seed stopped (exit $e139_code); preserving results, continuing with next independent seed."
    fi
}
run_or_record_failure bash "$e138_script" train --through-episode 60 --device cuda
run_or_record_failure bash "$e139_script" train --model-seed 1 --through-episode 60 --device cuda
run_or_record_failure bash "$e139_script" train --model-seed 2 --through-episode 60 --device cuda
echo 'E139 queue finished. Inspect all three seeds, including any stopped runs.'
if ((e139_failures > 0)); then exit 1; fi
