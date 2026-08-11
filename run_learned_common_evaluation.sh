#!/usr/bin/env bash
set -euo pipefail

# Run from any directory after the project's .venv has been activated.
PYTHON="/home/jupyter/Roland Cubahiro/hrl-project/.venv/bin/python"
EVAL="/home/bukuj/hrl-eval/common_evaluate.py"
export PYTHONPATH="/home/bukuj/hrl-eval:/home/bukuj/hrl-eval/example"
OUT="${1:-/home/bukuj/hrl-eval/learned_common_eval_$(date +%Y%m%d-%H%M%S).csv}"
GA_ASSIGNMENT="/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/ga_20260311-093907-20 Gen/best_assignment.pkl"

while IFS='|' read -r METHOD LAM MU TRAIN_SEED CHECKPOINT; do
  for EVAL_SEED in 0 1 2 3 4; do
    ARGS=(--method "$METHOD" --lambda "$LAM" --mu "$MU"
      --train-seed "$TRAIN_SEED" --eval-seed "$EVAL_SEED"
      --checkpoint "$CHECKPOINT" --output "$OUT")
    if [[ "$METHOD" == "ga" ]]; then
      ARGS+=(--ga-assignment "$GA_ASSIGNMENT")
    fi
    "$PYTHON" "$EVAL" "${ARGS[@]}"
  done
done <<'RUNS'
hrl|0.2|20|0|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/HRL_sensitivity_20260314-182749/models/2026-03-14T20-26-34-971069_HRL.pth
hrl|0.2|20|1|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/HRL_sensitivity_20260314-182749/models/2026-03-14T22-26-09-908413_HRL.pth
hrl|0.2|20|2|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/HRL_sensitivity_20260314-182749/models/2026-03-15T00-26-27-963168_HRL.pth
hrl|0.2|20|3|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/learned_selector_diagpairs_seeds3-4_20260320-194522/models/2026-03-20T22-17-38-555831_HRL.pth
hrl|0.2|20|4|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/learned_selector_diagpairs_seeds3-4_20260320-194522/models/2026-03-21T00-54-53-363815_HRL.pth
hrl|0.5|50|0|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/HRL_sensitivity_20260314-182749/models/2026-03-15T22-10-05-517715_HRL.pth
hrl|0.5|50|1|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/HRL_sensitivity_20260314-182749/models/2026-03-16T00-29-26-601078_HRL.pth
hrl|0.5|50|2|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/HRL_sensitivity_20260314-182749/models/2026-03-16T02-47-27-326619_HRL.pth
hrl|0.5|50|3|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/learned_selector_diagpairs_seeds3-4_20260320-194522/models/2026-03-21T03-57-41-176879_HRL.pth
hrl|0.5|50|4|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/learned_selector_diagpairs_seeds3-4_20260320-194522/models/2026-03-21T06-58-05-344461_HRL.pth
hrl|1.0|80|0|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/HRL_sensitivity_20260314-182749/models/2026-03-17T01-57-21-964192_HRL.pth
hrl|1.0|80|1|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/HRL_sensitivity_20260314-182749/models/2026-03-17T04-23-05-293297_HRL.pth
hrl|1.0|80|2|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/HRL_sensitivity_20260314-182749/models/2026-03-17T06-47-32-809199_HRL.pth
hrl|1.0|80|3|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/learned_selector_diagpairs_seeds3-4_20260320-194522/models/2026-03-21T10-02-19-765590_HRL.pth
hrl|1.0|80|4|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/learned_selector_diagpairs_seeds3-4_20260320-194522/models/2026-03-21T13-06-44-235267_HRL.pth
ga|0.2|20|0|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/GA_sensitivity_20260314-182806/models/2026-03-14T20-20-20-841731_HRL.pth
ga|0.2|20|1|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/GA_sensitivity_20260314-182806/models/2026-03-14T22-12-13-679410_HRL.pth
ga|0.2|20|2|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/GA_sensitivity_20260314-182806/models/2026-03-15T00-03-37-766032_HRL.pth
ga|0.2|20|3|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/learned_GA_selector_diagpairs_seeds3-4_20260320-194457/models/2026-03-20T22-34-38-533223_HRL.pth
ga|0.2|20|4|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/learned_GA_selector_diagpairs_seeds3-4_20260320-194457/models/2026-03-21T01-25-39-180187_HRL.pth
ga|0.5|50|0|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/GA_sensitivity_20260314-182806/models/2026-03-15T20-39-20-375181_HRL.pth
ga|0.5|50|1|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/GA_sensitivity_20260314-182806/models/2026-03-15T22-54-13-635065_HRL.pth
ga|0.5|50|2|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/GA_sensitivity_20260314-182806/models/2026-03-16T01-09-00-674097_HRL.pth
ga|0.5|50|3|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/learned_GA_selector_diagpairs_seeds3-4_20260320-194457/models/2026-03-21T04-43-29-577932_HRL.pth
ga|0.5|50|4|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/learned_GA_selector_diagpairs_seeds3-4_20260320-194457/models/2026-03-21T08-00-23-545383_HRL.pth
ga|1.0|80|0|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/GA_sensitivity_20260314-182806/models/2026-03-16T23-30-50-356503_HRL.pth
ga|1.0|80|1|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/GA_sensitivity_20260314-182806/models/2026-03-17T01-46-29-520555_HRL.pth
ga|1.0|80|2|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/GA_sensitivity_20260314-182806/models/2026-03-17T04-02-39-500580_HRL.pth
ga|1.0|80|3|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/learned_GA_selector_diagpairs_seeds3-4_20260320-194457/models/2026-03-21T11-20-43-323161_HRL.pth
ga|1.0|80|4|/home/jupyter/Roland Cubahiro/hrl-project/HRL_Management_DQN/results/learned_GA_selector_diagpairs_seeds3-4_20260320-194457/models/2026-03-21T14-39-43-740668_HRL.pth
RUNS

echo "Saved: $OUT"
