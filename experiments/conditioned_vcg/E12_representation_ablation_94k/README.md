# E12 — Representation attribution under distribution shift

E12 asks whether VCG's relational aggregation and explicit counterfactual
successor representation explain the transfer observed in E11. It compares
three independently trained controllers:

| Arm | Graph edges | Candidate successor | Q parameters |
|---|---|---|---|
| `full_relational_successor` | yard adjacency | yes | 122,977 |
| `nonrelational_successor` | empty edge set | yes | 122,977 |
| `relational_current_candidate` | yard adjacency | no; current graph is repeated | 122,977 |

The nonrelational arm retains all three encoder layers but removes their edge
input, making updates node-local. The successor-free arm repeats the current
embedding in the successor slot, making the difference slot zero without
changing network width. This avoids a parameter-count confound.

Everything else is fixed: exact certification, candidate generation, liveness
guard, action features, temporal mode aggregation, Bellman target, training
instances, validation instances, budgets, preferences, and evaluation
instances. The full arm is retrained; E11's checkpoint is not reused as its
control.

## Staged run

The seed-0 stage contains six training runs: three operational-Q runs of 500
episodes, followed by three conditioned-handling runs of 8×50 episodes. The
pilot then evaluates 180 frozen rollouts: 3 representations × 4 preferences ×
5 regimes × 3 common 94k instances.

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/conditioned_vcg/E12_representation_ablation_94k/run.sh prepare
bash experiments/conditioned_vcg/E12_representation_ablation_94k/run.sh train-operational-pilot
bash experiments/conditioned_vcg/E12_representation_ablation_94k/run.sh train-handling-pilot
bash experiments/conditioned_vcg/E12_representation_ablation_94k/run.sh evaluate-pilot
```

Each command resumes from its latest episode/round boundary. A single arm can
be resumed independently:

```bash
bash experiments/conditioned_vcg/E12_representation_ablation_94k/run.sh \
  train-operational-arm nonrelational_successor 0
```

Only if the pilot is informative, train seeds 1 and 2 and confirm on the
already frozen E11 93k EpisodeInstances:

```bash
bash experiments/conditioned_vcg/E12_representation_ablation_94k/run.sh train-confirmation
bash experiments/conditioned_vcg/E12_representation_ablation_94k/run.sh evaluate-confirmation
```

The confirmation has 7,560 rollouts (3 representations × 3 model seeds × 4
preferences × 7 regimes × 30 common instances), so it is not part of the pilot
command.

## Estimand

For lower-is-better loss `L`, analysis computes
`D_A(r) = L_A(r) - L_A(reference)` on matched coordinates and reports
`D_ablation(r) - D_full(r)`. Positive values mean the ablation deteriorated
more under the shift. Absolute completion, MAE, rehandles, steps,
within-window rate, and return remain beside the interaction. If a required
row is incomplete, the coordinate's quality metrics are suppressed.

The outcome pilot decides whether full confirmation is warranted. A matched
state/candidate bank is the follow-up mechanism analysis for prediction and
ranking errors; it is built only after the frozen pilot and cannot revise the
architectures.
