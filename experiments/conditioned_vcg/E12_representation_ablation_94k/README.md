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

The E11 instances are authenticated from the contract and manifest bytes
frozen into E12. Confirmation does not rebuild E11's historical upstream
checkpoint chain; that chain can change as the repository is organized even
though the serialized E11 panel has not changed.

All nine representation–seed training pairs are complete. For a more
observable, resumable confirmation, evaluate one model seed at a time (2,520
rollouts each):

```bash
bash experiments/conditioned_vcg/E12_representation_ablation_94k/run.sh evaluate-confirmation-seed 0
bash experiments/conditioned_vcg/E12_representation_ablation_94k/run.sh evaluate-confirmation-seed 1
bash experiments/conditioned_vcg/E12_representation_ablation_94k/run.sh evaluate-confirmation-seed 2
bash experiments/conditioned_vcg/E12_representation_ablation_94k/run.sh analyze-confirmation
```

Completed ledgers are authenticated and skipped. Do not overlap these CPU
runs with E13/E14 isolated latency measurements.

## Estimand

For lower-is-better loss `L`, analysis computes
`D_A(r) = L_A(r) - L_A(reference)` on matched coordinates and reports
`D_ablation(r) - D_full(r)`. Positive values mean the ablation deteriorated
more under the shift. Absolute completion, MAE, rehandles, steps,
within-window rate, and return remain beside the interaction. If a required
row is incomplete, the coordinate's quality metrics are suppressed.

## Confirmation result

The confirmation is complete: 7,560/7,560 rows were evaluated and 7,557
strictly completed. The full relational-successor arm accounts for all three
incomplete rows; all of its selected candidates were recorded exact-SAFE and
delivery execution ended with `direct_delivery_live_replan_failed`.

The clearest supported representation effect is delivery-time MAE. Removing
the explicit successor gives a mean shift-degradation interaction of +9.30
simulation steps across 72 seed–preference–regime cells, positive in 51/72;
the three model-seed means are +9.67, +8.53, and +9.70. Removing relational
edges gives +7.32 and 51/72 positive cells, but its seed means are +15.44,
+6.38, and +0.13. Neither ablation establishes a consistent rehandling or
episode-step contribution.

Render the paper-facing descriptive figure and companion table with:

```bash
bash experiments/conditioned_vcg/E12_representation_ablation_94k/run.sh figures
```

The figure displays model-seed variation explicitly. It does not claim an
inferential interval from only three independently trained seeds.

## Execution-consistency audit

The three incomplete full-arm rows selected exact-SAFE delivery candidates but
failed inside the bound executor with `direct_delivery_live_replan_failed`.
The follow-up audit replays those three rows and one predeclared successful
control while observing the certificate/executor boundary:

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/conditioned_vcg/E12_representation_ablation_94k/run.sh \
  audit-execution
```

This performs four inference-only runs and no training. It records the
certified recovery action and successor, live path-planner calls, primitive
state changes, replans, queue arrivals, fixed obstacles, and the realized
post-macro state. Instrumentation must reproduce the parent behavior digest,
outcome, failure reason, macro count, and step count exactly or the audit fails.
The authenticated E12 ledgers are read-only; audit artifacts are written to
`results/vcg-conditioned-e12-execution-consistency-audit-94k/`.

The cases can also be run separately and safely resumed:

```bash
bash experiments/conditioned_vcg/E12_representation_ablation_94k/run.sh \
  audit-execution-case failure-93008-lambda-0p05
bash experiments/conditioned_vcg/E12_representation_ablation_94k/run.sh \
  audit-execution-case failure-93028-lambda-0p10
bash experiments/conditioned_vcg/E12_representation_ablation_94k/run.sh \
  audit-execution-case failure-93028-lambda-0p20
bash experiments/conditioned_vcg/E12_representation_ablation_94k/run.sh \
  audit-execution-case control-93000-lambda-0p10
```

The completed audit is frozen evidence for the pre-correction executor. It
showed that the failing delivery paths were recomputed through the reserved
pickup cell, where a modeled queue arrival then blocked execution. The original
audit artifacts remain immutable after the executor source is corrected.

### Corrected-executor confirmation

The shared delivery and relocation executors now bind the exact initial paths
from their `RecoveryAction`. Any live repair retains the same fixed-obstacle
set used by certification. Confirm this change on the same three failures and
matched successful control with:

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/conditioned_vcg/E12_representation_ablation_94k/run.sh \
  confirm-execution-fix
```

This is again four inference-only runs with no training. The confirmation
authenticates the saved historical E12 and audit artifacts without rebuilding
their pre-correction source contracts. It requires 4/4 strict completion, zero
macro failures and illegal drops, all eight deliveries per case, and equality
between every successful macro's realized physical successor and its certified
successor. Results are written separately to
`results/vcg-conditioned-e12-execution-fix-confirmation-94k/`.

### Corrected-executor parity screen

Before reopening the complete 7,560-rollout aggregate, run the bounded frozen-
policy screen:

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/conditioned_vcg/E12_representation_ablation_94k/run.sh \
  parity-screen
```

This performs 108 inference-only runs and no training: all 3 representations ×
3 model seeds × 4 deployment preferences on reference instance 93000 and
combined-shift instances 93008 and 93028. The latter two include all three
historical E12 failures, so the predeclared gate asks whether the corrected
executor retains 105/105 historical successes and repairs 3/3 failures. It
also reports behavior-digest and metric equality separately; the gate does not
silently equate successful completion with exact trajectory parity.

The run is resumable by model seed (36 inference runs each):

```bash
bash experiments/conditioned_vcg/E12_representation_ablation_94k/run.sh \
  parity-screen-seed 0
bash experiments/conditioned_vcg/E12_representation_ablation_94k/run.sh \
  parity-screen-seed 1
bash experiments/conditioned_vcg/E12_representation_ablation_94k/run.sh \
  parity-screen-seed 2
bash experiments/conditioned_vcg/E12_representation_ablation_94k/run.sh \
  parity-screen-analyze
```

Artifacts are isolated under
`results/vcg-conditioned-e12-executor-parity-screen-94k/`. This is a diagnostic
screen, not a replacement for a corrected full E12 aggregate confirmation.

### Full corrected-executor confirmation

The separately versioned full confirmation reuses all frozen E12 checkpoints
and the exact E11 EpisodeInstances, but executes every coordinate with the
corrected macro executor. It performs 7,560 inference-only evaluations and no
training. Its contract binds the immutable original E12 ledger set, all frozen
checkpoints, the 4-case fix confirmation, the 108-case parity screen, and the
corrected executor sources. Results are written under
`results/vcg-conditioned-e12-corrected-executor-full-94k/`; the original E12
ledgers and their 7,557/7,560 result are never modified.

Prepare the contract, then run all three model seeds:

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/conditioned_vcg/E12_representation_ablation_94k/run.sh \
  corrected-full-prepare
bash experiments/conditioned_vcg/E12_representation_ablation_94k/run.sh \
  corrected-full
```

The run is resumable. For separate overnight jobs, use
`corrected-full-seed 0`, `corrected-full-seed 1`, and
`corrected-full-seed 2`, followed by `corrected-full-analyze`. The final report
states the original and corrected completion counts side by side and regenerates
the E12 aggregate/shift-interaction analysis from the corrected rows.

Render the separately versioned corrected paper figure and summary with:

```bash
bash experiments/conditioned_vcg/E12_representation_ablation_94k/run.sh \
  corrected-full-figures
```

The corrected assets are written beside the corrected report under
`results/vcg-conditioned-e12-corrected-executor-full-94k/`. The historical
figure under the original E12 output is left unchanged.
