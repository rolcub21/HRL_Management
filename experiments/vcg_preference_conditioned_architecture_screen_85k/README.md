# Preference-conditioned VCG architecture screen (85k development panel)

This is the lean architecture-selection experiment for the proposed final
VCG controller. It compares:

- **A — historical anchor:** authenticated VCG 1.1 plus its detached
  Monte-Carlo handling head. The completed 85k report is reused exactly; A is
  not retrained or rerun.
- **B — conditioned vector critic:** one jointly trained two-head critic
  `Q(s,c,lambda) = [Q_op,Q_N]` receives the fixed episode preference.
- **C — masked/unconditioned vector critic:** the same training recipe and
  two-head critic, but its lambda input is fixed to zero. Lambda still enters
  action merit at deployment, so C isolates what the preference input adds.

B and C use three paired model seeds, 500 training episodes per seed, the
balanced preference grid `{0, 0.025, 0.05, 0.1, 0.2}`, and fixed terminal
checkpoints. This first screen sets `preference_relabels=1`: every replay item
uses its real behavior lambda and behavior episodes are balanced across all
five levels, but no extra replay relabels are added. A later relabel ablation
can increase this value without confusing the first architecture comparison.

Evaluation reuses the exact serialized, already-opened EpisodeInstances
85000--85011. All arms use the real simulator macro path, deterministic
fixed-lambda deployment, the exact fail-closed verifier, and the same
recovery-witness liveness rule. A fresh weight-only B/C agent is loaded for
each row. There is no training or checkpoint selection on the evaluation
panel.

If any B or C row is unsafe or incomplete, that architecture's entire metric
table is suppressed. There is no complete-case averaging. Eligible metrics
are first averaged over 12 EpisodeInstances within each model seed and then
over the three seeds with equal weight.

## Commands

From the repository root:

```bash
bash experiments/vcg_preference_conditioned_architecture_screen_85k/run.sh prepare
bash experiments/vcg_preference_conditioned_architecture_screen_85k/run.sh smoke
bash experiments/vcg_preference_conditioned_architecture_screen_85k/run.sh train-all
bash experiments/vcg_preference_conditioned_architecture_screen_85k/run.sh evaluate
bash experiments/vcg_preference_conditioned_architecture_screen_85k/run.sh analyze
```

The full sequence is sequential:

```bash
bash experiments/vcg_preference_conditioned_architecture_screen_85k/run.sh run-all
```

Training can be resumed or limited to one seed:

```bash
bash experiments/vcg_preference_conditioned_architecture_screen_85k/run.sh train-conditioned 1
bash experiments/vcg_preference_conditioned_architecture_screen_85k/run.sh train-unconditioned 1
bash experiments/vcg_preference_conditioned_architecture_screen_85k/run.sh train-all 1
```

`train-*` resumes from `latest.pth` when present and skips an existing terminal
checkpoint. The evaluator authenticates every terminal checkpoint before use;
skipping a path is therefore not acceptance of an unauthenticated model.

As soon as all three conditioned checkpoints exist, B can be evaluated before
C finishes:

```bash
bash experiments/vcg_preference_conditioned_architecture_screen_85k/run.sh run-conditioned-analysis
```

This executes the contracted 180 B rows and writes
`conditioned-B-interim-report.json`. It compares A with B and applies the B
performance gate, but it remains provisional: C is still required to identify
whether preference conditioning itself caused the result. The final `analyze`
command writes the separate A/B/C report after C is complete.

Set `DEVICE=cpu` or `PYTHON_BIN=...` when needed. The default device is CUDA.
The default output is
`results/vcg-preference-conditioned-architecture-screen-85k-development`.

## Advancement rule

B advances to a new unseen confirmation panel only if all of the following
hold on this development screen:

1. all 180 B rows are strict-safe-complete;
2. at lambda 0, B meets or beats A in both mean dense return and MAE;
3. at lambda 0.2, B meets or beats A in rehandles per 100 deliveries;
4. B has at least three sampled nondominated MAE–rehandle points; and
5. B's aggregate rehandling rate is nonincreasing over the lambda grid.

The report also gives paired B-minus-C deltas and behavior-digest differences.
Those are provisional evidence about conditioning. This screen deliberately
does not add a prediction-calibration branch, so it does not claim that either
head is calibrated across lambda. If B advances, bounded held-out calibration
should accompany the unseen-panel confirmation.
