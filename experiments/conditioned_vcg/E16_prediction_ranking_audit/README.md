# E16 — Prediction-to-ranking mechanism audit

E16 asks whether candidate-relative future-handling forecasts change the
deployed ranking, and whether those changes explain useful episode outcomes.

## A. Offline stage

The first stage launches no training and no environment rollouts. It reuses:

- the completed fixed bank: 34 states, 686 candidates, 17 predictor inputs,
  and three model seeds;
- the exact hierarchical selector;
- all 270 paired E5(c) future-versus-immediate episode outcomes.

For the causal E5(c) comparison, the future predictor input remains clamped at
`.10`; the external deployment weight takes `.05`, `.10`, and `.20`. The audit
reconstructs both merits on identical states and candidate sets:

```text
M_immediate(c) = Qop(c) - lambda I_reconfigure(c)
M_future(c)    = M_immediate(c) - lambda N_future(c, .10)
```

Run:

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/conditioned_vcg/E16_prediction_ranking_audit/run.sh \
  analyze-offline
```

Outputs are written under
`results/vcg-conditioned-e16-prediction-ranking-audit/`.

### Offline result

The authenticated reanalysis completed with no new runs. On 102 fixed
frontiers per deployment preference, adding the fixed learned-future signal
changed the selected candidate on 4 frontiers at lambda=.05, 6 at lambda=.10,
and 8 at lambda=.20. Every changed winner had lower predicted future handling.
Seventeen of the 18 changes remained within the same mode; 17 were alternative
Accept placements and one changed Reconfigure to Accept. Thus the main
mechanism is candidate-relative placement selection, rather than wholesale
switching among macro families.

The effect below the winner is broader: the future term changed at least one
within-mode ordering on 70/102, 86/102, and 93/102 frontiers as lambda
increased. This is consistent with a margin-crossing mechanism: many rankings
move, but only near-top changes alter the deterministic decision.

The completed E5(c) pairs independently show different episode trajectories
on 50/90, 56/90, and 62/90 coordinates. Mean rehandling improved at all three
preferences; MAE was nearly unchanged/slightly worse at .05 and improved at
.10 and .20. These episode outcomes and the fixed-bank ranking results are
aggregate-linked evidence, not matched decision-level continuations.

The fixed-bank result is heterogeneous by model seed. Seed 0 supplied most
winner changes, seed 1 supplied four, and seed 2 supplied none, even though all
three seeds produced changed E5(c) trajectories. The continuation stage must
therefore sample actually visited deployment states across seeds rather than
treating the old 34-state bank as representative.

## B. Paired continuation bank

The bounded causal stage uses new EpisodeInstances frozen before any outcomes
are observed. For each of three model seeds and four regimes (`reference`,
`dwell_short`, `dwell_long`, and `mirrored_entry`), it searches three instance
seeds in their declared order. It captures the earliest actually visited,
unguarded frontier where the immediate-only and fixed-future rankings choose
different exact-SAFE candidates.

From that identical state, two branches are evaluated:

```text
branch A: force the immediate-only winner once
branch B: force the learned-future winner once
both:     continue with the same frozen fixed-future policy
```

The deployment multiplier and predictor input are both fixed at `.10`. The
realized prediction target excludes the forced current macro, matching the
trained predictor's post-current-macro target. The audit reports prediction
error for each candidate, candidate-relative ordering correctness, completion,
future and total rehandles, MAE, and steps. Liveness-forced frontiers are not
eligible because they do not expose a learned-ranking choice.

Prepare and run the one-coordinate pilot first:

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/conditioned_vcg/E16_prediction_ranking_audit/run.sh \
  prepare-continuations
bash experiments/conditioned_vcg/E16_prediction_ranking_audit/run.sh \
  run-continuation-pilot
```

If its branch semantics and runtime are satisfactory, run all 12 bounded
coordinates (at most 24 branch rollouts; no training):

```bash
bash experiments/conditioned_vcg/E16_prediction_ranking_audit/run.sh \
  run-continuations
```

Outputs are written under
`results/vcg-conditioned-e16-continuation-bank-96k/`. A coordinate with no
crossing in its three predeclared scout instances remains explicitly recorded;
it is not replaced by a post-hoc state.

## C. Signal-component and signed-timing decomposition

After E16-B completes, reconstruct its ten captured candidate frontiers and
replay the unchanged selector with the common, action-type, and within-type
candidate-residual components separated:

```bash
bash experiments/conditioned_vcg/E16_prediction_ranking_audit/run.sh \
  analyze-components
```

This performs ten deterministic prefix replays to recover complete candidate
score vectors. It does not train a model or launch new counterfactual
continuations. Each reconstructed state, selected pair, and score is checked
against E16-B before analysis. The same report also decomposes paired MAE into
earliness and tardiness from the already saved delivery deviations.

### Component result

The completed decomposition reconstructed all ten E16-B crossings exactly.
The candidate-specific within-action-type residual alone reproduced the full
exact choice in 8/10 cases; the action-type component alone did so in 2/10.
Seven cases were residual-only reproductions, one was action-type-only, one
was reproduced by either component, and one required their combination with
the hierarchical selector. All six within-action-type switches required the
candidate residual and were reproduced by it.

Prediction magnitude and decision mechanism should not be conflated: pooled
score variation was 85.8% between action types, yet the smaller within-type
residual usually determined which exact candidate crossed the selection
boundary. Among the four action-type switches, the type-only replay reproduced
the full action type in three cases.

The mean paired MAE change of -1.560 decomposed into -0.593 earliness and
-0.967 tardiness. Thus both contributed on average, with the larger component
coming from reduced tardiness, but individual cases were heterogeneous.

## D. Matched component rollout ablation

E16-D tests whether the selector-level component diagnosis reproduces at the
episode level. It uses the same frozen 92k EpisodeInstances as E5(c), fixes
both deployment and predictor input to lambda=.10, and changes only the part
of the frozen future prediction admitted to the merit:

```text
immediate only:   Qop - lambda I_reconfigure
action type only: Qop - lambda (I_reconfigure + mu + beta_type)
residual only:    Qop - lambda (I_reconfigure + mu + r_candidate)
full future:      Qop - lambda (I_reconfigure + N_future)
```

The common term `mu` is retained in both component arms so the decomposition
is exact, although it cannot change this shift-invariant selector. No weights
are trained or selected. All four arms use the current corrected macro
executor; historical E5(c) immediate/full rows are checked for parity but are
not substituted into the causal comparison.

First run the 20-row pilot: four arms on the first five instances for model
seed 0.

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/conditioned_vcg/E16_prediction_ranking_audit/run.sh \
  prepare-component-rollouts
bash experiments/conditioned_vcg/E16_prediction_ranking_audit/run.sh \
  run-component-pilot
```

After inspecting its runtime and execution-parity report, the complete
confirmation is 360 current-executor rollouts: four arms, three model seeds,
and 30 matched instances. The 20 pilot rows are reused.

```bash
bash experiments/conditioned_vcg/E16_prediction_ranking_audit/run.sh \
  run-component-confirmation
```

Outputs are written under
`results/vcg-conditioned-e16-component-rollout-ablation-92k/`.

### E16-D pilot result

The seed-0 five-instance pilot completed 20/20 strict-safe rows. Both current
control replays matched their historical E5(c) behavior digests and outcome
fields on 5/5 instances. Relative to immediate-only, the action-type arm
changed four trajectories and reduced mean MAE by 2.775 and rehandles/100 by
2.5. The candidate-residual arm changed all five trajectories and reduced
mean MAE by 4.700 and rehandles/100 by 7.5. The full signal changed all five
and reduced mean MAE by 4.200 and rehandles/100 by 10.0. These small seed-0
results authorize the confirmation run; they are not the final estimate of
either component's contribution.

### E16-D confirmation result

The confirmation completed all 360 current-executor rows (four arms, three
model seeds, and 30 matched instances) with 360/360 strict-safe completion.
Both immediate-only and full-future controls reproduced their historical
E5(c) behavior and outcomes on all 90 coordinates, establishing complete
execution parity for this panel.

Against immediate-only, the action-type component reduced rehandles/100 by
4.861 (normal-approximation 95% CI `[-7.174, -2.548]`) and steps by 14.889
(`[-20.706, -9.072]`). The full signal reduced them by 5.972
(`[-8.576, -3.369]`) and 17.344 (`[-24.210, -10.479]`). Directly comparing
action-type-only with full produced differences of +1.111 rehandles/100 and
+2.456 steps, with both intervals including zero. Thus the action-type
component accounts for most of the established handling-efficiency effect.

The candidate-residual arm had the best point estimates for return (172.27),
MAE (13.43), and within-window rate (0.789), but its paired improvements over
immediate-only had intervals spanning zero. It was also worse than the full
signal by 4.444 rehandles/100 and 12.800 steps, with both intervals excluding
zero. The residual therefore supplies candidate-level refinements and a
possible timing-quality effect, but it does not reproduce the established
handling-efficiency benefit alone.

The decision counts give the clearest operational mechanism. Across the 90
episodes, immediate-only selected 61 Reconfigure and 404 Defer macros;
action-type-only reduced those counts to 26 and 339, while the full signal
reduced them to 18 and 322. The main supported explanation is therefore
state-dependent action-type restraint. Timing changes remain a balance between
reduced earliness and increased tardiness rather than a uniformly improving
effect.

## Claim boundary

This stage directly tests prediction-to-ranking perturbation and separately
summarizes E5(c)'s paired deployment effects. Existing records do not contain
candidate-specific realized continuation targets on the fixed bank, nor the
candidate score vectors at every E5(c)/E12 decision. Therefore the offline
stage alone does not establish that each changed ranking selected the candidate
with the best realized continuation. E16-B supplies bounded intervention
evidence at predeclared visited crossings; E16-D supplies a matched
episode-level component intervention at lambda=.10. E16-B remains descriptive
mechanism evidence rather than an unbiased estimate of average episode-level
treatment effect.
