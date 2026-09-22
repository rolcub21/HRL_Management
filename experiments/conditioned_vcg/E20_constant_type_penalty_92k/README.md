# E20 — Constant action-type penalty ablation

E20 asks whether the learned handling model earns its deployment complexity,
or whether its benefit at `lambda=.10` can be reproduced by four fixed
action-type constants.

The new arm uses

```text
M_constant(s,c) = Qop(s,c)
                  - .10 [I_reconfigure(c) + b_action_type(c)].
```

Each model seed has four constants (`Accept`, `Deliver`, `Reconfigure`, and
`Defer`). They are candidate-weighted means of that seed's frozen future-head
outputs on all unique frontiers visited by the nine predeclared **training
probes**. The procedure uses no 92k evaluation state, target, or outcome. The
constants and their provenance are written first, then cryptographically bound
into the evaluation contract.

The matched controls are:

- `learned_state_dependent_type`: E16-D's action-type-only signal, whose type
  penalty varies with the current state;
- `full_future`: E16-D's full candidate-level future signal.

Those 180 rows are authenticated and reused from E16-D. They are reported as
reanalyses, not independent replications. E20 launches only the 90 fixed-type
rollouts (three model seeds by 30 frozen EpisodeInstances), or five new
rollouts for the pilot. No network is trained or selected.

## Run order

Prepare the non-test calibration and freeze the contract:

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/conditioned_vcg/E20_constant_type_penalty_92k/run.sh prepare
```

Inspect `e20-type-penalty-calibration.json`, then run the five-instance seed-0
pilot:

```bash
bash experiments/conditioned_vcg/E20_constant_type_penalty_92k/run.sh run-pilot
```

If the pilot is sound, run the full 90-rollout confirmation. The five pilot
ledgers are reused:

```bash
bash experiments/conditioned_vcg/E20_constant_type_penalty_92k/run.sh run-confirmation
```

Results are written to
`results/vcg-conditioned-e20-constant-type-penalty-92k/`.

## Predeclared comparisons

All comparisons are paired by model seed and EpisodeInstance. Besides paired
95% Student-*t* intervals, E20 reports a practical-equivalence check: the
paired 90% *t* interval for constant minus control must lie strictly within
the declared symmetric margin.

| Metric | Margin | Operational interpretation |
|---|---:|---|
| MAE | ±1.0 step | one step of mean delivery error |
| Rehandles/100 | ±2.5 | one relocation per 40 required deliveries |
| Episode steps | ±5 | five primitive steps per episode |
| Within-window rate | ±0.025 | one delivery per 40 required deliveries |

The test is deliberately limited to `lambda=.10`. Equivalence here would
support a simpler deployment rule at this operating point; it would not prove
equivalence over the full preference family. Conversely, an advantage for the
state-dependent-type control would show that contextual type penalties add
value, while a further full-future advantage would support candidate-specific
consequence modeling.

## Confirmation result

The confirmation completed all 90 new constant-type rollouts strictly. The
180 matched controls are authenticated E16-D reanalyses and were not rerun.

| Arm | Strict complete | Return | MAE | Within ±20 | Steps | Rehandles/100 |
|---|---:|---:|---:|---:|---:|---:|
| Fixed action-type constants | 90/90 | 154.74 | 14.76 | 0.756 | 163.74 | 6.25 |
| Learned state-dependent type | 90/90 | 165.16 | 13.92 | 0.764 | 154.04 | 3.61 |
| Full learned future | 90/90 | 168.83 | 13.68 | 0.779 | 151.59 | 2.50 |

Relative to learned state-dependent type, fixed constants produced 0.843
higher MAE (paired 95% CI `[0.360, 1.326]`), 2.639 more rehandles per 100
required deliveries (`[0.777, 4.500]`), 9.70 more steps (`[5.37, 14.03]`),
and 10.42 lower return (`[-17.07, -3.77]`). Practical equivalence failed for
MAE, rehandles, and steps; it held only for within-window rate.

The result shows that this **mean-distilled** fixed-type construction is
insufficient at `lambda=.10`; it does not exclude every validation-tuned fixed
penalty. The constants were candidate-weighted means of the complete frozen
future prediction within each type, not direct state-weighted means of
`beta_type(s)` and not values optimized for deployed performance. Relative to
this baseline, state dependence of the type projection supplied most of the
remaining handling and step improvement. A separately trained state-to-type
head was not tested: the state-dependent arm is a projection of the existing
candidate-level predictor over the candidates present in each frontier.

The full-future arm has the best point estimates, but its incremental
advantage over the state-dependent-type arm is smaller and was not the
predeclared E20 equivalence comparison. E20 therefore supports the usefulness
of the existing context-dependent type projection, not by itself a claim that
candidate-specific forecasting is indispensable. A stronger fixed baseline
requires constants selected outside the test panel for deployed performance.
