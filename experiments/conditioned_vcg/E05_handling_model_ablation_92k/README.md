# E5 — Handling-model validation and conditioning-input ablation

E5 contains two distinct pieces of evidence.

## E5(a): completed model validation

No new rollout or training is performed. `prepare` assembles the final
Monte-Carlo calibration and fixed-bank diagnostics for seeds 0 and 2 from D9,
and for seed 1 from its completed round-10 continuation. It records prediction
MAE/bias, smoothness of `Q_N`, direction reversals, monotonicity of effective
`lambda Q_N`, and decision-switch incidence.

This validates the internal frozen consequence model. It is not an ablation.

## E5(b): lean conditioning-input ablation

The full control uses the final frozen network as trained:

`Q_N(h, lambda)`.

The ablation uses the same weights but clamps only the network input:

`Q_N(h, .10)`.

Deployment still computes `M = Q_op - lambda Q_N`, so lambda continues to set
the handling penalty. The experiment isolates whether the learned
preference-dependent prediction contributes beyond applying different scalar
weights to one fixed handling prediction. It does **not** establish whether
conditioned training is necessary; that stronger claim would require separately
training an unconditioned model.

The full conditioned controls are reused from E4. At `lambda=.10`, full and
clamped computations are identical by construction, so the E4 row is reused as
both conditions. Only `.05` and `.20` require new rollouts.

## Pilot

This command first resumes/prepares the matching E4 seed-0 five-instance pilot,
then runs the ten new clamped rows:

```bash
bash experiments/conditioned_vcg/E05_handling_model_ablation_92k/run.sh run-pilot
```

The pilot is a debugging/mechanism check, not paper evidence.

## Full experiment

Complete E4 first, then run E5:

```bash
bash experiments/conditioned_vcg/E04_safe_frontier_ranking_92k/run.sh run-all
bash experiments/conditioned_vcg/E05_handling_model_ablation_92k/run.sh run-all
```

E5(b) adds 180 resumable rows rather than rerunning the 270 conditioned E4
controls. Outputs are written to
`results/vcg-conditioned-e05-handling-ablation-92k/`.

### E5(b) conditioning mechanism diagnostic

The rollout ablation tests deployment benefit. The complementary fixed-bank
diagnostic asks whether the predictor responds to its input, whether that
response is candidate-relative, and whether it crosses ranking or selection
margins:

```bash
bash experiments/conditioned_vcg/E05_handling_model_ablation_92k/run.sh mechanism
```

It reuses 34 fixed states, 686 candidates, and 17 lambda values for each of the
three final model seeds. It reproduces the stored full selector before applying
the `.10`-clamped counterfactual, then reports prediction changes, centered
within-frontier changes, within-mode order changes, selected-action changes,
and the already-completed E5(b) rollout trajectory differences. It performs no
training, checkpoint selection, or new rollout. It does not test correctness
against lambda-specific Monte-Carlo continuation policies or the necessity of
conditioned training.

## E5(c): learned future consequence versus immediate cost

E5(c) isolates the value of the learned future-handling estimate. Its control
uses the frozen predictor at input `lambda=.10` and then applies the requested
deployment multiplier:

`Q_op - lambda * (1_Reconfigure + N_future(h, .10))`.

The ablation retains only the current macro cost:

`Q_op - lambda * 1_Reconfigure`.

The complete E4 and E5(b) rows supply all 270 fixed-future controls, so only
the immediate-only arm requires new execution. Start with the declared
seed-0/five-instance mechanism pilot:

```bash
bash experiments/conditioned_vcg/E05_handling_model_ablation_92k/run_future_consequence.sh run-pilot
```

If the implementation checks pass, the resumable full comparison is:

```bash
bash experiments/conditioned_vcg/E05_handling_model_ablation_92k/run_future_consequence.sh run-all
```

The full program adds 270 rollouts and performs no training or checkpoint
selection. Outputs are written to
`results/vcg-conditioned-e05c-future-consequence-ablation-92k/`.

## E5(c) paper figure

After the full analysis completes, render the authenticated paired comparison:

```bash
bash experiments/conditioned_vcg/E05_handling_model_ablation_92k/run_future_consequence.sh figures
```

The arrow in each `lambda` color runs from the immediate-cost-only operating
point to the learned-future operating point. Bold endpoints aggregate the 90
matched model-seed/EpisodeInstance pairs; the lighter arrows expose the three
model-seed means. The renderer writes PDF, PNG, SVG, and its source data into
the E5(c) result directory without rerunning any policy.
