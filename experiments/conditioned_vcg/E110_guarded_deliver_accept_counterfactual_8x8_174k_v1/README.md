# E110: guarded Deliver-versus-Accept counterfactual

E109 found that later learning sometimes changed an identical frontier from
Deliver to Accept. Replay records cannot be restarted faithfully because they
do not contain the full future workload or live guard state. E110 therefore
recaptures the disagreement on reproducible full-episode prefixes.

For each of the five already-open development seeds, E110 runs the frozen
episode-269 controller and finds its first unforced Accept decision for which
the episode-240 network selects Deliver on the identical admissible exact-SAFE
frontier. The native episode is the Accept arm. A second matched episode forces
the episode-240 Deliver once and then resumes the same frozen episode-269
guarded controller. Prefix state, frontier, guard, action history, and reward
must match exactly.

This diagnoses the local action reversal under one fixed continuation. It is
not training, a general policy comparison, an independent test, or paper
evidence.

## Prepare and verify

If E108 advanced after the original E110 preparation, first migrate E110 to
the immutable compact episode-240/269 copies. This rewrites provenance only;
it performs no training or episode run and preserves completed capture rows.

```bash
bash experiments/conditioned_vcg/E110_guarded_deliver_accept_counterfactual_8x8_174k_v1/run.sh repair-contract
```

```bash
cd /home/ai_diagnosis/HRL_Management

bash experiments/conditioned_vcg/E110_guarded_deliver_accept_counterfactual_8x8_174k_v1/run.sh prepare
bash experiments/conditioned_vcg/E110_guarded_deliver_accept_counterfactual_8x8_174k_v1/run.sh freeze-sources
bash experiments/conditioned_vcg/E110_guarded_deliver_accept_counterfactual_8x8_174k_v1/run.sh self-check
```

## Capture disagreements and run matched Deliver branches

```bash
for seed in 112800 112801 112802 112803 112804; do
  bash experiments/conditioned_vcg/E110_guarded_deliver_accept_counterfactual_8x8_174k_v1/run.sh \
    capture --seed "$seed" --device cuda
done

for seed in 112800 112801 112802 112803 112804; do
  bash experiments/conditioned_vcg/E110_guarded_deliver_accept_counterfactual_8x8_174k_v1/run.sh \
    run-deliver --seed "$seed" --device cuda
done

bash experiments/conditioned_vcg/E110_guarded_deliver_accept_counterfactual_8x8_174k_v1/run.sh analyze
```

`run-deliver` reports `skipped_no_disagreement` when its capture found no
eligible reversal. Existing rows are never overwritten.
