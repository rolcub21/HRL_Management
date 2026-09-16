# E3 — Why certification matters

E3 isolates the recoverability verifier in the final conditioned VCG. It uses
the same three frozen final checkpoints, deterministic hierarchical selector,
recovery-witness liveness guard, fixed preference `lambda = .10`, and 30
serialized E1 EpisodeInstances. The paired factor is only candidate eligibility:

- `physical_feasibility_only`: every physically executable candidate may be
  scored;
- `recoverability_certified`: only exact-SAFE candidates may be scored.

The 90 certified controls are authenticated and reused from E1. Therefore E3
requires only 90 new physical-arm runs, not 180 runs and no retraining.

The physical arm computes exact certificates in shadow solely to diagnose
whether an admitted candidate is SAFE, UNSAFE, or UNKNOWN. These labels do not
filter its frontier. Exact-SAFE candidates retain byte-for-byte-equivalent
network features. An added UNSAFE or UNKNOWN candidate uses the existing
`rank_after_unavailable` encoding because no valid recovery rank exists.

Primary outcomes are strict completion, exact recoverability deadlock,
self-blocking, an empty certified successor frontier, unsafe admission, and
previously accepted unfinished workloads rendered unrecoverable. Timing and
handling are descriptive only among strict completers if the physical arm has
failures.

## Run

A five-instance seed-0 pilot checks that the mechanism is present without
changing the frozen full design:

```bash
bash experiments/conditioned_vcg/E03_certification_ablation_90k/run.sh run-pilot
```

Then complete all remaining rows; existing pilot ledgers are reused:

```bash
bash experiments/conditioned_vcg/E03_certification_ablation_90k/run.sh run-all
```

The runner automatically selects the strongest qualifying explanatory case:
a physically legal UNSAFE choice, a different SAFE counterfactual choice, and
preferably a positive critic-merit gap. We will render that companion trajectory
only after observing the frozen E3 result rather than hand-picking a state in
advance.

After the complete report has selected that case, render the three standalone
paper figures:

```bash
bash experiments/conditioned_vcg/E03_certification_ablation_90k/run.sh figures
```

This produces:

1. `e03-a-paired-outcomes`: the 90 paired model-instance outcomes, arranged by
   model seed so the four seed-0 certification gains remain visible;
2. `e03-b-rare-candidate-flow`: the candidate-count mechanism from 27,298
   physically feasible candidates to four unsafe selections and their two
   candidate-source consequences;
3. `e03-c-certified-decision-case`: the selected seed-0/instance-90016 yard
   trajectory, including the rejected B8 acceptance and the certified terminal
   state.

The case replay is inference-only and explicitly post-hoc explanatory
evidence; it does not add, replace, or select any benchmark row. The aggregate
figures are rendered directly from the authenticated E3 report and all 90
physical-arm ledgers.

Outputs are written to
`results/vcg-conditioned-e03-certification-ablation-90k-v2/`.

The `-v2` protocol corrects the physical arm's dense-return normalization.
The initial `-v1` execution stored the legacy environment return under the
dense-return name; its safety, frontier, timing, handling, and trajectory
records were unaffected and remain retained as superseded diagnostic evidence.
