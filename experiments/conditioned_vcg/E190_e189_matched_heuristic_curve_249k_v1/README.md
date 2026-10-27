# E190 — matched heuristic curve for E189

E190 runs the two inexpensive heuristic baselines once on the exact five E189
development days, then compares those fixed rows with every available E189
greedy validation checkpoint.  It answers baseline competitiveness while the
learning curve grows; E187 remains a guard ablation, not a baseline.

The panel contains five deterministic Dynamic PSLAP episodes and four rolling
GA replicates on each of the same five instances (20 GA episodes).  GA is
averaged within instance before the five paired instances are aggregated.
The sample size for paired comparisons is therefore five, never twenty.

Both baselines execute directly in E189's corrected environment and are scored
with the exact objective

```text
J = 2 T_storage + E_storage + 0.5 W_release_to_admission + 0.1 L + 10 R
```

where the storage-clock tolerance is +/-20 and waiting has no grace period.
Strict completion is reported before J.  An incomplete trajectory is never
allowed to look better by ending early.

The contract authenticates the E189 problem definition, instance seeds and
evaluator sources.  It intentionally does not hash the growing E189 training
summary, so episodes 40, 60 and later can be added to the same comparison by
rerunning `analyze`.  The sealed test panel remains closed.  Certified Dynamic
is excluded from E190 and belongs in a separate, slower arm.

From `/home/ai_diagnosis/HRL_Management`:

```bash
bash experiments/conditioned_vcg/E190_e189_matched_heuristic_curve_249k_v1/run.sh \
  prepare

bash experiments/conditioned_vcg/E190_e189_matched_heuristic_curve_249k_v1/run.sh \
  self-check

bash experiments/conditioned_vcg/E190_e189_matched_heuristic_curve_249k_v1/run.sh \
  run-all

bash experiments/conditioned_vcg/E190_e189_matched_heuristic_curve_249k_v1/run.sh \
  analyze
```

Rows are resumable.  To rerun or inspect one coordinate:

```bash
bash experiments/conditioned_vcg/E190_e189_matched_heuristic_curve_249k_v1/run.sh \
  run-one --method dynamic_pslap --seed 246187000 --replicate 0
```

After each later E189 validation, `analyze` reuses the same 25 heuristic rows;
it does not execute another episode or alter either model.

