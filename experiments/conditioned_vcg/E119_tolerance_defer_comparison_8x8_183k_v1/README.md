# E119: frozen comparison of nominal and tolerance-aware Defer

E119 evaluates the completed E118 `schedule_only` **episode-120 endpoint**, not
its best development checkpoint. Two rules run on the same five E115/E118
development instances (112800–112804), ten evaluation episodes in total:

- `nominal_due`: the original `min(d_i-t) > 0` waiting eligibility.
- `due_plus_20`: use `min(d_i+20-t) > 0` for eligibility and the bounded waiting
  horizon. Each Defer still lasts at most 10 primitive steps, and the 16-Defer
  streak bound is retained.

The pickup exclusion, exact recoverability checks, recovery witness guard,
nominal due slack shown to the critic, and objective `J=2T+E+0.1L+10R` remain
as in the original evaluation. T/E use release-fixed due dates and a ±20
tolerance. The changed rule gives additional waiting choices; it does not
guarantee that starting retrieval at d+20 completes within the window.

The model is frozen, epsilon is zero, no replay is inserted, and no optimizer
or target update occurs. E119 records parameter digests before and after every
episode. It extracts a separate small model snapshot and authenticates its
source and code. E118 training and results are untouched, including a running
`early125` arm.

All five nominal-rule rows must reproduce the original E118 episode-120
validation trajectory before intervention rows can run. Parity checks the
instance, delivery errors, steps, macro counts, action/source counts,
completion, and J; wall time is excluded. Device-dependent ranking changes
will fail this check rather than silently contaminate the comparison.

Per-decision and per-delivery logs capture Defer availability and rejection
reason, both rules' eligibility on the same state, inbound presence, due
slack, guard forcing, remaining alternatives, and predicted combined costs.
The report separates early deliveries with Defer available, early deliveries
with Defer excluded, and guard-forced early deliveries. Native and changed
trajectories can diverge after the first action difference; local attribution
is descriptive and is not an oracle assessment of an alternative's return.

Each episode has the original 3,600-step cap and a 300-second wall cap. Failed
completion never wins through a smaller accumulated cost. This is one frozen
model and a reused development panel; it does not establish retrained quality
or a new formal liveness guarantee.

```bash
cd /home/ai_diagnosis/HRL_Management
E119=experiments/conditioned_vcg/E119_tolerance_defer_comparison_8x8_183k_v1/run.sh
bash "$E119" self-check
bash "$E119" run-panel --device cpu &&
bash "$E119" analyze
```

The CPU default uses one thread. The source checkpoint is large, so initial
extraction can take tens of seconds; subsequent starts load the small pinned
snapshot. CUDA can be selected with `--device cuda` if available; the launcher
sets the cuBLAS reproducibility environment before Python starts.

To execute a single native coordinate:

```bash
bash "$E119" run-one --arm nominal_due --seed 112800 --device cpu
```

`run-panel` reuses completed authenticated rows. It stops if native parity
fails. Results are in
`results/vcg-conditioned-e119-tolerance-defer-comparison-8x8-183k-v1-repair1/`, including
`e119-report.json`, `e119-table.md`, and detailed `rows/<arm>/instance-*.json`.
Use `status` for a partial report without running episodes. No training is
performed by any E119 command.

The active output is `repair1`. The first attempt completed one native episode
but could not serialize the inherited undefined evaluation-loss metric. The
repair writes undefined metrics as JSON null and canonicalizes tuples/lists
for parity comparisons. The first attempt's output is retained; the policy,
weights, and experiment design are unchanged.
