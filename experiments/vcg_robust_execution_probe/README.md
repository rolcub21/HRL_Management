# Bounded-execution VCG mechanism probe

This is a small, deterministic, no-training experiment for the proposed
set-valued execution extension. It uses the repository's existing exact
closed-admission recovery verifier and exhaustively checks a finite declared
set of execution realizations.

The probe demonstrates:

- a nominally safe relocation that is rejected when a one-step execution
  delay is included in the declared set;
- a delivery with enough slack to remain safe under its declared delays;
- rejection when an intermediate primitive trajectory intersects a declared
  route-obstacle envelope even though every terminal successor remains
  recoverable;
- nominal versus radius-one terminal-position containment for an exact-SAFE,
  physically executable `Accept` candidate on the real 5x5 frontier;
- fail-closed treatment of an `UNKNOWN` successor certificate;
- different handling preferences selecting different actions from the same
  robustly certified frontier; and
- explicit labeling of an unsafe observation outside the declared uncertainty
  set.

Run from the repository root:

```bash
bash experiments/vcg_robust_execution_probe/run.sh
```

The report is written to
`results/vcg-robust-execution-probe/robust-execution-report.json`.

## Claim boundary

This is a one-step finite-set mechanism probe for strict `Deliver` and
`Reconfigure` recovery macros plus bounded terminal-position error after a
successful, live-frontier `Accept` putdown. It
does not construct a recursively robust viability kernel, cover complete
episodes, model unbounded disturbances, or validate hardware execution.
`Accept` and `Hold` do not yet have robust execution semantics here.
