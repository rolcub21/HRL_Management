# E84: unsafe-legal exposure pilot on 5×5

E83 saved certification work but encountered no physically legal candidate
that certification rejected. E84 repeats its matched, single-seed training
comparison on a deliberately dense 5×5 start that exposes a real disagreement.
The geometry comes from the E03 certification-ablation companion case:
seven stored blocks, one block released one step later, and the transporter
at `(2, 3)`. Both arms first execute the same certified, one-step Defer.
At the resulting boundary, `accept:B8:1:2` is physically executable but
**exhaustively UNSAFE**; four alternative macros are SAFE. The `exposure`
command checks this exact property, separates UNKNOWN from UNSAFE, and checks
feature parity before training can start. This placement ensures the unsafe
frontier enters an actual TD backup from the Defer transition, not merely
the first action selection. The preflight runs one macro but no episode.

Both arms use E83's graph network, masked proof-dependent action features,
dense base-40 reward, γ=.99, exploration, optimizer, Double-Q backup, and
30-episode budget. They differ only in training-time frontier construction:

- `full_frontier` proves every legal successor and backs up over SAFE candidates.
- `rank_then_certify` backs up over all legal candidates but proves proposals
  in rank order before executing one. It cannot execute an unproved move.

At evaluation, both frozen rankers act on the complete certified frontier.
Five new dense instances vary storage durations while keeping the known
unsafe geometry; five new empty-start instances check whether the stress
training sacrifices ordinary 5×5 behavior. These are development checks,
not paper evidence or a direct replication of the original 500-episode VCG
recipe. The fixed geometry and one model seed cannot establish generality.

Run from the repository root. The preflight is cheap and has already been
verified during preparation, but should be rerun after moving servers:

```bash
bash experiments/conditioned_vcg/E84_unsafe_legal_5x5_pilot_149k_v1/run.sh exposure
```

Training is resumable. To inspect a three-episode smoke before committing to
both 30-episode arms:

```bash
bash experiments/conditioned_vcg/E84_unsafe_legal_5x5_pilot_149k_v1/run.sh train --arm full_frontier --through-episode 3
bash experiments/conditioned_vcg/E84_unsafe_legal_5x5_pilot_149k_v1/run.sh train --arm rank_then_certify --through-episode 3
```

Then finish and evaluate:

```bash
bash experiments/conditioned_vcg/E84_unsafe_legal_5x5_pilot_149k_v1/run.sh train --arm full_frontier --through-episode 30
bash experiments/conditioned_vcg/E84_unsafe_legal_5x5_pilot_149k_v1/run.sh train --arm rank_then_certify --through-episode 30
bash experiments/conditioned_vcg/E84_unsafe_legal_5x5_pilot_149k_v1/run.sh evaluate
bash experiments/conditioned_vcg/E84_unsafe_legal_5x5_pilot_149k_v1/run.sh analyze
```

The report is
`results/vcg-conditioned-e84-unsafe-legal-5x5-149k-v1/e84-report.json`.
It compares proof searches, total training time, training exposure and rejected
proposals, strict completion, timing MAE, steps, return, and rehandles on both
evaluation panels. If the control sees no rejected legal candidates, treat the
quality comparison as uninformative and investigate the workload setup; do
not interpret a clean result as permission to skip certification. Even with
exposure, one seed and five instances per panel are only a mechanism pilot.
Do not substitute E84 into E81, E78/E79, or the manuscript without a new,
multi-seed, target-size protocol.
