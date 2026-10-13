# E83: legal-proposal training with certified deployment (5×5 pilot)

This is a single-seed, development-only test of the question: can the VCG
ranker learn while considering **all physically executable candidates**, with
exact recoverability checks restricted to the action it proposes to execute?
It does not replace the original VCG training or provide paper evidence.

The two arms share the 5×5/8-job instances, initial graph network, dense
base-40 reward, γ=.99, exploration schedule, action features, optimizer,
and nested Double-Q target. To keep the input comparable before and after
proof, both arms mask the certificate-rank and guard-context action features.
The pilot uses batch size 32 and one update every four transitions in **both**
arms so its training time remains modest. It is therefore not a direct
replication of the original 500-episode training recipe.

- `full_frontier` certifies all successors before ranking and uses only the
  certified next frontier in the TD backup.
- `rank_then_certify` ranks the full physically executable frontier, checks
  proposals in rank order until one is certified, and uses the **full legal**
  next frontier in the TD backup. This target can overvalue an action that
  certification would reject; the pilot measures whether that hurts the
  resulting policy. The current-state recovery witness is forced after the
  bounded nonprogress limit; otherwise ranked proposals are tested in turn.
  No unproved action is executed or called SAFE.

Both frozen rankers are evaluated on the **complete certified frontier** for
the same five fresh instances. The matched control makes training cost and
quality interpretable within this pilot. The reported comparison cannot by
itself establish multi-seed stability or larger-yard quality.

Run from the repository root. First verify physical/certified feature parity
at two consecutive decision boundaries:

```bash
bash experiments/conditioned_vcg/E83_legal_proposal_5x5_pilot_148k_v1/run.sh self-check
```

You can pause both 30-episode arms after episode 3, inspect the logged `proof`
and `learn` times, then resume. Each arm saves an atomic resumable checkpoint
after every episode; rerunning with a higher `--through-episode` resumes it.

```bash
bash experiments/conditioned_vcg/E83_legal_proposal_5x5_pilot_148k_v1/run.sh train --arm full_frontier --episodes 30 --through-episode 3
bash experiments/conditioned_vcg/E83_legal_proposal_5x5_pilot_148k_v1/run.sh train --arm rank_then_certify --episodes 30 --through-episode 3

bash experiments/conditioned_vcg/E83_legal_proposal_5x5_pilot_148k_v1/run.sh train --arm full_frontier --episodes 30 --through-episode 30
bash experiments/conditioned_vcg/E83_legal_proposal_5x5_pilot_148k_v1/run.sh train --arm rank_then_certify --episodes 30 --through-episode 30

bash experiments/conditioned_vcg/E83_legal_proposal_5x5_pilot_148k_v1/run.sh evaluate
bash experiments/conditioned_vcg/E83_legal_proposal_5x5_pilot_148k_v1/run.sh analyze
```

Results are written under
`results/vcg-conditioned-e83-legal-proposal-5x5-148k-v1/`.
`e83-report.json` compares training wall time, exact-search time and counts,
learning time, strict completion, timing MAE, steps, and rehandles. Interpret
proof savings separately from total training savings: on a light 5×5 yard,
neural updates can dominate wall time even when most exact searches disappear.
The report also counts legal candidates rejected by certification in the
control trajectories. If this count is zero, the two target frontiers had no
observed safety disagreement on this small workload, so equal quality is not
yet evidence that legal-frontier training survives unsafe-action exposure.

If the proposal arm is fast but loses certified-deployment quality, the legal
TD target is a likely culprit, not a safety failure. If it is comparable on
the held-out panel, the next step is a separately versioned larger-yard pilot
with additional seeds—not immediate substitution into E81 or the paper.
