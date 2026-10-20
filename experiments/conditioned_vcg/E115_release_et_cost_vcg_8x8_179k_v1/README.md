# E115: Release-Clock ET Cost VCG (8×8)

E115 is a fresh, one-seed development pilot for the **Release-Clock ET Cost
VCG**. It does not continue E113 or E114 and it does not alter their artifacts.

The active protocol is `repair2`. The initial launch completed zero episodes:
the inherited runner's offline two-reward rescorer correctly noticed that its
legacy/base-dense totals differed from E115's new third objective. Repair1 maps
both compatibility audit slots to the logged E115 return while retaining the
stronger checks that primitive cost heads reconstruct every reward and that
dense ET accrual equals the terminal ET formula on completion. It writes to a
new `...-repair2` result directory; both failed zero-episode outputs are retained.
Repair2 additionally supplies the legacy summary aliases
`selected_objective_return`, `selected_objective`, and `non_delivery_return`;
repair1 passed the step-level verifier but omitted these serializer fields.

The operational objective is

```text
min J = sum_i (2 T_i + E_i) + 0.1 L + 10 R
```

where each due date is fixed at release,
`d_i = arrival_i + storage_duration_i`; `T_i` and `E_i` are tardiness and
earliness beyond ±20 steps; `L` counts non-WAIT primitive equipment actions;
and `R` counts physical rehandles. There is no positive delivery base.
Completion remains a hard requirement supplied by exact certification and the
recovery-witness guard.

Tardiness is charged densely—one unit on every primitive interval for every
released, undelivered job more than 20 steps past due. Earliness is charged at
delivery. The implementation verifies that the dense totals equal the terminal
earliness/tardiness formula on every completed validation episode.

The workload is also changed deliberately. The yard starts empty, durations
use the frozen Poisson-80 generator, and releases are 30 steps apart rather
than 10. This avoids evaluating a timing controller under permanent overload.
The critic sees release-relative due slack for released jobs, but it cannot see
future releases or unreleased durations.

The critic has four nonnegative cost heads matching the four terms above.
Candidate merit is the negative weighted predicted cost. The exact certified
frontier, hierarchical selector, guard, cached exact vector Double-DQN backup,
and conservative frozen-target margin remain in place. Training uses γ=1 so
the learned episodic sum matches `J` rather than a differently discounted
objective.

This pilot fixes β=10. A β sweep or Stage B is not authorized until the new
objective is shown to learn with exact accounting.

## Prepare and verify (no episodes)

```bash
cd /home/ai_diagnosis/HRL_Management

bash experiments/conditioned_vcg/E115_release_et_cost_vcg_8x8_179k_v1/run.sh \
  prepare
bash experiments/conditioned_vcg/E115_release_et_cost_vcg_8x8_179k_v1/run.sh \
  self-check
```

## First pilot segment

The user launches all training. Start at episode 36:

```bash
bash experiments/conditioned_vcg/E115_release_et_cost_vcg_8x8_179k_v1/run.sh \
  train-seed --through-episode 36 --device cuda
bash experiments/conditioned_vcg/E115_release_et_cost_vcg_8x8_179k_v1/run.sh \
  analyze
```

The inherited banner may display the old dense objective. The authoritative
objective is the authenticated E115 contract and the `E115 train`/validation
lines, which print `J`, release timing, equipment work, and rehandles.

At episode 36 the gate checks only that all validation runs complete, all
values are finite, and dense/terminal accounting agrees. It does not claim
quality. If that gate passes, resume only to a prespecified checkpoint:

```bash
for episode in 60 84 112 144 180 216 240; do
  bash experiments/conditioned_vcg/E115_release_et_cost_vcg_8x8_179k_v1/run.sh \
    train-seed --through-episode "$episode" --device cuda
  bash experiments/conditioned_vcg/E115_release_et_cost_vcg_8x8_179k_v1/run.sh \
    analyze
done
```

Epsilon falls from 0.90 to 0.05 over 24,000 decisions, so it remains active
well beyond the first 36 episodes. One seed and the reused development panel
are mechanism evidence only; matched baselines and fresh instances are needed
before a comparative claim.
