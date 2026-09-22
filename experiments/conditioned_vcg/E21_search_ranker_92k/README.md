# E21 — certified search-ranker baseline

E21 asks whether learned VCG ranking adds operational value beyond a stronger,
deterministic, non-learned planner on the **same exact-SAFE frontier**.

The new arm performs exhaustive depth-two macro lookahead.  It executes every
root candidate and every exact-SAFE second-step candidate in a cloned simulator,
scores the realized dense reward, and then values the remaining admitted work
with the candidate's validated completion witness.  Because the verifier uses
goal-directed search, witness length is reported only as a constructive
completion upper bound—not a shortest recovery rank.

Future arrivals not observable at the real root decision are masked in every
simulated branch.  The recovery-witness guard, hierarchical mode aggregation,
tie-breaking, macro executor, objective, and E4 EpisodeInstances remain fixed.
The carrier is the frozen seed-0 VCG configuration, but **no neural score is
used by the E21 ranker**.

The E4 heuristic-safe, frozen-Qop, and conditioned-VCG (`lambda=.10`) rows are
authenticated and reused as controls.  They are reanalyses, not independent
replications.  E21 is explicitly post-hoc because the 92k panel was already
opened.

## Run order

Preparation is fast and launches no rollouts:

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/conditioned_vcg/E21_search_ranker_92k/run.sh prepare
```

Run the five-instance runtime/mechanism pilot first:

```bash
bash experiments/conditioned_vcg/E21_search_ranker_92k/run.sh run-pilot
```

If the pilot completes cleanly and its planning time is acceptable, complete
the 30-instance new arm:

```bash
bash experiments/conditioned_vcg/E21_search_ranker_92k/run.sh run-all
```

The pilot reuses its five authenticated ledgers during `run-all`.  There are no
training runs.  Exact search inside the planner can make this arm materially
slower than neural inference; the pilot exists to measure that cost before the
remaining 25 rollouts.

Outputs are written under
`results/vcg-conditioned-e21-search-ranker-92k/`, including the contract,
manifest, per-instance ledgers, report, and comparison table.
