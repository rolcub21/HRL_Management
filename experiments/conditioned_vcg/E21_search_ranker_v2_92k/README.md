# E21-v2 — accounting-consistent certified search ranker

E21-v1 completed safely but exposed a terminal-value pathology. Its
goal-directed recovery witness was treated as a timing policy, even though the
witness delivers admitted jobs as soon as feasible. Unadmitted jobs were valued
as if they would later receive perfect placement and delivery rewards. The
resulting asymmetry encouraged repeated out-and-back relocations.

V2 retains the same depth-two exhaustive exact-SAFE rollout search and changes
only the terminal evaluation:

- every undelivered job retains the same maximum delivery-reward potential;
- every unplaced job retains the same placement-reward potential;
- a validated witness contributes remaining primitive service steps, but its
  ASAP delivery times are **not** scored as a timing policy;
- unplaced jobs receive a geometry-only optimistic service-step lower bound;
- a depth-two branch returning to the root physical configuration is pruned,
  ignoring clocks, so relocation cannot serve as a liveness-bypassing wait.
- the exact inverse of a selected relocation is tabu at the next real decision;
  any non-reconfiguration action clears the tabu, the recovery guard bypasses
  it, and it is ignored if it would empty the admissible frontier.
- reconfiguring a block is structurally dominated whenever that same block has
  an exact-SAFE direct-delivery candidate. Other reconfigurations remain live,
  so obstruction-clearing and genuine recovery moves are unaffected.

Future arrival times remain masked. The exact verifier, guard, hierarchical
selector, executor, dense objective, E4 instances, and authenticated E4
controls remain unchanged. There is no training or neural score.

This is explicitly post-hoc and was designed after diagnosing E21-v1. It is a
sensitivity/strong-baseline test, not a new prospective confirmation.

## Run order

Preparation only:

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/conditioned_vcg/E21_search_ranker_v2_92k/run.sh prepare
```

Five-instance pilot:

```bash
bash experiments/conditioned_vcg/E21_search_ranker_v2_92k/run.sh run-pilot
```

After inspecting the pilot for completion, out-and-back cycles, and runtime:

```bash
bash experiments/conditioned_vcg/E21_search_ranker_v2_92k/run.sh run-all
```

The full command reuses the five pilot ledgers and runs the remaining 25
instances. Outputs are written to
`results/vcg-conditioned-e21-search-ranker-v2-92k/`.

## Derived E4+E21 paper figure

After the complete 30-instance analysis, render the combined comparison with:

```bash
bash experiments/conditioned_vcg/E21_search_ranker_v2_92k/run.sh figures
```

This command leaves the prospective E4 figure untouched. It writes a separately
versioned E4+E21 figure, authenticated figure data, and a companion table into
the E21-v2 results directory. The search point is explicitly marked post-hoc;
the learned-family zoom remains the original E4 view.
