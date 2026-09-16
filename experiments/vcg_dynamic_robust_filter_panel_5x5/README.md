# Full-dynamic 5x5 robust-recovery filter stress test

This additive, evaluation-only experiment reuses the authenticated frozen 89k
EpisodeInstances, model-seed-0 VCG 1.1 checkpoint, and detached handling-cost
head. It compares the Cartesian grid

- nominal, one-step robust, and recursively robust **recovery filters**; and
- handling lambda 0 and 0.2.

The four-instance pilot contains 24 frozen-policy episodes. The full 30-instance
panel contains 180 episodes. Rows are written atomically and authenticated
before reuse, so either panel can resume after interruption.

## Decision and execution contract

At every live environment decision boundary, the runner first enumerates the
repository's existing exact-SAFE frontier. Deliver and Reconfigure candidates
are intersected with the selected finite-budget recovery certificate. UNKNOWN
is rejected fail-closed. Accept and Defer retain their existing nominal
exact-SAFE and bounded-event semantics; the robust recovery abstraction does
not certify them.

Every arm uses the same stateless frozen selector over its filtered frontier:
the operational Q network, optional handling-cost head, and the original
cardinality-normalized mode aggregation. Its recovery-witness guard context is
fixed to the unforced state. This explicit policy-semantic choice avoids using
a retained nominal witness that robust filtering removed or that a realized
endpoint invalidated.

For each selected, nonterminal recovery macro, the live disturbance rule adds
one real WAIT and then uses the lexicographically first declared clear adjacent
stop when one exists. Both primitives reach the authoritative `env.step`, so
block clocks and observable arrivals advance. A Delivery uses the nominal
realization only when its target is the sole undelivered block in the entire
live episode; an empty closed-admission recovery successor is not sufficient.
Accept and Defer execute through the normal live macro executor with no added
disturbance.

## Run

The wrapper uses CPU by default, including on machines without CUDA:

```bash
bash experiments/vcg_dynamic_robust_filter_panel_5x5/run.sh prepare-pilot
bash experiments/vcg_dynamic_robust_filter_panel_5x5/run.sh run-pilot
```

After inspecting pilot runtime and failure modes, run the complete panel:

```bash
bash experiments/vcg_dynamic_robust_filter_panel_5x5/run.sh prepare-full
bash experiments/vcg_dynamic_robust_filter_panel_5x5/run.sh run-full
```

The full grid has 7.5 times as many rows as the pilot, but runtime need not
scale linearly because trajectories diverge and an interrupted/resumed process
starts with an empty in-memory certificate memo. Measure the pilot locally
before launching the full panel; on CPU, the pilot can take tens of minutes and
the full panel can take hours.

CUDA can be requested explicitly without changing the wrapper:

```bash
DEVICE=cuda bash experiments/vcg_dynamic_robust_filter_panel_5x5/run.sh run-pilot
```

The CPU and CUDA contracts use separate bound device values and must not be
mixed in one output directory.

Principal outputs are:

- `dynamic-filter-contract.json`;
- `run-ledger/<method>/lambda-<value>/instance-<seed>.json`; and
- `dynamic-filter-report.json`.

The report emits return, MAE, steps, and rehandling aggregates only when every
row in that method/lambda cell strictly completes. It never silently averages
only successful episodes.

Per-row wall times and memo hits are diagnostic only. They are not compared
between methods because cache misses are charged to earlier loop cells and the
in-memory memo is not persisted across process restarts. The comparative
computation fields are the selected method's counterfactual cold expanded-node
and cutoff counts summed over live decisions.

## Claim boundary

This is a full-dynamic **descriptive stress test of recovery filtering**, not a
full dynamic robust-viability theorem. Recursive certificates close admission
over currently stored work and use the frozen recovery abstraction. They do not
model future arrivals, robust Accept/Defer semantics, or the live clock changes
caused by the injected WAIT. The experiment is not hardware validation and does
not support confirmatory conclusions from the four-instance pilot.
