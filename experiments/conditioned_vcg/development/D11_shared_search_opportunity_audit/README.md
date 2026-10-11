# D11 — Shared-search opportunity audit

D11 measures whether the remaining E14 exact-search cost contains repeated
canonical intermediate-state work. It is a bounded development audit, not a
new safety mechanism and not a training run.

The input is the frozen E14 medium-occupancy query trace. The deterministic
sample contains the smallest, median, and widest completed frontiers in full,
plus up to five role-stratified queries from every other completed frontier.
This gives approximately one thousand root queries while retaining both
within-frontier and between-decision comparisons.

For every expanded state, D11 records the timing-erased physical key, action
enumeration time, successor-application time, action count, transition
signature, root/frontier identity, and membership in the returned completion
witness. Every instrumented certificate and witness must exactly equal E14's
stored path-cleanup result.

The report separates repeated work:

- within one root search;
- across candidate searches in one frontier;
- across different decision frontiers.

The conservative implementation gate uses deterministic action-enumeration
reuse. Successor-application time is reported separately because caching
complete successor objects has a higher memory cost. A repeated state is
**not** treated as a SAFE certificate.
Positive proof reuse still requires a constructive witness; UNSAFE requires
an exhaustive proof; UNKNOWN is never a proof. Consequently proof-validation
cost is zero for the transition-only estimate and is reported explicitly as
not applicable.

## Run

Run sequentially on an otherwise idle CPU:

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/conditioned_vcg/development/D11_shared_search_opportunity_audit/run.sh run
```

The accumulated instrumented-search budget is 900 seconds and the frozen
sample cap is 1,100 root queries. If the time limit is reached, the partial
sample remains an authenticated censored measurement and is not interpreted
as failure or infeasibility.

The implementation gate is predeclared: advance to a shared action/transition
prototype only if estimated net savings exceed 20% of the sampled reference exact
search time with acceptable memory. Otherwise stop without adding another
runtime mechanism.
