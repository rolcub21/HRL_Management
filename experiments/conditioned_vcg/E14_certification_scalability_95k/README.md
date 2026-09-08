# E14 — Exact-certificate reuse and latency scalability

E14 asks how much of the current verifier's latency is repeated computation
that can be removed without weakening the certification contract. It does no
training and does not modify the production verifier. The frozen model seed
0 controller at `lambda=.10` supplies an ordered query workload on the same
10×10 medium- and high-occupancy seed-95100 instances used by D10.

The experiment evaluates four cumulative stages:

| Stage | Change | Required validity check |
|---|---|---|
| `current` | Complete state-and-budget outcome cache | Recorded baseline |
| `timing_invariant_key` | Remove verifier-irrelevant block clocks from cache identity | Same legal actions, abstract successors, status, and witness |
| `path_enumeration_cleanup` | One canonical transport BFS per block; do not revalidate an already enumerated action | Exact certificate and witness equality |
| `witness_suffix_store` | Retain every validated suffix of a successful completion witness | Replay every served witness to completion |

Positive constructive proofs are stored separately from budgeted search
outcomes. `UNKNOWN` is never a proof. `UNSAFE` remains an ordinary exact
outcome only when the search itself proves it exhaustively. A stored witness
length is a completion upper bound, not a claim of shortest recovery rank.

## Run order

Do not run this alongside the D10 high-occupancy timing pilot. CPU contention
would invalidate latency comparisons. Start with medium only:

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/conditioned_vcg/E14_certification_scalability_95k/run.sh prepare
bash experiments/conditioned_vcg/E14_certification_scalability_95k/run.sh capture-medium
bash experiments/conditioned_vcg/E14_certification_scalability_95k/run.sh replay-medium
bash experiments/conditioned_vcg/E14_certification_scalability_95k/run.sh analyze-medium
```

`capture-medium` has a predeclared one-hour limit. Reaching it is recorded as
right-censoring, not as failure or infeasibility. The completed query prefix
is still valid. `replay-medium` stops between checks after roughly one hour
and is resumable; repeat the same command until it reports `complete`.

Only after reviewing medium should the high workload be captured and replayed:

```bash
bash experiments/conditioned_vcg/E14_certification_scalability_95k/run.sh capture-high
bash experiments/conditioned_vcg/E14_certification_scalability_95k/run.sh replay-high
bash experiments/conditioned_vcg/E14_certification_scalability_95k/run.sh analyze-high
```

The replay reports both cold unique-query time and accumulated ordered-query
time. It measures outcome-cache hits separately from constructive suffix
hits, validates the timing abstraction on every unique physical state, and
requires exact baseline equality for the path cleanup wherever the captured
baseline query completed.

After the replay identifies useful stages, their optional end-to-end checks
are deliberately separate. Run one arm at a time:

```bash
bash experiments/conditioned_vcg/E14_certification_scalability_95k/run.sh confirm-timing-medium
bash experiments/conditioned_vcg/E14_certification_scalability_95k/run.sh confirm-path-medium
bash experiments/conditioned_vcg/E14_certification_scalability_95k/run.sh confirm-suffix-medium
bash experiments/conditioned_vcg/E14_certification_scalability_95k/run.sh analyze-online
```

The timing-key and path-cleanup arms must reproduce every common completed
baseline frontier. The suffix arm may expose a different frontier or policy
trajectory only through a previously validated constructive completion
witness; its frontier, selected actions, and behavior digest are retained.
Each confirmation arm has the same one-hour censoring rule. `confirm-medium`
runs all three sequentially, but the individual commands are preferable for
observability.

The existing D10 high pilot is intentionally not reused as the query trace:
its instrumentation retained aggregate timings, not the exact ordered states
needed for a controlled replay. E14 therefore captures its own authenticated
workload later, under isolated CPU conditions.

## Decision gate

The shared-search-DAG stage is intentionally deferred. It is warranted only
if substantial latency remains after exact timing-key reuse, path cleanup,
and constructive witness suffixes. This keeps E14 focused on measured reuse
rather than adding another mechanism speculatively.

## Medium-occupancy outcome

Timing-key reuse removed only 25 of 6,540 ordered queries. The exact path and
enumeration cleanup reduced replayed search time from 3,502 to 2,024 seconds
while preserving all 6,514 comparable certificates and witnesses. The
end-to-end confirmation preserved all 21 captured baseline frontiers and
strictly completed all 34 deliveries in 2,641 seconds; the unoptimized run
had been right-censored after one hour.

Constructive suffixes were sound but avoided only 154 searches, and their
184-second validation cost exceeded the saved search time. Timing-only and
suffix-only online confirmations are therefore stopped. The remaining
shared-search opportunity is measured by the bounded [D11 audit](../development/D11_shared_search_opportunity_audit/).
