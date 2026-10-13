# E85: direct 10×10 cost audit, no training

E84 reduced proof searches on a dense 5×5 case but reduced total wall time much
less. E85 checks the actual 10×10 bottleneck before allocating another training
run. It loads a frozen E79-v2 **latest** checkpoint, replays only a bounded
prefix of a fresh development instance with E79's exact frontier, original
exploratory selector and epsilon schedule, guard, base-100 objective,
relocation-family acceleration, and E14 timing-invariant certificate cache.
It never writes to E79 or updates its
checkpoint. The E79 source contract must still authenticate.

Each boundary records full-frontier wall time, exact verifier and relocation-
family certification time, cache statistics, original selection/scoring time,
legal physical-frontier shadow
time, physical-versus-certified key coverage, and primitive execution time.
The legal frontier is generated **after** the certified frontier, for audit
only; it does not affect the chosen action. One in-memory TD update from the
checkpoint replay is timed separately, using E79's cached target-score backup.
The prefix writes in-memory replay transitions but makes no gradient updates;
the sampled TD update is also in memory. No network, optimizer, or replay
change is saved. The report separates UNSAFE from UNKNOWN and does not call a
legal proposal safe.

Run this on the 10×10 server when the E79 training process is stopped or idle;
running both on the same GPU would distort the timings. Do not copy a partially
written checkpoint. From the repository root, after pulling E85:

```bash
bash experiments/conditioned_vcg/E85_direct_10x10_cost_audit_150k_v1/run.sh self-check
bash experiments/conditioned_vcg/E85_direct_10x10_cost_audit_150k_v1/run.sh preflight --model-seed 0 --device cuda
bash experiments/conditioned_vcg/E85_direct_10x10_cost_audit_150k_v1/run.sh profile --model-seed 0 --device cuda --max-decisions 8
bash experiments/conditioned_vcg/E85_direct_10x10_cost_audit_150k_v1/run.sh analyze
```

The eight-decision prefix is a smoke/cost sample, **not** a representative
64-job training episode. If proof dominates there, collect a second, longer
development prefix before deciding whether a matched 10×10 training pilot is
worth its cost:

```bash
bash experiments/conditioned_vcg/E85_direct_10x10_cost_audit_150k_v1/run.sh profile --model-seed 0 --instance-seed 150100 --max-decisions 24 --device cuda
bash experiments/conditioned_vcg/E85_direct_10x10_cost_audit_150k_v1/run.sh profile --model-seed 0 --instance-seed 150101 --max-decisions 24 --device cuda
bash experiments/conditioned_vcg/E85_direct_10x10_cost_audit_150k_v1/run.sh analyze
```

Results go only to `results/vcg-conditioned-e85-direct-10x10-cost-150k-v1`.
Use `--source-output` for a nondefault E79-v2 output or `--output` for a new
E85 directory. Existing profile files are never overwritten. The analysis
reports an *optimistic* upper bound on speedup from eliminating measured
certification time. It uses the longest captured prefix per instance and keeps
each analysis versioned. It does not authorize skipping certification or changing the running
E79 protocol. A proposal-trained 10×10 model would need a separate matched
study: E79's unmasked rank/guard features are unavailable before proof, and
legal-action TD backups need a quality check on fresh certified executions.
