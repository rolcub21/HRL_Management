# E27: certified Dynamic PSLAP under sustained arrivals

This post-hoc experiment compares the frozen seed-0 VCG ranker at λ=.10 with
a versioned Dynamic PSLAP rule operating **inside the same exact-SAFE frontier**.
The active protocol is **v2**, writing to
`results/vcg-conditioned-e27-certified-dynamic-sustained-99k-v2/`. The v1
development output records a failed integration attempt and is not evidence.
Both arms use the same E13 physical candidate generator, 20,000-node
goal-directed verifier budget, E14 cleanup and timing-invariant cache, D12
relocation-family proofs, recovery-witness guard, macro executor, hardware
configuration, and matched EpisodeInstances. No network is trained.

The certified Dynamic adaptation preserves the source's placement ordering
(feasible location, then exit distance, travel distance, cell) and due-first
duration-aware dispatch. A one-hot merit passes its chosen SAFE candidate
through the existing normalized mode selector; the adapter asserts that
aggregation preserves that choice. If Dynamic's unconstrained placement is not
SAFE, the next eligible SAFE placement is chosen and the projection is counted.
When no admission or delivery is suitable, a deterministic reconfiguration
choice uses the first blocker of the least-slack retrieval plan when available,
otherwise the shortest SAFE relocation. This is an adaptation, not a claim of
bitwise parity with the old uncertified Track-B scheduler. The guard may
override either arm.

The first panel is intentionally compact: 5×5, 8×8, and 10×10 at approximately
50% *initial usable-storage occupancy*, with as many arriving jobs as initially
stored jobs (8, 36, and 64 total jobs). Arrivals begin at step 20 and recur
every 10 steps. Initial, time-weighted mean, and peak occupancy are recorded;
the nominal initial density is not treated as the realized occupancy profile.
All seeds and methods are fixed before the 99k confirmation manifest is opened.
Each episode has a declared 1,800-second wall limit; a censored episode is not
called complete or infeasible. The larger configurations can take a long time.

Run the 5×5 development integration pilot first:

```bash
bash experiments/conditioned_vcg/E27_certified_dynamic_sustained_99k/run.sh prepare
bash experiments/conditioned_vcg/E27_certified_dynamic_sustained_99k/run.sh run-pilot
```

The pilot checks that both arms receive the same first certified frontier and
that the adapter executes. Quality in this development episode does not tune
the rule. Once the gate passes, freeze and run the nine paired confirmation
instances (18 episodes), then analyze:

```bash
bash experiments/conditioned_vcg/E27_certified_dynamic_sustained_99k/run.sh freeze-confirmation
bash experiments/conditioned_vcg/E27_certified_dynamic_sustained_99k/run.sh run-all
bash experiments/conditioned_vcg/E27_certified_dynamic_sustained_99k/run.sh analyze
```

The run is resumable via authenticated coordinate ledgers. This is the first
certified-baseline comparator; GA/A3C would require separately frozen
candidate-ordering adapters and fidelity checks before extending the panel.
The 5×5 gate passed, and a separate 8×8 development pair completed with the
same first-frontier digest. Neither development result is a confirmation
comparison. The 99k manifest is frozen; the 18 confirmation episodes have not
been launched by the preparation workflow.
