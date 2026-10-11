# E25-v4: action-interface-matched scale baselines

This post-hoc follow-up repeats the E25-v3 baseline adaptations on the same
frozen E13 warm-start instances. It adds a legal, standalone storage relocation
macro to their common duration-aware scheduler. The assignment sources,
admission contract, stochastic seeds, and evaluator budgets are otherwise
unchanged. The VCG-family rows are reused from E25-v2, not new replications.

At a decision epoch, the new macro is considered only when neither admission
nor strict retrieval is executable. The scheduler enumerates physically legal
relocations using the same physical action model, without requesting a
recoverability certificate. It first prefers the blocker named by its own
least-slack retrieval plan (if a legal relocation exists), then chooses the
shortest complete relocation macro. Exact ties use block label and destination
coordinates. Repeated physical states are avoided; this is a deterministic
cycle precaution, not a viability test. An unresolved or exhausted yard may
still fail. This rule was fixed before opening the new full grid.

The evaluator counts every storage-to-storage PUTDOWN, distinguishing
standalone reconfigurations from relocations within retrieval. Strict
completion also requires no option failures, invalid assignments, fallbacks,
or illegal drops, plus intact reserved-cell commitments. Failed episodes
remain in the completion denominator; quality metrics are conditional on
strict completion.

The integration pilot is deliberately diagnostic. Dynamic PSLAP matches its
E25-v3 trajectory on the 5×5 low case, and a single standalone move rescues
the previously failing 5×5 medium case. On the sampled 8×8 high case, four
legal standalone moves still fail to expose a retrieval before an inbound job
arrives. Thus this removes the *missing action type* confound, but it does not
equalize planning ability or establish that baseline placement/dispatch is the
sole cause of any remaining gap. Do not present the pilot as a positive scale
result, or tune the frozen heuristic from that pilot and call it confirmation.

Prepare and run the small integration pilot:

```bash
bash experiments/conditioned_vcg/E25_v4_reconfiguration_baselines_95k/run.sh prepare
bash experiments/conditioned_vcg/E25_v4_reconfiguration_baselines_95k/run.sh run-pilot
```

If the integration checks pass, run the full 540 new baseline episodes, one
method at a time or all together. The coordinates resume from hashed ledgers:

```bash
bash experiments/conditioned_vcg/E25_v4_reconfiguration_baselines_95k/run.sh run-all
bash experiments/conditioned_vcg/E25_v4_reconfiguration_baselines_95k/run.sh analyze
```

For an individual coordinate, use `run-one --method METHOD --scenario SCENARIO
--seed SEED --replicate REPLICATE`. `analyze-partial` is for progress inspection,
not paper evidence. This experiment does not retrain any model.
