# E8 — Liveness intervention audit

E8 asks how much of the implemented controller's action sequence is learned
choice and how much is intervention by the exact recovery-witness liveness
guard. It is an attribution audit, not a guard-off safety ablation.

## Existing-log stage

The first stage reads authenticated artifacts and performs no new episodes:

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/conditioned_vcg/E08_liveness_audit_95k/run.sh \
  analyze-existing
```

E13 is the aggregate source because all 45 completed episodes retain a
decision-level `liveness_forced` flag and executed action type. The complete
D12 medium/high traces add primitive duration and physical-relocation counts
for forced versus unforced macros. They overlap E13 seed 95100 and are not
added to its denominator.

E1, E4, E5, E11, and E12 retain episode outcomes but not contemporaneous
unrestricted proposals. Their missing proposal fields are reported explicitly,
not interpreted as learned/guard agreement.

## Instrumented proposal pilot

Existing logs identify guard activation but cannot determine whether the guard
actually changed the learned choice. The predeclared pilot replays E13's
`size_8x8_occ_high`, seed 95100 coordinate using the same frozen model seed 0,
λ=.10, exact checker, E14 cleanup, D12 relocation-family strategy, and
hierarchical selector. It uses the corrected model-faithful macro executor.

At every decision it records:

- the learned proposal on the complete exact-SAFE frontier;
- the guard-executed candidate;
- agreement versus actual override;
- activation reason versus witness persistence;
- proposed/executed operational value, handling prediction, merit, and margins;
- forced/unforced primitive duration and physical relocations.

Prepare the frozen contract while other inference is running:

```bash
bash experiments/conditioned_vcg/E08_liveness_audit_95k/run.sh prepare
```

Run the single episode only after other CPU measurements/evaluations finish:

```bash
bash experiments/conditioned_vcg/E08_liveness_audit_95k/run.sh run-pilot
```

The historical episode took about 50 seconds on the recorded machine. The new
pilot has a 600-second diagnostic wall limit and performs no training or
checkpoint selection. Results are written to
`results/vcg-conditioned-e08-liveness-audit-95k/`.

After a completed pilot, derive the paper-facing summary without rerunning the
episode:

```bash
bash experiments/conditioned_vcg/E08_liveness_audit_95k/run.sh \
  summarize-pilot
```

The guard prepares the next Bellman frontier in `observe_outcome`, so a witness
can activate before the audit wrapper enters the next `select`. The immutable
pilot trace keeps the original pre-select instrumentation; the derived summary
correctly identifies activations as starts of contiguous forced runs.

The pilot measures policy attribution. Forced/unforced outcome differences are
descriptive because the guard activates in a different state population. A
with/without-guard rollout would be a separate simulation-only causal ablation
and is not required for E8.
