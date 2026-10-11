# V2.3 capacity-aware rolling-GA repair (opened 85xxx development panel)

This is a development-only repair comparison. It authenticates and reuses the
completed 156-row matched comparison, then runs exactly four new versioned
rolling-GA assignment sources on the same 12 already-opened EpisodeInstances
(`85000..85011`). It never opens `86xxx`.

The repair changes feasibility semantics, not the four GA objectives. At an
inbound placement decision it optimizes the mandatory current pickup block as
gene 0 plus the earliest-arrived FIFO prefix that fits the current shared
candidate mask. Every excess arrived block is recorded as nonbinding queue
overflow. Overflow is neither silently discarded nor executed as the
scheduler's `Defer` action, and it receives no virtual penalty. Only gene 0 is
reserved and committed.

Historical method IDs and results are immutable. New source IDs end in
`_capacity_aware_partial`; the expanded report keeps the old failed adapters as
safety-suppressed lineage diagnostics.

## Commands

Authenticate all inputs and prepare the exact 48-row contract without running
an environment:

```bash
bash experiments/vcg_v2_3_capacity_aware_ga_repair_85k/run.sh prepare
```

Execute the four repaired methods x 12 instances:

```bash
bash experiments/vcg_v2_3_capacity_aware_ga_repair_85k/run.sh run
```

If interrupted, resume only exact contract-matching atomic ledgers:

```bash
bash experiments/vcg_v2_3_capacity_aware_ga_repair_85k/run.sh resume
```

The final artifacts are written to
`results/vcg-v2-3-capacity-aware-ga-repair-85k/`:

- `repair-contract.json`
- `run-ledger/<new-method>/seed-85xxx.json` (48 exact rows)
- `expanded-runs.csv`
- `expanded-report.json`
- `expanded-audit.json`

The report uses whole-method safety exclusion: any missing, incomplete,
invalid, unauthenticated, or failed row suppresses that method's numerical
summary. There is no 11-of-12 filtering. Rankings are metric-specific, and the
screen is a one-training-seed development method-capacity/Pareto triage—not a
seed-stability, superiority, or confirmatory claim.

