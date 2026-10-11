# VCG V2.3 matched 85xxx online-baseline comparison

This is a development-only comparison on the already-opened EpisodeInstance
panel `85000..85011`. It does not open or sample the reserved `86xxx` final
panel and cannot authorize a confirmatory or final performance claim.

## Ranked primary methods

The complete-system ranking contains exactly eight online methods:

1. VCG constrained V2.3 gamma=1, selected episode 160. Its authenticated
   `12 instances x 4 fixed action RNGs` validation ledger is reused; the policy
   is not rerun.
2. VCG-Dense V1.1 selected procedure. Its authenticated
   `3 selected training seeds x 12 instances` control ledger is reused; the
   policies are not rerun.
3. duration-aware scheduler + nearest-free assignment;
4. duration-aware scheduler + dynamic PSLAP;
5. duration-aware scheduler + 2009 rolling GA;
6. duration-aware scheduler + duration-aware rolling GA;
7. duration-aware scheduler + operational rolling GA; and
8. duration-aware scheduler + enhanced-complete rolling GA.

Each deterministic method runs exactly once on each materialized instance.
All six use the same source-neutral duration-aware scheduler and differ only in
the named online assignment source.

V2.2 episode 80 is preserved as a superseded diagnostic, not ranked. The
offline full-schedule GA is recorded separately because it has future-schedule
information. REG-v5 and Kim-style learned assignment remain pending a
geometry-matched checkpoint/protocol; existing 10x10/40-block and other-panel
results are explicitly incompatible with this 5x5/8-block ranking.

## Estimands and safety

- V2.3: average four action-RNG realizations inside each EpisodeInstance, then
  weight the 12 EpisodeInstances equally.
- V1.1: weight the three training seeds equally and the 12 EpisodeInstances
  equally. Three per-seed summaries and their descriptive variability are
  retained; the 36 rows are not treated as independent samples.
- Deterministic methods: one result per EpisodeInstance, equally weighted.

The common statistical unit is the EpisodeInstance cluster (`n=12`). The
nominal one-sided 95% cluster bound uses `t(.95, df=11)=1.7958848187036691`
and is descriptive only. V2.3's original seven-look Bonferroni budget UCB is
preserved separately and is not replaced by the common-panel bound.

A method enters numeric ranks and Pareto fronts only if its entire expected
grid is present, strict, fully complete, and free of safety/fallback failures.
There is no 11-of-12 or other complete-case filtering. Rankings are reported
per metric, alongside several Pareto fronts; no scalar overall winner is
created.

The common rehandling metric is total physical storage relocations per 100
required deliveries. Target-bound obstruction clearances are a separate
mechanism field and never substitute for the total. V2.3's source ledger does
not authenticate the target-bound/standalone decomposition, so those fields
remain unavailable even though its selected Reconfigure count equals its total
physical moves.

## Commands

Authentication and execution-plan check only (the default; no rollout):

```bash
cd /home/ai_diagnosis/HRL_Management

PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -u compare_vcg_v2_3_matched_baselines.py \
  --device cpu \
  --output-dir results/vcg-v2-3-matched-baselines-85k
```

Execute/resume the 72 deterministic rows after reviewing the contract:

```bash
cd /home/ai_diagnosis/HRL_Management

PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -u compare_vcg_v2_3_matched_baselines.py \
  --device cpu \
  --execute-baselines \
  --resume-existing \
  --output-dir results/vcg-v2-3-matched-baselines-85k
```

The run is sequential and resumable. Every row is atomically written under
`run-ledger/<method>/seed-<seed>.json`; a resumed row must match the frozen
input contract, exact schema, method identity, and dense-rescore semantics.

## Outputs

- `comparison-contract.json`: self-hashed, pre-outcome method/estimand/safety
  contract and source-code/artifact fingerprints.
- `preflight.json`: authentication/execution-grid status.
- `matched-runs.csv`: all 156 source rows (48 + 36 + 72).
- `matched-report.json`: self-hashed method summaries, position-wise timing,
  metric ranks, Pareto fronts, paired descriptive contrasts, registries, and
  the explicitly non-confirmatory capacity screen.
- `matched-audit.json`: self-hashed provenance and all 72 atomic ledger hashes.

The protocol-named stability screen is scientifically only a one-training-seed
method-capacity/Pareto triage: passing it means no safety-eligible matched
online comparator dominates V2.3 jointly on MAE and total physical rehandles.
It does not establish training-seed stability or superiority; independent V2.3
training-seed replication remains required.

