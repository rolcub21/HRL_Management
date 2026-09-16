# E1 — Prospective common-panel benchmark

E1 is the paper-facing comparison of the preference-conditioned VCG family
against historical VCG 2.3, duration-aware Dynamic PSLAP, capacity-aware GA,
and the A3C adaptation on the same 30 frozen 90k EpisodeInstances.

The figure hierarchy is intentional:

1. `e1-benchmark-reliability-quality.pdf` is the preferred main figure. Its
   first panel reports strict-completion rates; its second panel plots MAE
   against physical rehandles per 100 required deliveries. Incomplete methods
   are retained as hollow, success-conditioned points with their completion
   rates annotated.
2. `e1-benchmark-table-inclusive.md` and `.csv` report reliability separately
   from operational quality. Metrics marked with a dagger are conditional on
   strict completion, descriptive, and excluded from unconditional Pareto and
   paired-superiority claims. The original authenticated strict-gate report and
   `e1-benchmark-table.md` remain unchanged.
3. `e1-operating-points.pdf` retains the uncluttered complete-method operating
   family and the three frozen model-seed paths.
4. `e1-supplement-cumulative.pdf` is a supplementary diagnostic showing how
   common-instance running means stabilize and where incomplete methods stop.

The completed ledgers can be reanalyzed and rendered without rerunning any
policy:

```bash
bash experiments/conditioned_vcg/E01_benchmark_90k/run.sh inspect
bash experiments/conditioned_vcg/E01_benchmark_90k/run.sh analyze
bash experiments/conditioned_vcg/E01_benchmark_90k/run.sh plot
```

To regenerate only the paper-facing E1/E2 operating-point figure from the
serialized contract, report, manifest, and 900 conditioned ledgers—without
reopening historical training dependencies—run:

```bash
bash experiments/conditioned_vcg/E01_benchmark_90k/run.sh \
  plot-operating-points
```

Render the reliability--quality Figure 5, inclusive tables, and ready-to-paste
reporting language from the same closed E1 rows with:

```bash
bash experiments/conditioned_vcg/E01_benchmark_90k/run.sh \
  plot-reliability-quality
```

This is a separately versioned post-hoc descriptive addendum. It performs no
training or evaluation runs, imputes no failure penalty, and does not alter the
predeclared strict-completion gate or the authenticated E1 report.

This wrapper preserves the authenticated launcher at
`experiments/vcg_conditioned_final_comparison_90k/`. See that directory's
README for the protocol identities and failure details.

## Current-method mechanism illustrations

Two inference-only filmstrips replay the final seed-0 conditioned checkpoint
on E1 instance 90001 at lambda zero and lambda `.10`:

- `storage-placement` shows the immediate placement fork and aligns every
  subsequent column on the same operational event: admission of B1 and B8,
  retrieval of B5 and B6, and final retrieval of B8. The rows are event-aligned,
  not wall-clock aligned; ellipses denote policy-specific decisions omitted
  between those events. Lambda zero finishes with three rehandles and lambda
  `.10` with none;
- `rehandle-vs-delivery` follows a common lambda-zero prefix to decision 12,
  where lambda zero reconfigures B6 and lambda `.10` delivers B6. The five
  consecutive post-fork decisions contain one versus zero additional
  rehandles.

Generate both from the authenticated checkpoint and serialized instance with:

```bash
bash experiments/conditioned_vcg/E01_benchmark_90k/render_illustrations.sh
```

Outputs are written below
`results/vcg-conditioned-final-comparison-90k-cpu-v3/illustrations/`. These
are post-hoc mechanism illustrations, not additional E1 performance evidence.
The earlier VCG 2.3-based filmstrips are not relabeled or reused.
