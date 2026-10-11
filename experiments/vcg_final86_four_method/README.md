# Final 86xxx four-method comparison

This is the prospective confirmation experiment for:

- VCG 1.1 (three frozen selected model seeds)
- VCG 2.3 (three frozen stability-selected model seeds)
- Duration-aware Dynamic PSLAP
- Capacity-aware Park--Seo-2009 rolling GA

Every method uses the exact same 30 serialized `EpisodeInstance` objects,
seeds 86000--86029. The frozen generator is the legacy matched regime:
5x5, 8 blocks, exponential arrival **rate** 10 (mean gap 0.1), and Poisson
storage duration mean 80.

The complete grid has 600 rows: 90 VCG 1.1, 360 VCG 2.3, 30 PSLAP, and
120 GA. Analysis first averages each method's nuisance/model realizations
within an EpisodeInstance, leaving 30 paired observations per method.

## Run sequence

Preparation authenticates and freezes inputs but does not construct or load
an 86xxx instance:

```bash
bash experiments/vcg_final86_four_method/run.sh prepare
```

Open the final panel exactly once, explicitly:

```bash
bash experiments/vcg_final86_four_method/run.sh open-panel OPEN_FINAL_86XXX_ONCE
```

Run or resume all 600 rows:

```bash
bash experiments/vcg_final86_four_method/run.sh run
```

Then authenticate and aggregate:

```bash
bash experiments/vcg_final86_four_method/run.sh analyze
```

`status` is read-only. Individual methods can be resumed with
`run-method vcg_1_1`, `vcg_2_3`, `duration_aware_dynamic_pslap`, or
`duration_aware_pslap_ga_2009_rolling_capacity_aware_partial`.

Learned policies require CUDA; baselines run on CPU. Do not edit bound source
or checkpoint artifacts between panel activation, evaluation, and analysis.

