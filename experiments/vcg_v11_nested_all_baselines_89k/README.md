# Nested VCG endpoints with all matched comparators on 89k

This additive, inference-only experiment evaluates the four missing comparators
on the exact serialized `89000..89029` EpisodeInstances already used by the
successful nested-VCG confirmation. It does not regenerate the panel, retrain a
policy, or rerun VCG. The plots retain only the two requested nested operating
points: VCG (`lambda = 0`) and VCG with handling (`lambda = 0.2`).

The four new resumable grids follow the established matched-baseline protocol:

- historical VCG 2.3: 3 model seeds x 4 policy rolls x 30 instances = 360 rows;
- Duration-aware Dynamic PSLAP: 30 deterministic rows;
- capacity-aware rolling GA: 4 optimizer rolls x 30 instances = 120 rows; and
- Kim2020 adaptation: 3 model seeds x 5 policy rolls x 30 instances = 450 rows.

That is 960 new rows. The two VCG endpoints reuse 180 existing authenticated
rows. No complete-case filtering is allowed: if any cell of a method fails,
its aggregate endpoint is suppressed. The cumulative plot may show only its
contiguous safe prefix ending before the first failed instance.

Run all phases sequentially:

```bash
bash experiments/vcg_v11_nested_all_baselines_89k/run.sh run-all
```

Every phase is resumable. To run or resume phases separately, use `run-v23`,
`run-baselines`, and `run-kim`, followed by `plot`. `inspect` reports ledger
counts without running policies. Outputs include a three-panel cumulative
return/MAE/rehandling figure and a matched MAE-rehandling endpoint plot.
