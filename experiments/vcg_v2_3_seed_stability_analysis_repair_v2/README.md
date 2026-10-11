# VCG V2.3 seed-stability analysis repair V2

This is an append-only deterministic analysis repair. It does not train or run
an environment. The frozen validation rows contain signed delivery deviations
and MAE, but the frozen analyzer also expected tardiness, earliness, and window
rate fields that the producer did not serialize.

The repair authenticates the original manifest, all three seed completions,
selected ledgers, frozen sources, and the repaired-baseline trust chain. It
derives the three summaries only in memory, verifies MAE against the signed
deviations, invokes the unchanged frozen aggregation and gate, and writes a
separate self-hashed contract/report/audit. The original result tree remains
byte-identical and the 86xxx panel remains sealed.

Run:

```bash
bash experiments/vcg_v2_3_seed_stability_analysis_repair_v2/run.sh
```
