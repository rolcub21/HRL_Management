# E270: 8x8 multiseed prefilled certification curve

Frozen evaluation counterpart to E269. It screens deterministic random 8x8
starts at 18, 25, and 32 occupied cells, retaining ten per fill only when the
exact `B=20,000` verifier returns `SAFE` with a concrete witness. `UNSAFE` and
`UNKNOWN` candidate-start discards remain in `instances.json`.

The live arms are the three E222 VCG checkpoints and one shared Dynamic ranker,
each with and without the certification filter. The unfiltered arms retain a
passive exact monitor. No training occurs. A3C placement is intentionally out
of scope until its separate 5x5 whole-controller certification contract passes.

Run the development gate before generating the protected state panel:

```bash
bash experiments/conditioned_vcg/E270_e222_multiseed_prefilled_certification_329k_v1/run.sh gate --device cuda
bash experiments/conditioned_vcg/E270_e222_multiseed_prefilled_certification_329k_v1/run.sh prepare --device cpu --workers 6
```

After preparation, run these four commands in parallel:

```bash
bash experiments/conditioned_vcg/E270_e222_multiseed_prefilled_certification_329k_v1/run.sh evaluate-vcg --model-seed 0 --device cuda
bash experiments/conditioned_vcg/E270_e222_multiseed_prefilled_certification_329k_v1/run.sh evaluate-vcg --model-seed 1 --device cuda
bash experiments/conditioned_vcg/E270_e222_multiseed_prefilled_certification_329k_v1/run.sh evaluate-vcg --model-seed 2 --device cuda
bash experiments/conditioned_vcg/E270_e222_multiseed_prefilled_certification_329k_v1/run.sh evaluate-dynamic --device cpu
```

Then run `analyze --device cpu`. The state panel is blocked before evaluation
if the aggregate `UNKNOWN` discard fraction exceeds 25% at any fill; that
predeclared rule prevents silently selecting an easy 90%-fill panel.

For an unattended server run, `run_overnight.sh` executes the gate and state
screen first, starts all four evaluation shards only after both pass, waits for
them, and analyzes the complete result. It is resumable at episode-row level.
