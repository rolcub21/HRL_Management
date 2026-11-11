# E270: dual-size multiranker prefilled certification curve

Frozen evaluation counterpart to E269. It screens deterministic random 8x8
starts at 18, 25, and 32 occupied cells, retaining ten per fill only when the
exact `B=20,000` verifier returns `SAFE` with a concrete witness. `UNSAFE` and
`UNKNOWN` candidate-start discards remain in `instances.json`.

The 8x8 live arms are the three E222 VCG checkpoints and one shared Dynamic
ranker, each with and without the certification filter. The suite also runs
the three finalized E247 A3C placement checkpoints at both 5x5 and 8x8, again
with paired certified and physical-unfiltered arms. The 5x5 A3C runs use the
same 100 witnessed-safe E263 states; the 8x8 A3C runs use E270's 30 states.
Every unfiltered arm retains a passive exact monitor. No training occurs.

For A3C, certification covers the whole controller: placement cells and
scheduler Reconfigures are filtered, while Deliver remains unfiltered because
removing a block cannot reduce retrievability. Both twins use the same fixed
macro scheduler, the same bounded-Defer rule, and the same stateless stochastic
placement order. Neither uses VCG's progress guard. Failures are separated into
unsafe selections and step-limit stalls.

Run the development gate before generating the protected state panel:

```bash
bash experiments/conditioned_vcg/E270_e222_multiseed_prefilled_certification_329k_v1/run.sh gate --device cuda
bash experiments/conditioned_vcg/E270_e222_multiseed_prefilled_certification_329k_v1/run.sh prepare --device cpu --workers 6
```

After preparation, the four VCG/Dynamic commands and these six A3C commands
may be run in parallel:

```bash
bash experiments/conditioned_vcg/E270_e222_multiseed_prefilled_certification_329k_v1/run.sh evaluate-vcg --model-seed 0 --device cuda
bash experiments/conditioned_vcg/E270_e222_multiseed_prefilled_certification_329k_v1/run.sh evaluate-vcg --model-seed 1 --device cuda
bash experiments/conditioned_vcg/E270_e222_multiseed_prefilled_certification_329k_v1/run.sh evaluate-vcg --model-seed 2 --device cuda
bash experiments/conditioned_vcg/E270_e222_multiseed_prefilled_certification_329k_v1/run.sh evaluate-dynamic --device cpu
for size in 5x5 8x8; do
  for seed in 0 1 2; do
    bash experiments/conditioned_vcg/E270_e222_multiseed_prefilled_certification_329k_v1/run.sh evaluate-a3c --size "$size" --model-seed "$seed" --device cpu &
  done
done
wait
```

Then run `analyze --device cpu`. The state panel is blocked before evaluation
if the aggregate `UNKNOWN` discard fraction exceeds 25% at any fill; that
predeclared rule prevents silently selecting an easy 90%-fill panel.

For an unattended server run, `run_overnight.sh` executes the eight-route gate
and state screen first, starts all ten evaluation shards only after both pass,
waits for them, and analyzes the complete result. It is resumable at
episode-row level. A3C shards use CPU, leaving the GPU to the VCG shards.
