# Target-size VCG-Dense Stage A v3

V3 retains the V2 dense objective, episodic agent, exact certified frontier,
D12/E14 acceleration, optimizer, replay sampling, and update cadence. It
changes only the implementation of no-gradient TD continuation scoring:
repeated references to the same immutable recovery state are encoded once and
then gathered back into the original candidate order.

The deployed selector and gradient-bearing chosen-action prediction use the
native implementation. CUDA retained-replay benchmarks passed numerical
parity and gave a 2.04x update speedup with one host thread. Eight threads were
slightly slower, so the default remains one.

V3 uses a fresh output directory. Do not copy or resume V2 checkpoints under
this protocol. Run the two-episode timing gate before considering the full
budget:

```bash
bash experiments/conditioned_vcg/target_size_dense_stage_a_v3/run.sh self-check
bash experiments/conditioned_vcg/target_size_dense_stage_a_v3/run.sh prepare
bash experiments/conditioned_vcg/target_size_dense_stage_a_v3/run.sh train-stage-a \
  --scenario empty_8x8_n36 --model-seed 0 --device cuda \
  --stop-after-episode 2
```

The full 500-episode budget remains unauthorised until episode-2 wall time and
the recorded state-identity reduction are reviewed. Stage B remains blocked.
