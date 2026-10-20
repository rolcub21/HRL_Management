# E123 v2: cache-corrected continuation

This resumes the stopped E123 v1 A/B experiment. It does not restart learning, change the objective, or enable arm C. The original code, outputs and checkpoints remain untouched.

Two execution corrections:

- Value-identical immutable current observations share one object instead of one copy per candidate. The sharing key includes the full observation, including block clocks, public queue context and progress-clock context. Candidate order, action features and successor states are unchanged.
- The weight-independent graph-input cache survives episode and validation boundaries. Its capacity is bounded at 1,000,000 state slots and it is released when the arm exits. The existing cache implementation safely clears on capacity exhaustion. No online-network embeddings or scores are cached; frozen-target score invalidation remains unchanged.

Migration preserves model/target weights, Adam state, replay transition values and order, local/global RNG states, training counters, epsilon, validation history and the 120-episode endpoint. It coalesces the old replay's duplicate current-state objects so existing samples benefit immediately. Every transition is compared by value before it is accepted. A `migration.json` records the source checkpoint hash and migration episode. Resume only from the new output after migration.

The CUDA preflight compares an actual 128-transition source replay batch using the learned weights. It requires exact equality of predictions, four-head TD targets, anchor terms, one Adam update including optimizer/RNG/counters, and a resumable checkpoint round trip. Those optimizer updates are disposable diagnostics; they never enter the resumed training files. The source checkpoint is hashed before and after the probe. Timing is batch-level evidence, not a whole-episode speedup claim.

## Launch or resume

```bash
cd /home/ai_diagnosis/HRL_Management
E123V2=experiments/conditioned_vcg/E123_absolute_progress_8x8_187k_v2/run.sh

# Stop the old E123 process before these steps.
bash "$E123V2" preflight --device cuda
bash "$E123V2" migrate --model-seed 0
bash "$E123V2" train-matched --model-seed 0 --through-episode 120 --device cuda
bash "$E123V2" analyze
```

Migration is one-time and refuses to overwrite an existing v2 checkpoint. After interruption, rerun only `train-matched` (and then `analyze`); do not migrate again. A and B remain sequential. Their saved episode numbers and exploration counters are retained, not reset. Arm C still requires the original explicit authorization and a separately migrated source checkpoint; this continuation does not implicitly launch it.

Outputs are under `results/vcg-conditioned-e123-absolute-progress-8x8-187k-v2`. `e123-v2-report.json` separates inherited v1 episode history from newly timed v2 episodes. No speed ratio should mix the two phases as if both used the corrected implementation. Hardware, cache warm-up, policy/frontier changes and checkpoint/validation overhead also affect full-run time.

If launched detached in a tmux session named `e123-v2`, inspect it with:

```bash
tmux attach -t e123-v2
```

Detach without stopping training with `Ctrl-b`, then `d`. This remains a development comparison; no independent test panel is opened.
