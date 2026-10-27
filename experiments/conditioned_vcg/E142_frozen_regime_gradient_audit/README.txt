E142 FROZEN REGIME GRADIENT AUDIT
================================
No training, optimizer steps, rollouts or test instances. Reads the current
E138/E139 checkpoint, including retained replay. Does not create episode-60
checkpoints or remove stopped-run gates.

Run from HRL_Management, one seed at a time:

  MPLCONFIGDIR=/tmp/e142-mpl .venv/bin/python -B -m experiments.conditioned_vcg.E142_frozen_regime_gradient_audit.run --model-seed 0

Repeat with --model-seed 1 and --model-seed 2. CPU defaults to one thread;
--device cuda is optional. Defaults: three disjointly drawn (not guaranteed
nonoverlapping) batches of eight transitions per regime. Samples are uniform
within regime for equal-size comparison. Gradient norms are BEFORE clipping.
Actual training uses mode-balanced rather than regime-balanced sampling; its
regime shares are estimated separately using 200 native-sampler batches.

Output: results/vcg-conditioned-e142-frozen-regime-gradient-audit/seed-N/result.json
and REPORT.md (compact exposure, TD-loss, gradient-norm and alignment tables).
Existing output is never overwritten; pass a new --output for another audit.
Each batch is saved immediately. Interrupted/failed runs retain status running
with their partial batches and final invariance checks; do not treat as complete.

Reports:
- completed-episode exposure and retained replay share per regime;
- episode J distributions from training history;
- per-head absolute TD/target distributions and Huber saturation fractions;
- native weighted per-head gradients and shared-trunk norms;
- conservative anchor loss, pairs, weighted gradient norm;
- complete native-loss preclip gradients and shared gradient alignment.

Regime labels are reconstructed from completed history decision counts and the
FIFO replay suffix. Every terminal boundary and total transition count must
agree. No inferred labels are accepted after an accounting mismatch.

All four heads retain the same native combined-cost-selected successor. The
audit never substitutes separate per-head optimal backups. Native Huber beta,
head loss weights and anchor weight are retained. No normalization is applied.
Parameter/target hashes, counters, replay length, agent RNG and source checkpoint
hash must remain unchanged. Diagnostic lists and input/target caches may grow
in memory only; optimizer state is neither stepped nor saved.

Interpretation limits:
This measures current frozen replay gradients, not the gradients seen throughout
training. A single checkpoint cannot establish that TD error stalled over time.
Three small batches give noisy gradient alignment; increase batches/size before
drawing a training-design conclusion. Current replay may omit early experience.
Large J alone does not imply gradient domination with Huber loss. Negative
shared-gradient cosine suggests local interference, not a causal proof of
forgetting. Inspect anchor contribution and native sampling shares as well.
No automatic retraining, cost scaling or stopping-gate bypass follows this audit.
