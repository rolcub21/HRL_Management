E144 WARM-START RESIDUAL MIXTURE ADAPTATION, SEED 0 ONLY
======================================================
Adaptation, not zero-shot transfer. No scaling, head reweighting, new features,
new certificate, new guard or objective change. Single seed first; no queue
for seeds 1/2. Long training is USER-LAUNCHED.

Start:
  cd /home/ai_diagnosis/HRL_Management
  MPLCONFIGDIR=/tmp/e144-mpl .venv/bin/python -B -m experiments.conditioned_vcg.E144_warm_start_mixture.run train --through-episode 360 --device cuda

For a bounded first chunk use --through-episode 6; the same command with a
larger endpoint resumes. Endpoints count NEW adaptation episodes, not the
source's 60 pretraining episodes. Maximum360 =60 new episodes per regime.
Total training lineage can therefore reach420 episodes. This may be a
multi-day run, not necessarily an overnight run. No fixed runtime promised.

Status:
  .venv/bin/python -B -m experiments.conditioned_vcg.E144_warm_start_mixture.run status

Initialization:
- Source: E141 authenticated compact original E131 residual seed0 endpoint60.
- Copy online AND target network weights; preserve source files read-only.
- Fresh optimizer, empty replay, zero adaptation counters/RNG seed0.
- Same mixture epsilon .90 -> .05 in6000 adaptation decisions. It is reset,
  NOT continued from the old checkpoint; random actions may temporarily damage
  performance. Greedy validation, not exploratory returns, decides quality.
- Same native loss, head weights, anchor, LR schedule and fast E123-v2 caches.
- Training/validation execution budget6000,1000macros,2000certificate nodes.
- CRITICAL: source queue/progress input normalization3600 is retained. Changing
  units to6000 would invalidate a weight-only warm start. Inputs clip beyond
  native normalization range; no new uniform completion-bound claim is made.
- First60 adaptation workloads match E138 instances/order; extend the same
  deterministic six-regime permutations and disjoint training seeds to360.

Validation:
- Frozen greedy at adaptation0,6,30,60,...360; all six fixed E141 development
  workloads plus five original native controls. Test panel stays sealed.
- Reuse E141 matched completed certified Dynamic rows as cost references.
- One changed-workload development instance per regime remains a limitation;
  passing is authorization to consider replication, not statistical paper proof.
- Development gate: all11 cases complete; easy/.60 J no more than10% above
  adaptation0 J (~73.1 for source seed0); each of four congested cases J no
  greater than certified Dynamic on that same instance. BOTH endpoints330 and
  360 must pass before additional seed training is proposed. No best-checkpoint
  selection or automatic seed expansion.
- Validation timeouts (900s, matching E141) or incomplete episodes are retained
  as failures but do NOT stop training. Unexpected errors still stop; they are
  not disguised as poor policy performance.
- Training incomplete/timeout/nonfinite/TD>1e6 stops, preserving the last full
  completed-episode checkpoint with optimizer/replay/RNG. No automatic retries.
- Ctrl-C/SIGTERM stop; unfinished episode work rolls back on future resume.
  A stopped.json requires explicit review before retry; no automatic bypass.

Outputs: results/vcg-conditioned-e144-warm-start-mixture/
  contract.json, seed-0/latest.pth, seed-0/summary.json,
  validation/ep-N-case-I.json, progress.json, optional stopped.json.
The source checksum, settings and implementation hashes are pinned in contract.
Existing contracts reject changed recipes/source code. No files are erased.

Interpretation:
This combines warm start and a larger exposure budget; success cannot isolate
which caused improvement. Replay share alone does not establish adequate
experience. Forgetting would motivate further work, but does not establish
normalization as its remedy. Native plain-baseline macro mismatch is unchanged.
