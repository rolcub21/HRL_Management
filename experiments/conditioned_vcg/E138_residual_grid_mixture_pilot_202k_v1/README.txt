E138 ONE-SEED RESIDUAL GRID MIXTURE TRAINING PILOT
================================================
This is NEW Stage-A training FROM SCRATCH, not E137's random-policy readiness
run and not continuation of the old easy-workload checkpoints. Seed 0 only.
No Stage B, sealed test generation, paper checkpoint eligibility or superiority
claim. Keeps E131 four signed cost heads, nominal K + residual, common selected
successor across heads, Double DQN, replay/target caches and fast E123-v2 inputs.

Repair1 fixes duplicate reserved metadata in the initial checkpoint save.
Default results use the separately contracted -repair1 output. The original
pre-training contract is preserved; no checkpoint/training work is overwritten.

    cd /home/ai_diagnosis/HRL_Management
    bash experiments/conditioned_vcg/E138_residual_grid_mixture_pilot_202k_v1/run.sh prepare
    bash experiments/conditioned_vcg/E138_residual_grid_mixture_pilot_202k_v1/run.sh train --through-episode 6 --device cuda

First run: SIX training episodes (one per regime), then six greedy development
episodes. It may take substantially longer than E137: it now performs TD updates.
Do not start alongside other GPU-heavy experiments if timing is being assessed.
Each training episode is capped at 20 minutes; each greedy episode at 5 minutes.
Worst-case first-block episode budget is about 2.5 hours plus generation/I/O;
actual runtime needs measurement. No promise based on inference-only E137 time.

    bash experiments/conditioned_vcg/E138_residual_grid_mixture_pilot_202k_v1/run.sh status

After reviewing the first block, the SAME command with --through-episode 60
resumes up to 60 TOTAL episodes, not 60 additional. Do not resume past a recorded
failure without inspection and explicit --retry-failed. GPU/CUDA is required for
the long launcher. No long training/validation was launched while preparing it.

Frozen pilot choices
--------------------
Balanced shuffled cycles of all six fill/load regimes; 60 episodes = ten each.
Fresh training seeds 21000001..21000060; greedy development seeds 22000000..22000005
(one fresh instance per regime, reused across checkpoints). Neither overlaps
E135 calibration or E137 readiness. Test panel remains sealed/unconstructed.
Fixed E135 calibrated gaps; independent references construct external due dates.
Reference traces are stored for verification, never attached to policy inputs.

Primary J = 2*T + E + 0.1*L + 10*R; tolerance +/-20, external release-fixed dates.
J_W is evaluation-only and does not train/select checkpoints. No fifth head.
One optimizer opportunity per macro, inherited batch/update/target/anchor
semantics, inherited learning-rate schedule, gamma=1. Epsilon .90 -> .05 linearly
over 6000 TRAINING decisions, then floor. Validation uses a separate frozen
evaluator at epsilon zero and restores training global RNG; it cannot consume
the training exploration schedule or mutate optimizer/replay.

Greedy validation every six episodes. Save latest after every completed episode,
with weights/target/optimizer/replay/agent sampler/global RNG and histories.
Checkpoints are for pilot continuation, not deployment. Validation rows retain
all endpoints, with prospective final reading of 48/54/60 rather than best-point
selection. Fixed dev panel is small: no statistical superiority conclusion.

Stops and recovery
------------------
Stop immediately on any incomplete training/greedy episode, wall/step/decision
cap, macro/certificate failure, nonfinite loss or |TD error|>1e6. Episodes capped
at 6000 primitive steps,1000 decisions; certificates at 2000 nodes. Limits are
pilot runtime rules, not a mathematical uniform completion guarantee. Prior
E137 timeout remains recorded; it did not prohibit this authorized pilot.

Interrupted/incomplete episode updates are NOT committed to latest.pth: resume
rolls back that episode, replay, optimizer and RNG to the last completed save.
stopped.json records the stop. --retry-failed acknowledges/archives the stop;
it is not automatic permission to change budgets or silently drop failures.
If scheduled validation is pending, resume validates the unchanged checkpoint
before doing any further training. Ctrl-C/SIGTERM saves a stop record and exits.
No in-episode continuation. Hard kill may prevent the stop record, but previous
atomic checkpoint remains. Do not run two training writers on the same output.

Limits deliberately retained
----------------------------
VCG and certified Dynamic share the A-outflow certified shell, guard and Defer.
Native plain baseline compound-versus-elementary macro alignment remains a final
comparison issue, NOT a blocker to this pilot. No joint-inflow guarantee or
uniform <=6000 completion theorem is claimed. Dense packing is corridor-
conditioned with avoidable obstruction, not proven necessary rehandling.
Source/contract/budget changes require a new version/output, never silent resume.
