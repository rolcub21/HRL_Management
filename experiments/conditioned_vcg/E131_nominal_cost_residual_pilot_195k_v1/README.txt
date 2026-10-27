E131 MATCHED NOMINAL-COST RESIDUAL TRAINING PILOT
===============================================

Accepted output (the default):
    results/vcg-conditioned-e131-nominal-cost-residual-pilot-195k-v1-repair2
Original and repair1 directories contain development smoke artifacts only.
Do not resume their checkpoints with the final source contract.

Question
--------
Does explicitly accounting for known nominal immediate costs improve learning
over a matched total-cost critic? This is not another reward-weight, epsilon,
checkpoint-selection or mode-selection sweep. No legacy model is initialized
or repaired: both experimental arms start from scratch, seed 0.

Arms
----
total_q: learn all four total costs directly.
residual_q: expose total cost K + R, where K is the full nominal macro vector
(tardiness accrual, delivery earliness, equipment work, rehandles) and R is learned.
Waiting also receives its nominal cost, not a movement-only or zero-cost hack.
Observed cost is stored separately from nominal K and nominal duration.

The nominal predictor never reads unreleased-job schedules or durations. K is
not claimed exact under interruption, route repair or arrival interference.
For example, Defer can end at an arrival before its cap. The correction C-K can
be negative, so BOTH arms use identical signed linear output heads, trunks,
head parameter initialization and optimizers. Thus the control is signed-head
total-Q, not bit-for-bit the old nonnegative/softplus E123 critic. This shared
link change is explicit; any later claim relative to old E123 needs its own
comparison. Do not conflate that historical comparison with the matched test.

Correct TD construction
-----------------------
    total = K + R
    target_R = C_observed - K_current
                 + gamma**actual_duration * (1-terminal) * next_total_selected

The implementation trains an algebraically equivalent loss in total space:
    Huber(K+R, C_observed + discounted_next_total).
K has no gradients. This gives identical errors and parameter gradients to
the explicit residual-space loss, without adding K twice. All four heads
bootstrap from ONE next candidate, chosen by the unchanged grouped online
selector using combined total costs. Target scoring, frozen-target caches,
action selection and conservative preference anchoring all see combined costs.
The guard remains authoritative. Terminal transitions never bootstrap.

Frozen protocol
---------------
8x8 empty start, 36 jobs, gap-30 arrivals, Poisson-80 stays, release-fixed due
dates; objective J = 2*T + E + 0.1*equipment_work + 10*rehandles, tolerance 20,
gamma 1. No delivery bonus, separate wait penalty or discounted timing change.
Use E123 A_outflow_bounded frontier and guard. Same fast-v2 input cache, target
refresh (200 updates), batch size (128), learning rate schedule and exploration
(0.9 to 0.05 over 8000 training decisions) in both arms. Epsilon is logged by
actual decision count, not assumed to reach the floor at any episode.
Training instances and five greedy development instances are inherited from
E123. The development panel is already used; it is NOT a fresh test. E126
release-informed baseline results remain references, not newly fitted arms.

Budget and stopping rule
------------------------
One seed per arm, 60 episodes maximum, greedy validation every six episodes.
Endpoint 36 is a sanity gate, NOT the improvement decision. Continuing beyond
36 requires all 36 training episodes and all five endpoint-36 validation
episodes to complete, finite recorded TD diagnostics and maximum sampled
absolute TD error <= 1e6. This is a coarse runaway detector, not a smoothness
or statistical stability claim. Failures save the endpoint and stop.

Decide ONLY on the paired mean over checkpoints 48, 54 and 60, with the same
five instances at each checkpoint. No best-checkpoint selection. Advance only
if residual J is at least 4 lower and both early/late violation counts AND
mean early/late excess magnitudes do not increase. This is a conservative
screening rule, not a significance test or multi-seed superiority claim.
Anything smaller or with tail regression is inconclusive: do not tune this
workload further. At most one additional matched seed is a later explicit
decision. The independent test panel remains closed.

Report J and components, equipment work and weighted equipment cost, timing
cost and tolerance-edge crossings side by side, rehandles, release MAE,
admission wait, within-tolerance rate, completion, decisions, epsilon and TD
diagnostics. Attribution to the proposed mechanism needs supporting equipment
and timing changes; lower J by itself is not a mechanism demonstration.
Existing instrumentation also retains steps, primitive work and service metrics.

Preflight and disposable smoke
------------------------------
    cd /home/ai_diagnosis/HRL_Management
    E131=experiments/conditioned_vcg/E131_nominal_cost_residual_pilot_195k_v1/run.sh
    bash "$E131" preflight
    bash "$E131" smoke

The physical preflight first certifies an overdue state, executes admission,
then constructs a NEW frontier with stored work one step before an arrival.
It checks positive tardiness including delivery and movement across the event.
It never edits a state after certification. E130 separately records interrupted
Defer examples. Signed C-K corrections remain allowed rather than asserting
nominal execution is universally exact.

Smoke runs one disposable CPU episode per arm. If fewer than 128 transitions
were collected, it repeats disposable transitions to exercise a real backward
and Adam update and a resumable checkpoint roundtrip. No smoke weights or
repeated transitions are saved into experimental checkpoints. Smoke returns
are not quality evidence. Training is gated on preflight.json and smoke.json.

Launch sequentially (recommended on one GPU)
-------------------------------------------
    bash "$E131" train --arm total_q --through-episode 36 --device cuda
    bash "$E131" train --arm residual_q --through-episode 36 --device cuda
    bash "$E131" analyze

After checking the endpoint-36 sanity results:
    bash "$E131" train --arm total_q --through-episode 60 --device cuda
    bash "$E131" train --arm residual_q --through-episode 60 --device cuda
    bash "$E131" analyze

The same commands resume each arm from latest.pth with optimizer, replay,
counters and global RNG state. Latest includes complete histories. Do not edit
the contracted source mid-run. Use a fresh, labelled output for a changed
protocol; never relabel old total-Q weights as residual weights.

Monitoring
----------
    bash "$E131" analyze
Read e131-report.json under the accepted output. Per-arm seed-0/summary.json
contains episode and greedy validation results, losses and sampled TD ranges.
Both arms must finish 60 before the decision is available. Concurrent runs on
one GPU are discouraged because contention invalidates runtime comparisons.
No Git commit, push or long training run is performed by preparing this code.
