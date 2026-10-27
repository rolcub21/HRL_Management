E134 PROSPECTIVE SIX-REGIME GRID AND WAIT-COST SENSITIVITY
========================================================

This records prospective design choices and implements trajectory rescoring.
It is NOT an executable workload generator or mixture-training launcher.
Generator/reference verification, load calibration and interface alignment
remain closed execution gates listed in protocol.json. No contender-quality
grid runs, training, test-instance generation or test-panel opening occurs here.

Primary objective stays unchanged:
    J = 2*T + E + 0.1*L + 10*R, release-fixed due dates, tolerance +/-20.
Only this objective trains the four-head nominal-cost residual controller.

Secondary evaluation-only sensitivity, fixed BEFORE grid tests:
    J_W = J + 0.5*sum_i(W_i).
Rescore EVERY method's SAME recorded trajectories. Do not retrain, change a
policy, select a different checkpoint or change the weight after seeing ranks.
Use the same per-cell comparator frozen on primary development J; report full
method rankings on both objectives and any reversals. J_W does not decide
primary hypotheses. No fifth head is added. An auxiliary wait-value head
would itself change training and would not establish arbitrary-preference
policy optimality without a suitable conditioning/improvement procedure.

Admission-wait accounting
-------------------------
W_i = first COMPLETED storage time minus release time for incoming jobs.
This is consistent with our previous wait-before-storage metric, but includes
admission travel and handling; it is not solely stationary pickup waiting.
It is the SUM of job-steps, not the mean. For example, 36 incoming jobs with
mean wait 40 add 0.5*36*40 = 720 cost units. This potentially large effect is
intentional as a fixed sensitivity, not a reason to retune it after tests.

Initial stored inventory contributes zero admission wait during the measured
episode; do not charge prehistory or dilute the incoming-job mean with initial
jobs. Reconfiguration never restarts W_i. At truncation, arrived/unadmitted
jobs accrue observed wait to the horizon and are explicitly right-censored.
Unreleased jobs accrue zero observed wait but remain in completion accounting.
Failed trajectories retain partial costs and censoring records, but cannot
win a completed-cost comparison by ending early. Completion is reported first.
J on incomplete trajectories uses observed tardiness accrual and observed
deliveries only, not a fabricated future completion date.

Locked design choices
----------------------
8x8 only first; 0/approximately40/approximately70 percent initial fill crossed
with nominal offered loads .60/.85. Same incoming-stream length in every cell,
with initial inventory additional. Calibrate measured reference incoming-job
equipment work per job divided by arrival interval; report initial-clearance
and total finite-episode burden separately. Freeze a bounded iteration rule
and intervals before evaluating contenders. Do not assume work rises with fill.

Reference must be an independent full-information dispatcher, not VCG's
certificate/witness/guard or any contender's own scheduler/retrieval planner.
Verify its primitive trace separately. Set d_i=C_reference_i+Uniform[0,20],
which places its deliveries within tolerance. Report reference work, rehandles,
admission wait, completion time and a transport lower bound. A legal reference
proves simulator feasibility, not absence of reference-distribution bias.

One mixture-trained residual model per seed (three seeds) covers all six cells.
Informed Dynamic, informed GA and certified Dynamic run every cell. The two
certified methods must share the actual frontier, guard and Defer conditions.
All methods share executable macros, due-date information, objective and horizon.
Eligibility filtering is not identical between plain and certified methods.

Twenty fresh test instances per cell, disjoint from calibration/train/dev.
Five-percent mean-J noninferiority margin for empty cells. Choose the primary
comparator per cell on development only. Completion first; conditional common-
completion costs explicitly labeled with sample size. Preserve every cell and
failure. Declare the exact hypothesis family/multiplicity procedure before
test execution. No claim that 20 observed completions proves universal safety.

Commands
---------
    cd /home/ai_diagnosis/HRL_Management
    .venv/bin/python -B -m experiments.conditioned_vcg.E134_regime_grid_protocol_198k_v1.run prepare

For a recorded physical trajectory:
    .venv/bin/python -B -m experiments.conditioned_vcg.E134_regime_grid_protocol_198k_v1.run rescore \
        --trajectory /absolute/path/trajectory.json \
        --rescore-output /absolute/path/new-sensitivity-result.json

Trajectory JSON schema:
    observation_end: nonnegative primitive time
    equipment_work: non-WAIT equipment action count
    rehandles: physical rehandle count
    strict_complete: boolean (must reflect physical and method success)
    jobs: array of {label, release, due, first_storage, delivery, initially_stored}
Timestamps are absolute; unobserved storage/delivery timestamps are null.
The input file is never changed. Results include its SHA256, primary components,
J, J_W, wait records/censoring, and completed-cost ranking eligibility. Existing
result files are not overwritten. A completed trace's reconstructed primary J
must also agree with the native execution log before inclusion in the grid.

Next implementation gates
--------------------------
Independent reference + primitive verifier; finite-horizon initial-state and
arrival generator; bounded calibration rule; shared certificate/guard/Defer and
information-access audit; training mixture/budget/exploration/checkpoint rule;
GA budget; disjoint seed manifests and full statistical plan. Finish these
before claiming the FULL grid execution protocol is frozen or launching training.
Stage B remains deferred until the grid result.
