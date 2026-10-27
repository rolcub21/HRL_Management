E164: fixed-order admission, storage-clock training
===================================================

Purpose
-------
Train the certified residual VCG in the static yard-allocation scenario used
by the literature baselines.  Every job is available at step zero.  Jobs are
admitted in the fixed B1..BN order as soon as a joint-SAFE Accept placement is
available.  VCG chooses the safe cell, reconfiguration and retrieval, but not
admission timing or order.

Timing and objective
--------------------
Each deadline starts when its block enters storage:

    d_i = stored_time_step_i + storage_steps_needed_i

Storage durations are positive Poisson with mean 300.  The objective is
J=2*T_storage + E_storage + 0.1*equipment_work + 10*rehandles.  Admission wait
has no cost because admission timing is not a controller decision.

Runtime design
--------------
The static batch contains 4--28 jobs (10--80% of the 35-cell capacity), the
primitive safety cap is 1200, replay batch size is 32, and learning runs every
two decisions.  Validation at episodes 3, 6 and 12 contains only three static
batches of 7, 18 and 28 jobs.  No working-day or full-day evaluation is run.

Run
---

    bash experiments/conditioned_vcg/E164_fixed_admission_storage_clock_223k_v1/run.sh prepare --device cpu
    bash experiments/conditioned_vcg/E164_fixed_admission_storage_clock_223k_v1/run.sh preflight --device cpu
    bash experiments/conditioned_vcg/E164_fixed_admission_storage_clock_223k_v1/run.sh smoke --device cpu
    bash experiments/conditioned_vcg/E164_fixed_admission_storage_clock_223k_v1/run.sh train --through-episode 3 --device cuda

Episode 3 is the first decision gate.  The test panel remains sealed.
