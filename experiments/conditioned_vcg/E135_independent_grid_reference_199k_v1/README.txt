E135 INDEPENDENT GRID REFERENCE — DEVELOPMENT GATE
================================================
Implements the generator, independent reference dispatcher, separately coded
primitive legality/replay verifier and bounded load calibration following E134.
No VCG, witness, certificate, guard or contender scheduler/planner is imported
by the reference or verifier. Shared primitive simulator is the physical model.
The verifier checks wall/block collisions explicitly: simulator movement alone
does not enforce all of them. It also checks pickup/drop identity, storage and
delivery legality, full completion, timestamps, work and constructed due windows.

    cd /home/ai_diagnosis/HRL_Management
    bash experiments/conditioned_vcg/E135_independent_grid_reference_199k_v1/run.sh calibrate

This is CPU-only reference construction, NOT training. No checkpoint needed.
Contracts authenticate design/source before calibration. Changed source requires
a new --output; existing case files cannot be overwritten with changed content.

Workloads: 36 incoming jobs at integer fixed intervals; Poisson-80 nominal stays.
Initial inventory is additional, with age zero. Initial fill floors 35*fill:
0,14,24 cells (0%,40%,68.6%). The distribution reserves the left column and
bottom interior row, independent of pickup occupancy. Other cells are shuffled
and filled to that count. This is a CORRIDOR-CONDITIONED distribution, not a
claim about arbitrary 70%-full packings. No contender result selects placements.

Reference: reachable retrievals timed to nominal storage stays have priority;
otherwise FIFO admission to deepest reachable free storage, otherwise WAIT.
Own BFS, deterministic tie breaking, no reconfiguration. Reference can wait for
retrieval to make admission possible; it is NOT a joint-flow-certified policy.
It has full information and sees nominal durations, not constructed due dates.
Due dates are assigned afterward: delivery time + independent integer U[0,20].
They are separate external_due_dates, NOT substituted into storage durations.
Their future manifest/reference trace must remain hidden from contenders.

All six cells use the same three calibration seeds, disjoint from historical
panels. Offered incoming load = mean incoming-attributed equipment work/job /
gap. Empty approach travel is attributed to the macro's serviced job. Initial
clearance work is separate; this load is NOT total finite-episode utilization.
The report includes both, reference makespan, wait, rehandles and a geometric
transport-work lower bound. Slow reference schedules can make timing too easy;
inspect/report this quality and do not equate feasibility with lack of bias.

Calibration starts at round(30/load), updates round(measured work/load), maximum
8 iterations, tolerance 0.03 absolute load. Cycle/stagnation fails the gate;
no silent relaxation or selective seed replacement. Each trace is verified.
The 10,000-step cap bounds reference construction ONLY, not the task horizon.

Still NOT a training launcher. External dates need explicit observation/critic
integration; VCG and certified Dynamic must pass actual frontier/guard/Defer
alignment. Existing 36-job/gap-30 liveness assumptions cannot be copied to this
grid with up to 60 jobs and calibrated intervals. Horizon, mixture schedule,
training budget, seed manifests, GA budget and statistical families remain gates.
Do not open test instances or start long training until those pass. E134 and
old experiments/checkpoints are unchanged. No fifth wait head is introduced.
