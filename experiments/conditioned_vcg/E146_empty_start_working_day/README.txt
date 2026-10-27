E146 FIXED WORKING DAY, EMPTY START, FROZEN ORIGINAL RESIDUAL VCG
================================================================
One day scenario shared by every volume level and every method. No training.
This replaces neither E144 nor E145; older results remain unchanged.

Frozen prospective settings:
- original8x8 layout with35storage cells; EMPTY at time0 at every level;
- day=5400primitive steps; positive Poisson1200 durations at every level;
- nominal overlap20/50/80% ->32/79/126blocks/day (round nearest);
- independent uniform random release times over the same day. Simulator
  requires the first release at0: one job is anchored there, the otherN-1
  releases are iid Uniform integer[1,5399], sorted. Simultaneous arrivals allowed.
- due=release+duration, tolerance20; unchanged J=2T+E+0.1L+10R;
- paired development seeds24060000..24060002, common random numbers across
  volume levels. Independent arrival/duration streams. No generated reference
  schedule, no witness dates, no initial inventory, no rejection by outcomes.

Capacity calculation (not a contender-performance calibration):
  N = round(target *35*5400/1200)
  planned transporter utilization = N*30/5400 =17.8%,43.9%,70.0%.
30 is a declared planning allowance in physical equipment steps/job, NOT a
proof of maximum crane work. Direct empty-yard observations were around20;
obstruction/rehandling can raise actual work. Report measured utilization,
queues and drain time rather than assert the heavy day is guaranteed feasible.
With800-duration jobs at equal target a30-step allowance would overload;1200
was chosen algebraically for70% allowance utilization in the heavy case.

The overlap formula is a nominal, long-run approximation, not actual average
stored occupancy. Empty-start/end transients, storage admission delay and early
deliveries affect actual fullness. Peak/mean stored occupancy and outside queue
are reported separately. If heavy-day occupancy stays low, report that result;
do NOT silently retune arrivals using method wins. Any extra level is a new
prospectively labelled extension, not replacement of these outcomes.

Methods: original frozen E131/E133 residual episode60 models,3seeds; informed
Dynamic PSLAP; informed native GA with same fixed compute budget; certified
Dynamic using the SAME original-model base preparation, certificate, witness
guard and E137 flow-event Defer semantics. All methods see current/past release
information only. Plain compound vs certified elementary macro mismatch remains.
The longer durations and random release times are an explicit common shift
from model training (36jobs,gap30,Poisson80). No normalization/input changes:
source3600 queue/progress units remain; clipping after3600 is a limitation.

Arrival cutoff is NOT episode termination. Execute until all jobs are delivered.
Common safety caps:16200steps,4000macros,2000certificate nodes,1800wall seconds.
Cap/timeout is truncation/failure, not completion. Never rank its partial J
against completed J. A finite cap is necessary operationally; it is no proof
of eventual completion. No unfinished work disappears at day-end.

Outputs per method/day:
completion, J/components, service MAE/window, rehandles, admission wait,
steps/delivery, decision latency, all primitive records;
day_statistics: peak stored/day peak/time-weighted day occupancy, peak outside
queue, non-WAIT transporter work divided by5400, total utilization, drain time,
unfinished jobs, censoring. For truncation before day-end, day-integrated
statistics are lower bounds. If all jobs finish early, remaining day is idle.
No storage/physical waiting time is mislabeled as equipment work.

Prepare:
  MPLCONFIGDIR=/tmp/e146-mpl .venv/bin/python -B -m experiments.conditioned_vcg.E146_empty_start_working_day.run prepare
Run/resume (54 episodes:9days x6controllers):
  MPLCONFIGDIR=/tmp/e146-mpl .venv/bin/python -u -B -m experiments.conditioned_vcg.E146_empty_start_working_day.run run
Status: same module with status. All files under:
  results/vcg-conditioned-e146-empty-start-working-day/
  contract.json, instances/, rows/, REPORT.md, report.json

CPU1thread; no GPU/training. At30min/episode the worst-case cap budget is27h,
NOT an expected runtime estimate. Long pilot is user-launched. Existing rows,
including failures, are reused; source/recipe changes reject existing contract.
Three instances/level are descriptive development data; test panel sealed.
