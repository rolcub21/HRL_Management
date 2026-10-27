E145 ORIGINAL-LAYOUT, OCCUPANCY-ONLY FROZEN PILOT
================================================
NO training. Original E131/E133 residual episode60 models, all3 seeds.
Development pilot3 paired instances/level, not the untouched statistical test.

  MPLCONFIGDIR=/tmp/e145-mpl .venv/bin/python -B -m experiments.conditioned_vcg.E145_occupancy_only_pilot.run run

Resume: same command reuses recorded rows, including failures/timeouts.
Status/setup: same module with status / prepare instead of run.
Output: results/vcg-conditioned-e145-occupancy-only-pilot/
  contract.json, setup.json, instances/, rows/, REPORT.md, report.json

Fixed before running any contender:
- original8x8 geometry, native36 incoming jobs, gap30, native Poisson80
  duration draws, incoming due=a+p. Incoming content identical across fills.
- initial fill0,14/35=40%,24/35=68.6%. No maintained-occupancy claim.
- independent integer Uniform[0,1050] initial residual deadlines at time0.
  Nominal initial durations independently Poisson80. This is an explicit
  standing-inventory convention, NOT a reconstructed stationary prehistory,
  and no reference/witness completion times determine due dates.
- uniform random placement over ALL usable storage cells, no reserved corridor,
  no due sorting, no rejection based on contestant outcomes. Initial layouts
  and deadlines are nested across fills using independent RNG streams.
- seeds24050000..24050002, disjoint from the known training/development panels.
- near-full31/35=88.6% also pre-generated. After primary results, run it ONLY
  if BOTH plain Dynamic and GA completed all3 highest-primary-fill cases:

  MPLCONFIGDIR=/tmp/e145-mpl .venv/bin/python -B -m experiments.conditioned_vcg.E145_occupancy_only_pilot.run run --panel near-full

Methods: original residualVCG3 seeds, informed native Dynamic, informed native
GA (unchanged fixed compute configuration), certified Dynamic.54 primary rows;
18 additional near-full rows if trigger passes. Same cases/J/information/caps
for every method. CPU1thread,900s/row,6000steps,1000decisions,2000certificate nodes.
Worst-case cap54*15min is NOT an expected runtime estimate. Long runs user-launched.

Certified pair alignment:
E137 executes the same certification, interface/liveness rule, flow-event Defer
counter and witness guard for both. Certified Dynamic is instantiated with
the ORIGINAL frozen model_class(0) base, not a different GridAgent shell.
It delegates _admissible_prepared and observe_outcome to that base; only ranking
is heuristic. Frozen VCG uses the same original model class/normalization.
Plain native compound retrieval vs certified elementary macros remains a
limitation: this does not isolate certification as the ONLY method difference.

Setup reports actual fill, initial deadline range/std and obstructed initial
jobs (no clear unloaded approach OR no loaded exit route without relocation).
Burials should arise from independent placement; observed obstruction is not
used to tune/reject instances. Rising rehandles is a RESULT, not a validity
condition. Never add/remove instances to manufacture a VCG advantage.

Random packing does NOT establish initial recoverability. Keep all outcomes,
but distinguish method failures on a known-feasible start from starts whose
completion feasibility is unknown. Do not claim certification can rescue all
random starts or interpret every plain-method failure as certification value.

Completion first, failures retained; rank J only on completed rows. Report J
components, incoming admission wait (initial jobs excluded), rehandles, timing
MAE/window,steps/delivery and latency/primitive traces. All available in row
JSONs; compact table in REPORT.md. No significance claims from3instances/cell.
More initial jobs adds work: compare methods WITHIN level, not raw J across fill.

This does not stop, replace or retrain E144. Running alongside it may slow both.
