E180: wait-priced, unmasked static-batch frozen pilot
=====================================================

This is a new development protocol. E164/E168/E172/E173/E174/E176/E177/E179
outputs are not rewritten or reinterpreted. E179 action-parity says only that
its optimization preserved the old Accept-first policy; its timing conclusions
do not apply to E180. The test panel remains sealed. No E180 training is run.

Active-rule inventory (the E177/E179 path before this change)
---------------------------------------------------------
Rule / source                                      Classification / disposition
Physical macro executability and FIFO inbound,
  E83 legal.physical_frontier and E164 env.step    Problem definition; retain.
All jobs available at zero, storage-clock due,
  20-step handoff window, E164/E168 objective      Problem definition; retain.
Event Defer to next observable window,
  E152 policy/option                               Problem definition; retain.
2T + E + 0.1L + 10R, E164/E168 objective           Problem definition; add 0.5W.
Joint outflow completion and inflow access,
  E122 flow and E172 all-free-cell reachability     Certification; retain as the
                                                   declared capacity invariant.
SAFE-only proposal selection, E158/E162            Certification; retain.
Residual critic's J ranking, E147/E159/E162        Ranking; retain frozen.
E156 scale-free softmax                             Training exploration only;
                                                   not active in frozen eval.
E176 event reset, Reconfigure bound and one
  witness step only if no SAFE objective action     Progress guard; retain.
E164/E168 safe-Accept-only when available           Preference disguised as
                                                   constraint; REMOVE.
E168 no inbound Defer                               Preference disguised as
                                                   constraint; REMOVE.
E171 intake-restoration lookahead                   Extra action filter; NOT
                                                   ACTIVE in E177; keep out.
Earlier E153/E162 guard-frontier patches            Superseded operationally by
                                                   E176 build/choose; E180
                                                   invokes these directly.
E158 lazy proof/cache and E161 batched targets      Exact computation; retain.
E179 lazy Accept certification                      Old-policy speed experiment;
                                                   not used by E180.

E180's effective path is physical actions -> joint certification (including
E172 access) -> frozen critic ranking under J with W -> E176 liveness guard.
The physical environment alone enforces FIFO. Accept, Deliver and event-Defer
can compete if physically proposed and joint-SAFE. Neither Accept-first nor
no-inbound-Defer filtering is called. The guard masks only Reconfigure after
its event-bounded allowance; it may take one witness step only if no other
certified action survives. Lazy certification and batching are unchanged.

W is counted per primitive step for every released, unadmitted block. Since
this static batch releases all jobs at step zero, completed W_i equals its
first storage time. The same fifth cost head and predeclared 0.5 weight are
used by the environment, nominal macro cost, replay, and reported J. The
frozen checkpoint has its wait head initialized to zero, so this pilot tests
the new interface; it does not claim a trained wait-aware policy.

From /home/ai_diagnosis/HRL_Management:

  .venv/bin/python -m pytest -q tests/test_vcg_e180_wait_cost_unmasked.py
  bash experiments/conditioned_vcg/E180_wait_cost_unmasked_pilot_239k_v1/run.sh preflight --device cpu

The recorded E175 boundary falsifies the proposed universal Deliver test:
all three physically legal Delivers fail the *native* E122 access test because
the agent cannot approach pickup after ending at the chosen exit, while three
jobs are still inbound. The stricter E172 all-free-cell rule is not the
reported rejection. Do not simply whitelist Deliver: physical legality and
joint-admission feasibility are different. The problem statement must settle
whether a transporter can return from the exit outside the grid. Under the
current grid-motion model it cannot, so those Delivers remain excluded.

Before the pilot, measure the heavy-day horizon using certified Dynamic on
the same instance and policy. The 2,600-step execution cap has a declared
20% headroom gate: the reference must finish by step 2,080. This is a gate,
not a change to its ranking. The reference has a separate 3,600-second wall
watchdog because the frozen heavy day already took 828 seconds and the
certified-Dynamic selector may certify more proposals. Run it separately:

  MPLCONFIGDIR=/tmp/e180-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  bash experiments/conditioned_vcg/E180_wait_cost_unmasked_pilot_239k_v1/run.sh \
    reference --device cuda

Only after the reference passes, the full three-day frozen pilot is a longer
run for the user to launch:

  set -o pipefail
  MPLCONFIGDIR=/tmp/e180-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  bash experiments/conditioned_vcg/E180_wait_cost_unmasked_pilot_239k_v1/run.sh \
    pilot --device cuda 2>&1 | \
  tee -a results/vcg-conditioned-e180-wait-cost-unmasked-pilot-239k-v1-repair2/run.log

Each case is resumable. Use --rho 0.2, 0.5, or 0.8 to run one development day.
Completion and achieved peak storage come first; compare J components only
after verifying that the three days complete. Do not train or open the test
panel on the basis of a single successful pilot.
Rows report the five J components, UNKNOWN selector queries, distinct UNKNOWN
successor certificates in the cache, and horizon headroom. UNKNOWN counts are
not automatically proven-unsafe counts. The access invariant is conservative:
it forbids disconnected free cells even if this day's remaining jobs would
never need them.
