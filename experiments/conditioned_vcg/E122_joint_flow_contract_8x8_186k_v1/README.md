# E122 — Joint-Flow VCG contract and frozen mechanism pilot

This is a separately versioned **controller/certificate experiment**, not a new
graph encoder and not an adopted training change. Existing E118--E121 outputs
and any ongoing training remain untouched. There is no training CLI here.

## Contract

Keep the exact current-work recovery certificate, and additionally require that
whenever usable free storage exists, at least one free cell is reachable from
the pickup and the empty transporter can approach that pickup. Check actual
stored occupancy, fixed obstacles and reservations. The pickup is an endpoint
for the admission check; its reserved waiting cell does not become a corridor.

The joint verifier searches the original strict recovery macros restricted to
access-preserving successors. Every prefix of every SAFE recovery witness is
checked. Accept and Reconfigure cannot create the two-entrance blockage seen
in E120. Recovery guard commitments use these joint witnesses; there is no
native-certificate fallback. UNSAFE and computational UNKNOWN remain rejected.

If no usable storage is free, admissions are closed. Bounded waiting or
recovery may occur; the first freeing recovery move must restore access.
There is no assertion that every full arrangement satisfies this contract.
An empty joint frontier is a failure, never hidden by a relaxed fallback.

The claim is **macro-boundary access preservation and current-work recovery
under the stated fixed-obstacle model**. It is not continuous primitive-time
ingress availability, recursive acceptance of arbitrary future jobs, an
unbounded-capacity guarantee, or a general throughput theorem.

## Waiting and liveness

The progress-only arm removes inbound, nominal-due and release-age exclusions.
Each Defer is capped at 10 primitive steps; after 16 consecutive no-event
Defers it is unavailable. The existing guard retains its persistent count
across Defers and commits after two non-progress recovery decisions. A
Reconfigure can reset the Defer streak but cannot reset that guard count.
Admissions reset the guard; their number is finite in this 36-job workload.
Observable arrival/status events can reset the Defer streak; these too must
be finite for this progress argument. No due-date guard trigger is enabled.

The preflight exercises alternating Defer/Reconfigure histories and forced
joint-witness binding. These are executable checks of the composition, not a
machine-checked universal liveness proof. A chosen finite bound can affect
quality even when its stated purpose is liveness, so report forced decisions.

## Comparison

| Arm | Feasibility | Waiting |
| --- | --- | --- |
| A_outflow_bounded | Existing outflow | E121 bounded blocked-pickup wait |
| B_joint_bounded | Joint flow | E121 bounded blocked-pickup wait |
| C_outflow_progress | Existing outflow | Progress-only |
| D_joint_progress | Joint flow | Progress-only |

All use the same frozen E118 schedule_only episode-120 model, seed 0, epsilon
zero, `J=2T+E+0.1L+10R`, release-fixed dates, ±20 tolerance, gamma 1. No replay
insertion, gradient steps, target updates, changed cost weights, or checkpoint
selection. Native/joint certificate caches are isolated per episode. The
relocation-family shortcut is disabled in all arms because its original proof
does not establish the augmented property. Graph tensor and timing-invariant
physical-certificate caches remain exact under their respective contracts.

Fresh arrival-driven 8x8 instances: 36 jobs, release gap 30, frozen Poisson(80)
durations. Pilot seeds 186990--186991 (8 total evaluations); larger mechanism
panel 186000--186009 (40 evaluations). Source training/validation/sealed seed
overlap is checked without opening any sealed trajectories. Both panels are
development mechanism evidence, not paper confirmation. Do not tune on the
larger panel and then call it a sealed test.

## Launch

Requires the existing E119 repair1 small frozen model, E120/E121 artifacts and
the repository Python environment. CPU uses one thread; CUDA is optional.
The per-episode wall bound is 300 seconds and primitive horizon 3600. Failed
or bounded rows are retained; quality means are suppressed for an arm if any
of its observed rows failed. Re-running commands reuses existing rows.

```bash
E122=experiments/conditioned_vcg/E122_joint_flow_contract_8x8_186k_v1/run.sh
bash "$E122" preflight --device cpu
bash "$E122" run-panel --panel pilot --device cpu
bash "$E122" status --panel pilot
```

Preflight first checks deterministic empty/trapped/open/full-yard fixtures,
native/joint scope, witness prefixes, and guard composition. It then replays
one E121 reference instance and requires exact action/time/cost parity. Fresh
panels remain closed if this gate fails. It is also run automatically before
the first panel when no gate artifact exists.

After inspecting the pilot, the larger panel is explicit:

```bash
bash "$E122" run-panel --panel mechanism --device cpu
bash "$E122" analyze --panel mechanism
```

Report every arm, completion, J/components, release/storage MAE, outside wait,
work, rehandles, occupancy, blocked admission, guard use and decision latency.
An adverse frozen result cannot establish that a controller retrained on the
new frontier would fail. Before any retraining, selection and TD backups must
share this complete controller frontier/memory, including the guard; use a
separate training protocol and a predeclared matched budget/exploration plan.
