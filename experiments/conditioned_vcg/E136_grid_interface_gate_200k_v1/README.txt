E136 EXPLICIT-DATE / SHARED CERTIFIED INTERFACE DEVELOPMENT GATE
===============================================================
    cd /home/ai_diagnosis/HRL_Management
    bash experiments/conditioned_vcg/E136_grid_interface_gate_200k_v1/run.sh preflight

No training entrypoint, no checkpoint mutation, no test-panel generation.
Runs one reserved calibration instance per cell. Both rankers use disposable
scratch weights/shells: results are NOT trained-policy performance comparisons.
CPU threads are limited to one; each live frontier check has a 30-second cap
and a 2000-node certificate budget. Each retained-witness stress check also has
a 30-second cap: explicitly inject the nonprogress trigger, then execute two
real forced macros and compare post-execution guard states/backup frontiers.
This does not claim full naturally occurring trajectory lifecycle coverage.
UNKNOWN/timeouts leave the gate incomplete;
they are not silently reclassified as UNSAFE or discarded.

GridEnv preserves nominal Poisson storage durations, but external release-fixed
dates drive dense tardiness, delivery earliness, graph/public due slack, inbound
and stored planning views, the inherited Defer rule and nominal residual K.
Reference trajectories must agree with E134 offline J/J_W accounting. Initial
inventory is excluded from in-window admission wait and does not create reset
admission progress events. Unreleased due dates are not exposed by the public
date accessor. Full reference data are only used for this development replay;
a future rollout harness must pass policies only the environment/interface.

VCG and certified Dynamic use literally the same GridAgent._admissible_prepared,
including cost records, queue/progress observation, E121 bounded-wait rule and
recovery guard. Certified Dynamic only replaces ranking. This is an outflow
contract, NOT the joint-inflow variant. Baseline E126 planning-view adapters use
the same external-date accessor in interface(); full plain-baseline harness
execution and full shared-guard trajectory lifecycle are still separate gates.
Existing sources/contracts are never edited; scoped patches restore at exit.

The inherited fast E123-v2 tensor-input/state-sharing path is retained. Online
scores are not cached; no candidate sampling or approximate backup is added.
The four-head residual definition and common successor choice are unchanged.

HORIZON IS NOT A PROOF: 6000 is a development execution cap. Initial inventory
needs one new delivery event, incoming work needs admission plus delivery:
24 + 2*36 = 96 events at the largest fill, not 120 newly required events.
The old <=30 release-gap / <=20 progress-prefix / 36-job 3600-step proposition
does not establish a uniform bound for the new grid or the legacy A guard.
We explicitly record completion_guarantee_established=false. Do not freeze
paper claims or a training launcher as if this issue were solved by a larger cap.

Obstruction audit counts stored blocks lacking an immediate approach/retrieval
path and tracks the count through the reference schedule. This is geometric
obstruction, NOT proof of necessary rehandling. Peeling accessible blocks can
still complete a dense yard with zero rehandles. Report both facts.

Before mixture training: full certified-pair lifecycle and plain baseline smoke,
decide the finite-progress guarantee/claim, then freeze balanced mixture, seed
manifests, budget/exploration/checkpoint rules, GA compute and statistical plan.
E134 primary J and evaluation-only J_W remain unchanged; no fifth head.
