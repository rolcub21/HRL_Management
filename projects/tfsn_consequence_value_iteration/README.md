# TF-SN consequence value iteration

> **Current method:** the cross-domain continuous reference is
> [`METHOD_REFERENCE.md`](METHOD_REFERENCE.md), with all fixed settings in
> [`shared/method_manifest.json`](shared/method_manifest.json). The PointMaze
> R11 freeze is [`pointmaze/R11_FREEZE.json`](pointmaze/R11_FREEZE.json).
> The remainder of this file is the preserved experimental history and is not
> the definition of the current method.

This isolated consumed-development arm keeps the frozen E377 Farama SAC
executor, consequence ensemble, reward-free preparation data, exact task
field, command duration, nine goals, and reset seeds. It replaces only the
behavior-successor future readout with finite-horizon dynamic programming over
a learned command-level transition graph.

The graph is reward-independent. Its 4,096 states are sampled uniformly over
space from reward-free command-boundary states (recorded starts and endpoints),
the state distribution on which the consequence model operates. Every state
optimizes over the same 128-coordinate, space-uniform sample of endpoints the
executor actually reached. The frozen consequence ensemble supplies five
mixture components and determines which commands are feasible from each state.
Eight samples per ensemble member are drawn from the complete mixture,
including within-component variance; the lower ensemble value supplies
pessimism.
Backups are undiscounted over the benchmark's ten-command physical horizon and
use the minimum expected value across ensemble members.

No hidden goal, maze graph, wall map, waypoint, task reward, or task gradient
enters graph construction. Supplied-goal evaluation uses the exact Gaussian
goal field at the frozen 0.2504 m landing scale.

Run both preparation and the matched nine-goal gate:

```bash
cd /home/ai_diagnosis/HRL_Management
bash projects/tfsn_consequence_value_iteration/antmaze/run.sh all
```

The frozen references are direct executor 2/9 and oracle waypoint route 7/9.
This is consumed-development diagnostic evidence only; it does not access the
sealed confirmation panel.

The preserved R1 graph used primitive mid-command states as nodes. Its
transition projection had a standardized p90 distance of 7.33 and its values
collapsed to zero; that invalid discretization scored 1/9. R2 corrects only
the node support and writes a new artifact rather than overwriting R1. R2
scored 2/9, but its localized proposal yielded only three unique commands per
node at the median and its nonzero value support stopped growing after one
backup. R3 removes that remaining behavior-policy action restriction while
preserving both earlier artifacts. R3 still used only component means for its
transition edges even though the model encoded a weighted 3.26 m of XY spread
inside components. That collapsed 56% of transition mass to self-loops and
scored 1/9. R4 uses the full sampled consequence distribution in both graph
construction and live command scoring. R4 also scored 1/9. Its full samples
exposed the remaining invalid approximation: predicted endpoints were an
average 7.15 standardized units from the graph node onto which they were
snapped (p90 13.82, maximum 671.13).

## R5/R6: recorded-transition graph

R5 preserves R1--R4 and changes only the graph representation. Every node is
an exact command-boundary state in the reward-free data. Every edge is an
actually recorded start, requested command, and realized endpoint. There is no
learned transition or endpoint projection inside the graph. Edge reliability
is the continuous kernel estimate of whether similar state-command records
produced a similar realized full-state change; it is not whether a fixed-time
command finished at its potentially distant requested target. There is no
success threshold. Preparation exposed that R5's first reliability definition
mistook "did not finish at the requested long-range target in 100 steps" for
"the recorded transition is unreliable": its median reliability was
1.52e-6. It was stopped before task evaluation and its artifact is preserved.
R6 uses realized full-state transition repeatability instead. A 512-edge
read-only check gave median reliability 0.500 (p10 0.401), so useful progress
edges retain value while inconsistent outcomes remain downweighted.

At deployment the frozen consequence ensemble is used only for the first step
from the live state into the empirical graph. Commands are sampled from those
actually issued at continuously similar recorded states. A predicted outcome
inherits graph continuation smoothly in proportion to its full-state support;
an unsupported outcome therefore cannot be snapped into a fictitious route.

Prepare the small empirical graph, then run the same consumed nine supplied
goals:

```bash
cd /home/ai_diagnosis/HRL_Management
bash projects/tfsn_consequence_value_iteration/antmaze/run.sh prepare-recorded
bash projects/tfsn_consequence_value_iteration/antmaze/run.sh evaluate-recorded
```

Preparation uses only the frozen 30,000 reward-free command transitions from
rounds 0--2. Evaluation remains consumed-development diagnostic evidence; it
does not access the sealed confirmation panel.

## Oracle command-contract localization

The old 7/9 oracle reference changed waypoints at primitive-step resolution,
whereas TF-SN commits to each command for 100 primitive steps. The R7
diagnostic keeps the same maze-cell routes but permits waypoint advancement
only at command boundaries. It saves every resulting trajectory so the frozen
round-2 consequence model is tested on exactly the same state-command pairs,
without a second environment run.

Run the two gates separately:

```bash
cd /home/ai_diagnosis/HRL_Management
bash projects/tfsn_consequence_value_iteration/antmaze/run.sh oracle-fixed-contract
bash projects/tfsn_consequence_value_iteration/antmaze/run.sh diagnose-oracle-commands
```

The model report compares command likelihood against the checkpoint's
reconstructed training-time validation split, measures distance to the nearest
predicted endpoint mode, and reports stall-probability calibration using the
existing 0.2 m stall definition. Both stages use the consumed nine-goal
development panel; the fresh panel remains sealed.

## R8: factorized consequence model

R7 localized the failure more narrowly. The frozen joint model accurately
separated stalls from movement, but on the 41 known-good fixed-contract route
commands its nearest spatial mode was 3.76 m from the realized endpoint and
only 1.3% of its predicted mass fell within 0.8 m of that endpoint. Full-state
likelihood hid this error because the two spatial coordinates shared a loss
with 27 posture coordinates.

R8 trains two independent distributions on the unchanged reward-free rounds
0--2. The spatial head predicts `endpoint_xy - command`; the posture head
predicts absolute endpoint posture conditioned on state, command, and the
spatial endpoint. The heads have separate parameters, likelihoods, and
validation selection, so posture error cannot alter the spatial fit. The
fixed-contract oracle commands are excluded from training and selection and
remain the untouched test.

Train first, then run the offline diagnostic:

```bash
cd /home/ai_diagnosis/HRL_Management
bash projects/tfsn_consequence_value_iteration/antmaze/run.sh train-factorized-consequence
bash projects/tfsn_consequence_value_iteration/antmaze/run.sh diagnose-factorized-consequence
```

The diagnostic reports spatial likelihood separately, probability mass within
0.8 m and within the frozen 0.2504 m landing scale of the actual endpoint, and
stall calibration. It compares the successful-command endpoint mass and stall
Brier score directly with the frozen joint-model diagnostic; it does not
invent a post-hoc pass threshold.

The R8 diagnostic raised endpoint mass within 0.8 m on the 41 known-good
commands from 1.3% to 91.3%, reduced median nearest-mode error from 3.76 m to
0.049 m, and retained separation between moving and stalled commands. R9
therefore reuses the exact R6 graph and changes only its one-step live
consequence model:

```bash
cd /home/ai_diagnosis/HRL_Management
bash projects/tfsn_consequence_value_iteration/antmaze/run.sh evaluate-factorized-recorded
```

The graph is not rebuilt: its 30,000 exact recorded edges, reliability,
proposal, and finite-horizon backup remain frozen. The evaluator verifies that
the graph's full-state normalization exactly matches the factorized model
before accessing the same consumed nine-goal supplied-goal panel.

## R10: mobility-weighted stitching

R9 reached 3/9 goals, but the empirical graph remained a collection of short
trajectory strands: only 2--16% of its exact nodes received supplied-goal
value. R10 preserves every exact R6 node and recorded edge and adds only
model-predicted bridges between strands.

Bridge destinations are a deterministic sigma-net of exact recorded nodes,
using the already frozen 0.2504 m landing scale. Every exact node retains its
own identity; landmarks are destinations, not merged representatives. Each
source considers the same global 128 nearest landmarks. A bridge is weighted
by the pessimistic ensemble expectation of the landing kernel times endpoint
mobility. Mobility is the best model-predicted probability of moving under
those same landmark commands, then amortized as a scalar function of the full
endpoint state. Thus a predicted fallen endpoint cannot inherit a standing
node's future, while harmless heading or joint-angle differences are not
forced to match one recorded posture.

The two expensive preparation stages are resumable and bind every shard to
the graph, factorized consequence checkpoint, landmarks, sample count, and
seed. Run them in order, followed by the unchanged consumed nine-goal gate:

```bash
cd /home/ai_diagnosis/HRL_Management
bash projects/tfsn_consequence_value_iteration/antmaze/run.sh prepare-stitched-mobility
bash projects/tfsn_consequence_value_iteration/antmaze/run.sh prepare-stitched-bridges
bash projects/tfsn_consequence_value_iteration/antmaze/run.sh evaluate-mobility-stitched
```

The report includes positive and meaningful value coverage over all exact
nodes, retains the fixed-contract oracle's 8/9 reference, and remains
consumed-development diagnostic evidence. The fresh panel stays sealed.

## R11/R12: locally shared command coordinates

R10 reached 5/9 and expanded meaningful ten-command value coverage from
2--16% to essentially every exact node. Its residual planner failure was more
specific: 14 of 16 apparent stalls were commands whose target was already
within 0.8 m, and roughly 26% of long-horizon graph choices were bridges under
0.25 m. The planner was spending a command entering a same-position landmark
to inherit that landmark's useful recorded action.

The R11 preparation attempted to keep every R10 bridge and every exact
recorded edge while sharing commands locally. Its pre-evaluation audit found
that it still attached each shared command to the old recorded endpoint and
multiplied its model score by the old edge reliability. Shared commands won
only 2--3% of offline choices and the same-position bridge rate was nearly
unchanged. R11 therefore has no task result and its preparation artifact is
preserved as a stopped construction diagnostic.

R12 keeps every R10 bridge and every exact recorded edge. At each exact or
live state it additionally considers every command recorded at a source
coordinate within the existing sigma scale. The factorized consequence model
evaluates each command from the state that will actually issue it, so no
recorded posture is copied. The continuation target is the sigma-net landmark
nearest the command coordinate, not the endpoint of the old execution, and no
old edge-reliability factor is applied. If the current Ant can execute a nearby
node's useful command it can do so immediately; if not, the state-conditioned
model assigns that option low support. Sigma is reused as the method's
established spatial identity scale, not introduced as a new AntMaze distance
setting.

Prepare the resumable shared-action weights, then evaluate the same nine
consumed supplied goals:

```bash
cd /home/ai_diagnosis/HRL_Management
bash projects/tfsn_consequence_value_iteration/antmaze/run.sh prepare-shared-actions
bash projects/tfsn_consequence_value_iteration/antmaze/run.sh evaluate-shared-actions
```

R10 already used the minimum connection weight across ensemble members. The
Task-6 bridge logged a pessimistic weight of 0.779, so no member supplied the
low warning that would have rejected its 2.75 m miss. R11 logs all member
connection weights explicitly for every selected command so any further
model outlier is directly attributable.

R12 remained at 5/9 while exchanging two successes for two failures and is
not retained. Before any further planner change, the frozen R10 result is
checked on the only six unused distinct goal cells. The canonical 26-cell
permutation reserves indices 11--19 for confirmation; the extension uses only
indices 20--25 and never reads or creates the confirmation panel. Frozen R10,
the fixed-100-step oracle, and the direct executor are run separately and the
report preserves original-nine, new-six, and combined-fifteen estimates.

The tasks within each arm are resumable and run concurrently:

```bash
cd /home/ai_diagnosis/HRL_Management
bash projects/tfsn_consequence_value_iteration/antmaze/run.sh expanded-all
```
