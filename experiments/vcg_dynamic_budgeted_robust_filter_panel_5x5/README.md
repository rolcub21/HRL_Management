# Committed-cohort dynamic robustness panel (v2)

This additive evaluation-only experiment corrects the recovery-horizon reset
in the earlier dynamic runner. It reuses the authenticated frozen 89k
EpisodeInstances, model-seed-0 VCG checkpoint, detached handling-cost head,
selector, and real 5x5 environment executor. It performs no training,
checkpoint selection, or lambda selection.

The comparison grid remains:

- nominal, one-step, and recursive recovery filters; and
- handling lambda 0 and 0.2.

The pilot has 4 instances and 24 rows. The full panel has 30 instances and 180
rows. This directory and protocol use new v2 output locations; they do not
alter or reuse v1 result ledgers.

## Committed recovery segment

Outside a committed segment, the runner computes a provisional budget

```text
B = min(exact witness primitive steps + 3 * witness macro count,
        remaining live episode horizon)
```

and filters recovery candidates at `B`. The unchanged frozen selector then
ranks the entire filtered frontier, including nominally SAFE Accept and Defer.
If it selects Accept or Defer, no robust segment has begun.

If the selector first chooses Deliver or Reconfigure, the runner commits the
current stored block-label cohort and `B`. Until that cohort is empty:

- Accept and Defer are masked;
- the same remaining primitive budget is carried across every live replanning
  boundary;
- recovery actions are recertified at that remaining budget;
- the selected recovery duration is subtracted exactly; and
- no delivery, relocation, arrival, or new decision replenishes the budget.

An added stored label during an active segment fails closed. When the committed
cohort becomes empty, the leftover budget is discarded and nominal admission
reopens at the next boundary. An arrived but unaccepted block can wait outside
the closed RecoveryState during the segment.

## State and execution authentication

Certification uses a timing-erased RecoveryState: every stored block's
`remaining_time` is set to zero. Timing error and return remain evaluation
metrics, but block clocks are not part of the recursive physical state.

Every live RecoveryAction must equal the canonical solver action in kind,
label, source, destination, approach path, transport path, and primitive step
count. Recovery does not invoke the historical options' A* replanning path.
The v2 exact executor sends the certified approach moves, PICKUP, certified
transport moves, and PUTDOWN literally through `env.step`, recording the full
action and cell trace. A selected recovery must then satisfy:

```text
observed base duration  = RecoveryAction.steps
observed total duration = RecoveryAction.steps + requested injected steps
```

The declared disturbed successor is compared with the live timing-erased
RecoveryState immediately after execution and again before filtering at the
next boundary. Grid geometry, storage and exit sets, block label-to-position
mapping, reservations, fixed/pickup/wait obstacles, and realized agent endpoint
must all match exactly. Only `remaining_time` is ignored.

The live experiment realizes one deterministic action-conditional member of
the certified uncertainty set: a nonterminal recovery receives one WAIT and
the lexicographically first declared clear adjacent stop when one exists; a
terminal completion uses delay zero. This is neither adversarial worst-case
selection nor exhaustive execution of every member of the uncertainty set.

UNKNOWN from the finite solver is rejected. The configured expansion limit is
reported; an UNKNOWN cutoff is not treated as a losing proof or a success.

## Run

The wrapper defaults to CPU:

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/vcg_dynamic_budgeted_robust_filter_panel_5x5/run.sh prepare-pilot
bash experiments/vcg_dynamic_budgeted_robust_filter_panel_5x5/run.sh run-pilot
```

After inspecting the pilot behavior and runtime, the full panel is:

```bash
bash experiments/vcg_dynamic_budgeted_robust_filter_panel_5x5/run.sh prepare-full
bash experiments/vcg_dynamic_budgeted_robust_filter_panel_5x5/run.sh run-full
```

CUDA is explicit:

```bash
DEVICE=cuda bash experiments/vcg_dynamic_budgeted_robust_filter_panel_5x5/run.sh run-pilot
```

Do not mix devices in one output directory. Rows are authenticated and written
atomically, so a run can resume after interruption. Performance aggregates are
suppressed unless every row in a method/lambda cell strictly completes.
The contract binds the exact executor, macro executor, selector hierarchy,
recovery/candidate/filter/dynamic-yard sources, timing-metric source, and
Python/NumPy/PyTorch/CUDA runtime identity; a changed dependency cannot resume
inside the same output contract.
Solver memo hits and all wall-clock fields are diagnostic and noncomparative:
they depend on arm order, process restart, resume state, hardware, and runtime.
The memo is shared only across the six arms of one instance and cleared before
the next instance.

## Claim boundary

The recursive arm supports a conditional theorem-facing interpretation for a
committed stored cohort only when the selected action was recursively WINNING,
not UNKNOWN; the realized disturbance belongs to the modeled set; the exact
path, duration, outcome, and next timing-erased successor all match; the
carried budget chain has no reset; and no unresolved cutoff was used for
admission. This is not a theorem of full-episode completion, arrival
robustness, timing robustness, robust Accept/Defer behavior, adversarial live
testing, or hardware execution. Nominal and one-step filters may fail under
later disturbances; they are comparison arms. The four-instance pilot is
diagnostic, not confirmatory.
