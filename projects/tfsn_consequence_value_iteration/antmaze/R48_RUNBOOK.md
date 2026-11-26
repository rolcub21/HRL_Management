# R48 rules-only AntMaze runbook

R48 is prepared for a rules-only comparison against active R39 plus R46. The
method and held changes are defined in
[`R48_RULE_FREE_R39_REFERENCE.md`](R48_RULE_FREE_R39_REFERENCE.md), and the
machine-readable contract is
[`r48_rule_free_protocol.json`](r48_rule_free_protocol.json).

The implementation never rewrites R31, R39, or R46 artifacts. The completed
two-task gate remains under:

```text
projects/tfsn_consequence_value_iteration/results/antmaze/r39_r46_rule_removal_r48
```

Its graph was prepared before the current manifest hash and is historical
development evidence. Do not reuse that preparation for the full panel.

## Full 15-task panel

The full panel uses a fresh, separate output root:

```text
projects/tfsn_consequence_value_iteration/results/antmaze/
  r39_r46_rule_removal_r48_full15
```

Prepare every reward-free artifact under the current manifest. This is the
first step of the panel and is resumable:

```bash
E600_DEVICE=cpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  projects/tfsn_consequence_value_iteration/antmaze/run.sh \
  prepare-r48-rule-removal-full15
```

Preparation now binds the manifest and method-reference hashes into the graph,
reach bank, mobility labels/checkpoint, bridge shards/artifact, and passage
geometry. Any stale or unbound artifact is rejected rather than relabeled.

After preparation, the passage audit and environment evaluation are
independent. They can be launched in separate terminals:

```bash
E600_DEVICE=cpu projects/tfsn_consequence_value_iteration/antmaze/run.sh \
  audit-r48-passage-matching-full15
```

```bash
E600_DEVICE=cpu projects/tfsn_consequence_value_iteration/antmaze/run.sh \
  hidden-r48-rule-removal-all15
```

The passage audit takes zero environment steps, uses no task reward or hidden
goal, and does not impose a new cutoff. It writes complete unweighted and
bridge-weighted match-error quantiles to:

```text
projects/tfsn_consequence_value_iteration/results/antmaze/
  r39_r46_rule_removal_r48_full15/diagnostics/
  passage_matching_audit.json
```

The panel summary will be written to:

```text
projects/tfsn_consequence_value_iteration/results/antmaze/
  r39_r46_rule_removal_r48_full15/evaluation_all15/summary.json
```

After the panel is complete, the zero-environment-step cause audit compares
the selected expected passage gain with actual new ground and endpoint
revisiting, and cross-checks the same relationship at episode level on the
saved PointMaze R11 shards:

```bash
projects/tfsn_consequence_value_iteration/antmaze/run.sh \
  audit-r48-passage-gain-cause-full15
```

It writes the report, command/episode tables, and PNG/PDF figure under:

```text
projects/tfsn_consequence_value_iteration/results/antmaze/
  r39_r46_rule_removal_r48_full15/diagnostics/passage_gain_cause_audit.*
```

The AntMaze comparison is command-level. Existing PointMaze R11 shards only
retain exact new visited cells per episode, so that cross-check is explicitly
episode-level and must not be described as command-level calibration.

The follow-up memory-scale audit uses the exact change in global residual
fine-memory mass between consecutive acquisition commands. It takes no
environment steps:

```bash
projects/tfsn_consequence_value_iteration/antmaze/run.sh \
  audit-r48-memory-scale-full15
```

It writes JSON/CSV and PNG/PDF outputs under:

```text
projects/tfsn_consequence_value_iteration/results/antmaze/
  r39_r46_rule_removal_r48_full15/diagnostics/memory_scale_cause_audit.*
```

This is an exact global memory-contraction diagnostic, but not an exact local
endpoint readout. The completed R48 logs did not retain primitive coordinates
or `TaskFields.search(actual_endpoint)` before updating memory, so the latter
cannot be reconstructed from the saved run.

The panel's claim level is `consumed_development_full_panel`, not
confirmation, because all 15 tasks have historical development exposure.

## Task-3 trace finding

R48 avoided the known flip transition; it did not demonstrate that the
transition itself is repaired. The historical failure went from
`(-7.392, 5.165)` toward `(-5.985, 1.132)`, ended inverted at
`(-6.057, 0.278)`, and then stalled. R48 instead reached
`(-7.658, 10.737)`, `(-5.704, 5.570)`, and `(-6.978, 0.651)`, then continued
moving through three more commands to contact. It never issued the known
flipping state-command pair. R48 did not log torso orientation, so the trace
rules out the same persistent flip-and-stall outcome but not every possible
transient inversion.

## Archived two-task gate

The completed gate summary is preserved at:

```text
projects/tfsn_consequence_value_iteration/results/antmaze/
  r39_r46_rule_removal_r48/evaluation_gate2/summary.json
```

Its artifacts correctly fail current-manifest verification. Do not regenerate
or relabel them; the full-panel directory is the current run target.

The current host reports no NVIDIA driver, so the examples select CPU. If a
working CUDA device is available, omit `E600_DEVICE=cpu`.

## Focused tests

```bash
cd /home/ai_diagnosis/HRL_Management
PYTHONPATH=/home/ai_diagnosis/SG/sn:/home/ai_diagnosis/HRL_Management \
  /home/ai_diagnosis/SG/.venv/bin/python -m unittest \
  projects.tfsn_consequence_value_iteration.antmaze.test_r48_rule_free_r39 -v
```

