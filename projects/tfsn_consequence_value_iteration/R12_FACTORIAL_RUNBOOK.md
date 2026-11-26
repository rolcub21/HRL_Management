# AntMaze R49 latent-detection × learned-passage development screen

This is a consumed-development AntMaze revision. It does not change the frozen
AntMaze R48 or PointMaze R11 sources, protocols, artifacts, or results.
PointMaze is deliberately outside this screen: the AntMaze causal question is
settled first.

The four arms are:

| Observation memory | Passage prediction |
| --- | --- |
| sigma-kernel control | nearest recorded control |
| latent radius | nearest recorded control |
| sigma-kernel control | shared learned passage |
| latent radius | shared learned passage |

The task set is fixed to AntMaze tasks 1 and 4, with three episodes per task.
These are consumed development cases; the result cannot be called
confirmation.

## Preparation (no environment evaluation)

Run the code audit:

```bash
E600_DEVICE=cpu projects/tfsn_consequence_value_iteration/run_r12_factorial.sh audit-code
```

Train the AntMaze passage artifact from existing reward-free transitions:

```bash
E600_DEVICE=cuda projects/tfsn_consequence_value_iteration/run_r12_factorial.sh train-passage
```

If CUDA is unavailable, `E600_DEVICE=cpu` is valid but slower. The AntMaze
adapter calls the domain-agnostic shared class, loss, target construction,
grouped split, optimizer, and schedule. Training reports every epoch and uses
the manifest's validation early-stopping rule; 200 epochs is a safety cap, not
a required duration. The runner holds a process lock and refuses a second
trainer targeting the same artifact. It also refuses to start expensive
training unless the target filesystem has at least 5 GiB free and checks
atomic-checkpoint capacity again immediately before saving.

Before any environment step, run the mandatory preflight:

```bash
E600_DEVICE=cpu projects/tfsn_consequence_value_iteration/run_r12_factorial.sh preflight
```

It refuses to continue unless:

- candidate goals are the unique recorded roadmap xy states with a uniform
  prior;
- `rho_min` and `rho_max` are derived from the deployed landing sigma and exact
  support diameter;
- the AntMaze passage checkpoint uses the shared model/loss/training contract;
- checkpoint and manifest hashes match;
- no domain-local passage model class exists; and
- the protocol declares fatal posterior exhaustion with no fallback.

The rendered report is saved to
`results/latent_detection_passage_r12/preflight.json`.

## Development evaluation (environment steps)

Run all four arms on AntMaze tasks 1 and 4:

```bash
E600_DEVICE=cpu projects/tfsn_consequence_value_iteration/run_r12_factorial.sh evaluate-2x2
```

The runner invokes preflight first. A posterior exhaustion writes a
`MODEL_FALSIFIED.json` artifact and terminates; it never resets the posterior to
uniform. Evaluation prints every task, episode, and command as it completes.

The evaluation command creates the AntMaze main-effect/interaction report and
plot. To regenerate only that report from completed arms, run:

```bash
projects/tfsn_consequence_value_iteration/run_r12_factorial.sh summarize
```

The outputs are
`results/latent_detection_passage_r12/summary/factorial_summary.json`,
`factorial_task_rows.csv`, and `factorial_acquisition.png`.
