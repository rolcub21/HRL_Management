# Kim spatial comparison: retrieval executor v3 protocol

This protocol separates executor repair from the final comparison.

1. `regression` reruns schedule seeds `44000..44049`. These seeds were already
   inspected under v2, so this artifact is diagnostic only. The command then
   verifies the eight exact historical route-failure tuples.
2. `verify` repeats the artifact-level eight-case check without rerunning.
3. `final` first requires the regression artifact to pass, then evaluates the
   preregistered, previously uninspected schedule seeds `45000..45049`.

The final seeds must not be used to modify the executor, scheduler, checkpoint,
or comparison protocol. If final execution exposes a protocol failure, retain
and report it; do not repair and reuse the same seeds as a clean final set.

Run from the project root:

```bash
./experiments/kim2020_v3_retrieval_regression/run.sh regression
```

After inspecting only the regression artifact and making no further changes:

```bash
./experiments/kim2020_v3_retrieval_regression/run.sh final
```

Outputs are written under `results/`. The runner refuses to overwrite either
artifact.

