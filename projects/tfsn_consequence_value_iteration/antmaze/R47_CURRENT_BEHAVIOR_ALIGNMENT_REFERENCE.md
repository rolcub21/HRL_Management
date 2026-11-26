# R47: current-behavior consequence alignment

R47 is one reward-free behavior-alignment collection for the frozen R39
acquisition planner with R46's within-task online correction.  It contains no
hidden goal, reward observation, oracle route, or task-time gradient update.
Its purpose is to expose the consequence and continuation models to the
first-encounter commands selected by the deployed acquisition behavior.

Each preparation task contains five physical episodes of ten fixed-duration
commands.  Search memory and online correction memory persist across the five
resets, exactly as they do across carried evaluation episodes, and are reset
between preparation tasks.  The environment is reset to the same seed within
a preparation task.  Forty preparation tasks give 2,000 command transitions;
this is a component-alignment budget, not hidden-goal evaluation evidence.

The shard stores every primitive state in each 100-step command.  Spatial and
posture consequences therefore use the actually observed endpoint.  It also
stores an observational continuation target for every nonterminal command:

```text
1 - exp(-||x_next_end - x_next_start||^2 / (2 sigma^2)).
```

This is the movement produced by the next command from the current endpoint.
It is continuous, reward-free, and uses the deployed data-derived landing
precision.  It does not use an uprightness threshold or a hand-authored flip
label.  Commands at physical-episode boundaries have no continuation target
because the next state is a reset.

R47 changes no planner score.  Any later model trained from these data must be
tested first on the consumed two-goal development gate.  The confirmation
panel remains sealed.
