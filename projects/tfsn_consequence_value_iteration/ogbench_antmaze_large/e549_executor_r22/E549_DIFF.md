# R22 audit against E549

R22 is a from-scratch E549-style executor for OGBench AntMaze Large. The
machine-readable comparison is produced by `verify.py` before training.

Unchanged from E549:

- Stable-Baselines3 SAC and every model keyword;
- seed 42, 16 environments, 4x256 ReLU policy;
- batch size 256, one-million-row replay, `auto_0.1` entropy;
- gamma 0.991, tau 0.03, learning rate 3e-4;
- one gradient step per environment step after 10,000 warmup steps;
- normal action noise with mean 0 and sigma 0.3;
- dense directional reward, control cost, and survival reward;
- contact-or-100-step command handoff without a physical reset;
- 1,000-step physical episodes and training from scratch.

Environment-adapter changes:

1. Gymnasium Robotics Medium Diverse is replaced by the pinned OGBench 1.2.1
   AntMaze Large environment. This necessarily changes its maze, Ant dynamics,
   reset support, and control timestep.
2. The command-distance replay is derived from every exact 100-step transition
   in OGBench's reward-free training dataset. It has no metric cutoff.
3. OGBench's complete native 29D observation is retained exactly and a 2D
   relative command target is appended, producing the required 31D
   goal-conditioned input. E549 used 27D Gym proprioception plus the same 2D
   relative target.
4. Physical reset cells come from reward-free training-trajectory starts.
   Native evaluation task rows remain unopened.

The R21 evaluator may command any exact held-out achieved state in a free maze
cell. This affects evaluation only: training goals retain E549's narrower
benchmark goal-noise support.

Operational changes are limited to local logging, reproducible checkpoint and
replay pairs, and audits at the frozen 1M, 2M, and 5M gates.

Render the line diff and the semantic audit with:

```bash
bash projects/tfsn_consequence_value_iteration/ogbench_antmaze_large/e549_executor_r22/run.sh source-diff
bash projects/tfsn_consequence_value_iteration/ogbench_antmaze_large/e549_executor_r22/run.sh verify
```
