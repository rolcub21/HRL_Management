# R52: E549-5M-matched AntMaze preparation

R52 rebuilds the reward-free AntMaze preparation for the frozen E549 5M
executor.  It does not reuse Farama transitions in the fitted consequence
model, recorded graph, mobility model, bridges, or passage geometry.

## Collection contract

The collector holds every command for exactly 100 primitive steps and resets
the physical environment after ten commands.  Task memory persists across
those resets and is cleared between preparation tasks.  The frozen R51 planner
and its frozen R48-full15 artifacts propose commands; they are a behavior
policy for data collection only.  Every executed action comes from the
hash-pinned E549 5M checkpoint.

The full dataset contains 36,000 reward-free commands, matching the prior
preparation budget: 30,000 transitions form the base recorded graph and 6,000
transitions form the behavior-alignment block.  Sixteen of the eighty complete
alignment tasks are held out.  The consequence model therefore trains on
30,000 base commands plus 4,800 alignment commands and validates on 1,200
alignment commands.

## Fitted artifacts

The spatial and posture mixture heads retain the R30 factorization and training
schedule.  Landing scale is recomputed from the E549 holdout.  The recorded
graph contains only exact E549 command-boundary starts and endpoints.  Mobility,
reach bridges, and empirical passages are rebuilt from that graph and the new
consequence model.

No hidden goal, task reward, evaluator success radius, maze route, or oracle
command enters collection, fitting, model selection, or graph construction.
E549's upstream executor-training reward is part of the supplied executor's
training budget; TF-SN preparation itself remains reward-free.

## Causal comparison

R53 retains R51's destination-arrival online correction and evaluates the same
consumed development tasks 2 and 13.  Compared with the existing Farama/R51
pair audit, both executor and executor-matched preparation change together.
That comparison attributes the deployed system change to adopting E549, not to
an online-correction or task-panel change.
