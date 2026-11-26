# R54: E549-10M-matched AntMaze preparation

R54 is the isolated ten-million-step counterpart to R52.  It uses the same
reward-free command proposal behavior, seeds, 100-step command contract,
30,000-command graph block, 6,000-command alignment block, whole-task holdout,
model architecture, and preparation schedule.  The only intended difference
is the hash-pinned E549 10M executor.

No R52 transition or artifact is reused.  The consequence model, landing
scale, recorded graph, mobility model, bridges, and empirical passages are
rebuilt exclusively from transitions executed by the 10M checkpoint.  This is
required because the 5M and 10M policies have measurably different success,
endpoint-error, and inversion distributions.

The upstream SAC reward is part of executor training only.  TF-SN preparation
does not consume task reward, hidden goals, evaluator success radii, routes, or
oracle commands.  R54 does not supersede R52; the two preparations form a
controlled executor-training-duration comparison.
