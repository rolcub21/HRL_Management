# PointMaze R12 latent-radius five-command behavior check

This is an isolated exploratory smoke check, not benchmark or factorial
evidence. It runs only latent-radius observation memory with nearest-recorded
passages on the two historically exposed PointMaze corner goals.

The physical and planning horizon is deliberately truncated from the
benchmark's 20 commands to five commands (500 primitive steps) solely to
inspect early behavior. Three episodes are retained so a contact in an early
episode can expose the unchanged frozen-R11 return behavior on a later reset.

All model, roadmap, executor, candidate support, radius prior, posterior, and
passage rules are the same as the corresponding arm of the R12 factorial. The
output is isolated and must not be pooled with or substituted for the
20-command factorial result.
