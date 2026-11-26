# PointMaze R14 fixed-sigma versus latent-radius passage-gain screen

This is a consumed-development, truncated-horizon comparison. It asks one
question only: does latent-radius observation memory improve hidden-goal
acquisition enough to replace the fixed-sigma observation used by R11 when the
passage gain and all search mechanics are held fixed?

Both arms use the exact unique xy support of the frozen R11 recorded roadmap,
the same uniform initial goal mass, the same nearest-recorded passage
realization, the same pessimistic consequence connection, the same local
depth-two/top-four consumptive search, and the same executor. Command regions
and the learned-passage arm are not loaded.

For the fixed-sigma arm, a no-contact path updates candidate `g` by the R11
Gaussian-complement likelihood

`1 - exp(-d(path,g)^2 / (2 sigma^2))`,

where `sigma` is the held-out landing spread of the deployed consequence model.
For the latent-radius arm, each candidate retains its analytic feasible radius
interval under the shared log-uniform `1/rho` prior from learned `sigma` to the
exact recorded-support diameter. No environment success radius is read by
either arm.

The panel contains the two exposed corner diagnostics `r01_c06` and `r06_c06`.
The ordinary controls are `r03_c03` and `r05_c03`, selected before evaluation
as the two non-selected exposed development goals nearest the mean of the
unique recorded R11 roadmap xy support. Task outcomes are not used for this
selection.

Each goal receives eight episodes of five commands. The five-command cap is a
deliberate efficiency screen, not the twenty-command benchmark horizon. The
sigma arm is the R11 observation realization inside the shared local R12
passage-search adapter; it is not represented as a bitwise rerun of R11's
global dynamic-programming backup. This shared adapter is necessary to isolate
observation memory rather than confounding memory with search.
