# R60: E549/R52/R51 latent-radius hidden-goal original-nine panel

R60 is a consumed-development comparison against R59 on the same original
nine AntMaze goals, resets, evaluation seed, five carried episodes, and fixed
ten-command physical-episode contract.  It retains the frozen E549 5M
executor, E549-matched R52 consequence preparation, and R51
destination-arrival online correction.

The only acquisition-method factor changes from R59's learned-landing-scale
Gaussian memory to the previously declared latent-radius observation model.
The hidden goal prior is uniform over the unique xy coordinates of the R52
recorded roadmap.  The unknown detection radius has a log-uniform prior from
the deployed R52 landing scale to the exact diameter of that support.  Each
observed primitive binary contact consequence updates the feasible interval
for every goal candidate analytically; there are no radius bins and the
evaluator's 0.8 m success radius is never read by the planner.

Candidate passages use the same locally matched reward-free empirical passage
geometry as the nearest-recorded R49 arm.  The advance prefilter is
observation-neutral maximum reliability, which is required to prevent R59's
Gaussian memory from leaking into the latent-radius arm.  Return behavior is
the unchanged R51/R52 return planner after first contact.  Posterior exhaustion
is a fatal model-falsification result; no fallback posterior is allowed.

Report acquisition by episodes one and five, conditional return, end-to-end
success, selected-destination arrival, and acquisition coverage separately for
each physical episode.  R60 is consumed-development component evidence, not
confirmation evidence.
