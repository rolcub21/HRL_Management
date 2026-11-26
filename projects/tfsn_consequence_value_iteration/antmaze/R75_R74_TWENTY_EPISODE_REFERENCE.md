# R75: R74 with twenty carried episodes

R75 is a budget extension of the adopted R74 AntMaze acquisition method.  It
retains the frozen E549-5M executor, R52 reward-free preparation, latent-radius
memory, dynamic area-weighted hubs, sigma-generalized link correction, and the
same empirical R52 reach covariance on both sides of route-level evidence.

The only method-facing evaluation change is the number of carried physical
episodes per task: twenty rather than ten.  Each physical episode still has
twenty commands, every command is still held for exactly 100 primitive steps,
and task-local search and consequence memory persist across resets.  No
trajectory or evidence from the completed ten-episode R74 run is reused.

The report includes acquisition at episode 10 and episode 20.  This consumed
nine-goal development panel tests whether goal-area hubs first selected near
the previous boundary, especially task 5 in episode 9, convert under additional
search budget.  It is not confirmation evidence.

Return remains unchanged from R74 for this matched acquisition-budget test.
The implementation audit records that return currently uses R51 link-level
correction but not the route-level outcome correction used by acquisition.
