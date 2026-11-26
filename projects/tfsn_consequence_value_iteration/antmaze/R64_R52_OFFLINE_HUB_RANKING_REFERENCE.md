# R64: R52 offline latent-coverage hub ranking audit

R64 is a reward-free, offline feasibility audit for ordered exploration hubs.
It does not create or step an environment and does not load a hidden-goal
panel.  Every input is a frozen R52 preparation artifact.

Hub locations are representative recorded roadmap nodes, one per occupied
R52 landing-scale alias cell.  This removes near-duplicate candidate states
at exactly the deployed landing-precision resolution without introducing a
new spatial scale.  The hidden-goal prior remains the R60/R61 prior: uniform
over every unique recorded roadmap coordinate, with a log-uniform detection
radius from R52 landing sigma to the exact support diameter.

For a possible goal coordinate \(g\) and hub \(h\), initial contact
probability is

\[
q_0(g,h)=
\begin{cases}
1, & \lVert g-h\rVert\leq\rho_{\min},\\
\log(\rho_{\max}/\lVert g-h\rVert)/
\log(\rho_{\max}/\rho_{\min}), & \rho_{\min}<\lVert g-h\rVert<\rho_{\max},\\
0, & \lVert g-h\rVert\geq\rho_{\max}.
\end{cases}
\]

Hubs are greedily ordered by exact marginal expected contact probability.  If
\(S\) is the already ordered prefix, the next hub maximizes

\[
\mathbb E_g\left[
\max\{q_0(g,h),\max_{s\in S}q_0(g,s)\}
-\max_{s\in S}q_0(g,s)
\right].
\]

There is no coverage threshold and no deployed hub count.  The first twenty
are materialized only for this requested visualization and feasibility
audit.

Reachability is not used to alter that coverage ranking.  It is reported as a
separate check.  For every selected hub, ten-command Bellman backups run over
all R52 recorded edges and bridges, with their frozen reliability weights and
failure self-loops.  The initial states are every physical-episode reset state
recorded in R52 base-graph collection, mapped through the graph's exact source
indices.  Thus both coverage and reachability use preparation data alone.

R64 is an offline component audit, not hidden-goal evaluation evidence.
