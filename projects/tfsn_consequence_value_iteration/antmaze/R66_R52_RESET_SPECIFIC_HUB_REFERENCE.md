# R66: reset-specific reach-weighted hub diagnostic

R66 is a consumed-development diagnostic built on the first twenty
area-corrected R65 hubs.  It asks whether continuous reach weighting changes
their useful order for the fixed reset states of failed development tasks 1,
2, and 8.

For each task, the environment is reset with its frozen development reset
seed but is never stepped.  The resulting full Ant state is associated with
the nearest R52 roadmap node in the R52 consequence model's standardized
state coordinates.  Ten-command reachability to each R65 hub is then computed
by the same Bellman recursion used in R64 and R65.

At every greedy selection, hub (h) receives

\[
\operatorname{score}(h\mid S,s_0)
=p_{\mathrm{reach}}(h\mid s_0)\,
\Delta_{\mathrm{area}}(h\mid S),
\]

where the area-corrected marginal coverage is exactly R65's and reachability
is specific to the task's reset state.  There is no reachability cutoff.

This audit reranks only the first twenty R65 hubs.  It is therefore a cheap
diagnostic, not the deployed all-candidate hub mechanism.  The development
goal coordinates are loaded only after all rankings are complete and are used
only as plot overlays and post-selection distance/contact diagnostics.  They
never enter coverage, reachability, or ranking.

R66 is not confirmation evidence and does not execute a hidden-goal episode.
