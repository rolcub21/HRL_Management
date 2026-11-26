# R65: area-weighted R52 exploration-hub audit

R65 corrects one diagnosed bias in the R64 offline hub ranking.  R64 gave
every unique recorded roadmap coordinate equal prior mass, so preparation
regions containing many recorded coordinates received more hidden-goal prior
mass.  R65 instead gives every occupied landing-scale spatial cell equal total
mass.  Coordinates inside one cell divide that cell's mass equally.

The spatial cells use the deployed R52 landing precision, with their origin at
the componentwise minimum of the recorded support.  Both scale and origin are
therefore data derived.  There is no density bandwidth, neighbor count,
distance cutoff, free-space raster, or environment-specific region rule.

If (c(g)) is the landing-scale cell containing recorded coordinate (g), the
prior weight is

\[
p_0(g)=\frac{1}{C\,n_{c(g)}},
\]

where (C) is the number of occupied cells and (n_c) is the number of
unique recorded coordinates in cell (c).  Thus every occupied cell has mass
(1/C), independent of how often preparation visited it.

The latent-radius contact model, hub candidates, greedy marginal-coverage
ranking, ten-command graph reachability audit, and twenty-hub visualization
remain exactly as in R64.  Reachability remains an independent diagnostic in
R65; it does not alter the ranking.  R65 uses only frozen R52 preparation
artifacts and neither creates nor steps an environment.

R65 is an offline component audit, not hidden-goal evaluation evidence.
