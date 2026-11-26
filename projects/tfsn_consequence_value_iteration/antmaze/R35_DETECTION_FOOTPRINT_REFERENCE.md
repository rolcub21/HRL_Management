# R35 detection-footprint hidden-goal acquisition

R35 is R34 with one semantic separation.  Landing precision and the reward
observation footprint no longer share one bandwidth.

- The learned R30 landing precision remains the scale for spatial aliasing,
  supplied hypotheses, consequence connections, and the return field.
- Before contact, the residual search field uses the benchmark-defined contact
  radius.  This is the area in which a hidden goal would have produced an
  observable reward, so it is the appropriate scale for negative evidence.

For visited primitive positions \(x_j\), acquisition reads

\[
A_t(x)=1-\max_{j\leq t}\exp\left(
-\frac{\lVert x-x_j\rVert^2}{2\rho^2}
\right),
\]

where \(\rho\) is the declared reward-contact radius.  R34's posterior mass and
particle-survival ratio read this field.  Its sampled hypothesis is still routed
by the frozen supplied-goal planner at landing precision.  After first contact,
return is exactly R31/R34 and uses landing precision.

The contact radius is task observation-model knowledge, not the hidden goal,
maze geometry, a routing threshold, or a tuned planner value.  It is read from
the same environment contract used by the evaluator and is reported explicitly.
No confirmation goal is accessed.  This is a consumed two-goal development
gate.
