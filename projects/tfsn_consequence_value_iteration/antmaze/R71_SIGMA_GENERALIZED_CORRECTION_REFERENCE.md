# R71: sigma-generalized online consequence correction

R71 tests the recurring coverage failure across all nine consumed AntMaze
development goals. It uses the frozen E549-5M executor and R52 reward-free
preparation, R69's 20-command physical episode and ten-episode task budget,
the complete data-derived dynamic hub bank, and R70's exact corrected
reachability selection.

The only method change is how online destination-arrival evidence generalizes
between links. R51 compared the complete 29-dimensional start states. Two
crossings a few centimetres apart could therefore receive virtually no shared
evidence when the Ant's posture differed. R71 represents a link by the spatial
geometry relevant to its consequence: source position, issued command target,
and intended graph destination. For an observation `o` and link `e`, its
continuous similarity is

```text
exp(-0.5 * (
    ||source_e - source_o||^2 / sigma^2
  + ||command_e - command_o||^2 / sigma^2
  + ||destination_e - destination_o||^2 / sigma^2
))
```

where `sigma` is the deployed, held-out landing precision of the R52
consequence model. There is no distance cutoff, neighbour count, wall test, or
environment-specific constant. The same kernel updates live routing candidates
and all modeled graph bridges used for hub reachability. Recorded transitions
remain unchanged because they are witnessed preparation edges.

Hub selection is the exact global maximizer of posterior coverage times
corrected reachability. As in R70, branch-and-bound uses coverage as a valid
upper bound and stops only after no unevaluated hub can beat the best exact
score. Maze geometry and the hidden goal are unavailable to the planner.

The nine-goal panel is development evidence. Its purpose is to determine the
dominant remaining failure across goals rather than optimize another component
on task 8.

