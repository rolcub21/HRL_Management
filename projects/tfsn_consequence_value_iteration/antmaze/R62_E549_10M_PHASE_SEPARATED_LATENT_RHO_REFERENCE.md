# R62: E549 10M phase-separated latent-radius original-nine panel

R62 is the executor-duration comparison to R61 on the same consumed
original-nine AntMaze development goals, resets, evaluation seed, five
episodes, and ten fixed 100-step commands per physical episode.

The deployed executor is the frozen E549 10M checkpoint.  Every learned
planning artifact comes from its matched R54 preparation: consequence model,
recorded graph, mobility model, bridges, landmarks, and empirical passage
geometry.  No R52/5M planning artifact or trajectory is admissible.

The task-time method is otherwise R61.  The analytic latent-radius posterior
receives primitive no-contact observations only while acquisition is active.
Within the first-contact command it consumes only the no-contact prefix, then
is permanently retired.  Positive and post-contact observations update only
the existing return evidence field.  Pre-contact posterior exhaustion remains
fatal, with no fallback or relaxation.  The evaluator success radius is never
read by the planner.

All nine tasks execute fresh because changing both executor and matched
preparation invalidates R60/R61 trajectory reuse.  R62 is
consumed-development component evidence, not confirmation evidence.
