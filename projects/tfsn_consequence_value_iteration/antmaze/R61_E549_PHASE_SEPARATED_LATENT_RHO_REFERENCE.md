# R61: phase-separated latent-radius search on E549/R52/R51

R61 corrects the phase-boundary error exposed by R60 on the same consumed
original-nine AntMaze development panel.  The E549 5M executor, R52
preparation, R51 destination-arrival correction, latent-radius acquisition
model, task rows, resets, evaluation seed, and five-episode contract are
unchanged.

The latent posterior is search memory.  While acquisition is active, every
observed primitive no-contact consequence updates it.  Within the command
that first contacts the hidden goal, only the no-contact prefix before the
first positive observation is incorporated.  At first contact the search
posterior is permanently retired.  The positive observation and all later
observations update only the existing reward/evidence field used by the
unchanged R51/R52 return planner.  The evaluator's 0.8 m success radius is
never read by the planner; the planner receives only the observed binary
contact consequence.

This is phase separation, not a fallback posterior.  Exhaustion before first
contact remains a fatal falsification result, and there is no posterior reset,
relaxation, or environment-specific threshold.

R60 tasks 0--6 may be reused only after the R61 runner proves execution
equivalence from their saved rows and command logs.  No-contact tasks contain
only negative acquisition evidence, so the update is identical.  For an
acquired task, acquisition decisions through first contact are identical; no
later acquisition decision exists, and the return policy does not read the
latent posterior.  Reused rows retain hashes of the source task and log and
are explicitly marked as execution-equivalent reuse.  Tasks 7 and 8 execute
fresh because task 7 was interrupted by the R60 post-contact failure.

R61 remains consumed-development component evidence, not confirmation
evidence.
