# R57 E549 task-13 posture replay

R57 is a consumed-development diagnostic, not method or performance evidence.
It replays all ten commands recorded by the R56 supplied-goal task-13 audit
from the same reset, using the same hash-pinned E549 5M executor.  It records
torso height, quaternion-derived uprightness, and position at every primitive
step, together with command-level movement and graph-destination error.

The focus is zero-based command index 7, after which the R56 trace lost useful
mobility.  Commands 8 and 9 are also replayed so the diagnostic can distinguish
a transient bridge miss from a persistent physical state.  The complete
recorded sequence is a post-hoc oracle input.  Posture never gates, terminates,
or changes execution, and no task reward or hidden goal is consumed.
