# R41 task-3 stall posture replay

This is a consumed-development diagnostic, not method evidence.  It replays the
first four commands recorded by the frozen R40 task-3 run from the same reset
and with the same frozen Farama SAC executor.  It records torso height and raw
quaternion-derived uprightness after every primitive step.  Posture is observed
only after the run and never gates, terminates, or changes a command.

The replay answers whether the first lasting command-level stall coincides with
an inverted or collapsed Ant.  The command sequence is a post-hoc oracle input
copied from the existing evaluation log.  Therefore no acquisition or planner
performance claim may be made from this replay.

