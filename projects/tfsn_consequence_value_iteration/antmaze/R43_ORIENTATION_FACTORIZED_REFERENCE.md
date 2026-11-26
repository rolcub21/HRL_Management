# R43 orientation-factorized consequence model

R43 changes only the posture parameterization of the frozen R30 consequence
model.  The spatial mixture, reward-free training data, whole-task split, and
training schedule remain those of R30.

The endpoint distribution is factorized as

1. endpoint position relative to the command (the frozen R30 spatial head),
2. torso rotation conditioned on state, command, and endpoint position, and
3. the remaining 23 posture coordinates conditioned additionally on rotation.

Rotation is represented during learning by the first two columns of its
rotation matrix.  Each orientation head is a five-component mixture.  Samples
are projected by Gram--Schmidt to a valid rotation and converted back to a
unit quaternion.  Quaternion sign is chosen to be closest to the command's
start quaternion, matching trajectory continuity.  This representation has no
uprightness label or posture threshold: upright and inverted modes must emerge
from the reward-free endpoint data.

R43 is a component model only.  It does not replace R30/R39, rebuild planner
artifacts, or access hidden goals.  The consumed R40 flip is reserved for the
subsequent diagnostic and is not used for fitting or model selection.

