# R42 posture-risk audit

R42 is a consumed-development diagnostic of the frozen R30 consequence model
used by R39.  It does not train or modify a model and does not evaluate a new
hidden-goal policy.

The diagnostic answers three questions:

1. Is the deployed posture head multimodal?
2. At the exact R40 task-3 command that flipped the Ant, how much probability
   does the frozen model assign to an inverted endpoint, and how much of that
   risk survives the deployed mobility readout?
3. On the untouched R29 task-held-out split, are predicted endpoint-flip
   probabilities calibrated to observed flips?

The command state is reconstructed by deterministically replaying the first
two recorded R40 commands.  Those commands and the reset seed are oracle inputs
for diagnosis only.  R29 held-out tasks remain excluded from model fitting.

An endpoint is called inverted only for reporting, using the same geometric
quantity as R41: the torso local vertical axis has negative world-z component.
This diagnostic threshold is not available to the planner and is not a method
rule.

