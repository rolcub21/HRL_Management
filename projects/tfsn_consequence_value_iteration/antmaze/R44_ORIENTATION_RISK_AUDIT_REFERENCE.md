# R44 orientation-risk audit

R44 applies the frozen R43 orientation-factorized consequence model to the
same consumed R40 task-3 command diagnosed by R41 and R42.  It does not train
the model or evaluate hidden-goal performance.

The exact command state is reconstructed by deterministic replay.  R44 reports
the learned orientation mixture at both the actual and commanded endpoint,
the joint predicted flip probability, and the response of the still-frozen R31
mobility readout to R43's physical endpoint samples.  R30/R39 artifacts remain
unchanged.

