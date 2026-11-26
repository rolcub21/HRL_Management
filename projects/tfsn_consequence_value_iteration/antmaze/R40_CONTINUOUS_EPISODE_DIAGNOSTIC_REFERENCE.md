# R40 continuous-episode acquisition diagnostic

R40 changes no TF-SN component.  It runs the frozen R39 acquisition method on
the same consumed two-goal development gate and under the same total budget of
50 commands (5,000 primitive steps), but removes physical reset fragmentation:

- R39 reference: five physical episodes of 10 commands;
- R40 diagnostic: one physical episode of 50 commands.

R39's 10-command computational lookahead is unchanged.  Its lookahead counter
restarts every ten commands without resetting the environment, so the only
intervention is physical continuity.  Task memory, consequence model, graph,
executor, posterior, candidate set, and command duration are frozen.

This is not benchmark evidence because the AntMaze benchmark defines a
1,000-step physical episode.  It is a consumed-development acquisition
diagnostic answering whether repeated physical resets help or hinder finding a
hidden goal under equal interaction.
