# AntMaze R50 latent-radius eight-episode miss extension

This is an isolated consumed-development extension, not confirmation or
factorial evidence. It runs only latent-radius observation memory with the
nearest-recorded passage adapter on AntMaze tasks 0 and 2 for eight episodes
per task.

Both tasks were unacquired in the R48 full 15-task panel. Tasks 1 and 4 are
excluded before choosing replacements because they were already used by the
R49 factorial and contact can turn later episodes into return trials. Among the
remaining R48 misses, tasks 0 and 2 are selected by the deterministic rule
"lowest two task IDs"; no R50 behavior is consulted.

Every episode retains AntMaze's benchmark horizon of 10 commands (1,000
primitive steps). Task memory persists across all eight episodes. Each result
labels whether an episode began in acquisition or return and records the
maximum Euclidean distance between any actual command endpoint and that
episode's observed reset position. This report-only displacement uses neither
the hidden goal nor the evaluator contact radius and never enters planning.

All R49 latent-radius and nearest-recorded passage mathematics, R48 artifacts,
R39 immediate adaptive-greedy search, and R46 online correction semantics are
unchanged. The output is isolated and must not be pooled with the four-arm R49
factorial.
