# PointMaze R10: two-command consumptive acquisition lookahead

R10 is an isolated consumed-development test over R9. The executor,
consequence ensemble, recorded road map, movement-conditioned reach bank,
bridge weights, exact 0.4 m negative-observation posterior, return planner,
tasks, starts, and seeds are unchanged.

R9 computes its value-iteration tail against one fixed search posterior. It can
therefore credit an imagined chain more than once for passages through the
same still-unsearched region. R10 tests the same global consumptive-lookahead
rule as AntMaze R32:

1. Rank all live acquisition candidates with the unchanged R9 score.
2. Keep the four highest-ranked candidates, where four is the global
   computational beam width shared by both domains.
3. For each candidate, match its reward-free primitive passage and consume the
   exact detection footprint along that imagined passage.
4. Recompute the unchanged bounded value-iteration tail on the consumed
   posterior and choose the highest pessimistic ensemble score.

This is called depth two because the first passage is scored and consumed
explicitly before the recomputed future tail is read. Return remains exactly
R8/R9. No task reward is used for learning, the goal coordinate remains
hidden, and no confirmation task is accessed.

The first gate uses only the two exposed development goals missed by R9. It is
not confirmation evidence. The diagnostic question is whether later episodes
select the last unresolved maze region instead of inheriting high value from
already consumed passages. A full nine-goal run is permitted only if this gate
shows that behavior without degrading physical coverage.
