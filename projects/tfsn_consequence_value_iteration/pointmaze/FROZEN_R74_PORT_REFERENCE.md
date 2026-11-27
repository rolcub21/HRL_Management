# PointMaze conformance port of frozen R74

The canonical cross-domain equations, setting provenance, and rebuild
procedure are in
[`../FROZEN_R74_METHOD_SPECIFICATION.md`](../FROZEN_R74_METHOD_SPECIFICATION.md).
The machine-readable inventory is
[`../shared/frozen_r74_settings.json`](../shared/frozen_r74_settings.json).

This revision applies the frozen AntMaze R74 decision method to the unchanged
PointMaze R11 reward-free preparation.  PointMaze contributes only its state
and action adapter, executor, recorded data, and benchmark horizon.

Acquisition uses the same area-weighted latent-radius posterior, dynamic hub
bank, exact coverage-times-reachability argmax, destination-arrival link
correction at the deployed landing scale, and route correction at the
empirical movement-weighted reach covariance.  Neither correction has a
cutoff.  The benchmark success radius, maze geometry, hidden goal coordinate,
and task reward are absent from every planner input.

Return uses the pre-existing R11 supplied-target router with the frozen
destination-arrival link correction.  R74 did not adopt the rejected R76--R78
return-route variants, so this port does not add them.

The conformance gate consists of the exact consumed R11 development panels:
the original nine tasks and the disjoint fifteen tasks, eight episodes each.
R11 acquired 7/9 and 15/15, or 22/24.  This revision passes only if it acquires
at least 22/24 without accessing the sealed confirmation panel.
