# PointMaze R13 command-region acquisition screen

This is an isolated consumed-development screen, not confirmation evidence. It
runs latent-radius observation memory with command-scale region gain on the two
historically exposed PointMaze corner goals for eight episodes each.

The region scale is derived from reward-free recorded transitions. For every
record, requested and achieved movement receive the already-shared continuous
weight

`omega = (1 - kappa_sigma(requested)) (1 - kappa_sigma(achieved))`.

The cover radius is the weighted median achieved displacement, with no cutoff
or multiplier. This prevents recorded stalls from defining the planner's
movement unit while retaining them continuously. Candidate goals are the exact
unique recorded roadmap xy states. The first centre is the candidate nearest
the support mean. Each subsequent centre is the candidate farthest from its
nearest existing centre, until every candidate is within one derived reach of
a centre. Thus the radius means "one command from the centre"; a complete
region may span two reaches.

Centre selection determines only the number and locations of centres. Region
assignment is a directed multi-source shortest-path calculation over all
positive-reliability recorded and model-bridge links, running outward from the
centres. A link with reliability `w` costs `1/w`, its expected number of retry
commands. Duplicate links retain their maximum reliability. Zero reliability
has infinite cost; there is no positive-reliability cutoff. If any candidate is
unreachable outward from all geometrically selected centres, the cover applies
the planner's existing alias resolution before declaring a roadmap gap.

Specifically, every initially unreachable exact-state candidate may receive at
most one zero-command attachment from its nearest originally reachable
candidate within learned landing sigma. Directed shortest paths are then rerun
with those attachments and the ordinary roadmap links. Alias attachments never
chain: transitive sigma-neighborhood closure would let dense samples percolate
across a macroscopic part of the maze, contradicting the meaning of one
controllable place. Candidates still unreachable after this step remain in the
goal posterior, because the hidden goal may lie there, but belong to no
targetable region. Their current posterior mass is reported at every decision.
The cover never falls back to unrestricted straight-line assignment.

At each acquisition decision, a region's value is the sum of the current
latent-radius goal posterior over candidates assigned to it. Candidate commands
retain their original node targets and consequence connection weights. Within
each proposed target region, the retained node maximizes its exact node-local
latent contact probability times its consequence connection. Regions are then
scored by their remaining posterior mass times that representative's
connection.

Depth-two lookahead conditions its imagined no-contact posterior only on the
selected target node. Its no-contact branch weight is likewise one minus that
node-local contact probability; it does not remove or declare consumed the
rest of the region. No passage model or nearest-trajectory match is read by
acquisition. Candidate coordinates must exactly match the posterior ordering
before masses can be accumulated, preventing silent probability reordering.
After contact, frozen R11 return is unchanged.

The screen retains R12's local depth-two/top-four acquisition backup so that
only the gain representation changes relative to the latent-radius plus
nearest-recorded arm. Integrating the same region gains into a new global
backup would be a separate search change and is intentionally not mixed into
this causal check.
