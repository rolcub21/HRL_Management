# Contention and storage-rehandle metrics

## Why this metric contract exists

The Park--Seo planar storage location assignment problem (PSLAP) minimizes
obstructive object moves, and the Kim--Jeong--Shin stockyard controller reports
rearrangements.  Those quantities are not transporter travel, primitive action
count, or episode return.  They also are not automatically identical to every
storage-to-storage move made by a controller.

The simulator historically exposed `info["relocated_block"]` and several
evaluators reported its total as `obstructive_moves`.  The event itself has a
narrower, policy-independent meaning: it is emitted once when an already stored,
undelivered block is successfully put down at a different storage location.  It
does not encode why the controller moved the block.  Calling every such event an
obstruction clearance was therefore incorrect for controllers that can choose a
standalone `ReconfigureOption`.

## Versioned event contract

New evaluations use `physical_storage_relocation_decomposition_v1`.

`physical_storage_relocations`
: All completed storage-to-storage block moves.  This is the common operational
  resource metric and the corrected name for the historical `relocations`
  count.  During migration, `relocations` may remain as an exact deprecated
  alias.

`target_bound_obstruction_clearances`
: Physical relocations executed inside a retrieval macro for a named target
  because blocks on that target's selected path had to be cleared.  This is the
  closest simulator-side mechanism to the traditional PSLAP rearrangement
  count.  It remains an adaptation: the original PSLAP model can include both
  inbound and outbound obstruction events and may restore displaced blockers.

`standalone_reconfigurations`
: Physical relocations selected as independent reconfiguration macros rather
  than as substeps of a named target retrieval.

`standalone_with_direct_delivery_available`
: Standalone reconfigurations selected at a decision boundary where at least one
  exactly safe direct-delivery action was also available.  This is evidence of a
  discretionary reconfiguration opportunity, but it does not by itself prove
  that the move was unrelated to every other blocked target.

`standalone_without_direct_delivery_available`
: Standalone reconfigurations selected at a boundary with no exactly safe direct
  delivery.  This is a recovery-pressure proxy, not automatically the exact
  Park--Seo target-specific obstruction count because the VCG controller has not
  first fixed a retrieval target.

`directly_deliverable_self_reconfigurations`
: A subset of `standalone_with_direct_delivery_available` in which the moved
  block itself had an exactly safe direct-delivery action.  This is the strongest
  observable evidence of proactive self-repositioning.

Every row must satisfy

```text
physical_storage_relocations
  = target_bound_obstruction_clearances + standalone_reconfigurations

standalone_reconfigurations
  = standalone_with_direct_delivery_available
  + standalone_without_direct_delivery_available

0 <= directly_deliverable_self_reconfigurations
   <= standalone_with_direct_delivery_available
```

Counts are accumulated from completed environment events, not from proposed or
selected actions.  A failed or truncated reconfiguration that never reaches the
new storage-cell putdown contributes zero.

## Reporting contract

The primary common contention-consequence coordinate is

```text
100 * sum(physical_storage_relocations) / sum(delivery_count)
```

and is reported as physical storage relocations per 100 deliveries.  It retains
the operational cost of both reactive and proactive rehandling and therefore
remains fair across controller architectures.

The cause fields are reported alongside it as mechanism diagnostics.  A
comparison must not replace total physical relocations with only target-bound
clearances: doing so would structurally give a controller whose relocations are
packaged as standalone macros an artificially low cost.  Timing MAE, tardiness,
window compliance, primitive steps, return, completion, and failures remain
separate outcomes.

Low relocation counts also do not establish that a benchmark activated
contention.  Evaluations should expose the number of decision boundaries with
no safe direct delivery, occupancy pressure, and recovery demand when those
signals are available.

## Controller mapping

- VCG direct delivery performs no storage relocation.  Its physical relocation
  events arise from standalone reconfiguration macros and are divided by the
  safe-direct-delivery state visible before execution.
- The source-neutral Track-B deterministic systems relocate blocks inside their
  named retrieval executor.  Their physical relocation events are therefore
  target-bound obstruction clearances under this execution interface.
- Track-A and other legacy executors may mix named-retrieval relocation and
  autonomous recovery fallback.  Their old `obstructive_moves` fields must not
  be assumed target-bound without the same origin instrumentation.

## GA objective

The current Park--Seo-style GA implementation does not use environment return
as its primary fitness.  `AssignmentCost` minimizes infeasible events first,
then predicted obstructive moves, and then route steps.  Duration-aware and
operational rolling variants add urgency and route terms under separately named
extensions.  Any manuscript text describing the current GA as return-fitted is
out of date.

## Protocol migration

Historical result files and their hashes remain immutable.  Their
`relocations` values are valid totals of physical storage rehandles, while their
VCG `obstructive_moves` alias has no causal interpretation.  Corrected historical
decompositions must be written as authenticated sidecars or obtained from a new
instrumented evaluation; old ledgers must never be silently rewritten.

The prospective VCG-Dense V1.2 checkpoint selector continues to rank the exact
same physical event count.  Renaming that coordinate and recording its
decomposition is a semantic repair, not a change to its selection objective or
training recipe.

## Authenticated development reanalysis

The frozen 80000--80029 development ledgers preserve every ordered exact VCG
frontier.  `reanalyze_vcg_contention_metrics.py` authenticates the original
ledger hashes and infers each executed macro by matching a candidate successor
certificate to the next recorded frontier.  All reconstructed reconfiguration
counts equal the historical physical-event totals; no trajectory is rerun.

For the three selected VCG checkpoints, the repaired 720-delivery result is:

| Quantity | Count | Per 100 deliveries |
|---|---:|---:|
| Physical storage relocations | 172 | 23.889 |
| Standalone with any safe direct delivery available | 167 | 23.194 |
| Directly-deliverable self-reconfigurations | 166 | 23.056 |
| Other reconfiguration while another direct delivery was available | 1 | 0.139 |
| Standalone with no safe direct delivery available | 5 | 0.694 |

Thus the historical `obstructive_moves = 172` label was causally misleading.
The 166 self-reconfigurations are strong evidence of proactive repositioning;
the five no-direct-delivery moves are recovery-pressure events but are not
automatically the exact Park--Seo target-bound count.  Only 135 of 2,059 exact
decision frontiers had no safe direct delivery, and five of those 135 led to a
reconfiguration.  This panel therefore contains relatively little forced
retrieval contention.

The complete development-only report is stored under
`results/vcg-contention-metric-reanalysis-development-30seed/`.  It retains
failed baseline rows and suppresses their primary numeric rates rather than
forming a complete-case comparison.

## Literature anchors

- Park, C., and Seo, J. (2009), *Mathematical modeling and solving procedure of
  the planar storage location assignment problem*, DOI:
  <https://doi.org/10.1016/j.cie.2009.04.010>.
- Kim, B., Jeong, Y., and Shin, J. G. (2020), *Spatial arrangement using deep
  reinforcement learning to minimise rearrangement in ship block stockyards*,
  DOI: <https://doi.org/10.1080/00207543.2020.1748247>.
