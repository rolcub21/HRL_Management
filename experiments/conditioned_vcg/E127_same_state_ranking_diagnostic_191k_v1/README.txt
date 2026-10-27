E127: frozen same-state proposals and two guarded counterfactuals
==============================================================

This bounded diagnostic uses E123 A seed 0's episode-120 candidate and the
first existing development seed 187800. No training, new panel or fitting.
The CPU native reference reproduces the saved validation cost and step count.
Model/target weights and replay are unchanged, and the source file is hashed.

At each native macro boundary, clone the physical yard and ask the unchanged
release-informed Dynamic PSLAP scheduler for one proposal. Use E126's input
clock substitution for placement previews and projected admission. Never
execute a baseline action on the native yard. The query reconstructs a fresh
baseline scheduler (empty private history) on that physical state; it is a
baseline-inspired local suggestion, NOT a counterfactual full baseline history.
An obstructed retrieval maps to its first route-clearing relocation; unobstructed
retrieval and admission match explicit block/destination candidates. Defer
matches intent only, since the two controllers may have different event bounds.
No Defer proposal is used as an intervention in this pilot.

Separate absence from the exact SAFE frontier, guard/interface exclusion,
critic merit preference and hierarchical rejection of a higher candidate
merit. Absence alone is not called UNSAFE: exact UNKNOWN or interface mismatch
would require further attribution. All 118 non-Defer suggestions in this pilot
were represented in the admissible certified frontier.

Capture the FIRST admissible disagreement of each prespecified category:
  1. both choose admission but disagree on cell;
  2. baseline proposes delivery while native chooses something else.
Do not select cases by rollout benefit. For each category, replay from reset
with identical prefix keys, physical-state and guard digests, candidate keys
and critic scores. Execute exactly one different already-certified candidate
at the fork, then resume native frozen VCG with the original guard. Never
override a forced/committed guard action. Compare complete episode J; common
prefixes make this equivalent to a remaining-cost contrast.

Maximum three episode executions: reference plus one branch per category.
The launcher has a separate Linux timeout of 600 seconds, TERM then KILL after
10 seconds; this does not depend on Python handling Ctrl-C. Partial completed
episode artifacts can be reused; unfinished work is not promoted to results.

Run:
  bash experiments/conditioned_vcg/E127_same_state_ranking_diagnostic_191k_v1/run.sh run-all
  bash experiments/conditioned_vcg/E127_same_state_ranking_diagnostic_191k_v1/run.sh analyze

Artifacts:
  results/vcg-conditioned-e127-same-state-ranking-diagnostic-191k-v1/
    reference.json, branch-admission_cell.json, branch-retrieval_timing.json
    contract.json, e127-report.json, INTERPRETATION.txt

This first instance already has zero timing penalties under VCG, so its small
cost gap to Dynamic PSLAP is equipment work, not an outside-tolerance timing
failure. Results cannot explain all five instances or prove global optimality.
Counterfactual signs are conditional on the native learned continuation: a
good baseline action need not be good when subsequent choices remain VCG's.
