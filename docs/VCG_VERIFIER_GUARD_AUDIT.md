# VCG verifier and recovery-guard implementation audit

## Status and scope

This document maps the implemented deterministic recoverability verifier,
finite recursive disturbance verifier, and recovery-witness guard to their
algorithmic contracts and focused correctness evidence. It is intended to make
the implementation independently inspectable and to delimit what the evidence
does and does not establish.

The native verifier supplies constructive positive certificates for the
closed-admission strict-macro recovery problem. The recursive verifier is a
separate robust extension over declared finite disturbance sets. The tests and
instrumented replay below support implementation correspondence; they do not
replace the paper's explicit assumption that a returned positive certificate
is sound for the stated transition model, nor do they establish correctness
under unmodeled execution errors.

## 1. Native deterministic recovery verifier

### 1.1 Recovery state and macro successors

`RecoveryState` is immutable and contains grid dimensions, traversable and
storage cells, exits, block identities and locations, agent position, fixed
obstacles, storage reservations, pickup cells, and waiting cells. At a strict
decision boundary no block is carried and no assignment is half committed.

For each stored block, `legal_recovery_actions`:

1. Finds a deterministic shortest empty-hand approach path with four-neighbor
   motion. All other blocks and fixed obstacles are impassable.
2. Treats the selected block as removed while it is carried.
3. Generates a delivery macro for each reachable exit.
4. Generates a relocation macro for each reachable empty, unreserved storage
   cell.
5. Sorts macros by delivery before relocation, then block label, destination,
   approach path, and transport path.

`apply_recovery_action` re-enumerates the legal actions before applying a
macro. Delivery removes the block; relocation changes its storage position;
the agent finishes at the destination.

### 1.2 Native search pseudocode

```text
CERTIFY-NATIVE(state s, depth cap H, node cap N, step horizon P, order O)
    validate s and all limits
    if s has no stored blocks:
        return SAFE(empty witness)

    root <- NODE-KEY(s, depth=0, steps=0)
    parent[root] <- null
    OPEN <- FIFO(root)                    if O = breadth_first
            priority queue(root)          if O = goal_directed
    CUT <- empty set
    explored <- 0

    while OPEN is not empty:
        if N is finite and explored >= N:
            return UNKNOWN(node budget exhausted)

        u <- POP(OPEN)
        explored <- explored + 1
        A <- LEGAL-RECOVERY-ACTIONS(u.state) in canonical order

        if H is finite and u.depth >= H:
            record every novel successor that would otherwise be explored in CUT
            continue

        for action a in A:
            next_steps <- u.steps + primitive_steps(a)
            if P is finite and next_steps > P:
                continue
            v <- APPLY-LEGAL-ACTION(u.state, a)
            key <- NODE-KEY(v, u.depth + 1, next_steps)
            if key was already generated:
                continue
            store v and parent[key] <- (u, a)
            if v has no stored blocks:
                return SAFE(RECONSTRUCT-WITNESS(parent, key))
            PUSH(OPEN, key)

    if CUT is nonempty:
        return UNKNOWN(depth budget exhausted)
    return UNSAFE(reachable state space exhausted within P, if supplied)
```

Breadth-first OPEN is a FIFO queue. Goal-directed OPEN uses the deterministic
tuple `(remaining block count, macro depth, primitive steps, insertion serial)`.
The latter is a feasibility order, not an assertion of shortest recovery rank.

### 1.3 Duplicate handling and termination

Within one search, the physical configuration key contains fixed obstacles,
reservations, agent position, and sorted `(block label, position)` pairs. Static
geometry is constant within that search. When a depth or primitive-step bound
is active, depth and consumed steps are also retained in the node key so a
longer early discovery cannot suppress a later resource-feasible route.

The three terminal statuses are:

- `SAFE`: a finite generated witness reaches an empty admitted workload;
- `UNSAFE`: the reachable graph was exhausted, optionally inside the declared
  primitive-step horizon;
- `UNKNOWN`: the node or macro-depth computation budget was exhausted.

Only `SAFE` successors enter the controller frontier. `UNSAFE` and `UNKNOWN`
both fail closed for candidate admission, but remain distinct audit outcomes.

### 1.4 Cache identity

The native cache key is:

```text
(complete RecoveryState, max_depth, max_nodes, max_primitive_steps, search_order)
```

It stores certificates rather than preferences. Consequently, learned scores
cannot change cache results. Search outcomes, including `UNKNOWN`, may be reused
only for the identical recovery state and computation contract.

The optional E14 timing-invariant cache uses a separately tested state
projection justified only for the recovery transition model; operational clocks
remain available to ranking and performance measurement. D12 relocation-family
proofs are constructive certificates and are deliberately not inserted into
the native outcome cache. An unresolved family attempt falls back to native
search.

## 2. Finite recursive disturbance verifier

The robust extension represents each action by a nonempty, uniquely identified
finite outcome set. Each outcome contains an endpoint, an intermediate-tube
safety result, a primitive duration, and a disturbance identifier.

### 2.1 Fixed-point pseudocode

```text
ROBUST-COMPLETION-KERNEL(game G)
    K <- terminal states
    rank(s) <- 0 for s in K
    j <- 0
    repeat
        ADD <- all nonterminal, nonunsafe states s outside K for which
               there exists an action c such that every declared outcome w:
                   T(s,c,w) is safe, and F(s,c,w) is in K
        if ADD is empty:
            return K and rank
        j <- j + 1
        assign rank(s) <- j for every s in ADD
        K <- K union ADD
```

Updates are synchronous: a state added in iteration `j` cannot certify another
state until iteration `j+1`. Therefore cycles outside the existing completion
set cannot certify one another. A malformed or unresolved outcome fails closed.
At deployment, the robust frontier requires every branch to remain inside the
fixed point. A bounded liveness rule permits at most one same-rank macro before
forcing an action whose every outcome has strictly smaller robust rank.

For the live 5x5 exhaustive tree, each allowed disturbance is executed through
a deep-copied authoritative environment and replanning occurs at every realized
successor. Convergent suffixes use a complete causal live-state key; terminal
aliases, resource cutoffs, failures, and invalid executions remain distinct.
`PASS` requires universal completion of all declared branches; any cutoff gives
`UNKNOWN`, not success.

## 3. Recovery-witness guard

The guard state is `(n, W, i)`: the number `n` of consecutive nonprogress
recovery choices, a retained finite witness `W`, and its cursor `i`.

```text
GUARDED-SELECT(snapshot S)
    trigger <- n >= nonprogress_limit or (force_when_due and a block is due)
    if W is inactive and trigger and S contains a recovery candidate:
        require S.current_certificate = SAFE with a nonempty witness
        W <- that witness; i <- 0

    if W is active:
        require exactly one SAFE frontier candidate matching W[i]
        return that singleton candidate as witness-forced
    return the learned selection over the complete exact-SAFE frontier

OBSERVE(decision d, observed next snapshot S', terminal done)
    if d was witness-forced:
        require the observed physical successor equals the certified successor
        i <- i + 1; n <- 0
        if i = length(W) or done: clear W, i, and n
        return

    if d is Accept:
        clear W, i, and n
    else if d is Recover:
        if d is Deliver or exactly decreases authenticated recovery rank: n <- 0
        else: n <- n + 1
    if done: clear W, i, and n
```

An active witness is never replaced by a certificate from a later snapshot.
The exact frontier is reduced to the uniquely matching witness macro, so Accept
and Defer cannot interrupt the commitment. A missing, duplicated, or
successor-inconsistent witness action is a method failure, not permission to
return to the learned frontier. Once the witness finishes, later admission may
resume under a newly certified recovery problem.

### 3.1 Connection to the finite-progress proposition

The implementation supplies the operational conditions used by the paper's
finite-progress proposition:

| Proposition condition | Implementation mechanism |
|---|---|
| Only certified actions execute | exact-SAFE candidate construction |
| Learned nonprogress is bounded | explicit `nonprogress_recovery_decisions` counter |
| A finite completion witness is committed | copied `current_certificate.witness` |
| Commitment cannot be interrupted | singleton liveness-restricted frontier |
| The realized transition matches the proof | physical successor equality check |
| Progress through the proof is finite | monotone witness cursor |
| New admission changes the recovery problem | admission allowed only after commitment clears; unforced Accept resets guard state |

The proposition remains conditional on the certificate's transition model and
on model-faithful execution. The guard does not cover disturbances outside that
contract.

## 4. Correctness evidence and traceability

| Mechanism | Principal implementation | Focused evidence |
|---|---|---|
| Legal macro enumeration and application | `PSLAP/viability.py` | `tests/test_viability.py::LegalRecoveryActionTests` |
| Witness discovery and legal replay | `PSLAP/viability.py::analyze_recoverability` | `test_finds_and_replays_a_deterministic_relocation_witness` |
| Cyclic duplicate handling | same | `test_exhaustive_cycle_search_proves_unsafe` |
| Node/depth budget semantics | same | `test_depth_cutoff_is_unknown_not_unsafe`; `test_node_cutoff_is_unknown_not_unsafe` |
| Primitive horizon | same | `test_primitive_step_horizon_is_part_of_the_certificate` |
| Fail-closed frontier | `PSLAP/viability_candidates.py` | `test_unknown_accept_certificates_are_audited_and_fail_closed`; recovery-successor counterpart |
| Cache contract | same | `test_complete_state_and_budget_cache_is_reused`; `test_cache_identity_separates_budget_and_recovery_contract` |
| Guard activation and Bellman frontier | `viability_graph_hierarchy.py::RecoveryWitnessGuard` | `test_exact_witness_guard_forces_and_bellman_frontier_matches` |
| Cursor persistence | same | `test_retained_multistep_witness_restricts_replayed_next_frontier` |
| No witness replacement/admission interruption | same | `test_active_witness_is_not_replaced_and_masks_admission` |
| Counter/reset rules | same | `test_guard_counter_update_and_reset_rules` |
| One-step robust fail-closed behavior | `vcg_robust_execution_probe.py` | `tests/test_vcg_robust_execution_probe.py` |
| Recursive rank/progress and cycle exclusion | `vcg_recursive_robust_kernel_experiment.py` | `tests/test_vcg_recursive_robust_kernel_experiment.py` |
| Exhaustive live-tree accounting | `run_vcg_exhaustive_live_recovery_tree_5x5.py` | `tests/test_vcg_exhaustive_live_recovery_tree_5x5.py` |

Run the focused audit suite from the repository root:

```bash
.venv/bin/python -m pytest -q \
  tests/test_viability.py \
  tests/test_viability_candidates.py \
  tests/test_viability_graph_hierarchy.py \
  tests/test_vcg_e14_certificate_reuse.py \
  tests/test_vcg_d12_relocation_family.py \
  tests/test_vcg_robust_execution_probe.py \
  tests/test_vcg_recursive_robust_kernel_experiment.py \
  tests/test_vcg_exhaustive_live_recovery_tree_5x5.py \
  tests/test_vcg_conditioned_e08_liveness_audit.py
```

On 22 September 2026 this command completed with **106 passed tests and five
passed subtests**. The E8 trace assertions and some amortization checks require
the authenticated result artifacts named by their experiment directories; they
are not source-only tests in a clean clone. The release described in Section 7
must therefore include those inputs or provide a separately downloadable test
fixture bundle.

## 5. Instrumented guard replay

E8 already supplies the instrumented replay proposed for transition-level
inspection. On `size_8x8_occ_high`, instance seed 95100, frozen model seed 0 and
lambda `.10`, it records at every decision the pre-selection nonprogress
counter, trigger, active-witness state, cursor, learned unrestricted proposal,
and executed guarded candidate.

The episode completed strictly safely in 42 macro decisions. One witness was
committed at the nonprogress limit and governed decisions 4--30: 27 forced
macros, 26 actual proposal overrides, and 16 learned Accept proposals diverted
to recovery. The cursor advanced from 0 through 26, the commitment then
cleared, and admission resumed at decision 31. The derived summary reconstructs
the activation boundary because the base controller prepares the next guarded
frontier inside `observe_outcome`, before the audit wrapper enters the following
`select`.

Across all 45 E13 episodes, 663 of 1,747 decisions (38.0%) were witness-forced.
Those aggregate logs identify forced execution but not unrestricted proposals;
the single E8 replay supplies that causal selection attribution.

## 6. Claim boundary

The evidence supports the following implementation statement:

> Under the declared deterministic strict-macro recovery model, positive native
> certificates contain finite generated recovery witnesses; budget exhaustion
> remains UNKNOWN and is rejected; and the implemented guard retains and
> executes a witness without learned-policy interruption, failing closed on a
> mismatch.

It does not establish recovery under arbitrary real execution errors, future
arrivals excluded by closed admission, or disturbances outside a declared
robust envelope. The reduced recursive games and exhaustive live trees have
their own narrower contracts and should not be silently generalized to the
nominal E-series benchmark.

## 7. Public artifacts

The configured source remote is
<https://github.com/rolcub21/HRL_Management>. A paper release should identify an
immutable commit or tag and archive the manuscript-facing result ledgers and E8
trace under a persistent release or data DOI. Generated `results/` paths are
currently ignored and are not supplied by a clean source clone; SHA-256 values
authenticate a retrieved artifact but do not make it retrievable.

At minimum, the release manifest should map every reported hash to a relative
artifact path, byte size, producing command, source commit, and download URL.
