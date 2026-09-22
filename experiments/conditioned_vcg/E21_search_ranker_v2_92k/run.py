#!/usr/bin/env python3
"""E21-v2: accounting-consistent certified lookahead ranker.

Version 1 exposed a terminal-value pathology: an ASAP recovery witness was
incorrectly treated as a timing policy, while unadmitted jobs received perfect
future reward.  V2 retains depth-two exact-SAFE rollout search but uses the
witness only for remaining service steps.  Potential placement and delivery
rewards are carried for every unfinished job regardless of admission state.
Pure physical out-and-back cycles are pruned at the depth-two boundary.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean
import sys
from time import perf_counter
from typing import Mapping, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import benchmark_viability_critic_priority as benchmark
from experiments.conditioned_vcg.E21_search_ranker_92k import run as v1
from PSLAP.viability import ViabilityStatus
from PSLAP.viability_candidates import (
    EXACT_ONLY_RECOVERY_CERTIFICATION,
    ViabilityCertificateCache,
    enumerate_viability_candidates,
)


PROTOCOL = "vcg_conditioned_e21_v2_consistent_certified_search_ranker_92k_v1"
SCHEMA_VERSION = 1
INSTANCE_SEEDS = v1.INSTANCE_SEEDS
PILOT_INSTANCE_COUNT = 5
SEARCH_RANKER = "depth2_exact_safe_consistent_witness_search"
CONTRACT_NAME = "e21-v2-contract.json"
MANIFEST_NAME = "e21-v2-manifest.json"
REPORT_NAME = "e21-v2-report.json"
TABLE_NAME = "e21-v2-results-table.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/vcg-conditioned-e21-search-ranker-v2-92k"
V1_OUTPUT = v1.DEFAULT_OUTPUT


class E21V2Error(RuntimeError):
    pass


def _canonical(value: Mapping) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: Mapping, field: Optional[str] = None) -> str:
    payload = dict(value)
    if field is not None:
        payload.pop(field, None)
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _with_hash(value: Mapping, field: str) -> dict:
    result = dict(value)
    result[field] = _digest(result)
    return result


def _sha(path: Path) -> str:
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink():
        raise E21V2Error(f"missing regular input: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E21V2Error(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise E21V2Error(f"{label} must contain an object")
    return value


def _verify_hash(value: Mapping, field: str, label: str) -> None:
    if value.get(field) != _digest(value, field):
        raise E21V2Error(f"{label} self-hash mismatch")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _source_hashes() -> dict:
    paths = (
        Path(__file__).resolve(),
        Path(v1.__file__).resolve(),
        PROJECT_ROOT / "benchmark_viability_critic_priority.py",
        PROJECT_ROOT / "PSLAP/viability.py",
        PROJECT_ROOT / "PSLAP/viability_candidates.py",
        PROJECT_ROOT / "train_viability_graph_smdp.py",
        PROJECT_ROOT / "viability_graph_hierarchy.py",
        PROJECT_ROOT / "viability_graph_preference_conditioned.py",
        PROJECT_ROOT / "vcg_objective_audit.py",
    )
    return {str(path.relative_to(PROJECT_ROOT)): _sha(path) for path in paths}


def _v1_diagnostic() -> dict:
    report = _load_json(V1_OUTPUT / v1.REPORT_NAME, "completed E21-v1 report")
    _verify_hash(report, "report_sha256", "completed E21-v1 report")
    if (
        report.get("protocol") != v1.PROTOCOL
        or report.get("status") != "complete"
        or report.get("new_search_rows") != len(INSTANCE_SEEDS)
    ):
        raise E21V2Error("E21-v1 diagnostic is incomplete")
    return report


def _parent_bundle() -> dict:
    controls = v1._control_bundle()
    diagnostic = _v1_diagnostic()
    return {**controls, "v1_report": diagnostic}


def _contract() -> dict:
    parent = _parent_bundle()
    arm = v1._frozen_carrier()
    return _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "scientific_question": (
                "does_learning_improve_ranking_over_an_accounting_consistent_"
                "deterministic_short_horizon_search_on_the_same_SAFE_frontier"
            ),
            "post_hoc": True,
            "derived_after_e21_v1_diagnostic": True,
            "instance_seeds": list(INSTANCE_SEEDS),
            "new_arm": SEARCH_RANKER,
            "new_rollouts": len(INSTANCE_SEEDS),
            "pilot_rollouts": PILOT_INSTANCE_COUNT,
            "carrier_model_seed": v1.CARRIER_MODEL_SEED,
            "carrier_checkpoint_sha256": arm.checkpoint_sha256,
            "carrier_policy_digest": arm.deployment_policy_digest,
            "lookahead_macros": v1.LOOKAHEAD_MACROS,
            "lookahead_branching": "all_exact_SAFE_candidates_at_depth_two",
            "terminal_value": {
                "delivery_potential_per_undelivered_job": float(
                    v1.pilot.v11.FROZEN_OBJECTIVE_SPEC.dense_b
                ),
                "placement_potential_per_unplaced_job": v1.PLACEMENT_REWARD,
                "admitted_service_cost": (
                    "step_cost_times_validated_witness_primitive_steps"
                ),
                "unplaced_service_cost": (
                    "step_cost_times_geometry_only_optimistic_lower_bound"
                ),
                "witness_delivery_timing_reward": False,
            },
            "cycle_rule": (
                "prune_depth2_branch_returning_to_root_physical_signature_"
                "while_ignoring_clocks_and_tabu_the_immediate_inverse_"
                "relocation_at_the_next_real_decision_and_prune_"
                "reconfiguration_of_any_exactly_directly_deliverable_block"
            ),
            "cycle_rule_rationale": (
                "prevent_relocation_from_serving_as_a_liveness_bypassing_wait"
            ),
            "information_contract": (
                "job_count_and_root_observed_state_only; future_arrival_times_"
                "masked_before_every_simulated_branch"
            ),
            "objective": v1.pilot.v11.FROZEN_OBJECTIVE_SPEC.to_dict(),
            "step_cost": v1.STEP_COST,
            "placement_reward": v1.PLACEMENT_REWARD,
            "reused_controls": {
                "heuristic_safe": "E4 nonlearned heuristic-safe rows",
                "frozen_qop": "E4 three-model-seed frozen-Qop rows",
                "conditioned_lambda_0.10": (
                    "E4 three-model-seed conditioned VCG lambda=.10 rows"
                ),
            },
            "controls_are_independent_replications": False,
            "training_or_checkpoint_selection": False,
            "held_fixed": [
                "frozen_E4_EpisodeInstances",
                "exact_SAFE_candidate_source",
                "recovery_witness_liveness_guard",
                "normalized_mode_logmeanexp_aggregation",
                "candidate_and_mode_tie_breaking",
                "macro_executor",
                "dense_operational_objective",
            ],
            "parent_e4_contract_sha256": parent["contract"]["contract_sha256"],
            "parent_e4_manifest_sha256": parent["manifest"]["manifest_sha256"],
            "parent_e4_report_sha256": parent["report"]["report_sha256"],
            "parent_e21_v1_report_sha256": parent["v1_report"]["report_sha256"],
            "source_sha256": _source_hashes(),
        },
        "contract_sha256",
    )


def _manifest(contract: Mapping, parent_manifest: Mapping) -> dict:
    return _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "parent_e4_manifest_sha256": parent_manifest["manifest_sha256"],
            "instance_count": len(parent_manifest["instances"]),
            "instances": [dict(item) for item in parent_manifest["instances"]],
            "instance_storage": "byte_frozen_parent_E4_files",
        },
        "manifest_sha256",
    )


def prepare(output: Path) -> dict:
    output = output.resolve()
    parent = _parent_bundle()
    contract = _contract()
    manifest = _manifest(contract, parent["manifest"])
    contract_path = output / CONTRACT_NAME
    manifest_path = output / MANIFEST_NAME
    if contract_path.is_file():
        observed = _load_json(contract_path, "E21-v2 contract")
        _verify_hash(observed, "contract_sha256", "E21-v2 contract")
        if observed != contract:
            raise E21V2Error("E21-v2 contract, sources, or frozen inputs changed")
    else:
        if output.exists() and any(output.iterdir()):
            raise E21V2Error("nonempty E21-v2 output has no contract")
        _atomic_json(contract_path, contract)
    if manifest_path.is_file():
        observed = _load_json(manifest_path, "E21-v2 manifest")
        _verify_hash(observed, "manifest_sha256", "E21-v2 manifest")
        if observed != manifest:
            raise E21V2Error("E21-v2 manifest changed")
    else:
        _atomic_json(manifest_path, manifest)
    return {
        "status": "prepared",
        "training_runs": 0,
        "new_pilot_rollouts": PILOT_INSTANCE_COUNT,
        "new_confirmation_rollouts": len(INSTANCE_SEEDS),
        "reused_control_rows": sum(
            len(rows) for rows in parent["controls"].values()
        ),
        "contract": str(contract_path),
        "manifest": str(manifest_path),
    }


def authenticate(output: Path):
    parent = _parent_bundle()
    contract = _load_json(output / CONTRACT_NAME, "E21-v2 contract")
    manifest = _load_json(output / MANIFEST_NAME, "E21-v2 manifest")
    _verify_hash(contract, "contract_sha256", "E21-v2 contract")
    _verify_hash(manifest, "manifest_sha256", "E21-v2 manifest")
    if contract != _contract():
        raise E21V2Error("E21-v2 contract, sources, or frozen inputs changed")
    if manifest != _manifest(contract, parent["manifest"]):
        raise E21V2Error("E21-v2 manifest changed")
    return contract, manifest, parent


def _physical_signature(env) -> tuple:
    """Clock-free physical state used only for within-search cycle pruning."""

    blocks = tuple(
        sorted(
            (
                str(block.label),
                None if block.position is None else tuple(block.position),
                (
                    None
                    if block.storage_location is None
                    else tuple(block.storage_location)
                ),
                bool(block.carrying),
                bool(block.stored),
                bool(block.delivered),
            )
            for block in env.blocks
        )
    )
    return tuple(env.current_state), blocks


def _minimum_unplaced_service_steps(env, candidate) -> int:
    """Geometry-only lower bound: accept then eventually deliver one job."""

    pickup = tuple(env.pickup_cell)
    storage = tuple(candidate.successor_state.storage_cells)
    exits = tuple(candidate.successor_state.exits)
    if not storage or not exits:
        raise E21V2Error("terminal service bound needs storage and exit cells")

    def distance(left, right):
        return abs(left[0] - right[0]) + abs(left[1] - right[1])

    # Two PICKUP/PUTDOWN pairs add four primitive actions.  Agent approach,
    # obstacles, queueing, and due-time waiting are deliberately omitted, so
    # this remains an optimistic lower bound rather than a hidden policy.
    return min(
        distance(pickup, cell) + distance(cell, exit_cell) + 4
        for cell in storage
        for exit_cell in exits
    )


def _consistent_terminal_value(env, candidate) -> float:
    certificate = candidate.certificate
    if certificate.status is not ViabilityStatus.SAFE:
        raise E21V2Error("terminal value requires a SAFE certificate")
    witness_steps = certificate.witness_primitive_steps
    if witness_steps is None:
        if certificate.witness:
            raise E21V2Error("SAFE witness lacks primitive-step accounting")
        witness_steps = 0
    summed = sum(int(action.steps) for action in certificate.witness)
    if int(witness_steps) != summed:
        raise E21V2Error("completion witness step accounting changed")

    undelivered = sum(not block.delivered for block in env.blocks)
    unplaced = sum(
        not block.stored and not block.delivered for block in env.blocks
    )
    lower_bound = _minimum_unplaced_service_steps(env, candidate)
    potential = (
        float(v1.pilot.v11.FROZEN_OBJECTIVE_SPEC.dense_b) * undelivered
        + v1.PLACEMENT_REWARD * unplaced
    )
    remaining_steps = int(witness_steps) + unplaced * lower_bound
    return float(potential + v1.STEP_COST * remaining_steps)


def _inverse_relocation_signature(candidate):
    if candidate.action_type is not v1.ViabilityActionType.RECONFIGURE:
        return None
    return (
        str(candidate.target_label),
        tuple(candidate.destination),
        tuple(candidate.source),
    )


def _is_inverse_relocation(candidate, signature) -> bool:
    if signature is None:
        return False
    if candidate.action_type is not v1.ViabilityActionType.RECONFIGURE:
        return False
    return (
        str(candidate.target_label),
        tuple(candidate.source),
        tuple(candidate.destination),
    ) == tuple(signature)


def _directly_deliverable_labels(candidates) -> frozenset[str]:
    return frozenset(
        str(candidate.target_label)
        for candidate in candidates
        if candidate.action_type is v1.ViabilityActionType.DELIVER
    )


def _is_dominated_self_reconfiguration(candidate, deliverable_labels) -> bool:
    return bool(
        candidate.action_type is v1.ViabilityActionType.RECONFIGURE
        and str(candidate.target_label) in deliverable_labels
    )


class ConsistentDepthTwoSearchAgent(v1.CertifiedDepthTwoSearchAgent):
    """V2 search with consistent reward potential and physical-cycle pruning."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inverse_relocation_tabu = None

    def reset_episode_state(self):
        self.inverse_relocation_tabu = None
        return super().reset_episode_state()

    on_episode_reset = reset_episode_state

    def _scores(self, snapshot, prepared) -> torch.Tensor:
        candidates = tuple(
            snapshot.candidates[index] for index in prepared.source_indices
        )
        live_env = self._live_env(candidates)
        deliverable = _directly_deliverable_labels(candidates)
        inverse_indices = {
            index
            for index, candidate in enumerate(candidates)
            if _is_inverse_relocation(candidate, self.inverse_relocation_tabu)
        }
        dominated_indices = {
            index
            for index, candidate in enumerate(candidates)
            if _is_dominated_self_reconfiguration(candidate, deliverable)
        }
        tabu = tuple(sorted(inverse_indices | dominated_indices))
        # If the inverse is literally the only admissible action, retain it;
        # the tabu is cycle control, never a new source of an empty frontier.
        active_tabu = set(tabu if len(tabu) < len(candidates) else ())
        self.diagnostics["online_inverse_candidates_pruned"] += len(
            active_tabu & inverse_indices
        )
        self.diagnostics["online_direct_delivery_reconfigs_pruned"] += len(
            active_tabu & dominated_indices
        )
        values = [
            -1.0e9
            if index in active_tabu
            else self._score_root(live_env, candidate)
            for index, candidate in enumerate(candidates)
        ]
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def _score_root(self, live_env, candidate) -> float:
        root_time = int(live_env.time_steps)
        initial_signature = _physical_signature(live_env)
        root_env, root_candidate = deepcopy((live_env, candidate))
        self.diagnostics["masked_future_arrivals"] += v1._mask_unobserved_arrivals(
            root_env, root_time
        )
        root_execution = self._simulate_macro(root_env, root_candidate)
        self.diagnostics["root_macro_simulations"] += 1
        root_reward = v1._dense_macro_reward(root_execution)
        if root_execution.env_terminal:
            return root_reward

        child_defer = v1._next_defer_count(
            self.consecutive_defer, root_candidate
        )
        frontier_started = perf_counter()
        child_snapshot = enumerate_viability_candidates(
            root_env,
            consecutive_defer_decisions=child_defer,
            search_config=self.search_config,
            liveness_rule=self.liveness_rule,
            cache=self.planning_cache,
            state_prioritizer=None,
            recovery_certification_strategy=EXACT_ONLY_RECOVERY_CERTIFICATION,
        )
        self.diagnostics["simulated_frontier_seconds"] += (
            perf_counter() - frontier_started
        )
        self.diagnostics["simulated_child_frontiers"] += 1
        self.diagnostics["simulated_exact_analysis_seconds"] += float(
            child_snapshot.audit.exact_analysis_seconds
        )
        if not child_snapshot.candidates:
            self.diagnostics["empty_simulated_child_frontiers"] += 1
            return root_reward + _consistent_terminal_value(
                root_env, root_candidate
            )

        best = -math.inf
        child_deliverable = _directly_deliverable_labels(
            child_snapshot.candidates
        )
        for child in child_snapshot.candidates:
            if _is_dominated_self_reconfiguration(child, child_deliverable):
                self.diagnostics[
                    "simulated_direct_delivery_reconfigs_pruned"
                ] += 1
                continue
            child_env, child_candidate = deepcopy((root_env, child))
            child_execution = self._simulate_macro(child_env, child_candidate)
            self.diagnostics["child_macro_simulations"] += 1
            if _physical_signature(child_env) == initial_signature:
                self.diagnostics["physical_cycles_pruned"] += 1
                continue
            score = (
                v1._dense_macro_reward(child_execution)
                + _consistent_terminal_value(child_env, child_candidate)
            )
            best = max(best, score)
        if not math.isfinite(best):
            self.diagnostics["all_children_cycle_pruned"] += 1
            best = _consistent_terminal_value(root_env, root_candidate)
        return float(root_reward + best)

    def observe_outcome(self, decision, *, next_snapshot, done):
        self.inverse_relocation_tabu = _inverse_relocation_signature(
            decision.candidate
        )
        return super().observe_outcome(
            decision, next_snapshot=next_snapshot, done=done
        )


def _ledger_path(output: Path, instance_seed: int) -> Path:
    return output / "run-ledger" / f"instance-{int(instance_seed)}.json"


def _v2_row(raw, compact, identity, diagnostics) -> dict:
    row = v1._row(raw, compact, identity, diagnostics)
    row.update(
        {
            "protocol": PROTOCOL,
            "ranking_signal": SEARCH_RANKER,
            "search_version": "v2_consistent_terminal",
        }
    )
    return row


def _failed_row(error: Exception, identity: Mapping) -> dict:
    row = v1._failed_row(error, identity)
    row.update(
        {
            "protocol": PROTOCOL,
            "ranking_signal": SEARCH_RANKER,
            "search_version": "v2_consistent_terminal",
        }
    )
    return row


def run(output: Path, *, instance_limit: Optional[int], device_name: str) -> dict:
    contract, manifest, _parent = authenticate(output)
    device = v1.pilot._device(device_name)
    if device.type != "cpu":
        raise E21V2Error("E21-v2 is frozen to CPU execution")
    torch.set_num_threads(1)
    arm = v1._frozen_carrier()
    search_config = benchmark._search_config(arm.payload)
    liveness_rule = benchmark._liveness_rule(arm.payload)
    records = list(manifest["instances"])
    if instance_limit is not None:
        if instance_limit < 1 or instance_limit > len(records):
            raise E21V2Error("invalid E21-v2 instance limit")
        records = records[:instance_limit]

    complete = safe = 0
    for record in records:
        identity = v1._identity(record)
        path = _ledger_path(output, identity["instance_seed"])
        if path.is_file():
            ledger = _load_json(path, "E21-v2 rollout ledger")
            _verify_hash(ledger, "ledger_sha256", "E21-v2 rollout ledger")
            if (
                ledger.get("contract_sha256") != contract["contract_sha256"]
                or ledger.get("manifest_sha256") != manifest["manifest_sha256"]
            ):
                raise E21V2Error("E21-v2 rollout ledger binding changed")
            row = ledger["run"]
        else:
            instance = v1.e4._load_instance(v1.E4_OUTPUT, record)
            holders = []

            def factory(base):
                agent = ConsistentDepthTwoSearchAgent(
                    base,
                    search_config=search_config,
                    liveness_rule=liveness_rule,
                )
                holders.append(agent)
                return agent

            try:
                raw = v1.pilot._run_raw(
                    arm, instance, device=device, wrapper_factory=factory
                )
                compact = v1.pilot._compact_row(raw, instance)
                if len(holders) != 1:
                    raise E21V2Error("E21-v2 wrapper construction changed")
                row = _v2_row(
                    raw,
                    compact,
                    identity,
                    holders[0].diagnostic_record(),
                )
            except Exception as error:  # preserve prospective failures
                row = _failed_row(error, identity)
            ledger = _with_hash(
                {
                    "schema_version": SCHEMA_VERSION,
                    "protocol": PROTOCOL,
                    "contract_sha256": contract["contract_sha256"],
                    "manifest_sha256": manifest["manifest_sha256"],
                    "run": row,
                },
                "ledger_sha256",
            )
            _atomic_json(path, ledger)
        complete += 1
        safe += int(row["strict_safe_complete"])
        print(
            f"E21-v2 {complete}/{len(records)} | "
            f"instance={identity['instance_seed']} | "
            f"safe={int(row['strict_safe_complete'])}",
            flush=True,
        )
    return {
        "status": "complete",
        "rows": complete,
        "strict_safe_complete_rows": safe,
        "training_runs": 0,
    }


def _new_rows(output: Path) -> list[dict]:
    rows = []
    root = output / "run-ledger"
    if not root.is_dir():
        return rows
    for path in sorted(root.glob("instance-*.json")):
        ledger = _load_json(path, "E21-v2 rollout ledger")
        _verify_hash(ledger, "ledger_sha256", "E21-v2 rollout ledger")
        rows.append(ledger["run"])
    return rows


def _table(report: Mapping) -> str:
    labels = {
        SEARCH_RANKER: "Certified depth-2 search v2",
        "heuristic_safe": "Heuristic-safe (reused E4)",
        "frozen_qop": "Frozen Qop (reused E4)",
        "conditioned_lambda_0.10": "Conditioned VCG lambda=.10 (reused E4)",
    }
    lines = [
        "# E21-v2 accounting-consistent search comparison",
        "",
        "| Ranker | Strict complete | Return | MAE | Within ±20 | Steps | Rehandles/100 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in (
        SEARCH_RANKER,
        "heuristic_safe",
        "frozen_qop",
        "conditioned_lambda_0.10",
    ):
        item = report["aggregate"][key]

        def value(name: str, digits: int = 2) -> str:
            observed = item[name]
            return "—" if observed is None else f"{observed:.{digits}f}"

        lines.append(
            f"| {labels[key]} | {item['strict_safe_complete']}/{item['rows']} | "
            f"{value('dense_return')} | {value('mean_absolute_error')} | "
            f"{value('within_target_window_rate', 3)} | {value('steps')} | "
            f"{value('physical_rehandles_per_100_required_deliveries')} |"
        )
    lines.extend(
        [
            "",
            "E21-v2 is post-hoc and was specified after diagnosing E21-v1.",
            "E4 controls are authenticated reanalyses, not new replications.",
            "The witness contributes service steps only; its ASAP delivery times are not treated as a timing policy.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(output: Path, *, allow_partial: bool) -> dict:
    contract, manifest, parent = authenticate(output)
    search_rows = _new_rows(output)
    if not allow_partial and len(search_rows) != len(INSTANCE_SEEDS):
        raise E21V2Error(
            f"E21-v2 grid incomplete: {len(search_rows)}/{len(INSTANCE_SEEDS)}"
        )
    aggregate = {SEARCH_RANKER: v1._summarize(search_rows)}
    aggregate.update(
        {name: v1._summarize(rows) for name, rows in parent["controls"].items()}
    )
    paired = {
        name: v1._paired_differences(
            search_rows,
            rows,
            average_model_seeds=name != "heuristic_safe",
        )
        for name, rows in parent["controls"].items()
    }
    planning = [
        row.get("planner_diagnostics", {})
        for row in search_rows
        if row.get("planner_diagnostics")
    ]
    report = _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": (
                "complete"
                if len(search_rows) == len(INSTANCE_SEEDS)
                else "partial"
            ),
            "paper_evidence": False,
            "post_hoc": True,
            "derived_after_e21_v1_diagnostic": True,
            "contract_sha256": contract["contract_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "new_search_rows": len(search_rows),
            "expected_new_search_rows": len(INSTANCE_SEEDS),
            "training_runs": 0,
            "reused_controls_are_independent_replications": False,
            "aggregate": aggregate,
            "paired_search_minus_control": paired,
            "planning_cost": {
                "episodes": len(planning),
                "total_seconds": sum(
                    float(item.get("planning_seconds", 0.0)) for item in planning
                ),
                "mean_seconds_per_episode": (
                    fmean(float(item.get("planning_seconds", 0.0)) for item in planning)
                    if planning
                    else None
                ),
                "root_macro_simulations": sum(
                    int(item.get("root_macro_simulations", 0)) for item in planning
                ),
                "child_macro_simulations": sum(
                    int(item.get("child_macro_simulations", 0)) for item in planning
                ),
                "physical_cycles_pruned": sum(
                    int(item.get("physical_cycles_pruned", 0)) for item in planning
                ),
            },
            "interpretation_limits": [
                "post_hoc_test_on_an_already_opened_panel",
                "terminal_rule_was_designed_after_diagnosing_E21_v1",
                "witness_length_is_a_constructive_upper_bound_not_shortest_rank",
                "E4_controls_are_reanalyses_not_new_replications",
            ],
        },
        "report_sha256",
    )
    _atomic_json(output / REPORT_NAME, report)
    _atomic_text(output / TABLE_NAME, _table(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--instance-limit", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "run":
        result = run(
            output,
            instance_limit=args.instance_limit,
            device_name=args.device,
        )
    else:
        result = analyze(output, allow_partial=args.allow_partial)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
