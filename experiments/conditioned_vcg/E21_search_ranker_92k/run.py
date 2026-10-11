#!/usr/bin/env python3
"""E21: deterministic certified lookahead versus learned VCG ranking.

The only new arm is a non-learned depth-two model-predictive ranker.  Every
simulated macro is drawn from an exact-SAFE frontier.  At the depth limit, a
validated completion witness supplies a deterministic terminal value.  The
experiment reuses the frozen E4 92k EpisodeInstances and E4 learned controls;
those controls are not rerun or presented as independent replications.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean, stdev
import sys
from time import perf_counter
from typing import Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from scipy.stats import t as student_t

import benchmark_viability_critic_priority as benchmark
from experiments.conditioned_vcg.E04_safe_frontier_ranking_92k import run as e4
import run_vcg_conditioned_final_comparison_90k as final90
import run_vcg_final86_four_method as final86
import run_vcg_v11_nested_handling_pilot as pilot
from PSLAP.viability import RecoveryActionKind, ViabilityStatus
from PSLAP.viability_candidates import (
    EXACT_ONLY_RECOVERY_CERTIFICATION,
    ViabilityActionType,
    ViabilityCertificateCache,
    ViabilityMode,
    enumerate_viability_candidates,
)
from train_viability_graph_smdp import execute_certified_macro
from vcg_objective_audit import delivery_reward
from viability_graph_hierarchy import ID_TO_MODE, ViabilityGraphDecision
from viability_graph_preference_conditioned import select_hierarchical_index


PROTOCOL = "vcg_conditioned_e21_certified_depth2_search_ranker_92k_v1"
SCHEMA_VERSION = 1
INSTANCE_SEEDS = e4.INSTANCE_SEEDS
PILOT_INSTANCE_COUNT = 5
CARRIER_MODEL_SEED = 0
LOOKAHEAD_MACROS = 2
STEP_COST = -0.09
PLACEMENT_REWARD = 5.0
MASKED_ARRIVAL_BASE = 1_000_000_000
SEARCH_RANKER = "depth2_exact_safe_witness_search"

CONTRACT_NAME = "e21-contract.json"
MANIFEST_NAME = "e21-manifest.json"
REPORT_NAME = "e21-report.json"
TABLE_NAME = "e21-results-table.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/vcg-conditioned-e21-search-ranker-92k"
E4_OUTPUT = e4.DEFAULT_OUTPUT

CONTROL_GROUPS = {
    "heuristic_safe": (e4.HEURISTIC_SAFE, None),
    "frozen_qop": (e4.QOP_SAFE, None),
    "conditioned_lambda_0.10": (e4.CONDITIONED_SAFE, 0.10),
}
METRICS = (
    "dense_return",
    "mean_absolute_error",
    "within_target_window_rate",
    "steps",
    "physical_rehandles_per_100_required_deliveries",
)


class E21Error(RuntimeError):
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
        raise E21Error(f"missing regular input: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E21Error(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise E21Error(f"{label} must contain an object")
    return value


def _verify_hash(value: Mapping, field: str, label: str) -> None:
    if value.get(field) != _digest(value, field):
        raise E21Error(f"{label} self-hash mismatch")


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
        PROJECT_ROOT / "benchmark_viability_critic_priority.py",
        PROJECT_ROOT / "PSLAP/viability.py",
        PROJECT_ROOT / "PSLAP/viability_candidates.py",
        PROJECT_ROOT / "train_viability_graph_smdp.py",
        PROJECT_ROOT / "viability_graph_hierarchy.py",
        PROJECT_ROOT / "viability_graph_preference_conditioned.py",
        PROJECT_ROOT / "vcg_objective_audit.py",
    )
    return {str(path.relative_to(PROJECT_ROOT)): _sha(path) for path in paths}


def _frozen_e4_parent() -> tuple[dict, dict, dict]:
    """Authenticate E4 by saved identities, without rebuilding old sources."""

    contract = _load_json(E4_OUTPUT / e4.CONTRACT_NAME, "frozen E4 contract")
    manifest = _load_json(E4_OUTPUT / e4.MANIFEST_NAME, "frozen E4 manifest")
    report = _load_json(E4_OUTPUT / e4.REPORT_NAME, "frozen E4 report")
    _verify_hash(contract, "contract_sha256", "frozen E4 contract")
    _verify_hash(manifest, "manifest_sha256", "frozen E4 manifest")
    _verify_hash(report, "report_sha256", "frozen E4 report")
    records = manifest.get("instances")
    if (
        contract.get("protocol") != e4.PROTOCOL
        or manifest.get("protocol") != e4.PROTOCOL
        or manifest.get("contract_sha256") != contract.get("contract_sha256")
        or report.get("contract_sha256") != contract.get("contract_sha256")
        or report.get("manifest_sha256") != manifest.get("manifest_sha256")
        or report.get("status") != "complete"
        or not isinstance(records, list)
        or [int(item.get("seed", -1)) for item in records]
        != list(INSTANCE_SEEDS)
    ):
        raise E21Error("frozen E4 parent identity is incomplete or inconsistent")
    for record in records:
        path = E4_OUTPUT / str(record["relative_path"])
        if _sha(path) != record["raw_sha256"]:
            raise E21Error("frozen E4 EpisodeInstance bytes changed")
    return contract, manifest, report


def _frozen_carrier():
    arms, _records = final86._authenticate_v11()
    by_seed = {int(arm.model_seed): arm for arm in arms}
    if CARRIER_MODEL_SEED not in by_seed:
        raise E21Error("frozen VCG 1.1 seed-0 carrier is unavailable")
    return by_seed[CARRIER_MODEL_SEED]


def _control_bundle() -> dict:
    contract, manifest, report = _frozen_e4_parent()
    controls = {name: [] for name in CONTROL_GROUPS}
    files = []
    for path in sorted((E4_OUTPUT / "run-ledger").glob("**/instance-*.json")):
        ledger = _load_json(path, "frozen E4 control ledger")
        _verify_hash(ledger, "ledger_sha256", "frozen E4 control ledger")
        if (
            ledger.get("contract_sha256") != contract["contract_sha256"]
            or ledger.get("manifest_sha256") != manifest["manifest_sha256"]
        ):
            raise E21Error("frozen E4 control ledger binding changed")
        row = ledger.get("run", {})
        for name, (ranking, preference) in CONTROL_GROUPS.items():
            if row.get("ranking_signal") != ranking:
                continue
            if preference is not None and not math.isclose(
                float(row.get("preference_lambda", -1.0)), preference
            ):
                continue
            controls[name].append(row)
            files.append(
                {
                    "relative_path": str(path.relative_to(PROJECT_ROOT)),
                    "raw_sha256": _sha(path),
                }
            )
    expected = {
        "heuristic_safe": len(INSTANCE_SEEDS),
        "frozen_qop": len(e4.MODEL_SEEDS) * len(INSTANCE_SEEDS),
        "conditioned_lambda_0.10": len(e4.MODEL_SEEDS) * len(INSTANCE_SEEDS),
    }
    observed = {name: len(rows) for name, rows in controls.items()}
    if observed != expected:
        raise E21Error(f"frozen E4 controls incomplete: {observed} != {expected}")
    return {
        "contract": contract,
        "manifest": manifest,
        "report": report,
        "controls": controls,
        "files": files,
    }


def _contract() -> dict:
    parent = _control_bundle()
    arm = _frozen_carrier()
    return _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "scientific_question": (
                "does_learning_improve_ranking_over_a_deterministic_short_"
                "horizon_search_ranker_on_the_same_exact_safe_frontier"
            ),
            "post_hoc": True,
            "instance_seeds": list(INSTANCE_SEEDS),
            "new_arm": SEARCH_RANKER,
            "new_rollouts": len(INSTANCE_SEEDS),
            "pilot_rollouts": PILOT_INSTANCE_COUNT,
            "carrier_model_seed": CARRIER_MODEL_SEED,
            "carrier_checkpoint_sha256": arm.checkpoint_sha256,
            "carrier_policy_digest": arm.deployment_policy_digest,
            "lookahead_macros": LOOKAHEAD_MACROS,
            "lookahead_branching": "all_exact_SAFE_candidates_at_depth_two",
            "terminal_value": (
                "dense_value_of_validated_goal_directed_completion_witness_"
                "plus_optimistic_constant_for_currently_observed_unadmitted_work"
            ),
            "witness_interpretation": (
                "constructive_completion_upper_bound_not_shortest_recovery_rank"
            ),
            "information_contract": (
                "only_root_observed_arrivals_visible; future_arrival_times_masked_"
                "before_every_simulated_branch"
            ),
            "objective": pilot.v11.FROZEN_OBJECTIVE_SPEC.to_dict(),
            "step_cost": STEP_COST,
            "placement_reward": PLACEMENT_REWARD,
            "reused_controls": {
                "heuristic_safe": "E4 nonlearned heuristic-safe rows",
                "frozen_qop": "E4 three-model-seed frozen-Qop rows",
                "conditioned_lambda_0.10": (
                    "E4 three-model-seed conditioned VCG lambda=.10 rows"
                ),
            },
            "controls_are_independent_replications": False,
            "held_fixed": [
                "frozen_E4_EpisodeInstances",
                "exact_SAFE_candidate_source",
                "recovery_witness_liveness_guard",
                "normalized_mode_logmeanexp_aggregation",
                "candidate_and_mode_tie_breaking",
                "macro_executor",
                "dense_operational_objective",
            ],
            "training_or_checkpoint_selection": False,
            "parent_e4_contract_sha256": parent["contract"]["contract_sha256"],
            "parent_e4_manifest_sha256": parent["manifest"]["manifest_sha256"],
            "parent_e4_report_sha256": parent["report"]["report_sha256"],
            "parent_control_set_sha256": hashlib.sha256(
                json.dumps(
                    parent["files"], sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "source_sha256": _source_hashes(),
        },
        "contract_sha256",
    )


def _manifest(contract: Mapping, parent_manifest: Mapping) -> dict:
    records = [dict(item) for item in parent_manifest["instances"]]
    return _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "parent_e4_manifest_sha256": parent_manifest["manifest_sha256"],
            "instance_count": len(records),
            "instances": records,
            "instance_storage": "byte_frozen_parent_E4_files",
            "all_new_rollouts_load_exact_parent_instances": True,
        },
        "manifest_sha256",
    )


def prepare(output: Path) -> dict:
    output = output.resolve()
    parent = _control_bundle()
    contract = _contract()
    manifest = _manifest(contract, parent["manifest"])
    contract_path = output / CONTRACT_NAME
    manifest_path = output / MANIFEST_NAME
    if contract_path.is_file():
        observed = _load_json(contract_path, "E21 contract")
        _verify_hash(observed, "contract_sha256", "E21 contract")
        if observed != contract:
            raise E21Error("E21 contract, sources, or frozen inputs changed")
    else:
        if output.exists() and any(output.iterdir()):
            raise E21Error("nonempty E21 output has no contract")
        _atomic_json(contract_path, contract)
    if manifest_path.is_file():
        observed = _load_json(manifest_path, "E21 manifest")
        _verify_hash(observed, "manifest_sha256", "E21 manifest")
        if observed != manifest:
            raise E21Error("E21 instance manifest changed")
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
    parent = _control_bundle()
    contract = _load_json(output / CONTRACT_NAME, "E21 contract")
    manifest = _load_json(output / MANIFEST_NAME, "E21 manifest")
    _verify_hash(contract, "contract_sha256", "E21 contract")
    _verify_hash(manifest, "manifest_sha256", "E21 manifest")
    expected = _contract()
    if contract != expected:
        raise E21Error("E21 contract, sources, or frozen inputs changed")
    if manifest != _manifest(contract, parent["manifest"]):
        raise E21Error("E21 instance manifest changed")
    return contract, manifest, parent


def _mask_unobserved_arrivals(env, root_time: int) -> int:
    """Hide schedule information not observable at the root decision."""

    masked = 0
    for index, block in enumerate(env.blocks):
        arrival = getattr(block, "arrival_step", None)
        unobserved = bool(
            arrival is not None
            and int(arrival) > int(root_time)
            and block.position is None
            and not block.carrying
            and not block.stored
            and not block.delivered
        )
        if unobserved:
            block.arrival_step = MASKED_ARRIVAL_BASE + index
            masked += 1
    return masked


def _visible_unadmitted_count(env) -> int:
    return sum(
        1
        for block in env.blocks
        if (
            block.position is not None
            and not block.carrying
            and not block.stored
            and not block.delivered
        )
    )


def _dense_macro_reward(execution) -> float:
    value = STEP_COST * int(execution.duration)
    if (
        execution.action_type == ViabilityActionType.ACCEPT.value
        and execution.option_success
    ):
        value += PLACEMENT_REWARD
    value += sum(
        delivery_reward(error, pilot.v11.FROZEN_OBJECTIVE_SPEC)
        for error in execution.delivery_deviations
    )
    return float(value)


def _witness_terminal_value(env, candidate) -> float:
    """Value one valid completion witness; no shortest-path claim is made."""

    certificate = candidate.certificate
    if certificate.status is not ViabilityStatus.SAFE:
        raise E21Error("terminal heuristic requires a SAFE certificate")
    elapsed = 0
    value = 0.0
    for action in certificate.witness:
        elapsed += int(action.steps)
        if action.kind is RecoveryActionKind.DELIVERY:
            block = candidate.successor_state.block(action.block_label)
            if block is None:
                raise E21Error("completion witness delivery has no successor block")
            predicted_error = float(elapsed) - float(block.remaining_time)
            value += delivery_reward(
                predicted_error, pilot.v11.FROZEN_OBJECTIVE_SPEC
            )
    if certificate.witness_primitive_steps is None:
        if certificate.witness:
            raise E21Error("SAFE witness is missing its primitive-step count")
        witness_steps = 0
    else:
        witness_steps = int(certificate.witness_primitive_steps)
    if witness_steps != elapsed:
        raise E21Error("completion witness primitive-step accounting changed")
    value += STEP_COST * witness_steps
    observed_external_value = (
        PLACEMENT_REWARD + float(pilot.v11.FROZEN_OBJECTIVE_SPEC.dense_b)
    )
    value += observed_external_value * _visible_unadmitted_count(env)
    return float(value)


def _next_defer_count(current: int, candidate) -> int:
    if candidate.action_type is not ViabilityActionType.DEFER:
        return 0
    outcome = getattr(candidate.option, "last_outcome", None)
    observed_event = bool(
        isinstance(outcome, Mapping) and outcome.get("reason") == "observed_event"
    )
    return 0 if observed_event else int(current) + 1


class CertifiedDepthTwoSearchAgent:
    """Non-learned exhaustive depth-two planner over exact-SAFE macros."""

    def __init__(self, base_agent, *, search_config, liveness_rule):
        self.base_agent = base_agent
        self.config = base_agent.config
        self.device = base_agent.device
        self.search_config = search_config
        self.liveness_rule = liveness_rule
        self.planning_cache = ViabilityCertificateCache()
        self.consecutive_defer = 0
        self.diagnostics = defaultdict(float)
        self.decision_planning_seconds = []

    def reset_episode_state(self):
        self.planning_cache = ViabilityCertificateCache()
        self.consecutive_defer = 0
        self.diagnostics.clear()
        self.decision_planning_seconds.clear()
        return self.base_agent.reset_episode_state()

    on_episode_reset = reset_episode_state

    @staticmethod
    def _live_env(candidates) -> object:
        envs = {id(candidate.option.env): candidate.option.env for candidate in candidates}
        if len(envs) != 1:
            raise E21Error("candidate frontier is not bound to one live environment")
        return next(iter(envs.values()))

    def _simulate_macro(self, env, candidate):
        execution = execute_certified_macro(
            env,
            candidate,
            gamma=float(self.config.gamma),
            remaining_steps=pilot.v11.MAX_STEPS,
            evaluation=True,
        )
        if (
            not execution.option_success
            or execution.truncated
            or execution.duration <= 0
            or execution.illegal_drops
        ):
            raise E21Error(
                "exact-SAFE simulated macro failed: "
                f"{candidate.key}: {execution.failure_reason}"
            )
        return execution

    def _score_root(self, live_env, candidate) -> float:
        root_time = int(live_env.time_steps)
        root_env, root_candidate = deepcopy((live_env, candidate))
        self.diagnostics["masked_future_arrivals"] += _mask_unobserved_arrivals(
            root_env, root_time
        )
        root_execution = self._simulate_macro(root_env, root_candidate)
        self.diagnostics["root_macro_simulations"] += 1
        root_reward = _dense_macro_reward(root_execution)
        if root_execution.env_terminal:
            return root_reward

        child_defer = _next_defer_count(
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
            return root_reward + _witness_terminal_value(root_env, root_candidate)

        best = -math.inf
        for child in child_snapshot.candidates:
            child_env, child_candidate = deepcopy((root_env, child))
            child_execution = self._simulate_macro(child_env, child_candidate)
            self.diagnostics["child_macro_simulations"] += 1
            score = (
                _dense_macro_reward(child_execution)
                + _witness_terminal_value(child_env, child_candidate)
            )
            best = max(best, score)
        if not math.isfinite(best):
            raise E21Error("depth-two search produced no finite continuation")
        return float(root_reward + best)

    def _scores(self, snapshot, prepared) -> torch.Tensor:
        candidates = tuple(
            snapshot.candidates[index] for index in prepared.source_indices
        )
        live_env = self._live_env(candidates)
        values = [self._score_root(live_env, candidate) for candidate in candidates]
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def select(self, snapshot, *, training=False, epsilon=0.0):
        if training or float(epsilon) != 0.0:
            raise E21Error("E21 permits deterministic frozen deployment only")
        prepared, liveness_forced = self.base_agent._admissible_prepared(snapshot)
        if not prepared.records:
            raise E21Error("E21 ranker received an empty exact-SAFE frontier")
        started = perf_counter()
        if liveness_forced:
            merits = torch.zeros(
                len(prepared.records), dtype=torch.float32, device=self.device
            )
        else:
            merits = self._scores(snapshot, prepared)
        hierarchy = select_hierarchical_index(
            merits,
            prepared.mode_ids,
            self.base_agent.within_temperatures,
            candidate_keys=tuple(record.key for record in prepared.records),
        )
        selected_index = 0 if liveness_forced else hierarchy.selected_index
        planning_seconds = perf_counter() - started
        self.decision_planning_seconds.append(planning_seconds)
        self.diagnostics["planning_seconds"] += planning_seconds
        self.diagnostics["planning_decisions"] += int(not liveness_forced)
        self.diagnostics["guard_forced_decisions"] += int(liveness_forced)

        source_index = prepared.source_indices[selected_index]
        candidate = snapshot.candidates[source_index]
        exact_rank_progress = bool(
            snapshot.audit.recovery_rank_exact
            and candidate.mode is ViabilityMode.RECOVER
            and candidate.rank_delta is not None
            and candidate.rank_delta > 0
        )
        source = (
            "exact_recovery_witness_guard"
            if liveness_forced
            else "e21_depth2_exact_safe_witness_search"
        )
        decision = ViabilityGraphDecision(
            candidate=candidate,
            record=prepared.records[selected_index],
            prepared_snapshot=prepared,
            q_values=tuple(float(value) for value in merits.detach().cpu()),
            mode_values=tuple(
                (ID_TO_MODE[mode], value) for mode, value in hierarchy.mode_values
            ),
            explored=False,
            selection_source=source,
            liveness_forced=liveness_forced,
            exact_rank_progress=exact_rank_progress,
        )
        base = self.base_agent
        base.decision_count += 1
        base.mode_decisions[candidate.mode.value] += 1
        base.action_decisions[candidate.action_type.value] += 1
        base.selection_sources[source] += 1
        base.safe_candidates_scored += len(prepared.records)
        base.interface_rejections += prepared.interface_rejection_count
        base.exact_rejections_seen += int(snapshot.audit.fail_closed_rejection_count)
        return decision

    def observe_outcome(self, decision, *, next_snapshot, done):
        self.consecutive_defer = _next_defer_count(
            self.consecutive_defer, decision.candidate
        )
        return self.base_agent.observe_outcome(
            decision, next_snapshot=next_snapshot, done=done
        )

    def diagnostic_record(self) -> dict:
        values = list(self.decision_planning_seconds)
        return {
            key: int(value) if float(value).is_integer() else float(value)
            for key, value in sorted(self.diagnostics.items())
        } | {
            "planning_cache_entries": len(self.planning_cache),
            "planning_seconds_mean": fmean(values) if values else 0.0,
            "planning_seconds_max": max(values) if values else 0.0,
        }


def _ledger_path(output: Path, instance_seed: int) -> Path:
    return output / "run-ledger" / f"instance-{int(instance_seed)}.json"


def _identity(record: Mapping) -> dict:
    return {
        "instance_seed": int(record["seed"]),
        "instance_index": int(record["instance_index"]),
        "episode_instance_id": record["episode_instance_id"],
        "schedule_id": record["schedule_id"],
        "episode_instance_sha256": record["canonical_sha256"],
    }


def _row(raw: Mapping, compact: Mapping, identity: Mapping, diagnostics: Mapping) -> dict:
    strict = bool(compact["strict_safe_complete"])
    timing = final90._timing(raw["delivery_deviations"]) if strict else {
        "mean_signed_deviation": None,
        "mean_absolute_error": None,
        "mean_tardiness": None,
        "mean_earliness": None,
        "within_target_window_rate": None,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        **dict(identity),
        "ranking_signal": SEARCH_RANKER,
        "model_seed": None,
        "carrier_model_seed": CARRIER_MODEL_SEED,
        "preference_lambda": None,
        "strict_safe_complete": strict,
        "dense_return": float(compact["dense_return"]) if strict else None,
        **timing,
        "steps": int(compact["steps"]) if strict else None,
        "physical_storage_relocations": (
            int(compact["physical_rehandles"]) if strict else None
        ),
        "physical_rehandles_per_100_required_deliveries": (
            float(compact["physical_rehandles_per_100"]) if strict else None
        ),
        "required_deliveries": e4.EXPECTED_BLOCKS,
        "all_selected_candidates_exact_safe": bool(
            raw.get("complete_frontier_exactly_verified", False)
        ),
        "method_failure_reason": raw.get("method_failure_reason"),
        "behavior_digest": raw.get("behavior_digest"),
        "episode_wall_seconds": float(raw.get("episode_wall_seconds", 0.0)),
        "planner_diagnostics": dict(diagnostics),
        "evaluation_learning": False,
    }


def _failed_row(error: Exception, identity: Mapping) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        **dict(identity),
        "ranking_signal": SEARCH_RANKER,
        "model_seed": None,
        "carrier_model_seed": CARRIER_MODEL_SEED,
        "preference_lambda": None,
        "strict_safe_complete": False,
        **{metric: None for metric in METRICS},
        "mean_signed_deviation": None,
        "mean_tardiness": None,
        "mean_earliness": None,
        "physical_storage_relocations": None,
        "required_deliveries": e4.EXPECTED_BLOCKS,
        "all_selected_candidates_exact_safe": None,
        "method_failure_reason": f"{type(error).__name__}: {error}",
        "behavior_digest": None,
        "episode_wall_seconds": None,
        "planner_diagnostics": {},
        "evaluation_learning": False,
    }


def run(output: Path, *, instance_limit: Optional[int], device_name: str) -> dict:
    contract, manifest, _parent = authenticate(output)
    device = pilot._device(device_name)
    if device.type != "cpu":
        raise E21Error("E21 is frozen to CPU execution")
    torch.set_num_threads(1)
    arm = _frozen_carrier()
    search_config = benchmark._search_config(arm.payload)
    liveness_rule = benchmark._liveness_rule(arm.payload)
    records = list(manifest["instances"])
    if instance_limit is not None:
        if instance_limit < 1 or instance_limit > len(records):
            raise E21Error("invalid E21 instance limit")
        records = records[:instance_limit]

    complete = safe = 0
    for record in records:
        identity = _identity(record)
        path = _ledger_path(output, identity["instance_seed"])
        if path.is_file():
            ledger = _load_json(path, "E21 rollout ledger")
            _verify_hash(ledger, "ledger_sha256", "E21 rollout ledger")
            if (
                ledger.get("contract_sha256") != contract["contract_sha256"]
                or ledger.get("manifest_sha256") != manifest["manifest_sha256"]
            ):
                raise E21Error("E21 rollout ledger binding changed")
            row = ledger["run"]
        else:
            instance = e4._load_instance(E4_OUTPUT, record)
            holders = []

            def factory(base):
                agent = CertifiedDepthTwoSearchAgent(
                    base,
                    search_config=search_config,
                    liveness_rule=liveness_rule,
                )
                holders.append(agent)
                return agent

            try:
                raw = pilot._run_raw(
                    arm, instance, device=device, wrapper_factory=factory
                )
                compact = pilot._compact_row(raw, instance)
                if len(holders) != 1:
                    raise E21Error("E21 wrapper was not constructed exactly once")
                row = _row(
                    raw,
                    compact,
                    identity,
                    holders[0].diagnostic_record(),
                )
            except Exception as error:  # prospective failed cells remain visible
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
            f"E21 {complete}/{len(records)} | instance={identity['instance_seed']} "
            f"| safe={int(row['strict_safe_complete'])}",
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
        ledger = _load_json(path, "E21 rollout ledger")
        _verify_hash(ledger, "ledger_sha256", "E21 rollout ledger")
        rows.append(ledger["run"])
    return rows


def _summarize(rows: Sequence[Mapping]) -> dict:
    safe = [row for row in rows if row["strict_safe_complete"]]
    summary = {
        "rows": len(rows),
        "strict_safe_complete": len(safe),
        "strict_completion_rate": len(safe) / len(rows) if rows else None,
        "complete_case_metrics_suppressed": len(safe) != len(rows),
    }
    for metric in METRICS:
        values = [float(row[metric]) for row in safe]
        summary[metric] = (
            fmean(values) if rows and len(values) == len(rows) else None
        )
        summary[f"{metric}_sd"] = (
            stdev(values) if rows and len(values) == len(rows) and len(values) > 1 else None
        )
    return summary


def _paired_differences(search_rows, control_rows, *, average_model_seeds: bool):
    search = {int(row["instance_seed"]): row for row in search_rows}
    grouped = defaultdict(list)
    for row in control_rows:
        grouped[int(row["instance_seed"])].append(row)
    result = {}
    for metric in METRICS:
        differences = []
        for seed in sorted(set(search) & set(grouped)):
            left = search[seed]
            controls = grouped[seed]
            if not left["strict_safe_complete"] or not all(
                row["strict_safe_complete"] for row in controls
            ):
                continue
            control_value = (
                fmean(float(row[metric]) for row in controls)
                if average_model_seeds
                else float(controls[0][metric])
            )
            differences.append(float(left[metric]) - control_value)
        if not differences:
            result[metric] = {"n": 0, "mean_search_minus_control": None, "ci95": None}
            continue
        mean = fmean(differences)
        if len(differences) < 2:
            interval = None
        else:
            half = float(student_t.ppf(0.975, len(differences) - 1)) * stdev(
                differences
            ) / math.sqrt(len(differences))
            interval = [mean - half, mean + half]
        result[metric] = {
            "n": len(differences),
            "mean_search_minus_control": mean,
            "ci95": interval,
        }
    return result


def _table(report: Mapping) -> str:
    labels = {
        SEARCH_RANKER: "Certified depth-2 search",
        "heuristic_safe": "Heuristic-safe (reused E4)",
        "frozen_qop": "Frozen Qop (reused E4)",
        "conditioned_lambda_0.10": "Conditioned VCG lambda=.10 (reused E4)",
    }
    lines = [
        "# E21 certified search-ranker comparison",
        "",
        "| Ranker | Strict complete | Return | MAE | Within ±20 | Steps | Rehandles/100 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in (SEARCH_RANKER, "heuristic_safe", "frozen_qop", "conditioned_lambda_0.10"):
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
            "The learned and heuristic controls are authenticated E4 rows, not new replications.",
            "The search witness is constructive but goal-directed; its length is an upper bound, not a shortest recovery rank.",
            "Complete-case performance is suppressed whenever an arm has a failed row.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(output: Path, *, allow_partial: bool) -> dict:
    contract, manifest, parent = authenticate(output)
    search_rows = _new_rows(output)
    if not allow_partial and len(search_rows) != len(INSTANCE_SEEDS):
        raise E21Error(
            f"E21 search grid incomplete: {len(search_rows)}/{len(INSTANCE_SEEDS)}"
        )
    aggregate = {SEARCH_RANKER: _summarize(search_rows)}
    aggregate.update(
        {name: _summarize(rows) for name, rows in parent["controls"].items()}
    )
    paired = {
        name: _paired_differences(
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
            "paper_evidence": len(search_rows) == len(INSTANCE_SEEDS),
            "post_hoc": True,
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
                "simulated_child_frontiers": sum(
                    int(item.get("simulated_child_frontiers", 0)) for item in planning
                ),
            },
            "interpretation_limits": [
                "post_hoc_test_on_an_already_opened_panel",
                "search_uses_constructive_goal_directed_witness_upper_bounds",
                "search_carrier_uses_seed0_configuration_but_no_learned_scores",
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
