#!/usr/bin/env python3
"""Replay E12's delivery failures at the certificate/executor boundary.

This is an inference-only diagnostic.  It authenticates and reads the frozen
E12/E11 artifacts, writes to a separate output directory, and never changes an
E12 evaluation ledger.  The instrumentation is observational: the original
enumerator, selector, option, environment step, and path planner still perform
all decisions and state transitions.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Optional, Sequence
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import benchmark_viability_critic_priority as benchmark
from example.Options.DirectDeliverOption import DirectDeliverOption
from example.small_rooms_env import SmallRoomsEnv
from experiments.conditioned_vcg.E11_distribution_shift_93k import run as e11
from experiments.conditioned_vcg.E12_representation_ablation_94k import (
    evaluate as core,
    frozen_parent_evaluate,
    program,
)
from methods.conditioned_vcg.representation_ablation import (
    FULL_RELATIONAL_SUCCESSOR,
)
from PSLAP.dynamic_yard import YardSnapshot
from PSLAP.relocation_family_certification import physical_recovery_state
from PSLAP.viability import RecoveryState
from PSLAP.viability_candidates import ViabilityActionType
from PSLAP.viability_dataset import recovery_state_to_dict
from PSLAP.viability_filter import online_fixed_obstacles
import run_vcg_v11_nested_handling_pilot as pilot


PROTOCOL = "vcg_conditioned_e12_execution_consistency_audit_94k_v1"
SCHEMA_VERSION = 1
DEFAULT_E12_OUTPUT = program.DEFAULT_OUTPUT
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results/vcg-conditioned-e12-execution-consistency-audit-94k"
)

# Three observed failures and one predeclared matched successful control.  The
# two 93028 failures historically have the same behavior digest, but both are
# replayed so preference invariance at this failure is observed, not assumed.
CASES = (
    {
        "case_id": "failure-93008-lambda-0p05",
        "role": "observed_failure",
        "regime_id": "combined_shift",
        "instance_seed": 93008,
        "model_seed": 2,
        "preference_lambda": 0.05,
    },
    {
        "case_id": "failure-93028-lambda-0p10",
        "role": "observed_failure",
        "regime_id": "combined_shift",
        "instance_seed": 93028,
        "model_seed": 2,
        "preference_lambda": 0.10,
    },
    {
        "case_id": "failure-93028-lambda-0p20",
        "role": "observed_failure",
        "regime_id": "combined_shift",
        "instance_seed": 93028,
        "model_seed": 2,
        "preference_lambda": 0.20,
    },
    {
        "case_id": "control-93000-lambda-0p10",
        "role": "matched_successful_control",
        "regime_id": "combined_shift",
        "instance_seed": 93000,
        "model_seed": 2,
        "preference_lambda": 0.10,
    },
)

SOURCE_PATHS = (
    "benchmark_viability_critic_priority.py",
    "train_viability_graph_smdp.py",
    "example/Options/DirectDeliverOption.py",
    "example/small_rooms_env.py",
    "PSLAP/viability.py",
    "PSLAP/viability_candidates.py",
    "PSLAP/viability_filter.py",
    "methods/conditioned_vcg/representation_ablation.py",
    "experiments/conditioned_vcg/E12_representation_ablation_94k/"
    "audit_execution_consistency.py",
)


class ExecutionConsistencyAuditError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ExecutionConsistencyAuditError(f"missing regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    if hasattr(value, "value"):
        return _jsonable(value.value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _case_map() -> dict[str, dict]:
    return {str(item["case_id"]): dict(item) for item in CASES}


def _lambda_slug(value: float) -> str:
    return str(float(value)).replace(".", "p")


def _parent_ledger_path(e12_output: Path, case: Mapping) -> Path:
    return (
        e12_output
        / "evaluation"
        / core.CONFIRMATION_PANEL
        / str(case["regime_id"])
        / FULL_RELATIONAL_SUCCESSOR
        / f"seed-{case['model_seed']}"
        / f"lambda-{_lambda_slug(case['preference_lambda'])}"
        / f"instance-{case['instance_seed']}.json"
    )


def _run_path(output: Path, case_id: str) -> Path:
    return output / "runs" / f"{case_id}.json"


def _find_instance_record(manifest: Mapping, case: Mapping) -> Mapping:
    matches = [
        item
        for item in manifest["records"]
        if item["regime_id"] == case["regime_id"]
        and int(item["seed"]) == int(case["instance_seed"])
    ]
    if len(matches) != 1:
        raise ExecutionConsistencyAuditError(
            f"expected one E11 record for {case['case_id']}, found {len(matches)}"
        )
    return matches[0]


def _load_parent_ledger(e12_output: Path, case: Mapping) -> dict:
    path = _parent_ledger_path(e12_output, case)
    ledger = program.load_json(path, label="parent E12 evaluation ledger")
    if ledger.get("ledger_sha256") != program.digest(
        ledger, hash_field="ledger_sha256"
    ):
        raise ExecutionConsistencyAuditError("parent E12 ledger hash mismatch")
    expected_spec = {
        "representation_variant": FULL_RELATIONAL_SUCCESSOR,
        "model_seed": int(case["model_seed"]),
        "preference_lambda": float(case["preference_lambda"]),
    }
    if ledger.get("spec") != expected_spec:
        raise ExecutionConsistencyAuditError("parent E12 ledger spec mismatch")
    row = ledger["row"]
    expected_failure = case["role"] == "observed_failure"
    if bool(row.get("strict_safe_complete")) == expected_failure:
        raise ExecutionConsistencyAuditError(
            f"parent outcome no longer matches role for {case['case_id']}"
        )
    if expected_failure and row.get("method_failure_reason") != (
        "macro_failure:deliver:direct_delivery_live_replan_failed"
    ):
        raise ExecutionConsistencyAuditError(
            f"parent failure changed for {case['case_id']}"
        )
    return ledger


def _contract(e12_output: Path) -> dict:
    e12_contract, _ = program.authenticate(e12_output)
    _e11_contract, e11_manifest = (
        frozen_parent_evaluate.authenticate_frozen_e11_parent(
            e12_output, PROJECT_ROOT, e11.DEFAULT_OUTPUT
        )
    )
    bound_cases = []
    for case in CASES:
        parent_path = _parent_ledger_path(e12_output, case)
        parent = _load_parent_ledger(e12_output, case)
        instance_record = _find_instance_record(e11_manifest, case)
        instance_path = e11.DEFAULT_OUTPUT / instance_record["relative_path"]
        bound_cases.append(
            {
                **dict(case),
                "episode_instance_id": instance_record["episode_instance_id"],
                "instance_raw_sha256": _sha256(instance_path),
                "parent_ledger": str(parent_path.relative_to(PROJECT_ROOT)),
                "parent_ledger_sha256": _sha256(parent_path),
                "parent_ledger_self_hash": parent["ledger_sha256"],
                "expected_behavior_digest": parent["row"]["behavior_digest"],
                "expected_strict_safe_complete": parent["row"][
                    "strict_safe_complete"
                ],
                "expected_failure_reason": parent["row"][
                    "method_failure_reason"
                ],
            }
        )
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scientific_question": (
            "did_E12_exact_certification_and_live_macro_execution_use_"
            "the_same_modeled_state_action_and_successor_contract"
        ),
        "inference_only": True,
        "training_or_checkpoint_selection": False,
        "parent_e12_contract_sha256": e12_contract["contract_sha256"],
        "parent_e11_manifest_sha256": e12_contract["e11_manifest_sha256"],
        "cases": bound_cases,
        "instrumentation": [
            "certified_recovery_action_and_successor",
            "live_environment_before_and_after_each_selected_macro",
            "direct_delivery_path_planner_calls",
            "direct_delivery_action_validity_and_replans",
            "primitive_environment_transitions_during_direct_delivery",
            "queue_arrivals_and_fixed_obstacles",
        ],
        "original_E12_ledgers_are_read_only": True,
        "source_sha256": {
            path: _sha256(PROJECT_ROOT / path) for path in SOURCE_PATHS
        },
    }
    return program.with_hash(semantic, "contract_sha256")


def prepare(output: Path, e12_output: Path) -> dict:
    output = output.resolve()
    expected = _contract(e12_output.resolve())
    path = output / "audit-contract.json"
    if path.is_file():
        observed = program.load_json(path, label="execution audit contract")
        if observed != expected:
            raise ExecutionConsistencyAuditError(
                "execution audit contract, parents, or sources changed"
            )
    else:
        if output.exists() and any(output.iterdir()):
            raise ExecutionConsistencyAuditError(
                "nonempty execution-audit output has no contract"
            )
        program.atomic_json(path, expected)
    return {
        "status": "prepared",
        "output": str(output),
        "inference_runs": len(CASES),
        "failure_replays": sum(
            item["role"] == "observed_failure" for item in CASES
        ),
        "matched_controls": sum(
            item["role"] == "matched_successful_control" for item in CASES
        ),
        "training_runs": 0,
    }


def _authenticate(output: Path, e12_output: Path) -> dict:
    observed = program.load_json(
        output / "audit-contract.json", label="execution audit contract"
    )
    if observed.get("contract_sha256") != program.digest(
        observed, hash_field="contract_sha256"
    ) or observed != _contract(e12_output.resolve()):
        raise ExecutionConsistencyAuditError(
            "execution audit contract, parents, or sources changed"
        )
    return observed


def _block_signature(block) -> dict:
    return {
        "label": str(block.label),
        "arrival_step": int(getattr(block, "arrival_step", 0)),
        "position": None if block.position is None else list(block.position),
        "storage_location": (
            None
            if block.storage_location is None
            else list(block.storage_location)
        ),
        "stored": bool(block.stored),
        "carrying": bool(block.carrying),
        "delivered": bool(block.delivered),
        "stored_time_step": (
            None if block.stored_time_step is None else int(block.stored_time_step)
        ),
        "storage_steps_elapsed": int(block.storage_steps_elapsed),
        "storage_steps_needed": int(block.storage_steps_needed),
    }


def _environment_signature(env) -> dict:
    fixed = online_fixed_obstacles(env, reserve_queue_cells=True)
    return {
        "time_step": int(env.time_steps),
        "agent_position": list(env.current_state),
        "pickup_cell": list(env.pickup_cell),
        "waiting_cell": list(env.waiting_cell),
        "exit_cells": [list(cell) for cell in env.exit_cells],
        "fixed_obstacles_with_queue_reservation": [
            list(cell) for cell in sorted(fixed)
        ],
        "blocks": [
            _block_signature(block)
            for block in sorted(env.blocks, key=lambda item: str(item.label))
        ],
    }


def _live_recovery_state(env) -> RecoveryState:
    yard = YardSnapshot.from_env(env)
    agent = tuple(env.current_state)
    fixed = online_fixed_obstacles(env, reserve_queue_cells=True) - {agent}
    return RecoveryState.from_yard_snapshot(
        yard,
        agent,
        fixed_obstacles=fixed,
        reserved_cells=(),
        pickup_cells=(tuple(env.pickup_cell),),
        wait_cells=(tuple(env.waiting_cell),),
    )


def _path_cells(env, start, actions) -> list[list[int]]:
    cell = tuple(start)
    cells = [list(cell)]
    for action in actions:
        name = env.ACTION_NAMES[int(action)]
        if name not in ("UP", "DOWN", "LEFT", "RIGHT"):
            continue
        cell = tuple(env._get_intended_cell(cell, int(action)))
        cells.append(list(cell))
    return cells


def _action_names(env, actions) -> list[str]:
    return [str(env.ACTION_NAMES[int(action)]) for action in actions]


def _recovery_action_signature(action) -> Optional[dict]:
    if action is None:
        return None
    return {
        "kind": action.kind.value,
        "block_label": str(action.block_label),
        "source": list(action.source),
        "destination": list(action.destination),
        "approach_path": [list(cell) for cell in action.approach_path],
        "transport_path": [list(cell) for cell in action.transport_path],
        "steps": int(action.steps),
    }


def _newly_materialized(before: Mapping, after: Mapping) -> list[dict]:
    left = {item["label"]: item for item in before["blocks"]}
    events = []
    for item in after["blocks"]:
        old = left[item["label"]]
        if old["position"] is None and item["position"] is not None:
            events.append(
                {
                    "label": item["label"],
                    "arrival_step": item["arrival_step"],
                    "position": item["position"],
                }
            )
    return events


class BoundaryTracker:
    def __init__(self) -> None:
        self.macros: list[dict] = []
        self.active: Optional[dict] = None

    def begin(self, env, candidate) -> dict:
        recovery_action = candidate.recovery_action
        item = {
            "candidate_key": candidate.key,
            "action_type": candidate.action_type.value,
            "target_label": candidate.target_label,
            "source": _jsonable(candidate.source),
            "destination": _jsonable(candidate.destination),
            "certificate": benchmark._certificate_signature(
                candidate.certificate
            ),
            "recovery_action": _recovery_action_signature(recovery_action),
            "certified_successor": recovery_state_to_dict(
                candidate.successor_state
            ),
            "environment_before": _environment_signature(env),
            "live_recovery_state_before": recovery_state_to_dict(
                _live_recovery_state(env)
            ),
            "path_planner_calls": [],
            "strict_action_builds": [],
            "action_validity_checks": [],
            "replans": [],
            "primitive_transitions": [],
        }
        if candidate.action_type is ViabilityActionType.DELIVER:
            self.active = item
        return item

    def finish(self, env, candidate, execution, item) -> None:
        actual = _live_recovery_state(env)
        expected = candidate.successor_state
        item.update(
            {
                "execution": _jsonable(asdict(execution)),
                "option_outcome": _jsonable(
                    getattr(candidate.option, "last_outcome", None)
                ),
                "environment_after": _environment_signature(env),
                "actual_recovery_state_after": recovery_state_to_dict(actual),
                "exact_successor_match": actual == expected,
                "physical_successor_match": (
                    physical_recovery_state(actual)
                    == physical_recovery_state(expected)
                ),
                "newly_materialized_blocks": _newly_materialized(
                    item["environment_before"], _environment_signature(env)
                ),
            }
        )
        self.macros.append(item)
        self.active = None


def _instrumentation(tracker: BoundaryTracker):
    original_execute = benchmark.execute_certified_macro
    original_path = SmallRoomsEnv.plan_path_heuristic
    original_strict = DirectDeliverOption._strict_actions
    original_valid = DirectDeliverOption._action_is_valid
    original_replan = DirectDeliverOption._replan
    original_step = SmallRoomsEnv.step

    def traced_execute(env, candidate, **kwargs):
        item = tracker.begin(env, candidate)
        execution = original_execute(env, candidate, **kwargs)
        tracker.finish(env, candidate, execution, item)
        return execution

    def traced_path(env, start, goal, ignore_block=None):
        before = _environment_signature(env)
        actions = original_path(
            env, start, goal, ignore_block=ignore_block
        )
        if tracker.active is not None:
            tracker.active["path_planner_calls"].append(
                {
                    "time_step": int(env.time_steps),
                    "start": list(start),
                    "goal": list(goal),
                    "ignore_label": (
                        None if ignore_block is None else str(ignore_block.label)
                    ),
                    "occupied_obstacles": [
                        list(block.position)
                        for block in env.blocks
                        if block.position is not None
                        and not block.delivered
                        and not block.carrying
                        and block is not ignore_block
                    ],
                    "fixed_obstacles_with_queue_reservation": before[
                        "fixed_obstacles_with_queue_reservation"
                    ],
                    "action_names": _action_names(env, actions),
                    "path_cells": _path_cells(env, start, actions),
                    "path_found": bool(tuple(start) == tuple(goal) or actions),
                }
            )
        return actions

    def traced_strict(option, block):
        before = _environment_signature(option.env)
        result = original_strict(option, block)
        if tracker.active is not None:
            tracker.active["strict_action_builds"].append(
                {
                    "time_step": int(option.env.time_steps),
                    "agent_position": list(option.env.current_state),
                    "block": None if block is None else _block_signature(block),
                    "success": result is not None,
                    "action_names": (
                        None
                        if result is None
                        else _action_names(option.env, result)
                    ),
                    "fixed_obstacles_with_queue_reservation": before[
                        "fixed_obstacles_with_queue_reservation"
                    ],
                }
            )
        return result

    def traced_valid(option, action):
        valid = original_valid(option, action)
        if tracker.active is not None:
            tracker.active["action_validity_checks"].append(
                {
                    "time_step": int(option.env.time_steps),
                    "agent_position": list(option.env.current_state),
                    "action": str(option.env.ACTION_NAMES[int(action)]),
                    "valid": bool(valid),
                }
            )
        return valid

    def traced_replan(option):
        before = _environment_signature(option.env)
        prior_count = int(option.replan_count)
        original_replan(option)
        if tracker.active is not None:
            tracker.active["replans"].append(
                {
                    "time_step": int(option.env.time_steps),
                    "count_before": prior_count,
                    "count_after": int(option.replan_count),
                    "failed": bool(option.failed),
                    "failure_reason": option.failure_reason,
                    "queued_action_names": _action_names(
                        option.env, tuple(option._actions)
                    ),
                    "environment": before,
                }
            )

    def traced_step(env, action):
        active = tracker.active
        before = _environment_signature(env) if active is not None else None
        result = original_step(env, action)
        if active is not None:
            after = _environment_signature(env)
            next_state, reward, terminal, info = result
            active["primitive_transitions"].append(
                {
                    "time_before": before["time_step"],
                    "time_after": after["time_step"],
                    "action": str(env.ACTION_NAMES[int(action)]),
                    "agent_before": before["agent_position"],
                    "agent_after": after["agent_position"],
                    "reward": float(reward),
                    "terminal": bool(terminal),
                    "info": _jsonable(info),
                    "newly_materialized_blocks": _newly_materialized(
                        before, after
                    ),
                    "fixed_obstacles_before": before[
                        "fixed_obstacles_with_queue_reservation"
                    ],
                    "fixed_obstacles_after": after[
                        "fixed_obstacles_with_queue_reservation"
                    ],
                    "state_returned": _jsonable(next_state),
                }
            )
        return result

    return (
        patch.object(benchmark, "execute_certified_macro", traced_execute),
        patch.object(SmallRoomsEnv, "plan_path_heuristic", traced_path),
        patch.object(DirectDeliverOption, "_strict_actions", traced_strict),
        patch.object(DirectDeliverOption, "_action_is_valid", traced_valid),
        patch.object(DirectDeliverOption, "_replan", traced_replan),
        patch.object(SmallRoomsEnv, "step", traced_step),
    )


def _run_one(
    output: Path,
    e12_output: Path,
    contract: Mapping,
    e11_manifest: Mapping,
    case: Mapping,
    *,
    device: torch.device,
) -> dict:
    run_path = _run_path(output, str(case["case_id"]))
    if run_path.is_file():
        observed = program.load_json(run_path, label="execution audit run")
        if (
            observed.get("run_sha256")
            != program.digest(observed, hash_field="run_sha256")
            or observed.get("contract_sha256") != contract["contract_sha256"]
        ):
            raise ExecutionConsistencyAuditError("saved audit run changed")
        return observed

    record = _find_instance_record(e11_manifest, case)
    instance = e11._load_instance(e11.DEFAULT_OUTPUT, record)
    parent = _load_parent_ledger(e12_output, case)
    op_payload, factory, checkpoint_hashes = core._checkpoint_factory(
        e12_output,
        variant=FULL_RELATIONAL_SUCCESSOR,
        model_seed=int(case["model_seed"]),
        value=float(case["preference_lambda"]),
        device=device,
    )
    regime = e11.REGIME_BY_ID[str(case["regime_id"])]
    tracker = BoundaryTracker()
    with ExitStack() as stack:
        stack.enter_context(core._runtime(factory, regime))
        for context in _instrumentation(tracker):
            stack.enter_context(context)
        raw = benchmark.run_arm(
            arm=benchmark.EXACT_FULL,
            controller_payload=op_payload,
            instance=instance,
            instance_seed=int(instance.seed),
            search_config=benchmark._search_config(op_payload),
            liveness_rule=benchmark._liveness_rule(op_payload),
            prioritizer=None,
            max_steps=program.MAX_STEPS,
            device=device,
        )

    expected = parent["row"]
    replay = {
        "strict_safe_complete": bool(
            raw["strict_method_success"]
            and raw["terminal"]
            and raw["method_failure_reason"] is None
        ),
        "method_failure_reason": raw["method_failure_reason"],
        "behavior_digest": raw["behavior_digest"],
        "macro_decisions": int(raw["macro_decisions"]),
        "steps": int(raw["steps"]),
        "terminal": bool(raw["terminal"]),
        "complete_frontier_exactly_verified": bool(
            raw["complete_frontier_exactly_verified"]
        ),
    }
    comparisons = {
        "behavior_digest_matches_parent": (
            replay["behavior_digest"] == expected["behavior_digest"]
        ),
        "strict_outcome_matches_parent": (
            replay["strict_safe_complete"]
            == bool(expected["strict_safe_complete"])
        ),
        "failure_reason_matches_parent": (
            replay["method_failure_reason"]
            == expected["method_failure_reason"]
        ),
        "macro_count_matches_parent": (
            replay["macro_decisions"] == int(expected["macro_decisions"])
        ),
        "step_count_matches_parent": (
            replay["steps"] == int(expected["observed_steps_to_stop"])
        ),
    }
    if not all(comparisons.values()):
        raise ExecutionConsistencyAuditError(
            f"instrumented replay changed behavior for {case['case_id']}"
        )
    value = program.with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "case": dict(case),
            "episode_instance_id": instance.instance_id,
            "parent_ledger_sha256": parent["ledger_sha256"],
            "checkpoints": checkpoint_hashes,
            "replay": replay,
            "parent_comparisons": comparisons,
            "macros": tracker.macros,
        },
        "run_sha256",
    )
    program.atomic_json(run_path, value)
    return value


def _failure_diagnostic(run: Mapping) -> Optional[dict]:
    failed = [
        item
        for item in run["macros"]
        if not bool(item["execution"]["option_success"])
    ]
    if not failed:
        return None
    if len(failed) != 1:
        raise ExecutionConsistencyAuditError("expected one failed macro per replay")
    item = failed[0]
    transitions = item["primitive_transitions"]
    new_arrivals = [
        event
        for transition in transitions
        for event in transition["newly_materialized_blocks"]
    ]
    empty_live_paths = [
        event for event in item["path_planner_calls"] if not event["path_found"]
    ]
    action = item["recovery_action"]
    initial_paths = item["path_planner_calls"][:2]
    certified_approach = [] if action is None else action["approach_path"]
    certified_transport = [] if action is None else action["transport_path"]
    live_approach = [] if not initial_paths else initial_paths[0]["path_cells"]
    live_transport = (
        [] if len(initial_paths) < 2 else initial_paths[1]["path_cells"]
    )
    fixed = item["environment_before"][
        "fixed_obstacles_with_queue_reservation"
    ]
    fixed_cells = {tuple(cell) for cell in fixed}
    arrival_cells = {
        tuple(event["position"])
        for event in new_arrivals
        if event["position"] is not None
    }
    certified_cells = {
        tuple(cell) for cell in certified_approach + certified_transport
    }
    live_cells = {tuple(cell) for cell in live_approach + live_transport}
    paths_match = bool(
        certified_approach == live_approach
        and certified_transport == live_transport
    )
    if (
        not paths_match
        and arrival_cells & live_cells
        and not (arrival_cells & certified_cells)
        and empty_live_paths
    ):
        classification = (
            "executor_departed_from_certified_path_then_modeled_queue_arrival_"
            "invalidated_live_route"
        )
    elif new_arrivals and empty_live_paths:
        classification = "in_macro_arrival_invalidated_live_route"
    else:
        classification = "unclassified_delivery_execution_failure"
    return {
        "candidate_key": item["candidate_key"],
        "action_type": item["action_type"],
        "failure_reason": item["execution"]["failure_reason"],
        "environment_time_before": item["environment_before"]["time_step"],
        "environment_time_after": item["environment_after"]["time_step"],
        "certified_recovery_action": item["recovery_action"],
        "initial_live_approach_path": live_approach,
        "initial_live_transport_path": live_transport,
        "initial_live_paths_match_certified_paths": paths_match,
        "certified_path_intersects_fixed_obstacles": bool(
            certified_cells & fixed_cells
        ),
        "initial_live_path_intersects_fixed_obstacles": bool(
            live_cells & fixed_cells
        ),
        "newly_materialized_blocks_during_macro": new_arrivals,
        "arrival_intersects_certified_path": bool(
            arrival_cells & certified_cells
        ),
        "arrival_intersects_initial_live_path": bool(arrival_cells & live_cells),
        "live_replans": len(item["replans"]),
        "empty_live_path_calls": empty_live_paths,
        "exact_successor_match": item["exact_successor_match"],
        "physical_successor_match": item["physical_successor_match"],
        "option_outcome": item["option_outcome"],
        "classification": classification,
    }


def analyze(output: Path, e12_output: Path, *, allow_partial: bool) -> dict:
    contract = _authenticate(output.resolve(), e12_output.resolve())
    runs = []
    missing = []
    for case in CASES:
        path = _run_path(output, str(case["case_id"]))
        if not path.is_file():
            missing.append(str(case["case_id"]))
            continue
        run = program.load_json(path, label="execution audit run")
        if (
            run.get("run_sha256")
            != program.digest(run, hash_field="run_sha256")
            or run.get("contract_sha256") != contract["contract_sha256"]
        ):
            raise ExecutionConsistencyAuditError("saved audit run changed")
        runs.append(run)
    if missing and not allow_partial:
        raise ExecutionConsistencyAuditError(
            f"missing audit cases: {', '.join(missing)}"
        )

    diagnostics = []
    for run in runs:
        diagnostic = _failure_diagnostic(run)
        diagnostics.append(
            {
                "case_id": run["case"]["case_id"],
                "role": run["case"]["role"],
                "replay": run["replay"],
                "parent_comparisons": run["parent_comparisons"],
                "failure_diagnostic": diagnostic,
            }
        )
    report = program.with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "status": "partial" if missing else "complete",
            "completed_cases": len(runs),
            "expected_cases": len(CASES),
            "missing_cases": missing,
            "instrumentation_preserved_parent_behavior": all(
                all(item["parent_comparisons"].values()) for item in diagnostics
            ),
            "observed_failures_reproduced": sum(
                item["role"] == "observed_failure"
                and item["replay"]["method_failure_reason"]
                == "macro_failure:deliver:direct_delivery_live_replan_failed"
                for item in diagnostics
            ),
            "diagnostics": diagnostics,
        },
        "report_sha256",
    )
    program.atomic_json(output / "execution-consistency-report.json", report)
    return report


def run(
    output: Path,
    e12_output: Path,
    *,
    device_name: str,
    case_ids: Sequence[str],
) -> dict:
    prepare(output, e12_output)
    contract = _authenticate(output.resolve(), e12_output.resolve())
    _e11_contract, manifest = (
        frozen_parent_evaluate.authenticate_frozen_e11_parent(
            e12_output.resolve(), PROJECT_ROOT, e11.DEFAULT_OUTPUT
        )
    )
    selected = tuple(case_ids) if case_ids else tuple(_case_map())
    unknown = set(selected) - set(_case_map())
    if unknown:
        raise ExecutionConsistencyAuditError(
            f"unknown case ids: {sorted(unknown)}"
        )
    device = pilot._device(device_name)
    completed = []
    for index, case_id in enumerate(selected, start=1):
        result = _run_one(
            output.resolve(),
            e12_output.resolve(),
            contract,
            manifest,
            _case_map()[case_id],
            device=device,
        )
        completed.append(case_id)
        print(
            json.dumps(
                {
                    "audit_case": f"{index}/{len(selected)}",
                    "case_id": case_id,
                    "failure_reason": result["replay"]["method_failure_reason"],
                    "macros": result["replay"]["macro_decisions"],
                    "steps": result["replay"]["steps"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return {
        "status": "ran_selected_cases",
        "completed": completed,
        "remaining": [case for case in _case_map() if case not in completed],
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--e12-output", type=Path, default=DEFAULT_E12_OUTPUT
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--case", action="append", choices=tuple(_case_map()))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "prepare":
        result = prepare(args.output, args.e12_output)
    elif args.command == "run":
        result = run(
            args.output,
            args.e12_output,
            device_name=args.device,
            case_ids=tuple(args.case or ()),
        )
        if not args.case:
            result = analyze(args.output, args.e12_output, allow_partial=False)
    else:
        result = analyze(
            args.output, args.e12_output, allow_partial=args.allow_partial
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
