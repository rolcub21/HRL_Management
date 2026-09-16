#!/usr/bin/env python3
"""Authenticated 5x5 frozen-snapshot robust-recovery bridge experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean
import tempfile
from types import MethodType
from typing import Mapping, Optional, Sequence

import torch

from PSLAP.viability import ViabilityStatus
import run_vcg_v11_nested_handling_pilot as pilot
import run_vcg_v11_nested_lambda_frontier_confirmation_89k as source89
from vcg_robust_recovery_snapshot_5x5 import (
    PROTOCOL as SOLVER_PROTOCOL,
    UNCERTAINTY_CONTRACT,
    SolveStatus,
    certify_snapshot,
    recovery_state_from_dict,
    recovery_state_to_dict,
    snapshot_certificate_to_dict,
)
from vcg_v11_nested_handling import score_cost_records
from viability_graph_hierarchy import prepare_viability_snapshot


PROTOCOL = "vcg_5x5_authenticated_snapshot_robust_recovery_panel_v1"
SCHEMA_VERSION = 1
SOURCE_MODEL_SEED = 0
SOURCE_LAMBDA = 0.0
TARGET_OCCUPANCIES = (2, 4, 6, 8)
HANDLING_LAMBDAS = (0.0, 0.2)
MAX_EXPANSIONS = 1_000
BUDGET_SLACK_PER_NOMINAL_WITNESS_MACRO = 3

HERE = Path(__file__).resolve().parent
SOURCE_OUTPUT = (
    HERE / "results/vcg-v1-1-nested-lambda-frontier-confirmation-89k"
)
DEFAULT_OUTPUT = HERE / "results/vcg-robust-recovery-snapshot-panel-5x5"
CONTRACT_NAME = "bridge-contract.json"
PANEL_NAME = "snapshot-panel.json"
REPORT_NAME = "bridge-report.json"


class BridgeError(RuntimeError):
    pass


def _target_search_order(assigned: int) -> tuple[int, ...]:
    if assigned not in TARGET_OCCUPANCIES:
        raise BridgeError(f"unsupported assigned occupancy: {assigned}")
    return tuple(
        sorted(
            TARGET_OCCUPANCIES,
            key=lambda value: (abs(value - assigned), -value),
        )
    )


def _canonical(value: Mapping, *, drop: Optional[str] = None) -> bytes:
    payload = dict(value)
    if drop is not None:
        payload.pop(drop, None)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Mapping, *, drop: Optional[str] = None) -> str:
    return hashlib.sha256(_canonical(value, drop=drop)).hexdigest()


def _sha(path: Path) -> str:
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise BridgeError(f"expected canonical regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise BridgeError(f"expected canonical regular JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BridgeError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise BridgeError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping) -> None:
    path = Path(path).absolute()
    if path.is_symlink():
        raise BridgeError(f"refusing to replace symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.resolve() != path.parent:
        raise BridgeError(f"output parent must be canonical: {path.parent}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _verify_self_hash(value: Mapping, field: str, *, label: str) -> None:
    received = value.get(field)
    if not isinstance(received, str) or received != _digest(value, drop=field):
        raise BridgeError(f"{label} self hash mismatch")


def _historical_source_auth(project_root: Path) -> dict:
    """Authenticate 89k without requiring its original CUDA hardware."""

    root = project_root.resolve()
    output = (
        root
        / "results/vcg-v1-1-nested-lambda-frontier-confirmation-89k"
    ).resolve()
    contract_path = output / source89.CONTRACT_NAME
    contract = source89._load_json(contract_path, label="89k contract")
    source89._verify_hash(contract, "contract_sha256", label="89k contract")
    expected_sources = source89._source_hashes(root)
    if contract.get("source_sha256") != expected_sources:
        raise BridgeError("89k source files changed")
    if (
        contract.get("protocol") != source89.PROTOCOL
        or contract.get("output_dir") != str(output)
        or contract.get("training_or_learning") is not False
        or contract.get("lambda_grid") != list(source89.LAMBDA_GRID)
        or contract.get("instance_seeds") != list(source89.INSTANCE_SEEDS)
    ):
        raise BridgeError("89k contract role changed")

    arms, _ = source89._authenticate_inputs(root)
    arm = arms[SOURCE_MODEL_SEED]
    if arm.checkpoint_sha256 != source89.SELECTED_CHECKPOINT_SHA256[0]:
        raise BridgeError("89k source checkpoint changed")

    activation = source89._load_json(
        output / source89.ACTIVATION_NAME, label="89k activation"
    )
    source89._verify_hash(
        activation, "activation_sha256", label="89k activation"
    )
    if (
        activation.get("panel_opened") is not True
        or activation.get("contract_sha256") != contract["contract_sha256"]
        or activation.get("authorized_seeds") != list(source89.INSTANCE_SEEDS)
    ):
        raise BridgeError("89k activation changed")

    manifest = source89._load_json(
        output / source89.INSTANCE_MANIFEST_NAME,
        label="89k instance manifest",
    )
    source89._verify_hash(
        manifest, "manifest_sha256", label="89k instance manifest"
    )
    records = manifest.get("instances")
    if (
        manifest.get("contract_sha256") != contract["contract_sha256"]
        or manifest.get("activation_sha256") != activation["activation_sha256"]
        or manifest.get("instance_count") != len(source89.INSTANCE_SEEDS)
        or manifest.get("panel_opened") is not True
        or not isinstance(records, list)
        or len(records) != 30
        or [record.get("instance_seed") for record in records]
        != list(source89.INSTANCE_SEEDS)
    ):
        raise BridgeError("89k manifest changed")
    for record in records:
        instance = source89._load_instance(output, record)
        if instance.seed != int(record["instance_seed"]):
            raise BridgeError("89k instance seed changed")
    expected_instance_paths = {
        (output / str(record["relative_path"])).absolute()
        for record in records
    }
    observed_instance_paths = {
        path.absolute()
        for path in (output / "episode-instances").iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if observed_instance_paths != expected_instance_paths:
        raise BridgeError("89k instance file tree changed")

    report = source89._load_json(
        output / source89.REPORT_NAME, label="89k report"
    )
    source89._verify_hash(report, "report_sha256", label="89k report")
    if (
        report.get("status") != "passed"
        or report.get("confirmation_gate", {}).get("passed") is not True
        or report.get("row_count") != 450
        or report.get("strict_safe_row_count") != 450
        or report.get("training_or_learning") is not False
        or report.get("contract_sha256") != contract["contract_sha256"]
        or report.get("manifest_sha256") != manifest["manifest_sha256"]
    ):
        raise BridgeError("89k confirmation report is not authoritative")
    return {
        "contract": contract,
        "activation": activation,
        "manifest": manifest,
        "report": report,
        "arm": arm,
        "source_output": output,
        "artifact_authentication": (
            "historical_contract_self_hash_sources_checkpoints_instances_"
            "ledgers_and_report_authenticated_without_requiring_original_gpu"
        ),
    }


def _contract(project_root: Path, output_dir: Path) -> dict:
    auth = _historical_source_auth(project_root)
    source_contract = auth["contract"]
    source_manifest = auth["manifest"]
    source_report = auth["report"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "prepared",
        "scientific_role": (
            "authenticated_5x5_closed_admission_robust_recovery_bridge"
        ),
        "output_dir": str(output_dir.resolve()),
        "solver_protocol": SOLVER_PROTOCOL,
        "uncertainty_contract": UNCERTAINTY_CONTRACT,
        "source": {
            "protocol": source89.PROTOCOL,
            "output": str(auth["source_output"]),
            "contract_sha256": source_contract["contract_sha256"],
            "contract_raw_sha256": _sha(
                auth["source_output"] / source89.CONTRACT_NAME
            ),
            "activation_sha256": auth["activation"]["activation_sha256"],
            "manifest_sha256": source_manifest["manifest_sha256"],
            "report_sha256": source_report["report_sha256"],
            "model_seed": SOURCE_MODEL_SEED,
            "handling_lambda": SOURCE_LAMBDA,
            "checkpoint_sha256": source89.SELECTED_CHECKPOINT_SHA256[0],
            "cost_head_sha256": source89.COST_HEAD_SHA256[0],
            "trajectory_authentication": (
                "exact_behavior_digest_and_compact_terminal_row"
            ),
        },
        "snapshot_selection": {
            "one_snapshot_per_episode_instance": True,
            "assigned_target_formula": "(2,4,6,8)[instance_index mod 4]",
            "target_occupancies": list(TARGET_OCCUPANCIES),
            "assigned_target_quotas": {"2": 8, "4": 8, "6": 7, "8": 7},
            "selected_target_rule": (
                "nearest_available_target_with_higher_occupancy_tie_break"
            ),
            "earliest_matching_executed_decision_boundary": True,
            "current_inbound_may_be_present": True,
            "inbound_excluded_from_stored_recovery_workload": True,
            "inbound_retained_in_fixed_queue_obstacle_envelope": True,
            "require_nonterminal": True,
            "require_current_exact_safe_certificate": True,
            "require_complete_exact_frontier": True,
            "deterministic_nearest_occupancy_substitution": True,
        },
        "completion_budget": {
            "formula": (
                "source_nominal_witness_primitive_steps_plus_3_times_"
                "source_nominal_witness_macro_count"
            ),
            "slack_per_witness_macro": (
                BUDGET_SLACK_PER_NOMINAL_WITNESS_MACRO
            ),
            "role": "predeclared_snapshot_recovery_horizon_not_block_deadline",
        },
        "max_expansions_per_method_per_snapshot": MAX_EXPANSIONS,
        "handling_lambdas_for_read_only_root_preference": list(
            HANDLING_LAMBDAS
        ),
        "root_preference_scope": (
            "candidate_level_scalar_merit_diagnostic_not_full_mode_"
            "aggregated_policy_replay"
        ),
        "training_or_learning": False,
        "future_arrival_schedule_available_to_solver": False,
        "full_dynamic_episode_claim": False,
        "hardware_claim": False,
        "source_sha256": {
            "solver": _sha(HERE / "vcg_robust_recovery_snapshot_5x5.py"),
            "runner": _sha(Path(__file__).resolve()),
        },
    }
    payload["contract_sha256"] = _digest(payload)
    return payload


def prepare_contract(project_root: Path, output_dir: Path) -> dict:
    expected = _contract(project_root, output_dir)
    path = output_dir.resolve() / CONTRACT_NAME
    if path.exists():
        observed = _read(path)
        _verify_self_hash(observed, "contract_sha256", label="bridge contract")
        if observed != expected:
            raise BridgeError("bridge contract or bound sources changed")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise BridgeError("nonempty bridge output has no contract")
        _atomic_json(path, expected)
    return expected


def _source_ledger(
    auth: Mapping,
    record: Mapping,
) -> dict:
    path = source89._ledger_path(
        auth["source_output"],
        SOURCE_LAMBDA,
        SOURCE_MODEL_SEED,
        int(record["instance_seed"]),
    )
    ledger = source89._load_json(path, label="89k source ledger")
    row = source89._validate_ledger(
        ledger,
        contract=auth["contract"],
        manifest=auth["manifest"],
        value=SOURCE_LAMBDA,
        model_seed=SOURCE_MODEL_SEED,
        record=record,
    )
    return {
        "row": row,
        "ledger_sha256": ledger["ledger_sha256"],
        "ledger_raw_sha256": _sha(path),
        "relative_path": str(path.relative_to(auth["source_output"])),
    }


def _assert_compact_replay(expected: Mapping, observed: Mapping) -> None:
    exact = (
        "instance_seed",
        "episode_instance_id",
        "schedule_id",
        "behavior_digest",
        "strict_safe_complete",
        "steps",
        "physical_rehandles",
        "delivery_deviations",
    )
    for field in exact:
        if observed.get(field) != expected.get(field):
            raise BridgeError(f"89k CPU replay changed {field}")
    for field in ("dense_return", "mean_absolute_error"):
        if not math.isclose(
            float(observed[field]),
            float(expected[field]),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise BridgeError(f"89k CPU replay changed {field}")


def _capture_factory(
    *,
    target_occupancies: Sequence[int],
    captures: dict[int, dict],
    cost_network,
):
    target_set = frozenset(int(value) for value in target_occupancies)

    def factory(base_agent):
        original_select = base_agent.select
        local_decision_index = 0

        def select(self, snapshot, *, training=True, epsilon=None):
            nonlocal local_decision_index
            current_decision_index = local_decision_index
            local_decision_index += 1
            occupancy = len(snapshot.recovery_state.blocks)
            eligible = bool(
                occupancy in target_set
                and occupancy not in captures
                and snapshot.current_certificate.status is ViabilityStatus.SAFE
                and snapshot.audit.complete_frontier_exactly_verified
                and not snapshot.audit.terminal
            )
            pending = None
            if eligible:
                context = self.recovery_witness_guard.feature_context(
                    snapshot,
                    forced_frontier=False,
                )
                prepared = prepare_viability_snapshot(
                    snapshot,
                    guard_context=context,
                )
                with torch.inference_mode():
                    q_operational = self._score_records(prepared.records)
                    q_handling = score_cost_records(
                        cost_network,
                        self.Q_local,
                        prepared.records,
                    )
                source_rows = []
                for index, (record, source_index) in enumerate(
                    zip(prepared.records, prepared.source_indices)
                ):
                    candidate = snapshot.candidates[source_index]
                    source_rows.append(
                        {
                            "key": record.key,
                            "action_type": record.action_type,
                            "target_label": candidate.target_label,
                            "source": (
                                None
                                if candidate.source is None
                                else list(candidate.source)
                            ),
                            "destination": (
                                None
                                if candidate.destination is None
                                else list(candidate.destination)
                            ),
                            "q_operational": float(q_operational[index]),
                            "q_predicted_rehandles": float(q_handling[index]),
                            "source_exact_safe": True,
                        }
                    )
                certificate = snapshot.current_certificate
                pending = {
                    "source_decision_index": int(current_decision_index),
                    "decision_epoch": int(snapshot.decision_epoch),
                    "target_occupancy": int(occupancy),
                    "observed_occupancy": int(occupancy),
                    "inbound_label": snapshot.inbound_label,
                    "recovery_state": recovery_state_to_dict(
                        snapshot.recovery_state
                    ),
                    "source_recovery_state_digest": hashlib.sha256(
                        _canonical(
                            recovery_state_to_dict(snapshot.recovery_state)
                        )
                    ).hexdigest(),
                    "source_current_certificate": {
                        "status": certificate.status.value,
                        "reason": certificate.reason,
                        "witness_macro_count": certificate.witness_macro_count,
                        "witness_primitive_steps": (
                            certificate.witness_primitive_steps
                        ),
                        "exhaustive": bool(certificate.exhaustive),
                    },
                    "source_frontier": {
                        "candidate_count": len(snapshot.candidates),
                        "candidate_keys": [
                            candidate.key for candidate in snapshot.candidates
                        ],
                        "fail_closed_rejections": int(
                            snapshot.audit.fail_closed_rejection_count
                        ),
                        "complete_frontier_exactly_verified": bool(
                            snapshot.audit.complete_frontier_exactly_verified
                        ),
                    },
                    "unforced_guard_context": list(context),
                    "frozen_candidate_values": source_rows,
                }
            decision = original_select(
                snapshot,
                training=training,
                epsilon=epsilon,
            )
            if pending is not None:
                pending["authenticated_source_selected_key"] = (
                    decision.candidate.key
                )
                pending["authenticated_source_liveness_forced"] = bool(
                    decision.liveness_forced
                )
                captures[int(occupancy)] = pending
            return decision

        base_agent.select = MethodType(select, base_agent)
        return base_agent

    return factory


def _replay_snapshot(
    *,
    auth: Mapping,
    record: Mapping,
    target_occupancy: int,
    cost_network,
    device: torch.device,
) -> dict:
    source_ledger = _source_ledger(auth, record)
    expected = source_ledger["row"]
    instance = source89._load_instance(auth["source_output"], record)
    captures: dict[int, dict] = {}
    raw = pilot._run_raw(
        auth["arm"],
        instance,
        device=device,
        wrapper_factory=_capture_factory(
            target_occupancies=TARGET_OCCUPANCIES,
            captures=captures,
            cost_network=cost_network,
        ),
    )
    compact = pilot._compact_row(raw, instance)
    _assert_compact_replay(expected, compact)
    search_order = _target_search_order(target_occupancy)
    selected_target = next(
        (value for value in search_order if value in captures),
        None,
    )
    if selected_target is None:
        raise BridgeError(
            f"instance {instance.seed} has no exact-SAFE boundary at any "
            f"target occupancy"
        )
    capture = captures[selected_target]
    certificate = capture["source_current_certificate"]
    witness_steps = certificate["witness_primitive_steps"]
    witness_macros = certificate["witness_macro_count"]
    if (
        isinstance(witness_steps, bool)
        or not isinstance(witness_steps, int)
        or witness_steps < 1
        or isinstance(witness_macros, bool)
        or not isinstance(witness_macros, int)
        or witness_macros < 1
    ):
        raise BridgeError("source SAFE certificate lacks a finite witness")
    capture.update(
        {
            "instance_index": int(record["instance_index"]),
            "instance_seed": int(record["instance_seed"]),
            "episode_instance_id": record["episode_instance_id"],
            "schedule_id": record["schedule_id"],
            "source_behavior_digest": raw["behavior_digest"],
            "source_ledger_relative_path": source_ledger["relative_path"],
            "source_ledger_sha256": source_ledger["ledger_sha256"],
            "source_ledger_raw_sha256": source_ledger[
                "ledger_raw_sha256"
            ],
            "source_ledger_strict_safe_complete": expected[
                "strict_safe_complete"
            ],
            "source_replay_exactly_authenticated": True,
            "assigned_target_occupancy": int(target_occupancy),
            "target_occupancy_search_order": list(search_order),
            "unavailable_preceding_target_occupancies": list(
                search_order[: search_order.index(selected_target)]
            ),
            "stored_recovery_workload_only": True,
            "current_inbound_excluded_from_recovery_workload": (
                capture["inbound_label"] is not None
            ),
            "current_inbound_retained_in_fixed_obstacle_envelope": (
                capture["inbound_label"] is None
                or {
                    tuple(cell)
                    for cell in capture["recovery_state"]["pickup_cells"]
                }
                <= {
                    tuple(cell)
                    for cell in capture["recovery_state"]["fixed_obstacles"]
                }
            ),
            "primitive_budget": int(
                witness_steps
                + BUDGET_SLACK_PER_NOMINAL_WITNESS_MACRO * witness_macros
            ),
            "future_schedule_used_by_solver": False,
        }
    )
    return capture


def _validate_panel(panel: Mapping, contract: Mapping) -> dict:
    _verify_self_hash(panel, "panel_sha256", label="snapshot panel")
    snapshots = panel.get("snapshots")
    if (
        panel.get("schema_version") != SCHEMA_VERSION
        or panel.get("protocol") != PROTOCOL
        or panel.get("contract_sha256") != contract["contract_sha256"]
        or panel.get("snapshot_count") != 30
        or panel.get("episode_instance_count") != 30
        or not isinstance(snapshots, list)
        or len(snapshots) != 30
    ):
        raise BridgeError("snapshot panel identity or grid changed")
    expected_seeds = list(source89.INSTANCE_SEEDS)
    if [item.get("instance_seed") for item in snapshots] != expected_seeds:
        raise BridgeError("snapshot panel EpisodeInstance order changed")
    if [item.get("instance_index") for item in snapshots] != list(range(30)):
        raise BridgeError("snapshot panel instance indices changed")
    observed_counts = Counter(item.get("observed_occupancy") for item in snapshots)
    serialized_counts = {
        str(key): value for key, value in sorted(observed_counts.items())
    }
    if (
        panel.get("occupancy_counts") != serialized_counts
        or set(observed_counts) != set(TARGET_OCCUPANCIES)
    ):
        raise BridgeError("snapshot panel occupancy strata changed")
    for item in snapshots:
        index = int(item["instance_index"])
        assigned = TARGET_OCCUPANCIES[index % len(TARGET_OCCUPANCIES)]
        search_order = _target_search_order(assigned)
        target = item.get("target_occupancy")
        if target not in search_order:
            raise BridgeError("snapshot target is outside the declared strata")
        target_index = search_order.index(target)
        state = item.get("recovery_state")
        if not isinstance(state, Mapping):
            raise BridgeError("snapshot panel has no serialized RecoveryState")
        observed_digest = hashlib.sha256(_canonical(state)).hexdigest()
        if (
            item.get("assigned_target_occupancy") != assigned
            or item.get("target_occupancy_search_order") != list(search_order)
            or item.get("unavailable_preceding_target_occupancies")
            != list(search_order[:target_index])
            or item.get("observed_occupancy") != target
            or item.get("source_recovery_state_digest") != observed_digest
            or item.get("source_replay_exactly_authenticated") is not True
            or item.get("stored_recovery_workload_only") is not True
            or item.get("current_inbound_excluded_from_recovery_workload")
            != (item.get("inbound_label") is not None)
            or item.get(
                "current_inbound_retained_in_fixed_obstacle_envelope"
            ) is not True
        ):
            raise BridgeError(
                f"snapshot panel row changed: {item.get('instance_seed')}"
            )
    return dict(panel)


def prepare_panel(project_root: Path, output_dir: Path) -> dict:
    contract = prepare_contract(project_root, output_dir)
    path = output_dir.resolve() / PANEL_NAME
    if path.exists():
        panel = _read(path)
        return _validate_panel(panel, contract)

    auth = _historical_source_auth(project_root)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    base = pilot._fresh_base(auth["arm"], device)
    cost_network = source89._load_bound_cost(
        project_root,
        auth["arm"],
        device=device,
        config=base.config,
    )
    snapshots = []
    for record in auth["manifest"]["instances"]:
        target = TARGET_OCCUPANCIES[int(record["instance_index"]) % 4]
        snapshot = _replay_snapshot(
            auth=auth,
            record=record,
            target_occupancy=target,
            cost_network=cost_network,
            device=device,
        )
        snapshots.append(snapshot)
        print(
            json.dumps(
                {
                    "snapshot_progress": f"{len(snapshots)}/30",
                    "instance_seed": record["instance_seed"],
                    "assigned_target_occupancy": target,
                    "selected_target_occupancy": snapshot[
                        "target_occupancy"
                    ],
                }
            ),
            flush=True,
        )
    quotas = Counter(item["observed_occupancy"] for item in snapshots)
    if set(quotas) != set(TARGET_OCCUPANCIES):
        raise BridgeError(f"snapshot occupancy strata changed: {quotas!r}")
    if len({item["instance_seed"] for item in snapshots}) != 30:
        raise BridgeError("snapshot panel lost EpisodeInstance independence")
    panel = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "role": "authenticated_frozen_5x5_closed_admission_snapshot_panel",
        "contract_sha256": contract["contract_sha256"],
        "snapshot_count": len(snapshots),
        "episode_instance_count": 30,
        "occupancy_counts": {
            str(key): value for key, value in sorted(quotas.items())
        },
        "training_or_learning": False,
        "future_arrival_schedule_excluded_after_snapshot": True,
        "snapshots": snapshots,
    }
    panel["panel_sha256"] = _digest(panel)
    _validate_panel(panel, contract)
    _atomic_json(path, panel)
    return panel


def _select_from_certified_frontier(
    values: Sequence[Mapping],
    admitted: Sequence[str],
    handling_lambda: float,
) -> Optional[str]:
    admitted_set = set(admitted)
    rows = [
        row
        for row in values
        if row["key"] in admitted_set
        and row["action_type"] in ("deliver", "reconfigure")
    ]
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: (
            -(
                float(row["q_operational"])
                - float(handling_lambda)
                * float(row["q_predicted_rehandles"])
            ),
            str(row["key"]),
        ),
    )["key"]


def _evaluate_one(snapshot: Mapping) -> dict:
    state = recovery_state_from_dict(snapshot["recovery_state"])
    if len(state.blocks) != int(snapshot["target_occupancy"]):
        raise BridgeError("serialized snapshot occupancy changed")
    if snapshot.get(
        "current_inbound_retained_in_fixed_obstacle_envelope"
    ) is not True:
        raise BridgeError("current inbound was not retained as a fixed obstacle")
    certificate = certify_snapshot(
        state,
        primitive_budget=int(snapshot["primitive_budget"]),
        max_expansions=MAX_EXPANSIONS,
    )
    certificate_dict = snapshot_certificate_to_dict(certificate)
    selections = {}
    for method_name in ("nominal", "one_step", "recursive"):
        method = getattr(certificate, method_name)
        selections[method_name] = {
            str(value): _select_from_certified_frontier(
                snapshot["frozen_candidate_values"],
                method.admitted_action_keys,
                value,
            )
            for value in HANDLING_LAMBDAS
        }
    return {
        "instance_seed": snapshot["instance_seed"],
        "episode_instance_id": snapshot["episode_instance_id"],
        "target_occupancy": snapshot["target_occupancy"],
        "assigned_target_occupancy": snapshot[
            "assigned_target_occupancy"
        ],
        "decision_epoch": snapshot["decision_epoch"],
        "primitive_budget": snapshot["primitive_budget"],
        "source_recovery_state_digest": snapshot[
            "source_recovery_state_digest"
        ],
        "certificate": certificate_dict,
        "root_preference_selection": selections,
        "certified_set_independent_of_lambda": True,
        "training_or_learning": False,
    }


def _validate_result_row(
    row: Mapping,
    *,
    contract: Mapping,
    panel: Mapping,
    snapshot: Mapping,
) -> dict:
    _verify_self_hash(row, "row_sha256", label="snapshot result")
    certificate = row.get("certificate")
    if not isinstance(certificate, Mapping):
        raise BridgeError("snapshot result has no certificate")
    if (
        row.get("schema_version") != SCHEMA_VERSION
        or row.get("protocol") != PROTOCOL
        or row.get("contract_sha256") != contract["contract_sha256"]
        or row.get("panel_sha256") != panel["panel_sha256"]
        or row.get("instance_seed") != snapshot["instance_seed"]
        or row.get("episode_instance_id") != snapshot["episode_instance_id"]
        or row.get("target_occupancy") != snapshot["target_occupancy"]
        or row.get("assigned_target_occupancy")
        != snapshot["assigned_target_occupancy"]
        or row.get("primitive_budget") != snapshot["primitive_budget"]
        or row.get("source_recovery_state_digest")
        != snapshot["source_recovery_state_digest"]
        or certificate.get("primitive_budget") != snapshot["primitive_budget"]
    ):
        raise BridgeError(
            f"snapshot result identity changed: {snapshot['instance_seed']}"
        )
    return dict(row)


def _status_counts(rows: Sequence[Mapping], method: str) -> dict:
    counts = Counter(
        row["certificate"][method]["state"]["status"] for row in rows
    )
    return {key: counts.get(key, 0) for key in SolveStatus._value2member_map_}


def _method_summary(rows: Sequence[Mapping], method: str) -> dict:
    if not rows:
        raise BridgeError("cannot summarize an empty snapshot group")
    return {
        "snapshot_count": len(rows),
        "snapshot_status_counts": _status_counts(rows, method),
        "admitted_snapshot_count": sum(
            row["certificate"][method]["state"]["admitted"] for row in rows
        ),
        "mean_admitted_root_actions": fmean(
            len(row["certificate"][method]["admitted_action_keys"])
            for row in rows
        ),
        "total_expanded_nodes": sum(
            row["certificate"][method]["computation"]["expanded_nodes"]
            for row in rows
        ),
        "snapshots_hitting_compute_cutoff": sum(
            row["certificate"][method]["computation"]["cutoff_count"] > 0
            for row in rows
        ),
    }


def evaluate_panel(project_root: Path, output_dir: Path) -> dict:
    contract = prepare_contract(project_root, output_dir)
    panel = prepare_panel(project_root, output_dir)
    rows = []
    row_dir = output_dir.resolve() / "snapshot-results"
    for index, snapshot in enumerate(panel["snapshots"], start=1):
        path = row_dir / f"instance-{int(snapshot['instance_seed'])}.json"
        if path.exists():
            row = _read(path)
            row = _validate_result_row(
                row,
                contract=contract,
                panel=panel,
                snapshot=snapshot,
            )
        else:
            row = {
                "schema_version": SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "contract_sha256": contract["contract_sha256"],
                "panel_sha256": panel["panel_sha256"],
                **_evaluate_one(snapshot),
            }
            row["row_sha256"] = _digest(row)
            _validate_result_row(
                row,
                contract=contract,
                panel=panel,
                snapshot=snapshot,
            )
            _atomic_json(path, row)
        rows.append(row)
        print(
            json.dumps(
                {
                    "certification_progress": f"{index}/30",
                    "instance_seed": snapshot["instance_seed"],
                    "recursive_status": row["certificate"]["recursive"][
                        "state"
                    ]["status"],
                }
            ),
            flush=True,
        )

    summaries = {
        method: _method_summary(rows, method)
        for method in ("nominal", "one_step", "recursive")
    }
    summaries_by_occupancy = {
        str(occupancy): {
            method: _method_summary(
                [row for row in rows if row["target_occupancy"] == occupancy],
                method,
            )
            for method in ("nominal", "one_step", "recursive")
        }
        for occupancy in TARGET_OCCUPANCIES
    }
    lambda_changes = sum(
        row["root_preference_selection"]["recursive"]["0.0"]
        != row["root_preference_selection"]["recursive"]["0.2"]
        and row["root_preference_selection"]["recursive"]["0.0"] is not None
        and row["root_preference_selection"]["recursive"]["0.2"] is not None
        for row in rows
    )
    checks = {
        "all_30_source_replays_authenticated": panel["snapshot_count"] == 30
        and all(
            snapshot["source_replay_exactly_authenticated"]
            for snapshot in panel["snapshots"]
        ),
        "all_four_occupancy_strata_present": bool(panel["occupancy_counts"])
        and set(panel["occupancy_counts"])
        == {str(value) for value in TARGET_OCCUPANCIES},
        "inbound_treatment_is_explicit_and_physical_queue_cells_are_retained": all(
            snapshot["stored_recovery_workload_only"] is True
            and snapshot[
                "current_inbound_retained_in_fixed_obstacle_envelope"
            ]
            for snapshot in panel["snapshots"]
        ),
        "all_queries_fail_closed_on_unknown": all(
            not row["certificate"][method]["state"]["admitted"]
            for row in rows
            for method in ("nominal", "one_step", "recursive")
            if row["certificate"][method]["state"]["status"]
            == SolveStatus.UNKNOWN.value
        ),
        "lambda_never_changes_certified_set": all(
            row["certified_set_independent_of_lambda"] for row in rows
        ),
    }
    if not all(checks.values()):
        raise BridgeError(f"bridge checks failed: {checks!r}")
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "passed",
        "contract_sha256": contract["contract_sha256"],
        "panel_sha256": panel["panel_sha256"],
        "source_episode_instances": 30,
        "snapshot_count": len(rows),
        "occupancy_counts": panel["occupancy_counts"],
        "uncertainty_contract": UNCERTAINTY_CONTRACT,
        "completion_budget_formula": contract["completion_budget"],
        "max_expansions_per_method_per_snapshot": MAX_EXPANSIONS,
        "method_summary": summaries,
        "method_summary_by_occupancy": summaries_by_occupancy,
        "assigned_target_substitution_count": sum(
            row["assigned_target_occupancy"] != row["target_occupancy"]
            for row in rows
        ),
        "snapshots_with_current_unaccepted_inbound": sum(
            snapshot["inbound_label"] is not None
            for snapshot in panel["snapshots"]
        ),
        "recursive_root_selection_changed_between_lambda_0_and_0p2": (
            lambda_changes
        ),
        "checks": checks,
        "claim": (
            "exact bounded-disturbance closed-admission robust recovery for "
            "WINNING snapshots/actions; deterministic-count-budget UNKNOWN "
            "otherwise"
        ),
        "limitations": {
            "full_dynamic_episode_robustness": False,
            "future_arrivals_modeled": False,
            "current_unaccepted_inbound_is_excluded_from_recovery_workload": (
                True
            ),
            "block_timing_advanced_inside_recovery_abstraction": False,
            "authoritative_environment_disturbance_injection": False,
            "hardware_validation": False,
            "root_preferences_use_frozen_historical_q_inputs": True,
            "root_preference_is_candidate_level_scalar_diagnostic_not_full_"
            "mode_aggregated_policy": True,
        },
        "training_or_learning": False,
        "checkpoint_or_lambda_selection": False,
        "rows": rows,
    }
    report["report_sha256"] = _digest(report)
    report_path = output_dir.resolve() / REPORT_NAME
    if report_path.exists():
        observed = _read(report_path)
        _verify_self_hash(observed, "report_sha256", label="bridge report")
        if observed != report:
            raise BridgeError("existing bridge report changed")
    else:
        _atomic_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "evaluate", "run"), nargs="?", default="run"
    )
    parser.add_argument("--project-root", type=Path, default=HERE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    if args.command == "prepare":
        result = prepare_panel(project_root, output_dir)
        summary = {
            "status": "prepared",
            "snapshot_count": result["snapshot_count"],
            "output": str((output_dir / PANEL_NAME).resolve()),
        }
    elif args.command == "evaluate":
        result = evaluate_panel(project_root, output_dir)
        summary = {
            "status": result["status"],
            "snapshot_count": result["snapshot_count"],
            "output": str((output_dir / REPORT_NAME).resolve()),
        }
    else:
        prepare_panel(project_root, output_dir)
        result = evaluate_panel(project_root, output_dir)
        summary = {
            "status": result["status"],
            "snapshot_count": result["snapshot_count"],
            "output": str((output_dir / REPORT_NAME).resolve()),
        }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
