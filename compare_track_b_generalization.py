#!/usr/bin/env python3
"""Paired complete-system evaluation over sealed Track-B regime suites."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from example.Options.selector_v5 import ExplicitCellAssignmentRegistry
from example.controller_observation import OnlineManifestTimingObservationEncoder
from example.controller_options import build_controller_options
from example.track_b_regimes import (
    BUILTIN_REGIME_SUITES,
    canonical_manifest,
    manifest_regimes,
)
from example.yard_geometry import geometry_metadata
from fully_learned_hierarchy import (
    FULLY_LEARNED_ACTION_INTERFACE,
    FULLY_LEARNED_CHECKPOINT_SCHEMA_VERSION,
    FullyLearnedConfig,
    FullyLearnedHierarchyAgent,
    validate_fully_learned_checkpoint_metadata,
)
from PSLAP.reg_selector_v5 import REGV5AssignmentSource
from PSLAP.checkpoint_identity import selector_deployment_digest
from PSLAP.track_a import (
    TRACK_A_DYNAMIC,
    TRACK_A_NEAREST_FREE,
    TRACK_A_REG_SELECTOR_V5,
)
from track_b_urgency_evaluate import (
    DECISION_EPOCH_RESERVED,
    evaluate_assignment_ablation_one,
)
from train_fully_learned_track_b import (
    CURRICULUM_CONTRACT,
    FAILURE_CONTRACT,
    FAILURE_PENALTY_CONTRACT,
    FULLY_LEARNED_TRAINER_VERSION,
    MODEL_SELECTION_CONTRACT,
    RAW_MACRO_RETURN_CONTRACT,
    TRUNCATION_CONTRACT,
    json_safe,
    resolve_device,
    run_episode,
    seed_everything,
)
from train_fully_learned_track_b_mixed import (
    BASE_PROVENANCE_FIELDS,
    MIXED_BASE_DEPLOYMENT_CONTRACT,
    MIXED_CHECKPOINT_SCHEMA_VERSION,
    MIXED_CURRICULUM,
    MIXED_INSTANCE_SEED_CONTRACT,
    MIXED_METHOD,
    MIXED_MODEL_SELECTION_CONTRACT,
    MIXED_REPLAY_CONTRACT,
    MIXED_SAMPLER_CONTRACT,
    MIXED_TARGET_SYNC_CONTRACT,
    MIXED_TRAINER_VERSION,
    MIXED_TRANSFER_CONTRACT,
    _json_digest,
    _sha256,
    base_deployment_provenance,
)
from track_b_fully_learned_evaluate import validate_deployment_payload


FULL_METHOD = "fully_learned_mixed"
ZERO_SHOT_METHOD = "fully_learned_v4_1_zero_shot"
BASELINE_METHODS = (
    TRACK_A_REG_SELECTOR_V5,
    TRACK_A_DYNAMIC,
    TRACK_A_NEAREST_FREE,
)
DEFAULT_METHODS = (FULL_METHOD, *BASELINE_METHODS)
ALL_METHODS = (FULL_METHOD, ZERO_SHOT_METHOD, *BASELINE_METHODS)
PROTOCOL_VERSION = "paired_complete_system_mixed_generalization_v3"
GENERALIZATION_AXES = ("load", "grid_size", "exit_width", "block_count")
SEALED_EVALUATION_SUITES = frozenset(("mixed_ood_v1", "stress_v1"))
STRESS_STAGE_SEEDS = {
    "calibration": (68000, 68001, 68002),
    "holdout": tuple(range(69000, 69010)),
}
STRESS_ACTIVATION_CONTRACT = (
    "dynamic_reference_strict_complete_peak50_candidate3_relocation_v1"
)
ZERO_SHOT_AUTH_CONTRACT = (
    "exact_mixed_warm_start_file_sha256_plus_semantic_provenance_v1"
)
ZERO_SHOT_SHIFT_CONTRACT = (
    "sealed_suite_axes_within_shared_open_yard_geometry_contract_v1"
)


def validate_mixed_checkpoint(payload):
    expected = {
        "mixed_checkpoint_schema_version": MIXED_CHECKPOINT_SCHEMA_VERSION,
        "mixed_trainer_version": MIXED_TRAINER_VERSION,
        "method": MIXED_METHOD,
        "checkpoint_kind": "deployment_best",
        "resumable": False,
        "curriculum_contract": MIXED_CURRICULUM,
        "replay_sampling_contract": MIXED_REPLAY_CONTRACT,
        "regime_sampler_contract": MIXED_SAMPLER_CONTRACT,
        "instance_seed_contract": MIXED_INSTANCE_SEED_CONTRACT,
        "transfer_contract": MIXED_TRANSFER_CONTRACT,
        "warm_start_target_sync": MIXED_TARGET_SYNC_CONTRACT,
        "model_selection_contract": MIXED_MODEL_SELECTION_CONTRACT,
        "macro_return_contract": RAW_MACRO_RETURN_CONTRACT,
        "truncation_contract": TRUNCATION_CONTRACT,
        "failure_contract": FAILURE_CONTRACT,
        "failure_penalty_contract": FAILURE_PENALTY_CONTRACT,
    }
    mismatches = {
        key: {"expected": value, "found": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    validate_fully_learned_checkpoint_metadata(payload)
    if not isinstance(payload.get("selector_checkpoint"), dict):
        mismatches["selector_checkpoint"] = "missing"
    if not isinstance(payload.get("agent_checkpoint_state"), dict):
        mismatches["agent_checkpoint_state"] = "missing"
    else:
        state = payload["agent_checkpoint_state"]
        if not isinstance(state.get("Q_local"), dict):
            mismatches["agent_Q_local"] = "missing"
        if not isinstance(state.get("Q_target"), dict):
            mismatches["agent_Q_target"] = "missing"
        if state.get("training_phase") != "joint":
            mismatches["agent_training_phase"] = state.get(
                "training_phase"
            )
        forbidden = set(state).intersection(
            {"optimizer", "replay", "rng_state"}
        )
        if forbidden:
            mismatches["deployment_state"] = {
                "unexpected_resumable_fields": sorted(forbidden)
            }
    selector_payload = payload.get("selector_checkpoint")
    if isinstance(selector_payload, dict):
        try:
            digest = selector_deployment_digest(selector_payload)
        except (KeyError, TypeError, ValueError) as exc:
            mismatches["selector_checkpoint"] = f"invalid: {exc}"
        else:
            if digest != payload.get("selector_deployment_digest"):
                mismatches["selector_deployment_digest"] = "mismatch"
    base_provenance = payload.get("base_deployment_provenance")
    base_path = payload.get("base_checkpoint")
    if not isinstance(base_path, str) or not base_path:
        mismatches["base_checkpoint"] = "missing"
    base_sha256 = payload.get("base_checkpoint_sha256")
    if (
        not isinstance(base_sha256, str)
        or len(base_sha256) != 64
        or any(character not in "0123456789abcdef" for character in base_sha256)
    ):
        mismatches["base_checkpoint_sha256"] = "invalid_or_missing"
    if not isinstance(base_provenance, dict):
        mismatches["base_deployment_provenance"] = "missing"
    else:
        if base_provenance.get("contract") != MIXED_BASE_DEPLOYMENT_CONTRACT:
            mismatches["base_deployment_contract"] = {
                "expected": MIXED_BASE_DEPLOYMENT_CONTRACT,
                "found": base_provenance.get("contract"),
            }
        fields = base_provenance.get("fields")
        if not isinstance(fields, dict):
            mismatches["base_deployment_fields"] = "missing"
        elif base_provenance.get("sha256") != _json_digest(fields):
            mismatches["base_deployment_sha256"] = "mismatch"
        else:
            if set(fields) != set(BASE_PROVENANCE_FIELDS):
                mismatches["base_deployment_field_set"] = {
                    "missing": sorted(set(BASE_PROVENANCE_FIELDS) - set(fields)),
                    "unknown": sorted(set(fields) - set(BASE_PROVENANCE_FIELDS)),
                }
            base_expected = {
                "fully_learned_checkpoint_schema_version": (
                    FULLY_LEARNED_CHECKPOINT_SCHEMA_VERSION
                ),
                "fully_learned_trainer_version": (
                    FULLY_LEARNED_TRAINER_VERSION
                ),
                "curriculum_contract": CURRICULUM_CONTRACT,
                "checkpoint_kind": "deployment_best",
                "resumable": False,
                "macro_return_contract": RAW_MACRO_RETURN_CONTRACT,
                "truncation_contract": TRUNCATION_CONTRACT,
                "failure_contract": FAILURE_CONTRACT,
                "failure_penalty_contract": FAILURE_PENALTY_CONTRACT,
                "model_selection_contract": MODEL_SELECTION_CONTRACT,
                "controller_action_interface": FULLY_LEARNED_ACTION_INTERFACE,
                "common_continuation": True,
                "deployment_reg_policy_query": False,
                "selector_frozen_independent_replay_disabled": True,
                "selector_deployment_digest": payload.get(
                    "selector_deployment_digest"
                ),
            }
            base_mismatches = {
                key: {"expected": value, "found": fields.get(key)}
                for key, value in base_expected.items()
                if fields.get(key) != value
            }
            anchor = BUILTIN_REGIME_SUITES["mixed_train_v1"][0]
            anchor_geometry = anchor.provenance()["geometry"]
            geometry = fields.get("geometry")
            anchor_expected = {
                "training_lambda": anchor.arrival_rate,
                "training_mu": anchor.proc_mean,
            }
            base_mismatches.update(
                {
                    key: {"expected": value, "found": fields.get(key)}
                    for key, value in anchor_expected.items()
                    if fields.get(key) != value
                }
            )
            if payload.get("base_training_seed") != fields.get(
                "training_seed"
            ):
                base_mismatches["base_training_seed"] = {
                    "expected": fields.get("training_seed"),
                    "found": payload.get("base_training_seed"),
                }
            if not isinstance(geometry, dict):
                base_mismatches["geometry"] = "missing"
            else:
                for key in (
                    "geometry_signature",
                    "grid_rows",
                    "grid_cols",
                    "requested_exit_width",
                    "block_count",
                ):
                    if geometry.get(key) != anchor_geometry[key]:
                        base_mismatches[f"geometry.{key}"] = {
                            "expected": anchor_geometry[key],
                            "found": geometry.get(key),
                        }
            if base_mismatches:
                mismatches["base_deployment_fields"] = base_mismatches
    if not isinstance(payload.get("fully_learned_config"), dict):
        mismatches["fully_learned_config"] = "missing"
    joint_expected = {
        "training_phase": "joint",
        "common_continuation": True,
        "teacher_coefficient": 0.0,
        "accept_policy_lock": False,
        "spatial_trainable": True,
        "frozen_reg_runtime_policy_query": False,
        "deployment_reg_policy_query": False,
        "replay_sampling": "regime_mode_balanced",
    }
    mismatches.update(
        {
            key: {"expected": value, "found": payload.get(key)}
            for key, value in joint_expected.items()
            if payload.get(key) != value
        }
    )
    try:
        regimes = manifest_regimes(payload["training_manifest"])
    except (KeyError, TypeError, ValueError) as exc:
        mismatches["training_manifest"] = f"invalid: {exc}"
    else:
        expected_manifest = canonical_manifest(
            BUILTIN_REGIME_SUITES["mixed_train_v1"]
        )
        if payload["training_manifest"] != expected_manifest:
            mismatches["training_manifest"] = (
                "does not match the sealed mixed_train_v1 suite"
            )
        if not regimes:
            mismatches["training_manifest"] = "empty"
    if mismatches:
        raise ValueError(f"invalid mixed deployment checkpoint: {mismatches}")


def validate_zero_shot_checkpoint(payload, mixed_payload, checkpoint_path=None):
    """Authenticate the explicit v4.1 comparator used to warm-start mixed RL.

    Semantic provenance prevents a merely compatible v4.1 artifact from being
    mislabeled as the mixed run's pre-adaptation control.  When a path is
    supplied, the byte-level digest additionally proves it is the exact source
    checkpoint recorded by the mixed deployment artifact.
    """

    validate_deployment_payload(payload)
    mismatches = {}
    try:
        actual_provenance = base_deployment_provenance(payload)
    except (KeyError, TypeError, ValueError) as exc:
        mismatches["base_deployment_provenance"] = f"invalid: {exc}"
    else:
        expected_provenance = mixed_payload.get("base_deployment_provenance")
        if actual_provenance != expected_provenance:
            mismatches["base_deployment_provenance"] = (
                "does_not_match_mixed_warm_start"
            )
    selector_payload = payload.get("selector_checkpoint")
    if not isinstance(selector_payload, dict):
        mismatches["selector_checkpoint"] = "missing"
    else:
        try:
            digest = selector_deployment_digest(selector_payload)
        except (KeyError, TypeError, ValueError) as exc:
            mismatches["selector_checkpoint"] = f"invalid: {exc}"
        else:
            expected_digest = mixed_payload.get("selector_deployment_digest")
            if digest != payload.get("selector_deployment_digest"):
                mismatches["selector_deployment_digest"] = (
                    "does_not_match_embedded_selector"
                )
            elif digest != expected_digest:
                mismatches["selector_deployment_digest"] = (
                    "does_not_match_mixed_checkpoint"
                )
    if payload.get("training_seed") != mixed_payload.get("base_training_seed"):
        mismatches["training_seed"] = {
            "expected": mixed_payload.get("base_training_seed"),
            "found": payload.get("training_seed"),
        }
    if checkpoint_path is not None:
        found_sha256 = _sha256(Path(checkpoint_path))
        expected_sha256 = mixed_payload.get("base_checkpoint_sha256")
        if found_sha256 != expected_sha256:
            mismatches["checkpoint_sha256"] = {
                "expected": expected_sha256,
                "found": found_sha256,
            }
    if mismatches:
        raise ValueError(
            f"invalid authenticated v4.1 zero-shot checkpoint: {mismatches}"
        )


def zero_shot_shift_axes(payload, regime):
    """Return audited source-to-target shifts within the supported yard family."""

    source_geometry = payload.get("geometry")
    if not isinstance(source_geometry, dict):
        raise ValueError("v4.1 zero-shot checkpoint geometry is missing")
    target_geometry = regime.provenance()["geometry"]
    source_contract = source_geometry.get("geometry_contract")
    target_contract = target_geometry.get("geometry_contract")
    if source_contract != target_contract:
        raise ValueError(
            "zero-shot geometry contract mismatch: "
            f"source={source_contract!r}, target={target_contract!r}"
        )
    axes = []
    if (
        payload.get("training_lambda") != regime.arrival_rate
        or payload.get("training_mu") != regime.proc_mean
    ):
        axes.append("load")
    if (
        source_geometry.get("grid_rows") != regime.grid_rows
        or source_geometry.get("grid_cols") != regime.grid_cols
    ):
        axes.append("grid_size")
    if source_geometry.get("actual_exit_width") != target_geometry.get(
        "actual_exit_width"
    ):
        axes.append("exit_width")
    if source_geometry.get("block_count") != regime.number_blocks:
        axes.append("block_count")
    unknown = set(axes) - set(GENERALIZATION_AXES)
    if unknown:
        raise ValueError(f"unsupported zero-shot shift axes: {sorted(unknown)}")
    return tuple(axes)


def validate_runtime_geometry(env, regime):
    """Prove the constructed runtime is the canonical manifest geometry."""

    expected = regime.provenance()["geometry"]
    actual = geometry_metadata(env, requested_exit_width=regime.exit_width)
    fields = (
        "geometry_contract",
        "geometry_signature",
        "grid_rows",
        "grid_cols",
        "requested_exit_width",
        "actual_exit_width",
        "block_count",
        "storage_cell_count",
        "nominal_storage_density",
        "room_rows",
    )
    mismatches = {
        key: {"expected": expected.get(key), "found": actual.get(key)}
        for key in fields
        if actual.get(key) != expected.get(key)
    }
    if mismatches:
        raise ValueError(
            f"runtime geometry does not match regime {regime.regime_id!r}: "
            f"{mismatches}"
        )
    return actual


def target_training_overlap(training_manifest, regimes):
    """Return exact profile signatures shared by training and target suites."""

    train_signatures = {
        item.regime_signature for item in manifest_regimes(training_manifest)
    }
    target_signatures = {item.regime_signature for item in regimes}
    return sorted(train_signatures.intersection(target_signatures))


def validate_suite_separation(suite_name, training_manifest, regimes):
    """Enforce each sealed target-suite promise before any episode runs."""

    overlap = target_training_overlap(training_manifest, regimes)
    if suite_name in SEALED_EVALUATION_SUITES and overlap:
        raise ValueError(
            f"sealed {suite_name} suite overlaps training: {overlap}"
        )
    return overlap


def build_runtime(regime, selector_payload, *, device, seed):
    env = regime.make_env()
    validate_runtime_geometry(env, regime)
    source = REGV5AssignmentSource.from_checkpoint(
        env,
        selector_payload,
        learning_enabled=False,
        device=device,
        seed=seed,
    )
    selector = ExplicitCellAssignmentRegistry(env, source)
    build_controller_options(
        env,
        selector,
        controller_action_interface=FULLY_LEARNED_ACTION_INTERFACE,
        max_defer_steps=10,
    )
    env.reset(instance=env.sample_episode_instance(0))
    return env, selector, OnlineManifestTimingObservationEncoder(env), source


def build_agent(payload, first_runtime, *, device):
    env, _, encoder, source = first_runtime
    config = FullyLearnedConfig.from_dict(payload["fully_learned_config"])
    agent = FullyLearnedHierarchyAgent(
        env,
        encoder,
        config=config,
        spatial_network=deepcopy(source.network),
        seed=int(payload["training_seed"]),
        device=device,
        gamma=float(payload["gamma"]),
        reward_scale=float(payload["reward_scale"]),
        epsilon=0.0,
        replay_sampling=payload.get("replay_sampling", "mode_balanced"),
    )
    agent.load_checkpoint_state(
        payload["agent_checkpoint_state"], resumable=False
    )
    agent.epsilon = 0.0
    agent.set_training_phase("joint", clear_replay=False)
    agent.Q_local.eval()
    agent.Q_target.eval()
    return agent


def _safe_ratio(numerator, denominator):
    denominator = float(denominator)
    return (
        None
        if denominator <= 0.0
        else float(float(numerator) / denominator)
    )


def _optional_mean(values):
    usable = [float(value) for value in values if value is not None]
    return float(np.mean(usable)) if usable else None


def _optional_min(values):
    usable = [float(value) for value in values if value is not None]
    return float(min(usable)) if usable else None


def _optional_max(values):
    usable = [float(value) for value in values if value is not None]
    return float(max(usable)) if usable else None


def compact(run, regime):
    episode_return = float(run["return"])
    steps = int(run["steps"])
    delivery_count = int(run["delivery_count"])
    manifest_block_count = int(regime.number_blocks)
    storage_capacity_cells = int(
        regime.provenance()["geometry"]["storage_cell_count"]
    )
    relocations = int(run["obstructive_moves"])
    result = {
        "return": episode_return,
        "strict_method_success": float(run["strict_method_success"]),
        "reservation_integrity": float(run["reservation_integrity"]),
        "steps": steps,
        "delivery_count": delivery_count,
        "manifest_block_count": manifest_block_count,
        "storage_capacity_cells": storage_capacity_cells,
        "completion_rate": _safe_ratio(
            delivery_count, manifest_block_count
        ),
        "return_per_manifest_block": _safe_ratio(
            episode_return, manifest_block_count
        ),
        "return_per_delivery": _safe_ratio(
            episode_return, delivery_count
        ),
        "steps_per_delivery": _safe_ratio(steps, delivery_count),
        "mean_absolute_error": float(run["mean_absolute_error"]),
        "mean_tardiness": float(run["mean_tardiness"]),
        "within_target_window_rate": float(
            run["within_target_window_rate"]
        ),
        "relocations": relocations,
        "relocations_per_100_deliveries": (
            None
            if delivery_count <= 0
            else float(100.0 * relocations / delivery_count)
        ),
        "invalid_assignments": int(
            run["selector_audit"].get("invalid_assignment_count", 0)
        ),
        "fallbacks": int(run["selector_audit"].get("fallback_count", 0)),
        "method_failure_reason": run["method_failure_reason"],
    }
    for field in (
        "occupancy_pressure_metric_contract",
        "peak_active_stored_count",
        "peak_active_stored_fraction",
        "peak_physical_storage_occupancy_count",
        "peak_physical_storage_occupancy_fraction",
        "minimum_free_storage_cells",
        "storage_pressure_observed_steps",
        "storage_steps_at_or_above_80pct_occupied",
        "storage_step_fraction_at_or_above_80pct_occupied",
        "maximum_inbound_waiting_count",
        "maximum_inbound_queue_count",
        "minimum_live_candidate_count",
        "minimum_live_accept_candidate_count",
    ):
        result[field] = run.get(field)
    reported_capacity = run.get("storage_capacity_cells")
    if (
        reported_capacity is not None
        and int(reported_capacity) != storage_capacity_cells
    ):
        raise ValueError(
            "episode occupancy capacity does not match regime geometry: "
            f"reported={reported_capacity}, expected={storage_capacity_cells}"
        )
    reported_pressure_steps = run.get("storage_pressure_observed_steps")
    if (
        reported_pressure_steps is not None
        and int(reported_pressure_steps) != steps
    ):
        raise ValueError(
            "occupancy pressure must be sampled once per completed "
            f"transition: reported={reported_pressure_steps}, steps={steps}"
        )
    return result


def summarize(runs, regimes, methods):
    regimes = tuple(regimes)
    methods = tuple(methods)
    regime_ids = [item.regime_id for item in regimes]
    if not regime_ids or not methods:
        raise ValueError("summary requires nonempty regimes and methods")
    if len(regime_ids) != len(set(regime_ids)):
        raise ValueError("summary regime ids must be unique")
    expected_cells = {
        (regime_id, method)
        for regime_id in regime_ids
        for method in methods
    }
    grouped = {}
    for item in runs:
        key = (item.get("regime_id"), item.get("method"))
        if key not in expected_cells:
            raise ValueError(f"unexpected run cell: {key}")
        eval_seed = item.get("eval_seed")
        if eval_seed in grouped.setdefault(key, {}):
            raise ValueError(
                f"duplicate run for cell {key}, seed {eval_seed}"
            )
        grouped[key][eval_seed] = item
    missing = sorted(expected_cells - set(grouped))
    if missing:
        raise ValueError(f"missing run cells: {missing}")
    expected_seeds = None
    for key in sorted(expected_cells):
        seeds = set(grouped[key])
        if expected_seeds is None:
            expected_seeds = seeds
        elif seeds != expected_seeds:
            raise ValueError(
                "methods/regimes were not evaluated on one complete paired "
                f"instance matrix; cell={key}"
            )
    for regime_id in regime_ids:
        for eval_seed in sorted(expected_seeds):
            matched = [
                grouped[(regime_id, method)][eval_seed]
                for method in methods
            ]
            instance_ids = {item.get("instance_id") for item in matched}
            schedule_ids = {item.get("schedule_id") for item in matched}
            if len(instance_ids) != 1 or None in instance_ids:
                raise ValueError(
                    "paired method instance mismatch for "
                    f"regime={regime_id}, seed={eval_seed}"
                )
            if len(schedule_ids) != 1 or None in schedule_ids:
                raise ValueError(
                    "paired method schedule mismatch for "
                    f"regime={regime_id}, seed={eval_seed}"
                )

    cells = []
    for regime in regimes:
        for method in methods:
            selected = [
                item
                for item in runs
                if item["regime_id"] == regime.regime_id
                and item["method"] == method
            ]
            pressure_contracts = {
                item.get("occupancy_pressure_metric_contract")
                for item in selected
            }
            if len(pressure_contracts) != 1 or None in pressure_contracts:
                raise ValueError(
                    "missing or inconsistent occupancy-pressure contract for "
                    f"regime={regime.regime_id}, method={method}: "
                    f"{pressure_contracts}"
                )
            cells.append(
                {
                    "regime_id": regime.regime_id,
                    "method": method,
                    "episodes": len(selected),
                    "manifest_block_count": int(regime.number_blocks),
                    "storage_capacity_cells": int(
                        regime.provenance()["geometry"][
                            "storage_cell_count"
                        ]
                    ),
                    "mean_return": float(
                        np.mean([item["return"] for item in selected])
                    ),
                    "return_std": float(
                        np.std(
                            [item["return"] for item in selected],
                            ddof=1,
                        )
                    ) if len(selected) > 1 else 0.0,
                    "mean_return_per_manifest_block": float(
                        np.mean(
                            [
                                item["return_per_manifest_block"]
                                for item in selected
                            ]
                        )
                    ),
                    "mean_return_per_delivery": _optional_mean(
                        [item["return_per_delivery"] for item in selected]
                    ),
                    "mean_delivery_count": float(
                        np.mean(
                            [item["delivery_count"] for item in selected]
                        )
                    ),
                    "mean_completion_rate": float(
                        np.mean(
                            [item["completion_rate"] for item in selected]
                        )
                    ),
                    "strict_method_success_rate": float(
                        np.mean(
                            [
                                item["strict_method_success"]
                                for item in selected
                            ]
                        )
                    ),
                    "reservation_integrity_rate": float(
                        np.mean(
                            [item["reservation_integrity"] for item in selected]
                        )
                    ),
                    "mean_absolute_error": float(
                        np.mean(
                            [item["mean_absolute_error"] for item in selected]
                        )
                    ),
                    "mean_tardiness": float(
                        np.mean([item["mean_tardiness"] for item in selected])
                    ),
                    "within_target_window_rate": float(
                        np.mean(
                            [
                                item["within_target_window_rate"]
                                for item in selected
                            ]
                        )
                    ),
                    "mean_steps": float(
                        np.mean([item["steps"] for item in selected])
                    ),
                    "mean_steps_per_delivery": _optional_mean(
                        [item["steps_per_delivery"] for item in selected]
                    ),
                    "total_relocations": int(
                        sum(item["relocations"] for item in selected)
                    ),
                    "mean_relocations_per_100_deliveries": _optional_mean(
                        [
                            item["relocations_per_100_deliveries"]
                            for item in selected
                        ]
                    ),
                    "episodes_with_relocation": int(
                        sum(item["relocations"] > 0 for item in selected)
                    ),
                    "occupancy_pressure_metric_contract": next(
                        iter(pressure_contracts)
                    ),
                    "mean_peak_active_stored_fraction": _optional_mean(
                        [
                            item["peak_active_stored_fraction"]
                            for item in selected
                        ]
                    ),
                    "mean_peak_physical_storage_occupancy_fraction": (
                        _optional_mean(
                            [
                                item[
                                    "peak_physical_storage_occupancy_fraction"
                                ]
                                for item in selected
                            ]
                        )
                    ),
                    "maximum_peak_physical_storage_occupancy_fraction": (
                        _optional_max(
                            [
                                item[
                                    "peak_physical_storage_occupancy_fraction"
                                ]
                                for item in selected
                            ]
                        )
                    ),
                    "minimum_free_storage_cells": _optional_min(
                        [
                            item["minimum_free_storage_cells"]
                            for item in selected
                        ]
                    ),
                    "mean_storage_step_fraction_at_or_above_80pct_occupied": (
                        _optional_mean(
                            [
                                item[
                                    "storage_step_fraction_at_or_above_80pct_occupied"
                                ]
                                for item in selected
                            ]
                        )
                    ),
                    "episodes_at_or_above_80pct_occupied": int(
                        sum(
                            int(
                                item[
                                    "storage_steps_at_or_above_80pct_occupied"
                                ]
                                or 0
                            )
                            > 0
                            for item in selected
                        )
                    ),
                    "mean_maximum_inbound_queue_count": _optional_mean(
                        [
                            item["maximum_inbound_queue_count"]
                            for item in selected
                        ]
                    ),
                    "minimum_live_candidate_count": _optional_min(
                        [
                            item["minimum_live_candidate_count"]
                            for item in selected
                        ]
                    ),
                    "minimum_live_accept_candidate_count": _optional_min(
                        [
                            item["minimum_live_accept_candidate_count"]
                            for item in selected
                        ]
                    ),
                    "total_invalid_assignments": int(
                        sum(item["invalid_assignments"] for item in selected)
                    ),
                    "total_fallbacks": int(
                        sum(item["fallbacks"] for item in selected)
                    ),
                    "method_failures": [
                        item["method_failure_reason"]
                        for item in selected
                        if item["method_failure_reason"] is not None
                    ],
                }
            )
    macro = []
    for method in methods:
        selected = [item for item in cells if item["method"] == method]
        macro.append(
            {
                "method": method,
                "regime_count": len(selected),
                "macro_mean_return": float(
                    np.mean([item["mean_return"] for item in selected])
                ),
                "worst_regime_return": float(
                    min(item["mean_return"] for item in selected)
                ),
                "macro_mean_return_per_manifest_block": float(
                    np.mean(
                        [
                            item["mean_return_per_manifest_block"]
                            for item in selected
                        ]
                    )
                ),
                "worst_regime_return_per_manifest_block": float(
                    min(
                        item["mean_return_per_manifest_block"]
                        for item in selected
                    )
                ),
                "macro_mean_completion_rate": float(
                    np.mean(
                        [item["mean_completion_rate"] for item in selected]
                    )
                ),
                "macro_mean_steps_per_delivery": _optional_mean(
                    [item["mean_steps_per_delivery"] for item in selected]
                ),
                "macro_mean_relocations_per_100_deliveries": _optional_mean(
                    [
                        item["mean_relocations_per_100_deliveries"]
                        for item in selected
                    ]
                ),
                "macro_mean_absolute_error": float(
                    np.mean([item["mean_absolute_error"] for item in selected])
                ),
                "worst_regime_absolute_error": float(
                    max(item["mean_absolute_error"] for item in selected)
                ),
                "minimum_strict_success_rate": float(
                    min(
                        item["strict_method_success_rate"] for item in selected
                    )
                ),
                "total_relocations": int(
                    sum(item["total_relocations"] for item in selected)
                ),
            }
        )
    paired = []
    if FULL_METHOD not in methods:
        return cells, macro, paired
    learned = {
        (item["regime_id"], item["eval_seed"]): item
        for item in runs
        if item["method"] == FULL_METHOD
    }
    for method in methods:
        if method == FULL_METHOD:
            continue
        deltas = []
        for item in runs:
            if item["method"] != method:
                continue
            left = learned[(item["regime_id"], item["eval_seed"])]
            if left["instance_id"] != item["instance_id"]:
                raise RuntimeError("paired method instance mismatch")
            if left.get("schedule_id") != item.get("schedule_id"):
                raise RuntimeError("paired method schedule mismatch")
            deltas.append(
                {
                    "regime_id": item["regime_id"],
                    "eval_seed": item["eval_seed"],
                    "instance_id": item["instance_id"],
                    "return_advantage": left["return"] - item["return"],
                    "return_per_manifest_block_advantage": (
                        left["return_per_manifest_block"]
                        - item["return_per_manifest_block"]
                    ),
                    "completion_rate_advantage": (
                        left["completion_rate"] - item["completion_rate"]
                    ),
                    "steps_per_delivery_reduction": (
                        None
                        if left["steps_per_delivery"] is None
                        or item["steps_per_delivery"] is None
                        else item["steps_per_delivery"]
                        - left["steps_per_delivery"]
                    ),
                    "absolute_error_reduction": (
                        item["mean_absolute_error"]
                        - left["mean_absolute_error"]
                    ),
                    "relocation_reduction": (
                        item["relocations"] - left["relocations"]
                    ),
                    "relocations_per_100_deliveries_reduction": (
                        None
                        if left["relocations_per_100_deliveries"] is None
                        or item["relocations_per_100_deliveries"] is None
                        else item["relocations_per_100_deliveries"]
                        - left["relocations_per_100_deliveries"]
                    ),
                }
            )
        paired.append(
            {
                "comparison_method": method,
                "n": len(deltas),
                "mean_return_advantage": float(
                    np.mean([item["return_advantage"] for item in deltas])
                ),
                "mean_return_per_manifest_block_advantage": float(
                    np.mean(
                        [
                            item["return_per_manifest_block_advantage"]
                            for item in deltas
                        ]
                    )
                ),
                "mean_completion_rate_advantage": float(
                    np.mean(
                        [
                            item["completion_rate_advantage"]
                            for item in deltas
                        ]
                    )
                ),
                "mean_steps_per_delivery_reduction": _optional_mean(
                    [
                        item["steps_per_delivery_reduction"]
                        for item in deltas
                    ]
                ),
                "mean_absolute_error_reduction": float(
                    np.mean(
                        [item["absolute_error_reduction"] for item in deltas]
                    )
                ),
                "total_relocation_reduction": int(
                    sum(item["relocation_reduction"] for item in deltas)
                ),
                "mean_relocations_per_100_deliveries_reduction": (
                    _optional_mean(
                        [
                            item[
                                "relocations_per_100_deliveries_reduction"
                            ]
                            for item in deltas
                        ]
                    )
                ),
                "per_instance": deltas,
            }
        )
    return cells, macro, paired


def audit_stress_activation(runs, regimes):
    """Audit pressure/integrity without consulting return or timing quality."""

    per_regime = []
    for regime in regimes:
        selected = [
            item
            for item in runs
            if item["regime_id"] == regime.regime_id
            and item["method"] == TRACK_A_DYNAMIC
        ]
        if not selected:
            raise ValueError(
                "stress activation requires the dynamic PSLAP reference in "
                f"regime {regime.regime_id}"
            )
        candidate_counts = [
            item["minimum_live_accept_candidate_count"] for item in selected
        ]
        criteria = {
            "strict_completion_every_episode": all(
                item["strict_method_success"] == 1.0
                and item["completion_rate"] == 1.0
                for item in selected
            ),
            "peak_physical_occupancy_at_least_50pct_every_episode": all(
                item["peak_physical_storage_occupancy_fraction"] >= 0.50
                for item in selected
            ),
            "minimum_accept_candidates_at_most_3_every_episode": all(
                value is not None and value <= 3
                for value in candidate_counts
            ),
            "relocation_observed_every_episode": all(
                item["relocations"] > 0 for item in selected
            ),
        }
        per_regime.append(
            {
                "regime_id": regime.regime_id,
                "episodes": len(selected),
                "passed": all(criteria.values()),
                "criteria": criteria,
                "minimum_peak_physical_storage_occupancy_fraction": float(
                    min(
                        item[
                            "peak_physical_storage_occupancy_fraction"
                        ]
                        for item in selected
                    )
                ),
                "maximum_minimum_live_accept_candidate_count": (
                    None
                    if any(value is None for value in candidate_counts)
                    else int(max(candidate_counts))
                ),
                "minimum_relocations": int(
                    min(item["relocations"] for item in selected)
                ),
            }
        )
    return {
        "contract": STRESS_ACTIVATION_CONTRACT,
        "reference_method": TRACK_A_DYNAMIC,
        "performance_metrics_used_for_profile_acceptance": [],
        "passed": all(item["passed"] for item in per_regime),
        "per_regime": per_regime,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--zero-shot-checkpoint",
        type=Path,
        help=(
            "Exact authenticated v4.1 deployment checkpoint recorded as the "
            "mixed run's warm start. Supplying it activates the zero-shot "
            "complete-system comparator."
        ),
    )
    parser.add_argument(
        "--suite", choices=tuple(BUILTIN_REGIME_SUITES), required=True
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--stress-stage",
        choices=tuple(STRESS_STAGE_SEEDS),
        help=(
            "Use the predeclared stress_v1 calibration or untouched holdout "
            "seed namespace. Omit only for explicitly exploratory runs."
        ),
    )
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--target-window", type=float, default=20.0)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.stress_stage is not None:
        if args.suite != "stress_v1":
            parser.error("--stress-stage is valid only with --suite stress_v1")
        if args.seeds is not None:
            parser.error(
                "--stress-stage supplies its sealed seeds; do not pass --seeds"
            )
        args.seeds = STRESS_STAGE_SEEDS[args.stress_stage]
    else:
        if args.seeds is None:
            parser.error(
                "--seeds is required unless --stress-stage supplies them"
            )
        args.seeds = tuple(dict.fromkeys(args.seeds))
    requested_methods = (
        tuple(args.methods) if args.methods is not None else DEFAULT_METHODS
    )
    if args.zero_shot_checkpoint is not None:
        requested_methods = (*requested_methods, ZERO_SHOT_METHOD)
    elif ZERO_SHOT_METHOD in requested_methods:
        parser.error(
            f"method {ZERO_SHOT_METHOD!r} requires --zero-shot-checkpoint"
        )
    args.methods = tuple(dict.fromkeys(requested_methods))
    if (
        args.stress_stage is not None
        and TRACK_A_DYNAMIC not in args.methods
    ):
        parser.error(
            "sealed stress stages require dynamic_pslap for the fixed "
            "activation audit"
        )
    if not args.seeds or args.max_steps <= 0:
        parser.error("seeds and max-steps must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be absent or empty")
    return args


def main(argv=None):
    args = parse_args(argv)
    payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must be a mapping")
    validate_mixed_checkpoint(payload)
    zero_shot_payload = None
    zero_shot_shift_map = None
    if args.zero_shot_checkpoint is not None:
        zero_shot_payload = torch.load(
            args.zero_shot_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(zero_shot_payload, dict):
            raise ValueError("zero-shot checkpoint must be a mapping")
        validate_zero_shot_checkpoint(
            zero_shot_payload,
            payload,
            args.zero_shot_checkpoint,
        )
    regimes = BUILTIN_REGIME_SUITES[args.suite]
    manifest = canonical_manifest(regimes)
    overlap = validate_suite_separation(
        args.suite, payload["training_manifest"], regimes
    )
    if zero_shot_payload is not None:
        zero_shot_shift_map = {
            regime.regime_id: list(
                zero_shot_shift_axes(zero_shot_payload, regime)
            )
            for regime in regimes
        }

    device = resolve_device(args.device)
    selector_payload = payload["selector_checkpoint"]
    runtimes = {}
    for index, regime in enumerate(regimes):
        runtimes[regime.regime_id] = build_runtime(
            regime,
            selector_payload,
            device=device,
            seed=int(payload["training_seed"]) * 100 + index,
        )
    first_runtime = runtimes[regimes[0].regime_id]
    agents = {}
    if FULL_METHOD in args.methods:
        agents[FULL_METHOD] = build_agent(
            payload, first_runtime, device=device
        )
    if ZERO_SHOT_METHOD in args.methods:
        agents[ZERO_SHOT_METHOD] = build_agent(
            zero_shot_payload, first_runtime, device=device
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    instance_root = args.output_dir / "instances"
    runs = []
    for regime in regimes:
        env, selector, encoder, _ = runtimes[regime.regime_id]
        for agent in agents.values():
            agent.bind_runtime(env, encoder, regime_id=regime.regime_id)
            agent.Q_local.eval()
            agent.Q_target.eval()
        regime_dir = instance_root / regime.regime_id
        regime_dir.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            seed_everything(seed)
            instance = env.sample_episode_instance(seed)
            (regime_dir / f"seed-{seed}.json").write_text(
                instance.to_json() + "\n"
            )
            for method in args.methods:
                if method in agents:
                    with torch.inference_mode():
                        raw = run_episode(
                            agents[method],
                            selector,
                            env,
                            instance,
                            max_steps=args.max_steps,
                            target_window=args.target_window,
                            training=False,
                            phase="joint",
                        )
                else:
                    runtime_args = SimpleNamespace(
                        lam=regime.arrival_rate,
                        mu=regime.proc_mean,
                        grid_rows=regime.grid_rows,
                        grid_cols=regime.grid_cols,
                        exit_width=regime.exit_width,
                        number_blocks=regime.number_blocks,
                        instance=None,
                        device=str(device),
                        max_steps=args.max_steps,
                        max_defer_steps=10,
                        lookahead_margin_steps=2.0,
                        assignment_commitment=DECISION_EPOCH_RESERVED,
                        target_window=args.target_window,
                        save_instances_dir=None,
                        allow_selector_regime_shift=True,
                    )
                    raw = evaluate_assignment_ablation_one(
                        runtime_args,
                        seed,
                        selector_payload,
                        assignment_source=method,
                        episode_instance=instance,
                    )
                result = {
                    "protocol_version": PROTOCOL_VERSION,
                    "suite": args.suite,
                    "regime_id": regime.regime_id,
                    "regime_signature": regime.regime_signature,
                    "method": method,
                    "eval_seed": int(seed),
                    "instance_id": instance.instance_id,
                    "schedule_id": instance.schedule_id,
                    **compact(raw, regime),
                }
                runs.append(result)
                print(
                    f"[{regime.regime_id}/{method}] seed={seed} "
                    f"R={result['return']:.2f} "
                    f"MAE={result['mean_absolute_error']:.3f} "
                    f"strict={result['strict_method_success']:.0f} "
                    f"reloc={result['relocations']}",
                    flush=True,
                )

    cells, macro, paired = summarize(runs, regimes, args.methods)
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "suite": args.suite,
        "suite_manifest": manifest,
        "training_manifest_sha256": payload["training_manifest"][
            "manifest_sha256"
        ],
        "target_train_signature_overlap": overlap,
        "paired_instance_contract": "same_EpisodeInstance_per_regime_seed_v1",
        "regime_aggregation": (
            "equal_weight_macro_with_manifest_normalized_primary_return_v2"
        ),
        "cross_regime_primary_return_metric": (
            "return_per_manifest_block_with_completion_rate_v1"
        ),
        "raw_return_interpretation": (
            "within_regime_only_when_manifest_block_counts_differ"
        ),
        "occupancy_sampling_contract": (
            "post_transition_physical_storage_and_decision_epoch_candidates_v1"
        ),
        "methods": list(args.methods),
        "seeds": list(args.seeds),
        "stress_stage": (
            args.stress_stage
            if args.suite == "stress_v1"
            else None
        ),
        "stress_seed_contract": (
            "predeclared_stage_namespace_v1"
            if args.stress_stage is not None
            else (
                "exploratory_unsealed"
                if args.suite == "stress_v1"
                else None
            )
        ),
        "stress_activation_contract": (
            STRESS_ACTIVATION_CONTRACT
            if args.suite == "stress_v1"
            else None
        ),
        "frozen_reg_baseline_transfer": (
            "explicit_feature_compatible_zero_shot_v1"
        ),
        "fully_learned_runtime_reg_query": False,
        "strict_no_fallback": True,
        "checkpoint": str(args.checkpoint.resolve()),
    }
    if zero_shot_payload is not None:
        protocol["zero_shot_comparator"] = {
            "method": ZERO_SHOT_METHOD,
            "authentication_contract": ZERO_SHOT_AUTH_CONTRACT,
            "geometry_shift_contract": ZERO_SHOT_SHIFT_CONTRACT,
            "checkpoint": str(args.zero_shot_checkpoint.resolve()),
            "checkpoint_sha256": _sha256(args.zero_shot_checkpoint),
            "authenticated_against_mixed_base_sha256": payload[
                "base_checkpoint_sha256"
            ],
            "base_deployment_provenance_sha256": payload[
                "base_deployment_provenance"
            ]["sha256"],
            "source_geometry": zero_shot_payload["geometry"],
            "authorized_shift_source": f"sealed_suite:{args.suite}",
            "observed_generalization_axes_by_regime": zero_shot_shift_map,
            "runtime_reg_policy_query": False,
        }
    summary = {
        "protocol": protocol,
        "cells": cells,
        "macro_summaries": macro,
        "paired_comparisons": paired,
    }
    if args.suite == "stress_v1":
        summary["stress_activation_audit"] = audit_stress_activation(
            runs, regimes
        )
    (args.output_dir / "generalization-summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n"
    )
    (args.output_dir / "generalization-runs.json").write_text(
        json.dumps(
            json_safe({"protocol": protocol, "runs": runs}),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    fields = list(runs[0])
    with (args.output_dir / "generalization-results.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(runs)
    print(json.dumps(json_safe(macro), indent=2), flush=True)
    print(json.dumps(json_safe(paired), indent=2), flush=True)


if __name__ == "__main__":
    main()
