#!/usr/bin/env python3
"""Extract one authenticated VCG decision boundary for a paper illustration.

The replay follows one persisted 87k confirmation row exactly.  At every
decision boundary it evaluates the *same* frozen two-head critic and the same
exact-safe candidate set under lambda=0 and lambda=.05.  No second action is
sampled: the counterfactual consists only of read-only policy calculations.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
from types import MethodType, SimpleNamespace
from typing import Mapping

import torch
import torch.nn.functional as F

import evaluate_vcg_unified_frozen_lambda_confirmation as confirmation
import evaluate_vcg_unified_frozen_lambda_seed_stability as development
import evaluate_vcg_unified_frozen_lambda_sweep as base
import render_vcg_unified_behavior_gifs as behavior
import train_vcg_constrained_v2_3 as trainer
import viability_graph_constrained_v2 as v2
import viability_graph_constrained_v2_3 as v23core


PROTOCOL = "vcg_unified_exact_safe_preference_probe_v1"
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    HERE
    / "results"
    / "vcg-unified-frozen-lambda-confirmation-87k-behavior-view"
    / "decision-mechanism-probe.json"
)
DEFAULT_MODEL_SEED = 16
DEFAULT_INSTANCE_SEED = 87_013
DEFAULT_RNG_INDEX = 3
LAMBDAS = (0.0, 0.05)


class DecisionProbeError(RuntimeError):
    pass


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_sha(value: Mapping) -> str:
    raw = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _probe_runtime(parent: Mapping, *, device: str):
    """Load the frozen parent, allowing a clearly labelled CPU replay.

    CUDA is the authoritative confirmation device.  The CPU path changes only
    the post-hoc replay device and is accepted only when the entire observed
    rollout still matches the persisted CUDA row exactly.
    """

    if device == str(parent["base_contract"].get("device")):
        return base._runtime(parent, device=device)
    if device != "cpu":
        raise DecisionProbeError("probe device must be the frozen device or cpu")
    contract = deepcopy(parent["base_contract"])
    contract["device"] = "cpu"
    runtime = trainer.load_default_runtime(SimpleNamespace(device="cpu"), contract)
    runtime.agent = v23core.ConstrainedV23HierarchyAgent.from_checkpoint(
        parent["checkpoint"]["agent_state"],
        device="cpu",
        resumable=False,
        seed=int(parent["base_contract"]["model_seed"]),
    )
    if runtime.agent.config.to_dict() != parent["base_contract"]["agent_config"]:
        raise DecisionProbeError("CPU probe agent configuration drifted")
    return runtime


def _policy_view(agent, prepared, q_op, q_n, operational_weight: float, lam: float):
    merits = operational_weight * q_op - float(lam) * q_n
    group_ids = torch.as_tensor(
        prepared.mode_ids, dtype=torch.long, device=agent.device
    )
    _, _, group_merits, live_groups = v2.shared_lagrangian_policy_values(
        q_op,
        q_n,
        q_op,
        q_n,
        group_ids,
        agent.within_group_temperatures,
        agent.group_temperature,
        float(lam),
        operational_weight=operational_weight,
    )
    group_probs = F.softmax(
        group_merits / agent.group_temperature, dim=0
    )
    marginal = torch.zeros_like(merits)
    within = torch.zeros_like(merits)
    group_position = {
        int(group.item()): index for index, group in enumerate(live_groups)
    }
    for group_id in live_groups.tolist():
        indices = torch.where(group_ids == int(group_id))[0]
        probabilities = F.softmax(
            merits[indices]
            / float(agent.within_group_temperature_values[int(group_id)]),
            dim=0,
        )
        within[indices] = probabilities
        marginal[indices] = group_probs[group_position[int(group_id)]] * probabilities
    family_probs = defaultdict(float)
    for index, group_id in enumerate(prepared.mode_ids):
        family_probs[v2.ID_TO_CONTROL_GROUP[int(group_id)]] += float(marginal[index])
    result = {
        "lambda": float(lam),
        "candidate_merits": tuple(float(value) for value in merits.detach().cpu()),
        "candidate_within_group_probabilities": tuple(
            float(value) for value in within.detach().cpu()
        ),
        "candidate_marginal_probabilities": tuple(
            float(value) for value in marginal.detach().cpu()
        ),
        "group_merits": {
            v2.ID_TO_CONTROL_GROUP[int(group.item())]: float(group_merits[index])
            for index, group in enumerate(live_groups)
        },
        "group_probabilities": {
            v2.ID_TO_CONTROL_GROUP[int(group.item())]: float(group_probs[index])
            for index, group in enumerate(live_groups)
        },
        "action_family_probabilities": dict(sorted(family_probs.items())),
        "expected_predicted_rehandles": float(torch.sum(marginal * q_n)),
        "probability_sum": float(marginal.sum()),
    }
    if not math.isclose(result["probability_sum"], 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise DecisionProbeError("nested policy probabilities do not sum to one")
    return result


def _candidate_rows(frontier, prepared, q_op, q_n, policies):
    rows = []
    for index, (record, source_index) in enumerate(
        zip(prepared.records, prepared.source_indices)
    ):
        candidate = frontier.candidates[source_index]
        if candidate.key != record.key:
            raise DecisionProbeError("prepared/source candidate mapping drifted")
        certificate = candidate.certificate
        rows.append(
            {
                "prepared_index": index,
                "source_index": int(source_index),
                "key": candidate.key,
                "action_type": candidate.action_type.value,
                "group": v2.ID_TO_CONTROL_GROUP[int(record.group_id)],
                "target_label": candidate.target_label,
                "source": candidate.source,
                "destination": candidate.destination,
                "hold_horizon_steps": candidate.horizon_steps,
                "recovery_rank_delta": candidate.rank_delta,
                "certificate_status": certificate.status.value,
                "certificate_reason": certificate.reason,
                "certificate_witness_macros": certificate.witness_macro_count,
                "certificate_witness_primitive_steps": (
                    certificate.witness_primitive_steps
                ),
                "q_operational": float(q_op[index]),
                "q_predicted_physical_rehandles": float(q_n[index]),
                "lambda0_merit": policies[0.0]["candidate_merits"][index],
                "lambda005_merit": policies[0.05]["candidate_merits"][index],
                "lambda0_probability": policies[0.0][
                    "candidate_marginal_probabilities"
                ][index],
                "lambda005_probability": policies[0.05][
                    "candidate_marginal_probabilities"
                ][index],
            }
        )
    return rows


def _probe_factory(
    runtime, probes: list[dict], *, target_occupancy: int | None = None,
):
    original_factory = runtime._evaluation_agent

    def factory(*, seed):
        clone = original_factory(seed=seed)
        original_select = clone.select

        def probed_select(self, snapshot, *, training=True, epsilon=None):
            physical = behavior._snapshot(runtime.env)
            storage_cells = {
                tuple(cell) for cell in physical["storage_positions"]
            }
            occupied_storage_cells = sum(
                1
                for block in physical["blocks"]
                if not block["delivered"]
                and block["position"] is not None
                and tuple(block["position"]) in storage_cells
            )
            rng_before_selection = self.behavior_rng.getstate()
            decision = original_select(
                snapshot, training=training, epsilon=epsilon
            )
            rng_after_selection = self.behavior_rng.getstate()
            prepared = decision.prepared_snapshot
            frontier, hold_audit = v2._unwrap_snapshot(snapshot)
            with torch.inference_mode():
                q_op, q_n = self._score_records(prepared.records)
                elapsed = {record.elapsed_steps for record in prepared.records}
                if len(elapsed) != 1:
                    raise DecisionProbeError("mixed elapsed contexts in one frontier")
                operational_weight = self._operational_weight(elapsed.pop())
                policies = {
                    lam: _policy_view(
                        self, prepared, q_op, q_n, operational_weight, lam
                    )
                    for lam in LAMBDAS
                }
            shared_draw_rng = random.Random()
            shared_draw_rng.setstate(rng_before_selection)
            shared_draws = {
                "outer_mode_draw": shared_draw_rng.random(),
                "within_mode_draw": shared_draw_rng.random(),
            }
            counterfactual_samples = {}
            for lam in LAMBDAS:
                policy = policies[lam]
                sample_rng = random.Random()
                sample_rng.setstate(rng_before_selection)
                live_groups = tuple(policy["group_probabilities"])
                selected_group = sample_rng.choices(
                    live_groups,
                    weights=tuple(
                        policy["group_probabilities"][group]
                        for group in live_groups
                    ),
                    k=1,
                )[0]
                indices = tuple(
                    index
                    for index, group_id in enumerate(prepared.mode_ids)
                    if v2.ID_TO_CONTROL_GROUP[int(group_id)] == selected_group
                )
                selected_index = sample_rng.choices(
                    indices,
                    weights=tuple(
                        policy["candidate_within_group_probabilities"][index]
                        for index in indices
                    ),
                    k=1,
                )[0]
                counterfactual_samples[lam] = {
                    "selected_group": selected_group,
                    "selected_prepared_index": selected_index,
                    "selected_key": prepared.records[selected_index].key,
                    "selected_action_type": prepared.records[selected_index].action_type,
                }
            if counterfactual_samples[0.0]["selected_key"] != decision.record.key:
                raise DecisionProbeError(
                    "cloned shared RNG did not reproduce the authenticated selection"
                )
            if self.behavior_rng.getstate() != rng_after_selection:
                raise DecisionProbeError("counterfactual scoring consumed policy RNG")
            p0 = policies[0.0]["candidate_marginal_probabilities"]
            p1 = policies[0.05]["candidate_marginal_probabilities"]
            total_variation = 0.5 * sum(abs(a - b) for a, b in zip(p0, p1))
            candidates = _candidate_rows(
                frontier, prepared, q_op, q_n, policies
            )
            probes.append(
                {
                    "decision_index": len(probes),
                    "decision_epoch": int(frontier.decision_epoch),
                    "environment_time": int(runtime.env.time_steps),
                    "physical_snapshot": physical,
                    "frontier_audit": frontier.audit.audit_dict(),
                    "hold_audit": (
                        None if hold_audit is None else hold_audit.audit_dict()
                    ),
                    "exact_safe_frontier_keys": tuple(
                        candidate.key for candidate in frontier.candidates
                    ),
                    "admissible_prepared_keys": tuple(
                        record.key for record in prepared.records
                    ),
                    "liveness_forced": bool(decision.liveness_forced),
                    "selected_key_in_authenticated_replay": decision.record.key,
                    "selected_action_in_authenticated_replay": (
                        decision.candidate.action_type.value
                    ),
                    "selection_source": decision.selection_source,
                    "shared_policy_rng_unit_draws": shared_draws,
                    "same_rng_counterfactual_samples": {
                        "lambda0": counterfactual_samples[0.0],
                        "lambda005": counterfactual_samples[0.05],
                    },
                    "one_macro_counterfactual_forks": None,
                    "operational_weight": float(operational_weight),
                    "occupied_storage_cells": occupied_storage_cells,
                    "safe_candidate_count": len(prepared.records),
                    "live_group_count": len(
                        policies[0.0]["group_probabilities"]
                    ),
                    "total_variation_probability_shift": float(total_variation),
                    "expected_qn_shift": float(
                        policies[0.05]["expected_predicted_rehandles"]
                        - policies[0.0]["expected_predicted_rehandles"]
                    ),
                    "marginal_argmax_key_lambda0": candidates[
                        max(range(len(p0)), key=lambda index: p0[index])
                    ]["key"],
                    "marginal_argmax_key_lambda005": candidates[
                        max(range(len(p1)), key=lambda index: p1[index])
                    ]["key"],
                    "candidates": candidates,
                    "policies": {"lambda0": policies[0.0], "lambda005": policies[0.05]},
                }
            )
            return decision

        clone.select = MethodType(probed_select, clone)
        return clone

    return original_factory, factory


def _replay_one_decision_fork(
    *, parent: Mapping, entry: Mapping, chosen: Mapping, model_seed: int,
    instance_seed: int, rng_index: int, device: str,
) -> dict:
    """Replay the common prefix independently for each one-macro arm."""

    class ForkCaptured(Exception):
        pass

    class ForkProbeFailure(Exception):
        pass

    def run_arm(label: str, lam: float) -> dict:
        captured: dict = {}
        diagnostic = {"decision_count": 0, "runtime_failure": None}
        selection = chosen["same_rng_counterfactual_samples"][label]
        with confirmation._runtime_context(model_seed):
            runtime = _probe_runtime(parent, device=device)
            trainer._install_schedule(
                runtime,
                trainer.schedule_for_episode(
                    confirmation.EPISODE, validation=True
                ),
            )
            runtime.set_dual_lambda(0.0)
            instance = confirmation._load_instance(
                confirmation.DEFAULT_OUTPUT / entry["relative_path"],
                entry,
                runtime.env,
            )
            original_factory = runtime._evaluation_agent

            def factory(*, seed):
                clone = original_factory(seed=seed)
                original_select = clone.select
                decision_counter = 0

                def select(self, snapshot, *, training=True, epsilon=None):
                    nonlocal decision_counter
                    current_index = decision_counter
                    decision_counter += 1
                    diagnostic["decision_count"] = decision_counter
                    if current_index != int(chosen["decision_index"]):
                        return original_select(
                            snapshot, training=training, epsilon=epsilon
                        )
                    physical = behavior._snapshot(runtime.env)
                    if _json_safe(physical) != _json_safe(chosen["physical_snapshot"]):
                        raise ForkProbeFailure(
                            "fork replay did not reach the same state"
                        )
                    frontier, _ = v2._unwrap_snapshot(snapshot)
                    if tuple(
                        candidate.key for candidate in frontier.candidates
                    ) != tuple(chosen["exact_safe_frontier_keys"]):
                        raise ForkProbeFailure("fork replay SAFE frontier drifted")
                    draw_rng = random.Random()
                    draw_rng.setstate(self.behavior_rng.getstate())
                    observed_draws = (
                        draw_rng.random(),
                        draw_rng.random(),
                    )
                    expected_draws = chosen["shared_policy_rng_unit_draws"]
                    if observed_draws != (
                        expected_draws["outer_mode_draw"],
                        expected_draws["within_mode_draw"],
                    ):
                        raise ForkProbeFailure("fork replay policy RNG state drifted")
                    prepared, liveness_forced = self._admissible_prepared(snapshot)
                    if liveness_forced:
                        raise ForkProbeFailure("fork unexpectedly became liveness-forced")
                    matching_records = [
                        (index, record)
                        for index, record in enumerate(prepared.records)
                        if record.key == selection["selected_key"]
                    ]
                    matching_candidates = [
                        candidate
                        for candidate in frontier.candidates
                        if candidate.key == selection["selected_key"]
                    ]
                    if len(matching_records) != 1 or len(matching_candidates) != 1:
                        raise ForkProbeFailure(
                            "counterfactual candidate is missing or duplicated"
                        )
                    _, record = matching_records[0]
                    candidate = matching_candidates[0]
                    try:
                        execution = runtime._execute_macro(
                            candidate,
                            remaining_steps=record.remaining_episode_steps,
                            evaluation=True,
                        )
                    except (RuntimeError, ValueError, TypeError) as error:
                        raise ForkProbeFailure(
                            f"counterfactual macro failed: {type(error).__name__}: {error}"
                        ) from error
                    captured.update(
                        {
                            "lambda": lam,
                            "selected_key": candidate.key,
                            "selected_action_type": candidate.action_type.value,
                            "duration": int(execution.duration),
                            "raw_environment_return": float(execution.raw_return),
                            "discounted_environment_return": float(
                                execution.discounted_return
                            ),
                            "physical_rehandles": int(execution.relocations),
                            "illegal_drops": int(execution.illegal_drops),
                            "delivery_deviations": tuple(
                                float(value)
                                for value in execution.delivery_deviations
                            ),
                            "option_success": bool(execution.option_success),
                            "failure_reason": execution.failure_reason,
                            "truncated": bool(execution.truncated),
                            "post_action_snapshot": behavior._snapshot(runtime.env),
                        }
                    )
                    raise ForkCaptured()

                clone.select = MethodType(select, clone)
                return clone

            runtime._evaluation_agent = factory
            runtime.begin_validation_batch()
            original_environment = runtime.env
            runtime.env = confirmation._BoundEnvironment(
                original_environment, instance
            )
            try:
                try:
                    observed = runtime.run_episode(
                        instance_seed=instance_seed,
                        training=False,
                        max_steps=confirmation.MAX_STEPS,
                        policy_rng_index=rng_index,
                        policy_rng_seed=confirmation._policy_rng(
                            instance_seed - confirmation.INSTANCE_SEEDS[0],
                            rng_index,
                        ),
                    )
                    diagnostic["runtime_failure"] = observed.get(
                        "method_failure_reason"
                    )
                except ForkCaptured:
                    pass
                except ForkProbeFailure as error:
                    raise DecisionProbeError(str(error)) from error
            finally:
                runtime.env = original_environment
                runtime._evaluation_agent = original_factory
                batch = dict(runtime.end_validation_batch())
            if batch.get("training_agent_unchanged") is not True:
                raise DecisionProbeError(
                    "fork replay mutated the frozen training agent"
                )
        if not captured:
            raise DecisionProbeError(
                f"{label} fork not reached: decisions={diagnostic['decision_count']}, "
                f"runtime_failure={diagnostic['runtime_failure']!r}"
            )
        return captured

    return {
        "lambda0": run_arm("lambda0", 0.0),
        "lambda005": run_arm("lambda005", 0.05),
    }


def _rank_probe(
    probe: Mapping,
    *,
    target_occupancy: int | None = None,
    allow_single_group: bool = False,
    target_decision_index: int | None = None,
):
    occupancy_matches = (
        target_occupancy is None
        or int(probe["occupied_storage_cells"]) == int(target_occupancy)
    )
    eligible = (
        occupancy_matches
        and (
            target_decision_index is None
            or int(probe["decision_index"]) == int(target_decision_index)
        )
        and not probe["liveness_forced"]
        # The figure groups candidates by action family and displays only the
        # most consequential individual options, so a modest 10-way frontier
        # remains readable while preserving the complete SAFE set in JSON.
        and 2 <= probe["safe_candidate_count"]
        and (
            target_occupancy is not None
            or probe["safe_candidate_count"] <= 12
        )
        and probe["live_group_count"] >= (1 if allow_single_group else 2)
    )
    rejects = int(probe["frontier_audit"]["fail_closed_rejection_count"])
    argmax_changed = (
        probe["marginal_argmax_key_lambda0"]
        != probe["marginal_argmax_key_lambda005"]
    )
    reduced_expected_qn = probe["expected_qn_shift"] < 0.0
    score = (
        float(probe["total_variation_probability_shift"])
        + (0.10 if rejects else 0.0)
        + (0.10 if argmax_changed else 0.0)
        + (0.05 if reduced_expected_qn else 0.0)
    )
    return (eligible, score, -probe["safe_candidate_count"], -probe["decision_index"])


def _short_candidate_label(candidate: Mapping) -> str:
    action = str(candidate["action_type"])
    target = candidate.get("target_label")
    destination = candidate.get("destination")
    if action == "deliver":
        return f"Deliver {target} · exit {tuple(destination)}"
    if action == "reconfigure":
        return f"Move {target} · cell {tuple(destination)}"
    if action == "accept":
        return f"Store {target} at {tuple(destination)}"
    if action == "defer":
        return f"Hold {candidate.get('hold_horizon_steps')} steps"
    return str(candidate["key"])


def _figure_probe(
    *, chosen: Mapping, saved: Mapping, parent: Mapping, model_seed: int,
    instance_seed: int, rng_index: int,
) -> dict:
    """Adapt the complete probe to the deliberately small renderer contract."""

    snapshot = chosen["physical_snapshot"]
    rooms = snapshot["rooms"]
    rows, cols = len(rooms), len(rooms[0])
    storage = {tuple(cell) for cell in snapshot["storage_positions"]}
    exits = {tuple(cell) for cell in snapshot["exit_cells"]}
    pickup = tuple(snapshot["pickup_cell"])
    waiting = tuple(snapshot["waiting_cell"])
    active_by_cell = defaultdict(list)
    for block in snapshot["blocks"]:
        position = block.get("position")
        if position is not None and not block.get("delivered"):
            active_by_cell[tuple(position)].append({"label": block["label"]})
    cells = []
    for row in range(rows):
        for col in range(cols):
            cell = (row, col)
            if rooms[row][col] == "#":
                kind = "wall"
            elif cell in exits:
                kind = "exit"
            elif cell == pickup:
                kind = "pickup"
            elif cell == waiting:
                kind = "waiting"
            elif cell in storage:
                kind = "storage"
            else:
                kind = "path"
            cells.append(
                {"row": row, "col": col, "kind": kind, "blocks": active_by_cell[cell]}
            )

    safe_digest = hashlib.sha256(
        "\n".join(chosen["admissible_prepared_keys"]).encode("utf-8")
    ).hexdigest()
    p0 = chosen["policies"]["lambda0"]
    p1 = chosen["policies"]["lambda005"]
    candidate_values = list(chosen["candidates"])
    priority_order = sorted(
        range(len(candidate_values)),
        key=lambda index: max(
            candidate_values[index]["lambda0_probability"],
            candidate_values[index]["lambda005_probability"],
        ),
        reverse=True,
    )
    priorities = {index: len(candidate_values) - rank for rank, index in enumerate(priority_order)}
    figure_candidates = []
    for index, candidate in enumerate(candidate_values):
        figure_candidates.append(
            {
                "id": candidate["key"],
                "mode": candidate["group"],
                "short_label": _short_candidate_label(candidate),
                "q_op": candidate["q_operational"],
                "q_n": candidate["q_predicted_physical_rehandles"],
                "figure_priority": priorities[index],
                "show_arrow": index in priority_order[:2],
                "geometry": {
                    "kind": candidate["action_type"],
                    "from": candidate.get("source"),
                    "to": candidate.get("destination"),
                },
            }
        )

    conditions = []
    for lam, label, policy in (
        (0.0, "lambda = 0", p0),
        (0.05, "lambda = .05", p1),
    ):
        conditions.append(
            {
                "lambda": lam,
                "label": label,
                "mode_probabilities": [
                    {"mode": mode, "probability": probability}
                    for mode, probability in policy["group_probabilities"].items()
                ],
                "candidate_probabilities": [
                    {
                        "candidate_id": candidate["key"],
                        "marginal_probability": policy[
                            "candidate_marginal_probabilities"
                        ][index],
                        "conditional_probability": policy[
                            "candidate_within_group_probabilities"
                        ][index],
                    }
                    for index, candidate in enumerate(candidate_values)
                ],
            }
        )
    audit = chosen["frontier_audit"]
    return {
        "schema_version": "vcg-single-state-probe/v1",
        "title": "VCG: hard viability, soft handling preference",
        "provenance": {
            "status": "authenticated_replay",
            "source": (
                f"87k confirmation · model seed {model_seed} · instance "
                f"{instance_seed} · RNG {rng_index}"
            ),
            "checkpoint": str(parent["checkpoint_path"]),
            "checkpoint_raw_sha256": parent["checkpoint_sha256"],
            "confirmation_row_exactly_reproduced": True,
        },
        "state": {
            "instance_id": saved["episode_instance_id"],
            "decision_index": chosen["decision_index"],
            "sim_time": chosen["environment_time"],
            "yard": {
                "rows": rows,
                "cols": cols,
                "stack_capacity": 1,
                "cells": cells,
                "inbound_queue": [
                    block["label"]
                    for block in snapshot["blocks"]
                    if not block["stored"] and not block["delivered"]
                ],
            },
        },
        "viability": {
            "safe_exposed_count": len(candidate_values),
            "fail_closed_unknown_excluded_count": int(audit["unknown_accept_count"])
            + int(audit["unknown_recovery_count"]),
            "proven_unsafe_excluded_count": int(audit["unsafe_accept_count"])
            + int(audit["unsafe_recovery_count"]),
            "safe_set_digest": safe_digest,
            "complete_frontier_exactly_verified": audit[
                "complete_frontier_exactly_verified"
            ],
        },
        "policy": {
            "type": "nested_stochastic_softmax",
            "same_safe_set": True,
            "operational_weight": chosen["operational_weight"],
            "candidates": figure_candidates,
            "conditions": conditions,
        },
    }


def collect(
    *, model_seed: int, instance_seed: int, rng_index: int, device: str,
    target_occupancy: int | None = None,
    allow_single_group: bool = False,
    target_decision_index: int | None = None,
) -> dict:
    # Authenticate the original CUDA-bound experiment before any device-local
    # explanatory replay is constructed.
    contract = confirmation.prepare(
        confirmation.DEFAULT_OUTPUT, confirmation.DEFAULT_PARENT, device="cuda"
    )
    manifest = confirmation._validate_manifest(
        confirmation.DEFAULT_OUTPUT, contract
    )
    entries = {int(item["instance_seed"]): item for item in manifest["instances"]}
    if instance_seed not in entries:
        raise DecisionProbeError("requested instance is outside the frozen 87k panel")
    parent = development._parent(confirmation.DEFAULT_PARENT, model_seed)
    probes: list[dict] = []
    saved = behavior._saved_row(
        confirmation.DEFAULT_OUTPUT,
        model_seed,
        0.0,
        instance_seed,
        rng_index,
    )
    with confirmation._runtime_context(model_seed):
        runtime = _probe_runtime(parent, device=device)
        trainer._install_schedule(
            runtime, trainer.schedule_for_episode(confirmation.EPISODE, validation=True)
        )
        runtime.set_dual_lambda(0.0)
        instance = confirmation._load_instance(
            confirmation.DEFAULT_OUTPUT / entries[instance_seed]["relative_path"],
            entries[instance_seed],
            runtime.env,
        )
        original_factory, factory = _probe_factory(
            runtime, probes, target_occupancy=target_occupancy
        )
        runtime._evaluation_agent = factory
        runtime.begin_validation_batch()
        original_environment = runtime.env
        runtime.env = confirmation._BoundEnvironment(original_environment, instance)
        try:
            observed = dict(
                runtime.run_episode(
                    instance_seed=instance_seed,
                    training=False,
                    max_steps=confirmation.MAX_STEPS,
                    policy_rng_index=rng_index,
                    policy_rng_seed=confirmation._policy_rng(
                        instance_seed - confirmation.INSTANCE_SEEDS[0], rng_index
                    ),
                )
            )
        finally:
            runtime.env = original_environment
            runtime._evaluation_agent = original_factory
            batch = dict(runtime.end_validation_batch())
        if batch.get("training_agent_unchanged") is not True:
            raise DecisionProbeError("probe replay mutated the frozen training agent")
        behavior._assert_replay(saved, observed)
        if not probes:
            raise DecisionProbeError("authenticated replay exposed no decisions")
        rank_key = lambda probe: _rank_probe(
            probe,
            target_occupancy=target_occupancy,
            allow_single_group=allow_single_group,
            target_decision_index=target_decision_index,
        )
        chosen = max(probes, key=rank_key)
        if not rank_key(chosen)[0]:
            raise DecisionProbeError("no reader-suitable multi-action decision was found")
    samples = chosen["same_rng_counterfactual_samples"]
    if (
        target_occupancy is not None
        and samples["lambda0"]["selected_key"]
        != samples["lambda005"]["selected_key"]
    ):
        chosen["one_macro_counterfactual_forks"] = _replay_one_decision_fork(
            parent=parent,
            entry=entries[instance_seed],
            chosen=chosen,
            model_seed=model_seed,
            instance_seed=instance_seed,
            rng_index=rng_index,
            device=device,
        )
    result = {
        "protocol": PROTOCOL,
        "role": "post_hoc_mechanism_illustration_not_performance_evidence",
        "mechanism": (
            "exact verifier fixes the SAFE frontier; lambda shifts a nested "
            "stochastic preference over that unchanged frontier"
        ),
        "counterfactual_lambdas": LAMBDAS,
        "model_seed": model_seed,
        "instance_seed": instance_seed,
        "policy_rng_index": rng_index,
        "policy_rng_seed": confirmation._policy_rng(
            instance_seed - confirmation.INSTANCE_SEEDS[0], rng_index
        ),
        "authoritative_confirmation_device": "cuda",
        "probe_execution_device": device,
        "exactly_reproduces_persisted_confirmation_row": True,
        "confirmation_row_summary": {
            key: saved[key]
            for key in (
                "steps",
                "physical_rehandles",
                "dense_return",
                "mean_absolute_error",
                "selected_action_counts",
            )
        },
        "decision_count": len(probes),
        "requested_target_occupancy": target_occupancy,
        "selected_probe": chosen,
        "all_probe_rankings": tuple(
            {
                "decision_index": item["decision_index"],
                "environment_time": item["environment_time"],
                "occupied_storage_cells": item["occupied_storage_cells"],
                "safe_candidate_count": item["safe_candidate_count"],
                "live_group_count": item["live_group_count"],
                "fail_closed_rejection_count": item["frontier_audit"][
                    "fail_closed_rejection_count"
                ],
                "liveness_forced": item["liveness_forced"],
                "total_variation_probability_shift": item[
                    "total_variation_probability_shift"
                ],
                "expected_qn_shift": item["expected_qn_shift"],
            }
            for item in sorted(probes, key=rank_key, reverse=True)
        ),
    }
    result["figure_probe"] = _figure_probe(
        chosen=chosen,
        saved=saved,
        parent=parent,
        model_seed=model_seed,
        instance_seed=instance_seed,
        rng_index=rng_index,
    )
    result["probe_sha256"] = _canonical_sha(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-seed", type=int, default=DEFAULT_MODEL_SEED)
    parser.add_argument("--instance-seed", type=int, default=DEFAULT_INSTANCE_SEED)
    parser.add_argument("--rng-index", type=int, default=DEFAULT_RNG_INDEX)
    parser.add_argument(
        "--target-occupancy",
        type=int,
        choices=range(0, 9),
        default=None,
        help="prefer an exact number of occupied storage cells",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--allow-single-group",
        action="store_true",
        help="allow a within-mode mechanism probe (for example, storage-cell choice)",
    )
    parser.add_argument(
        "--target-decision-index",
        type=int,
        default=None,
        help="select one exact zero-based decision boundary",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=None,
        help="renderer-ready JSON (defaults beside --output)",
    )
    args = parser.parse_args()
    result = collect(
        model_seed=args.model_seed,
        instance_seed=args.instance_seed,
        rng_index=args.rng_index,
        device=args.device,
        target_occupancy=args.target_occupancy,
        allow_single_group=args.allow_single_group,
        target_decision_index=args.target_decision_index,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(result), indent=2) + "\n", encoding="utf-8")
    figure_output = (
        args.figure_output.resolve()
        if args.figure_output is not None
        else output.with_name("vcg-mechanism-figure-input.json")
    )
    figure_output.write_text(
        json.dumps(_json_safe(result["figure_probe"]), indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "figure_output": str(figure_output),
        "probe_sha256": result["probe_sha256"],
        "selected_decision": result["selected_probe"]["decision_index"],
        "environment_time": result["selected_probe"]["environment_time"],
        "safe_candidates": result["selected_probe"]["safe_candidate_count"],
        "occupied_storage_cells": result["selected_probe"]["occupied_storage_cells"],
        "fail_closed_rejections": result["selected_probe"]["frontier_audit"]["fail_closed_rejection_count"],
        "total_variation_shift": result["selected_probe"]["total_variation_probability_shift"],
    }, indent=2))


if __name__ == "__main__":
    main()
