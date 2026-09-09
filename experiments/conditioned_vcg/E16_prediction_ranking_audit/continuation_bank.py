#!/usr/bin/env python3
"""E16-B: frozen paired continuations from observed ranking crossings.

The experiment follows the fixed-future controller until the immediate-only
and learned-future merits choose different exact-SAFE candidates. From that
identical state it forces each candidate once, then gives both branches back
to the same frozen fixed-future controller. No weights are trained or chosen.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from statistics import fmean
import sys
from typing import Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import benchmark_viability_critic_priority as benchmark
from example.episode_instance import EpisodeInstance
from experiments.conditioned_vcg.E05_handling_model_ablation_92k.run import (
    ClampedPreferenceNetwork,
)
from experiments.conditioned_vcg.E11_distribution_shift_93k import run as e11
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from PSLAP.viability import ViabilityStatus
from PSLAP.viability_candidates import (
    RELOCATION_FAMILY_CERTIFICATION,
    ViabilityActionType,
)
import run_vcg_final86_four_method as final86
import run_vcg_v11_nested_handling_pilot as pilot
from train_viability_graph_smdp import execute_certified_macro
from vcg_v11_conditioned_handling import ConditionedHandlingAgent
from vcg_v11_nested_handling import detached_v11_features
from viability_graph_hierarchy import ID_TO_MODE, ViabilityGraphDecision
from viability_graph_preference_conditioned import select_hierarchical_index


PROTOCOL = "vcg_conditioned_e16_paired_continuation_bank_96k_v1"
SCHEMA_VERSION = 1
MODEL_SEEDS = (0, 1, 2)
INSTANCE_SEEDS = (96_000, 96_001, 96_002)
REGIME_IDS = ("reference", "dwell_short", "dwell_long", "mirrored_entry")
DEPLOYMENT_LAMBDA = 0.10
PREDICTOR_INPUT_LAMBDA = 0.10
MAX_STEPS_AFTER_CAPTURE = 2_000
SEARCH_MAX_NODES = 20_000
PILOT_MODEL_SEED = 0
PILOT_REGIME_ID = "reference"

CONTRACT_NAME = "e16b-contract.json"
MANIFEST_NAME = "e16b-episode-instance-manifest.json"
REPORT_NAME = "e16b-continuation-report.json"
TABLE_NAME = "e16b-continuation-summary.md"
CASE_ROWS_NAME = "e16b-continuation-cases.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/vcg-conditioned-e16-continuation-bank-96k"

TERMINAL_PATHS = {
    0: PROJECT_ROOT
    / "results/vcg-v1-1-conditioned-handling-seed0-damped-convergence/terminal.pth",
    1: PROJECT_ROOT
    / "results/vcg-v1-1-conditioned-handling-seed1-convergence-continuation/terminal.pth",
    2: PROJECT_ROOT
    / "results/vcg-v1-1-conditioned-handling-seeds12-two-phase-development/seed-2/terminal.pth",
}
TERMINAL_SHA256 = {
    0: "842e4ed28f1a8b14fe2eb8d58bd58cae4e8d739ba4e057c8754d2ba06db23d29",
    1: "4235565890a381f3199c51fcbd7de7c941c778b1e2453d35d6ee4fe3457a2f71",
    2: "c857c0ec6699997dcc7da307c05c97fe843b207f7b6b874261b4dee5eb4b64c6",
}


class E16BError(RuntimeError):
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
        raise E16BError(f"missing regular input: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E16BError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise E16BError(f"{label} must contain an object")
    return value


def _verify_hash(value: Mapping, field: str, label: str) -> None:
    if value.get(field) != _digest(value, field):
        raise E16BError(f"{label} self-hash mismatch")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _frozen_arms() -> dict[int, object]:
    arms, _records = final86._authenticate_v11()
    result = {int(arm.model_seed): arm for arm in arms}
    if tuple(sorted(result)) != MODEL_SEEDS:
        raise E16BError("frozen VCG 1.1 arms are incomplete")
    return result


def _terminal(seed: int) -> dict:
    path = TERMINAL_PATHS[int(seed)]
    if _sha(path) != TERMINAL_SHA256[int(seed)]:
        raise E16BError(f"conditioned seed-{seed} terminal changed")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping) or not isinstance(
        value.get("agent_checkpoint"), Mapping
    ):
        raise E16BError(f"conditioned seed-{seed} terminal is malformed")
    return dict(value)


def _new_agent(seed: int, device: torch.device, *, arms=None):
    arms = _frozen_arms() if arms is None else arms
    arm = arms[int(seed)]
    checkpoint = _terminal(seed)["agent_checkpoint"]
    base = pilot._fresh_base(arm, device)
    agent = ConditionedHandlingAgent.from_checkpoint(
        checkpoint,
        base_agent=base,
        expected_base_checkpoint_sha256=arm.checkpoint_sha256,
        expected_base_policy_digest=arm.deployment_policy_digest,
        expected_source_cost_sha256=checkpoint["source_cost_sha256"],
        seed=int(seed),
    )
    agent.handling_network = ClampedPreferenceNetwork(
        agent.handling_network, PREDICTOR_INPUT_LAMBDA
    ).to(device)
    agent.handling_network.requires_grad_(False).eval()
    agent.set_epsilon(0.0)
    agent.reset_episode_state()
    return agent, arm


def _source_hashes() -> dict:
    paths = (
        Path(__file__).resolve(),
        Path(e11.__file__).resolve(),
        PROJECT_ROOT / "benchmark_viability_critic_priority.py",
        PROJECT_ROOT / "vcg_v11_conditioned_handling.py",
        PROJECT_ROOT / "vcg_v11_nested_handling.py",
        PROJECT_ROOT / "viability_graph_hierarchy.py",
        PROJECT_ROOT / "viability_graph_preference_conditioned.py",
        PROJECT_ROOT / "PSLAP/viability_candidates.py",
        PROJECT_ROOT / "PSLAP/relocation_family_certification.py",
        Path(reuse.__file__).resolve(),
        PROJECT_ROOT / "example/Options/certified_path.py",
    )
    return {str(path.relative_to(PROJECT_ROOT)): _sha(path) for path in paths}


def _contract() -> dict:
    arms = _frozen_arms()
    for seed in MODEL_SEEDS:
        terminal = _terminal(seed)["agent_checkpoint"]
        if (
            terminal.get("base_checkpoint_sha256")
            != arms[seed].checkpoint_sha256
            or terminal.get("base_policy_digest")
            != arms[seed].deployment_policy_digest
        ):
            raise E16BError(f"conditioned seed-{seed} does not bind its base arm")
    return _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "scientific_question": (
                "do_candidate_relative_future_handling_predictions_track_"
                "realized_continuations_and_explain_ranking_benefit"
            ),
            "model_seeds": list(MODEL_SEEDS),
            "regime_ids": list(REGIME_IDS),
            "ordered_scout_instance_seeds": list(INSTANCE_SEEDS),
            "deployment_lambda": DEPLOYMENT_LAMBDA,
            "predictor_input_lambda": PREDICTOR_INPUT_LAMBDA,
            "capture_rule": (
                "earliest_unguarded_visited_frontier_where_exact_hierarchical_"
                "immediate_only_and_fixed_future_winners_differ"
            ),
            "maximum_cases": len(MODEL_SEEDS) * len(REGIME_IDS),
            "branches_per_case": 2,
            "continuation_policy": (
                "same_frozen_fixed_future_policy_after_one_forced_candidate"
            ),
            "future_target": (
                "undiscounted_physical_rehandles_after_forced_current_macro"
            ),
            "search_max_nodes": SEARCH_MAX_NODES,
            "max_steps_after_capture": MAX_STEPS_AFTER_CAPTURE,
            "recovery_certification_strategy": RELOCATION_FAMILY_CERTIFICATION,
            "path_cleanup_enabled": True,
            "training_or_checkpoint_selection": False,
            "liveness_forced_frontiers_eligible": False,
            "no_crossing_coordinates_retained": True,
            "base_checkpoint_sha256": {
                str(seed): arms[seed].checkpoint_sha256 for seed in MODEL_SEEDS
            },
            "base_policy_digest": {
                str(seed): arms[seed].deployment_policy_digest
                for seed in MODEL_SEEDS
            },
            "conditioned_terminal_sha256": {
                str(seed): TERMINAL_SHA256[seed] for seed in MODEL_SEEDS
            },
            "source_sha256": _source_hashes(),
        },
        "contract_sha256",
    )


def _instance_path(output: Path, regime_id: str, seed: int) -> Path:
    return output / "episode-instances" / regime_id / f"seed-{seed}.json"


def _ledger_path(output: Path, regime_id: str, model_seed: int) -> Path:
    return output / "run-ledger" / regime_id / f"model-seed-{model_seed}.json"


def prepare(output: Path) -> dict:
    output = output.resolve()
    expected = _contract()
    contract_path = output / CONTRACT_NAME
    if contract_path.is_file():
        observed = _load_json(contract_path, "E16-B contract")
        _verify_hash(observed, "contract_sha256", "E16-B contract")
        if observed != expected:
            raise E16BError("E16-B contract, sources, or frozen artifacts changed")
    else:
        if output.exists() and any(output.iterdir()):
            raise E16BError("nonempty E16-B output has no contract")
        _atomic_json(contract_path, expected)

    records = []
    for regime_id in REGIME_IDS:
        regime = e11.REGIME_BY_ID[regime_id]
        for order, seed in enumerate(INSTANCE_SEEDS):
            path = _instance_path(output, regime_id, seed)
            expected_instance = e11._instance(regime, seed)
            if path.is_file():
                observed = EpisodeInstance.from_json(path.read_text(encoding="utf-8"))
                if observed != expected_instance:
                    raise E16BError(f"frozen instance changed: {regime_id}/{seed}")
            else:
                _atomic_text(path, expected_instance.to_json() + "\n")
                observed = EpisodeInstance.from_json(path.read_text(encoding="utf-8"))
            records.append(
                {
                    "regime_id": regime_id,
                    "scout_order": order,
                    "instance_seed": seed,
                    "relative_path": str(path.relative_to(output)),
                    "raw_sha256": _sha(path),
                    "episode_instance_id": observed.instance_id,
                    "schedule_id": observed.schedule_id,
                }
            )
    manifest = _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": expected["contract_sha256"],
            "instances_frozen_before_continuation_outcomes": True,
            "records": records,
        },
        "manifest_sha256",
    )
    manifest_path = output / MANIFEST_NAME
    if manifest_path.is_file():
        observed = _load_json(manifest_path, "E16-B manifest")
        _verify_hash(observed, "manifest_sha256", "E16-B manifest")
        if observed != manifest:
            raise E16BError("E16-B instance manifest changed")
    else:
        _atomic_json(manifest_path, manifest)
    return {
        "status": "prepared",
        "training_runs": 0,
        "coordinates": len(MODEL_SEEDS) * len(REGIME_IDS),
        "scout_instances_per_coordinate": len(INSTANCE_SEEDS),
        "maximum_branch_rollouts": 2 * len(MODEL_SEEDS) * len(REGIME_IDS),
        "contract": str(contract_path),
        "manifest": str(manifest_path),
    }


def authenticate(output: Path) -> tuple[dict, dict]:
    contract = _load_json(output / CONTRACT_NAME, "E16-B contract")
    _verify_hash(contract, "contract_sha256", "E16-B contract")
    if contract != _contract():
        raise E16BError("E16-B contract, sources, or frozen artifacts changed")
    manifest = _load_json(output / MANIFEST_NAME, "E16-B manifest")
    _verify_hash(manifest, "manifest_sha256", "E16-B manifest")
    if (
        manifest.get("contract_sha256") != contract["contract_sha256"]
        or len(manifest.get("records", ()))
        != len(REGIME_IDS) * len(INSTANCE_SEEDS)
    ):
        raise E16BError("E16-B manifest binding or size changed")
    for record in manifest["records"]:
        if _sha(output / record["relative_path"]) != record["raw_sha256"]:
            raise E16BError("E16-B frozen instance bytes changed")
    return contract, manifest


def _instance(output: Path, manifest: Mapping, regime_id: str, seed: int):
    matches = [
        record
        for record in manifest["records"]
        if record["regime_id"] == regime_id
        and int(record["instance_seed"]) == int(seed)
    ]
    if len(matches) != 1:
        raise E16BError("frozen E16-B instance record is not unique")
    record = matches[0]
    value = EpisodeInstance.from_json(
        (output / record["relative_path"]).read_text(encoding="utf-8")
    )
    value.validate_for(e11.REGIME_BY_ID[regime_id].make_env())
    if value.instance_id != record["episode_instance_id"]:
        raise E16BError("E16-B instance identity changed")
    return value


def _search_config(arm):
    config = benchmark._search_config(arm.payload)
    if int(config.max_nodes) != SEARCH_MAX_NODES:
        raise E16BError(
            f"frozen verifier has max_nodes={config.max_nodes}, expected {SEARCH_MAX_NODES}"
        )
    return config


def _score_frontier(agent: ConditionedHandlingAgent, snapshot) -> dict:
    prepared, liveness_forced = agent.base_agent._admissible_prepared(snapshot)
    if not prepared.records:
        raise E16BError("exact-SAFE frontier became empty at the learned interface")
    features = detached_v11_features(agent.base_agent.Q_local, prepared.records)
    with torch.no_grad():
        operational = agent.base_agent.Q_local.q_head(features).squeeze(-1)
        immediate = torch.as_tensor(
            [
                1.0
                if record.action_type == ViabilityActionType.RECONFIGURE.value
                else 0.0
                for record in prepared.records
            ],
            dtype=features.dtype,
            device=features.device,
        )
        future = agent.handling_network(features, PREDICTOR_INPUT_LAMBDA)
        immediate_merit = operational - DEPLOYMENT_LAMBDA * immediate
        future_merit = immediate_merit - DEPLOYMENT_LAMBDA * future
    keys = tuple(record.key for record in prepared.records)
    immediate_selection = select_hierarchical_index(
        immediate_merit,
        prepared.mode_ids,
        agent.base_agent.within_temperatures,
        candidate_keys=keys,
    )
    future_selection = select_hierarchical_index(
        future_merit,
        prepared.mode_ids,
        agent.base_agent.within_temperatures,
        candidate_keys=keys,
    )
    rows = []
    for index, record in enumerate(prepared.records):
        rows.append(
            {
                "prepared_index": index,
                "source_index": int(prepared.source_indices[index]),
                "candidate_key": record.key,
                "mode": ID_TO_MODE[int(record.mode_id)],
                "action_type": record.action_type,
                "qop": float(operational[index]),
                "immediate_rehandle": int(immediate[index].item()),
                "predicted_future_rehandles": float(future[index]),
                "immediate_only_merit": float(immediate_merit[index]),
                "fixed_future_merit": float(future_merit[index]),
            }
        )
    return {
        "prepared": prepared,
        "liveness_forced": bool(liveness_forced),
        "rows": rows,
        "immediate_index": int(immediate_selection.selected_index),
        "future_index": int(future_selection.selected_index),
        "immediate_selection": immediate_selection,
        "future_selection": future_selection,
        "immediate_merits": immediate_merit,
        "future_merits": future_merit,
    }


def _forced_decision(snapshot, scores: Mapping, branch: str) -> ViabilityGraphDecision:
    if branch not in ("immediate_only_winner", "fixed_future_winner"):
        raise E16BError(f"unknown continuation branch: {branch}")
    use_immediate = branch == "immediate_only_winner"
    index = int(
        scores["immediate_index"] if use_immediate else scores["future_index"]
    )
    hierarchy = (
        scores["immediate_selection"]
        if use_immediate
        else scores["future_selection"]
    )
    merits = scores["immediate_merits"] if use_immediate else scores["future_merits"]
    prepared = scores["prepared"]
    candidate = snapshot.candidates[prepared.source_indices[index]]
    exact_rank_progress = bool(
        snapshot.audit.recovery_rank_exact
        and candidate.mode.value == "recover"
        and candidate.rank_delta is not None
        and candidate.rank_delta > 0
    )
    return ViabilityGraphDecision(
        candidate=candidate,
        record=prepared.records[index],
        prepared_snapshot=prepared,
        q_values=tuple(float(value) for value in merits.detach().cpu()),
        mode_values=tuple(
            (ID_TO_MODE[int(mode_id)], float(value))
            for mode_id, value in hierarchy.mode_values
        ),
        explored=False,
        selection_source=f"e16_forced_{branch}",
        liveness_forced=False,
        exact_rank_progress=exact_rank_progress,
    )


def _next_defer_count(current: int, decision, execution) -> int:
    if execution.action_type != ViabilityActionType.DEFER.value:
        return 0
    outcome = getattr(decision.option, "last_outcome", None)
    observed_event = bool(
        isinstance(outcome, dict) and outcome.get("reason") == "observed_event"
    )
    return 0 if observed_event else int(current) + 1


def _timing(deviations: Sequence[float]) -> dict:
    values = tuple(float(value) for value in deviations)
    if not values:
        return {"mean_absolute_error": None, "within_target_window_rate": None}
    return {
        "mean_absolute_error": fmean(abs(value) for value in values),
        "within_target_window_rate": fmean(abs(value) <= 20.0 for value in values),
    }


def _enumerate(env, consecutive_defer, arm, cache):
    return benchmark._enumerate_frontier(
        env,
        consecutive_defer=consecutive_defer,
        search_config=_search_config(arm),
        liveness_rule=benchmark._liveness_rule(arm.payload),
        cache=cache,
        prioritizer=None,
        recovery_certification_strategy=RELOCATION_FAMILY_CERTIFICATION,
    )


def _run_branch(
    *,
    branch: str,
    model_seed: int,
    arm,
    captured_env,
    captured_snapshot,
    guard_state: Mapping,
    consecutive_defer: int,
    expected_keys: Mapping,
    device: torch.device,
    arms,
) -> dict:
    # Candidate options retain strict bindings to the environment and its
    # block objects. Copy the connected object graph as one unit.
    env, snapshot = deepcopy((captured_env, captured_snapshot))
    agent, loaded_arm = _new_agent(model_seed, device, arms=arms)
    if loaded_arm.checkpoint_sha256 != arm.checkpoint_sha256:
        raise E16BError("branch loaded a different operational checkpoint")
    agent.base_agent.recovery_witness_guard.load_state_dict(dict(guard_state))
    scores = _score_frontier(agent, snapshot)
    if scores["liveness_forced"]:
        raise E16BError("captured unforced frontier became guard-forced in branch")
    observed_keys = {
        "immediate_only_winner": scores["rows"][scores["immediate_index"]][
            "candidate_key"
        ],
        "fixed_future_winner": scores["rows"][scores["future_index"]][
            "candidate_key"
        ],
    }
    if observed_keys != dict(expected_keys):
        raise E16BError("captured ranking crossing did not reproduce in branch")
    decision = _forced_decision(snapshot, scores, branch)
    selected_index = (
        scores["immediate_index"]
        if branch == "immediate_only_winner"
        else scores["future_index"]
    )
    selected_row = scores["rows"][selected_index]

    delivered_before = sum(bool(block.delivered) for block in env.blocks)
    required_remaining = len(env.blocks) - delivered_before
    steps = 0
    total_return = 0.0
    total_rehandles = 0
    future_rehandles = 0
    deviations = []
    illegal_drops = 0
    macro_failures = 0
    guard_forced = 0
    exact_frontiers = True
    decisions = []
    cache = reuse.TimingInvariantCertificateCache()
    pending = snapshot
    first = True
    failure = None

    while steps < MAX_STEPS_AFTER_CAPTURE and not env.is_state_terminal(env.current_state):
        if not first:
            if pending is None or not pending.candidates:
                failure = "no_exact_safe_candidate"
                break
            decision = agent.select(
                pending,
                preference_lambda=DEPLOYMENT_LAMBDA,
                training=False,
                epsilon=0.0,
            )
        guard_forced += int(decision.liveness_forced)
        execution = execute_certified_macro(
            env,
            decision.candidate,
            gamma=agent.config.gamma,
            remaining_steps=MAX_STEPS_AFTER_CAPTURE - steps,
            evaluation=True,
        )
        steps += int(execution.duration)
        total_return += float(execution.raw_return)
        total_rehandles += int(execution.relocations)
        if not first:
            future_rehandles += int(execution.relocations)
        deviations.extend(float(value) for value in execution.delivery_deviations)
        illegal_drops += int(execution.illegal_drops)
        macro_failures += int(not execution.option_success)
        decisions.append(
            {
                "candidate_key": decision.candidate.key,
                "action_type": execution.action_type,
                "mode": execution.mode,
                "forced_first_action": bool(first),
                "liveness_forced": bool(decision.liveness_forced),
                "duration": int(execution.duration),
                "physical_rehandles": int(execution.relocations),
                "delivery_deviations": list(execution.delivery_deviations),
            }
        )
        consecutive_defer = _next_defer_count(consecutive_defer, decision, execution)
        boundary = bool(
            execution.env_terminal
            or execution.truncated
            or not execution.option_success
            or execution.duration == 0
        )
        next_snapshot = None
        if not boundary:
            next_snapshot, frontier = _enumerate(env, consecutive_defer, arm, cache)
            exact_frontiers = bool(
                exact_frontiers and frontier["complete_frontier_exactly_verified"]
            )
            if not next_snapshot.candidates:
                boundary = True
                failure = "no_exact_safe_candidate"
                next_snapshot = None
        agent.observe_outcome(
            decision,
            next_snapshot=None if boundary else next_snapshot,
            done=boundary,
        )
        if not execution.option_success and failure is None:
            failure = f"macro_failure:{execution.failure_reason or 'unknown'}"
        pending = next_snapshot
        first = False
        if boundary:
            break

    terminal = bool(env.is_state_terminal(env.current_state))
    if steps >= MAX_STEPS_AFTER_CAPTURE and not terminal and failure is None:
        failure = "continuation_step_limit"
    strict = bool(
        terminal
        and failure is None
        and macro_failures == 0
        and illegal_drops == 0
        and len(deviations) == required_remaining
        and exact_frontiers
    )
    first_realized = int(decisions[0]["physical_rehandles"])
    if first_realized != int(selected_row["immediate_rehandle"]):
        raise E16BError("forced macro violated the exact immediate-rehandle contract")
    timing = (
        _timing(deviations)
        if strict
        else {"mean_absolute_error": None, "within_target_window_rate": None}
    )
    return {
        "branch": branch,
        "forced_candidate_key": decisions[0]["candidate_key"],
        "forced_action_type": selected_row["action_type"],
        "forced_mode": selected_row["mode"],
        "predicted_future_rehandles": float(selected_row["predicted_future_rehandles"]),
        "forced_immediate_rehandles": first_realized,
        "realized_future_rehandles_after_forced_macro": int(future_rehandles),
        "realized_total_rehandles_from_state": int(total_rehandles),
        "prediction_error": float(
            selected_row["predicted_future_rehandles"] - future_rehandles
        ),
        "strict_safe_complete": strict,
        "failure_reason": failure,
        "required_deliveries_after_capture": required_remaining,
        "deliveries_after_capture": len(deviations),
        "steps_after_capture": steps,
        "return_after_capture": total_return,
        **timing,
        "liveness_forced_continuation_decisions": guard_forced,
        "macro_decisions_after_capture": len(decisions),
        "illegal_drops": illegal_drops,
        "macro_failures": macro_failures,
        "complete_frontiers_exactly_verified": exact_frontiers,
        "decision_trace": decisions,
        "final_state": benchmark._environment_signature(env),
    }


def _pair_summary(branches: Mapping[str, Mapping]) -> dict:
    immediate = branches["immediate_only_winner"]
    future = branches["fixed_future_winner"]
    predicted_delta = float(
        immediate["predicted_future_rehandles"]
        - future["predicted_future_rehandles"]
    )
    realized_delta = int(
        immediate["realized_future_rehandles_after_forced_macro"]
        - future["realized_future_rehandles_after_forced_macro"]
    )
    both_complete = bool(
        immediate["strict_safe_complete"] and future["strict_safe_complete"]
    )
    return {
        "predicted_immediate_minus_future_candidate": predicted_delta,
        "realized_future_rehandles_immediate_minus_future_candidate": realized_delta,
        "future_prediction_order_correct": realized_delta > 0,
        "realized_future_rehandles_tied": realized_delta == 0,
        "pair_prediction_delta_error": predicted_delta - realized_delta,
        "both_branches_strict_safe_complete": both_complete,
        "future_minus_immediate_completion": int(future["strict_safe_complete"])
        - int(immediate["strict_safe_complete"]),
        "future_minus_immediate_total_rehandles": int(
            future["realized_total_rehandles_from_state"]
            - immediate["realized_total_rehandles_from_state"]
        ),
        "future_minus_immediate_future_rehandles": int(
            future["realized_future_rehandles_after_forced_macro"]
            - immediate["realized_future_rehandles_after_forced_macro"]
        ),
        "future_minus_immediate_steps": (
            int(future["steps_after_capture"] - immediate["steps_after_capture"])
            if both_complete
            else None
        ),
        "future_minus_immediate_mae": (
            float(future["mean_absolute_error"] - immediate["mean_absolute_error"])
            if both_complete
            else None
        ),
    }


def _state_digest(env) -> str:
    return _digest(benchmark._environment_signature(env))


def _run_coordinate(
    output: Path,
    contract: Mapping,
    manifest: Mapping,
    regime_id: str,
    model_seed: int,
    *,
    device: torch.device,
) -> dict:
    path = _ledger_path(output, regime_id, model_seed)
    if path.is_file():
        ledger = _load_json(path, "E16-B coordinate ledger")
        _verify_hash(ledger, "ledger_sha256", "E16-B coordinate ledger")
        if ledger.get("contract_sha256") != contract["contract_sha256"]:
            raise E16BError("E16-B coordinate ledger binding changed")
        return ledger

    arms = _frozen_arms()
    arm = arms[model_seed]
    searched = []
    case = None
    with reuse.path_cleanup_active():
        for instance_seed in INSTANCE_SEEDS:
            instance = _instance(output, manifest, regime_id, instance_seed)
            env = e11.REGIME_BY_ID[regime_id].make_env()
            env.current_episode = 1
            env.reset(instance=instance)
            agent, _ = _new_agent(model_seed, device, arms=arms)
            cache = reuse.TimingInvariantCertificateCache()
            consecutive_defer = 0
            steps = 0
            decisions = 0
            crossing = None
            pending = None
            while steps < MAX_STEPS_AFTER_CAPTURE and not env.is_state_terminal(
                env.current_state
            ):
                if pending is None:
                    pending, _frontier = _enumerate(env, consecutive_defer, arm, cache)
                snapshot = pending
                pending = None
                if not snapshot.candidates:
                    break
                if not all(
                    candidate.certificate.status is ViabilityStatus.SAFE
                    for candidate in snapshot.candidates
                ):
                    raise E16BError("scout frontier contains a non-SAFE candidate")
                scores = _score_frontier(agent, snapshot)
                immediate_key = scores["rows"][scores["immediate_index"]]["candidate_key"]
                future_key = scores["rows"][scores["future_index"]]["candidate_key"]
                if not scores["liveness_forced"] and immediate_key != future_key:
                    expected_keys = {
                        "immediate_only_winner": immediate_key,
                        "fixed_future_winner": future_key,
                    }
                    selected_rows = {
                        "immediate_only_winner": scores["rows"][scores["immediate_index"]],
                        "fixed_future_winner": scores["rows"][scores["future_index"]],
                    }
                    captured_env, captured_snapshot = deepcopy((env, snapshot))
                    guard_state = deepcopy(
                        agent.base_agent.recovery_witness_guard.state_dict()
                    )
                    branches = {
                        branch: _run_branch(
                            branch=branch,
                            model_seed=model_seed,
                            arm=arm,
                            captured_env=captured_env,
                            captured_snapshot=captured_snapshot,
                            guard_state=guard_state,
                            consecutive_defer=consecutive_defer,
                            expected_keys=expected_keys,
                            device=device,
                            arms=arms,
                        )
                        for branch in ("immediate_only_winner", "fixed_future_winner")
                    }
                    crossing = {
                        "regime_id": regime_id,
                        "model_seed": model_seed,
                        "instance_seed": instance_seed,
                        "episode_instance_id": instance.instance_id,
                        "decision_index": decisions,
                        "primitive_steps_before_capture": steps,
                        "state_digest": _state_digest(env),
                        "candidate_count": len(scores["rows"]),
                        "guard_active_at_capture": bool(guard_state.get("active_witness")),
                        "liveness_forced_at_capture": False,
                        "selected_candidates": selected_rows,
                        "branches": branches,
                        "pair": _pair_summary(branches),
                    }
                    break

                decision = agent.select(
                    snapshot,
                    preference_lambda=DEPLOYMENT_LAMBDA,
                    training=False,
                    epsilon=0.0,
                )
                if decision.candidate.key != future_key:
                    raise E16BError("scout future selection disagrees with score audit")
                execution = execute_certified_macro(
                    env,
                    decision.candidate,
                    gamma=agent.config.gamma,
                    remaining_steps=MAX_STEPS_AFTER_CAPTURE - steps,
                    evaluation=True,
                )
                steps += int(execution.duration)
                decisions += 1
                consecutive_defer = _next_defer_count(consecutive_defer, decision, execution)
                boundary = bool(
                    execution.env_terminal
                    or execution.truncated
                    or not execution.option_success
                    or execution.duration == 0
                )
                if not boundary:
                    pending, _frontier = _enumerate(env, consecutive_defer, arm, cache)
                    if not pending.candidates:
                        boundary = True
                        pending = None
                agent.observe_outcome(
                    decision,
                    next_snapshot=None if boundary else pending,
                    done=boundary,
                )
                if boundary:
                    break
            searched.append(
                {
                    "instance_seed": instance_seed,
                    "episode_instance_id": instance.instance_id,
                    "decisions_examined": decisions + int(crossing is not None),
                    "primitive_steps_examined": steps,
                    "crossing_found": crossing is not None,
                }
            )
            print(
                json.dumps(
                    {
                        "regime": regime_id,
                        "model_seed": model_seed,
                        "scout_seed": instance_seed,
                        "decisions": decisions,
                        "crossing": crossing is not None,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if crossing is not None:
                case = crossing
                break

    ledger = _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "status": "crossing_captured" if case is not None else "no_crossing",
            "regime_id": regime_id,
            "model_seed": model_seed,
            "scouts": searched,
            "case": case,
            "training_runs": 0,
        },
        "ledger_sha256",
    )
    _atomic_json(path, ledger)
    return ledger


def run(output: Path, *, pilot_only: bool, device_name: str) -> dict:
    prepare(output)
    contract, manifest = authenticate(output.resolve())
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise E16BError("CUDA requested but unavailable")
    coordinates = (
        ((PILOT_REGIME_ID, PILOT_MODEL_SEED),)
        if pilot_only
        else tuple(
            (regime_id, seed)
            for regime_id in REGIME_IDS
            for seed in MODEL_SEEDS
        )
    )
    captured = 0
    for index, (regime_id, seed) in enumerate(coordinates, 1):
        ledger = _run_coordinate(
            output.resolve(),
            contract,
            manifest,
            regime_id,
            seed,
            device=device,
        )
        captured += int(ledger["case"] is not None)
        print(
            f"E16-B {index}/{len(coordinates)} | {regime_id} seed={seed} "
            f"| {ledger['status']}",
            flush=True,
        )
    return {
        "status": "pilot_complete" if pilot_only else "run_complete",
        "coordinates": len(coordinates),
        "crossings_captured": captured,
        "training_runs": 0,
    }


def _all_ledgers(output: Path) -> list[dict]:
    result = []
    for path in sorted((output / "run-ledger").glob("**/model-seed-*.json")):
        value = _load_json(path, "E16-B coordinate ledger")
        _verify_hash(value, "ledger_sha256", "E16-B coordinate ledger")
        result.append(value)
    return result


def _aggregate_cases(cases: Sequence[Mapping]) -> dict:
    pairs = [case["pair"] for case in cases]
    both = [pair for pair in pairs if pair["both_branches_strict_safe_complete"]]
    return {
        "captured_cases": len(cases),
        "branch_rollouts": 2 * len(cases),
        "both_branches_strict_safe_complete": len(both),
        "future_prediction_order_correct": sum(
            bool(pair["future_prediction_order_correct"]) for pair in pairs
        ),
        "realized_future_rehandles_tied": sum(
            bool(pair["realized_future_rehandles_tied"]) for pair in pairs
        ),
        "mean_predicted_immediate_minus_future_candidate": (
            fmean(pair["predicted_immediate_minus_future_candidate"] for pair in pairs)
            if pairs
            else None
        ),
        "mean_realized_future_rehandles_immediate_minus_future_candidate": (
            fmean(
                pair["realized_future_rehandles_immediate_minus_future_candidate"]
                for pair in pairs
            )
            if pairs
            else None
        ),
        "mean_future_minus_immediate_total_rehandles": (
            fmean(pair["future_minus_immediate_total_rehandles"] for pair in pairs)
            if pairs
            else None
        ),
        "mean_future_minus_immediate_mae_complete_pairs": (
            fmean(pair["future_minus_immediate_mae"] for pair in both)
            if both
            else None
        ),
        "mean_future_minus_immediate_steps_complete_pairs": (
            fmean(pair["future_minus_immediate_steps"] for pair in both)
            if both
            else None
        ),
    }


def _table(report: Mapping) -> str:
    lines = [
        "# E16-B paired continuation bank",
        "",
        "| Regime | Seed | Scout seed | Decision | Immediate-only candidate | Future candidate | Predicted saving | Realized future saving | Both complete | Delta MAE | Delta total rehandles |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        pair = case["pair"]
        immediate = case["selected_candidates"]["immediate_only_winner"]
        future = case["selected_candidates"]["fixed_future_winner"]
        delta_mae = pair["future_minus_immediate_mae"]
        lines.append(
            f"| {case['regime_id']} | {case['model_seed']} | "
            f"{case['instance_seed']} | {case['decision_index']} | "
            f"{immediate['action_type']} `{immediate['candidate_key']}` | "
            f"{future['action_type']} `{future['candidate_key']}` | "
            f"{pair['predicted_immediate_minus_future_candidate']:.3f} | "
            f"{pair['realized_future_rehandles_immediate_minus_future_candidate']} | "
            f"{int(pair['both_branches_strict_safe_complete'])} | "
            f"{'--' if delta_mae is None else f'{delta_mae:+.3f}'} | "
            f"{pair['future_minus_immediate_total_rehandles']:+d} |"
        )
    lines.extend(
        [
            "",
            "Predicted and realized savings are immediate-only candidate minus future-selected candidate; positive values favor the learned-future ranking.",
            "Delta outcome columns are future-selected branch minus immediate-only branch; negative values favor the learned-future ranking.",
            "Only the first candidate is forced. Both branches then use the identical frozen fixed-future controller.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(output: Path, *, allow_partial: bool) -> dict:
    output = output.resolve()
    contract, manifest = authenticate(output)
    ledgers = _all_ledgers(output)
    expected = len(MODEL_SEEDS) * len(REGIME_IDS)
    if not allow_partial and len(ledgers) != expected:
        raise E16BError(f"E16-B grid incomplete: {len(ledgers)}/{expected}")
    if any(
        item.get("contract_sha256") != contract["contract_sha256"]
        for item in ledgers
    ):
        raise E16BError("E16-B ledger contract mismatch")
    cases = [item["case"] for item in ledgers if item.get("case") is not None]
    grouped = defaultdict(list)
    for case in cases:
        grouped[case["regime_id"]].append(case)
    report = _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "status": "complete" if len(ledgers) == expected else "partial",
            "paper_evidence": len(ledgers) == expected,
            "expected_coordinates": expected,
            "observed_coordinates": len(ledgers),
            "coordinates_without_crossing": sum(
                item.get("case") is None for item in ledgers
            ),
            "cases": cases,
            "aggregate": _aggregate_cases(cases),
            "by_regime": {
                regime_id: _aggregate_cases(grouped[regime_id])
                for regime_id in REGIME_IDS
            },
            "claim_boundary": (
                "bounded_descriptive_continuation_interventions_on_earliest_"
                "predeclared_visited_crossings_not_an_unbiased_episode_average"
            ),
            "training_runs": 0,
        },
        "report_sha256",
    )
    _atomic_json(output / REPORT_NAME, report)
    _atomic_json(
        output / CASE_ROWS_NAME,
        {
            "protocol": PROTOCOL,
            "report_sha256": report["report_sha256"],
            "cases": cases,
        },
    )
    _atomic_text(output / TABLE_NAME, _table(report))
    return {
        "status": report["status"],
        "coordinates": len(ledgers),
        "crossings_captured": len(cases),
        "branch_rollouts": 2 * len(cases),
        "report": str(output / REPORT_NAME),
        "table": str(output / TABLE_NAME),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("prepare", "run-pilot", "run-all", "analyze", "analyze-partial"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command in ("run-pilot", "run-all"):
        result = run(
            args.output,
            pilot_only=args.command == "run-pilot",
            device_name=args.device,
        )
    else:
        result = analyze(
            args.output, allow_partial=args.command == "analyze-partial"
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
