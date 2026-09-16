#!/usr/bin/env python3
"""E3: paired recoverability-certificate candidate-source ablation.

The frozen final conditioned controller, deterministic hierarchical selector,
and serialized E1 EpisodeInstances are held fixed.  The only deployment-side
factor is whether executable physical candidates are filtered by the exact
recoverability certificate before scoring.  Exact certificates are still
computed in the physical-only arm as shadow labels; they have no eligibility
authority there.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from statistics import fmean
import sys
from typing import Mapping, Optional, Sequence
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import benchmark_viability_critic_priority as benchmark
from example.Options.DirectDeliverOption import DirectDeliverOption
from example.Options.ExplicitAcceptOption import ExplicitAcceptOption
from example.Options.ReconfigureOption import ReconfigureOption
from PSLAP.neutral_protocol import shared_candidate_mask
from PSLAP.viability import (
    RecoveryActionKind,
    ViabilityStatus,
    apply_recovery_action,
    legal_recovery_actions,
)
import PSLAP.viability_candidates as viability_candidates
from PSLAP.viability_candidates import (
    ViabilityActionType,
    ViabilityCandidateSnapshot,
    ViabilityCertificateCache,
    ViabilityMode,
)
from PSLAP.viability_filter import post_accept_recovery_state
import render_vcg_unified_behavior_gifs as behavior_view
import run_vcg_conditioned_final_comparison_90k as final90
import run_vcg_v11_conditioned_handling_seed0_85k as conditioned_seed0
import run_vcg_v11_nested_handling_pilot as pilot
from viability_graph_preference_conditioned import select_hierarchical_index


PROTOCOL = "vcg_conditioned_e03_certificate_candidate_source_ablation_90k_v2"
SCHEMA_VERSION = 1
MODEL_SEEDS = final90.MODEL_SEEDS
INSTANCE_SEEDS = final90.INSTANCE_SEEDS
PREFERENCE_LAMBDA = 0.1
EXPECTED_BLOCKS = final90.EXPECTED_BLOCKS
MAX_STEPS = final90.MAX_STEPS

PHYSICAL = "physical_feasibility_only"
CERTIFIED = "recoverability_certified"
CANDIDATE_SOURCES = (PHYSICAL, CERTIFIED)

CONTRACT_NAME = "e03-contract.json"
REPORT_NAME = "e03-report.json"
TABLE_NAME = "e03-results-table.md"
COMPANION_NAME = "e03-companion-case.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results/vcg-conditioned-e03-certification-ablation-90k-v2"
)
E1_OUTPUT = final90.DEFAULT_OUTPUT

PHYSICAL_INTERFACE = "e03_executable_physical_frontier_v1"
SHADOW_CONTRACT = "exact_recoverability_shadow_label_no_eligibility_authority_v1"
ELIGIBILITY_PROXY_REASON = "e03_physical_eligibility_proxy"


class E3Error(RuntimeError):
    pass


def _canonical_bytes(value: Mapping) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: Mapping, *, hash_field: Optional[str] = None) -> str:
    payload = dict(value)
    if hash_field is not None:
        payload.pop(hash_field, None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _with_hash(value: Mapping, field: str) -> dict:
    result = dict(value)
    result[field] = _digest(result)
    return result


def _load_json(path: Path, *, label: str) -> dict:
    if not path.is_file() or path.is_symlink():
        raise E3Error(f"missing canonical {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E3Error(f"invalid {label}") from error
    if not isinstance(value, dict):
        raise E3Error(f"{label} must contain an object")
    return value


def _verify_hash(value: Mapping, field: str, *, label: str) -> None:
    if value.get(field) != _digest(value, hash_field=field):
        raise E3Error(f"{label} self hash mismatch")


def _atomic_json(path: Path, value: Mapping) -> None:
    final90._atomic_json(path, value)


@dataclass(frozen=True)
class PhysicalCandidate:
    """Executor-compatible candidate carrying a non-authoritative shadow label."""

    key: str
    mode: ViabilityMode
    action_type: ViabilityActionType
    target_label: Optional[str]
    source: Optional[tuple[int, int]]
    destination: Optional[tuple[int, int]]
    successor_state: object
    certificate: object
    shadow_certificate: object
    option: object
    recovery_action: object = None
    horizon_steps: Optional[int] = None
    recovery_rank_before: Optional[int] = None
    recovery_rank_after: Optional[int] = None

    @property
    def rank_delta(self) -> Optional[int]:
        if self.recovery_rank_before is None or self.recovery_rank_after is None:
            return None
        return self.recovery_rank_before - self.recovery_rank_after


def _shadow_certificate(candidate):
    return getattr(candidate, "shadow_certificate", candidate.certificate)


def _eligibility_certificate(shadow):
    if shadow.status is ViabilityStatus.SAFE:
        return shadow
    return replace(
        shadow,
        status=ViabilityStatus.SAFE,
        reason=(
            f"{ELIGIBILITY_PROXY_REASON};shadow_status={shadow.status.value};"
            f"shadow_reason={shadow.reason}"
        ),
    )


def _proxy(
    *,
    key: str,
    mode: ViabilityMode,
    action_type: ViabilityActionType,
    target_label,
    source,
    destination,
    successor_state,
    shadow_certificate,
    option,
    recovery_action=None,
    horizon_steps=None,
    recovery_rank_before=None,
    recovery_rank_after=None,
) -> PhysicalCandidate:
    return PhysicalCandidate(
        key=key,
        mode=mode,
        action_type=action_type,
        target_label=target_label,
        source=source,
        destination=destination,
        successor_state=successor_state,
        certificate=_eligibility_certificate(shadow_certificate),
        shadow_certificate=shadow_certificate,
        option=option,
        recovery_action=recovery_action,
        horizon_steps=horizon_steps,
        recovery_rank_before=recovery_rank_before,
        recovery_rank_after=recovery_rank_after,
    )


def _proxy_safe(candidate) -> PhysicalCandidate:
    return _proxy(
        key=candidate.key,
        mode=candidate.mode,
        action_type=candidate.action_type,
        target_label=candidate.target_label,
        source=candidate.source,
        destination=candidate.destination,
        successor_state=candidate.successor_state,
        shadow_certificate=candidate.certificate,
        option=candidate.option,
        recovery_action=candidate.recovery_action,
        horizon_steps=candidate.horizon_steps,
        recovery_rank_before=candidate.recovery_rank_before,
        recovery_rank_after=candidate.recovery_rank_after,
    )


def _physical_snapshot(
    env,
    certified: ViabilityCandidateSnapshot,
    *,
    search_config,
    cache: ViabilityCertificateCache,
    max_replans: int = 8,
) -> ViabilityCandidateSnapshot:
    """Restore every executable physical candidate from the shadow cache."""

    safe_by_key = {candidate.key: candidate for candidate in certified.candidates}
    result = []
    inbound = viability_candidates._strict_inbound(env)
    current_rank = certified.audit.current_recovery_rank

    if inbound is not None and not certified.audit.terminal:
        inbound_view = viability_candidates._inbound_view(env, inbound)
        cells = shared_candidate_mask(
            certified.yard,
            inbound_view,
            tuple(inbound.position),
        )
        for cell in cells:
            key = f"accept:{inbound.label}:{cell[0]}:{cell[1]}"
            if key in safe_by_key:
                result.append(_proxy_safe(safe_by_key[key]))
                continue
            option = ExplicitAcceptOption(
                env, inbound.label, cell, max_replans=max_replans
            )
            if not option.initiation(env.get_current_state()):
                continue
            successor = post_accept_recovery_state(
                env,
                certified.yard,
                inbound_view,
                cell,
                reserve_queue_cells=search_config.reserve_queue_cells,
            )
            shadow = cache.get(
                viability_candidates._certificate_key(successor, search_config)
            )
            if shadow is None:
                raise E3Error("physical Accept successor lacks a shadow certificate")
            result.append(
                _proxy(
                    key=key,
                    mode=ViabilityMode.ACCEPT,
                    action_type=ViabilityActionType.ACCEPT,
                    target_label=str(inbound.label),
                    source=tuple(inbound.position),
                    destination=tuple(cell),
                    successor_state=successor,
                    shadow_certificate=shadow,
                    option=option,
                    recovery_rank_before=current_rank,
                    recovery_rank_after=None,
                )
            )

    if not certified.audit.terminal:
        for action in legal_recovery_actions(certified.recovery_state):
            successor = apply_recovery_action(certified.recovery_state, action)
            key = viability_candidates._recovery_candidate_key(action)
            if key in safe_by_key:
                result.append(_proxy_safe(safe_by_key[key]))
                continue
            shadow = cache.get(
                viability_candidates._certificate_key(successor, search_config)
            )
            if shadow is None:
                raise E3Error("physical recovery successor lacks a shadow certificate")
            if action.kind is RecoveryActionKind.DELIVERY:
                action_type = ViabilityActionType.DELIVER
                option = DirectDeliverOption.from_recovery_action(
                    env,
                    action,
                    max_replans=max_replans,
                    fixed_obstacles=certified.recovery_state.fixed_obstacles,
                )
            elif action.kind is RecoveryActionKind.RELOCATION:
                action_type = ViabilityActionType.RECONFIGURE
                option = ReconfigureOption.from_recovery_action(
                    env,
                    action,
                    max_replans=max_replans,
                    fixed_obstacles=certified.recovery_state.fixed_obstacles,
                )
            else:  # pragma: no cover - closed enum
                raise E3Error("unknown physical recovery action")
            if not option.initiation(env.get_current_state()):
                continue
            result.append(
                _proxy(
                    key=key,
                    mode=ViabilityMode.RECOVER,
                    action_type=action_type,
                    target_label=action.block_label,
                    source=action.source,
                    destination=action.destination,
                    successor_state=successor,
                    shadow_certificate=shadow,
                    option=option,
                    recovery_action=action,
                    recovery_rank_before=current_rank,
                    recovery_rank_after=None,
                )
            )

    # Defer remains governed by the same bounded-liveness rule in both arms.
    result.extend(
        _proxy_safe(candidate)
        for candidate in certified.candidates
        if candidate.action_type is ViabilityActionType.DEFER
    )
    keys = tuple(candidate.key for candidate in result)
    if len(keys) != len(set(keys)):
        raise E3Error("physical frontier contains duplicate candidates")
    safe_keys = tuple(
        candidate.key
        for candidate in result
        if candidate.shadow_certificate.status is ViabilityStatus.SAFE
    )
    expected_safe = tuple(candidate.key for candidate in certified.candidates)
    if safe_keys != expected_safe:
        raise E3Error("physical frontier does not preserve the certified subsequence")

    type_counts = Counter(candidate.action_type for candidate in result)
    audit = replace(
        certified.audit,
        interface=PHYSICAL_INTERFACE,
        certification_contract=SHADOW_CONTRACT,
        verifier_authority="none_for_eligibility_exact_shadow_for_diagnosis",
        candidate_count=len(result),
        accept_candidate_count=type_counts[ViabilityActionType.ACCEPT],
        deliver_candidate_count=type_counts[ViabilityActionType.DELIVER],
        reconfigure_candidate_count=type_counts[ViabilityActionType.RECONFIGURE],
        defer_candidate_count=type_counts[ViabilityActionType.DEFER],
        rank_reducing_recovery_count=sum(
            candidate.mode is ViabilityMode.RECOVER
            and candidate.rank_delta is not None
            and candidate.rank_delta > 0
            for candidate in result
        ),
    )
    return ViabilityCandidateSnapshot(
        episode_instance_id=certified.episode_instance_id,
        decision_epoch=certified.decision_epoch,
        agent_position=certified.agent_position,
        yard=certified.yard,
        recovery_state=certified.recovery_state,
        current_certificate=certified.current_certificate,
        inbound_label=certified.inbound_label,
        candidates=tuple(result),
        audit=audit,
    )


def _empty_copy(snapshot: ViabilityCandidateSnapshot) -> ViabilityCandidateSnapshot:
    audit = replace(
        snapshot.audit,
        candidate_count=0,
        accept_candidate_count=0,
        deliver_candidate_count=0,
        reconfigure_candidate_count=0,
        defer_candidate_count=0,
        rank_reducing_recovery_count=0,
    )
    return replace(snapshot, candidates=(), audit=audit)


def _candidate_record(candidate, merit: float) -> dict:
    shadow = _shadow_certificate(candidate)
    return {
        "key": candidate.key,
        "mode": candidate.mode.value,
        "action_type": candidate.action_type.value,
        "target_label": candidate.target_label,
        "source": candidate.source,
        "destination": candidate.destination,
        "merit": float(merit),
        "shadow_status": shadow.status.value,
        "shadow_reason": shadow.reason,
        "shadow_exhaustive": bool(shadow.exhaustive),
    }


class EpisodeTracker:
    def __init__(self) -> None:
        self.selection_count = 0
        self.selected_statuses = Counter()
        self.frontier_count = 0
        self.physical_candidates = 0
        self.safe_candidates = 0
        self.unsafe_candidates = 0
        self.unknown_candidates = 0
        self.physical_empty_frontiers = 0
        self.safe_empty_frontiers = 0
        self.selection_divergences = 0
        self.events = []
        self.pending_selection = None
        self.stop_after_non_safe = False
        self.blocking_event = None
        self.post_failure_frontier = None
        self.latest_before = None

    def record_frontier(self, env, physical: ViabilityCandidateSnapshot) -> None:
        statuses = Counter(
            _shadow_certificate(candidate).status for candidate in physical.candidates
        )
        self.frontier_count += 1
        self.physical_candidates += len(physical.candidates)
        self.safe_candidates += statuses[ViabilityStatus.SAFE]
        self.unsafe_candidates += statuses[ViabilityStatus.UNSAFE]
        self.unknown_candidates += statuses[ViabilityStatus.UNKNOWN]
        self.physical_empty_frontiers += int(not physical.candidates)
        self.safe_empty_frontiers += int(statuses[ViabilityStatus.SAFE] == 0)
        self.latest_before = behavior_view._snapshot(env)

    def record_selection(self, snapshot, decision, within_temperatures) -> None:
        prepared = decision.prepared_snapshot
        base_decision = getattr(decision, "base_decision", decision)
        merits = tuple(float(value) for value in base_decision.q_values)
        if len(merits) != len(prepared.records):
            raise E3Error("selector merit vector does not align with the frontier")
        records = []
        safe_positions = []
        selected_position = None
        for position, (source_index, merit) in enumerate(
            zip(prepared.source_indices, merits)
        ):
            candidate = snapshot.candidates[source_index]
            records.append(_candidate_record(candidate, merit))
            if _shadow_certificate(candidate).status is ViabilityStatus.SAFE:
                safe_positions.append(position)
            if candidate.key == decision.candidate.key:
                selected_position = position
        if selected_position is None:
            raise E3Error("selected candidate is absent from prepared frontier")

        certified_position = None
        if safe_positions:
            if decision.liveness_forced:
                if selected_position not in safe_positions:
                    raise E3Error("liveness guard selected a shadow-unsafe candidate")
                certified_position = selected_position
            else:
                selection = select_hierarchical_index(
                    [merits[position] for position in safe_positions],
                    [prepared.records[position].mode_id for position in safe_positions],
                    within_temperatures,
                    candidate_keys=tuple(
                        prepared.records[position].key for position in safe_positions
                    ),
                )
                certified_position = safe_positions[selection.selected_index]

        selected = records[selected_position]
        certified = (
            None if certified_position is None else records[certified_position]
        )
        diverges = certified is None or selected["key"] != certified["key"]
        self.selection_count += 1
        self.selected_statuses[selected["shadow_status"]] += 1
        self.selection_divergences += int(diverges)
        event = {
            "decision_index": self.selection_count - 1,
            "decision_epoch": int(snapshot.decision_epoch),
            "current_recovery_status": snapshot.current_certificate.status.value,
            "physical_candidate_count": len(records),
            "certified_candidate_count": len(safe_positions),
            "selected": selected,
            "certified_counterfactual": certified,
            "selection_diverges": bool(diverges),
            "liveness_forced": bool(decision.liveness_forced),
            "before": self.latest_before,
        }
        self.pending_selection = event
        if diverges or selected["shadow_status"] != ViabilityStatus.SAFE.value:
            self.events.append(event)

    def record_execution(self, env, candidate, execution) -> None:
        event = self.pending_selection
        if event is None or event["selected"]["key"] != candidate.key:
            raise E3Error("selected candidate and macro execution drifted")
        event["duration"] = int(execution.duration)
        event["option_success"] = bool(execution.option_success)
        event["physical_rehandles"] = int(execution.relocations)
        event["after"] = behavior_view._snapshot(env)
        shadow = _shadow_certificate(candidate)
        if shadow.status is not ViabilityStatus.SAFE and execution.option_success:
            unfinished = tuple(
                sorted(
                    str(block.label)
                    for block in env.blocks
                    if block.stored and not block.delivered
                )
            )
            self.stop_after_non_safe = True
            self.blocking_event = {
                **event,
                "selected_shadow_status": shadow.status.value,
                "self_blocking": shadow.status is ViabilityStatus.UNSAFE,
                "unresolved_successor": shadow.status is ViabilityStatus.UNKNOWN,
                "unsafe_admission": bool(
                    shadow.status is ViabilityStatus.UNSAFE
                    and candidate.action_type is ViabilityActionType.ACCEPT
                ),
                "unfinished_accepted_workloads": unfinished,
            }
        self.pending_selection = None

    def audit_dict(self) -> dict:
        return {
            "frontier_count": self.frontier_count,
            "physical_candidates_exposed": self.physical_candidates,
            "safe_candidates_exposed": self.safe_candidates,
            "unsafe_candidates_exposed": self.unsafe_candidates,
            "unknown_candidates_exposed": self.unknown_candidates,
            "physical_empty_frontiers": self.physical_empty_frontiers,
            "safe_empty_frontiers": self.safe_empty_frontiers,
            "selection_divergences": self.selection_divergences,
            "selected_shadow_status_counts": dict(self.selected_statuses),
            "post_failure_frontier": self.post_failure_frontier,
            "blocking_event": self.blocking_event,
            "diagnostic_events": tuple(self.events),
        }


class DiagnosticFixedLambdaAgent:
    def __init__(self, agent, tracker: EpisodeTracker) -> None:
        self.agent = agent
        self.tracker = tracker
        self.config = agent.config

    def reset_episode_state(self):
        return self.agent.reset_episode_state()

    def select(self, snapshot, *, training=False, epsilon=0.0):
        decision = self.agent.select(
            snapshot, training=training, epsilon=epsilon
        )
        within_temperatures = self.agent.agent.base_agent.within_temperatures
        self.tracker.record_selection(
            snapshot, decision, within_temperatures
        )
        return decision

    def observe_outcome(self, decision, *, next_snapshot, done):
        return self.agent.observe_outcome(
            decision, next_snapshot=next_snapshot, done=done
        )


def _source_contract(project_root: Path) -> dict:
    e1_contract = final90.authenticate_contract(project_root, E1_OUTPUT)
    manifest = final90.authenticate_manifest(project_root, E1_OUTPUT)
    auth = final90._authenticate_inputs(project_root)["conditioned"]
    return {
        "e1_contract_sha256": e1_contract["contract_sha256"],
        "e1_manifest_sha256": manifest["manifest_sha256"],
        "conditioned_terminal_sha256": {
            str(seed): auth["terminal_sha256"][seed] for seed in MODEL_SEEDS
        },
    }


def _contract(project_root: Path, output_dir: Path) -> dict:
    source = _source_contract(project_root)
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "prepared",
        "scientific_question": "does_exact_recoverability_candidate_filtering_matter",
        "candidate_sources": list(CANDIDATE_SOURCES),
        "only_experimental_factor": "candidate_eligibility_source",
        "model_seeds": list(MODEL_SEEDS),
        "instance_seeds": list(INSTANCE_SEEDS),
        "preference_lambda": PREFERENCE_LAMBDA,
        "new_physical_rows": len(MODEL_SEEDS) * len(INSTANCE_SEEDS),
        "certified_control_rows_reused_from_e1": True,
        "training_or_learning": False,
        "checkpoint_selection": False,
        "instance_selection": False,
        "same_frozen_checkpoint_within_pair": True,
        "same_deterministic_hierarchical_selector_within_pair": True,
        "same_recovery_witness_liveness_guard_within_pair": True,
        "physical_arm_shadow_certification": {
            "computed_for_diagnosis": True,
            "eligibility_authority": False,
            "unsafe_and_unknown_rank_features": "rank_after_unavailable",
            "safe_candidate_features_identical_to_certified_arm": True,
            "stop_after_first_non_safe_realized_successor": True,
        },
        "primary_metrics": [
            "strict_completion",
            "recoverability_deadlock",
            "self_blocking",
            "empty_certified_frontier_after_selection",
            "unsafe_admission",
            "accepted_workloads_rendered_impossible",
        ],
        "secondary_metrics": [
            "dense_return",
            "mean_absolute_error",
            "steps",
            "physical_rehandles_per_100_required_deliveries",
        ],
        "secondary_metrics_for_incomplete_arm": "strict_completers_only_descriptive",
        "output_dir": str(output_dir.resolve()),
        **source,
        "source_sha256": {"run.py": _sha256(Path(__file__).resolve())},
    }
    return _with_hash(semantic, "contract_sha256")


def prepare(project_root: Path, output_dir: Path) -> dict:
    expected = _contract(project_root, output_dir)
    path = output_dir / CONTRACT_NAME
    if path.is_file():
        observed = _load_json(path, label="E3 contract")
        _verify_hash(observed, "contract_sha256", label="E3 contract")
        if observed != expected:
            raise E3Error("E3 contract, sources, or parent evidence changed")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise E3Error("nonempty E3 output has no contract")
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(path, expected)
    return {
        "status": "prepared",
        "new_runs": len(MODEL_SEEDS) * len(INSTANCE_SEEDS),
        "certified_rows_reused": len(MODEL_SEEDS) * len(INSTANCE_SEEDS),
        "contract": str(path.resolve()),
    }


def authenticate_contract(project_root: Path, output_dir: Path) -> dict:
    observed = _load_json(output_dir / CONTRACT_NAME, label="E3 contract")
    _verify_hash(observed, "contract_sha256", label="E3 contract")
    if observed != _contract(project_root, output_dir):
        raise E3Error("E3 contract, sources, or parent evidence changed")
    return observed


def _ledger_path(output_dir: Path, model_seed: int, instance_seed: int) -> Path:
    return (
        output_dir
        / "run-ledger"
        / PHYSICAL
        / f"seed-{model_seed}"
        / f"instance-{instance_seed}.json"
    )


def _timing_or_none(deviations: Sequence[float]) -> dict:
    if len(deviations) != EXPECTED_BLOCKS:
        return {
            "mean_absolute_error": None,
            "within_target_window_rate": None,
        }
    timing = final90._timing(deviations)
    return {
        "mean_absolute_error": timing["mean_absolute_error"],
        "within_target_window_rate": timing["within_target_window_rate"],
    }


def _run_physical_row(
    project_root: Path,
    instance,
    record: Mapping,
    *,
    model_seed: int,
    arm,
    auth: Mapping,
    device: torch.device,
) -> dict:
    tracker = EpisodeTracker()
    original_enumerator = benchmark.enumerate_viability_candidates
    original_execute = benchmark.execute_certified_macro

    def physical_enumerator(env, **kwargs):
        cache = kwargs.get("cache")
        if not isinstance(cache, ViabilityCertificateCache):
            raise E3Error("E3 requires the episode's exact certificate cache")
        certified = original_enumerator(env, **kwargs)
        physical = _physical_snapshot(
            env,
            certified,
            search_config=kwargs["search_config"],
            cache=cache,
            max_replans=int(kwargs.get("max_replans", 8)),
        )
        tracker.record_frontier(env, physical)
        if tracker.stop_after_non_safe:
            statuses = Counter(
                candidate.shadow_certificate.status.value
                for candidate in physical.candidates
            )
            tracker.post_failure_frontier = {
                "current_recovery_status": certified.current_certificate.status.value,
                "physical_candidate_count": len(physical.candidates),
                "certified_candidate_count": statuses[ViabilityStatus.SAFE.value],
                "unsafe_candidate_count": statuses[ViabilityStatus.UNSAFE.value],
                "unknown_candidate_count": statuses[ViabilityStatus.UNKNOWN.value],
            }
            return _empty_copy(physical)
        return physical

    def traced_execute(env, candidate, **kwargs):
        execution = original_execute(env, candidate, **kwargs)
        tracker.record_execution(env, candidate, execution)
        return execution

    def factory(base):
        conditioned = final90._load_conditioned_agent(
            project_root,
            auth,
            model_seed=model_seed,
            base=base,
            device=device,
        )
        conditioned.set_epsilon(0.0)
        fixed = conditioned_seed0._FixedLambdaAgent(
            conditioned, PREFERENCE_LAMBDA
        )
        return DiagnosticFixedLambdaAgent(fixed, tracker)

    with (
        patch.object(
            benchmark, "enumerate_viability_candidates", physical_enumerator
        ),
        patch.object(benchmark, "execute_certified_macro", traced_execute),
    ):
        raw = pilot._run_raw(
            arm, instance, device=device, wrapper_factory=factory
        )

    audit = tracker.audit_dict()
    blocking = audit["blocking_event"]
    self_blocking = bool(blocking and blocking["self_blocking"])
    unresolved = bool(blocking and blocking["unresolved_successor"])
    strict = bool(
        raw["strict_method_success"]
        and raw["terminal"]
        and raw["method_failure_reason"] is None
        and blocking is None
        and len(raw["delivery_deviations"]) == EXPECTED_BLOCKS
    )
    if self_blocking:
        failure_reason = "self_blocking_nonrecoverable_successor"
    elif unresolved:
        failure_reason = "selected_unknown_recoverability_successor"
    else:
        failure_reason = raw["method_failure_reason"]
    post = audit["post_failure_frontier"] or {}
    unfinished = () if blocking is None else tuple(
        blocking["unfinished_accepted_workloads"]
    )
    legacy_return = float(raw["return"])
    _legacy_check, dense_return = pilot.normalized._dual_rescore_from_legacy_return(
        legacy_return,
        raw["delivery_deviations"],
        pilot.v11.FROZEN_OBJECTIVE_SPEC,
    )
    row = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "candidate_source": PHYSICAL,
        **final90._identity(record),
        "model_seed": int(model_seed),
        "preference_lambda": PREFERENCE_LAMBDA,
        "checkpoint_sha256": auth["conditioned"]["terminal_sha256"][model_seed],
        "strict_completion": strict,
        "recoverability_deadlock": self_blocking,
        "self_blocking": self_blocking,
        "unresolved_successor": unresolved,
        "empty_certified_frontier_after_selection": bool(
            blocking is not None and int(post.get("certified_candidate_count", 0)) == 0
        ),
        "empty_physical_frontier": bool(
            audit["physical_empty_frontiers"]
            or (
                blocking is not None
                and int(post.get("physical_candidate_count", 0)) == 0
            )
        ),
        "unsafe_admission": bool(blocking and blocking["unsafe_admission"]),
        "accepted_workloads_rendered_impossible": bool(
            self_blocking and unfinished
        ),
        "unfinished_accepted_workloads": list(unfinished),
        "method_failure_reason": failure_reason,
        "dense_return": float(dense_return),
        "legacy_environment_return": legacy_return,
        **_timing_or_none(raw["delivery_deviations"]),
        "steps": int(raw["steps"]),
        "physical_storage_relocations": int(raw["relocations"]),
        "physical_rehandles_per_100_required_deliveries": float(
            100.0 * int(raw["relocations"]) / EXPECTED_BLOCKS
        ),
        "completed_deliveries": len(raw["delivery_deviations"]),
        "required_deliveries": EXPECTED_BLOCKS,
        "illegal_drops": int(raw["illegal_drops"]),
        "macro_failures": int(raw["macro_failures"]),
        "behavior_digest": raw["behavior_digest"],
        "shadow_audit": audit,
        "training_or_learning": False,
    }
    return row


def _validate_physical_row(
    row: Mapping, record: Mapping, *, model_seed: int
) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "candidate_source": PHYSICAL,
        **final90._identity(record),
        "model_seed": model_seed,
        "preference_lambda": PREFERENCE_LAMBDA,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise E3Error(f"physical ledger binding changed: {key}")
    for key in (
        "strict_completion",
        "recoverability_deadlock",
        "self_blocking",
        "unresolved_successor",
        "empty_certified_frontier_after_selection",
        "empty_physical_frontier",
        "unsafe_admission",
        "accepted_workloads_rendered_impossible",
    ):
        if type(row.get(key)) is not bool:
            raise E3Error(f"physical row has invalid Boolean metric: {key}")
    if row["strict_completion"] and row["method_failure_reason"] is not None:
        raise E3Error("strict physical row has a failure reason")
    if row["self_blocking"] != row["recoverability_deadlock"]:
        raise E3Error("E3 exact deadlock and self-blocking labels disagree")
    if row["unsafe_admission"] and not row["self_blocking"]:
        raise E3Error("unsafe admission must be a self-blocking event")


def run_physical(
    project_root: Path,
    output_dir: Path,
    *,
    selected_seed: Optional[int] = None,
    instance_limit: Optional[int] = None,
) -> dict:
    contract = authenticate_contract(project_root, output_dir)
    manifest = final90.authenticate_manifest(project_root, E1_OUTPUT)
    auth = final90._authenticate_inputs(project_root)
    seeds = MODEL_SEEDS if selected_seed is None else (selected_seed,)
    if any(seed not in MODEL_SEEDS for seed in seeds):
        raise E3Error("invalid conditioned model seed")
    records = list(manifest["instances"])
    if instance_limit is not None:
        if instance_limit < 1 or instance_limit > len(records):
            raise E3Error("instance limit is out of range")
        records = records[:instance_limit]
    device = torch.device("cpu")
    completed = strict = blocked = 0
    total = len(seeds) * len(records)
    for model_seed in seeds:
        arm = auth["conditioned"]["inputs"]["arms"][model_seed]
        for record in records:
            instance_seed = int(record["seed"])
            path = _ledger_path(output_dir, model_seed, instance_seed)
            if path.is_file():
                ledger = _load_json(path, label="E3 physical ledger")
                _verify_hash(ledger, "ledger_sha256", label="E3 physical ledger")
                if ledger.get("contract_sha256") != contract["contract_sha256"]:
                    raise E3Error("physical ledger contract binding changed")
                row = ledger["run"]
                _validate_physical_row(row, record, model_seed=model_seed)
            else:
                instance = final90._load_instance(E1_OUTPUT, record)
                row = _run_physical_row(
                    project_root,
                    instance,
                    record,
                    model_seed=model_seed,
                    arm=arm,
                    auth=auth,
                    device=device,
                )
                _validate_physical_row(row, record, model_seed=model_seed)
                ledger = _with_hash(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "protocol": PROTOCOL,
                        "contract_sha256": contract["contract_sha256"],
                        "e1_manifest_sha256": manifest["manifest_sha256"],
                        "candidate_source": PHYSICAL,
                        "model_seed": model_seed,
                        "instance_seed": instance_seed,
                        "run": row,
                    },
                    "ledger_sha256",
                )
                _atomic_json(path, ledger)
            completed += 1
            strict += int(row["strict_completion"])
            blocked += int(row["self_blocking"])
            print(
                f"E3 physical {completed}/{total} | strict={strict} | "
                f"self_blocking={blocked}",
                flush=True,
            )
    return {
        "status": "complete",
        "rows": completed,
        "strict": strict,
        "self_blocking": blocked,
        "model_seeds": list(seeds),
        "instance_limit": instance_limit,
    }


def _physical_rows(
    output_dir: Path,
    manifest: Mapping,
    *,
    allow_partial: bool,
) -> list[dict]:
    records = {int(record["seed"]): record for record in manifest["instances"]}
    rows = []
    missing = []
    for model_seed in MODEL_SEEDS:
        for instance_seed in INSTANCE_SEEDS:
            path = _ledger_path(output_dir, model_seed, instance_seed)
            if not path.is_file():
                missing.append((model_seed, instance_seed))
                continue
            ledger = _load_json(path, label="E3 physical ledger")
            _verify_hash(ledger, "ledger_sha256", label="E3 physical ledger")
            row = dict(ledger["run"])
            _validate_physical_row(
                row, records[instance_seed], model_seed=model_seed
            )
            rows.append(row)
    if missing and not allow_partial:
        raise E3Error(f"E3 physical grid is incomplete: {len(missing)} rows missing")
    return rows


def _certified_row(record: Mapping, *, model_seed: int) -> dict:
    path = final90._conditioned_ledger_path(
        E1_OUTPUT, PREFERENCE_LAMBDA, model_seed, int(record["seed"])
    )
    ledger = final90._load_json(path, label="E1 certified control ledger")
    final90._verify_hash(
        ledger, "ledger_sha256", label="E1 certified control ledger"
    )
    row = dict(ledger["run"])
    final90._validate_conditioned_row(
        row,
        identity=final90._identity(record),
        value=PREFERENCE_LAMBDA,
        seed=model_seed,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "candidate_source": CERTIFIED,
        **final90._identity(record),
        "model_seed": model_seed,
        "preference_lambda": PREFERENCE_LAMBDA,
        "checkpoint_sha256": row["checkpoint_sha256"],
        "strict_completion": bool(row["strict_safe_complete"]),
        "recoverability_deadlock": False,
        "self_blocking": False,
        "unresolved_successor": False,
        "empty_certified_frontier_after_selection": False,
        "empty_physical_frontier": False,
        "unsafe_admission": False,
        "accepted_workloads_rendered_impossible": False,
        "dense_return": row["dense_return"],
        "mean_absolute_error": row["mean_absolute_error"],
        "within_target_window_rate": row["within_target_window_rate"],
        "steps": row["steps"],
        "physical_storage_relocations": row["physical_storage_relocations"],
        "physical_rehandles_per_100_required_deliveries": row[
            "physical_rehandles_per_100_required_deliveries"
        ],
    }


def _source_summary(rows: Sequence[Mapping]) -> dict:
    rows = tuple(rows)
    complete = tuple(row for row in rows if row["strict_completion"])
    booleans = (
        "strict_completion",
        "recoverability_deadlock",
        "self_blocking",
        "unresolved_successor",
        "empty_certified_frontier_after_selection",
        "empty_physical_frontier",
        "unsafe_admission",
        "accepted_workloads_rendered_impossible",
    )
    result = {
        "model_instance_rows": len(rows),
        **{
            name: {
                "count": sum(int(row[name]) for row in rows),
                "rate": (
                    None
                    if not rows
                    else float(fmean(int(row[name]) for row in rows))
                ),
            }
            for name in booleans
        },
        "strict_completer_secondary_metrics": None,
    }
    if complete:
        result["strict_completer_secondary_metrics"] = {
            "denominator": len(complete),
            "dense_return": float(fmean(row["dense_return"] for row in complete)),
            "mean_absolute_error": float(
                fmean(row["mean_absolute_error"] for row in complete)
            ),
            "within_target_window_rate": float(
                fmean(row["within_target_window_rate"] for row in complete)
            ),
            "steps": float(fmean(row["steps"] for row in complete)),
            "physical_rehandles_per_100_required_deliveries": float(
                fmean(
                    row["physical_rehandles_per_100_required_deliveries"]
                    for row in complete
                )
            ),
        }
    return result


def _companion(rows: Sequence[Mapping]) -> Optional[dict]:
    cases = []
    for row in rows:
        event = row["shadow_audit"].get("blocking_event")
        if not event or event.get("selected_shadow_status") != "UNSAFE":
            continue
        alternative = event.get("certified_counterfactual")
        if alternative is None or alternative["key"] == event["selected"]["key"]:
            continue
        gap = float(event["selected"]["merit"]) - float(alternative["merit"])
        cases.append(
            {
                "model_seed": int(row["model_seed"]),
                "instance_seed": int(row["instance_seed"]),
                "episode_instance_id": row["episode_instance_id"],
                "decision_index": int(event["decision_index"]),
                "critic_merit_gap_selected_minus_certified": gap,
                "critic_value_prefers_unsafe": gap > 0.0,
                "event": event,
            }
        )
    if not cases:
        return None
    preferred = [case for case in cases if case["critic_value_prefers_unsafe"]]
    pool = preferred or cases
    selected = max(
        pool,
        key=lambda case: (
            case["critic_merit_gap_selected_minus_certified"],
            -case["instance_seed"],
            -case["model_seed"],
        ),
    )
    return {
        "protocol": PROTOCOL,
        "selection_rule": (
            "largest_positive_unsafe_vs_certified_merit_gap; deterministic "
            "instance/model tie break"
        ),
        "post_hoc_explanatory_case_not_performance_evidence": True,
        **selected,
    }


def _fmt_count(metric: Mapping, denominator: int) -> str:
    return f"{int(metric['count'])}/{denominator}"


def _fmt_metric(metrics: Optional[Mapping], name: str) -> str:
    if metrics is None:
        return "—"
    return f"{float(metrics[name]):.2f}"


def _table(summaries: Mapping[str, Mapping]) -> str:
    lines = [
        "# E3 certification candidate-source ablation",
        "",
        "Secondary performance metrics are descriptive among strict completers only.",
        "",
        "| Candidate source | Strict completion | Recoverability deadlock | Self-blocking | Empty certified frontier | Unsafe admission | Accepted workloads impossible | MAE* | Rehandles/100* |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        CERTIFIED: "Recoverability-certified",
        PHYSICAL: "Physical-feasibility-only",
    }
    for source in (CERTIFIED, PHYSICAL):
        summary = summaries[source]
        denominator = int(summary["model_instance_rows"])
        metrics = summary["strict_completer_secondary_metrics"]
        lines.append(
            "| "
            + " | ".join(
                (
                    labels[source],
                    _fmt_count(summary["strict_completion"], denominator),
                    _fmt_count(summary["recoverability_deadlock"], denominator),
                    _fmt_count(summary["self_blocking"], denominator),
                    _fmt_count(
                        summary["empty_certified_frontier_after_selection"],
                        denominator,
                    ),
                    _fmt_count(summary["unsafe_admission"], denominator),
                    _fmt_count(
                        summary["accepted_workloads_rendered_impossible"],
                        denominator,
                    ),
                    _fmt_metric(metrics, "mean_absolute_error"),
                    _fmt_metric(
                        metrics,
                        "physical_rehandles_per_100_required_deliveries",
                    ),
                )
            )
            + " |"
        )
    lines.extend(("", "\\* Strict completers only; denominator is reported in JSON."))
    return "\n".join(lines) + "\n"


def analyze(
    project_root: Path,
    output_dir: Path,
    *,
    allow_partial: bool,
) -> dict:
    contract = authenticate_contract(project_root, output_dir)
    manifest = final90.authenticate_manifest(project_root, E1_OUTPUT)
    physical = _physical_rows(
        output_dir, manifest, allow_partial=allow_partial
    )
    records = {int(record["seed"]): record for record in manifest["instances"]}
    certified = [
        _certified_row(records[int(row["instance_seed"])], model_seed=int(row["model_seed"]))
        for row in physical
    ]
    physical_keys = {
        (int(row["model_seed"]), int(row["instance_seed"])) for row in physical
    }
    certified_keys = {
        (int(row["model_seed"]), int(row["instance_seed"])) for row in certified
    }
    if physical_keys != certified_keys:
        raise E3Error("paired candidate-source grid is not identical")
    summaries = {
        CERTIFIED: _source_summary(certified),
        PHYSICAL: _source_summary(physical),
    }
    paired = {
        "pairs": len(physical),
        "certified_success_physical_failure": sum(
            int(c["strict_completion"] and not p["strict_completion"])
            for c, p in zip(certified, physical)
        ),
        "physical_success_certified_failure": sum(
            int(p["strict_completion"] and not c["strict_completion"])
            for c, p in zip(certified, physical)
        ),
    }
    companion = _companion(physical)
    companion_path = None
    if companion is not None:
        companion = _with_hash(companion, "case_sha256")
        companion_path = output_dir / COMPANION_NAME
        _atomic_json(companion_path, companion)
    report = _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": (
                "complete"
                if len(physical) == len(MODEL_SEEDS) * len(INSTANCE_SEEDS)
                else "partial"
            ),
            "contract_sha256": contract["contract_sha256"],
            "preference_lambda": PREFERENCE_LAMBDA,
            "candidate_sources": list(CANDIDATE_SOURCES),
            "summaries": summaries,
            "paired_strict_completion": paired,
            "companion_case": (
                None
                if companion is None
                else {
                    "path": str(companion_path.resolve()),
                    "case_sha256": companion["case_sha256"],
                }
            ),
            "interpretation_limits": {
                "physical_arm_stops_at_first_non_safe_successor": True,
                "unknown_is_not_labeled_deadlock": True,
                "secondary_metrics_are_not_whole_method_comparisons_when_failures_exist": True,
                "shadow_verifier_does_not_filter_physical_candidates": True,
            },
        },
        "report_sha256",
    )
    _atomic_json(output_dir / REPORT_NAME, report)
    (output_dir / TABLE_NAME).write_text(_table(summaries), encoding="utf-8")
    return {
        "status": report["status"],
        "physical_rows": len(physical),
        "certified_rows": len(certified),
        "strict": {
            source: summaries[source]["strict_completion"]
            for source in (CERTIFIED, PHYSICAL)
        },
        "self_blocking": summaries[PHYSICAL]["self_blocking"],
        "companion_case": report["companion_case"],
        "report": str((output_dir / REPORT_NAME).resolve()),
        "table": str((output_dir / TABLE_NAME).resolve()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "run", "analyze", "run-all")
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--instance-limit", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    torch.set_num_threads(1)
    if args.command == "prepare":
        result = prepare(project_root, output_dir)
    elif args.command == "run":
        result = run_physical(
            project_root,
            output_dir,
            selected_seed=args.seed,
            instance_limit=args.instance_limit,
        )
    elif args.command == "analyze":
        result = analyze(
            project_root, output_dir, allow_partial=args.allow_partial
        )
    else:
        prepared = prepare(project_root, output_dir)
        execution = run_physical(project_root, output_dir)
        report = analyze(project_root, output_dir, allow_partial=False)
        result = {"prepare": prepared, "run": execution, "analysis": report}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
