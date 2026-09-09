#!/usr/bin/env python3
"""One frozen E13 replay that records learned proposals under liveness forcing."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from statistics import fmean
import sys
from typing import Mapping, Optional, Sequence
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import benchmark_viability_critic_priority as benchmark
from experiments.conditioned_vcg.E08_liveness_audit_95k import (
    analyze_existing,
)
from experiments.conditioned_vcg.E13_operational_scalability_95k import (
    program as e13,
)
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from experiments.conditioned_vcg.development.D10_scalability_support_screen import (
    occupancy_extension as occupancy,
)
from PSLAP.relocation_family_certification import (
    RELOCATION_FAMILY_CERTIFICATION,
)
from PSLAP.viability_candidates import ViabilityMode
import run_vcg_conditioned_final_comparison_90k as final90
import run_vcg_v11_conditioned_handling_seed0_85k as conditioned_seed0
import run_vcg_v11_nested_handling_pilot as pilot
from vcg_v11_conditioned_handling import detached_v11_features
from viability_graph_hierarchy import ID_TO_MODE, prepare_viability_snapshot
from viability_graph_preference_conditioned import select_hierarchical_index


PROTOCOL = "vcg_conditioned_e8_instrumented_liveness_pilot_95k_v1"
SCHEMA_VERSION = 1
DEFAULT_OUTPUT = analyze_existing.DEFAULT_OUTPUT
E13_OUTPUT = analyze_existing.E13_ROOT
SCENARIO_ID = "size_8x8_occ_high"
INSTANCE_SEED = 95_100
MODEL_SEED = 0
PREFERENCE_LAMBDA = 0.10
SEARCH_MAX_NODES = 20_000
WALL_LIMIT_SECONDS = 600
CONDITIONED_TERMINAL = e13.CONDITIONED_TERMINAL

SOURCE_PATHS = (
    "benchmark_viability_critic_priority.py",
    "vcg_v11_conditioned_handling.py",
    "viability_graph_hierarchy.py",
    "viability_graph_preference_conditioned.py",
    "example/Options/certified_path.py",
    "example/Options/DirectDeliverOption.py",
    "example/Options/ReconfigureOption.py",
    "example/small_rooms_env.py",
    "PSLAP/relocation_family_certification.py",
    "PSLAP/viability_candidates.py",
    "PSLAP/viability_filter.py",
    "experiments/conditioned_vcg/E14_certification_scalability_95k/reuse.py",
    "experiments/conditioned_vcg/E08_liveness_audit_95k/analyze_existing.py",
    "experiments/conditioned_vcg/E08_liveness_audit_95k/instrumented_pilot.py",
)


class E8PilotError(RuntimeError):
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
    if not path.is_file() or path.is_symlink():
        raise E8PilotError(f"missing regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E8PilotError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise E8PilotError(f"{label} must contain an object")
    return value


def _self_hashed(path: Path, field: str, label: str) -> dict:
    value = _load(path, label)
    if value.get(field) != _digest(value, field):
        raise E8PilotError(f"{label} self-hash mismatch")
    return value


def _atomic_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _frozen_parent() -> dict:
    contract_path = E13_OUTPUT / e13.CONTRACT_NAME
    manifest_path = E13_OUTPUT / e13.MANIFEST_NAME
    ledger_path = e13._ledger_path(E13_OUTPUT, SCENARIO_ID, INSTANCE_SEED)
    contract = _self_hashed(contract_path, "contract_sha256", "E13 contract")
    manifest = _self_hashed(manifest_path, "manifest_sha256", "E13 manifest")
    ledger = _self_hashed(ledger_path, "ledger_sha256", "E13 ledger")
    if (
        contract.get("protocol") != e13.PROTOCOL
        or manifest.get("contract_sha256") != contract["contract_sha256"]
        or ledger.get("contract_sha256") != contract["contract_sha256"]
    ):
        raise E8PilotError("E13 frozen-parent binding changed")
    record = e13._record(manifest, SCENARIO_ID, INSTANCE_SEED)
    row = ledger["row"]
    if (
        row.get("scenario_id") != SCENARIO_ID
        or int(row.get("instance_seed", -1)) != INSTANCE_SEED
        or row.get("strict_safe_complete") is not True
        or int(row.get("macro_decisions", -1)) != 42
        or sum(
            bool(item["liveness_forced"])
            for item in row.get("decision_costs", ())
        )
        != 27
    ):
        raise E8PilotError("predeclared E13 guard-heavy parent changed")
    instance_path = e13.PARENT_D10 / record["relative_path"]
    if _sha(instance_path) != record["raw_sha256"]:
        raise E8PilotError("E13 parent instance bytes changed")
    if _sha(CONDITIONED_TERMINAL) != contract["conditioned_terminal_sha256"]:
        raise E8PilotError("frozen conditioned checkpoint changed")
    return {
        "contract": contract,
        "manifest": manifest,
        "ledger": ledger,
        "record": record,
        "paths": {
            "e13_contract": contract_path,
            "e13_manifest": manifest_path,
            "e13_ledger": ledger_path,
            "episode_instance": instance_path,
            "conditioned_terminal": CONDITIONED_TERMINAL,
        },
    }


def _contract(output: Path) -> dict:
    parent = _frozen_parent()
    existing_report_path = output / "e8-existing-log-report.json"
    existing = _self_hashed(
        existing_report_path,
        "report_sha256",
        "E8 existing-log report",
    )
    if (
        existing.get("status") != "complete_existing_log_reanalysis"
        or existing["e13_global"]["macro_decisions"] != 1747
        or existing["e13_global"]["liveness_forced_decisions"] != 663
    ):
        raise E8PilotError("E8 existing-log audit is incomplete")
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "question": (
            "when_the_liveness_guard_activates_does_it_change_the_"
            "contemporaneous_unrestricted_learned_choice"
        ),
        "pilot_is_descriptive_not_guard_off_causal_ablation": True,
        "scenario_id": SCENARIO_ID,
        "instance_seed": INSTANCE_SEED,
        "model_seed": MODEL_SEED,
        "preference_lambda": PREFERENCE_LAMBDA,
        "historical_parent": {
            "macro_decisions": 42,
            "liveness_forced_decisions": 27,
            "episode_wall_seconds": parent["ledger"]["row"][
                "episode_wall_seconds"
            ],
        },
        "selection": {
            "learned_proposal": (
                "same full exact-SAFE frontier, same learned merits, same "
                "normalized hierarchical selector, without witness restriction"
            ),
            "executed_choice": "implemented selector with liveness guard",
            "override": "forced and proposal candidate key differs from executed key",
        },
        "search": {
            "max_nodes": SEARCH_MAX_NODES,
            "wall_limit_seconds": WALL_LIMIT_SECONDS,
            "timing_invariant_cache": True,
            "path_enumeration_cleanup": True,
            "relocation_family_certification": RELOCATION_FAMILY_CERTIFICATION,
        },
        "training_or_checkpoint_selection": False,
        "parents": {
            name: _sha(path) for name, path in parent["paths"].items()
        },
        "e8_existing_log_report_sha256": _sha(existing_report_path),
        "source_sha256": {
            path: _sha(ROOT / path) for path in SOURCE_PATHS
        },
    }
    return _with_hash(semantic, "contract_sha256")


def prepare(output: Path) -> dict:
    output = output.resolve()
    analyze_existing.analyze(output)
    expected = _contract(output)
    path = output / "e8-pilot-contract.json"
    if path.is_file():
        observed = _load(path, "E8 pilot contract")
        if observed != expected:
            raise E8PilotError("E8 pilot contract, sources, or parents changed")
    else:
        _atomic_json(path, expected)
    return {
        "status": "prepared",
        "training_runs": 0,
        "pilot_episode_runs": 1,
        "scenario_id": SCENARIO_ID,
        "instance_seed": INSTANCE_SEED,
        "historical_wall_seconds": expected["historical_parent"][
            "episode_wall_seconds"
        ],
        "output": str(output),
    }


def _authenticate(output: Path) -> dict:
    observed = _self_hashed(
        output / "e8-pilot-contract.json",
        "contract_sha256",
        "E8 pilot contract",
    )
    if observed != _contract(output):
        raise E8PilotError("E8 pilot contract, sources, or parents changed")
    return observed


def _margin(values: Sequence[float], selected_index: int, indices) -> Optional[float]:
    alternatives = [float(values[index]) for index in indices if index != selected_index]
    if not alternatives:
        return None
    return float(values[selected_index]) - max(alternatives)


class ProposalAuditAgent:
    """Observe the unrestricted learned proposal without changing guard state."""

    def __init__(self, fixed_agent) -> None:
        self.fixed_agent = fixed_agent
        self.config = fixed_agent.config
        self.records: list[dict] = []

    @property
    def conditioned(self):
        return self.fixed_agent.agent

    def reset_episode_state(self):
        return self.fixed_agent.reset_episode_state()

    def select(self, snapshot, *, training=False, epsilon=0.0):
        base = self.conditioned.base_agent
        guard = base.recovery_witness_guard
        pre_active = bool(guard.active)
        pre_nonprogress = int(guard.nonprogress_recovery_decisions)
        pre_activations = int(guard.activations)
        due_trigger = bool(
            guard.force_when_due
            and any(
                float(block.remaining_time) <= 0.0
                for block in snapshot.recovery_state.blocks
            )
        )
        nonprogress_trigger = bool(
            pre_nonprogress >= guard.max_nonprogress_recovery_decisions
        )
        has_recovery = any(
            candidate.mode is ViabilityMode.RECOVER
            for candidate in snapshot.candidates
        )

        decision = self.fixed_agent.select(
            snapshot, training=training, epsilon=epsilon
        )
        forced = bool(decision.liveness_forced)
        prepared = prepare_viability_snapshot(
            snapshot,
            guard_context=guard.feature_context(
                snapshot, forced_frontier=forced
            ),
        )
        if not prepared.records:
            raise E8PilotError("learned proposal received an empty SAFE frontier")
        features = detached_v11_features(base.Q_local, prepared.records)
        value = float(self.fixed_agent.value)
        with torch.no_grad():
            operational = base.Q_local.q_head(features).squeeze(-1)
            immediate = torch.as_tensor(
                [
                    1.0 if item.action_type == "reconfigure" else 0.0
                    for item in prepared.records
                ],
                dtype=features.dtype,
                device=features.device,
            )
            future = self.conditioned.handling_network(features, value)
            handling = immediate + future
            merit = operational - value * handling
        hierarchy = select_hierarchical_index(
            merit,
            prepared.mode_ids,
            base.within_temperatures,
            candidate_keys=tuple(item.key for item in prepared.records),
        )
        proposal_index = int(hierarchy.selected_index)
        proposal_source = prepared.source_indices[proposal_index]
        proposal_candidate = snapshot.candidates[proposal_source]
        executed_matches = [
            index
            for index, item in enumerate(prepared.records)
            if item.key == decision.candidate.key
        ]
        if len(executed_matches) != 1:
            raise E8PilotError("executed candidate does not bind to full SAFE frontier")
        executed_index = executed_matches[0]
        mode_values = dict(hierarchy.mode_values)
        selected_mode_value = float(mode_values[hierarchy.selected_mode_id])
        other_mode_values = [
            float(score)
            for mode_id, score in hierarchy.mode_values
            if mode_id != hierarchy.selected_mode_id
        ]
        proposed_mode_indices = [
            index
            for index, mode_id in enumerate(prepared.mode_ids)
            if mode_id == hierarchy.selected_mode_id
        ]
        activation_started = bool(forced and guard.activations > pre_activations)
        if forced and pre_active:
            activation_reason = "persisted_active_witness"
        elif activation_started:
            reasons = []
            if due_trigger:
                reasons.append("due_trigger")
            if nonprogress_trigger:
                reasons.append("nonprogress_limit")
            activation_reason = "+".join(reasons) if reasons else "other_trigger"
        else:
            activation_reason = None
        self.records.append(
            {
                "decision_epoch": int(snapshot.decision_epoch),
                "full_exact_safe_candidate_count": len(prepared.records),
                "full_exact_safe_candidate_keys": [
                    item.key for item in prepared.records
                ],
                "liveness_forced": forced,
                "guard_active_before_select": pre_active,
                "guard_activation_started": activation_started,
                "guard_activation_reason": activation_reason,
                "due_trigger": due_trigger,
                "nonprogress_trigger": nonprogress_trigger,
                "has_recovery_candidate": has_recovery,
                "nonprogress_count_before_select": pre_nonprogress,
                "active_witness_length_after_select": len(guard.active_witness),
                "active_witness_cursor_after_select": int(guard.witness_cursor),
                "learned_proposal_key": proposal_candidate.key,
                "learned_proposal_mode": proposal_candidate.mode.value,
                "learned_proposal_action_type": proposal_candidate.action_type.value,
                "executed_key": decision.candidate.key,
                "executed_mode": decision.candidate.mode.value,
                "executed_action_type": decision.candidate.action_type.value,
                "proposal_agrees_with_executed": (
                    proposal_candidate.key == decision.candidate.key
                ),
                "actual_override": bool(
                    forced and proposal_candidate.key != decision.candidate.key
                ),
                "proposal_operational_value": float(operational[proposal_index]),
                "proposal_immediate_handling": float(immediate[proposal_index]),
                "proposal_future_handling": float(future[proposal_index]),
                "proposal_total_handling": float(handling[proposal_index]),
                "proposal_merit": float(merit[proposal_index]),
                "executed_operational_value": float(operational[executed_index]),
                "executed_immediate_handling": float(immediate[executed_index]),
                "executed_future_handling": float(future[executed_index]),
                "executed_total_handling": float(handling[executed_index]),
                "executed_merit": float(merit[executed_index]),
                "proposal_minus_executed_merit": float(
                    merit[proposal_index] - merit[executed_index]
                ),
                "proposal_mode_margin": (
                    selected_mode_value - max(other_mode_values)
                    if other_mode_values
                    else None
                ),
                "proposal_within_mode_margin": _margin(
                    merit.tolist(), proposal_index, proposed_mode_indices
                ),
                "learned_mode_values": {
                    ID_TO_MODE[int(mode_id)]: float(score)
                    for mode_id, score in hierarchy.mode_values
                },
            }
        )
        return decision

    def observe_outcome(self, decision, *, next_snapshot, done):
        return self.fixed_agent.observe_outcome(
            decision, next_snapshot=next_snapshot, done=done
        )


def _counter(records, field: str, *, where=lambda _item: True) -> dict:
    return dict(
        sorted(Counter(str(item[field]) for item in records if where(item)).items())
    )


def _mean(records, field: str) -> Optional[float]:
    values = [float(item[field]) for item in records if item[field] is not None]
    return fmean(values) if values else None


def run(output: Path) -> dict:
    output = output.resolve()
    prepare(output)
    contract = _authenticate(output)
    report_path = output / "e8-instrumented-pilot-report.json"
    if report_path.is_file():
        report = _self_hashed(report_path, "report_sha256", "E8 pilot report")
        if report.get("contract_sha256") != contract["contract_sha256"]:
            raise E8PilotError("E8 pilot report binding changed")
        return report

    parent = _frozen_parent()
    inputs = e13._parent_inputs()
    conditioned = inputs["conditioned"]
    arm = conditioned["inputs"]["arms"][MODEL_SEED]
    instance = e13._load_instance(parent["record"])
    scenario = occupancy.SCENARIO_BY_ID[SCENARIO_ID]
    holder = []

    def agent_factory(base):
        agent = final90._load_conditioned_agent(
            ROOT,
            {"conditioned": conditioned},
            model_seed=MODEL_SEED,
            base=base,
            device=torch.device("cpu"),
        )
        agent.set_epsilon(0.0)
        fixed = conditioned_seed0._FixedLambdaAgent(agent, PREFERENCE_LAMBDA)
        audited = ProposalAuditAgent(fixed)
        holder.append(audited)
        return audited

    def env_factory(_payload):
        return occupancy.OccupancyTrackingEnv(scenario)

    try:
        cache = reuse.TimingInvariantCertificateCache()
        with (
            e13._WallClock(WALL_LIMIT_SECONDS),
            reuse.path_cleanup_active(),
            patch.object(benchmark, "_make_env", env_factory),
            pilot._agent_factory(agent_factory),
            patch.object(benchmark, "ViabilityCertificateCache", lambda: cache),
        ):
            raw = benchmark.run_arm(
                arm=benchmark.EXACT_FULL,
                controller_payload=arm.payload,
                instance=instance,
                instance_seed=int(instance.seed),
                search_config=e13._search_config(arm, SEARCH_MAX_NODES),
                liveness_rule=benchmark._liveness_rule(arm.payload),
                prioritizer=None,
                max_steps=scenario.max_steps,
                device=torch.device("cpu"),
                recovery_certification_strategy=RELOCATION_FAMILY_CERTIFICATION,
            )
    except e13.EpisodeWallLimit as error:
        raise E8PilotError(str(error)) from error
    if len(holder) != 1:
        raise E8PilotError("instrumented agent was not constructed exactly once")
    records = holder[0].records
    decisions = list(raw["decisions"])
    if len(records) != len(decisions):
        raise E8PilotError("proposal and executed decision records do not align")
    for index, (record, decision) in enumerate(zip(records, decisions)):
        if (
            record["decision_epoch"] != int(decision["decision_epoch"])
            or record["executed_key"] != decision["selected_key"]
            or record["liveness_forced"] != bool(decision["liveness_forced"])
        ):
            raise E8PilotError("proposal instrumentation changed decision alignment")
        record.update(
            {
                "decision_index": index,
                "duration": int(decision["duration"]),
                "physical_storage_relocations": int(
                    decision["physical_storage_relocations"]
                ),
                "option_success": bool(decision["option_success"]),
                "frontier_index": int(decision["frontier_index"]),
            }
        )

    forced = [item for item in records if item["liveness_forced"]]
    unforced = [item for item in records if not item["liveness_forced"]]
    overrides = [item for item in forced if item["actual_override"]]
    activations = [item for item in forced if item["guard_activation_started"]]
    strict = bool(
        raw["strict_method_success"]
        and raw["terminal"]
        and raw["method_failure_reason"] is None
        and raw["complete_frontier_exactly_verified"]
        and raw["illegal_drops"] == 0
        and raw["macro_failures"] == 0
        and len(raw["delivery_deviations"]) == scenario.total_jobs
    )
    historical = parent["ledger"]["row"]
    report = _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "status": "complete",
            "strict_safe_complete": strict,
            "model_seed": MODEL_SEED,
            "preference_lambda": PREFERENCE_LAMBDA,
            "scenario_id": SCENARIO_ID,
            "instance_seed": INSTANCE_SEED,
            "macro_decisions": len(records),
            "liveness_forced_decisions": len(forced),
            "liveness_forced_rate": len(forced) / len(records),
            "actual_overrides": len(overrides),
            "actual_override_rate_among_forced": (
                len(overrides) / len(forced) if forced else None
            ),
            "forced_agreements_with_learned_proposal": len(forced) - len(overrides),
            "guard_activations": len(activations),
            "guard_persistence_decisions": len(forced) - len(activations),
            "guard_activation_reasons": _counter(
                activations, "guard_activation_reason"
            ),
            "forced_proposal_action_types": _counter(
                forced, "learned_proposal_action_type"
            ),
            "forced_executed_action_types": _counter(
                forced, "executed_action_type"
            ),
            "forced_proposal_to_executed": _counter(
                [
                    {
                        "pair": (
                            f"{item['learned_proposal_action_type']}->"
                            f"{item['executed_action_type']}"
                        )
                    }
                    for item in forced
                ],
                "pair",
            ),
            "forced_outcomes": {
                "primitive_steps": sum(item["duration"] for item in forced),
                "physical_relocations": sum(
                    item["physical_storage_relocations"] for item in forced
                ),
            },
            "unforced_outcomes": {
                "primitive_steps": sum(item["duration"] for item in unforced),
                "physical_relocations": sum(
                    item["physical_storage_relocations"] for item in unforced
                ),
            },
            "override_margins": {
                "mean_proposal_minus_executed_merit": _mean(
                    overrides, "proposal_minus_executed_merit"
                ),
                "mean_proposal_mode_margin": _mean(
                    overrides, "proposal_mode_margin"
                ),
                "mean_proposal_within_mode_margin": _mean(
                    overrides, "proposal_within_mode_margin"
                ),
            },
            "recovery_certification": {
                "strategy": raw.get("recovery_certification_strategy"),
                "relocation_family_attempt_count": raw.get(
                    "relocation_family_attempt_count"
                ),
                "relocation_family_proof_count": raw.get(
                    "relocation_family_proof_count"
                ),
                "relocation_family_miss_count": raw.get(
                    "relocation_family_miss_count"
                ),
                "native_recovery_search_count": raw.get(
                    "native_recovery_search_count"
                ),
            },
            "historical_e13_comparison": {
                "historical_behavior_digest": historical["behavior_digest"],
                "corrected_behavior_digest": raw["behavior_digest"],
                "behavior_digest_equal": (
                    historical["behavior_digest"] == raw["behavior_digest"]
                ),
                "historical_macro_decisions": historical["macro_decisions"],
                "historical_liveness_forced_decisions": sum(
                    bool(item["liveness_forced"])
                    for item in historical["decision_costs"]
                ),
            },
            "decisions": records,
            "interpretation": {
                "override": (
                    "guard forced a different candidate than the same learned "
                    "selector proposed on the full exact-SAFE frontier"
                ),
                "not_causal": (
                    "this observes implemented attribution; it is not a "
                    "with-versus-without-guard rollout comparison"
                ),
            },
        },
        "report_sha256",
    )
    _atomic_json(report_path, report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "run":
        result = run(args.output)
    else:
        result = _authenticate(args.output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
