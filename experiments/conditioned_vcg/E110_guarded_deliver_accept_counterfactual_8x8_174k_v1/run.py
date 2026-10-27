"""Capture and replay matched guarded Deliver-versus-Accept continuations."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import gc
import json
import math
import os
from pathlib import Path
from statistics import mean, median
from types import SimpleNamespace

import torch

import benchmark_viability_critic_priority as benchmark
import run_vcg_objective_gamma_audit as audit
from PSLAP.viability_candidates import (
    BoundedEventDeferRule,
    ViabilityActionType,
    ViabilityMode,
)
from PSLAP.viability_dataset import recovery_state_to_dict
from PSLAP.viability_filter import ViabilitySearchConfig
from experiments.conditioned_vcg.E14_certification_scalability_95k import (
    reuse,
)
from experiments.conditioned_vcg.E105_wait_aware_stage_a_8x8_169k_v1.rewards import (
    WAIT_CAP,
    WaitAwareDenseEnv,
    episode_service_metrics,
)
from experiments.conditioned_vcg.E107_queue_context_stage_a_8x8_171k_v1.representation import (
    QueueContextAgent,
    deduplicated_queue_scores,
)
from viability_graph_hierarchy import ID_TO_MODE, regularized_mode_values

from . import protocol as p


class E110Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E110Error(message)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    require(path.is_file(), f"missing E110 artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid E110 artifact: {path}")
    return value


def prepare(output: Path) -> dict:
    output = output.resolve()
    expected = p.contract()
    path = output / "contract.json"
    if path.is_file():
        require(read_json(path) == expected, "E110 contract, code, or source changed")
    else:
        require(
            not output.exists() or not any(output.iterdir()),
            "E110 refuses a nonempty uncontracted output",
        )
        write_json(path, expected)
    return {
        "status": "prepared",
        "contract_sha256": expected["contract_sha256"],
        "development_seeds": list(p.DEVELOPMENT_SEEDS),
        "anchor_episode": p.ANCHOR_EPISODE,
        "late_episode": p.LATE_EPISODE,
        "maximum_episode_runs": 2 * len(p.DEVELOPMENT_SEEDS),
        "training_runs": 0,
        "episode_runs": 0,
    }


def authenticate(output: Path) -> dict:
    expected = p.contract()
    observed = read_json(output.resolve() / "contract.json")
    require(observed == expected, "E110 contract, code, or source changed")
    return expected


def _frozen_paths(output: Path) -> tuple[Path, Path, Path]:
    root = output.resolve() / "frozen-sources"
    return root / "episode-240.pth", root / "episode-269.pth", root / "manifest.json"


def _write_torch(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def repair_contract(output: Path) -> dict:
    """Migrate the original contract to immutable frozen-source authentication.

    E108's resumable ``latest.pth`` advanced after E110 had already copied the
    intended episode-269 behavior weights.  The old contract correctly records
    the source that was copied, but the first implementation unnecessarily
    re-read the moving E108 path in every later process.  This one-time repair
    preserves and re-hashes those compact copies and any completed rows; it
    never reads a replacement E108 checkpoint and never runs an episode.
    """
    output = output.resolve()
    expected = p.contract()
    contract_path = output / "contract.json"
    observed = read_json(contract_path)
    if observed == expected:
        return {
            "status": "already_repaired",
            "contract_sha256": expected["contract_sha256"],
            "training_runs": 0,
            "episode_runs": 0,
        }
    require(
        observed.get("contract_sha256") == p.ORIGINAL_CONTRACT_SHA256,
        "E110 contract is neither the original nor repair1 contract",
    )
    original_payload = dict(observed)
    original_digest = original_payload.pop("contract_sha256")
    require(
        p.digest(original_payload) == original_digest,
        "E110 original contract digest is invalid",
    )
    require(
        observed.get("source_identity") == p.FROZEN_SOURCE_IDENTITY,
        "E110 original source identity differs from the frozen episode-269 source",
    )

    anchor_path, late_path, manifest_path = _frozen_paths(output)
    manifest = read_json(manifest_path)
    require(
        manifest.get("contract_sha256") == p.ORIGINAL_CONTRACT_SHA256,
        "E110 original frozen manifest contract differs",
    )
    for key, path, episode, role in (
        ("anchor", anchor_path, p.ANCHOR_EPISODE, "episode_240_anchor"),
        ("late", late_path, p.LATE_EPISODE, "episode_269_continuation"),
    ):
        require(path.is_file(), f"E110 repair is missing frozen {key} checkpoint")
        require(
            p.sha256(path) == manifest[key]["sha256"]
            and path.stat().st_size == manifest[key]["bytes"],
            f"E110 frozen {key} checkpoint differs before repair",
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require(
            payload.get("e110_contract_sha256") == p.ORIGINAL_CONTRACT_SHA256
            and payload.get("e110_source_episode") == episode
            and payload.get("e110_frozen_role") == role,
            f"E110 frozen {key} checkpoint provenance differs before repair",
        )
        payload["e110_migrated_from_contract_sha256"] = p.ORIGINAL_CONTRACT_SHA256
        payload["e110_contract_repair"] = "repair1_frozen_source_authentication"
        payload["e110_contract_sha256"] = expected["contract_sha256"]
        _write_torch(path, payload)
        manifest[key]["sha256"] = p.sha256(path)
        manifest[key]["bytes"] = path.stat().st_size

    migrated_captures = []
    for path in sorted((output / "captures").glob("seed-*.json")):
        row = read_json(path)
        require(
            row.get("contract_sha256") == p.ORIGINAL_CONTRACT_SHA256,
            f"E110 capture has unexpected pre-repair contract: {path}",
        )
        row["migrated_from_contract_sha256"] = p.ORIGINAL_CONTRACT_SHA256
        row["contract_sha256"] = expected["contract_sha256"]
        write_json(path, row)
        migrated_captures.append(str(path))

    migrated_branches = []
    for path in sorted((output / "deliver-branches").glob("seed-*.json")):
        row = read_json(path)
        require(
            row.get("contract_sha256") == p.ORIGINAL_CONTRACT_SHA256,
            f"E110 branch has unexpected pre-repair contract: {path}",
        )
        capture_path = _capture_path(output, int(row["instance_seed"]))
        require(capture_path.is_file(), f"E110 branch source capture is missing: {path}")
        row["migrated_from_contract_sha256"] = p.ORIGINAL_CONTRACT_SHA256
        row["contract_sha256"] = expected["contract_sha256"]
        row["source_capture_sha256"] = p.sha256(capture_path)
        write_json(path, row)
        migrated_branches.append(str(path))

    manifest["migrated_from_contract_sha256"] = p.ORIGINAL_CONTRACT_SHA256
    manifest["contract_repair"] = "repair1_frozen_source_authentication"
    manifest["contract_sha256"] = expected["contract_sha256"]
    write_json(manifest_path, manifest)
    write_json(contract_path, expected)
    repair_path = output / "contract-repair1.json"
    repair = {
        "protocol": p.PROTOCOL,
        "status": "complete",
        "reason": "E108 latest advanced after immutable episode-269 weights were frozen",
        "old_contract_sha256": p.ORIGINAL_CONTRACT_SHA256,
        "new_contract_sha256": expected["contract_sha256"],
        "frozen_anchor_sha256": manifest["anchor"]["sha256"],
        "frozen_late_sha256": manifest["late"]["sha256"],
        "migrated_captures": migrated_captures,
        "migrated_branches": migrated_branches,
        "source_episode_replaced": False,
        "episode_runs": 0,
        "training_runs": 0,
    }
    write_json(repair_path, repair)
    return {**repair, "report": str(repair_path)}


def _compact_payload(payload: dict, *, role: str, episode: int, contract: dict) -> dict:
    require(
        int(payload.get("completed_training_episodes", -1)) == episode,
        f"E110 {role} checkpoint episode differs",
    )
    state = dict(payload["agent_state"])
    state.pop("replay", None)
    state.pop("optimizer", None)
    state.pop("rng_state", None)
    result = dict(payload)
    result["agent_state"] = state
    result["e110_frozen_role"] = role
    result["e110_source_episode"] = episode
    result["e110_contract_sha256"] = contract["contract_sha256"]
    return result


def freeze_sources(output: Path) -> dict:
    contract = authenticate(output)
    anchor_path, late_path, manifest_path = _frozen_paths(output)
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        require(
            manifest.get("contract_sha256") == contract["contract_sha256"]
            and anchor_path.is_file()
            and late_path.is_file()
            and p.sha256(anchor_path) == manifest["anchor"]["sha256"]
            and p.sha256(late_path) == manifest["late"]["sha256"],
            "E110 frozen source manifest differs",
        )
        return {
            "status": "already_frozen",
            "manifest": str(manifest_path),
            "training_runs": 0,
            "episode_runs": 0,
        }
    require(
        not anchor_path.exists() and not late_path.exists(),
        "E110 has partial frozen sources",
    )
    anchor = torch.load(p.ANCHOR_SOURCE, map_location="cpu", weights_only=False)
    late = torch.load(p.LATE_SOURCE, map_location="cpu", weights_only=False)
    compact = {
        "anchor": _compact_payload(
            anchor, role="episode_240_anchor", episode=p.ANCHOR_EPISODE,
            contract=contract,
        ),
        "late": _compact_payload(
            late, role="episode_269_continuation", episode=p.LATE_EPISODE,
            contract=contract,
        ),
    }
    anchor_path.parent.mkdir(parents=True, exist_ok=True)
    for name, path in (("anchor", anchor_path), ("late", late_path)):
        temporary = path.with_suffix(".pth.tmp")
        torch.save(compact[name], temporary)
        os.replace(temporary, path)
    manifest = {
        "contract_sha256": contract["contract_sha256"],
        "anchor": {
            "episode": p.ANCHOR_EPISODE,
            "path": str(anchor_path),
            "sha256": p.sha256(anchor_path),
            "bytes": anchor_path.stat().st_size,
            "source_sha256": contract["source_identity"]["anchor"]["sha256"],
        },
        "late": {
            "episode": p.LATE_EPISODE,
            "path": str(late_path),
            "sha256": p.sha256(late_path),
            "bytes": late_path.stat().st_size,
            "source_sha256": contract["source_identity"]["late"]["sha256"],
        },
        "replay_removed": True,
        "optimizer_removed": True,
        "behavior_weights_unchanged": True,
    }
    write_json(manifest_path, manifest)
    return {
        "status": "frozen",
        "manifest": str(manifest_path),
        "anchor_bytes": anchor_path.stat().st_size,
        "late_bytes": late_path.stat().st_size,
        "training_runs": 0,
        "episode_runs": 0,
    }


def _load_frozen(output: Path, device: str) -> tuple[QueueContextAgent, QueueContextAgent, dict]:
    contract = authenticate(output)
    anchor_path, late_path, manifest_path = _frozen_paths(output)
    manifest = read_json(manifest_path)
    require(
        manifest.get("contract_sha256") == contract["contract_sha256"]
        and p.sha256(anchor_path) == manifest["anchor"]["sha256"]
        and p.sha256(late_path) == manifest["late"]["sha256"],
        "E110 frozen checkpoints changed",
    )
    anchor_payload = torch.load(anchor_path, map_location="cpu", weights_only=False)
    late_payload = torch.load(late_path, map_location="cpu", weights_only=False)
    require(
        anchor_payload.get("e110_source_episode") == p.ANCHOR_EPISODE
        and late_payload.get("e110_source_episode") == p.LATE_EPISODE,
        "E110 compact checkpoint roles differ",
    )
    anchor = QueueContextAgent.from_checkpoint(
        anchor_payload, device=device, resumable=False, seed=0
    )
    late = QueueContextAgent.from_checkpoint(
        late_payload, device=device, resumable=False, seed=0
    )
    for agent in (anchor, late):
        agent.Q_local.eval()
        agent.Q_target.eval()
    return anchor, late, manifest


def self_check(output: Path) -> dict:
    prepared = prepare(output)
    anchor, late, manifest = _load_frozen(output, "cpu")
    require(
        math.isclose(anchor.config.gamma, late.config.gamma)
        and anchor.config.to_dict() == late.config.to_dict(),
        "E110 checkpoint configurations differ",
    )
    require(
        tuple(p.DEVELOPMENT_SEEDS) == tuple(range(112_800, 112_805)),
        "E110 development panel changed",
    )
    return {
        "status": "passed",
        "contract_sha256": prepared["contract_sha256"],
        "frozen_source_manifest": str(_frozen_paths(output)[2]),
        "anchor_episode": manifest["anchor"]["episode"],
        "late_episode": manifest["late"]["episode"],
        "same_agent_configuration": True,
        "no_learning": True,
        "training_runs": 0,
        "episode_runs": 0,
    }


def _stable_selection(records, q: torch.Tensor, within_temperatures) -> dict:
    modes = torch.as_tensor(
        [record.mode_id for record in records], dtype=torch.long, device=q.device
    )
    mode_values, live_modes = regularized_mode_values(q, modes, within_temperatures)
    pairs = sorted(
        (
            (int(mode.item()), float(value.item()))
            for mode, value in zip(live_modes, mode_values)
        ),
        key=lambda item: item[0],
    )
    maximum_mode = max(value for _, value in pairs)
    selected_mode = min(mode for mode, value in pairs if value == maximum_mode)
    indices = [
        index for index, record in enumerate(records)
        if int(record.mode_id) == selected_mode
    ]
    maximum_q = max(float(q[index].item()) for index in indices)
    selected_index = min(
        (index for index in indices if float(q[index].item()) == maximum_q),
        key=lambda index: records[index].key,
    )
    by_type = {
        action_type: max(
            float(q[index].item())
            for index, record in enumerate(records)
            if record.action_type == action_type
        )
        for action_type in sorted({record.action_type for record in records})
    }
    delivery = by_type.get("deliver")
    non_delivery = [value for key, value in by_type.items() if key != "deliver"]
    return {
        "selected_index": selected_index,
        "selected_key": records[selected_index].key,
        "selected_mode": ID_TO_MODE[selected_mode],
        "selected_action_type": records[selected_index].action_type,
        "selected_q": float(q[selected_index].item()),
        "delivery_vs_best_non_delivery_margin": (
            None if delivery is None or not non_delivery
            else delivery - max(non_delivery)
        ),
    }


def _state_payload(state) -> dict:
    result = {"recovery_state": recovery_state_to_dict(state)}
    if hasattr(state, "queue_context"):
        result["queue_context"] = list(state.queue_context)
    return result


def _record_bank_digest(records) -> str:
    return p.digest([
        {
            "key": record.key,
            "mode_id": int(record.mode_id),
            "action_type": record.action_type,
            "current": _state_payload(record.current_state),
            "successor": _state_payload(record.successor_state),
            "action_features": list(record.action_features),
        }
        for record in records
    ])


def _guard_digest(agent: QueueContextAgent) -> str:
    state = agent.recovery_witness_guard.state_dict()
    serializable = {key: value for key, value in state.items() if key != "active_witness"}
    serializable["active_witness"] = [
        reuse.action_to_dict(action) for action in state["active_witness"]
    ]
    return p.digest(serializable)


def _environment_digest(env) -> str:
    value = benchmark._environment_signature(env)
    value["episode_instance_id"] = env.current_episode_instance.instance_id
    value["jobs"] = [
        {
            "label": str(block.label),
            "arrival_step": int(block.arrival_step),
            "storage_steps_needed": int(block.storage_steps_needed),
            "delivered_time_step": (
                None if block.delivered_time_step is None
                else int(block.delivered_time_step)
            ),
        }
        for block in sorted(env.blocks, key=lambda item: str(item.label))
    ]
    return p.digest(value)


def _prefix_return(env) -> float:
    return float(sum(
        float(record["selected_step_reward"])
        for record in env.objective_audit_records
    ))


def _force_prepared_index(agent, decision, snapshot, index: int):
    prepared = decision.prepared_snapshot
    source_index = prepared.source_indices[index]
    candidate = snapshot.candidates[source_index]
    record = prepared.records[index]
    require(
        candidate.key == record.key
        and candidate.action_type is ViabilityActionType.DELIVER,
        "E110 forced candidate is not the authenticated Deliver",
    )
    revised = replace(
        decision,
        candidate=candidate,
        record=record,
        selection_source="e110_forced_episode240_deliver",
        explored=False,
        liveness_forced=False,
        exact_rank_progress=bool(
            snapshot.audit.recovery_rank_exact
            and candidate.mode is ViabilityMode.RECOVER
            and candidate.rank_delta is not None
            and candidate.rank_delta > 0
        ),
    )
    native = decision.candidate
    agent.mode_decisions[native.mode.value] -= 1
    agent.action_decisions[native.action_type.value] -= 1
    agent.selection_sources[decision.selection_source] -= 1
    agent.mode_decisions[candidate.mode.value] += 1
    agent.action_decisions[candidate.action_type.value] += 1
    agent.selection_sources[revised.selection_source] += 1
    if agent.decision_log:
        agent.decision_log[-1].update({
            "selected_key": candidate.key,
            "selected_mode": candidate.mode.value,
            "selected_action_type": candidate.action_type.value,
            "selection_source": revised.selection_source,
            "explored": False,
            "liveness_forced": False,
            "exact_rank_progress": revised.exact_rank_progress,
        })
    return revised


class MatchedPolicy:
    """Late frozen policy with read-only anchor scoring and one optional fork."""

    def __init__(self, late, anchor, env, *, expected_fork=None):
        self.late = late
        self.anchor = anchor
        self.env = env
        self.expected_fork = expected_fork
        self.native_keys: list[str] = []
        self.executed_keys: list[str] = []
        self.fork: dict | None = None
        self.interventions = 0

    def __getattr__(self, name):
        return getattr(self.late, name)

    def reset_episode_state(self):
        self.native_keys.clear()
        self.executed_keys.clear()
        self.fork = None
        self.interventions = 0
        return self.late.reset_episode_state()

    def _anchor_selection(self, records) -> dict:
        with torch.no_grad():
            q = deduplicated_queue_scores(records, self.anchor.Q_local)
        return _stable_selection(records, q, self.anchor.within_temperatures)

    def _identity(self, decision, anchor_selection: dict, index: int) -> dict:
        records = tuple(decision.prepared_snapshot.records)
        late_q = torch.as_tensor(
            decision.q_values,
            dtype=torch.float32,
            device=self.late.device,
        )
        late_selection = _stable_selection(
            records, late_q, self.late.within_temperatures
        )
        require(
            late_selection["selected_key"] == decision.candidate.key,
            "E110 late stable selector disagrees with executed native choice",
        )
        deliver_index = int(anchor_selection["selected_index"])
        source_index = decision.prepared_snapshot.source_indices[deliver_index]
        deliver_candidate = self._snapshot.candidates[source_index]
        return {
            "decision_index": index,
            "decision_epoch": int(self._snapshot.decision_epoch),
            "time_steps": int(self.env.time_steps),
            "prefix_return": _prefix_return(self.env),
            "prefix_native_keys": list(self.native_keys),
            "state_sha256": _environment_digest(self.env),
            "candidate_bank_sha256": _record_bank_digest(records),
            "guard_sha256": _guard_digest(self.late),
            "candidate_keys": [record.key for record in records],
            "late_selection": late_selection,
            "anchor_selection": anchor_selection,
            "accept_key": decision.candidate.key,
            "accept_label": decision.candidate.target_label,
            "deliver_key": records[deliver_index].key,
            "deliver_label": deliver_candidate.target_label,
            "frontier_candidates": len(records),
        }

    def select(self, snapshot, *, training=False, epsilon=0.0):
        require(not training and float(epsilon) == 0.0, "E110 forbids learning/exploration")
        self._snapshot = snapshot
        decision = self.late.select(snapshot, training=False, epsilon=0.0)
        index = len(self.native_keys)
        chosen = decision

        if self.expected_fork is not None and index < self.expected_fork["decision_index"]:
            expected = self.expected_fork["prefix_native_keys"][index]
            require(
                decision.candidate.key == expected,
                f"E110 matched prefix diverged at decision {index}",
            )

        should_score = (
            not decision.liveness_forced
            and (
                (self.expected_fork is None and self.fork is None)
                or (
                    self.expected_fork is not None
                    and index == self.expected_fork["decision_index"]
                )
            )
        )
        if should_score:
            anchor_selection = self._anchor_selection(
                decision.prepared_snapshot.records
            )
            disagreement = (
                decision.candidate.action_type is ViabilityActionType.ACCEPT
                and anchor_selection["selected_action_type"] == "deliver"
            )
            if self.expected_fork is None and disagreement:
                self.fork = self._identity(decision, anchor_selection, index)
            elif self.expected_fork is not None and index == self.expected_fork["decision_index"]:
                require(disagreement, "E110 expected Deliver-to-Accept fork disappeared")
                observed = self._identity(decision, anchor_selection, index)
                for key in (
                    "decision_index", "decision_epoch", "time_steps", "prefix_return",
                    "prefix_native_keys", "state_sha256", "candidate_bank_sha256",
                    "guard_sha256", "candidate_keys", "accept_key", "accept_label",
                    "deliver_key", "deliver_label",
                ):
                    require(
                        observed[key] == self.expected_fork[key],
                        f"E110 matched fork differs in {key}",
                    )
                self.fork = observed
                chosen = _force_prepared_index(
                    self.late,
                    decision,
                    snapshot,
                    int(anchor_selection["selected_index"]),
                )
                self.interventions += 1

        if self.expected_fork is not None and index == self.expected_fork["decision_index"]:
            require(self.interventions == 1, "E110 failed to execute its one intervention")
        self.native_keys.append(decision.candidate.key)
        self.executed_keys.append(chosen.candidate.key)
        return chosen


def _search_config() -> ViabilitySearchConfig:
    return ViabilitySearchConfig(
        max_depth=None,
        max_nodes=20_000,
        max_primitive_steps=None,
        reserve_queue_cells=True,
        search_order="goal_directed",
    )


def _liveness_rule() -> BoundedEventDeferRule:
    return BoundedEventDeferRule(
        max_option_steps=10,
        max_consecutive_defer_decisions=16,
    )


def _block_timing(env, labels) -> dict:
    result = {}
    by_label = {str(block.label): block for block in env.blocks}
    for label in labels:
        block = by_label[label]
        stored = block.stored_time_step
        delivered = block.delivered_time_step
        result[label] = {
            "arrival_step": int(block.arrival_step),
            "stored_step": None if stored is None else int(stored),
            "delivered_step": None if delivered is None else int(delivered),
            "required_storage_steps": int(block.storage_steps_needed),
            "wait_before_storage": None if stored is None else int(stored - block.arrival_step),
            "storage_error": (
                None if stored is None or delivered is None
                else int(delivered - stored - block.storage_steps_needed)
            ),
            "release_error": (
                None if delivered is None
                else int(delivered - block.arrival_step - block.storage_steps_needed)
            ),
        }
    return result


def _outcome(run: dict, service: dict, *, prefix_return: float) -> dict:
    return {
        "strict_method_success": bool(run["strict_method_success"]),
        "method_failure_reason": run["method_failure_reason"],
        "return": float(run["return"]),
        "continuation_return": float(run["return"] - prefix_return),
        "steps": int(run["steps"]),
        "macro_decisions": int(run["macro_decisions"]),
        "mean_absolute_error": float(run["mean_absolute_error"]),
        "mean_signed_deviation": float(run["mean_signed_deviation"]),
        "within_target_window_rate": float(run["within_target_window_rate"]),
        "rehandles": int(run["audit"].get("relocations", 0)),
        "selected_action_counts": run["selected_action_counts"],
        "mean_wait_before_storage": service["mean_wait_before_storage"],
        "release_clock_mae": service["release_clock_mae"],
        "storage_clock_mae": service["storage_clock_mae"],
        "storage_clock_signed_error": service["storage_clock_mean_signed_error"],
    }


def _run(output: Path, seed: int, device: str, expected_fork=None):
    require(seed in p.DEVELOPMENT_SEEDS, "E110 seed is outside the frozen panel")
    require(device in ("cpu", "cuda"), "E110 device must be cpu or cuda")
    if device == "cuda":
        require(torch.cuda.is_available(), "E110 CUDA requested but unavailable")
    anchor, late, _ = _load_frozen(output, device)
    env = WaitAwareDenseEnv(p.SCENARIO, p.e107.e105.e78.SPEC)
    env.current_episode = f"E110-{seed}"
    late.bind_context_environment(
        env, wait_cap=WAIT_CAP, required_jobs=36, max_steps=p.MAX_STEPS
    )
    policy = MatchedPolicy(late, anchor, env, expected_fork=expected_fork)
    run = audit.run_objective_audit_episode(
        policy,
        env,
        objective_spec=p.e107.e105.e78.SPEC,
        instance_seed=seed,
        training=False,
        max_steps=p.MAX_STEPS,
        search_config=_search_config(),
        liveness_rule=_liveness_rule(),
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_decisions=1,
        updates_per_macro=0,
        epsilon_warmup_decisions=0,
        run_protocol=p.PROTOCOL,
    )
    service = episode_service_metrics(env)
    return run, service, policy, env


def _capture_path(output: Path, seed: int) -> Path:
    return output.resolve() / "captures" / f"seed-{seed}.json"


def _branch_path(output: Path, seed: int) -> Path:
    return output.resolve() / "deliver-branches" / f"seed-{seed}.json"


def capture(output: Path, seed: int, device: str) -> dict:
    contract = authenticate(output)
    path = _capture_path(output, seed)
    require(not path.exists(), "E110 capture already exists; refusing overwrite")
    run, service, policy, env = _run(output, seed, device)
    fork = policy.fork
    status = "captured" if fork is not None else "no_disagreement"
    row = {
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "status": status,
        "instance_seed": seed,
        "instance_id": run["episode_instance_id"],
        "fork": fork,
        "native_accept": _outcome(
            run, service, prefix_return=(fork["prefix_return"] if fork else 0.0)
        ),
        "fork_block_timing": (
            None if fork is None else _block_timing(
                env, (fork["accept_label"], fork["deliver_label"])
            )
        ),
        "episode_runs": 1,
        "training_runs": 0,
    }
    write_json(path, row)
    return {
        "status": status,
        "seed": seed,
        "fork_decision": None if fork is None else fork["decision_index"],
        "accept_key": None if fork is None else fork["accept_key"],
        "deliver_key": None if fork is None else fork["deliver_key"],
        "strict_complete": bool(run["strict_method_success"]),
        "report": str(path),
        "episode_runs": 1,
        "training_runs": 0,
    }


def run_deliver(output: Path, seed: int, device: str) -> dict:
    contract = authenticate(output)
    capture_path = _capture_path(output, seed)
    source = read_json(capture_path)
    require(
        source.get("contract_sha256") == contract["contract_sha256"]
        and source.get("instance_seed") == seed,
        "E110 capture provenance differs",
    )
    if source["status"] == "no_disagreement":
        return {
            "status": "skipped_no_disagreement",
            "seed": seed,
            "episode_runs": 0,
            "training_runs": 0,
        }
    require(source["status"] == "captured", "E110 capture is incomplete")
    path = _branch_path(output, seed)
    require(not path.exists(), "E110 Deliver branch already exists; refusing overwrite")
    run, service, policy, env = _run(output, seed, device, source["fork"])
    require(policy.interventions == 1, "E110 did not execute exactly one intervention")
    require(
        run["episode_instance_id"] == source["instance_id"],
        "E110 branch EpisodeInstance differs",
    )
    outcome = _outcome(run, service, prefix_return=source["fork"]["prefix_return"])
    row = {
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "status": "complete" if run["strict_method_success"] else "branch_failure_retained",
        "instance_seed": seed,
        "instance_id": run["episode_instance_id"],
        "source_capture": str(capture_path),
        "source_capture_sha256": p.sha256(capture_path),
        "fork": policy.fork,
        "forced_deliver": outcome,
        "fork_block_timing": _block_timing(
            env, (source["fork"]["accept_label"], source["fork"]["deliver_label"])
        ),
        "continuation_return_difference_deliver_minus_accept": (
            outcome["continuation_return"]
            - source["native_accept"]["continuation_return"]
        ),
        "episode_runs": 1,
        "training_runs": 0,
    }
    write_json(path, row)
    return {
        "status": row["status"],
        "seed": seed,
        "continuation_return_difference_deliver_minus_accept": row[
            "continuation_return_difference_deliver_minus_accept"
        ],
        "strict_complete": bool(run["strict_method_success"]),
        "report": str(path),
        "episode_runs": 1,
        "training_runs": 0,
    }


def _difference(branch: dict, capture_row: dict, key: str) -> float:
    return float(branch["forced_deliver"][key]) - float(capture_row["native_accept"][key])


def _render(report: dict) -> str:
    lines = [
        "# E110 guarded Deliver-versus-Accept counterfactual",
        "",
        f"Development instances scanned: **{report['captures']}**; matched disagreements: **{report['matched_pairs']}**.",
        "",
        "| Seed | Decision | Deliver − Accept continuation return | Δ steps | Δ storage MAE | Verdict |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["pairs"]:
        lines.append(
            f"| {row['instance_seed']} | {row['decision_index']} | "
            f"{row['continuation_return_difference_deliver_minus_accept']:.2f} | "
            f"{row['steps_difference']:+.0f} | {row['storage_mae_difference']:+.2f} | "
            f"{row['pair_verdict']} |"
        )
    lines.extend([
        "",
        f"Directional result: **{report['directional_result']}**.",
        "",
        "Positive continuation-return differences mean that executing Deliver once was better under the identical frozen episode-269 guarded continuation.",
        "",
        "This is a development-only one-action causal diagnostic. It does not establish a general policy or deployment advantage.",
        "",
    ])
    return "\n".join(lines)


def analyze(output: Path) -> dict:
    contract = authenticate(output)
    captures = []
    pairs = []
    missing = []
    for seed in p.DEVELOPMENT_SEEDS:
        capture_path = _capture_path(output, seed)
        if not capture_path.is_file():
            missing.append(f"capture:{seed}")
            continue
        source = read_json(capture_path)
        captures.append(source)
        if source["status"] == "no_disagreement":
            continue
        branch_path = _branch_path(output, seed)
        if not branch_path.is_file():
            missing.append(f"deliver:{seed}")
            continue
        branch = read_json(branch_path)
        require(
            branch["contract_sha256"] == contract["contract_sha256"]
            and branch["source_capture_sha256"] == p.sha256(capture_path),
            "E110 pair provenance differs",
        )
        advantage = float(branch["continuation_return_difference_deliver_minus_accept"])
        pairs.append({
            "instance_seed": seed,
            "decision_index": int(source["fork"]["decision_index"]),
            "time_steps": int(source["fork"]["time_steps"]),
            "accept_key": source["fork"]["accept_key"],
            "deliver_key": source["fork"]["deliver_key"],
            "continuation_return_difference_deliver_minus_accept": advantage,
            "steps_difference": _difference(branch, source, "steps"),
            "storage_mae_difference": _difference(
                branch, source, "storage_clock_mae"
            ),
            "wait_difference": _difference(
                branch, source, "mean_wait_before_storage"
            ),
            "release_mae_difference": _difference(
                branch, source, "release_clock_mae"
            ),
            "rehandles_difference": _difference(branch, source, "rehandles"),
            "both_strict_complete": bool(
                source["native_accept"]["strict_method_success"]
                and branch["forced_deliver"]["strict_method_success"]
            ),
            "pair_verdict": (
                "Deliver better" if advantage > 0
                else "Accept better" if advantage < 0
                else "tie"
            ),
        })
    complete = not missing
    advantages = [
        row["continuation_return_difference_deliver_minus_accept"] for row in pairs
    ]
    positives = sum(value > 0 for value in advantages)
    negatives = sum(value < 0 for value in advantages)
    if len(pairs) < p.MIN_MATCHED_PAIRS:
        direction = "insufficient_matched_disagreements"
    elif mean(advantages) > 0 and positives / len(pairs) >= 2 / 3:
        direction = "supports_harmful_late_accept_reversal"
    elif mean(advantages) < 0 and negatives / len(pairs) >= 2 / 3:
        direction = "contradicts_harmful_late_accept_reversal"
    else:
        direction = "mixed_local_effects"
    report = {
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "status": "complete" if complete else "partial",
        "captures": len(captures),
        "disagreements_found": sum(row["status"] == "captured" for row in captures),
        "matched_pairs": len(pairs),
        "missing": missing,
        "pairs": pairs,
        "summary": {
            "deliver_better_pairs": positives,
            "accept_better_pairs": negatives,
            "ties": len(advantages) - positives - negatives,
            "mean_continuation_return_difference": (
                mean(advantages) if advantages else None
            ),
            "median_continuation_return_difference": (
                median(advantages) if advantages else None
            ),
        },
        "directional_result": direction,
        "interpretation_boundary": (
            "one forced action under the frozen episode-269 continuation on the "
            "reused development panel; no general policy claim"
        ),
        "paper_evidence": False,
        "episode_runs": len(captures) + len(pairs),
        "training_runs": 0,
    }
    report_path = output.resolve() / "e110-report.json"
    table_path = output.resolve() / "e110-results.md"
    write_json(report_path, report)
    table_path.write_text(_render(report), encoding="utf-8")
    return {
        "status": report["status"],
        "captures": report["captures"],
        "matched_pairs": report["matched_pairs"],
        "directional_result": direction,
        "report": str(report_path),
        "table": str(table_path),
        "episode_runs": report["episode_runs"],
        "training_runs": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare", "repair-contract", "freeze-sources", "self-check", "capture",
            "run-deliver", "analyze",
        ),
    )
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "repair-contract":
        result = repair_contract(args.output)
    elif args.command == "freeze-sources":
        result = freeze_sources(args.output)
    elif args.command == "self-check":
        result = self_check(args.output)
    elif args.command == "capture":
        require(args.seed is not None, "capture requires --seed")
        result = capture(args.output, args.seed, args.device)
    elif args.command == "run-deliver":
        require(args.seed is not None, "run-deliver requires --seed")
        result = run_deliver(args.output, args.seed, args.device)
    else:
        result = analyze(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
