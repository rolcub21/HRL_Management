"""Seed, continue, and analyze E108 without changing E107 learning state."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import run_vcg_objective_gamma_audit as audit
import train_vcg_dense_proper as dense
from viability_graph_episodic_audit import EpisodicViabilityGraphHierarchyAgent
from experiments.conditioned_vcg.E100_accelerated_release_clock_163k_v1 import (
    fast_inputs,
)
from experiments.conditioned_vcg.E107_queue_context_stage_a_8x8_171k_v1 import (
    instrumentation,
)
from experiments.conditioned_vcg.E107_queue_context_stage_a_8x8_171k_v1.representation import (
    QUEUE_CONTEXT_NAMES,
    QueueContextAgent,
    acceleration_enabled,
)
from experiments.conditioned_vcg.target_size_dense_stage_a_v2 import run as v2run
from experiments.conditioned_vcg.target_size_dense_stage_a_v4 import acceleration
from . import protocol as p


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing E108 artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid E108 artifact: {path}")
    return value


def prepare(output: Path) -> dict:
    output = output.resolve()
    expected = p.contract()
    path = output / "contract.json"
    if path.is_file():
        if read_json(path) != expected:
            raise RuntimeError("E108 contract, source checkpoint, or code changed")
    else:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError("E108 refuses a nonempty uncontracted output")
        write_json(path, expected)
    return {
        "status": "prepared",
        "contract_sha256": expected["contract_sha256"],
        "source_episode": p.SOURCE_EPISODE,
        "maximum_episode": p.MAX_EPISODE,
        "decision_checkpoints": list(p.DECISION_CHECKPOINTS),
        "episode_runs": 0,
        "training_runs": 0,
    }


def destination(output: Path) -> Path:
    return output.resolve() / "training/operational" / p.SCENARIO / "seed-0"


def migrate_payload(source: dict, contract: dict) -> dict:
    value = dict(source)
    resume_contract = p.continuation_resume_contract(source, contract)
    value.update(
        {
            "protocol": p.PROTOCOL,
            "training_protocol": p.PROTOCOL,
            "method_version": p.METHOD_VERSION,
            "completed_training_episodes": p.SOURCE_EPISODE,
            "total_training_episodes": p.MAX_EPISODE,
            "next_training_episode": p.SOURCE_EPISODE + 1,
            "next_train_instance_seed": (
                int(resume_contract["train_instance_seed_base"])
                + p.SOURCE_EPISODE
            ),
            "resume_contract": resume_contract,
            "resume_contract_sha256": audit._contract_hash(resume_contract),
            # E107 remains available as the immutable source. E108 selects only
            # among newly evaluated continuation checkpoints.
            "best_validation_record": None,
            "best_candidate_checkpoint_filename": None,
            "best_candidate_checkpoint_sha256": None,
            "best_checkpoint_sha256": None,
            "protocol_training_complete": False,
            "selection_finalized_after_total_episodes": False,
            "selection_finalized_at_training_episode": None,
            "selected_checkpoint_episode": None,
            "deployment_checkpoint_eligible": False,
            "e108_contract_sha256": contract["contract_sha256"],
            "e108_source_checkpoint_sha256": contract["source"][
                "checkpoint_sha256"
            ],
            "training_horizon_rule": p.HORIZON_RULE,
            "development_only": True,
            "performance_claim_authorized": False,
            "test_panels_opened": False,
            "paper_evidence": False,
        }
    )
    value.pop("e107_contract_sha256", None)
    return value


def seed(output: Path) -> dict:
    prepared = prepare(output)
    output = output.resolve()
    target = destination(output)
    latest = target / "latest.pth"
    manifest = target / "training-contract.json"
    if latest.is_file() or manifest.is_file():
        if not (latest.is_file() and manifest.is_file()):
            raise RuntimeError("E108 seed is partial; refusing automatic repair")
        observed = torch.load(latest, map_location="cpu", weights_only=False)
        if (
            observed.get("e108_contract_sha256") != prepared["contract_sha256"]
            or observed.get("e108_source_checkpoint_sha256")
            != p.contract()["source"]["checkpoint_sha256"]
        ):
            raise RuntimeError("existing E108 seed has foreign provenance")
        return {
            "status": "already_seeded",
            "completed_training_episodes": observed[
                "completed_training_episodes"
            ],
            "latest_checkpoint": str(latest),
            "episode_runs": 0,
            "training_runs": 0,
        }

    source = torch.load(p.SOURCE_LATEST, map_location="cpu", weights_only=False)
    contract = p.contract()
    if p.sha256(p.SOURCE_LATEST) != contract["source"]["checkpoint_sha256"]:
        raise RuntimeError("E107 source checkpoint changed during E108 seeding")
    migrated = migrate_payload(source, contract)
    target.mkdir(parents=True, exist_ok=False)
    write_json(
        manifest,
        {
            "training_protocol": p.PROTOCOL,
            "method_version": p.METHOD_VERSION,
            "resume_contract_sha256": migrated["resume_contract_sha256"],
            "contract": migrated["resume_contract"],
        },
    )
    audit._atomic_torch_save(migrated, latest)
    state = migrated["agent_state"]
    return {
        "status": "seeded",
        "completed_training_episodes": p.SOURCE_EPISODE,
        "epsilon": state["epsilon"],
        "decision_count": state["decision_count"],
        "gradient_steps": state["gradient_steps"],
        "target_updates": state["target_updates"],
        "replay_size": len(state["replay"]["memory"]),
        "latest_checkpoint": str(latest),
        "source_checkpoint_sha256": contract["source"]["checkpoint_sha256"],
        "episode_runs": 0,
        "training_runs": 0,
    }


def _development_validation_record(*args, **kwargs):
    value = _BASE_VALIDATION_RECORD(*args, **kwargs)
    value["deployment_eligible"] = False
    value["development_selection_only"] = True
    value["e108_low_epsilon_continuation"] = True
    return value


_BASE_VALIDATION_RECORD = audit._validation_record


def _train_with_context(args, trainer_protocol):
    original_resume_contract = dense._resume_contract
    original_result = dense._result_payload
    original_checkpoint = EpisodicViabilityGraphHierarchyAgent.checkpoint
    original_validate = dense._validate_resume

    def resume_contract(*positional, **keywords):
        value = original_resume_contract(*positional, **keywords)
        value.update(
            {
                "e108_contract_sha256": p.contract()["contract_sha256"],
                "e108_source_checkpoint_sha256": p.contract()["source"][
                    "checkpoint_sha256"
                ],
                "agent_architecture": "e107_queue_observed_stage_a",
                "queue_context_names": list(QUEUE_CONTEXT_NAMES),
                "queue_context_stored_in_replay": True,
                "future_schedule_visible_to_policy": False,
                "training_horizon_rule": p.HORIZON_RULE,
            }
        )
        return value

    def result_payload(*positional, **keywords):
        value = original_result(*positional, **keywords)
        value.update(
            {
                "e108_contract_sha256": p.contract()["contract_sha256"],
                "e108_source_checkpoint_sha256": p.contract()["source"][
                    "checkpoint_sha256"
                ],
                "training_horizon_rule": p.HORIZON_RULE,
                "epsilon_schedule": {
                    "start": p.e107.EPSILON_START,
                    "end": p.e107.EPSILON_END,
                    "warmup_decisions": p.e107.EPSILON_WARMUP_DECISIONS,
                    "decay_decisions": p.e107.EPSILON_DECAY_DECISIONS,
                },
                "td_backup_acceleration": acceleration.audit_payload(
                    keywords["agent"]
                ),
                "development_only": True,
                "paper_evidence": False,
            }
        )
        return value

    def checkpoint(agent, *positional, **keywords):
        value = original_checkpoint(agent, *positional, **keywords)
        value.update(
            {
                "e108_contract_sha256": p.contract()["contract_sha256"],
                "e108_source_checkpoint_sha256": p.contract()["source"][
                    "checkpoint_sha256"
                ],
                "training_horizon_rule": p.HORIZON_RULE,
                "td_backup_implementation": p.e107.TD_BACKUP_IMPLEMENTATION,
                "td_backup_acceleration_audit": acceleration.audit_payload(agent),
                "performance_claim_authorized": False,
            }
        )
        return value

    def validate_resume(payload, contract):
        original_validate(payload, contract)
        expected = {
            "e108_contract_sha256": p.contract()["contract_sha256"],
            "e108_source_checkpoint_sha256": p.contract()["source"][
                "checkpoint_sha256"
            ],
            "training_horizon_rule": p.HORIZON_RULE,
            "td_backup_implementation": p.e107.TD_BACKUP_IMPLEMENTATION,
            "queue_context_stored_in_replay": True,
            "future_schedule_visible_to_policy": False,
        }
        mismatches = {
            name: (payload.get(name), wanted)
            for name, wanted in expected.items()
            if payload.get(name) != wanted
        }
        if mismatches:
            raise ValueError(f"E108 resume metadata mismatch: {mismatches!r}")

    with ExitStack() as stack:
        stack.enter_context(patch.object(v2run, "p", trainer_protocol))
        stack.enter_context(
            patch.object(
                EpisodicViabilityGraphHierarchyAgent,
                "_td_batch",
                acceleration.accelerated_cached_td_batch,
            )
        )
        stack.enter_context(
            patch.object(
                EpisodicViabilityGraphHierarchyAgent, "checkpoint", checkpoint
            )
        )
        for module in (dense, audit):
            stack.enter_context(
                patch.object(
                    module,
                    "EpisodicViabilityGraphHierarchyAgent",
                    QueueContextAgent,
                )
            )
        stack.enter_context(patch.object(dense, "_resume_contract", resume_contract))
        stack.enter_context(patch.object(dense, "_result_payload", result_payload))
        stack.enter_context(patch.object(dense, "_validate_resume", validate_resume))
        stack.enter_context(
            patch.object(dense, "FROZEN_EPSILON_START", p.e107.EPSILON_START)
        )
        stack.enter_context(
            patch.object(dense, "FROZEN_EPSILON_END", p.e107.EPSILON_END)
        )
        stack.enter_context(
            patch.object(
                dense,
                "FROZEN_EPSILON_WARMUP_DECISIONS",
                p.e107.EPSILON_WARMUP_DECISIONS,
            )
        )
        stack.enter_context(
            patch.object(
                dense,
                "FROZEN_EPSILON_DECAY_DECISIONS",
                p.e107.EPSILON_DECAY_DECISIONS,
            )
        )
        return v2run.train_stage_a(
            args.output,
            args.scenario,
            args.model_seed,
            device_name=args.device,
            stop_after_episode=args.stop_after_episode,
            allow_cpu_pilot=False,
        )


def self_check(output: Path) -> dict:
    prepared = prepare(output)
    source = torch.load(p.SOURCE_LATEST, map_location="cpu", weights_only=False)
    migrated = migrate_payload(source, p.contract())
    for name in (
        "agent_state",
        "global_rng_state",
        "training_history",
        "validation_history",
    ):
        if migrated[name] is not source[name]:
            raise RuntimeError(f"E108 migration copied or replaced {name}")
    if migrated["resume_contract"]["total_episodes"] != p.MAX_EPISODE:
        raise RuntimeError("E108 resume horizon was not changed to 320")
    state = migrated["agent_state"]
    if abs(float(state["epsilon"]) - p.e107.EPSILON_END) > 1.0e-12:
        raise RuntimeError("E108 source epsilon is not at the frozen floor")
    if int(state["decision_count"]) < p.e107.EPSILON_DECAY_DECISIONS:
        raise RuntimeError("E108 source has not completed epsilon decay")
    with ExitStack() as stack:
        stack.enter_context(patch.object(dense, "TRAINING_PROTOCOL", p.PROTOCOL))
        stack.enter_context(patch.object(dense, "METHOD_VERSION", p.METHOD_VERSION))
        stack.enter_context(patch.object(dense, "FROZEN_TOTAL_EPISODES", p.MAX_EPISODE))
        stack.enter_context(
            patch.object(dense, "FROZEN_OBJECTIVE_SPEC", p.e107.e105.e78.SPEC)
        )
        dense._validate_resume(migrated, migrated["resume_contract"])
    # Loading the migrated payload through the specialized class verifies that
    # the queue-aware networks and the complete replay remain resumable.
    restored = QueueContextAgent.from_checkpoint(
        migrated, device="cpu", resumable=True, seed=0
    )
    if len(restored.replay) != len(state["replay"]["memory"]):
        raise RuntimeError("E108 replay round-trip changed its size")
    return {
        "status": "passed",
        "contract_sha256": prepared["contract_sha256"],
        "source_episode": p.SOURCE_EPISODE,
        "maximum_episode": p.MAX_EPISODE,
        "epsilon": state["epsilon"],
        "decision_count": state["decision_count"],
        "gradient_steps": state["gradient_steps"],
        "target_updates": state["target_updates"],
        "replay_size": len(state["replay"]["memory"]),
        "learning_state_identity_preserved": True,
        "queue_context_replay_roundtrip": True,
        "base_trainer_resume_validation": "passed",
        "episode_runs": 0,
        "training_runs": 0,
    }


def train_seed(output: Path, through_episode: int, device: str) -> dict:
    seed(output)
    if not p.SOURCE_EPISODE <= through_episode <= p.MAX_EPISODE:
        raise ValueError(
            f"through-episode must be in [{p.SOURCE_EPISODE},{p.MAX_EPISODE}]"
        )
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("E108 continuation requires CUDA")
    trainer_protocol = p.trainer_protocol()
    args = SimpleNamespace(
        output=output.resolve(),
        scenario=p.SCENARIO,
        model_seed=0,
        device=device,
        stop_after_episode=through_episode,
    )
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(dense, "FROZEN_OBJECTIVE_SPEC", p.e107.e105.e78.SPEC)
        )
        stack.enter_context(
            patch.object(audit, "run_objective_audit_episode", instrumentation.run_episode)
        )
        stack.enter_context(
            patch.object(audit, "summarize_objective_runs", instrumentation.summarize_runs)
        )
        stack.enter_context(
            patch.object(audit, "_validation_record", _development_validation_record)
        )
        stack.enter_context(fast_inputs.enabled())
        stack.enter_context(acceleration_enabled())
        result = _train_with_context(args, trainer_protocol)
    return {
        **result,
        "e108_protocol": p.PROTOCOL,
        "source_episode": p.SOURCE_EPISODE,
        "through_episode": through_episode,
        "maximum_episode": p.MAX_EPISODE,
        "epsilon_floor": p.e107.EPSILON_END,
        "paper_evidence": False,
    }


def _row(record: dict, training_history: list[dict]) -> dict:
    summary = record["summary"]
    episode = int(record["checkpoint_episode"])
    training = training_history[episode - 1] if len(training_history) >= episode else {}
    return {
        "checkpoint_episode": episode,
        "phase": "source" if episode <= p.SOURCE_EPISODE else "continuation",
        "training_decision_count": training.get("training_decision_count_after"),
        "training_epsilon": training.get("scheduled_epsilon_after"),
        "strict_completion_rate": summary["strict_method_success_rate"],
        "dense_return": summary["mean_dense_rescored_return"],
        "mean_wait_before_storage": summary["mean_wait_before_storage"],
        "mean_jobs_exceeding_wait_cap": summary["mean_jobs_exceeding_wait_cap"],
        "wait_cap_compliance_rate": summary["mean_wait_cap_compliance_rate"],
        "release_clock_mae": summary["mean_release_clock_mae"],
        "storage_clock_mae": summary["mean_storage_clock_mae"],
        "storage_clock_signed_error": summary["mean_storage_clock_signed_error"],
        "within_20_rate": summary["within_target_window_rate"],
        "steps": summary["mean_steps"],
        "rehandles_per_100": summary["relocations_per_100_deliveries"],
        "guard_activations": summary["total_liveness_guard_activations"],
        "mean_outside_wait_penalty": summary["mean_outside_wait_penalty"],
        "wait_accounting_exact": summary[
            "wait_accounting_exact_for_all_complete_runs"
        ],
    }


def _gate(row: dict, thresholds: dict) -> dict:
    checks = {
        "strict_completion": row["strict_completion_rate"]
        >= thresholds["strict_completion_rate"],
        "wait_reduced": row["mean_wait_before_storage"]
        <= thresholds["mean_wait_before_storage_max"],
        "storage_mae_retained": row["storage_clock_mae"]
        <= thresholds["storage_clock_mae_max"],
        "within_window_retained": row["within_20_rate"]
        >= thresholds["within_20_rate_min"],
        "steps_bounded": row["steps"] <= thresholds["mean_steps_max"],
        "wait_accounting_exact": bool(row["wait_accounting_exact"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _stability_window(rows: list[dict], episode: int, thresholds: dict) -> dict:
    available = [
        row
        for row in rows
        if p.SOURCE_EPISODE < row["checkpoint_episode"] <= episode
    ][-p.STABILITY_WINDOW :]
    gates = [_gate(row, thresholds) for row in available]
    return {
        "episodes": [row["checkpoint_episode"] for row in available],
        "complete_window": len(available) == p.STABILITY_WINDOW,
        "all_joint_gates_pass": (
            len(available) == p.STABILITY_WINDOW
            and all(gate["passed"] for gate in gates)
        ),
        "joint_gate_pass_count": sum(gate["passed"] for gate in gates),
        "dense_return_range": (
            [min(row["dense_return"] for row in available),
             max(row["dense_return"] for row in available)]
            if available
            else None
        ),
        "storage_mae_range": (
            [min(row["storage_clock_mae"] for row in available),
             max(row["storage_clock_mae"] for row in available)]
            if available
            else None
        ),
        "within_20_range": (
            [min(row["within_20_rate"] for row in available),
             max(row["within_20_rate"] for row in available)]
            if available
            else None
        ),
    }


def analyze(output: Path) -> dict:
    prepare(output)
    summary = destination(output) / "training-summary.json"
    if summary.is_file():
        saved = read_json(summary)
        training_history = list(saved.get("training", {}).get("runs", ()))
    else:
        latest = destination(output) / "latest.pth"
        if not latest.is_file():
            raise RuntimeError("run E108 seed or train before analyze")
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        training_history = list(saved.get("training_history", ()))
    rows = [
        _row(record, training_history)
        for record in saved.get("validation_history", ())
    ]
    thresholds = p.contract()["joint_mechanism_thresholds"]
    gates = {
        str(episode): _gate(row, thresholds)
        for episode in p.DECISION_CHECKPOINTS
        for row in rows
        if row["checkpoint_episode"] == episode
    }
    stability = {
        str(episode): _stability_window(rows, episode, thresholds)
        for episode in p.DECISION_CHECKPOINTS
        if any(row["checkpoint_episode"] == episode for row in rows)
    }
    completed = int(saved["completed_training_episodes"])
    source_row = next(
        (row for row in rows if row["checkpoint_episode"] == p.SOURCE_EPISODE),
        None,
    )
    continuation_rows = [
        row for row in rows if row["checkpoint_episode"] > p.SOURCE_EPISODE
    ]
    report = {
        "protocol": p.PROTOCOL,
        "status": "complete" if completed == p.MAX_EPISODE else "partial",
        "completed_training_episodes": completed,
        "source_episode": p.SOURCE_EPISODE,
        "maximum_episode": p.MAX_EPISODE,
        "epsilon_floor": p.e107.EPSILON_END,
        "decision_checkpoints": list(p.DECISION_CHECKPOINTS),
        "source_row": source_row,
        "rows": rows,
        "decision_checkpoint_gates": gates,
        "stability_windows": stability,
        "first_nonnegative_continuation_checkpoint": next(
            (
                row["checkpoint_episode"]
                for row in continuation_rows
                if row["dense_return"] >= 0
            ),
            None,
        ),
        "interpretation": (
            "Exact low-epsilon development continuation from E107 episode 224. "
            "A single best row is not stable convergence; the prespecified "
            "three-validation windows are descriptive development gates."
        ),
        "stage_b_authorized": False,
        "independent_test_opened": False,
        "paper_evidence": False,
        "episode_runs": 0,
        "training_runs": 0,
    }
    report_path = output.resolve() / "e108-report.json"
    write_json(report_path, report)
    return {
        "status": report["status"],
        "completed_training_episodes": completed,
        "source_episode": p.SOURCE_EPISODE,
        "maximum_episode": p.MAX_EPISODE,
        "validation_checkpoints": [row["checkpoint_episode"] for row in rows],
        "decision_checkpoint_gates": gates,
        "stability_windows": stability,
        "report": str(report_path),
        "episode_runs": 0,
        "training_runs": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "self-check", "seed", "train-seed", "analyze")
    )
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--through-episode", type=int, default=240)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "self-check":
        result = self_check(args.output)
    elif args.command == "seed":
        result = seed(args.output)
    elif args.command == "train-seed":
        result = train_seed(args.output, args.through_episode, args.device)
    else:
        result = analyze(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

