"""Build an executor-matched AntMaze preparation for frozen E549 at 5M."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from projects.tfsn_consequence_value_iteration.antmaze import (
    aligned_factorized_consequence_r30 as r30,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r47_current_behavior_alignment as r47,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r48_rule_free_r39 as r48,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r51_destination_arrival_correction as r51,
)
from projects.tfsn_consequence_value_iteration.antmaze.e549_executor import (
    E549_CHECKPOINT,
    E549_CHECKPOINT_SHA256,
    E549SACExecutor,
    inspect_e549_checkpoint,
)
from projects.tfsn_consequence_value_iteration.antmaze.factorized_consequence import (
    FactorizedConsequenceEnsemble,
    _train_head,
)
from projects.tfsn_consequence_value_iteration.antmaze.recorded_transition_planning import (
    consequence_inputs,
)
from projects.tfsn_consequence_value_iteration.shared.reliability import (
    recorded_transition_reliability,
)
from projects.tfsn_consequence_value_iteration.shared.settings import (
    RECORDED_EVIDENCE_PRIOR_WEIGHT,
    RECORDED_RELIABILITY_NEIGHBORS,
)
from scripts.antmaze.current_tfsn.constants import (
    COMMANDS_PER_EPISODE,
    COMMAND_STEPS,
    STATE_DIM,
)
from scripts.antmaze.current_tfsn.executor import make_env


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
PROTOCOL = HERE / "r52_e549_preparation_protocol.json"
REFERENCE = HERE / "R52_E549_PREPARATION_REFERENCE.md"
METHOD_REVISION = "antmaze_e549_5m_preparation_r52"
CLAIM_LEVEL = "reward_free_executor_matched_preparation_only"
DEFAULT_OUTPUT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "e549_5m_preparation_r52"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def atomic_torch(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_protocol() -> dict:
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "oracle_inputs": [],
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "environment_reward_consumed": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
        "fixed_command_duration": COMMAND_STEPS,
        "commands_per_physical_episode": COMMANDS_PER_EPISODE,
        "base_graph_commands": 30_000,
        "alignment_commands": 6_000,
        "training_commands": 34_800,
        "validation_commands": 1_200,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"R52 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE.resolve():
        raise ValueError("R52 method reference changed")
    return value


PROTOCOL_VALUE = load_protocol()


def output_paths(output: Path) -> dict:
    output = Path(output).resolve()
    return {
        "root": output,
        "base": output / "collection/base_graph",
        "alignment": output / "collection/alignment",
        "collection_summary": output / "collection/summary.json",
        "consequence_directory": output / "consequence",
        "consequence": output / "consequence/consequence.pt",
        "consequence_summary": output / "consequence/summary.json",
        "runtime": output / "runtime",
        "graph": output / "runtime/recorded_transition_graph.npz",
    }


def dimensions(profile: str) -> tuple[int, int, int]:
    if profile == "full":
        return (
            int(PROTOCOL_VALUE["base_graph_tasks"]),
            int(PROTOCOL_VALUE["alignment_tasks"]),
            int(PROTOCOL_VALUE["commands_per_preparation_task"]),
        )
    # Exercise a physical reset and a whole-task holdout in smoke mode.
    return 2, 2, COMMANDS_PER_EPISODE + 2


def _part_path(directory: Path, local_task: int) -> Path:
    return Path(directory) / f"part_{int(local_task):05d}.npz"


def _validate_part(
    path: Path, global_task: int, commands: int,
) -> dict:
    return r47.validate_shard(path, int(global_task), int(commands))


def _write_block_manifest(
    directory: Path, name: str, global_offset: int, tasks: int, commands: int,
) -> dict:
    parts = []
    for local_task in range(tasks):
        path = _part_path(directory, local_task)
        _validate_part(path, global_offset + local_task, commands)
        parts.append({"file": path.name, "commands": int(commands)})
    manifest = {
        "schema_version": 1,
        "method_revision": METHOD_REVISION,
        "block": name,
        "global_task_offset": int(global_offset),
        "tasks": int(tasks),
        "commands_per_task": int(commands),
        "commands": int(tasks * commands),
        "parts": parts,
        "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
        "task_reward_used": False,
        "hidden_goals_used": False,
    }
    atomic_json(Path(directory) / "manifest.json", manifest)
    return manifest


def _collect_block(
    planner, executor, env, directory: Path, name: str, global_offset: int,
    tasks: int, commands: int, seed_base: int,
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    completed = 0
    while completed < tasks and _part_path(directory, completed).is_file():
        _validate_part(
            _part_path(directory, completed), global_offset + completed, commands
        )
        completed += 1
    existing = list(directory.glob("part_*.npz"))
    if len(existing) != completed:
        raise RuntimeError(f"R52 {name} parts are not a contiguous prefix")
    for local_task in range(completed, tasks):
        global_task = global_offset + local_task
        shard = r47.collect_task(
            planner,
            executor,
            env,
            global_task,
            seed_base + global_task,
            commands,
        )
        path = _part_path(directory, local_task)
        r47.atomic_savez(path, **shard)
        _validate_part(path, global_task, commands)
        movement = np.linalg.norm(
            shard["states"][:, -1, :2] - shard["states"][:, 0, :2], axis=1
        )
        print(
            f"r52_e549 {name} task={local_task + 1}/{tasks} "
            f"rows={(local_task + 1) * commands} "
            f"movement_median={np.median(movement):.3f}",
            flush=True,
        )
    return _write_block_manifest(
        directory, name, global_offset, tasks, commands
    )


def command_audit(_args) -> dict:
    executor = inspect_e549_checkpoint()
    report = {
        "schema_version": 1,
        "method_reference": str(REFERENCE),
        "method_reference_sha256": sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "executor": executor,
        "base_graph_commands": PROTOCOL_VALUE["base_graph_commands"],
        "alignment_commands": PROTOCOL_VALUE["alignment_commands"],
        "training_commands": PROTOCOL_VALUE["training_commands"],
        "validation_commands": PROTOCOL_VALUE["validation_commands"],
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "environment_reward_consumed": False,
        "confirmation_panel_accessed": False,
        "passed": True,
    }
    print(json.dumps(report, indent=2))
    return report


def command_collect(args) -> dict:
    base_tasks, alignment_tasks, commands = dimensions(args.profile)
    paths = output_paths(args.output)
    if paths["collection_summary"].exists():
        saved = json.loads(paths["collection_summary"].read_text())
        if (
            saved.get("profile") != args.profile
            or saved.get("executor_checkpoint_sha256")
            != E549_CHECKPOINT_SHA256
        ):
            raise ValueError("existing R52 collection has another contract")
        print(json.dumps(saved, indent=2))
        return saved

    # The old artifacts only define the reward-free command proposal behavior.
    # The returned Farama executor is deliberately discarded before collection.
    planner, old_executor, old_env = r51._load_runtime(args)
    old_env.close()
    del old_executor
    executor = E549SACExecutor(E549_CHECKPOINT, args.device)
    env = make_env(max_episode_steps=10_000_000)
    started = time.time()
    try:
        base_manifest = _collect_block(
            planner,
            executor,
            env,
            paths["base"],
            "base_graph",
            0,
            base_tasks,
            commands,
            int(PROTOCOL_VALUE["collection_seed_base"]),
        )
        alignment_manifest = _collect_block(
            planner,
            executor,
            env,
            paths["alignment"],
            "alignment",
            base_tasks,
            alignment_tasks,
            commands,
            int(PROTOCOL_VALUE["collection_seed_base"]),
        )
    finally:
        env.close()
    report = {
        "schema_version": 1,
        "experiment_id": PROTOCOL_VALUE["experiment_id"],
        "method_reference": str(REFERENCE),
        "method_reference_sha256": sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL if args.profile == "full" else "smoke_only",
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "profile": args.profile,
        "executor_checkpoint": str(E549_CHECKPOINT),
        "executor_checkpoint_sha256": executor.audit["sha256"],
        "executor_training_steps": executor.audit["num_timesteps"],
        "command_proposal_revision": r51.METHOD_REVISION,
        "command_proposal_artifacts": dict(planner.artifact_hashes),
        "base_graph_manifest": str(paths["base"] / "manifest.json"),
        "base_graph_manifest_sha256": sha256(paths["base"] / "manifest.json"),
        "alignment_manifest": str(paths["alignment"] / "manifest.json"),
        "alignment_manifest_sha256": sha256(
            paths["alignment"] / "manifest.json"
        ),
        "base_graph_tasks": int(base_tasks),
        "alignment_tasks": int(alignment_tasks),
        "commands_per_preparation_task": int(commands),
        "rows": int((base_tasks + alignment_tasks) * commands),
        "fixed_command_duration": COMMAND_STEPS,
        "commands_per_physical_episode": COMMANDS_PER_EPISODE,
        "task_memory_reset_per_preparation_task": True,
        "task_memory_persisted_across_physical_resets": True,
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "environment_reward_consumed": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "base_block": base_manifest,
        "alignment_block": alignment_manifest,
        "wall_clock_seconds_this_invocation": time.time() - started,
    }
    atomic_json(paths["collection_summary"], report)
    print(json.dumps(report, indent=2))
    return report


def _load_block(
    directory: Path, global_offset: int, tasks: int, commands: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = [
        _validate_part(
            _part_path(directory, local), global_offset + local, commands
        )
        for local in range(tasks)
    ]
    states = np.concatenate([row["states"][:, 0] for row in rows])
    command = np.concatenate([row["commands"] for row in rows])
    endpoint = np.concatenate([row["states"][:, -1] for row in rows])
    task = np.concatenate([row["task_id"] for row in rows])
    return states, command, endpoint, task


def split_alignment_tasks(
    base_tasks: int, alignment_tasks: int, profile: str,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(PROTOCOL_VALUE["split_seed"]))
    order = rng.permutation(alignment_tasks)
    held_count = (
        1 if profile == "smoke"
        else int(PROTOCOL_VALUE["held_out_alignment_tasks"])
    )
    held = np.sort(order[:held_count] + base_tasks).astype(np.int16)
    trained = np.sort(order[held_count:] + base_tasks).astype(np.int16)
    if not len(held) or not len(trained):
        raise ValueError("R52 needs whole training and held-out tasks")
    return trained, held


def _spatial_expected_endpoint(
    model: FactorizedConsequenceEnsemble,
    states: np.ndarray,
    commands: np.ndarray,
    batch: int = 4096,
) -> np.ndarray:
    rows = []
    for begin in range(0, len(states), batch):
        stop = min(begin + batch, len(states))
        inputs = consequence_inputs(states[begin:stop], commands[begin:stop])
        members = []
        for member in model.members:
            weight, residual, _scale = model.spatial_parameters(inputs, member)
            members.append(
                commands[begin:stop]
                + (weight[..., None] * residual).sum(axis=1)
            )
        rows.append(np.stack(members, axis=1).mean(axis=1))
    return np.concatenate(rows).astype(np.float32)


def command_train_consequence(args) -> dict:
    paths = output_paths(args.output)
    if paths["consequence_summary"].exists():
        saved = json.loads(paths["consequence_summary"].read_text())
        if saved.get("checkpoint_sha256") != sha256(paths["consequence"]):
            raise ValueError("saved R52 consequence hash changed")
        print(json.dumps(saved, indent=2))
        return saved
    base_tasks, alignment_tasks, commands_per_task = dimensions(args.profile)
    base = _load_block(paths["base"], 0, base_tasks, commands_per_task)
    alignment = _load_block(
        paths["alignment"], base_tasks, alignment_tasks, commands_per_task
    )
    trained_tasks, held_tasks = split_alignment_tasks(
        base_tasks, alignment_tasks, args.profile
    )
    alignment_train = np.isin(alignment[3], trained_tasks)
    alignment_held = np.isin(alignment[3], held_tasks)
    train_states = np.concatenate((base[0], alignment[0][alignment_train]))
    train_commands = np.concatenate((base[1], alignment[1][alignment_train]))
    train_endpoints = np.concatenate((base[2], alignment[2][alignment_train]))
    validation_states = alignment[0][alignment_held]
    validation_commands = alignment[1][alignment_held]
    validation_endpoints = alignment[2][alignment_held]
    all_states = np.concatenate((train_states, validation_states))
    all_commands = np.concatenate((train_commands, validation_commands))
    all_endpoints = np.concatenate((train_endpoints, validation_endpoints))
    train_index = np.arange(len(train_states), dtype=np.int64)
    validation_index = np.arange(
        len(train_states), len(all_states), dtype=np.int64
    )
    inputs = consequence_inputs(all_states, all_commands)
    spatial_target = all_endpoints[:, :2] - all_commands
    posture_input = np.concatenate((inputs, all_endpoints[:, :2]), axis=1)
    posture_target = all_endpoints[:, 2:]
    schedule = PROTOCOL_VALUE["training_schedule"]
    epochs = 2 if args.profile == "smoke" else int(schedule["epochs"])
    spatial = _train_head(
        inputs,
        spatial_target,
        train_index,
        validation_index,
        int(PROTOCOL_VALUE["split_seed"]) + 801,
        args.device,
        epochs,
        int(schedule["batch_size"]),
        float(schedule["learning_rate"]),
        "R52 E549 spatial",
    )
    posture = _train_head(
        posture_input,
        posture_target,
        train_index,
        validation_index,
        int(PROTOCOL_VALUE["split_seed"]) + 802,
        args.device,
        epochs,
        int(schedule["batch_size"]),
        float(schedule["learning_rate"]),
        "R52 E549 posture",
    )
    payload = {
        "artifact_type": "factorized_ant_consequence",
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL if args.profile == "full" else "smoke_only",
        "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
        "spatial_parameterization": "endpoint_xy_minus_command",
        "posture_parameterization": "absolute_endpoint_posture",
        "factorization": (
            "P(endpoint_xy-command|state,command) times "
            "P(endpoint_posture|state,command,endpoint_xy)"
        ),
        "training_source": [str(paths["base"]), str(paths["alignment"])],
        "training_commands": int(len(train_index)),
        "validation_commands": int(len(validation_index)),
        "split_seed": int(PROTOCOL_VALUE["split_seed"]),
        "split": "whole_E549_alignment_tasks_held_out",
        "held_out_alignment_tasks": held_tasks.tolist(),
        "oracle_command_dataset_used_for_training_or_selection": False,
        "base_input_mean": spatial["input_mean"],
        "base_input_std": spatial["input_std"],
        "spatial_target_mean": spatial["target_mean"],
        "spatial_target_std": spatial["target_std"],
        "posture_input_mean": posture["input_mean"],
        "posture_input_std": posture["input_std"],
        "posture_target_mean": posture["target_mean"],
        "posture_target_std": posture["target_std"],
        "spatial_config": spatial["config"],
        "posture_config": posture["config"],
        "spatial_member_states": spatial["member_states"],
        "posture_member_states": posture["member_states"],
        "epochs": int(epochs),
        "batch_size": int(schedule["batch_size"]),
        "learning_rate": float(schedule["learning_rate"]),
    }
    atomic_torch(paths["consequence"], payload)
    model = FactorizedConsequenceEnsemble(paths["consequence"], args.device)
    sigma, sigma_rows = r30.landing_precision(
        model, validation_states, validation_commands
    )
    predicted = _spatial_expected_endpoint(
        model, validation_states, validation_commands
    )
    error = np.linalg.norm(predicted - validation_endpoints[:, :2], axis=1)
    report = {
        "schema_version": 1,
        "experiment_id": PROTOCOL_VALUE["experiment_id"],
        "method_reference": str(REFERENCE),
        "method_reference_sha256": sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": payload["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "profile": args.profile,
        "checkpoint": str(paths["consequence"]),
        "checkpoint_sha256": sha256(paths["consequence"]),
        "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
        "training_commands": int(len(train_index)),
        "validation_commands": int(len(validation_index)),
        "trained_alignment_tasks": trained_tasks.tolist(),
        "held_out_alignment_tasks": held_tasks.tolist(),
        "landing_sigma_m": float(sigma),
        "landing_sigma_estimator": (
            "held_out_E549_mean_ensemble_and_mixture_weighted_"
            "within_mode_radial_scale"
        ),
        "endpoint_error_mean_m": float(error.mean()),
        "endpoint_error_median_m": float(np.median(error)),
        "endpoint_error_p90_m": float(np.quantile(error, 0.9)),
        "landing_scale_p90_m": float(np.quantile(sigma_rows, 0.9)),
        "spatial_best_validation_normalized_nll": [
            row["best_validation_normalized_nll"] for row in spatial["histories"]
        ],
        "posture_best_validation_normalized_nll": [
            row["best_validation_normalized_nll"] for row in posture["histories"]
        ],
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    atomic_json(paths["consequence_summary"], report)
    atomic_json(
        paths["consequence_directory"] / "training_history.json",
        {"spatial": spatial["histories"], "posture": posture["histories"]},
    )
    print(json.dumps(report, indent=2))
    return report


def _verified_runtime_identity(summary: dict) -> tuple[Path, str, str]:
    """Validate the immutable method identity recorded by a preparation.

    R52 historically configured every consumer with R52's module constants.
    Independent aligned rebuilds retain the same runtime layout but declare a
    different, frozen preparation identity.  Runtime consumers must therefore
    bind the identity carried by the verified preparation rather than silently
    relabeling its graph as historical R52.
    """
    required = ("method_reference", "method_reference_sha256",
                "method_revision", "claim_level")
    missing = [name for name in required if not summary.get(name)]
    if missing:
        raise ValueError(
            "preparation summary lacks runtime identity: "
            + ", ".join(missing)
        )
    reference = Path(summary["method_reference"])
    if not reference.is_absolute():
        reference = (MANAGEMENT_ROOT / reference).resolve()
    else:
        reference = reference.resolve()
    if not reference.is_file():
        raise FileNotFoundError(
            f"preparation method reference is absent: {reference}"
        )
    if summary["method_reference_sha256"] != sha256(reference):
        raise ValueError("preparation method reference changed")

    revision = str(summary["method_revision"])
    claim_level = str(summary["claim_level"])
    if revision == METHOD_REVISION:
        if reference != REFERENCE.resolve():
            raise ValueError("historical R52 preparation has another reference")
        if claim_level != CLAIM_LEVEL:
            raise ValueError("historical R52 preparation has another claim level")
    return reference, revision, claim_level


def configure_r52_runtime(output: Path) -> dict:
    paths = output_paths(output)
    if not paths["consequence_summary"].is_file():
        raise FileNotFoundError("run R52 consequence training first")
    summary = json.loads(paths["consequence_summary"].read_text())
    if summary.get("checkpoint_sha256") != sha256(paths["consequence"]):
        raise ValueError("R52 consequence checkpoint changed")
    reference, revision, claim_level = _verified_runtime_identity(summary)
    r48.REFERENCE = reference
    r48.METHOD_REVISION = revision
    r48.PREPARATION_CLAIM_LEVEL = claim_level
    r48.R30_CHECKPOINT = paths["consequence"]
    r48.R30_SUMMARY = paths["consequence_summary"]
    r48.SIGMA = float(summary["landing_sigma_m"])
    r48.configure_result_directory(paths["runtime"])
    r48._bind_adapter()
    return summary


def command_prepare_graph(args) -> dict:
    summary = configure_r52_runtime(args.output)
    paths = output_paths(args.output)
    args.graph = paths["graph"]
    if paths["graph"].exists() and paths["graph"].with_suffix(".json").exists():
        graph = r48.verify_graph(paths["graph"])
        report = {
            "artifact": str(paths["graph"]),
            "artifact_sha256": sha256(paths["graph"]),
            "nodes": graph.node_count,
            "edges": graph.edge_count,
            "resumed": True,
        }
        print(json.dumps(report, indent=2))
        return report
    base_tasks, _alignment_tasks, commands = dimensions(summary["profile"])
    states, command, endpoints, _task = _load_block(
        paths["base"], 0, base_tasks, commands
    )
    model = FactorizedConsequenceEnsemble(paths["consequence"], args.device)
    state_scale = model.input_std[:STATE_DIM].copy()
    state_scale[:2] = float(r48.SIGMA)
    nodes, inverse = np.unique(
        np.concatenate((states, endpoints), axis=0), axis=0, return_inverse=True
    )
    edge_count = len(states)
    source_index = inverse[:edge_count]
    endpoint_index = inverse[edge_count:]
    reliability, empirical, support = recorded_transition_reliability(
        states, command, endpoints, state_scale, r48.SIGMA
    )
    out_degree = np.bincount(source_index, minlength=len(nodes))
    graph = r48.ManifestRecordedTransitionGraph(
        nodes,
        source_index,
        endpoint_index,
        command,
        reliability,
        state_scale,
        metadata={
            "method_revision": METHOD_REVISION,
            **r48.manifest_contract(),
            "claim_level": CLAIM_LEVEL,
            "node_source": "all_exact_E549_reward_free_command_boundary_states",
            "edge_source": "all_exact_E549_recorded_reward_free_transitions",
            "reward_free_rounds": [str(paths["base"])],
            "reward_free_manifest_sha256": sha256(
                paths["base"] / "manifest.json"
            ),
            "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
            "recorded_reliability_neighbors": RECORDED_RELIABILITY_NEIGHBORS,
            "recorded_evidence_prior_weight": RECORDED_EVIDENCE_PRIOR_WEIGHT,
            "recorded_reliability_estimator": (
                "shared_kernel_posterior_of_repeatable_realized_deltas"
            ),
            "reliability_empirical_mean": float(empirical.mean()),
            "reliability_support_median": float(np.median(support)),
            "deployed_consequence_sha256": sha256(paths["consequence"]),
            "field_sigma_m": float(r48.SIGMA),
            "node_count": int(len(nodes)),
            "edge_count": int(edge_count),
            "fraction_edges_with_recorded_continuation": float(
                np.mean(out_degree[endpoint_index] > 0)
            ),
            "exact_recorded_start_reconstruction_max_error": float(
                np.abs(nodes[source_index] - states).max()
            ),
            "exact_recorded_endpoint_reconstruction_max_error": float(
                np.abs(nodes[endpoint_index] - endpoints).max()
            ),
            "task_reward_used": False,
            "hidden_goals_used": False,
            "oracle_inputs": [],
        },
    )
    graph.save(paths["graph"])
    r48.verify_graph(paths["graph"])
    report = {
        "artifact": str(paths["graph"]),
        "artifact_sha256": sha256(paths["graph"]),
        "nodes": graph.node_count,
        "edges": graph.edge_count,
        "reliability_mean": float(reliability.mean()),
        "landing_sigma_m": float(r48.SIGMA),
        "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
    }
    print(json.dumps(report, indent=2))
    return report


def _prepare_args(args) -> None:
    configure_r52_runtime(args.output)
    args.graph = output_paths(args.output)["graph"]


def command_prepare_mobility(args) -> None:
    _prepare_args(args)
    r48.command_prepare_mobility(args)


def command_prepare_bridges(args) -> None:
    _prepare_args(args)
    r48.command_prepare_bridges(args)


def command_prepare_passage(args) -> None:
    _prepare_args(args)
    r48.command_prepare_passage(args)


def command_prepare_all(args) -> None:
    command_prepare_graph(args)
    command_prepare_mobility(args)
    command_prepare_bridges(args)
    command_prepare_passage(args)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=652)
    value.add_argument("--profile", choices=("smoke", "full"), default="full")
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--base-output", type=Path, default=r51.R48_BASE_OUTPUT)
    value.add_argument("--graph", type=Path)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").set_defaults(func=command_audit)
    commands.add_parser("collect").set_defaults(func=command_collect)
    commands.add_parser("train-consequence").set_defaults(
        func=command_train_consequence
    )
    commands.add_parser("prepare-graph").set_defaults(func=command_prepare_graph)
    commands.add_parser("prepare-mobility").set_defaults(
        func=command_prepare_mobility
    )
    commands.add_parser("prepare-bridges").set_defaults(
        func=command_prepare_bridges
    )
    commands.add_parser("prepare-passage").set_defaults(
        func=command_prepare_passage
    )
    commands.add_parser("prepare-all").set_defaults(func=command_prepare_all)
    return value


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
