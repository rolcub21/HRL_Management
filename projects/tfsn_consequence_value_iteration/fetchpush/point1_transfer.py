"""Reward-free FetchPush transfer using the executor's native goal space.

The smoke profile exercises the complete artifact and hidden-goal interfaces.
It is intentionally too small to be behavioral evidence.  The full profile
uses the shared frozen preparation allocation and is resumable by task shard.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from scripts.fetch.hidden_goal.runtime import (
    get_task,
    hide_native_target,
    make_env,
    make_executor,
    robotics_source_audit,
    sanitize_observation,
    set_hidden_goal,
    task_coordinate,
)

from projects.tfsn_consequence_value_iteration.pointmaze.dataset import (
    CommandDataset,
    grouped_split,
)
from projects.tfsn_consequence_value_iteration.pointmaze.fields import (
    SuppliedGoalField,
    TaskFields,
)
from projects.tfsn_consequence_value_iteration.pointmaze.frozen_r74_planner import (
    FrozenR74PointPlanner,
)
from projects.tfsn_consequence_value_iteration.pointmaze.planner import (
    PointMazeRoadmapPlanner,
)
from projects.tfsn_consequence_value_iteration.pointmaze.prepare_bridges import (
    MobilityModel,
    prepare_bridge_weights,
    prepare_mobility_labels,
    train_mobility,
)
from projects.tfsn_consequence_value_iteration.pointmaze.roadmap import (
    RecordedRoadmap,
    graph_from_dataset,
    reach_targets,
)
from projects.tfsn_consequence_value_iteration.shared.density import (
    situation_density_weights,
)
from projects.tfsn_consequence_value_iteration.shared.factorized import (
    FactorizedConsequenceEnsemble,
    consequence_features,
    landing_precision,
    train_head,
)
from projects.tfsn_consequence_value_iteration.shared.settings import (
    BATCH_SIZE,
    BRIDGE_CANDIDATES,
    COMMAND_STEPS,
    EPOCHS,
    LEARNING_RATE,
    SITUATION_DENSITY_TRUNCATION_SIGMAS,
    VALIDATION_FRACTION,
    finite_horizon,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROTOCOL_PATH = HERE / "protocol.json"
METHOD_REFERENCE = HERE / "METHOD_REFERENCE.md"
RESULT_ROOT = ROOT / "projects/tfsn_consequence_value_iteration/results/fetchpush"
METHOD_REVISION = "fetchpush_executor_goal_space_transfer_r17"
TASK = get_task("push")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def protocol():
    value = json.loads(PROTOCOL_PATH.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "goal_dimension": 2,
        "distance_metric": "native_euclidean_executor_goal_coordinates",
        "reach_normalized_distance_enabled": False,
        "positive_evidence_projection": (
            "nearest_recorded_node_in_executor_goal_space"
        ),
        "fixed_command_duration": COMMAND_STEPS,
        "episode_primitive_steps": 200,
        "derived_command_horizon": finite_horizon(200),
        "preparation_commands_full": 36_000,
        "hidden_goals_accessed_during_preparation": False,
        "task_reward_used_during_preparation": False,
        "environment_reward_consumed_during_preparation": False,
        "success_radius_read_by_planner": False,
        "confirmation_panel_accessed": False,
    }
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            raise ValueError(f"FetchPush point-1 protocol changed {name}")
    if (ROOT / value["method_reference"]).resolve() != METHOD_REFERENCE:
        raise ValueError("FetchPush method reference changed")
    return value


PROTOCOL = protocol()


def dimensions(profile):
    if profile == "smoke":
        return 12, 4, 8
    return (
        int(PROTOCOL["bootstrap_tasks_full"]),
        int(PROTOCOL["alignment_tasks_per_phase_full"]),
        int(PROTOCOL["commands_per_task_full"]),
    )


def paths(profile):
    # Keep superseded smoke artifacts intact for auditability.  The v2 smoke
    # has enough distinct pushes to estimate a nondegenerate landing scale.
    profile_directory = "smoke_v2" if profile == "smoke" else profile
    root = RESULT_ROOT / f"point1_transfer_r17/{profile_directory}"
    return {
        "root": root,
        "bootstrap": root / "collection/bootstrap",
        "acquisition": root / "collection/alignment/acquisition",
        "return": root / "collection/alignment/return",
        "draft_model": root / "draft/models",
        "draft_runtime": root / "draft/runtime",
        "calibration_model": root / "calibration_model",
        "models": root / "models",
        "runtime": root / "runtime",
        "evaluation": root / "evaluation",
        "goal_bank": (
            root / "development_goals.json"
            if profile == "smoke"
            else HERE / "panels/development_goals.json"
        ),
    }


def state_from_observation(observation):
    view = sanitize_observation(observation)
    goal = task_coordinate(TASK, view.achieved_goal).astype(np.float32)
    state = np.concatenate((goal, view.vector.astype(np.float32)))
    return state.astype(np.float32), goal, view


def _reset(env, seed):
    observation, _info = env.reset(seed=int(seed))
    hide_native_target(env)
    return observation


def _execute(env, executor, command, hidden_goal=None):
    observation = env.unwrapped._get_obs()
    _state, _goal, view = state_from_observation(observation)
    executor.set_target(view, np.asarray(command, np.float32))
    states, goal_rows, reward_rows = [], [], []
    for _ in range(COMMAND_STEPS):
        observation, _reward, _terminated, _truncated, _info = env.step(
            executor.act(sanitize_observation(observation))
        )
        state, achieved, _view = state_from_observation(observation)
        states.append(state)
        goal_rows.append(achieved)
        reward_rows.append(
            0.0 if hidden_goal is None else float(
                np.linalg.norm(
                    sanitize_observation(observation).achieved_goal
                    - np.asarray(hidden_goal, np.float64)
                ) <= float(PROTOCOL["evaluator_success_radius_m"])
            )
        )
    return (
        np.asarray(states, np.float32),
        np.asarray(goal_rows, np.float32),
        np.asarray(reward_rows, np.float32),
    )


def _task_seed(phase, task, episode):
    phase_value = {"bootstrap": 0, "acquisition": 1, "return": 2}[phase]
    return int(17_000_000 + phase_value * 1_000_003 + task * 10_007 + episode)


def _shard(directory, task):
    return Path(directory) / f"task_{int(task):04d}.npz"


def _bootstrap_reset_coordinates(tasks, commands):
    env = make_env(TASK, int(PROTOCOL["episode_primitive_steps"]))
    horizon = int(PROTOCOL["derived_command_horizon"])
    episodes = int(np.ceil(commands / horizon))
    coordinates = []
    try:
        for task in range(tasks):
            for episode in range(episodes):
                observation = _reset(
                    env, _task_seed("bootstrap", task, episode)
                )
                coordinates.append(state_from_observation(observation)[1])
    finally:
        env.close()
    return np.asarray(coordinates, np.float32)


def collect_bootstrap(profile):
    bootstrap_tasks, _alignment_tasks, commands = dimensions(profile)
    output = paths(profile)["bootstrap"]
    output.mkdir(parents=True, exist_ok=True)
    pool = _bootstrap_reset_coordinates(bootstrap_tasks, commands).tolist()
    env = make_env(TASK, int(PROTOCOL["episode_primitive_steps"]))
    executor = make_executor(TASK)
    horizon = int(PROTOCOL["derived_command_horizon"])
    started = time.time()
    try:
        for task in range(bootstrap_tasks):
            path = _shard(output, task)
            if path.is_file():
                with np.load(path, allow_pickle=False) as stored:
                    pool.extend(stored["endpoint"][:, :2].tolist())
                continue
            rng = np.random.default_rng(_task_seed("bootstrap", task, 0))
            starts, targets, endpoints, primitives, episodes = [], [], [], [], []
            for row in range(commands):
                episode, index = divmod(row, horizon)
                if index == 0:
                    observation = _reset(
                        env, _task_seed("bootstrap", task, episode)
                    )
                state, _xy, _view = state_from_observation(observation)
                command = np.asarray(pool[int(rng.integers(len(pool)))], np.float32)
                primitive, _path_xy, _reward = _execute(env, executor, command)
                observation = env.unwrapped._get_obs()
                endpoint = primitive[-1]
                pool.append(endpoint[:2].tolist())
                starts.append(state); targets.append(command)
                endpoints.append(endpoint); primitives.append(primitive)
                episodes.append(episode)
            atomic_npz(
                path,
                start=np.asarray(starts, np.float32),
                command=np.asarray(targets, np.float32),
                endpoint=np.asarray(endpoints, np.float32),
                primitive=np.asarray(primitives, np.float32),
                episode=np.asarray(episodes, np.int32),
            )
            print(
                f"fetch bootstrap {task + 1}/{bootstrap_tasks} rows={commands}",
                flush=True,
            )
    finally:
        env.close()
    report = common_report(profile, "bootstrap")
    report.update({
        "tasks": bootstrap_tasks,
        "commands_per_task": commands,
        "rows": bootstrap_tasks * commands,
        "wall_clock_seconds_this_invocation": time.time() - started,
    })
    atomic_json(output / "summary.json", report)
    return report


def common_report(profile, phase):
    deviations = (
        [
            "smoke_only_160_command_preparation_instead_of_36000",
            "smoke_only_20_training_epochs_instead_of_200",
            "smoke_only_one_goal_two_episode_execution_check",
        ]
        if profile == "smoke" else []
    )
    return {
        "schema_version": 1,
        "method_reference": str(METHOD_REFERENCE),
        "method_reference_sha256": sha256(METHOD_REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": (
            "implementation_smoke_only" if profile == "smoke"
            else PROTOCOL["claim_level"]
        ),
        "declared_deviations": deviations,
        "oracle_inputs": [],
        "profile": profile,
        "phase": phase,
        "task_space": PROTOCOL["task_space"],
        "goal_dimension": PROTOCOL["goal_dimension"],
        "distance_metric": PROTOCOL["distance_metric"],
        "reach_normalized_distance_enabled": False,
        "positive_evidence_projection": PROTOCOL[
            "positive_evidence_projection"
        ],
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "environment_reward_consumed": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }


def load_dataset(profile, phases):
    roots = paths(profile)
    starts, commands, endpoints, primitives = [], [], [], []
    trajectories, sources = [], []
    source_names = []
    for source, phase in enumerate(phases):
        directory = roots[phase]
        files = sorted(directory.glob("task_*.npz"))
        if not files:
            raise RuntimeError(f"missing FetchPush {phase} shards")
        for task, path in enumerate(files):
            with np.load(path, allow_pickle=False) as stored:
                row_start = stored["start"].astype(np.float32)
                row_command = stored["command"].astype(np.float32)
                row_endpoint = stored["endpoint"].astype(np.float32)
                row_primitive = stored["primitive"].astype(np.float32)
                episode = stored["episode"].astype(np.int64)
            starts.append(row_start); commands.append(row_command)
            endpoints.append(row_endpoint); primitives.append(row_primitive)
            trajectories.append(
                source * 10_000_000 + task * 10_000 + episode
            )
            sources.append(np.full(len(row_start), source, np.int16))
        source_names.append(f"fetchpush_{phase}")
    return CommandDataset(
        starts=np.concatenate(starts),
        commands=np.concatenate(commands),
        endpoints=np.concatenate(endpoints),
        primitives=np.concatenate(primitives),
        trajectory_id=np.concatenate(trajectories),
        source_id=np.concatenate(sources),
        source_names=tuple(source_names),
        source_paths=tuple(roots[p] for p in phases),
    )


def _endpoint_metrics(model, dataset, index):
    features = consequence_features(
        dataset.starts[index], dataset.commands[index]
    )
    predicted, within = [], []
    for member in range(len(model.spatial_members)):
        weight, residual, std = model.spatial_parameters(features, member)
        predicted.append(
            dataset.commands[index]
            + (weight[..., None] * residual).sum(axis=1)
        )
        within.append((weight * np.sqrt(np.square(std).mean(axis=2))).sum(1))
    prediction = np.stack(predicted).mean(0)
    error = np.linalg.norm(
        prediction - dataset.endpoints[index, :2], axis=1
    )
    return {
        "endpoint_error_mean": float(error.mean()),
        "endpoint_error_median": float(np.median(error)),
        "endpoint_error_p90": float(np.quantile(error, 0.9)),
        "landing_sigma": float(np.stack(within).mean()),
    }


def train_model(profile, phases, output, device, label, density_sigma=None):
    dataset = load_dataset(profile, phases)
    train_index, validation_index = grouped_split(
        dataset, seed=17_031, validation_fraction=VALIDATION_FRACTION
    )
    features = consequence_features(dataset.starts, dataset.commands)
    sample_weight = None
    density_report = None
    if density_sigma is not None:
        sample_weight, density_report = situation_density_weights(
            dataset.starts,
            dataset.commands,
            train_index,
            float(density_sigma),
            SITUATION_DENSITY_TRUNCATION_SIGMAS,
        )
    epochs = 20 if profile == "smoke" else EPOCHS
    spatial = train_head(
        features,
        dataset.endpoints[:, :2] - dataset.commands,
        train_index,
        validation_index,
        17_041,
        device,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        label=f"fetchpush_{label}_spatial",
        sample_weight=sample_weight,
    )
    posture_features = np.concatenate(
        (features, dataset.endpoints[:, :2]), axis=1
    ).astype(np.float32)
    posture = train_head(
        posture_features,
        dataset.endpoints[:, 2:],
        train_index,
        validation_index,
        17_042,
        device,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        label=f"fetchpush_{label}_physical",
        sample_weight=sample_weight,
    )
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "factorized_consequence.pt"
    payload = {
        "artifact_type": "factorized_consequence",
        "schema_version": 1,
        "domain": "FetchPush-v4",
        "method_revision": METHOD_REVISION,
        "claim_level": "implementation_smoke_only" if profile == "smoke" else PROTOCOL["claim_level"],
        "state_dim": int(dataset.starts.shape[1]),
        "position_dim": 2,
        "posture_semantics": "goal_hidden_fetch_physical_state_except_block_xy",
        "spatial_parameterization": "endpoint_block_xy_minus_command_block_xy",
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
        "spatial_histories": spatial["histories"],
        "posture_histories": posture["histories"],
        "task_reward_used": False,
        "hidden_goals_used": False,
        "oracle_inputs": [],
    }
    atomic_torch(checkpoint, payload)
    model = FactorizedConsequenceEnsemble(checkpoint, device)
    metrics = _endpoint_metrics(model, dataset, validation_index)
    report = {
        **common_report(profile, f"train_{label}"),
        "artifact": str(checkpoint.resolve()),
        "artifact_sha256": sha256(checkpoint),
        "rows": dataset.rows,
        "training_rows": int(len(train_index)),
        "validation_rows": int(len(validation_index)),
        "kernel_sigma": metrics["landing_sigma"],
        "kernel_sigma_estimator": "held_out_ensemble_mixture_weighted_within_mode_radial_scale",
        "validation": metrics,
        "density_correction": density_report,
    }
    atomic_json(output / "factorized_consequence_metrics.json", report)
    return report


def build_runtime(profile, phases, models, output, device):
    dataset = load_dataset(profile, phases)
    metrics = json.loads((models / "factorized_consequence_metrics.json").read_text())
    sigma = float(metrics["kernel_sigma"])
    graph = graph_from_dataset(dataset, sigma)
    output.mkdir(parents=True, exist_ok=True)
    graph_path = output / "recorded_graph.npz"
    graph.save(graph_path)
    displacement = (
        graph.nodes[graph.endpoint_index, : graph.goal_dim]
        - graph.nodes[graph.source_index, : graph.goal_dim]
    )
    requested = np.linalg.norm(
        graph.commands
        - graph.nodes[graph.source_index, : graph.goal_dim], axis=1
    )
    achieved = np.linalg.norm(displacement, axis=1)
    reach_positive = int(np.count_nonzero(
        (requested > 0.0) & (achieved > 0.0)
    ))
    reach_count = min(BRIDGE_CANDIDATES, reach_positive)
    if reach_count < 1:
        raise RuntimeError("preparation produced no advancing reach samples")
    if profile == "full" and reach_count != BRIDGE_CANDIDATES:
        raise RuntimeError("full FetchPush preparation has too few reach samples")
    landmarks, targets, distance, offsets = reach_targets(
        graph, reach_count, sigma=sigma, continuous_weighting=True
    )
    atomic_npz(
        output / "reach_candidates.npz",
        landmark_index=landmarks,
        target_index=targets,
        snap_distance=distance,
        reach_offsets=offsets,
    )
    model = FactorizedConsequenceEnsemble(metrics["artifact"], device)
    labels = prepare_mobility_labels(
        graph, targets, model, sigma, output,
        sha256(graph_path), metrics["artifact_sha256"],
    )
    mobility, mobility_report = train_mobility(
        graph,
        labels,
        output,
        device,
        profile,
        method_revision=METHOD_REVISION,
        seed=17_051,
        artifact_type="executor_goal_endpoint_mobility",
    )
    member_weight, weight = prepare_bridge_weights(
        graph, targets, model, mobility, sigma, output
    )
    report = {
        **common_report(profile, "runtime"),
        "graph": str(graph_path.resolve()),
        "graph_sha256": sha256(graph_path),
        "reach_candidates": str((output / "reach_candidates.npz").resolve()),
        "factorized_consequence": metrics["artifact"],
        "factorized_consequence_sha256": metrics["artifact_sha256"],
        "mobility": mobility_report["artifact"],
        "mobility_sha256": mobility_report["artifact_sha256"],
        "bridges": str((output / "bridges.npz").resolve()),
        "bridges_sha256": sha256(output / "bridges.npz"),
        "nodes": graph.node_count,
        "recorded_edges": graph.edge_count,
        "bridges_count": int(weight.size),
        "bridge_candidates_per_node": int(reach_count),
        "smoke_only_bridge_capacity_reduction": (
            int(reach_count)
            if profile == "smoke" and reach_count < BRIDGE_CANDIDATES
            else None
        ),
        "bridge_weight_mean": float(weight.mean()),
        "landing_sigma": sigma,
        "state_dimension": int(graph.nodes.shape[1]),
        "goal_dimension": int(graph.goal_dim),
    }
    atomic_json(output / "runtime.json", report)
    return report


def load_runtime(profile, stage, device):
    root = paths(profile)
    models = root["draft_model"] if stage == "draft" else root["models"]
    runtime = root["draft_runtime"] if stage == "draft" else root["runtime"]
    metrics = json.loads((models / "factorized_consequence_metrics.json").read_text())
    meta = json.loads((runtime / "runtime.json").read_text())
    graph = RecordedRoadmap.load(meta["graph"])
    with np.load(runtime / "reach_candidates.npz", allow_pickle=False) as stored:
        landmarks = stored["landmark_index"]
        targets = stored["target_index"]
        offsets = stored["reach_offsets"]
    with np.load(runtime / "bridges.npz", allow_pickle=False) as stored:
        member_weight = stored["member_weight"]
    consequence = FactorizedConsequenceEnsemble(metrics["artifact"], device)
    mobility = MobilityModel(runtime / "mobility.pt", device)
    base = PointMazeRoadmapPlanner(
        graph, landmarks, targets, offsets, member_weight,
        consequence, mobility, None, float(metrics["kernel_sigma"]),
        finite_horizon(int(PROTOCOL["episode_primitive_steps"])), 17_061,
        advance_weighting=True, consumptive_lookahead=False,
        positive_evidence_goal_space_projection=True,
    )
    return FrozenR74PointPlanner(
        base,
        device,
        method_revision=METHOD_REVISION,
        protocol_value={
            "hub_target_batch_computational": PROTOCOL["hub_target_batch_computational"],
            "recorded_evidence_prior_weight": PROTOCOL["recorded_evidence_prior_weight"],
        },
    )


def collect_alignment(profile, device):
    _bootstrap_tasks, tasks, commands = dimensions(profile)
    roots = paths(profile)
    planner = load_runtime(profile, "draft", device)
    env = make_env(TASK, int(PROTOCOL["episode_primitive_steps"]))
    executor = make_executor(TASK)
    horizon = int(PROTOCOL["derived_command_horizon"])
    support = np.unique(planner.graph.nodes[:, :2], axis=0)
    try:
        for phase in ("acquisition", "return"):
            directory = roots[phase]
            directory.mkdir(parents=True, exist_ok=True)
            for task in range(tasks):
                path = _shard(directory, task)
                if path.is_file():
                    continue
                planner.reset_task_memory()
                rng = np.random.default_rng(_task_seed(phase, task, 0))
                fields = (
                    TaskFields(planner.sigma, 2)
                    if phase == "acquisition"
                    else SuppliedGoalField(
                        planner.sigma,
                        support[int(rng.integers(len(support)))],
                    )
                )
                starts, targets, endpoints, primitives, episodes = [], [], [], [], []
                for row in range(commands):
                    episode, index = divmod(row, horizon)
                    if index == 0:
                        observation = _reset(
                            env, _task_seed(phase, task, episode)
                        )
                        planner.begin_episode()
                    state, _xy, _view = state_from_observation(observation)
                    command, _diagnostic = planner.choose(state, index, fields)
                    primitive, path_xy, rewards = _execute(env, executor, command)
                    observation = env.unwrapped._get_obs()
                    endpoint = primitive[-1]
                    planner.observe_transition(state, command, endpoint)
                    if phase == "acquisition":
                        zero = np.zeros_like(rewards)
                        fields.extend(path_xy, zero)
                        planner.observe_passage(path_xy, zero)
                    starts.append(state); targets.append(command)
                    endpoints.append(endpoint); primitives.append(primitive)
                    episodes.append(episode)
                    if index == horizon - 1:
                        planner.end_episode(False, endpoint[:2])
                atomic_npz(
                    path,
                    start=np.asarray(starts, np.float32),
                    command=np.asarray(targets, np.float32),
                    endpoint=np.asarray(endpoints, np.float32),
                    primitive=np.asarray(primitives, np.float32),
                    episode=np.asarray(episodes, np.int32),
                )
                print(
                    f"fetch alignment {phase} {task + 1}/{tasks}", flush=True
                )
    finally:
        env.close()


def prepare(profile, device):
    collect_bootstrap(profile)
    root = paths(profile)
    if not (root["draft_runtime"] / "runtime.json").is_file():
        train_model(
            profile, ("bootstrap",), root["draft_model"], device, "draft"
        )
        build_runtime(
            profile, ("bootstrap",), root["draft_model"],
            root["draft_runtime"], device,
        )
    collect_alignment(profile, device)
    phases = ("bootstrap", "acquisition", "return")
    calibration = train_model(
        profile, phases, root["calibration_model"], device, "calibration"
    )
    final = train_model(
        profile, phases, root["models"], device, "final",
        density_sigma=float(calibration["kernel_sigma"]),
    )
    runtime = build_runtime(
        profile, phases, root["models"], root["runtime"], device
    )
    report = {
        **common_report(profile, "prepare_complete"),
        "rows": int(load_dataset(profile, phases).rows),
        "models": final,
        "runtime": runtime,
        "ready_for_evaluation": True,
    }
    atomic_json(root["root"] / "preparation_summary.json", report)
    print(json.dumps(report, indent=2))
    return report


def freeze_goals(profile, count):
    path = paths(profile)["goal_bank"]
    if path.is_file():
        value = json.loads(path.read_text())
        if len(value["goals_xyz"]) != count:
            raise ValueError("frozen FetchPush development bank count changed")
        return value
    env = make_env(TASK, int(PROTOCOL["episode_primitive_steps"]))
    goals = []
    try:
        for index in range(count):
            observation, _ = env.reset(
                seed=int(PROTOCOL["development_goal_seed"]) + index
            )
            goals.append(np.asarray(observation["desired_goal"], float).tolist())
    finally:
        env.close()
    value = {
        "schema_version": 1,
        "status": "frozen_before_point1_transfer_behavioral_evaluation",
        "seed": int(PROTOCOL["development_goal_seed"]),
        "goals_xyz": goals,
        "confirmation_panel": False,
    }
    atomic_json(path, value)
    return value


def evaluate(profile, device, smoke=False):
    planner = load_runtime(profile, "final", device)
    count = 1 if smoke else int(PROTOCOL["development_goal_count"])
    episodes = 2 if smoke else int(PROTOCOL["development_episodes_per_goal"])
    bank = freeze_goals(profile, count)
    env = make_env(TASK, int(PROTOCOL["episode_primitive_steps"]))
    executor = make_executor(TASK)
    rows = []
    try:
        for goal_index, raw_goal in enumerate(bank["goals_xyz"]):
            goal = np.asarray(raw_goal, np.float64)
            planner.reset_task_memory()
            fields = TaskFields(planner.sigma, 2)
            successes, first = [], None
            for episode in range(episodes):
                observation = _reset(
                    env,
                    int(PROTOCOL["development_reset_seed"])
                    + goal_index * 10_000 + episode,
                )
                set_hidden_goal(env, goal)
                hide_native_target(env)
                planner.begin_episode()
                episode_success = False
                for command_index in range(
                    int(PROTOCOL["derived_command_horizon"])
                ):
                    state, _xy, _view = state_from_observation(observation)
                    command, _diagnostic = planner.choose(
                        state, command_index, fields
                    )
                    primitive, path_xy, rewards = _execute(
                        env, executor, command, hidden_goal=goal
                    )
                    observation = env.unwrapped._get_obs()
                    endpoint = primitive[-1]
                    planner.observe_transition(state, command, endpoint)
                    fields.extend(path_xy, rewards)
                    planner.observe_passage(path_xy, rewards)
                    episode_success = episode_success or bool(np.any(rewards > 0))
                planner.end_episode(fields.contacted, endpoint[:2])
                if episode_success and first is None:
                    first = episode
                successes.append(bool(episode_success))
            row = {
                "goal_index": goal_index,
                "goal_xyz": goal.tolist(),
                "first_contact_episode": first,
                "successes": successes,
                "acquired": first is not None,
                "conditional_return": (
                    float(np.mean(successes[first + 1:]))
                    if first is not None and first + 1 < len(successes) else None
                ),
                "planner_diagnostics": planner.diagnostics(),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        env.close()
    acquired = sum(row["acquired"] for row in rows)
    returns = [
        row["conditional_return"] for row in rows
        if row["conditional_return"] is not None
    ]
    report = {
        **common_report(profile, "hidden_goal_development_evaluation"),
        "hidden_goal_accessed_by_evaluator": True,
        "hidden_goal_accessed_by_planner_executor_or_models": False,
        "goal_bank": str(paths(profile)["goal_bank"]),
        "goals": len(rows),
        "episodes_per_goal": episodes,
        "acquired_goals": acquired,
        "acquisition_rate": acquired / len(rows),
        "pooled_task_mean_conditional_return": (
            float(np.mean(returns)) if returns else None
        ),
        "rows": rows,
        "success_radius_role": "evaluator_only",
    }
    atomic_json(paths(profile)["evaluation"] / "summary.json", report)
    print(json.dumps(report, indent=2))
    return report


def audit():
    source = robotics_source_audit(require_commit=True)
    env = make_env(TASK, int(PROTOCOL["episode_primitive_steps"]))
    try:
        observation, _ = env.reset(seed=17)
        state, coordinate, view = state_from_observation(observation)
    finally:
        env.close()
    report = {
        **common_report("smoke", "adapter_audit"),
        "passed": bool(
            len(coordinate) == int(PROTOCOL["goal_dimension"])
            and len(state) == len(view.vector) + len(coordinate)
            and source["source_commit"] == PROTOCOL["gymnasium_robotics_commit"]
        ),
        "robotics_source": source,
        "executor_goal_coordinate": coordinate.tolist(),
        "goal_dimension": int(len(coordinate)),
        "physical_observation_dimension": int(len(view.vector)),
        "consequence_state_dimension": int(len(state)),
        "derived_horizon": finite_horizon(
            int(PROTOCOL["episode_primitive_steps"])
        ),
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("FetchPush point-1 adapter audit failed")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("audit", "freeze-goals", "prepare", "evaluate", "smoke"),
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "audit":
        audit()
    elif args.command == "freeze-goals":
        count = (
            1 if args.profile == "smoke"
            else int(PROTOCOL["development_goal_count"])
        )
        print(json.dumps(freeze_goals(args.profile, count), indent=2))
    elif args.command == "prepare":
        prepare(args.profile, args.device)
    elif args.command == "evaluate":
        evaluate(args.profile, args.device)
    else:
        audit()
        prepare("smoke", args.device)
        evaluate("smoke", args.device, smoke=True)


if __name__ == "__main__":
    main()
