"""Independent frozen-R74 PointMaze preparation for seeds 0--2."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from projects.tfsn_current_definition.pointmaze.core import ExperienceCommandPool
from projects.tfsn_consequence_value_iteration.shared.aligned_rebuild import (
    METHOD_REVISION,
    PROTOCOL,
    PROTOCOL_VALUE,
    REFERENCE,
    SPECIFICATION,
    atomic_json,
    atomic_npz,
    dimensions,
    output_paths,
    result_root,
    seed_values,
    sha256,
    task_seed,
)
from projects.tfsn_consequence_value_iteration.shared.factorized import (
    FactorizedConsequenceEnsemble,
    consequence_features,
    train_head,
)
from projects.tfsn_consequence_value_iteration.shared.settings import (
    ARTIFACT_NODE_SHARD,
    BATCH_SIZE,
    BRIDGE_CANDIDATES,
    EPOCHS,
    LEARNING_RATE,
    VALIDATION_FRACTION,
)

from .core import atomic_torch, load_json
from .dataset import CommandDataset, grouped_split
from .fields import SuppliedGoalField, TaskFields
from .frozen_r74_planner import load_frozen_r74_planner
from .passages import prepare_passages
from .roadmap import graph_from_dataset, reach_targets
from .train_factorized import endpoint_metrics


HERE = Path(__file__).resolve().parent
CLAIM_LEVEL = "reward_free_cross_domain_multiseed_preparation_rebuild_only"
PHASES = ("bootstrap", "acquisition", "return")


def _actor_contract(seed: int) -> dict:
    contract = load_json(HERE / "protocol.json")
    row = contract["executor"]["seeds"][str(int(seed))]
    for name in ("checkpoint", "manifest"):
        path = Path(row[name]).resolve()
        if not path.is_file() or sha256(path) != row[f"{name}_sha256"]:
            raise ValueError(f"PointMaze executor {seed} {name} changed")
    return row


def _current_paths() -> dict:
    base = HERE.parent / "results/pointmaze"
    row = PROTOCOL_VALUE["pointmaze"]
    return {
        name: (HERE.parents[2] / row[f"current_{name}"]).resolve()
        for name in ("models", "roadmap", "planner")
    }


def _shard_path(directory: Path, task: int) -> Path:
    return Path(directory) / f"task_{int(task):04d}.npz"


def _load_shard(path: Path, rows: int) -> dict:
    with np.load(path, allow_pickle=False) as stored:
        value = {name: stored[name] for name in stored.files}
    tau = int(PROTOCOL_VALUE["fixed_command_duration"])
    required = {
        "start_observation": (rows, 4),
        "command_xy": (rows, 2),
        "endpoint_observation": (rows, 4),
        "primitive_observation": (rows, tau, 4),
        "duration": (rows,),
        "reset_seed": (rows,),
        "physical_episode": (rows,),
        "command_index": (rows,),
        "task_id": (rows,),
    }
    for name, shape in required.items():
        array = value.get(name)
        if array is None or array.shape != shape:
            actual = None if array is None else array.shape
            raise ValueError(f"invalid {path}: {name} {actual}, expected {shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite {name} in {path}")
    if not np.all(value["duration"] == tau):
        raise ValueError(f"command duration changed in {path}")
    return value


def _point_runtime(seed: int, device: str):
    from scripts.pointmaze.gc_executor.runtime import (  # noqa: PLC0415
        GCSACExecutorRuntime,
        observation_vector,
    )
    from scripts.pointmaze.rnd_fb.env import PointMazeTimeStepEnv  # noqa: PLC0415

    actor = GCSACExecutorRuntime(Path(_actor_contract(seed)["checkpoint"]), device)
    env = PointMazeTimeStepEnv(
        PROTOCOL_VALUE["pointmaze"]["environment"],
        int(PROTOCOL_VALUE["episode_primitive_steps"]),
    )
    return actor, env, observation_vector


def _execute(actor, env, observation_vector, command, tau):
    rows = []
    for _ in range(int(tau)):
        env.step(actor.act(env.raw_obs, command))
        rows.append(observation_vector(env.raw_obs)[:4].astype(np.float32))
    return np.asarray(rows, np.float32)


def _reset_bootstrap(args, env, observation_vector, tasks, commands):
    path = output_paths(args.output)["collection"] / "reset_bootstrap.npz"
    horizon = int(PROTOCOL_VALUE["commands_per_physical_episode"])
    episodes = int(np.ceil(commands / horizon))
    expected_seed = np.empty((tasks, episodes), np.int64)
    expected_obs = np.empty((tasks, episodes, 4), np.float32)
    for task in range(tasks):
        for episode in range(episodes):
            seed = task_seed(args.preparation_seed, 0, task) + episode
            expected_seed[task, episode] = seed
            env.reset(seed=int(seed))
            expected_obs[task, episode] = observation_vector(env.raw_obs)[:4]
    if path.is_file():
        with np.load(path, allow_pickle=False) as stored:
            if not np.array_equal(stored["seed"], expected_seed):
                raise ValueError("PointMaze reset-bootstrap seeds changed")
            if not np.array_equal(stored["observation"], expected_obs):
                raise ValueError("PointMaze reset-bootstrap replay changed")
    else:
        atomic_npz(path, seed=expected_seed, observation=expected_obs)
    return expected_seed, expected_obs


def command_audit(args):
    actor = _actor_contract(args.preparation_seed)
    current = _current_paths()
    required = [
        current["models"] / "factorized_consequence.pt",
        current["roadmap"] / "recorded_graph.npz",
        current["planner"] / "bridges.npz",
    ]
    report = {
        "schema_version": 1,
        "method_reference": str(SPECIFICATION),
        "method_reference_sha256": sha256(SPECIFICATION),
        "rebuild_reference": str(REFERENCE),
        "rebuild_reference_sha256": sha256(REFERENCE),
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256(PROTOCOL),
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "declared_deviations": [],
        "oracle_inputs": [],
        "domain": "pointmaze",
        "preparation_seed": int(args.preparation_seed),
        "executor_checkpoint": actor["checkpoint"],
        "executor_checkpoint_sha256": actor["checkpoint_sha256"],
        "required_current_alignment_controller_artifacts": [
            {"path": str(path), "exists": path.is_file()} for path in required
        ],
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "confirmation_panel_accessed": False,
        "passed": all(path.is_file() for path in required),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(output_paths(args.output)["audit"] / "preflight.json", report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("PointMaze aligned rebuild preflight failed")
    return report


def command_collect_bootstrap(args):
    bootstrap_tasks, _aligned_tasks, commands = dimensions(args.profile)
    paths = output_paths(args.output)
    actor, env, observation_vector = _point_runtime(
        args.preparation_seed, args.device
    )
    tau = int(PROTOCOL_VALUE["fixed_command_duration"])
    horizon = int(PROTOCOL_VALUE["commands_per_physical_episode"])
    started = time.time()
    try:
        reset_seed, reset_obs = _reset_bootstrap(
            args, env, observation_vector, bootstrap_tasks, commands
        )
        pool = ExperienceCommandPool()
        for xy in reset_obs[:, :, :2].reshape(-1, 2):
            pool.add(xy)
        completed = 0
        while completed < bootstrap_tasks:
            path = _shard_path(paths["bootstrap"], completed)
            if not path.is_file():
                break
            shard = _load_shard(path, commands)
            for endpoint in shard["endpoint_observation"]:
                pool.add(endpoint[:2])
            completed += 1
        if len(list(paths["bootstrap"].glob("task_*.npz"))) != completed:
            raise RuntimeError("PointMaze bootstrap shards are not contiguous")
        for task in range(completed, bootstrap_tasks):
            rng = np.random.default_rng(task_seed(args.preparation_seed, 0, task))
            starts, targets, endpoints, primitive = [], [], [], []
            seeds, episodes, indices = [], [], []
            for row in range(commands):
                episode, command_index = divmod(row, horizon)
                if command_index == 0:
                    env.reset(seed=int(reset_seed[task, episode]))
                    now = observation_vector(env.raw_obs)[:4].astype(np.float32)
                    if not np.array_equal(now, reset_obs[task, episode]):
                        raise RuntimeError("PointMaze reset replay changed")
                state = observation_vector(env.raw_obs)[:4].astype(np.float32)
                _pool_index, command = pool.sample_uniform(rng)
                trajectory = _execute(
                    actor, env, observation_vector, command, tau
                )
                endpoint = trajectory[-1]
                pool.add(endpoint[:2])
                starts.append(state)
                targets.append(command)
                endpoints.append(endpoint)
                primitive.append(trajectory)
                seeds.append(reset_seed[task, episode])
                episodes.append(episode)
                indices.append(command_index)
            path = _shard_path(paths["bootstrap"], task)
            atomic_npz(
                path,
                start_observation=np.asarray(starts, np.float32),
                command_xy=np.asarray(targets, np.float32),
                endpoint_observation=np.asarray(endpoints, np.float32),
                primitive_observation=np.asarray(primitive, np.float32),
                duration=np.full(commands, tau, np.int16),
                reset_seed=np.asarray(seeds, np.int64),
                physical_episode=np.asarray(episodes, np.int16),
                command_index=np.asarray(indices, np.int16),
                task_id=np.full(commands, task, np.int16),
            )
            _load_shard(path, commands)
            print(
                f"point_r16 bootstrap seed={args.preparation_seed} "
                f"task={task + 1}/{bootstrap_tasks}", flush=True,
            )
    finally:
        env.close()
    report = _collection_summary(
        args, "bootstrap", bootstrap_tasks, commands, started
    )
    atomic_json(paths["bootstrap"] / "summary.json", report)
    print(json.dumps(report, indent=2))
    return report


def _collection_summary(args, phase, tasks, commands, started):
    actor = _actor_contract(args.preparation_seed)
    return {
        "schema_version": 1,
        "method_reference": str(SPECIFICATION),
        "method_reference_sha256": sha256(SPECIFICATION),
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL if args.profile == "full" else "smoke_only",
        "declared_deviations": [],
        "oracle_inputs": [],
        "domain": "pointmaze",
        "preparation_seed": int(args.preparation_seed),
        "profile": args.profile,
        "phase": phase,
        "tasks": int(tasks),
        "commands_per_task": int(commands),
        "rows": int(tasks * commands),
        "fixed_command_duration": int(PROTOCOL_VALUE["fixed_command_duration"]),
        "commands_per_physical_episode": int(
            PROTOCOL_VALUE["commands_per_physical_episode"]
        ),
        "executor_checkpoint_sha256": actor["checkpoint_sha256"],
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "environment_reward_consumed": False,
        "confirmation_panel_accessed": False,
        "wall_clock_seconds_this_invocation": time.time() - started,
    }


def command_collect_alignment(args):
    _bootstrap_tasks, tasks, commands = dimensions(args.profile)
    paths = output_paths(args.output)
    current = _current_paths()
    actor, env, observation_vector = _point_runtime(
        args.preparation_seed, args.device
    )
    tau = int(PROTOCOL_VALUE["fixed_command_duration"])
    horizon = int(PROTOCOL_VALUE["commands_per_physical_episode"])
    planner = load_frozen_r74_planner(
        models=current["models"], roadmap=current["roadmap"],
        planner=current["planner"], device=args.device,
        seed=seed_values(args.preparation_seed)["collection_seed_base"],
    )
    started = time.time()
    try:
        for phase_index, phase in enumerate(("acquisition", "return"), start=1):
            directory = paths[phase]
            completed = 0
            while completed < tasks and _shard_path(directory, completed).is_file():
                _load_shard(_shard_path(directory, completed), commands)
                completed += 1
            if len(list(directory.glob("task_*.npz"))) != completed:
                raise RuntimeError(f"PointMaze {phase} shards are not contiguous")
            for task in range(completed, tasks):
                seed = task_seed(args.preparation_seed, phase_index, task)
                rng = np.random.default_rng(seed)
                planner.reset_task_memory()
                if phase == "acquisition":
                    fields = TaskFields(planner.sigma)
                else:
                    anchor = planner.hub_xy[int(rng.integers(len(planner.hub_xy)))]
                    fields = SuppliedGoalField(planner.sigma, anchor)
                starts, targets, endpoints, primitive = [], [], [], []
                seeds, episodes, indices = [], [], []
                for row in range(commands):
                    episode, command_index = divmod(row, horizon)
                    if command_index == 0:
                        reset_seed = seed + episode
                        env.reset(seed=int(reset_seed))
                        planner.begin_episode()
                        if phase == "acquisition":
                            reset = observation_vector(env.raw_obs)[:4]
                            fields.extend(reset[None, :2], np.zeros(1, np.float32))
                    state = observation_vector(env.raw_obs)[:4].astype(np.float32)
                    command, _diagnostic = planner.choose(
                        state, command_index, fields
                    )
                    trajectory = _execute(
                        actor, env, observation_vector, command, tau
                    )
                    rewards = np.zeros(tau, np.float32)
                    if phase == "acquisition":
                        fields.extend(trajectory[:, :2], rewards)
                        planner.observe_passage(trajectory[:, :2], rewards)
                    endpoint = trajectory[-1]
                    planner.observe_transition(state, command, endpoint)
                    starts.append(state)
                    targets.append(command)
                    endpoints.append(endpoint)
                    primitive.append(trajectory)
                    seeds.append(reset_seed)
                    episodes.append(episode)
                    indices.append(command_index)
                    if command_index + 1 == horizon or row + 1 == commands:
                        planner.end_episode(False, endpoint[:2])
                path = _shard_path(directory, task)
                atomic_npz(
                    path,
                    start_observation=np.asarray(starts, np.float32),
                    command_xy=np.asarray(targets, np.float32),
                    endpoint_observation=np.asarray(endpoints, np.float32),
                    primitive_observation=np.asarray(primitive, np.float32),
                    duration=np.full(commands, tau, np.int16),
                    reset_seed=np.asarray(seeds, np.int64),
                    physical_episode=np.asarray(episodes, np.int16),
                    command_index=np.asarray(indices, np.int16),
                    task_id=np.full(commands, task, np.int16),
                )
                _load_shard(path, commands)
                print(
                    f"point_r16 {phase} seed={args.preparation_seed} "
                    f"task={task + 1}/{tasks}", flush=True,
                )
            report = _collection_summary(
                args, phase, tasks, commands, started
            )
            report["alignment_controller_revision"] = (
                planner.method_revision
            )
            atomic_json(directory / "summary.json", report)
    finally:
        env.close()
    summaries = {
        phase: load_json(paths[phase] / "summary.json")
        for phase in ("acquisition", "return")
    }
    print(json.dumps(summaries, indent=2))
    return summaries


def load_dataset(root: Path, profile: str) -> CommandDataset:
    bootstrap_tasks, aligned_tasks, commands = dimensions(profile)
    paths = output_paths(root)
    starts, targets, endpoints, primitive = [], [], [], []
    trajectory, source = [], []
    names, directories = [], []
    for source_id, (phase, tasks) in enumerate((
        ("bootstrap", bootstrap_tasks),
        ("acquisition", aligned_tasks),
        ("return", aligned_tasks),
    )):
        directory = paths[phase]
        for task in range(tasks):
            shard = _load_shard(_shard_path(directory, task), commands)
            starts.append(shard["start_observation"])
            targets.append(shard["command_xy"])
            endpoints.append(shard["endpoint_observation"])
            primitive.append(shard["primitive_observation"])
            trajectory.append(np.full(
                commands, source_id * 1_000_000 + task, np.int64
            ))
            source.append(np.full(commands, source_id, np.int16))
        names.append(f"r16_{phase}")
        directories.append(directory)
    return CommandDataset(
        starts=np.concatenate(starts), commands=np.concatenate(targets),
        endpoints=np.concatenate(endpoints), primitives=np.concatenate(primitive),
        trajectory_id=np.concatenate(trajectory), source_id=np.concatenate(source),
        source_names=tuple(names), source_paths=tuple(directories),
    )


def command_train(args):
    data = load_dataset(args.output, args.profile)
    seeds = seed_values(args.preparation_seed)
    train_index, validation_index = grouped_split(
        data, seeds["split_seed"], VALIDATION_FRACTION
    )
    features = consequence_features(data.starts, data.commands)
    spatial_target = data.endpoints[:, :2] - data.commands
    posture_features = np.concatenate(
        (features, data.endpoints[:, :2]), axis=1
    ).astype(np.float32)
    posture_target = data.endpoints[:, 2:]
    epochs = EPOCHS if args.profile == "full" else 2
    started = time.time()
    spatial = train_head(
        features, spatial_target, train_index, validation_index,
        seeds["training_seed"] + 1, args.device, epochs=epochs,
        batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
        label=f"point_r16_seed{args.preparation_seed}_spatial",
    )
    posture = train_head(
        posture_features, posture_target, train_index, validation_index,
        seeds["training_seed"] + 2, args.device, epochs=epochs,
        batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
        label=f"point_r16_seed{args.preparation_seed}_velocity",
    )
    paths = output_paths(args.output)
    checkpoint = paths["models"] / "factorized_consequence.pt"
    payload = {
        "artifact_type": "factorized_consequence",
        "schema_version": 1,
        "domain": "PointMaze_Medium-v3",
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL if args.profile == "full" else "smoke_only",
        "method_reference": str(SPECIFICATION),
        "preparation_seed": int(args.preparation_seed),
        "state_dim": 4,
        "position_dim": 2,
        "posture_semantics": "endpoint_velocity_xy",
        "spatial_parameterization": "endpoint_xy_minus_command_xy",
        "factorization": (
            "P(endpoint_xy-command|state,command) times "
            "P(endpoint_velocity|state,command,endpoint_xy)"
        ),
        "ensemble_bootstrap": True,
        "training_rows": int(len(train_index)),
        "validation_rows": int(len(validation_index)),
        "split": PROTOCOL_VALUE["training_schedule"]["split"],
        "split_seed": seeds["split_seed"],
        "training_seed": seeds["training_seed"],
        "source_names": list(data.source_names),
        "source_paths": [str(path) for path in data.source_paths],
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
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "task_reward_used": False,
        "hidden_goals_used": False,
        "oracle_inputs": [],
    }
    atomic_torch(checkpoint, payload)
    model = FactorizedConsequenceEnsemble(checkpoint, args.device)
    validation = endpoint_metrics(model, data, validation_index)
    report = {
        "schema_version": 1,
        "artifact": str(checkpoint),
        "artifact_sha256": sha256(checkpoint),
        "method_revision": METHOD_REVISION,
        "claim_level": payload["claim_level"],
        "schedule": args.profile,
        "preparation_seed": int(args.preparation_seed),
        "rows": data.rows,
        "training_rows": int(len(train_index)),
        "validation_rows": int(len(validation_index)),
        "source_names": list(data.source_names),
        "validation": validation,
        "kernel_sigma_m": validation["landing_precision_m"],
        "kernel_sigma_estimator": (
            "held_out_ensemble_mean_mixture_weighted_within_mode_spatial_spread"
        ),
        "task_reward_used": False,
        "hidden_goals_used": False,
        "oracle_inputs": [],
        "wall_clock_seconds": time.time() - started,
    }
    atomic_json(paths["models"] / "factorized_consequence_metrics.json", report)
    print(json.dumps(report, indent=2))
    return report


def command_prepare_roadmap(args):
    paths = output_paths(args.output)
    data = load_dataset(args.output, args.profile)
    metrics = load_json(paths["models"] / "factorized_consequence_metrics.json")
    if sha256(metrics["artifact"]) != metrics["artifact_sha256"]:
        raise ValueError("rebuilt PointMaze consequence checkpoint changed")
    sigma = float(metrics["kernel_sigma_m"])
    graph = graph_from_dataset(data, sigma)
    candidate_count = int(BRIDGE_CANDIDATES)
    if args.profile == "smoke":
        displacement = (
            graph.nodes[graph.endpoint_index, :2]
            - graph.nodes[graph.source_index, :2]
        )
        requested = np.linalg.norm(
            graph.commands - graph.nodes[graph.source_index, :2], axis=1
        )
        achieved = np.linalg.norm(displacement, axis=1)
        denominator = 2.0 * sigma**2
        positive = (
            (1.0 - np.exp(-np.square(requested) / denominator))
            * (1.0 - np.exp(-np.square(achieved) / denominator))
        ) > 0.0
        # Smoke has only 18 rows and cannot instantiate the declared 128
        # empirical representatives.  This exercises the same code path with
        # every positive-weight row; full preparation remains exactly 128.
        candidate_count = max(1, min(candidate_count, int(positive.sum())))
    landmarks, targets, distance, offsets = reach_targets(
        graph, candidate_count, sigma=sigma,
        continuous_weighting=True,
    )
    graph_path = paths["roadmap"] / "recorded_graph.npz"
    reach_path = paths["roadmap"] / "reach_candidates.npz"
    graph.save(graph_path)
    atomic_npz(
        reach_path, landmark_index=landmarks, target_index=targets,
        target_distance=distance, reach_offsets=offsets,
    )
    passage = prepare_passages(
        graph, targets, sigma, paths["roadmap"] / "passages",
        shard_nodes=ARTIFACT_NODE_SHARD,
    )
    report = {
        "schema_version": 1,
        "artifact_type": "pointmaze_recorded_roadmap",
        "method_reference": str(SPECIFICATION),
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL if args.profile == "full" else "smoke_only",
        "declared_deviations": [],
        "oracle_inputs": [],
        "schedule": args.profile,
        "preparation_seed": int(args.preparation_seed),
        "dataset_rows": data.rows,
        "dataset_sources": list(data.source_names),
        "graph": str(graph_path),
        "graph_sha256": sha256(graph_path),
        "reach_candidates": str(reach_path),
        "reach_candidates_sha256": sha256(reach_path),
        "factorized_consequence_sha256": metrics["artifact_sha256"],
        "landing_sigma_m": sigma,
        "recorded_nodes": graph.node_count,
        "recorded_edges": graph.edge_count,
        "recorded_nodes_deleted": False,
        "recorded_edges_deleted": False,
        "landmarks": int(len(landmarks)),
        "bridge_candidates_per_node": int(targets.shape[1]),
        "smoke_candidate_count_reduced_for_available_rows": bool(
            args.profile == "smoke" and candidate_count != BRIDGE_CANDIDATES
        ),
        "continuous_reach_weighting": True,
        "movement_request_threshold": None,
        "passage_geometry": passage,
        "task_reward_used": False,
        "hidden_goals_used": False,
    }
    atomic_json(paths["roadmap"] / "roadmap.json", report)
    print(json.dumps(report, indent=2))
    return report


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--preparation-seed", type=int, choices=(0, 1, 2), required=True)
    value.add_argument("--profile", choices=("smoke", "full"), default="full")
    value.add_argument("--device", default="cuda")
    value.add_argument("--output", type=Path)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").set_defaults(func=command_audit)
    commands.add_parser("collect-bootstrap").set_defaults(func=command_collect_bootstrap)
    commands.add_parser("collect-alignment").set_defaults(func=command_collect_alignment)
    commands.add_parser("train-consequence").set_defaults(func=command_train)
    commands.add_parser("prepare-roadmap").set_defaults(func=command_prepare_roadmap)
    return value


def main():
    args = parser().parse_args()
    if args.output is None:
        args.output = result_root("pointmaze", args.preparation_seed, args.profile)
    args.output = args.output.resolve()
    args.func(args)


if __name__ == "__main__":
    main()
