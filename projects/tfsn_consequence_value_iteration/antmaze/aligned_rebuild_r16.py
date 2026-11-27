"""Independent frozen-R74 AntMaze preparation for seeds 0--2.

The output deliberately retains the R52 runtime layout.  R74/R75 can therefore
consume a rebuilt seed by changing only ``--preparation``.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from projects.tfsn_current_definition.pointmaze.core import ExperienceCommandPool
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
    r52_e549_preparation as r52,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r53_e549_r51_evaluation as r53,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r56_e549_supplied_goal as r56,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r67_e549_hub_routed_exploration as r67,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r68_e549_dynamic_hubs as r68,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r71_sigma_generalized_correction as r71,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r74_reach_scale_route_correction as r74,
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
from projects.tfsn_consequence_value_iteration.shared.aligned_rebuild import (
    METHOD_REVISION,
    PROTOCOL,
    PROTOCOL_VALUE,
    REFERENCE,
    SPECIFICATION,
    atomic_json,
    dimensions,
    result_root,
    seed_values,
    sha256,
    task_seed,
)
from projects.tfsn_consequence_value_iteration.shared.reliability import (
    recorded_transition_reliability,
)
from projects.tfsn_consequence_value_iteration.shared.settings import (
    BATCH_SIZE,
    BRIDGE_CANDIDATES,
    EPOCHS,
    LEARNING_RATE,
    RECORDED_EVIDENCE_PRIOR_WEIGHT,
    RECORDED_RELIABILITY_NEIGHBORS,
    VALIDATION_FRACTION,
)
from scripts.antmaze.current_tfsn.collect import execute_command
from scripts.antmaze.current_tfsn.constants import COMMAND_STEPS, STATE_DIM
from scripts.antmaze.current_tfsn.executor import full_state, make_env


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
CLAIM_LEVEL = "reward_free_cross_domain_multiseed_preparation_rebuild_only"
PHASES = ("bootstrap", "acquisition", "return")


def output_paths(output: Path) -> dict[str, Path]:
    output = Path(output).resolve()
    return {
        "root": output,
        "collection": output / "collection",
        "bootstrap": output / "collection/bootstrap",
        "acquisition": output / "collection/alignment/acquisition",
        "return": output / "collection/alignment/return",
        "audit": output / "audit",
        "collection_summary": output / "collection/summary.json",
        "consequence_directory": output / "consequence",
        "consequence": output / "consequence/consequence.pt",
        "consequence_summary": output / "consequence/summary.json",
        "runtime": output / "runtime",
        "graph": output / "runtime/recorded_transition_graph.npz",
    }


def _part_path(directory: Path, task: int) -> Path:
    return Path(directory) / f"part_{int(task):05d}.npz"


def _validate_part(path: Path, task: int, commands: int) -> dict:
    with np.load(path, allow_pickle=False) as stored:
        value = {name: stored[name] for name in stored.files}
    required = {
        "states": (commands, COMMAND_STEPS + 1, STATE_DIM),
        "commands": (commands, 2),
        "task_id": (commands,),
        "episode_id": (commands,),
        "command_index": (commands,),
        "reset_seed": (commands,),
    }
    if set(value) != set(required):
        raise ValueError(f"invalid AntMaze rebuild shard keys in {path}")
    for name, shape in required.items():
        if value[name].shape != shape or not np.isfinite(value[name]).all():
            raise ValueError(f"invalid AntMaze rebuild {name} in {path}")
    if not np.all(value["task_id"] == int(task)):
        raise ValueError(f"AntMaze rebuild task id changed in {path}")
    index = np.arange(commands)
    horizon = int(PROTOCOL_VALUE["commands_per_physical_episode"])
    if not np.array_equal(value["episode_id"], index // horizon):
        raise ValueError(f"AntMaze rebuild episode ids changed in {path}")
    if not np.array_equal(value["command_index"], index % horizon):
        raise ValueError(f"AntMaze rebuild command indices changed in {path}")
    for row in range(1, commands):
        if value["command_index"][row] == 0:
            continue
        if not np.array_equal(value["states"][row, 0], value["states"][row - 1, -1]):
            raise ValueError(f"AntMaze rebuild command chain broke in {path}")
    return value


def _save_part(path: Path, **arrays) -> None:
    r47.atomic_savez(path, **arrays)


def _phase_summary(args, phase: str, tasks: int, commands: int, started: float) -> dict:
    return {
        "schema_version": 1,
        "method_reference": str(SPECIFICATION),
        "method_reference_sha256": sha256(SPECIFICATION),
        "rebuild_reference": str(REFERENCE),
        "rebuild_reference_sha256": sha256(REFERENCE),
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256(PROTOCOL),
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL if args.profile == "full" else "smoke_only",
        "declared_deviations": [],
        "oracle_inputs": [],
        "domain": "antmaze",
        "preparation_seed": int(args.preparation_seed),
        "profile": args.profile,
        "phase": phase,
        "tasks": int(tasks),
        "commands_per_task": int(commands),
        "rows": int(tasks * commands),
        "fixed_command_duration": COMMAND_STEPS,
        "commands_per_physical_episode": int(
            PROTOCOL_VALUE["commands_per_physical_episode"]
        ),
        "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "environment_reward_consumed": False,
        "confirmation_panel_accessed": False,
        "wall_clock_seconds_this_invocation": time.time() - started,
    }


def command_audit(args) -> dict:
    executor = inspect_e549_checkpoint()
    current = (MANAGEMENT_ROOT / PROTOCOL_VALUE["antmaze"]["current_preparation"]).resolve()
    required = [
        current / "consequence/consequence.pt",
        current / "runtime/recorded_transition_graph.npz",
        current / "runtime/planner/bridges.npz",
        current / "runtime/empirical_passage/metadata.json",
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
        "domain": "antmaze",
        "preparation_seed": int(args.preparation_seed),
        "executor": executor,
        "required_current_alignment_controller_artifacts": [
            {"path": str(path), "exists": path.is_file()} for path in required
        ],
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "confirmation_panel_accessed": False,
        "passed": bool(
            executor["sha256"] == E549_CHECKPOINT_SHA256
            and all(path.is_file() for path in required)
        ),
    }
    output_paths(args.output)["audit"].mkdir(parents=True, exist_ok=True)
    atomic_json(output_paths(args.output)["audit"] / "preflight.json", report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("AntMaze aligned rebuild preflight failed")
    return report


def _reset_bank(args, env, tasks: int, commands: int) -> tuple[np.ndarray, np.ndarray]:
    paths = output_paths(args.output)
    path = paths["collection"] / "reset_bootstrap.npz"
    horizon = int(PROTOCOL_VALUE["commands_per_physical_episode"])
    episodes = int(np.ceil(commands / horizon))
    seeds = np.empty((tasks, episodes), np.int64)
    states = np.empty((tasks, episodes, STATE_DIM), np.float32)
    for task in range(tasks):
        for episode in range(episodes):
            seed = task_seed(args.preparation_seed, 0, task) + episode
            observation, _ = env.reset(seed=int(seed))
            seeds[task, episode] = seed
            states[task, episode] = full_state(observation)
    if path.is_file():
        with np.load(path, allow_pickle=False) as stored:
            if not np.array_equal(stored["seed"], seeds):
                raise ValueError("AntMaze bootstrap reset seeds changed")
            if not np.array_equal(stored["state"], states):
                raise ValueError("AntMaze bootstrap reset replay changed")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        r47.atomic_savez(path, seed=seeds, state=states)
    return seeds, states


def command_collect_bootstrap(args) -> dict:
    tasks, _aligned, commands = dimensions(args.profile)
    paths = output_paths(args.output)
    executor = E549SACExecutor(E549_CHECKPOINT, args.device)
    env = make_env(max_episode_steps=10_000_000)
    horizon = int(PROTOCOL_VALUE["commands_per_physical_episode"])
    started = time.time()
    try:
        reset_seed, reset_state = _reset_bank(args, env, tasks, commands)
        pool = ExperienceCommandPool()
        for xy in reset_state[:, :, :2].reshape(-1, 2):
            pool.add(xy)
        completed = 0
        while completed < tasks and _part_path(paths["bootstrap"], completed).is_file():
            shard = _validate_part(_part_path(paths["bootstrap"], completed), completed, commands)
            for endpoint in shard["states"][:, -1, :2]:
                pool.add(endpoint)
            completed += 1
        if len(list(paths["bootstrap"].glob("part_*.npz"))) != completed:
            raise RuntimeError("AntMaze bootstrap parts are not contiguous")
        for task in range(completed, tasks):
            rng = np.random.default_rng(task_seed(args.preparation_seed, 0, task))
            trajectories, targets, seeds, episodes, indices = [], [], [], [], []
            state = None
            for row in range(commands):
                episode, command_index = divmod(row, horizon)
                if command_index == 0:
                    observation, _ = env.reset(seed=int(reset_seed[task, episode]))
                    state = full_state(observation)
                    if not np.array_equal(state, reset_state[task, episode]):
                        raise RuntimeError("AntMaze bootstrap reset replay changed")
                _pool_index, command = pool.sample_uniform(rng)
                trajectory, rewards = execute_command(
                    env, executor, state, command, hidden_goal=None
                )
                if rewards.any():
                    raise RuntimeError("task reward appeared in AntMaze bootstrap")
                trajectories.append(trajectory.astype(np.float32))
                targets.append(np.asarray(command, np.float32))
                seeds.append(reset_seed[task, episode])
                episodes.append(episode)
                indices.append(command_index)
                state = trajectory[-1]
                pool.add(state[:2])
            path = _part_path(paths["bootstrap"], task)
            _save_part(
                path,
                states=np.asarray(trajectories, np.float32),
                commands=np.asarray(targets, np.float32),
                task_id=np.full(commands, task, np.int16),
                episode_id=np.asarray(episodes, np.int16),
                command_index=np.asarray(indices, np.int16),
                reset_seed=np.asarray(seeds, np.int64),
            )
            _validate_part(path, task, commands)
            print(
                f"ant_r16 bootstrap seed={args.preparation_seed} "
                f"task={task + 1}/{tasks}", flush=True,
            )
    finally:
        env.close()
    report = _phase_summary(args, "bootstrap", tasks, commands, started)
    atomic_json(paths["bootstrap"] / "summary.json", report)
    print(json.dumps(report, indent=2))
    return report


class _OnDemandOfflineReachability:
    """Exact R69 table columns without materializing the historical table.

    R74 reads the frozen R69 table only before a task has online link
    evidence.  Its active corrected scorer already contains the equivalent
    exact batched Bellman solver.  This compatibility view evaluates the one
    ``[:, remaining, current_node]`` column requested by R74 and caches it.
    It therefore preserves the R69 recursion while avoiding a large,
    disposable preparation artifact in the independent rebuild.
    """

    def __init__(self, scorer, hub_nodes):
        self.scorer = scorer
        self.hub_nodes = np.asarray(hub_nodes, np.int64).reshape(-1)
        self._cache_key = None
        self._cache_value = None

    def _column(self, remaining: int, current_node: int) -> np.ndarray:
        key = (int(remaining), int(current_node))
        if key == self._cache_key:
            return self._cache_value
        solver = self.scorer.batched_reachability
        if solver is None:
            raise RuntimeError(
                "the frozen R74 exact batched reachability solver is absent"
            )
        solver.set_bridge_weight(self.scorer.bridge_weight)
        chunks = []
        for begin in range(0, len(self.hub_nodes), solver.target_batch):
            targets = self.hub_nodes[begin:begin + solver.target_batch]
            chunks.append(solver.solve(targets, key[0], key[1]))
        value = np.concatenate(chunks).astype(np.float32, copy=False)
        if value.shape != self.hub_nodes.shape:
            raise RuntimeError("on-demand R74 reachability shape changed")
        self._cache_key = key
        self._cache_value = value
        return value

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 3:
            raise IndexError(
                "reachability must be indexed as hubs, remaining, node"
            )
        hub_index, remaining, current_node = key
        return self._column(remaining, current_node)[hub_index]


def _current_r74_runtime(args):
    current = (
        MANAGEMENT_ROOT
        / PROTOCOL_VALUE["antmaze"]["current_preparation"]
    ).resolve()
    r74.configure_implementation()
    runtime_args = SimpleNamespace(
        root=args.root,
        device=args.device,
        seed=seed_values(args.preparation_seed)["collection_seed_base"],
        preparation=current,
        output=r74.DEFAULT_OUTPUT,
        graph=None,
    )
    # R74's historical loader enters through R69 and consequently requires
    # the complete cached (hub, horizon, node) table.  The table is merely a
    # computational cache: after the first task-local observation, R74 uses
    # its exact batched solver.  Build the identical runtime directly and
    # expose exact first-decision columns lazily instead of requiring that
    # cleaned historical artifact.
    r71.r69._install_contract()
    planner, executor, env = r53._load_runtime(runtime_args)
    planner.horizon = r74.HORIZON
    _contract, hub_nodes, hub_xy, arrays = r68._candidate_contract(
        runtime_args
    )

    r71._bind_r71_contract()
    planner.__class__ = r71.SigmaGeneralizedDestinationArrivalPlanner
    planner.method_revision = r71.METHOD_REVISION
    planner.planner_name = (
        "E549_R51_destination_arrival_sigma_link_generalization"
    )

    r74.implementation._bind_r72_contract()
    reach = r74.empirical_reach_covariance(current)
    planner.method_revision = r74.METHOD_REVISION
    planner.planner_name = (
        "E549_R51_sigma_link_and_empirical_reach_route_correction"
    )
    scorer = r74.ReachScaleHubRouteCorrectedScorer(
        planner,
        arrays,
        hub_nodes,
        reach["covariance_m2"],
        args.device,
    )
    planner.hub_score_override = scorer
    planner.hub_attempt_observer = scorer
    reachability = _OnDemandOfflineReachability(scorer, hub_nodes)
    return (
        planner, executor, env, hub_nodes, hub_xy, arrays[2], reachability,
    )


def _reset_planner_task(planner, seed: int) -> None:
    planner.rng = np.random.default_rng(int(seed))
    planner.seed = int(seed)
    planner.goal_key = None
    planner.reset_online_correction_task()
    observer = getattr(planner, "hub_attempt_observer", None)
    if observer is not None:
        observer.reset_task()


def _collect_aligned_task(
    args, phase: str, local_task: int, commands: int,
    planner, executor, env, hub_nodes, hub_xy, nodes, reachability,
) -> dict:
    phase_index = 1 if phase == "acquisition" else 2
    seed = task_seed(args.preparation_seed, phase_index, local_task)
    rng = np.random.default_rng(seed)
    _reset_planner_task(planner, seed)
    sigma = float(r48.SIGMA)
    horizon = int(PROTOCOL_VALUE["commands_per_physical_episode"])
    if phase == "acquisition":
        posterior = r67._area_posterior(nodes, sigma)
        fields = r48.TaskFields(
            sigma,
            search_mode=planner.acquisition_search_mode,
            coarse_sigma=planner.coarse_search_sigma,
            coverage_reference_xy=planner.coverage_reference_xy,
        )
        anchor = None
    else:
        anchor = np.asarray(hub_xy[int(rng.integers(len(hub_xy)))], np.float32)
        fields = r56.R56SuppliedGoalField(sigma, anchor)
        posterior = None
    trajectories, targets, seeds, episodes, indices = [], [], [], [], []
    online_states, completed = [], []
    current_memory = None
    active_hub = None
    observer = getattr(planner, "hub_attempt_observer", None)
    state = None
    for row in range(commands):
        episode, command_index = divmod(row, horizon)
        if command_index == 0:
            reset_seed = seed + episode
            observation, _ = env.reset(seed=int(reset_seed))
            state = full_state(observation)
            current_memory = [state.copy()]
            active_hub = None
            if phase == "acquisition":
                fields.add(state[:2], 0.0)
                r67._negative_path_update(posterior, state[:2].reshape(1, 2))
        if phase == "acquisition":
            if active_hub is None:
                remaining = horizon - command_index
                (
                    active_hub, _coverage, reach, _score, _current_node, _seconds,
                ) = planner.hub_score_override(
                    posterior, hub_xy, reachability, planner.graph,
                    state, remaining, online_states,
                )
                if observer is not None:
                    observer.start_attempt(
                        source_xy=state[:2],
                        hub_bank_index=int(active_hub),
                        hub_xy=hub_xy[active_hub],
                        remaining_commands=int(remaining),
                        predicted_reachability=float(reach[active_hub]),
                    )
            decision_field = r56.R56SuppliedGoalField(sigma, hub_xy[active_hub])
        else:
            decision_field = fields
        command, _diagnostic = planner.choose(
            state, command_index, decision_field, online_states,
            completed + [current_memory],
        )
        trajectory, rewards = execute_command(
            env, executor, state, command, hidden_goal=None
        )
        if rewards.any():
            raise RuntimeError("task reward appeared in AntMaze aligned collection")
        for next_state in trajectory[1:]:
            online_states.append(next_state.copy())
            current_memory.append(next_state.copy())
            if phase == "acquisition":
                fields.add(next_state[:2], 0.0)
        if phase == "acquisition":
            r67._negative_path_update(posterior, trajectory[1:, :2])
            arrived = bool(np.linalg.norm(
                trajectory[:, :2] - hub_xy[active_hub][None], axis=1
            ).min() <= sigma)
            if arrived:
                if observer is not None:
                    observer.finish_attempt(True, trajectory[-1, :2])
                active_hub = None
            elif command_index + 1 == horizon or row + 1 == commands:
                if observer is not None:
                    observer.finish_attempt(False, trajectory[-1, :2])
                active_hub = None
        trajectories.append(trajectory.astype(np.float32))
        targets.append(np.asarray(command, np.float32))
        seeds.append(reset_seed)
        episodes.append(episode)
        indices.append(command_index)
        state = trajectory[-1]
        if command_index + 1 == horizon or row + 1 == commands:
            completed.append(np.asarray(current_memory, np.float32))
            current_memory = None
    planner._ingest_pending(online_states)
    return {
        "states": np.asarray(trajectories, np.float32),
        "commands": np.asarray(targets, np.float32),
        "task_id": np.full(commands, local_task, np.int16),
        "episode_id": np.asarray(episodes, np.int16),
        "command_index": np.asarray(indices, np.int16),
        "reset_seed": np.asarray(seeds, np.int64),
    }


def command_collect_alignment(args) -> dict:
    _bootstrap, tasks, commands = dimensions(args.profile)
    paths = output_paths(args.output)
    started = time.time()
    loaded = _current_r74_runtime(args)
    planner, executor, env, hub_nodes, hub_xy, nodes, reachability = loaded
    try:
        summaries = {}
        for phase in ("acquisition", "return"):
            directory = paths[phase]
            completed = 0
            while completed < tasks and _part_path(directory, completed).is_file():
                _validate_part(_part_path(directory, completed), completed, commands)
                completed += 1
            if len(list(directory.glob("part_*.npz"))) != completed:
                raise RuntimeError(f"AntMaze {phase} parts are not contiguous")
            for task in range(completed, tasks):
                shard = _collect_aligned_task(
                    args, phase, task, commands, planner, executor, env,
                    hub_nodes, hub_xy, nodes, reachability,
                )
                path = _part_path(directory, task)
                _save_part(path, **shard)
                _validate_part(path, task, commands)
                print(
                    f"ant_r16 {phase} seed={args.preparation_seed} "
                    f"task={task + 1}/{tasks}", flush=True,
                )
            report = _phase_summary(args, phase, tasks, commands, started)
            report["alignment_controller_revision"] = r74.METHOD_REVISION
            report["synthetic_return_anchor"] = (
                "uniform_over_data_derived_alias_hub_bank"
                if phase == "return" else None
            )
            atomic_json(directory / "summary.json", report)
            summaries[phase] = report
    finally:
        env.close()
    print(json.dumps(summaries, indent=2))
    return summaries


def _load_rows(root: Path, profile: str):
    bootstrap_tasks, aligned_tasks, commands = dimensions(profile)
    paths = output_paths(root)
    starts, targets, endpoints, task_id, source_id = [], [], [], [], []
    source_names = []
    for source, (phase, tasks) in enumerate((
        ("bootstrap", bootstrap_tasks),
        ("acquisition", aligned_tasks),
        ("return", aligned_tasks),
    )):
        for task in range(tasks):
            shard = _validate_part(_part_path(paths[phase], task), task, commands)
            starts.append(shard["states"][:, 0])
            targets.append(shard["commands"])
            endpoints.append(shard["states"][:, -1])
            task_id.append(np.full(commands, source * 1_000_000 + task, np.int64))
            source_id.append(np.full(commands, source, np.int16))
        source_names.append(phase)
    return (
        np.concatenate(starts), np.concatenate(targets),
        np.concatenate(endpoints), np.concatenate(task_id),
        np.concatenate(source_id), tuple(source_names),
    )


def _ensure_phase_manifests(root: Path, profile: str) -> list[Path]:
    bootstrap_tasks, aligned_tasks, commands = dimensions(profile)
    paths = output_paths(root)
    directories = []
    for phase, tasks in (
        ("bootstrap", bootstrap_tasks),
        ("acquisition", aligned_tasks),
        ("return", aligned_tasks),
    ):
        directory = paths[phase]
        parts = []
        for task in range(tasks):
            path = _part_path(directory, task)
            _validate_part(path, task, commands)
            parts.append({
                "file": path.name,
                "commands": int(commands),
                "sha256": sha256(path),
            })
        atomic_json(directory / "manifest.json", {
            "schema_version": 1,
            "method_revision": METHOD_REVISION,
            "phase": phase,
            "tasks": int(tasks),
            "commands_per_task": int(commands),
            "commands": int(tasks * commands),
            "parts": parts,
            "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
            "task_reward_used": False,
            "hidden_goals_used": False,
        })
        directories.append(directory.resolve())
    return directories


def _grouped_split(task_id, source_id, source_count: int, seed: int):
    rng = np.random.default_rng(int(seed))
    held = []
    for source in range(source_count):
        groups = rng.permutation(np.unique(task_id[source_id == source]))
        count = max(1, int(round(VALIDATION_FRACTION * len(groups))))
        held.extend(groups[:count].tolist())
    validation = np.isin(task_id, held)
    return np.flatnonzero(~validation), np.flatnonzero(validation), held


def command_train(args) -> dict:
    paths = output_paths(args.output)
    states, commands, endpoints, task_id, source_id, source_names = _load_rows(
        args.output, args.profile
    )
    seeds = seed_values(args.preparation_seed)
    train_index, validation_index, held = _grouped_split(
        task_id, source_id, len(source_names), seeds["split_seed"]
    )
    inputs = consequence_inputs(states, commands)
    spatial_target = endpoints[:, :2] - commands
    posture_input = np.concatenate((inputs, endpoints[:, :2]), axis=1)
    posture_target = endpoints[:, 2:]
    epochs = EPOCHS if args.profile == "full" else 2
    started = time.time()
    spatial = _train_head(
        inputs, spatial_target, train_index, validation_index,
        seeds["training_seed"] + 1, args.device, epochs, BATCH_SIZE,
        LEARNING_RATE, f"Ant R16 seed{args.preparation_seed} spatial",
    )
    posture = _train_head(
        posture_input, posture_target, train_index, validation_index,
        seeds["training_seed"] + 2, args.device, epochs, BATCH_SIZE,
        LEARNING_RATE, f"Ant R16 seed{args.preparation_seed} posture",
    )
    payload = {
        "artifact_type": "factorized_ant_consequence",
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL if args.profile == "full" else "smoke_only",
        "method_reference": str(SPECIFICATION),
        "preparation_seed": int(args.preparation_seed),
        "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
        "spatial_parameterization": "endpoint_xy_minus_command",
        "posture_parameterization": "absolute_endpoint_posture",
        "factorization": (
            "P(endpoint_xy-command|state,command) times "
            "P(endpoint_posture|state,command,endpoint_xy)"
        ),
        "training_source": list(source_names),
        "training_commands": int(len(train_index)),
        "validation_commands": int(len(validation_index)),
        "split_seed": seeds["split_seed"],
        "training_seed": seeds["training_seed"],
        "split": PROTOCOL_VALUE["training_schedule"]["split"],
        "held_out_task_groups": [int(value) for value in held],
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
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
    }
    r52.atomic_torch(paths["consequence"], payload)
    model = FactorizedConsequenceEnsemble(paths["consequence"], args.device)
    sigma, sigma_rows = r30.landing_precision(
        model, states[validation_index], commands[validation_index]
    )
    predicted = r52._spatial_expected_endpoint(
        model, states[validation_index], commands[validation_index]
    )
    error = np.linalg.norm(predicted - endpoints[validation_index, :2], axis=1)
    report = {
        "schema_version": 1,
        "method_reference": str(SPECIFICATION),
        "method_reference_sha256": sha256(SPECIFICATION),
        "method_revision": METHOD_REVISION,
        "claim_level": payload["claim_level"],
        "declared_deviations": [],
        "oracle_inputs": [],
        "profile": args.profile,
        "preparation_seed": int(args.preparation_seed),
        "checkpoint": str(paths["consequence"]),
        "checkpoint_sha256": sha256(paths["consequence"]),
        "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
        "rows": int(len(states)),
        "training_commands": int(len(train_index)),
        "validation_commands": int(len(validation_index)),
        "source_names": list(source_names),
        "landing_sigma_m": float(sigma),
        "landing_sigma_estimator": (
            "held_out_E549_mean_ensemble_and_mixture_weighted_"
            "within_mode_radial_scale"
        ),
        "endpoint_error_mean_m": float(error.mean()),
        "endpoint_error_median_m": float(np.median(error)),
        "endpoint_error_p90_m": float(np.quantile(error, 0.9)),
        "landing_scale_p90_m": float(np.quantile(sigma_rows, 0.9)),
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "wall_clock_seconds": time.time() - started,
    }
    atomic_json(paths["consequence_summary"], report)
    print(json.dumps(report, indent=2))
    return report


def configure_runtime(output: Path) -> dict:
    paths = output_paths(output)
    summary = json.loads(paths["consequence_summary"].read_text())
    if summary.get("checkpoint_sha256") != sha256(paths["consequence"]):
        raise ValueError("rebuilt AntMaze consequence checkpoint changed")
    r48.REFERENCE = SPECIFICATION
    r48.METHOD_REVISION = METHOD_REVISION
    r48.PREPARATION_CLAIM_LEVEL = CLAIM_LEVEL
    r48.R30_CHECKPOINT = paths["consequence"]
    r48.R30_SUMMARY = paths["consequence_summary"]
    r48.SIGMA = float(summary["landing_sigma_m"])
    r48.configure_result_directory(paths["runtime"])
    r48._bind_adapter()
    return summary


def command_prepare_graph(args) -> dict:
    summary = configure_runtime(args.output)
    paths = output_paths(args.output)
    states, commands, endpoints, _task, _source, source_names = _load_rows(
        args.output, args.profile
    )
    reward_free_rounds = _ensure_phase_manifests(args.output, args.profile)
    model = FactorizedConsequenceEnsemble(paths["consequence"], args.device)
    state_scale = model.input_std[:STATE_DIM].copy()
    state_scale[:2] = float(r48.SIGMA)
    nodes, inverse = np.unique(
        np.concatenate((states, endpoints), axis=0), axis=0, return_inverse=True
    )
    count = len(states)
    source_index, endpoint_index = inverse[:count], inverse[count:]
    reliability, empirical, support = recorded_transition_reliability(
        states, commands, endpoints, state_scale, r48.SIGMA
    )
    out_degree = np.bincount(source_index, minlength=len(nodes))
    graph = r48.ManifestRecordedTransitionGraph(
        nodes, source_index, endpoint_index, commands, reliability, state_scale,
        metadata={
            "method_revision": METHOD_REVISION,
            **r48.manifest_contract(),
            "claim_level": summary["claim_level"],
            "node_source": "all_exact_reward_free_command_boundary_states",
            "edge_source": "all_exact_reward_free_recorded_transitions",
            "reward_free_rounds": [str(path) for path in reward_free_rounds],
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
            "edge_count": int(count),
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
        "collection_phases": list(source_names),
        "reliability_mean": float(reliability.mean()),
        "landing_sigma_m": float(r48.SIGMA),
        "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
    }
    print(json.dumps(report, indent=2))
    return report


def _prepare_args(args):
    configure_runtime(args.output)
    args.graph = output_paths(args.output)["graph"]
    args.seed = seed_values(args.preparation_seed)["training_seed"]
    if args.profile == "smoke":
        graph = r48.verify_graph(args.graph)
        positive = int(np.count_nonzero(r48.movement_reach_weight(graph) > 0.0))
        # The 18-row smoke dataset cannot instantiate the declared 128
        # representatives.  Full preparation never enters this branch.
        r48.CANDIDATE_BUDGET = max(
            1, min(int(BRIDGE_CANDIDATES), positive)
        )


def command_prepare_mobility(args):
    _prepare_args(args)
    return r48.command_prepare_mobility(args)


def command_prepare_bridges(args):
    _prepare_args(args)
    return r48.command_prepare_bridges(args)


def command_prepare_passage(args):
    _prepare_args(args)
    return r48.command_prepare_passage(args)


def command_prepare_all(args):
    command_prepare_graph(args)
    command_prepare_mobility(args)
    command_prepare_bridges(args)
    command_prepare_passage(args)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--preparation-seed", type=int, choices=(0, 1, 2), required=True)
    value.add_argument("--profile", choices=("smoke", "full"), default="full")
    value.add_argument("--device", default="cuda")
    value.add_argument("--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377"))
    value.add_argument("--output", type=Path)
    value.add_argument("--graph", type=Path)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").set_defaults(func=command_audit)
    commands.add_parser("collect-bootstrap").set_defaults(func=command_collect_bootstrap)
    commands.add_parser("collect-alignment").set_defaults(func=command_collect_alignment)
    commands.add_parser("train-consequence").set_defaults(func=command_train)
    commands.add_parser("prepare-graph").set_defaults(func=command_prepare_graph)
    commands.add_parser("prepare-mobility").set_defaults(func=command_prepare_mobility)
    commands.add_parser("prepare-bridges").set_defaults(func=command_prepare_bridges)
    commands.add_parser("prepare-passage").set_defaults(func=command_prepare_passage)
    commands.add_parser("prepare-all").set_defaults(func=command_prepare_all)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.output is None:
        args.output = result_root("antmaze", args.preparation_seed, args.profile)
    args.output = args.output.resolve()
    args.func(args)


if __name__ == "__main__":
    main()
