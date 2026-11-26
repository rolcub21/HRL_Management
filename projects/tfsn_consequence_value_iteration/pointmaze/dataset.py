"""Load the initial reward-free PointMaze command data without task labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .core import load_json, protocol, resolve_repository, sha256
from projects.tfsn_consequence_value_iteration.shared.settings import COMMAND_STEPS


@dataclass(frozen=True)
class CommandDataset:
    starts: np.ndarray
    commands: np.ndarray
    endpoints: np.ndarray
    primitives: np.ndarray
    trajectory_id: np.ndarray
    source_id: np.ndarray
    source_names: tuple
    source_paths: tuple

    @property
    def rows(self):
        return int(len(self.starts))


def _shard_paths(root, schema):
    pattern = "chain_*.npz" if schema == "bootstrap_chain" else "rollout_*.npz"
    return sorted((root / "shards").glob(pattern))


def _validate_shard(path, schema, expected_rows, tau):
    with np.load(path, allow_pickle=False) as stored:
        shard = {name: stored[name] for name in stored.files}
    required = {
        "start_observation": (expected_rows, 4),
        "command_xy": (expected_rows, 2),
        "endpoint_observation": (expected_rows, 4),
        "primitive_observation": (expected_rows, tau, 4),
        "duration": (expected_rows,),
    }
    identifier = "chain_id" if schema == "bootstrap_chain" else "rollout_id"
    required[identifier] = (expected_rows,)
    for name, shape in required.items():
        value = shard.get(name)
        if value is None or value.shape != shape:
            actual = None if value is None else value.shape
            raise ValueError(f"invalid {path}: {name} {actual}, expected {shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"non-finite {name} in {path}")
    if not np.all(shard["duration"] == int(tau)):
        raise ValueError(f"fixed command duration changed in {path}")
    return shard, identifier


def load_initial_dataset(max_groups_per_source=None):
    contract = protocol()
    tau = COMMAND_STEPS
    starts, commands, endpoints, primitives = [], [], [], []
    trajectory_ids, source_ids = [], []
    source_names, source_paths = [], []
    expected_total = 0
    for source_id, source in enumerate(
        contract["initial_reward_free_sources_seed0"]
    ):
        root = resolve_repository(source["path"])
        summary_path = root / "summary.json"
        if sha256(summary_path) != source["summary_sha256"]:
            raise ValueError(f"source summary changed: {source['name']}")
        summary = load_json(summary_path)
        if summary.get("task_reward_used") is not False:
            raise ValueError(f"source is not reward-free: {source['name']}")
        if summary.get("hidden_goals_used") is not False:
            raise ValueError(f"source exposed a hidden goal: {source['name']}")
        schema = source["schema"]
        paths = _shard_paths(root, schema)
        if max_groups_per_source is not None:
            paths = paths[: int(max_groups_per_source)]
        if not paths:
            raise ValueError(f"source has no command shards: {source['name']}")
        rows_per_group = int(summary.get("options_per_chain", summary.get("options_per_rollout")))
        for local_group, path in enumerate(paths):
            shard, identifier = _validate_shard(
                path, schema, rows_per_group, tau
            )
            stored_group = np.asarray(shard[identifier], np.int64)
            if not np.all(stored_group == local_group):
                raise ValueError(f"noncanonical group id in {path}")
            starts.append(shard["start_observation"].astype(np.float32))
            commands.append(shard["command_xy"].astype(np.float32))
            endpoints.append(shard["endpoint_observation"].astype(np.float32))
            primitives.append(shard["primitive_observation"].astype(np.float32))
            trajectory_ids.append(
                np.full(rows_per_group, source_id * 100_000 + local_group, np.int64)
            )
            source_ids.append(np.full(rows_per_group, source_id, np.int16))
        expected_total += len(paths) * rows_per_group
        source_names.append(str(source["name"]))
        source_paths.append(root)
    dataset = CommandDataset(
        starts=np.concatenate(starts),
        commands=np.concatenate(commands),
        endpoints=np.concatenate(endpoints),
        primitives=np.concatenate(primitives),
        trajectory_id=np.concatenate(trajectory_ids),
        source_id=np.concatenate(source_ids),
        source_names=tuple(source_names),
        source_paths=tuple(source_paths),
    )
    if dataset.rows != expected_total:
        raise ValueError("loaded PointMaze row count changed")
    return dataset


def _validate_current_alignment_shard(path, tau):
    with np.load(path, allow_pickle=False) as stored:
        shard = {name: stored[name] for name in stored.files}
    rows = len(shard.get("start_observation", ()))
    required = {
        "start_observation": (rows, 4),
        "command_xy": (rows, 2),
        "endpoint_observation": (rows, 4),
        "primitive_observation": (rows, tau, 4),
        "primitive_reward": (rows, tau),
        "task_id": (),
        "physical_episode": (),
    }
    if rows <= 0:
        raise ValueError(f"empty current alignment shard: {path}")
    for name, shape in required.items():
        value = shard.get(name)
        if value is None or value.shape != shape:
            actual = None if value is None else value.shape
            raise ValueError(
                f"invalid current alignment shard {path}: "
                f"{name} {actual}, expected {shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"non-finite {name} in {path}")
    return shard


def load_current_alignment_dataset(root, max_groups_per_source=None):
    """Load the current planner's reward-free acquisition and return round."""
    root = Path(root).resolve()
    contract = protocol()
    tau = COMMAND_STEPS
    starts, commands, endpoints, primitives = [], [], [], []
    trajectory_ids, source_ids = [], []
    source_names, source_paths = [], []
    expected_total = 0
    for source_id, phase in enumerate(("acquisition", "return")):
        summary_path = root / f"summary_{phase}_full.json"
        summary = load_json(summary_path)
        expected = {
            "method_revision": "pointmaze_current_planner_alignment_r2",
            "profile": "full",
            "phases": [phase],
            "hidden_goals_accessed": False,
            "task_reward_used": False,
            "environment_reward_consumed": False,
            "confirmation_panel_accessed": False,
        }
        for name, value in expected.items():
            if summary.get(name) != value:
                raise ValueError(
                    f"current {phase} alignment has another {name}"
                )
        paths = sorted((root / "shards" / phase).glob("task_*_episode_*.npz"))
        if max_groups_per_source is not None:
            paths = paths[: int(max_groups_per_source)]
        if not paths:
            raise ValueError(f"current {phase} alignment has no shards")
        rows_loaded = 0
        for path in paths:
            shard = _validate_current_alignment_shard(path, tau)
            task_id = int(shard["task_id"])
            episode = int(shard["physical_episode"])
            group = source_id * 100_000 + task_id * 100 + episode
            rows = len(shard["start_observation"])
            starts.append(shard["start_observation"].astype(np.float32))
            commands.append(shard["command_xy"].astype(np.float32))
            endpoints.append(shard["endpoint_observation"].astype(np.float32))
            primitives.append(shard["primitive_observation"].astype(np.float32))
            trajectory_ids.append(np.full(rows, group, np.int64))
            source_ids.append(np.full(rows, source_id, np.int16))
            rows_loaded += rows
        if max_groups_per_source is None:
            declared = int(summary["phase_summary"][phase]["rows"])
            if rows_loaded != declared:
                raise ValueError(
                    f"current {phase} alignment rows changed: "
                    f"{rows_loaded} != {declared}"
                )
        expected_total += rows_loaded
        source_names.append(f"current_{phase}_alignment_r2")
        source_paths.append(root / "shards" / phase)
    dataset = CommandDataset(
        starts=np.concatenate(starts),
        commands=np.concatenate(commands),
        endpoints=np.concatenate(endpoints),
        primitives=np.concatenate(primitives),
        trajectory_id=np.concatenate(trajectory_ids),
        source_id=np.concatenate(source_ids),
        source_names=tuple(source_names),
        source_paths=tuple(source_paths),
    )
    if dataset.rows != expected_total:
        raise ValueError("loaded current PointMaze alignment row count changed")
    return dataset


def concatenate_datasets(*datasets):
    if not datasets:
        raise ValueError("at least one command dataset is required")
    starts, commands, endpoints, primitives = [], [], [], []
    trajectory_ids, source_ids = [], []
    source_names, source_paths = [], []
    source_offset = 0
    for dataset in datasets:
        starts.append(dataset.starts)
        commands.append(dataset.commands)
        endpoints.append(dataset.endpoints)
        primitives.append(dataset.primitives)
        source_ids.append(dataset.source_id.astype(np.int64) + source_offset)
        # Source-local trajectory identifiers occupy disjoint 100,000 blocks.
        trajectory_ids.append(
            dataset.trajectory_id.astype(np.int64) + source_offset * 100_000
        )
        source_names.extend(dataset.source_names)
        source_paths.extend(dataset.source_paths)
        source_offset += len(dataset.source_names)
    return CommandDataset(
        starts=np.concatenate(starts),
        commands=np.concatenate(commands),
        endpoints=np.concatenate(endpoints),
        primitives=np.concatenate(primitives),
        trajectory_id=np.concatenate(trajectory_ids),
        source_id=np.concatenate(source_ids).astype(np.int16),
        source_names=tuple(source_names),
        source_paths=tuple(source_paths),
    )


def load_current_aligned_dataset(alignment_root, max_groups_per_source=None):
    initial = load_initial_dataset(
        max_groups_per_source=max_groups_per_source
    )
    current = load_current_alignment_dataset(
        alignment_root, max_groups_per_source=max_groups_per_source
    )
    return concatenate_datasets(initial, current)


def grouped_split(dataset, seed=731, validation_fraction=0.10):
    rng = np.random.default_rng(int(seed))
    validation_groups = []
    for source_id in range(len(dataset.source_names)):
        groups = np.unique(dataset.trajectory_id[dataset.source_id == source_id])
        groups = rng.permutation(groups)
        count = max(1, int(round(float(validation_fraction) * len(groups))))
        validation_groups.extend(groups[:count].tolist())
    validation = np.isin(dataset.trajectory_id, validation_groups)
    training = ~validation
    if not validation.any() or not training.any():
        raise ValueError("grouped split produced an empty partition")
    return np.flatnonzero(training), np.flatnonzero(validation)

