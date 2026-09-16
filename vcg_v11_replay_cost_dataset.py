"""Fail-closed offline cost labels from completed VCG 1.1 replay episodes.

VCG 1.1's scalar replay does not store a separate handling-cost field.  Its
chosen action record does, however, identify a completed reconfiguration.  For
the existing proper-training checkpoints, that indicator is an exact alias of
the executor's recorded physical storage relocation count.  This module
accepts the alias only after checking it episode-by-episode against the
authoritative training history.

Splitting is performed over complete episodes, never individual transitions,
so future-cost labels from one trajectory cannot leak across train and
validation sets.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence

from viability_graph_hierarchy import (
    ViabilityGraphCandidateRecord,
    ViabilityGraphTransition,
)


DATASET_SCHEMA_VERSION = 1
COST_DEFINITION = (
    "remaining_completed_reconfigure_macros_validated_against_"
    "episode_physical_storage_relocations_v1"
)
SPLIT_PROTOCOL = "sha256_ranked_complete_episode_split_v1"
_KNOWN_ACTION_TYPES = frozenset(("accept", "deliver", "reconfigure", "defer"))


class ReplayCostDatasetError(ValueError):
    """Raised when scalar replay cannot safely serve as a cost dataset."""


def _strict_nonnegative_int(value, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReplayCostDatasetError(f"{name} must be a nonnegative integer")
    return int(value)


def _strict_positive_int(value, *, name: str) -> int:
    result = _strict_nonnegative_int(value, name=name)
    if result == 0:
        raise ReplayCostDatasetError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class ReplayCostSample:
    """One chosen V1.1 macro with its realized remaining-episode cost."""

    model_seed: int
    episode_index: int
    instance_seed: int
    macro_index: int
    record: ViabilityGraphCandidateRecord
    immediate_physical_rehandles: int
    remaining_physical_rehandles: float

    def __post_init__(self) -> None:
        _strict_nonnegative_int(self.model_seed, name="model_seed")
        _strict_nonnegative_int(self.episode_index, name="episode_index")
        _strict_nonnegative_int(self.instance_seed, name="instance_seed")
        _strict_nonnegative_int(self.macro_index, name="macro_index")
        immediate = _strict_nonnegative_int(
            self.immediate_physical_rehandles,
            name="immediate_physical_rehandles",
        )
        if immediate not in (0, 1):
            raise ReplayCostDatasetError("one V1.1 macro can cost only zero or one")
        remaining = float(self.remaining_physical_rehandles)
        if not math.isfinite(remaining) or remaining < immediate:
            raise ReplayCostDatasetError(
                "remaining cost must be finite and include the immediate cost"
            )
        if not remaining.is_integer():
            raise ReplayCostDatasetError("remaining cost must be an event count")


@dataclass(frozen=True)
class ReplayCostEpisode:
    episode_index: int
    instance_seed: int
    samples: tuple[ReplayCostSample, ...]
    physical_rehandles: int

    def __post_init__(self) -> None:
        _strict_nonnegative_int(self.episode_index, name="episode_index")
        _strict_nonnegative_int(self.instance_seed, name="instance_seed")
        _strict_nonnegative_int(
            self.physical_rehandles, name="physical_rehandles"
        )
        if not self.samples:
            raise ReplayCostDatasetError("a cost episode cannot be empty")
        if any(sample.episode_index != self.episode_index for sample in self.samples):
            raise ReplayCostDatasetError("sample/episode indices do not align")
        if any(sample.instance_seed != self.instance_seed for sample in self.samples):
            raise ReplayCostDatasetError("sample/instance seeds do not align")
        if tuple(sample.macro_index for sample in self.samples) != tuple(
            range(len(self.samples))
        ):
            raise ReplayCostDatasetError("macro indices must be contiguous from zero")
        total = sum(sample.immediate_physical_rehandles for sample in self.samples)
        if total != self.physical_rehandles:
            raise ReplayCostDatasetError("episode cost does not equal immediate costs")
        for index, sample in enumerate(self.samples):
            continuation = (
                0.0
                if index + 1 == len(self.samples)
                else self.samples[index + 1].remaining_physical_rehandles
            )
            expected = sample.immediate_physical_rehandles + continuation
            if sample.remaining_physical_rehandles != expected:
                raise ReplayCostDatasetError("Monte Carlo cost recurrence is broken")


@dataclass(frozen=True)
class ReplayCostDataset:
    model_seed: int
    selected_episode_limit: int
    split_seed: int
    validation_fraction: float
    train_episodes: tuple[ReplayCostEpisode, ...]
    validation_episodes: tuple[ReplayCostEpisode, ...]
    source_completed_episodes: int
    source_transition_count: int

    @property
    def train_samples(self) -> tuple[ReplayCostSample, ...]:
        return tuple(sample for episode in self.train_episodes for sample in episode.samples)

    @property
    def validation_samples(self) -> tuple[ReplayCostSample, ...]:
        return tuple(
            sample for episode in self.validation_episodes for sample in episode.samples
        )

    def audit_dict(self) -> dict:
        train_indices = tuple(episode.episode_index for episode in self.train_episodes)
        validation_indices = tuple(
            episode.episode_index for episode in self.validation_episodes
        )
        return {
            "schema_version": DATASET_SCHEMA_VERSION,
            "cost_definition": COST_DEFINITION,
            "split_protocol": SPLIT_PROTOCOL,
            "model_seed": int(self.model_seed),
            "selected_episode_limit": int(self.selected_episode_limit),
            "source_completed_episodes": int(self.source_completed_episodes),
            "source_transition_count": int(self.source_transition_count),
            "used_episode_count": len(train_indices) + len(validation_indices),
            "used_transition_count": len(self.train_samples) + len(self.validation_samples),
            "train_episode_count": len(train_indices),
            "validation_episode_count": len(validation_indices),
            "train_sample_count": len(self.train_samples),
            "validation_sample_count": len(self.validation_samples),
            "train_episode_indices": train_indices,
            "validation_episode_indices": validation_indices,
            "split_seed": int(self.split_seed),
            "validation_fraction": float(self.validation_fraction),
            "total_physical_rehandles": sum(
                episode.physical_rehandles
                for episode in self.train_episodes + self.validation_episodes
            ),
            "episode_disjoint": not bool(set(train_indices) & set(validation_indices)),
            "episode_exhaustive": sorted(train_indices + validation_indices)
            == list(range(self.selected_episode_limit)),
            "reconfigure_alias_validated_against_training_history": True,
            "complete_done_bounded_episodes_only": True,
        }


def segment_complete_episodes(
    transitions: Sequence[ViabilityGraphTransition],
) -> tuple[tuple[ViabilityGraphTransition, ...], ...]:
    """Partition a non-truncated replay in order, rejecting partial tails."""

    transitions = tuple(transitions)
    if not transitions:
        raise ReplayCostDatasetError("source replay is empty")
    episodes = []
    pending = []
    for index, transition in enumerate(transitions):
        if not isinstance(transition, ViabilityGraphTransition):
            raise ReplayCostDatasetError(
                f"replay item {index} is not a ViabilityGraphTransition"
            )
        pending.append(transition)
        if transition.done:
            episodes.append(tuple(pending))
            pending = []
    if pending:
        raise ReplayCostDatasetError(
            "source replay ends inside an episode; Monte Carlo labels are invalid"
        )
    if not episodes:
        raise ReplayCostDatasetError("source replay contains no terminal boundary")
    return tuple(episodes)


def _episode_samples(
    transitions: Sequence[ViabilityGraphTransition],
    *,
    model_seed: int,
    episode_index: int,
    instance_seed: int,
) -> tuple[ReplayCostSample, ...]:
    costs = []
    for macro_index, transition in enumerate(transitions):
        action_type = transition.chosen.action_type
        if action_type not in _KNOWN_ACTION_TYPES:
            raise ReplayCostDatasetError(
                f"episode {episode_index} macro {macro_index} has unknown action type"
            )
        costs.append(int(action_type == "reconfigure"))
    remaining = 0
    reversed_samples = []
    for macro_index in range(len(transitions) - 1, -1, -1):
        remaining += costs[macro_index]
        reversed_samples.append(
            ReplayCostSample(
                model_seed=int(model_seed),
                episode_index=int(episode_index),
                instance_seed=int(instance_seed),
                macro_index=int(macro_index),
                record=transitions[macro_index].chosen,
                immediate_physical_rehandles=costs[macro_index],
                remaining_physical_rehandles=float(remaining),
            )
        )
    return tuple(reversed(reversed_samples))


def build_replay_cost_episodes(
    transitions: Sequence[ViabilityGraphTransition],
    training_history: Sequence[Mapping],
    *,
    model_seed: int,
    selected_episode_limit: int,
) -> tuple[ReplayCostEpisode, ...]:
    """Create labels and authenticate the inferred cost episode by episode."""

    segmented = segment_complete_episodes(transitions)
    limit = _strict_positive_int(
        selected_episode_limit, name="selected_episode_limit"
    )
    history = tuple(training_history)
    if limit > len(segmented) or limit > len(history):
        raise ReplayCostDatasetError(
            "selected episode limit exceeds replay or training history"
        )
    episodes = []
    for episode_index, (episode, row) in enumerate(zip(segmented[:limit], history[:limit])):
        if not isinstance(row, Mapping):
            raise ReplayCostDatasetError("training-history row is not a mapping")
        instance_seed = _strict_nonnegative_int(
            row.get("instance_seed"), name=f"history[{episode_index}].instance_seed"
        )
        expected_macros = _strict_positive_int(
            row.get("macro_decisions"),
            name=f"history[{episode_index}].macro_decisions",
        )
        if expected_macros != len(episode):
            raise ReplayCostDatasetError(
                f"episode {episode_index} macro count disagrees with training history"
            )
        audit = row.get("audit")
        if not isinstance(audit, Mapping):
            raise ReplayCostDatasetError(
                f"history[{episode_index}] has no authoritative audit"
            )
        expected_rehandles = _strict_nonnegative_int(
            audit.get("relocations"),
            name=f"history[{episode_index}].audit.relocations",
        )
        samples = _episode_samples(
            episode,
            model_seed=int(model_seed),
            episode_index=episode_index,
            instance_seed=instance_seed,
        )
        inferred_rehandles = sum(
            sample.immediate_physical_rehandles for sample in samples
        )
        if inferred_rehandles != expected_rehandles:
            raise ReplayCostDatasetError(
                f"episode {episode_index} reconfigure count {inferred_rehandles} "
                f"does not equal recorded physical relocations {expected_rehandles}"
            )
        episodes.append(
            ReplayCostEpisode(
                episode_index=episode_index,
                instance_seed=instance_seed,
                samples=samples,
                physical_rehandles=expected_rehandles,
            )
        )
    return tuple(episodes)


def split_complete_episodes(
    episodes: Sequence[ReplayCostEpisode],
    *,
    split_seed: int,
    validation_fraction: float = 0.2,
) -> tuple[tuple[ReplayCostEpisode, ...], tuple[ReplayCostEpisode, ...]]:
    """Return a reproducible exhaustive train/validation episode partition."""

    episodes = tuple(episodes)
    if len(episodes) < 2:
        raise ReplayCostDatasetError("episode split requires at least two episodes")
    indices = tuple(episode.episode_index for episode in episodes)
    if len(set(indices)) != len(indices):
        raise ReplayCostDatasetError("episode indices must be unique")
    fraction = float(validation_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ReplayCostDatasetError("validation_fraction must lie strictly in (0, 1)")
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise ReplayCostDatasetError("split_seed must be an integer")
    validation_count = min(
        len(episodes) - 1,
        max(1, int(math.floor(len(episodes) * fraction + 0.5))),
    )

    def rank(episode: ReplayCostEpisode) -> tuple[str, int]:
        identity = (
            f"{SPLIT_PROTOCOL}:{split_seed}:{episode.episode_index}:"
            f"{episode.instance_seed}"
        ).encode("ascii")
        return hashlib.sha256(identity).hexdigest(), episode.episode_index

    validation_indices = {
        episode.episode_index
        for episode in sorted(episodes, key=rank)[:validation_count]
    }
    train = tuple(
        episode for episode in episodes if episode.episode_index not in validation_indices
    )
    validation = tuple(
        episode for episode in episodes if episode.episode_index in validation_indices
    )
    if not train or not validation:
        raise ReplayCostDatasetError("episode split produced an empty partition")
    return train, validation


def build_replay_cost_dataset(
    checkpoint: Mapping,
    *,
    split_seed: int,
    validation_fraction: float = 0.2,
    selected_episode_limit: int | None = None,
) -> ReplayCostDataset:
    """Build a deterministic, episode-disjoint dataset from ``latest.pth``."""

    if not isinstance(checkpoint, Mapping):
        raise ReplayCostDatasetError("checkpoint must be a mapping")
    model_seed = _strict_nonnegative_int(
        checkpoint.get("model_seed"), name="checkpoint.model_seed"
    )
    completed = _strict_positive_int(
        checkpoint.get("completed_training_episodes"),
        name="checkpoint.completed_training_episodes",
    )
    history = checkpoint.get("training_history")
    if not isinstance(history, (list, tuple)) or len(history) != completed:
        raise ReplayCostDatasetError(
            "training history must contain every completed source episode"
        )
    state = checkpoint.get("agent_state")
    replay = state.get("replay") if isinstance(state, Mapping) else None
    if not isinstance(replay, Mapping):
        raise ReplayCostDatasetError("checkpoint has no resumable scalar replay")
    capacity = _strict_positive_int(replay.get("capacity"), name="replay.capacity")
    memory = replay.get("memory")
    if not isinstance(memory, (list, tuple)):
        raise ReplayCostDatasetError("replay.memory must be an ordered sequence")
    if len(memory) >= capacity:
        raise ReplayCostDatasetError(
            "full replay capacity cannot prove that the first episode is complete"
        )
    segmented = segment_complete_episodes(memory)
    if len(segmented) != completed:
        raise ReplayCostDatasetError(
            "terminal boundary count does not equal completed training episodes"
        )
    if sum(len(episode) for episode in segmented) != len(memory):
        raise ReplayCostDatasetError("episode segmentation is not exhaustive")

    limit_value = (
        checkpoint.get("selected_checkpoint_episode")
        if selected_episode_limit is None
        else selected_episode_limit
    )
    limit = _strict_positive_int(limit_value, name="selected_episode_limit")
    episodes = build_replay_cost_episodes(
        memory,
        history,
        model_seed=model_seed,
        selected_episode_limit=limit,
    )
    train, validation = split_complete_episodes(
        episodes,
        split_seed=split_seed,
        validation_fraction=validation_fraction,
    )
    dataset = ReplayCostDataset(
        model_seed=model_seed,
        selected_episode_limit=limit,
        split_seed=int(split_seed),
        validation_fraction=float(validation_fraction),
        train_episodes=train,
        validation_episodes=validation,
        source_completed_episodes=completed,
        source_transition_count=len(memory),
    )
    audit = dataset.audit_dict()
    if not audit["episode_disjoint"] or not audit["episode_exhaustive"]:
        raise ReplayCostDatasetError("dataset split invariants failed")
    return dataset


__all__ = (
    "COST_DEFINITION",
    "DATASET_SCHEMA_VERSION",
    "ReplayCostDataset",
    "ReplayCostDatasetError",
    "ReplayCostEpisode",
    "ReplayCostSample",
    "SPLIT_PROTOCOL",
    "build_replay_cost_dataset",
    "build_replay_cost_episodes",
    "segment_complete_episodes",
    "split_complete_episodes",
)
