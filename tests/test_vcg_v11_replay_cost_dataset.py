from copy import deepcopy
from types import SimpleNamespace

import pytest

from viability_graph_hierarchy import ViabilityGraphTransition
from vcg_v11_replay_cost_dataset import (
    ReplayCostDatasetError,
    build_replay_cost_dataset,
    segment_complete_episodes,
)


def _transition(action_type, *, done):
    chosen = SimpleNamespace(action_type=action_type, key=action_type)
    next_candidates = () if done else (SimpleNamespace(key="next"),)
    return ViabilityGraphTransition(
        chosen=chosen,
        reward=0.0,
        duration=1,
        next_candidates=next_candidates,
        done=done,
    )


def _checkpoint():
    action_episodes = (
        ("accept", "reconfigure", "deliver"),
        ("reconfigure", "reconfigure", "deliver"),
        ("accept", "deliver"),
        ("defer", "reconfigure", "deliver"),
    )
    memory = []
    history = []
    for episode_index, actions in enumerate(action_episodes):
        for macro_index, action in enumerate(actions):
            memory.append(
                _transition(action, done=macro_index + 1 == len(actions))
            )
        history.append(
            {
                "instance_seed": 50_000_000 + episode_index,
                "macro_decisions": len(actions),
                "audit": {
                    "relocations": sum(action == "reconfigure" for action in actions)
                },
            }
        )
    return {
        "model_seed": 0,
        "completed_training_episodes": len(action_episodes),
        "selected_checkpoint_episode": len(action_episodes),
        "training_history": history,
        "agent_state": {
            "replay": {"capacity": 100, "memory": memory},
        },
    }


def test_monte_carlo_labels_include_current_macro_and_recur_exactly():
    dataset = build_replay_cost_dataset(_checkpoint(), split_seed=31)
    episodes = sorted(
        dataset.train_episodes + dataset.validation_episodes,
        key=lambda episode: episode.episode_index,
    )
    assert [sample.remaining_physical_rehandles for sample in episodes[0].samples] == [
        1.0,
        1.0,
        0.0,
    ]
    assert [sample.remaining_physical_rehandles for sample in episodes[1].samples] == [
        2.0,
        1.0,
        0.0,
    ]
    assert [sample.remaining_physical_rehandles for sample in episodes[2].samples] == [
        0.0,
        0.0,
    ]


def test_split_is_deterministic_disjoint_exhaustive_and_episode_level():
    first = build_replay_cost_dataset(_checkpoint(), split_seed=31)
    second = build_replay_cost_dataset(_checkpoint(), split_seed=31)
    assert first.audit_dict() == second.audit_dict()
    audit = first.audit_dict()
    assert audit["episode_disjoint"] is True
    assert audit["episode_exhaustive"] is True
    assert audit["train_episode_count"] == 3
    assert audit["validation_episode_count"] == 1
    train_indices = {sample.episode_index for sample in first.train_samples}
    validation_indices = {
        sample.episode_index for sample in first.validation_samples
    }
    assert not train_indices & validation_indices
    assert train_indices | validation_indices == {0, 1, 2, 3}


def test_selected_checkpoint_limit_excludes_later_replay_episodes():
    dataset = build_replay_cost_dataset(
        _checkpoint(), split_seed=31, selected_episode_limit=3
    )
    assert dataset.selected_episode_limit == 3
    assert dataset.source_completed_episodes == 4
    assert {
        sample.episode_index
        for sample in dataset.train_samples + dataset.validation_samples
    } == {0, 1, 2}


def test_partial_final_episode_is_rejected():
    checkpoint = _checkpoint()
    checkpoint["agent_state"]["replay"]["memory"].append(
        _transition("accept", done=False)
    )
    with pytest.raises(ReplayCostDatasetError, match="ends inside an episode"):
        segment_complete_episodes(checkpoint["agent_state"]["replay"]["memory"])


def test_reconfigure_alias_must_match_authoritative_episode_metric():
    checkpoint = _checkpoint()
    checkpoint["training_history"][1]["audit"]["relocations"] = 1
    with pytest.raises(ReplayCostDatasetError, match="does not equal recorded"):
        build_replay_cost_dataset(checkpoint, split_seed=31)


def test_full_capacity_replay_is_rejected_as_potentially_left_truncated():
    checkpoint = _checkpoint()
    checkpoint["agent_state"]["replay"]["capacity"] = len(
        checkpoint["agent_state"]["replay"]["memory"]
    )
    with pytest.raises(ReplayCostDatasetError, match="cannot prove"):
        build_replay_cost_dataset(checkpoint, split_seed=31)


def test_history_macro_count_and_done_count_are_authenticated():
    macro_mismatch = _checkpoint()
    macro_mismatch["training_history"][0]["macro_decisions"] += 1
    with pytest.raises(ReplayCostDatasetError, match="macro count"):
        build_replay_cost_dataset(macro_mismatch, split_seed=31)

    done_mismatch = _checkpoint()
    done_mismatch["completed_training_episodes"] += 1
    done_mismatch["training_history"].append(deepcopy(done_mismatch["training_history"][-1]))
    with pytest.raises(ReplayCostDatasetError, match="boundary count"):
        build_replay_cost_dataset(done_mismatch, split_seed=31)
