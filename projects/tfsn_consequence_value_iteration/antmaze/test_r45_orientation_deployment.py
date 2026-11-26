import json
from types import SimpleNamespace

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r31_aligned_hidden_goal as r31,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r45_orientation_deployment as r45,
)


def test_r45_isolated_artifact_contract():
    protocol = r45.load_contract()
    assert protocol["gate_task_ids"] == [3, 5]
    assert protocol["confirmation_panel_accessed"] is False
    r45_paths = r45.artifact_paths()
    r31_paths = r31.artifact_paths()
    assert r45_paths["landmarks"] == r31_paths["landmarks"]
    assert r45_paths["mobility"] == r31_paths["mobility"]
    assert r45_paths["bridges"] != r31_paths["bridges"]
    assert r45_paths["bridge_shards"] != r31_paths["bridge_shards"]


def test_r45_reuses_exact_frozen_landmarks():
    graph = r31.verify_graph(r31.graph_path())
    args = SimpleNamespace(root=None, graph=r31.graph_path())
    artifact, landmarks, targets, distance, metadata = r45._frozen_landmarks(
        args, graph, r45.R43_CHECKPOINT
    )
    with np.load(r31.artifact_paths()["landmarks"]) as stored:
        assert np.array_equal(landmarks, stored["landmark_index"])
        assert np.array_equal(targets, stored["target_index"])
        assert np.array_equal(distance, stored["target_distance"])
    assert artifact["mobility"] == r31.artifact_paths()["mobility"]
    assert metadata["base_graph_sha256"] == r31.recorded.sha256_file(
        r31.graph_path()
    )


def test_r43_full_checkpoint_contract():
    report = r45.verify_r43()
    assert report["profile"] == "full"
    assert report["hidden_goals_accessed"] is False
    assert report["task_reward_used"] is False
    assert report["checkpoint_sha256"] == r31.sha256(r45.R43_CHECKPOINT)
