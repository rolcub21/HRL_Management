from pathlib import Path

import numpy as np

from experiments.conditioned_vcg.E03_certification_ablation_90k import (
    full_state_recoverability_atlas as atlas,
)
from PSLAP.viability import RecoveryState
from PSLAP.dynamic_yard import BlockView


def _state(label: str = "B1") -> RecoveryState:
    return RecoveryState(
        rows=3,
        cols=3,
        traversable=frozenset(
            (row, col) for row in range(3) for col in range(3)
        ),
        storage_cells=frozenset({(1, 1), (1, 2)}),
        exits=((2, 1),),
        blocks=(BlockView(label, (1, 1), -4.0),),
        agent_position=(1, 2),
        pickup_cells=frozenset({(0, 1)}),
        wait_cells=frozenset({(0, 2)}),
    )


def test_state_descriptor_contains_load_and_timing() -> None:
    payload = atlas.recovery_state_to_dict(_state())
    descriptor = atlas._state_descriptors(payload)

    assert descriptor["occupancy_fraction"] == 0.5
    assert descriptor["occupied_blocks"] == 1.0
    assert descriptor["overdue_fraction"] == 1.0
    assert descriptor["minimum_remaining_time"] == -4.0
    assert descriptor["mean_remaining_time"] == -4.0


def test_graph_input_is_invariant_to_arbitrary_block_label() -> None:
    first = atlas.encode_recovery_state(_state("B1"))
    renamed = atlas.encode_recovery_state(_state("renamed"))

    assert first.node_features == renamed.node_features
    assert first.edge_index == renamed.edge_index


def test_deterministic_pca_is_repeatable_and_centered() -> None:
    values = np.asarray(
        [[1.0, 2.0, 0.0], [2.0, 1.0, 1.0], [3.0, 4.0, 2.0], [4.0, 3.0, 5.0]]
    )
    first, axes, explained = atlas.deterministic_pca(values, components=3)
    second, axes_second, explained_second = atlas.deterministic_pca(
        values, components=3
    )

    assert np.allclose(first, second)
    assert np.allclose(axes, axes_second)
    assert np.allclose(explained, explained_second)
    assert np.allclose(first.mean(axis=0), 0.0)
    assert np.isclose(explained.sum(), 1.0)


def test_context_separates_absolute_time_and_workload_status() -> None:
    snapshot = {
        "time": 12,
        "blocks": [
            {
                "arrival_step": 0,
                "stored": True,
                "delivered": False,
                "carrying": False,
                "storage_needed": 20,
                "storage_elapsed": 8,
            },
            {
                "arrival_step": 15,
                "stored": False,
                "delivered": False,
                "carrying": False,
                "storage_needed": 10,
                "storage_elapsed": 0,
            },
        ],
    }
    context = atlas._context(snapshot)

    assert context["absolute_time"] == 12
    assert context["arrived_workloads"] == 1
    assert context["future_arrivals"] == 1
    assert context["minimum_remaining_time"] == 12.0
