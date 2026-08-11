from dataclasses import replace
import unittest

import torch

from PSLAP.dynamic_yard import BlockView
from PSLAP.viability import RecoveryState
from PSLAP.yard_graph import (
    NODE_FEATURE_DIM,
    NODE_FEATURE_NAMES,
    PaddedYardGraphBatch,
    YardGraphEncoder,
    encode_recovery_state,
    pad_yard_graphs,
    padded_batch_tensors,
)


def compact_state(remaining_time=-4.0):
    # ``(1, 1)`` is a wall.  The remaining traversable cells form one narrow
    # U-shaped route, making the sparse topology easy to audit exactly.
    return RecoveryState(
        rows=2,
        cols=3,
        traversable=frozenset(
            {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2)}
        ),
        storage_cells=frozenset({(0, 1), (0, 2), (1, 2)}),
        exits=((1, 2),),
        blocks=(BlockView("A", (0, 1), remaining_time),),
        agent_position=(0, 0),
        fixed_obstacles=frozenset({(1, 0)}),
        reserved_cells=frozenset({(0, 2)}),
        pickup_cells=frozenset({(0, 0)}),
        wait_cells=frozenset({(1, 0)}),
    )


def small_state():
    return RecoveryState(
        rows=1,
        cols=2,
        traversable=frozenset({(0, 0), (0, 1)}),
        storage_cells=frozenset({(0, 0)}),
        exits=((0, 1),),
        blocks=(),
        agent_position=(0, 1),
    )


class YardGraphEncodingTests(unittest.TestCase):
    def test_encodes_static_dynamic_and_masked_timing_features(self):
        graph = encode_recovery_state(compact_state(), timing_scale=2.0)

        self.assertEqual(graph.node_count, 6)
        self.assertEqual(graph.feature_dim, NODE_FEATURE_DIM)
        self.assertEqual(graph.feature_names, NODE_FEATURE_NAMES)
        self.assertEqual(
            graph.cells,
            ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
        )

        agent = graph.features_for((0, 0))
        self.assertEqual(agent["traversable"], 1.0)
        self.assertEqual(agent["pickup"], 1.0)
        self.assertEqual(agent["agent"], 1.0)
        self.assertEqual(agent["occupied"], 0.0)

        occupant = graph.features_for((0, 1))
        self.assertEqual(occupant["storage"], 1.0)
        self.assertEqual(occupant["occupied"], 1.0)
        self.assertEqual(occupant["occupant_time_known"], 1.0)
        self.assertEqual(occupant["occupant_remaining_time_scaled"], -2.0)
        self.assertEqual(
            occupant["occupant_abs_remaining_time_scaled"], 2.0
        )
        self.assertEqual(occupant["occupant_overdue"], 1.0)

        reserved = graph.features_for((0, 2))
        self.assertEqual(reserved["reserved"], 1.0)
        self.assertEqual(reserved["occupied"], 0.0)

        fixed = graph.features_for((1, 0))
        self.assertEqual(fixed["wait"], 1.0)
        self.assertEqual(fixed["fixed_obstacle"], 1.0)

        wall = graph.features_for((1, 1))
        self.assertEqual(wall["traversable"], 0.0)
        self.assertEqual(wall["row_fraction"], 1.0)
        self.assertEqual(wall["column_fraction"], 0.5)
        wall_index = graph.index((1, 1))
        self.assertFalse(
            any(wall_index in edge for edge in graph.edges),
            "walls must be isolated in the physical sparse graph",
        )

        exit_features = graph.features_for((1, 2))
        self.assertEqual(exit_features["exit"], 1.0)
        self.assertEqual(graph.edge_count, 8)
        self.assertEqual(set(graph.edges), {
            (graph.index((0, 0)), graph.index((0, 1))),
            (graph.index((0, 1)), graph.index((0, 0))),
            (graph.index((0, 0)), graph.index((1, 0))),
            (graph.index((1, 0)), graph.index((0, 0))),
            (graph.index((0, 1)), graph.index((0, 2))),
            (graph.index((0, 2)), graph.index((0, 1))),
            (graph.index((0, 2)), graph.index((1, 2))),
            (graph.index((1, 2)), graph.index((0, 2))),
        })

    def test_counterfactual_state_carries_pickup_and_wait_roles_by_default(self):
        state = compact_state()
        successor = replace(
            state,
            blocks=(BlockView("A", (1, 2), 4.0),),
        )

        for graph in (
            encode_recovery_state(state),
            encode_recovery_state(successor),
        ):
            self.assertEqual(graph.features_for((0, 0))["pickup"], 1.0)
            self.assertEqual(graph.features_for((1, 0))["wait"], 1.0)

    def test_unavailable_timing_is_zero_with_an_explicit_unknown_mask(self):
        graph = encode_recovery_state(compact_state(float("nan")))

        occupant = graph.features_for((0, 1))
        self.assertEqual(occupant["occupant_time_known"], 0.0)
        self.assertEqual(occupant["occupant_remaining_time_scaled"], 0.0)
        self.assertEqual(occupant["occupant_abs_remaining_time_scaled"], 0.0)
        self.assertEqual(occupant["occupant_overdue"], 0.0)

    def test_counterfactual_state_uses_same_policy_independent_encoder(self):
        current = compact_state(12.0)
        successor = replace(
            current,
            blocks=(BlockView("A", (1, 2), 12.0),),
        )

        current_graph = encode_recovery_state(current)
        successor_graph = encode_recovery_state(successor)

        self.assertEqual(current_graph.cells, successor_graph.cells)
        self.assertEqual(current_graph.edge_index, successor_graph.edge_index)
        self.assertEqual(
            current_graph.features_for((0, 1))["occupied"], 1.0
        )
        self.assertEqual(
            successor_graph.features_for((0, 1))["occupied"], 0.0
        )
        self.assertEqual(
            successor_graph.features_for((1, 2))["occupied"], 1.0
        )

    def test_rejects_units_and_special_cells_outside_the_graph(self):
        state = compact_state()

        with self.assertRaisesRegex(ValueError, "timing_scale"):
            encode_recovery_state(state, timing_scale=0.0)
        with self.assertRaisesRegex(ValueError, "pickup_cells.*out-of-bounds"):
            encode_recovery_state(state, pickup_cells=(3, 0))


class YardGraphBatchTests(unittest.TestCase):
    def test_pads_plain_data_and_offsets_sparse_edges_by_batch_stride(self):
        first = encode_recovery_state(small_state())
        second = encode_recovery_state(compact_state())

        batch = pad_yard_graphs((first, second), pad_value=-9.0)

        self.assertIsInstance(batch, PaddedYardGraphBatch)
        self.assertIsInstance(batch.node_features, tuple)
        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.max_nodes, 6)
        self.assertEqual(batch.node_counts, (2, 6))
        self.assertEqual(batch.node_mask[0], (True, True, False, False, False, False))
        self.assertEqual(batch.cells[0][2:], (None, None, None, None))
        self.assertEqual(
            batch.node_features[0][2], (-9.0,) * NODE_FEATURE_DIM
        )

        first_sources = [
            source
            for source in batch.edge_index[0]
            if source < batch.max_nodes
        ]
        first_destinations = [
            destination
            for source, destination in zip(*batch.edge_index)
            if source < batch.max_nodes
        ]
        self.assertEqual(set(first_sources), {0, 1})
        self.assertEqual(set(first_destinations), {0, 1})
        self.assertTrue(
            all(
                source >= batch.max_nodes and destination >= batch.max_nodes
                for source, destination in zip(*batch.edge_index)
                if source not in (0, 1)
            )
        )

        features, mask, edge_index = padded_batch_tensors(batch)
        self.assertEqual(features.shape, (2, 6, NODE_FEATURE_DIM))
        self.assertEqual(mask.shape, (2, 6))
        self.assertEqual(edge_index.shape[0], 2)
        self.assertEqual(features.dtype, torch.float32)
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(edge_index.dtype, torch.long)

    def test_empty_batch_is_rejected_explicitly(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            pad_yard_graphs(())


class YardGraphEncoderTests(unittest.TestCase):
    def test_encoder_is_padding_invariant_and_differentiable(self):
        torch.manual_seed(7)
        first = encode_recovery_state(small_state())
        second = encode_recovery_state(compact_state())
        encoder = YardGraphEncoder(
            hidden_dim=16,
            output_dim=12,
            message_passing_steps=2,
        )
        encoder.eval()

        alone = encoder(pad_yard_graphs((first,)))
        mixed = encoder(pad_yard_graphs((first, second)))

        self.assertEqual(alone.shape, (1, 12))
        self.assertEqual(mixed.shape, (2, 12))
        torch.testing.assert_close(alone[0], mixed[0], atol=1e-6, rtol=0)
        self.assertTrue(torch.isfinite(mixed).all())

        encoder.train()
        loss = encoder(pad_yard_graphs((first, second))).square().mean()
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in encoder.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_pooling_retains_sum_mean_and_valid_node_count(self):
        torch.manual_seed(11)
        encoder = YardGraphEncoder(
            hidden_dim=8,
            output_dim=6,
            message_passing_steps=0,
        )
        encoder.eval()
        features = torch.ones((2, 2, NODE_FEATURE_DIM))
        mask = torch.tensor(((True, False), (True, True)))
        edge_index = torch.empty((2, 0), dtype=torch.long)
        pooling_inputs = []
        handle = encoder.pool_projection.register_forward_pre_hook(
            lambda _module, arguments: pooling_inputs.append(
                arguments[0].detach()
            )
        )

        try:
            encoder(features, mask, edge_index)
        finally:
            handle.remove()

        pooled = pooling_inputs[0]
        hidden_dim = encoder.hidden_dim
        one_sum = pooled[0, :hidden_dim]
        two_sum = pooled[1, :hidden_dim]
        one_mean = pooled[0, hidden_dim : 2 * hidden_dim]
        two_mean = pooled[1, hidden_dim : 2 * hidden_dim]
        torch.testing.assert_close(two_sum, 2.0 * one_sum)
        torch.testing.assert_close(one_mean, two_mean)
        torch.testing.assert_close(
            pooled[:, -1],
            torch.log1p(torch.tensor((1.0, 2.0))),
        )

    def test_encoder_handles_an_edgeless_valid_graph(self):
        state = RecoveryState(
            rows=1,
            cols=1,
            traversable=frozenset({(0, 0)}),
            storage_cells=frozenset(),
            exits=((0, 0),),
            blocks=(),
            agent_position=(0, 0),
        )
        batch = pad_yard_graphs((encode_recovery_state(state),))
        encoder = YardGraphEncoder(hidden_dim=8, output_dim=6)

        output = encoder(batch)

        self.assertEqual(output.shape, (1, 6))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
