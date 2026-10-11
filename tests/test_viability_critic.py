import unittest

import torch

from PSLAP.dynamic_yard import BlockView
from PSLAP.viability import (
    RecoveryState,
    ViabilityStatus,
    analyze_recoverability,
)
from PSLAP.viability_critic import (
    CounterfactualViabilityCritic,
    DynamicsViabilityLabel,
    ViabilityCriticEnsemble,
    ViabilityCriticOutput,
    compute_viability_critic_loss,
    summarize_viability_ensemble,
)
from PSLAP.yard_graph import encode_recovery_state, pad_yard_graphs


def direct_recovery_state():
    return RecoveryState(
        rows=1,
        cols=3,
        traversable=frozenset({(0, 0), (0, 1), (0, 2)}),
        storage_cells=frozenset({(0, 1)}),
        exits=((0, 2),),
        blocks=(BlockView("A", (0, 1), 10.0),),
        agent_position=(0, 0),
    )


def empty_recovery_state():
    return RecoveryState(
        rows=2,
        cols=2,
        traversable=frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
        storage_cells=frozenset({(0, 1), (1, 0)}),
        exits=((1, 1),),
        blocks=(),
        agent_position=(0, 0),
    )


def output_from_tensors(logits, ranks, steps):
    return ViabilityCriticOutput(
        safety_logit=logits,
        safety_probability=torch.sigmoid(logits),
        recovery_rank=ranks,
        primitive_steps=steps,
    )


class DynamicsViabilityLabelTests(unittest.TestCase):
    def test_safe_certificate_supplies_witness_rank_and_steps(self):
        certificate = analyze_recoverability(
            direct_recovery_state(), max_nodes=None
        )

        label = DynamicsViabilityLabel.from_certificate(certificate)

        self.assertIs(label.status, ViabilityStatus.SAFE)
        self.assertEqual(label.safety_target, 1.0)
        self.assertEqual(label.recovery_rank, 1.0)
        self.assertEqual(label.primitive_steps, 4.0)

    def test_unknown_has_no_classification_or_regression_target(self):
        unknown = DynamicsViabilityLabel(ViabilityStatus.UNKNOWN)

        self.assertFalse(unknown.has_safety_target)
        self.assertIsNone(unknown.safety_target)
        self.assertIsNone(unknown.recovery_rank)
        self.assertIsNone(unknown.primitive_steps)
        with self.assertRaisesRegex(ValueError, "only SAFE"):
            DynamicsViabilityLabel(
                ViabilityStatus.UNKNOWN,
                recovery_rank=0,
            )
        with self.assertRaisesRegex(ValueError, "only SAFE"):
            DynamicsViabilityLabel(
                ViabilityStatus.UNSAFE,
                primitive_steps=12,
            )

    def test_goal_directed_witness_is_not_mislabeled_as_exact_rank(self):
        certificate = analyze_recoverability(
            direct_recovery_state(),
            max_nodes=None,
            search_order="goal_directed",
        )

        label = DynamicsViabilityLabel.from_certificate(certificate)

        self.assertIs(label.status, ViabilityStatus.SAFE)
        self.assertIsNone(label.recovery_rank)
        self.assertEqual(label.primitive_steps, 4.0)


class CounterfactualViabilityCriticTests(unittest.TestCase):
    def test_graph_batch_produces_separate_finite_nonnegative_outputs(self):
        torch.manual_seed(17)
        graphs = (
            encode_recovery_state(direct_recovery_state()),
            encode_recovery_state(empty_recovery_state()),
        )
        critic = CounterfactualViabilityCritic(
            graph_hidden_dim=16,
            graph_embedding_dim=12,
            head_hidden_dim=8,
            message_passing_steps=1,
        )

        output = critic(pad_yard_graphs(graphs))

        self.assertEqual(output.safety_logit.shape, (2,))
        self.assertEqual(output.recovery_rank.shape, (2,))
        self.assertEqual(output.primitive_steps.shape, (2,))
        torch.testing.assert_close(
            output.safety_probability,
            torch.sigmoid(output.safety_logit),
        )
        self.assertTrue(torch.isfinite(output.safety_logit).all())
        self.assertTrue((output.recovery_rank >= 0.0).all())
        self.assertTrue((output.primitive_steps >= 0.0).all())
        self.assertIsNot(
            critic.safety_head,
            critic.recovery_rank_head,
        )
        self.assertIsNot(
            critic.recovery_rank_head,
            critic.primitive_steps_head,
        )

        (
            output.safety_logit.mean()
            + output.recovery_rank.mean()
            + output.primitive_steps.mean()
        ).backward()
        gradients = [
            parameter.grad
            for parameter in critic.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in gradients)
        )

    def test_ensemble_container_runs_independent_graph_critics(self):
        torch.manual_seed(23)
        batch = pad_yard_graphs(
            (encode_recovery_state(empty_recovery_state()),)
        )
        ensemble = ViabilityCriticEnsemble(
            (
                CounterfactualViabilityCritic(
                    graph_hidden_dim=8,
                    graph_embedding_dim=6,
                    head_hidden_dim=4,
                    message_passing_steps=0,
                ),
                CounterfactualViabilityCritic(
                    graph_hidden_dim=8,
                    graph_embedding_dim=6,
                    head_hidden_dim=4,
                    message_passing_steps=0,
                ),
            )
        )

        outputs = ensemble(batch)
        estimate = ensemble.estimate(batch, lcb_scale=1.0)

        self.assertEqual(len(outputs), 2)
        self.assertEqual(estimate.member_count, 2)
        self.assertEqual(estimate.safety_probability_lcb.shape, (1,))
        self.assertTrue(
            torch.all(
                estimate.safety_probability_lcb
                <= estimate.safety_probability_mean
            )
        )


class ViabilityCriticLossTests(unittest.TestCase):
    def test_false_safe_errors_receive_extra_weight_and_unknown_is_masked(self):
        logits = torch.tensor((2.0, 20.0), requires_grad=True)
        ranks = torch.tensor((1.0, 1000.0), requires_grad=True)
        steps = torch.tensor((4.0, 1000.0), requires_grad=True)
        output = output_from_tensors(logits, ranks, steps)
        labels = (
            DynamicsViabilityLabel(ViabilityStatus.UNSAFE),
            DynamicsViabilityLabel(ViabilityStatus.UNKNOWN),
        )

        ordinary = compute_viability_critic_loss(
            output,
            labels,
            false_safe_weight=1.0,
        )
        asymmetric = compute_viability_critic_loss(
            output,
            labels,
            false_safe_weight=4.0,
        )

        torch.testing.assert_close(
            asymmetric.safety,
            4.0 * ordinary.safety,
        )
        self.assertEqual(asymmetric.known_safety_count, 1)
        self.assertEqual(asymmetric.recovery_rank_count, 0)
        self.assertEqual(asymmetric.primitive_steps_count, 0)
        asymmetric.total.backward()
        self.assertGreater(logits.grad[0].item(), 0.0)
        self.assertEqual(logits.grad[1].item(), 0.0)
        self.assertEqual(ranks.grad[1].item(), 0.0)
        self.assertEqual(steps.grad[1].item(), 0.0)

    def test_witness_regression_uses_only_safe_labels_with_values(self):
        logits = torch.zeros(3, requires_grad=True)
        ranks = torch.tensor((5.0, 100.0, 1000.0), requires_grad=True)
        steps = torch.tensor((18.0, 100.0, 1000.0), requires_grad=True)
        labels = (
            DynamicsViabilityLabel(
                ViabilityStatus.SAFE,
                recovery_rank=2,
                primitive_steps=8,
            ),
            DynamicsViabilityLabel(ViabilityStatus.UNSAFE),
            DynamicsViabilityLabel(ViabilityStatus.UNKNOWN),
        )

        loss = compute_viability_critic_loss(
            output_from_tensors(logits, ranks, steps),
            labels,
            safety_weight=0.0,
        )

        self.assertEqual(loss.known_safety_count, 2)
        self.assertEqual(loss.recovery_rank_count, 1)
        self.assertEqual(loss.primitive_steps_count, 1)
        self.assertGreater(loss.recovery_rank.item(), 0.0)
        self.assertGreater(loss.primitive_steps.item(), 0.0)
        loss.total.backward()
        self.assertNotEqual(ranks.grad[0].item(), 0.0)
        self.assertEqual(ranks.grad[1].item(), 0.0)
        self.assertEqual(ranks.grad[2].item(), 0.0)
        self.assertNotEqual(steps.grad[0].item(), 0.0)
        self.assertEqual(steps.grad[1].item(), 0.0)
        self.assertEqual(steps.grad[2].item(), 0.0)

    def test_all_unknown_batch_has_differentiable_zero_loss(self):
        logits = torch.tensor((4.0,), requires_grad=True)
        ranks = torch.tensor((9.0,), requires_grad=True)
        steps = torch.tensor((40.0,), requires_grad=True)

        loss = compute_viability_critic_loss(
            output_from_tensors(logits, ranks, steps),
            (DynamicsViabilityLabel(ViabilityStatus.UNKNOWN),),
        )

        self.assertEqual(loss.total.item(), 0.0)
        self.assertEqual(loss.known_safety_count, 0)
        loss.total.backward()
        self.assertEqual(logits.grad.item(), 0.0)
        self.assertEqual(ranks.grad.item(), 0.0)
        self.assertEqual(steps.grad.item(), 0.0)


class ViabilityEnsembleEstimateTests(unittest.TestCase):
    def test_lcb_uses_population_disagreement_and_stays_in_probability_range(self):
        first_logits = torch.logit(torch.tensor((0.9, 0.1)))
        second_logits = torch.logit(torch.tensor((0.7, 0.1)))
        first = output_from_tensors(
            first_logits,
            torch.tensor((2.0, 4.0)),
            torch.tensor((8.0, 12.0)),
        )
        second = output_from_tensors(
            second_logits,
            torch.tensor((4.0, 6.0)),
            torch.tensor((10.0, 14.0)),
        )

        estimate = summarize_viability_ensemble(
            (first, second), lcb_scale=2.0
        )

        torch.testing.assert_close(
            estimate.safety_probability_mean,
            torch.tensor((0.8, 0.1)),
        )
        torch.testing.assert_close(
            estimate.safety_probability_std,
            torch.tensor((0.1, 0.0)),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            estimate.safety_probability_lcb,
            torch.tensor((0.6, 0.1)),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            estimate.recovery_rank_mean,
            torch.tensor((3.0, 5.0)),
        )
        torch.testing.assert_close(
            estimate.primitive_steps_mean,
            torch.tensor((9.0, 13.0)),
        )


if __name__ == "__main__":
    unittest.main()
