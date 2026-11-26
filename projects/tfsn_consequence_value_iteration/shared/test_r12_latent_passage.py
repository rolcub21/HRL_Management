from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .latent_detection import (
    LatentRadiusPosterior,
    PosteriorFalsifiedError,
    exact_support_diameter,
)
from .learned_passage import (
    LOSS_NAME,
    MODEL_CLASS,
    PassageMLP,
    PassageTrainingConfig,
    passage_loss,
    passage_targets,
    train_passage_ensemble,
    LearnedPassageEnsemble,
)


class LatentRadiusPosteriorTest(unittest.TestCase):
    def test_candidates_are_unique_recorded_states_and_prior_is_uniform(self):
        posterior = LatentRadiusPosterior(
            np.asarray([[0, 0], [10, 0], [0, 0]], np.float32), sigma=1.0
        )
        np.testing.assert_array_equal(
            posterior.goal_xy, np.asarray([[0, 0], [10, 0]], np.float64)
        )
        np.testing.assert_allclose(posterior.goal_probability, [0.5, 0.5])
        self.assertEqual(posterior.summary().candidate_count, 2)

    def test_support_diameter_is_exact_not_a_grid_bound(self):
        value = exact_support_diameter(
            np.asarray([[0, 0], [3, 0], [0, 4], [1, 1]], np.float32)
        )
        self.assertEqual(value, 5.0)

    def test_log_uniform_radius_is_marginalized_analytically(self):
        posterior = LatentRadiusPosterior(
            np.asarray([[0, 0], [10, 0]], np.float32), sigma=1.0
        )
        # At the first support point, contact is certain for g0 and impossible
        # for g1.  Uniform p0 therefore gives exactly one half.
        self.assertAlmostEqual(
            float(posterior.contact_probability(np.asarray([[0, 0]]))), 0.5
        )
        posterior.update(np.asarray([[0, 0]]), np.asarray([0]))
        np.testing.assert_allclose(posterior.goal_probability, [0.0, 1.0])
        self.assertAlmostEqual(
            float(posterior.contact_probability(np.asarray([[9, 0]]))), 1.0
        )

    def test_positive_and_negative_observations_bracket_radius(self):
        posterior = LatentRadiusPosterior(
            np.asarray([[0, 0], [10, 0]], np.float32), sigma=1.0
        )
        posterior.update(np.asarray([[3, 0]]), np.asarray([1]))
        self.assertAlmostEqual(posterior.lower[0], 3.0)
        posterior.update(np.asarray([[4, 0]]), np.asarray([0]))
        self.assertAlmostEqual(posterior.upper[0], 4.0)

    def test_exhaustion_is_fatal_and_never_uniform(self):
        posterior = LatentRadiusPosterior(
            np.asarray([[0, 0], [10, 0]], np.float32), sigma=1.0
        )
        with self.assertRaisesRegex(PosteriorFalsifiedError, "no fallback"):
            posterior.update(
                np.asarray([[0, 0], [10, 0]]), np.asarray([0, 0])
            )
        self.assertTrue(posterior.exhausted)
        np.testing.assert_array_equal(posterior.goal_probability, [0.0, 0.0])


class SharedPassageModelTest(unittest.TestCase):
    def test_target_rule_is_shared_time_group_mean(self):
        states = np.asarray([[0, 0, 7]], np.float32)
        primitive = np.asarray([[[1, 0], [2, 0], [3, 0], [4, 0]]], np.float32)
        target = passage_targets(states, primitive, sigma=2.0, passage_bins=2)
        np.testing.assert_allclose(target, [[[0.75, 0], [1.75, 0]]])

    def test_single_declared_class_and_loss(self):
        self.assertEqual(MODEL_CLASS, "shared.learned_passage.PassageMLP")
        model = PassageMLP(7, 2, 8, 1)
        import torch
        output = model(torch.zeros((3, 7)))
        self.assertEqual(tuple(output.shape), (3, 2, 2))
        self.assertEqual(float(passage_loss(output, output).detach()), 0.0)
        self.assertIn("mean_squared_error", LOSS_NAME)

    def test_shared_trainer_artifact_round_trip(self):
        rows, steps = 12, 4
        states = np.zeros((rows, 4), np.float32)
        states[:, 0] = np.arange(rows)
        commands = states[:, :2] + np.asarray([1.0, 0.0], np.float32)
        fraction = np.linspace(0.25, 1.0, steps, dtype=np.float32)
        primitive = states[:, None, :2] + fraction[None, :, None] * np.asarray(
            [1.0, 0.0], np.float32
        )
        config = PassageTrainingConfig(
            passage_bins=2,
            ensemble_size=1,
            hidden_dimension=8,
            hidden_layers=1,
            batch_size=4,
            epochs=2,
            learning_rate=1e-3,
            validation_fraction=0.25,
            std_floor=1e-4,
            seed=7,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = train_passage_ensemble(
                states=states,
                commands=commands,
                primitive_xy=primitive,
                group_id=np.arange(rows),
                sigma=0.5,
                config=config,
                artifact_path=root / "passage.pt",
                metadata_path=root / "passage.json",
                provenance={"adapter": "unit_test"},
                device="cpu",
            )
            self.assertEqual(metadata["model_class"], MODEL_CLASS)
            self.assertFalse(metadata["reward_used"])
            saved = json.loads((root / "passage.json").read_text())
            self.assertEqual(saved["loss"], LOSS_NAME)
            runtime = LearnedPassageEnsemble(root / "passage.pt", "cpu")
            prediction = runtime.predict(states[:3], commands[:3])
            self.assertEqual(prediction.shape, (1, 3, 3, 2))


if __name__ == "__main__":
    unittest.main()
