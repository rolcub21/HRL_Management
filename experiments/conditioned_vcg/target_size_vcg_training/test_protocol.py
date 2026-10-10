"""Fast, episode-free checks for the staged target-size protocol."""

from types import SimpleNamespace
import unittest

from experiments.conditioned_vcg.target_size_vcg_training import evaluate
from experiments.conditioned_vcg.target_size_vcg_training import protocol as p


class TargetSizeProtocolTests(unittest.TestCase):
    def test_e28_workload_identity(self):
        p.check_splits()
        for name in p.SCENARIOS:
            seed = p.PILOT_SEEDS[name][0]
            expected = p.instance_for(name, seed)
            actual = p.TargetSizeEnv(name).sample_episode_instance(seed)
            self.assertEqual(actual.instance_id, expected.instance_id)
            self.assertEqual(actual.arrival_steps[-1], (actual.number_blocks - 1) * 10)

    def test_two_clock_identity(self):
        blocks = [
            SimpleNamespace(arrival_step=0, stored_time_step=5,
                            delivered_time_step=85, storage_steps_needed=80,
                            delivery_error_time=0, delivered=True),
            SimpleNamespace(arrival_step=10, stored_time_step=20,
                            delivered_time_step=105, storage_steps_needed=80,
                            delivery_error_time=5, delivered=True),
        ]
        result = evaluate._two_clock_metrics(
            SimpleNamespace(blocks=blocks), strict=True, required=2
        )
        self.assertEqual(result["mean_pre_storage_flow_steps"], 7.5)
        self.assertEqual(result["release_mean_absolute_error"], 10.0)
        self.assertTrue(result["timing_identity_verified"])
        blocks[0].delivery_error_time = 1
        with self.assertRaisesRegex(RuntimeError, "timing identity"):
            evaluate._two_clock_metrics(
                SimpleNamespace(blocks=blocks), strict=True, required=2
            )

    def test_baseline_identity_is_deterministic(self):
        name = "empty_8x8_n36"
        seed = p.TEST_SEEDS[name][0]
        first = evaluate._identity(evaluate.e28.GA, name, seed, 0, "test")
        second = evaluate._identity(evaluate.e28.GA, name, seed, 1, "test")
        self.assertNotEqual(first["rng_seed"], second["rng_seed"])
        self.assertEqual(first, evaluate._identity(evaluate.e28.GA, name, seed, 0, "test"))


if __name__ == "__main__":
    unittest.main()
