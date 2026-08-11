import dataclasses
import random
import unittest

import numpy as np

from example.episode_instance import EpisodeInstance
from example.small_rooms_env import SmallRoomsEnv
from PSLAP.run_pslap import run_pslap_episode


class EpisodeInstanceContracts(unittest.TestCase):
    def make_env(self, **kwargs):
        return SmallRoomsEnv(
            choose_storage=False,
            arrival_rate=kwargs.get("arrival_rate", 0.5),
            proc_mean=kwargs.get("proc_mean", 50),
            exit_cells=kwargs.get("exit_cells"),
        )

    def test_seeded_sampling_is_independent_of_global_numpy_state(self):
        env = self.make_env()
        np.random.seed(1)
        first = env.sample_episode_instance(101)
        np.random.seed(999)
        np.random.random(100)
        second = env.sample_episode_instance(101)

        self.assertEqual(first, second)
        self.assertEqual(first.instance_id, second.instance_id)

    def test_json_round_trip_preserves_identity(self):
        instance = self.make_env().sample_episode_instance(7)
        restored = EpisodeInstance.from_json(instance.to_json())

        self.assertEqual(restored, instance)
        self.assertEqual(restored.instance_id, instance.instance_id)

    def test_instance_is_immutable(self):
        instance = self.make_env().sample_episode_instance(7)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            instance.seed = 8

    def test_constructor_canonicalizes_mutable_sequences(self):
        sampled = self.make_env().sample_episode_instance(11)
        payload = sampled.to_dict()
        arrivals = list(payload["arrival_steps"])
        payload["arrival_steps"] = arrivals
        payload["start_state"] = list(payload["start_state"])

        instance = EpisodeInstance(**payload)
        arrivals[0] = 99

        self.assertIsInstance(instance.arrival_steps, tuple)
        self.assertIsInstance(instance.start_state, tuple)
        self.assertEqual(instance.arrival_steps[0], 0)

    def test_reset_replays_exact_arrivals_and_durations(self):
        env = self.make_env()
        instance = env.sample_episode_instance(19)

        env.reset(instance=instance)
        first = [
            (block.arrival_step, block.storage_steps_needed)
            for block in env.blocks
        ]
        np.random.seed(1234)
        np.random.random(200)
        env.reset(instance=instance)
        second = [
            (block.arrival_step, block.storage_steps_needed)
            for block in env.blocks
        ]

        expected = list(
            zip(instance.arrival_steps, instance.storage_steps_needed)
        )
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        self.assertEqual(env.current_episode_instance.instance_id, instance.instance_id)

    def test_geometry_mismatch_is_rejected(self):
        instance = self.make_env().sample_episode_instance(3)
        narrow = self.make_env(exit_cells=[(9, 8)])

        with self.assertRaisesRegex(ValueError, "exit_cells"):
            narrow.reset(instance=instance)

    def test_pslap_runner_consumes_the_supplied_instance(self):
        source = self.make_env()
        instance = source.sample_episode_instance(31)
        first_env = self.make_env()
        second_env = self.make_env()

        random.seed(1)
        np.random.seed(1)
        first = run_pslap_episode(
            first_env,
            max_steps=2000,
            baseline="dynamic_pslap",
            episode_instance=instance,
        )
        random.seed(999)
        np.random.seed(999)
        second = run_pslap_episode(
            second_env,
            max_steps=2000,
            baseline="dynamic_pslap",
            episode_instance=instance,
        )

        self.assertEqual(first_env.current_episode_instance, instance)
        self.assertEqual(second_env.current_episode_instance, instance)
        self.assertEqual(first["return"], second["return"])
        self.assertEqual(first["errors"], second["errors"])
        self.assertEqual(first["steps"], second["steps"])


if __name__ == "__main__":
    unittest.main()
