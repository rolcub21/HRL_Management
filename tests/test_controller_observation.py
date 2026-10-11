from dataclasses import replace
import unittest

import numpy as np

from example.controller_observation import (
    BLOCK_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    OnlineSignedTimingObservationEncoder,
    OnlineManifestTimingObservationEncoder,
    controller_observation_encoder_from_checkpoint,
)
from example.small_rooms_env import SmallRoomsEnv
from PSLAP.dynamic_yard import YardSnapshot
from PSLAP.retrieval_dispatch import plan_retrieval


def configure_single_stored_block(env, remaining):
    env.reset(instance=env.sample_episode_instance(700))
    for block in env.blocks:
        block.position = None
        block.storage_location = None
        block.carrying = False
        block.stored = False
        block.delivered = False
        block.stored_time_step = None
    env.time_steps = 50
    block = env.blocks[0]
    block.position = env.storage_positions[len(env.storage_positions) // 2]
    block.storage_location = block.position
    block.stored = True
    block.stored_time_step = 0
    block.storage_steps_elapsed = 50
    block.storage_steps_needed = 50 + remaining
    block.arrival_step = 0
    return block


class ControllerObservationTimingTests(unittest.TestCase):
    def make_env(self):
        return SmallRoomsEnv(
            choose_storage=False,
            arrival_rate=0.5,
            proc_mean=50,
        )

    def test_due_and_overdue_blocks_remain_distinct_and_signed(self):
        due_env = self.make_env()
        configure_single_stored_block(due_env, 0)
        due = OnlineSignedTimingObservationEncoder(due_env).capture()

        late_env = self.make_env()
        configure_single_stored_block(late_env, -10)
        late = OnlineSignedTimingObservationEncoder(late_env).capture()

        valid = BLOCK_FEATURE_NAMES.index("deadline_valid")
        remaining = BLOCK_FEATURE_NAMES.index("signed_remaining_time")
        self.assertEqual(due.block_features[0, valid], 1.0)
        self.assertEqual(late.block_features[0, valid], 1.0)
        self.assertEqual(due.remaining_steps[0], 0.0)
        self.assertEqual(late.remaining_steps[0], -10.0)
        self.assertAlmostEqual(due.block_features[0, remaining], 0.0)
        self.assertAlmostEqual(
            late.block_features[0, remaining], -10.0 / 150.0
        )
        self.assertFalse(np.array_equal(due.flatten(), late.flatten()))

    def test_values_above_legacy_timer_cap_do_not_collapse(self):
        env_20 = self.make_env()
        configure_single_stored_block(env_20, 20)
        obs_20 = OnlineSignedTimingObservationEncoder(env_20).capture()
        env_50 = self.make_env()
        configure_single_stored_block(env_50, 50)
        obs_50 = OnlineSignedTimingObservationEncoder(env_50).capture()
        index = BLOCK_FEATURE_NAMES.index("signed_remaining_time")
        self.assertAlmostEqual(obs_20.block_features[0, index], 20 / 150)
        self.assertAlmostEqual(obs_50.block_features[0, index], 50 / 150)
        self.assertNotEqual(
            obs_20.block_features[0, index], obs_50.block_features[0, index]
        )

    def test_missing_timing_has_an_explicit_validity_bit(self):
        env = self.make_env()
        env.reset(instance=env.sample_episode_instance(701))
        observation = OnlineSignedTimingObservationEncoder(env).capture()
        valid = BLOCK_FEATURE_NAMES.index("deadline_valid")
        remaining = BLOCK_FEATURE_NAMES.index("signed_remaining_time")
        self.assertEqual(observation.block_features[0, valid], 0.0)
        self.assertEqual(observation.block_features[0, remaining], 0.0)

    def test_retrieval_eta_and_slack_match_shared_planner(self):
        env = self.make_env()
        block = configure_single_stored_block(env, 30)
        encoder = OnlineSignedTimingObservationEncoder(env)
        observation = encoder.capture()
        plan = plan_retrieval(
            YardSnapshot.from_env(env), env.current_state, block.label
        )
        self.assertIsNotNone(plan)
        self.assertEqual(observation.retrieval_eta_steps[0], plan.estimated_steps)
        self.assertEqual(
            observation.signed_slack_steps[0],
            30 - plan.estimated_steps,
        )
        self.assertEqual(plan.slack, 30 - plan.estimated_steps)


class ControllerObservationInformationTests(unittest.TestCase):
    def make_env(self):
        return SmallRoomsEnv(
            choose_storage=False,
            arrival_rate=0.5,
            proc_mean=50,
        )

    def test_arrived_queue_and_current_time_are_observable(self):
        env = self.make_env()
        base = env.sample_episode_instance(702)
        arrivals = (0, 0, 0) + tuple(range(100, 137))
        instance = replace(base, arrival_steps=arrivals)
        env.reset(instance=instance)
        encoder = OnlineSignedTimingObservationEncoder(env)
        initial = encoder.capture()
        self.assertEqual(initial.global_raw["waiting_count"], 2.0)
        self.assertEqual(initial.global_raw["current_step"], 0.0)

        for _ in range(7):
            env.step(env.ACTION_IDS["WAIT"])
        later = encoder.capture()
        self.assertEqual(later.global_raw["current_step"], 7.0)
        current_step = GLOBAL_FEATURE_NAMES.index("current_step")
        self.assertGreater(
            later.global_features[current_step],
            initial.global_features[current_step],
        )

    def test_unannounced_future_schedule_and_durations_do_not_leak(self):
        env_a = self.make_env()
        base = env_a.sample_episode_instance(703)
        arrivals_a = (0,) + tuple(range(10, 49))
        arrivals_b = (0,) + tuple(range(100, 139))
        durations_a = (base.storage_steps_needed[0],) + (1,) * 39
        durations_b = (base.storage_steps_needed[0],) + (149,) * 39
        instance_a = replace(
            base,
            arrival_steps=arrivals_a,
            storage_steps_needed=durations_a,
        )
        instance_b = replace(
            base,
            arrival_steps=arrivals_b,
            storage_steps_needed=durations_b,
        )
        env_b = self.make_env()
        env_a.reset(instance=instance_a)
        env_b.reset(instance=instance_b)
        for _ in range(5):
            env_a.step(env_a.ACTION_IDS["WAIT"])
            env_b.step(env_b.ACTION_IDS["WAIT"])
        observation_a = OnlineSignedTimingObservationEncoder(env_a).capture()
        observation_b = OnlineSignedTimingObservationEncoder(env_b).capture()
        np.testing.assert_array_equal(
            observation_a.global_features, observation_b.global_features
        )
        np.testing.assert_array_equal(
            observation_a.block_mask, observation_b.block_mask
        )
        np.testing.assert_array_equal(
            observation_a.block_features, observation_b.block_features
        )
        np.testing.assert_array_equal(
            observation_a.flatten(), observation_b.flatten()
        )
        manifest_a = OnlineManifestTimingObservationEncoder(env_a).capture()
        manifest_b = OnlineManifestTimingObservationEncoder(env_b).capture()
        np.testing.assert_array_equal(manifest_a.flatten(), manifest_b.flatten())

    def test_feature_dimension_and_metadata_are_explicit(self):
        env = self.make_env()
        env.reset(instance=env.sample_episode_instance(704))
        encoder = OnlineSignedTimingObservationEncoder(env)
        observation = encoder.capture()
        self.assertEqual(len(GLOBAL_FEATURE_NAMES), 20)
        self.assertEqual(len(BLOCK_FEATURE_NAMES), 29)
        self.assertEqual(observation.flatten().shape, (1220,))
        self.assertEqual(
            encoder.metadata()["controller_observation_feature_size"], 1220
        )

    def test_checkpoint_metadata_is_complete_and_validated(self):
        env = self.make_env()
        env.reset(instance=env.sample_episode_instance(705))
        encoder = OnlineSignedTimingObservationEncoder(env)
        payload = encoder.metadata()
        restored = controller_observation_encoder_from_checkpoint(env, payload)
        self.assertEqual(restored.metadata(), payload)

        incomplete = dict(payload)
        incomplete.pop("controller_observation_block_size")
        with self.assertRaisesRegex(
            ValueError, "Incomplete controller-observation"
        ):
            controller_observation_encoder_from_checkpoint(env, incomplete)

        mismatched = dict(payload)
        mismatched["controller_observation_timing_scale"] = 15.0
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            controller_observation_encoder_from_checkpoint(env, mismatched)


if __name__ == "__main__":
    unittest.main()
