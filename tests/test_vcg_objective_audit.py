import math
import unittest

from example.small_rooms_env import SmallRoomsEnv
from vcg_objective_audit import (
    DENSE_PIECEWISE,
    LEGACY_CLIPPED,
    TIMING_OBJECTIVE_INFO_KEY,
    ObjectiveAuditSmallRoomsEnv,
    TimingObjectiveSpec,
    canonical_timing_objective,
    delivery_reward,
    delivery_reward_components,
    rescore_episode_return,
    rescore_logged_return,
    rescore_logged_return_with_spec,
    rescore_step_reward,
)


def _prepare_delivery(env, *, error):
    """Place the one-block test environment one PUTDOWN from delivery."""

    block = env.blocks[0]
    exit_cell = tuple(env.exit_cells[0])
    block.stored = True
    block.delivered = False
    block.carrying = True
    block.picked = True
    block.position = exit_cell
    block.storage_location = (1, 1)
    block.stored_time_step = 0
    block.storage_steps_needed = 100
    block.storage_steps_elapsed = 0
    env.current_state = exit_cell
    # step increments time before computing e = T_delivery - T_deadline.
    env.time_steps = 100 + int(error) - 1


class TimingObjectiveSpecTests(unittest.TestCase):
    def test_aliases_and_serialization_are_versioned(self):
        spec = TimingObjectiveSpec.dense(
            dense_b=41.0,
            lambda_abs=1.25,
            lambda_outside=0.75,
            window=20.0,
        )

        self.assertEqual(canonical_timing_objective("dense"), DENSE_PIECEWISE)
        self.assertEqual(
            canonical_timing_objective("legacy"), LEGACY_CLIPPED
        )
        self.assertEqual(TimingObjectiveSpec.from_dict(spec.to_dict()), spec)
        self.assertTrue(TimingObjectiveSpec.legacy().exact_legacy)
        self.assertTrue(
            TimingObjectiveSpec.dense().dense_matches_legacy_inside_window
        )

    def test_rejects_unknown_objective_and_invalid_coefficients(self):
        with self.assertRaisesRegex(ValueError, "unknown timing objective"):
            TimingObjectiveSpec(objective="mystery")
        with self.assertRaisesRegex(ValueError, "non-negative"):
            TimingObjectiveSpec.dense(lambda_outside=-0.1)
        with self.assertRaisesRegex(ValueError, "positive"):
            TimingObjectiveSpec.dense(window=0.0)


class TimingRewardFormulaTests(unittest.TestCase):
    def test_default_dense_matches_exact_legacy_inside_window_only(self):
        spec = TimingObjectiveSpec.dense()

        for error in (-20.0, -10.0, 0.0, 10.0, 20.0):
            self.assertAlmostEqual(
                delivery_reward(error, spec, LEGACY_CLIPPED),
                delivery_reward(error, spec, DENSE_PIECEWISE),
            )
        self.assertEqual(delivery_reward(30.0, spec, LEGACY_CLIPPED), 10.0)
        self.assertEqual(delivery_reward(30.0, spec, DENSE_PIECEWISE), -10.0)

    def test_dense_components_are_explicit_and_not_clipped(self):
        spec = TimingObjectiveSpec.dense(
            dense_b=50.0,
            lambda_abs=1.0,
            lambda_outside=2.0,
            window=20.0,
        )

        components = delivery_reward_components(-35.0, spec)

        self.assertEqual(components.signed_error, -35.0)
        self.assertEqual(components.absolute_error, 35.0)
        self.assertEqual(components.dense_excess_error, 15.0)
        self.assertEqual(components.dense_absolute_penalty, 35.0)
        self.assertEqual(components.dense_outside_penalty, 30.0)
        self.assertEqual(components.dense_delivery_reward, -15.0)

    def test_aggregate_rescoring_handles_incomplete_episodes(self):
        spec = TimingObjectiveSpec.dense()
        deviations = (-30.0, 0.0)

        legacy = rescore_episode_return(
            deviations,
            steps=50,
            delivery_count=2,
            placement_count=3,
            spec=spec,
            objective=LEGACY_CLIPPED,
        )
        dense = rescore_episode_return(
            deviations,
            steps=50,
            delivery_count=2,
            placement_count=3,
            spec=spec,
            objective=DENSE_PIECEWISE,
        )

        self.assertAlmostEqual(legacy, -4.5 + 15.0 + 10.0 + 40.0)
        self.assertAlmostEqual(dense, -4.5 + 15.0 - 10.0 + 40.0)
        with self.assertRaisesRegex(ValueError, "delivery_count"):
            rescore_episode_return(
                deviations,
                steps=50,
                delivery_count=1,
                placement_count=3,
                spec=spec,
            )


class ObjectiveAuditEnvironmentTests(unittest.TestCase):
    def _make_env(self, spec):
        return ObjectiveAuditSmallRoomsEnv(
            timing_objective=spec,
            grid_rows=6,
            grid_cols=6,
            number_blocks=1,
            arrival_rate=1.0,
            proc_mean=100,
        )

    def test_legacy_arm_is_step_exact_with_unmodified_environment(self):
        legacy = self._make_env(TimingObjectiveSpec.legacy())
        original = SmallRoomsEnv(
            grid_rows=6,
            grid_cols=6,
            number_blocks=1,
            arrival_rate=1.0,
            proc_mean=100,
        )
        instance = original.sample_episode_instance(123)
        legacy.reset(instance=instance)
        original.reset(instance=instance)
        _prepare_delivery(legacy, error=10)
        _prepare_delivery(original, error=10)

        _, audit_reward, _, info = legacy.step(legacy.ACTION_IDS["PUTDOWN"])
        _, original_reward, _, _ = original.step(
            original.ACTION_IDS["PUTDOWN"]
        )

        self.assertEqual(audit_reward, original_reward)
        self.assertEqual(audit_reward, -0.09 + 25.0)
        self.assertEqual(
            audit_reward,
            rescore_step_reward(info, LEGACY_CLIPPED),
        )

    def test_dense_arm_logs_dual_rescorable_delivery_components(self):
        spec = TimingObjectiveSpec.dense()
        env = self._make_env(spec)
        env.reset(instance=env.sample_episode_instance(456))
        # A non-delivery record is also dual-rescorable.
        _, wait_reward, _, wait_info = env.step(env.ACTION_IDS["WAIT"])
        self.assertEqual(wait_reward, -0.09)
        self.assertFalse(
            wait_info[TIMING_OBJECTIVE_INFO_KEY]["delivery_event"]
        )
        _prepare_delivery(env, error=30)

        _, dense_reward, _, delivery_info = env.step(
            env.ACTION_IDS["PUTDOWN"]
        )
        record = delivery_info[TIMING_OBJECTIVE_INFO_KEY]

        self.assertEqual(dense_reward, -10.09)
        self.assertEqual(record["delivery"]["signed_error"], 30.0)
        self.assertEqual(record["delivery"]["absolute_error"], 30.0)
        self.assertEqual(record["delivery"]["dense_excess_error"], 10.0)
        self.assertAlmostEqual(
            rescore_step_reward(delivery_info, LEGACY_CLIPPED), 9.91
        )
        self.assertAlmostEqual(
            rescore_step_reward(delivery_info, DENSE_PIECEWISE), -10.09
        )
        self.assertAlmostEqual(
            rescore_logged_return(env.objective_audit_records, LEGACY_CLIPPED),
            -0.09 + 9.91,
        )
        self.assertAlmostEqual(
            rescore_logged_return(env.objective_audit_records, DENSE_PIECEWISE),
            -0.09 - 10.09,
        )
        summary = env.objective_audit_summary()
        self.assertEqual(summary["steps"], 2)
        self.assertEqual(summary["delivery_count"], 1)
        self.assertEqual(summary["placement_count"], 0)

    def test_logged_error_can_be_rescored_with_new_dense_coefficients(self):
        env = self._make_env(TimingObjectiveSpec.dense())
        env.reset(instance=env.sample_episode_instance(789))
        _prepare_delivery(env, error=-30)
        env.step(env.ACTION_IDS["PUTDOWN"])
        alternative = TimingObjectiveSpec.dense(
            dense_b=50.0,
            lambda_abs=1.0,
            lambda_outside=0.25,
            window=20.0,
        )

        rescored = rescore_logged_return_with_spec(
            env.objective_audit_records,
            alternative,
            DENSE_PIECEWISE,
        )

        self.assertTrue(math.isclose(rescored, 17.41, abs_tol=1e-12))


if __name__ == "__main__":
    unittest.main()
