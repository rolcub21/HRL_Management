from __future__ import annotations

import unittest

import numpy as np

from .environment import dense_directional_reward
from .prepare_bank import derive_bank


class R22ContractTest(unittest.TestCase):
    def test_bank_uses_every_exact_in_trajectory_transition(self):
        observation = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 4.0, 0.0],
                [10.0, 9.0, 0.0],
            ],
            dtype=np.float32,
        )
        terminal = np.asarray([False, False, True, False, False, True])
        bank = derive_bank(observation, terminal, command_steps=1)
        np.testing.assert_allclose(bank["requested_distance_m"], [1, 2, 4, 5])
        np.testing.assert_allclose(bank["reset_xy"], [[0, 0], [10, 0]])

    def test_dense_reward_is_e549_equation(self):
        previous = np.asarray([1.0, 2.0])
        achieved = np.asarray([1.1, 1.8])
        desired = np.asarray([4.0, 3.0])
        action = np.asarray([0.2] * 8)
        dt = 0.1
        velocity = (achieved - previous) / dt
        direction = desired - achieved
        expected = (
            10.0 * np.dot(velocity, direction) / np.linalg.norm(direction)
            - 0.5 * np.sum(np.square(action))
            + 1.0
        )
        self.assertAlmostEqual(
            dense_directional_reward(previous, achieved, desired, action, dt),
            expected,
            places=14,
        )


if __name__ == "__main__":
    unittest.main()
