import unittest

from example.small_rooms_env import SmallRoomsEnv


class SmallRoomsPickupTests(unittest.TestCase):
    def test_pickup_carries_only_one_of_multiple_colocated_blocks(self):
        env = SmallRoomsEnv(number_blocks=2, choose_storage=False)
        env.reset(instance=env.sample_episode_instance(7))
        env.current_state = env.pickup_cell

        for block in env.blocks:
            block.position = env.pickup_cell
            block.carrying = False
            block.delivered = False

        _, _, _, info = env.step(env.ACTION_IDS["PICKUP"])

        carrying = [block for block in env.blocks if block.carrying]
        self.assertEqual([block.label for block in carrying], [env.blocks[0].label])
        self.assertTrue(env.blocks[0].picked)
        self.assertFalse(env.blocks[1].picked)
        self.assertEqual(info["picked_block"], env.blocks[0].label)


if __name__ == "__main__":
    unittest.main()
