import unittest

from projects.tfsn_consequence_value_iteration.antmaze import (
    r75_r74_twenty_episode as r75,
)


class R74TwentyEpisodeTest(unittest.TestCase):
    def test_only_budget_changes_from_r74(self):
        self.assertEqual(r75.EPISODES, 20)
        self.assertEqual(r75.HORIZON, 20)
        self.assertEqual(
            r75.PROTOCOL_VALUE["base_evaluation_revision"],
            r75.r74.METHOD_REVISION,
        )
        self.assertEqual(
            r75.PROTOCOL_VALUE["hub_route_similarity"],
            r75.r74.PROTOCOL_VALUE["hub_route_similarity"],
        )
        self.assertEqual(
            r75.PROTOCOL_VALUE["route_source_scale"],
            "R52_empirical_reach_covariance",
        )
        self.assertEqual(
            r75.PROTOCOL_VALUE["route_hub_scale"],
            "R52_empirical_reach_covariance",
        )

    def test_budget_accounting_keeps_command_contract(self):
        self.assertEqual(
            r75.PROTOCOL_VALUE["primitive_steps_per_episode"], 2000
        )
        self.assertEqual(
            r75.PROTOCOL_VALUE["maximum_primitive_steps_per_task"], 40000
        )
        self.assertEqual(
            r75.PROTOCOL_VALUE["fixed_command_duration"], 100
        )

    def test_return_blind_spot_is_declared_not_silently_changed(self):
        self.assertEqual(
            r75.PROTOCOL_VALUE["return_route_correction"],
            "none_R51_link_level_correction_only_unchanged_from_R74",
        )


if __name__ == "__main__":
    unittest.main()
