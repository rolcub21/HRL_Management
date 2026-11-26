import unittest

from projects.tfsn_consequence_value_iteration.antmaze import (
    r48_rule_free_r39 as r48,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r51_destination_arrival_correction as r51,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r53_e549_r51_evaluation as r53,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r67_e549_hub_routed_exploration as r67,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r68_e549_dynamic_hubs as r68,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r69_e549_dynamic_hub_long_budget as r69,
)


class R69LongBudgetTest(unittest.TestCase):
    def setUp(self):
        self.r48_horizon = r48.HORIZON
        self.r48_planner_horizon = r48.RuleFreeR39Planner.horizon
        self.r51_planner_horizon = r51.DestinationArrivalOnlinePlanner.horizon
        self.r53_planner_horizon = r53.E549DestinationArrivalPlanner.horizon
        self.r67_contract = (
            r67.REFERENCE,
            r67.PROTOCOL,
            r67.METHOD_REVISION,
            r67.PROTOCOL_VALUE,
            r67._score_hubs,
        )
        self.r68_contract = (
            r68.REFERENCE,
            r68.PROTOCOL,
            r68.METHOD_REVISION,
            r68.PROTOCOL_VALUE,
            r68.DEFAULT_OUTPUT,
        )

    def tearDown(self):
        r48.HORIZON = self.r48_horizon
        r48.RuleFreeR39Planner.horizon = self.r48_planner_horizon
        r51.DestinationArrivalOnlinePlanner.horizon = self.r51_planner_horizon
        r53.E549DestinationArrivalPlanner.horizon = self.r53_planner_horizon
        (
            r67.REFERENCE,
            r67.PROTOCOL,
            r67.METHOD_REVISION,
            r67.PROTOCOL_VALUE,
            r67._score_hubs,
        ) = self.r67_contract
        (
            r68.REFERENCE,
            r68.PROTOCOL,
            r68.METHOD_REVISION,
            r68.PROTOCOL_VALUE,
            r68.DEFAULT_OUTPUT,
        ) = self.r68_contract

    def test_budget_contract(self):
        self.assertEqual(r69.PROTOCOL_VALUE["episodes"], 10)
        self.assertEqual(r69.PROTOCOL_VALUE["commands_per_episode"], 20)
        self.assertEqual(r69.PROTOCOL_VALUE["primitive_steps_per_episode"], 2000)

    def test_contract_install_changes_planning_horizon(self):
        r69._install_contract()
        self.assertEqual(r48.HORIZON, 20)
        self.assertEqual(r48.RuleFreeR39Planner.horizon, 20)
        self.assertEqual(r53.E549DestinationArrivalPlanner.horizon, 20)

    def test_budget_is_declared_deviation(self):
        deviations = r69.PROTOCOL_VALUE["declared_deviations"]
        self.assertIn(
            "physical_episode_budget_extended_from_10_to_20_fixed_duration_commands",
            deviations,
        )
        self.assertIn("episodes_per_task_extended_from_5_to_10", deviations)


if __name__ == "__main__":
    unittest.main()
