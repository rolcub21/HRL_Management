import json
import unittest

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r32_consumptive_lookahead as r32,
)


class UnitSearchField:
    sigma = 1.0

    @staticmethod
    def search(xy):
        return np.ones(len(np.asarray(xy).reshape(-1, 2)), np.float32)


class R32ConsumptiveLookaheadTest(unittest.TestCase):
    def test_contract_is_narrow_and_development_only(self):
        protocol = r32.load_contract()
        self.assertEqual(protocol["gate_task_ids"], [3, 5])
        self.assertEqual(protocol["lookahead_depth_global"], 2)
        self.assertEqual(protocol["lookahead_beam_width_global"], 4)
        self.assertEqual(protocol["oracle_inputs"], [])
        self.assertFalse(protocol["confirmation_panel_accessed"])

    def test_consumed_field_removes_passage_and_preserves_remote_mass(self):
        passage = np.asarray([[0.0, 0.0], [1.0, 0.0]], np.float32)
        field = r32.ConsumedSearchField(UnitSearchField(), [passage], 1.0)
        value = field.search(np.asarray([[0.0, 0.0], [10.0, 0.0]]))
        self.assertAlmostEqual(float(value[0]), 0.0, places=6)
        self.assertGreater(float(value[1]), 0.999)

    def test_consumption_is_multiplicative_on_base_field(self):
        class HalfField:
            @staticmethod
            def search(xy):
                return np.full(len(np.asarray(xy).reshape(-1, 2)), 0.5, np.float32)

        field = r32.ConsumedSearchField(
            HalfField(), [np.asarray([[0.0, 0.0]], np.float32)], 1.0
        )
        expected = 0.5 * (1.0 - np.exp(-0.5))
        self.assertAlmostEqual(
            float(field.search(np.asarray([[1.0, 0.0]]))[0]), expected, places=6
        )

    def test_protocol_file_matches_loaded_contract(self):
        self.assertEqual(json.loads(r32.PROTOCOL.read_text()), r32.PROTOCOL_VALUE)


if __name__ == "__main__":
    unittest.main()
