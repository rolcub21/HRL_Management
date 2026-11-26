import unittest

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r78_soft_return_route_audit as r78,
)


class SoftReturnRouteAuditTest(unittest.TestCase):
    def test_no_evidence_is_bitwise_identity_even_above_one(self):
        base = np.asarray([[0.0, 0.0], [3.0, 2.0]], np.float32)
        corrected, reliability, support, arrivals = (
            r78.multiplicative_route_correction(
                base,
                [[0.0, 0.0], [1.0, 0.0]],
                [2.0, 0.0],
                [],
                np.eye(2),
            )
        )
        np.testing.assert_array_equal(corrected, base)
        np.testing.assert_array_equal(reliability, np.ones_like(base))
        self.assertFalse(np.any(support))
        self.assertFalse(np.any(arrivals))

    def test_failure_scales_additive_value_without_clipping(self):
        base = np.asarray([[0.0, 0.0], [3.0, 2.0]], np.float32)
        corrected, reliability, support, _arrivals = (
            r78.multiplicative_route_correction(
                base,
                [[0.0, 0.0], [3.0, 0.0]],
                [4.0, 0.0],
                [{
                    "source_xy": [0.0, 0.0],
                    "target_xy": [4.0, 0.0],
                    "remaining_commands": 1,
                    "arrived": False,
                }],
                np.eye(2),
            )
        )
        self.assertAlmostEqual(float(support[1, 0]), 1.0)
        self.assertAlmostEqual(float(reliability[1, 0]), 0.5)
        self.assertAlmostEqual(float(corrected[1, 0]), 1.5)
        self.assertGreater(float(corrected[1, 1]), 0.0)
        self.assertGreater(float(reliability[1, 1]), float(reliability[1, 0]))


if __name__ == "__main__":
    unittest.main()
