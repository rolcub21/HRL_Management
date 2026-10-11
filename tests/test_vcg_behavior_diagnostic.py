import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from diagnose_vcg_behavior import (
    _correlation,
    _entropy,
    _softmax,
    main,
)


class VcgBehaviorDiagnosticTests(unittest.TestCase):
    def test_softmax_and_entropy_are_finite_and_normalized(self):
        probabilities = _softmax((0.1, 0.2, 0.3), 0.1)

        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertTrue(all(value > 0.0 for value in probabilities))
        self.assertTrue(math.isfinite(_entropy(probabilities)))

    def test_correlation_handles_direction_and_degenerate_inputs(self):
        self.assertAlmostEqual(_correlation((1, 2, 3), (3, 2, 1)), -1.0)
        self.assertIsNone(_correlation((1, 1, 1), (1, 2, 3)))

    def test_runner_refuses_the_predeclared_in_regime_test_panel(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "sealed test seeds"):
                main(
                    (
                        "--checkpoint",
                        str(Path(directory) / "not-opened.pth"),
                        "--seeds",
                        "77000",
                        "--output-dir",
                        str(Path(directory) / "out"),
                    )
                )


if __name__ == "__main__":
    unittest.main()
