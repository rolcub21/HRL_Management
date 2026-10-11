from __future__ import annotations

import unittest

import evaluate_vcg_unified_frozen_lambda_confirmation as confirmation
import run_vcg_unified_frozen_lambda_confirmation as repair
import train_vcg_constrained_v2_3 as v23
import train_vcg_unified as unified


class FrozenLambdaConfirmationContractTests(unittest.TestCase):
    def test_confirmation_grid_is_exact_and_disjoint_from_prior_panels(self):
        self.assertEqual(confirmation.INSTANCE_SEEDS, tuple(range(87_000, 87_030)))
        self.assertTrue(set(confirmation.INSTANCE_SEEDS).isdisjoint(range(85_000, 85_012)))
        self.assertTrue(set(confirmation.INSTANCE_SEEDS).isdisjoint(range(86_000, 86_030)))
        self.assertEqual(confirmation._policy_rng(0, 0), 624_000_000)
        self.assertEqual(confirmation._policy_rng(29, 3), 624_000_119)
        self.assertEqual(
            len(confirmation.MODEL_SEEDS)
            * len(confirmation.LAMBDAS)
            * len(confirmation.INSTANCE_SEEDS)
            * len(confirmation.RNG_INDICES),
            720,
        )

    def test_grid_repair_extends_only_coordinate_domain_and_restores(self):
        original_seeds = v23.DEFAULT_VALIDATION_SEEDS
        original_v23_rng = v23.VALIDATION_POLICY_RNG_BASE
        original_unified_rng = unified.VALIDATION_POLICY_RNG_BASE
        with repair._thirty_instance_coordinate_domain():
            with confirmation._confirmation_rng_profile():
                self.assertEqual(
                    v23.validation_policy_rng_seed(29, 3), 624_000_119
                )
        self.assertEqual(v23.DEFAULT_VALIDATION_SEEDS, original_seeds)
        self.assertEqual(v23.VALIDATION_POLICY_RNG_BASE, original_v23_rng)
        self.assertEqual(unified.VALIDATION_POLICY_RNG_BASE, original_unified_rng)

    def test_primary_decision_rule_uses_two_frozen_endpoints(self):
        self.assertEqual(confirmation.MAE_NONINFERIORITY_MARGIN, 2.0)
        self.assertAlmostEqual(confirmation.PRIMARY_T_DF29, 2.045229642132703)
        self.assertEqual(confirmation.LAMBDAS, (0.0, 0.05))


if __name__ == "__main__":
    unittest.main()
