import unittest

from verify_kim2020_v3_retrieval_regression import (
    EXPECTED_COMPARISON_VERSION,
    EXPECTED_EXECUTOR_VERSION,
    EXPECTED_OPTION_VERSION,
    EXPECTED_START_CONTRACT,
    RETRIEVAL_V3_REGRESSION_CASES,
    verify_regression,
)


class Kim2020RetrievalV3RegressionTests(unittest.TestCase):
    @staticmethod
    def _artifacts():
        before_runs = []
        after_runs = []
        for method, schedule_seed, policy_seed, reason in (
            RETRIEVAL_V3_REGRESSION_CASES
        ):
            common = {
                "comparison_method": method,
                "schedule_seed": schedule_seed,
                "assignment_policy_seed": policy_seed,
            }
            before_runs.append(
                {
                    **common,
                    "method_failure_reason": (
                        "macro_failure:StrictRetrieveDeliverOption:" + reason
                    ),
                }
            )
            after_runs.append(
                {
                    **common,
                    "protocol_valid": True,
                    "strict_method_success": 1.0,
                    "method_failure_reason": None,
                    "reservation_integrity": True,
                    "illegal_drops": 0,
                    "urgency_scheduler_audit": {
                        "retrieve_deliver_option_version": (
                            EXPECTED_OPTION_VERSION
                        ),
                        "retrieval_executor_version": (
                            EXPECTED_EXECUTOR_VERSION
                        ),
                        "retrieval_start_contract": EXPECTED_START_CONTRACT,
                    },
                }
            )
        return (
            {"raw_runs": before_runs},
            {
                "config": {
                    "comparison_version": EXPECTED_COMPARISON_VERSION
                },
                "raw_runs": after_runs,
            },
        )

    def test_manifest_contains_the_eight_observed_failures(self):
        self.assertEqual(len(RETRIEVAL_V3_REGRESSION_CASES), 8)
        self.assertEqual(
            sum(case[3] == "delivery_route_failed" for case in RETRIEVAL_V3_REGRESSION_CASES),
            6,
        )
        self.assertEqual(
            sum(case[3] == "relocation_route_failed" for case in RETRIEVAL_V3_REGRESSION_CASES),
            2,
        )

    def test_exact_failure_manifest_passes_v3_contract(self):
        before, after = self._artifacts()
        verified = verify_regression(before, after)
        self.assertEqual(len(verified), 8)

    def test_verifier_rejects_a_reintroduced_route_failure(self):
        before, after = self._artifacts()
        after["raw_runs"][0]["protocol_valid"] = False
        after["raw_runs"][0]["method_failure_reason"] = (
            "macro_failure:StrictRetrieveDeliverOption:delivery_route_failed"
        )
        with self.assertRaisesRegex(AssertionError, "v3 regression failed"):
            verify_regression(before, after)


if __name__ == "__main__":
    unittest.main()
