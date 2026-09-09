import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PSLAP.dynamic_yard import BlockView, YardSnapshot
from PSLAP.viability import RecoveryState
from vcg_robust_execution_probe import (
    ExecutionRealization,
    PreferenceCandidate,
    RobustExecutionStatus,
    build_probe_report,
    certify_execution_envelope,
    main,
    robust_safe_frontier,
    select_preferred_robust,
)


def direct_successor() -> RecoveryState:
    yard = YardSnapshot(
        rows=1,
        cols=3,
        traversable=frozenset({(0, 0), (0, 1), (0, 2)}),
        storage_cells=frozenset({(0, 1)}),
        exits=((0, 2),),
        blocks=(),
    )
    return RecoveryState.from_yard_snapshot(yard, (0, 2))


def realization(
    identifier: str,
    *,
    blocked=(),
    inside=True,
) -> ExecutionRealization:
    return ExecutionRealization(
        disturbance_id=identifier,
        successor_state=direct_successor(),
        trajectory_cells=((0, 0), (0, 1), (0, 2)),
        realized_primitive_steps=4,
        primitive_step_budget=4,
        transient_blocked_cells=frozenset(blocked),
        in_declared_set=inside,
    )


class RobustExecutionCertificateTests(unittest.TestCase):
    def test_complete_safe_envelope_is_admitted(self):
        certificate = certify_execution_envelope(
            "deliver:A",
            (realization("nominal"), realization("bounded_delay")),
            nominal_disturbance_id="nominal",
        )

        self.assertIs(certificate.status, RobustExecutionStatus.SAFE)
        self.assertTrue(certificate.admitted)
        self.assertEqual(certificate.nominal_audit.disturbance_id, "nominal")

    def test_any_unsafe_tube_rejects_the_whole_envelope(self):
        certificate = certify_execution_envelope(
            "deliver:A",
            (
                realization("nominal"),
                realization("blocked", blocked=((0, 1),)),
            ),
            nominal_disturbance_id="nominal",
        )

        self.assertIs(certificate.status, RobustExecutionStatus.UNSAFE)
        self.assertFalse(certificate.admitted)
        self.assertTrue(certificate.nominal_audit.tube_safe)
        blocked = certificate.in_set_realizations[1]
        self.assertFalse(blocked.tube_safe)
        self.assertEqual(blocked.successor_status.value, "SAFE")

    def test_out_of_set_failure_is_labeled_but_not_folded_into_certificate(self):
        certificate = certify_execution_envelope(
            "deliver:A",
            (realization("nominal"),),
            nominal_disturbance_id="nominal",
            out_of_set_observations=(
                realization(
                    "larger_than_declared",
                    blocked=((0, 1),),
                    inside=False,
                ),
            ),
        )

        self.assertTrue(certificate.admitted)
        self.assertIs(
            certificate.out_of_set_observations[0].status,
            RobustExecutionStatus.UNSAFE,
        )
        payload = certificate.to_dict()["out_of_set_observations"][0]
        self.assertFalse(payload["covered_by_guarantee"])
        self.assertTrue(payload["assumption_breach"])

    def test_invalid_or_ambiguous_envelopes_fail_validation(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            certify_execution_envelope(
                "deliver:A", (), nominal_disturbance_id="nominal"
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            certify_execution_envelope(
                "deliver:A",
                (realization("same"), realization("same")),
                nominal_disturbance_id="same",
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            ExecutionRealization(
                disturbance_id="bad",
                successor_state=direct_successor(),
                trajectory_cells=((0, 2),),
                realized_primitive_steps=-1,
                primitive_step_budget=0,
            )


class ProbeScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_probe_report()

    def test_nominal_safe_actions_can_fail_bounded_set_certification(self):
        nominal = self.report["cases"]["duration_boundary_nominal"]
        robust = self.report["cases"]["duration_boundary_bounded_set"]
        endpoint_nominal = self.report["cases"]["endpoint_nominal"]
        endpoint_robust = self.report["cases"][
            "endpoint_radius_one_boundary"
        ]

        self.assertEqual(nominal["robust_status"], "SAFE")
        self.assertEqual(robust["robust_status"], "UNSAFE")
        self.assertEqual(endpoint_nominal["robust_status"], "SAFE")
        self.assertEqual(endpoint_robust["robust_status"], "UNSAFE")

    def test_unknown_is_fail_closed_and_tube_checks_intermediate_states(self):
        unknown = self.report["cases"]["unknown_fail_closed"]
        tube = self.report["cases"]["trajectory_tube_obstruction"]

        self.assertEqual(unknown["robust_status"], "UNKNOWN")
        self.assertFalse(unknown["admitted"])
        self.assertEqual(tube["robust_status"], "UNSAFE")
        self.assertTrue(
            all(
                row["successor_status"] == "SAFE"
                for row in tube["in_set_realizations"]
            )
        )

    def test_preference_changes_only_inside_the_fixed_robust_frontier(self):
        result = self.report["preference_invariance"]

        self.assertEqual(
            result["robust_frontier"], ["careful_robust", "fast_robust"]
        )
        self.assertEqual(
            result["selected_by_handling_lambda"],
            {"0.0": "fast_robust", "1.0": "careful_robust"},
        )
        self.assertEqual(
            result["rejected_before_ranking"], []
        )

    def test_report_is_deterministic_and_all_mechanism_checks_pass(self):
        second = build_probe_report()

        self.assertEqual(self.report, second)
        self.assertEqual(self.report["status"], "passed")
        self.assertTrue(all(self.report["checks"].values()))
        self.assertFalse(self.report["recursive_robust_kernel_constructed"])
        self.assertFalse(self.report["real_world_validation"])


if __name__ == "__main__":
    unittest.main()
