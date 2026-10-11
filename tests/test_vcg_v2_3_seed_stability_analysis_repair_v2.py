from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest import mock

import vcg_v2_3_seed_stability as frozen
from vcg_v2_3_seed_stability_protocol import StabilityProtocolError
import vcg_v2_3_seed_stability_analysis_repair_v2 as repair


HRL_ROOT = Path("/home/ai_diagnosis/HRL_Management").resolve()
ORIGINAL_ROOT = HRL_ROOT / "results/vcg-v2-3-seed-stability-85k"
FREEZE_SPEC = HRL_ROOT / "experiments/vcg_v2_3_seed_stability_85k/freeze-spec.json"
REPAIR_V2 = HRL_ROOT / "results/vcg-v2-3-capacity-aware-ga-repair-v2-85k"


def valid_row():
    deviations = [-25, -20, 0, 5, 20, 21, -1, 0]
    return {
        "delivery_deviations": deviations,
        "required_deliveries": 8,
        "delivery_count": 8,
        "mean_absolute_error": 11.5,
    }


class TimingDerivationTests(unittest.TestCase):
    def test_sign_boundary_and_mae_contract(self):
        row = valid_row()
        enriched = repair.derive_timing_fields(row)
        self.assertNotIn("mean_tardiness", row)
        self.assertAlmostEqual(enriched["mean_tardiness"], 5.75)
        self.assertAlmostEqual(enriched["mean_earliness"], 5.75)
        self.assertAlmostEqual(enriched["within_target_window_rate"], 0.75)

    def test_malformed_or_inconsistent_sufficient_statistics_fail_closed(self):
        mutations = []
        item = valid_row(); item["delivery_deviations"] = []
        mutations.append(item)
        item = valid_row(); item["delivery_deviations"][0] = float("nan")
        mutations.append(item)
        item = valid_row(); item["delivery_deviations"][0] = True
        mutations.append(item)
        item = valid_row(); item["delivery_count"] = 7
        mutations.append(item)
        item = valid_row(); item["required_deliveries"] = 7
        mutations.append(item)
        item = valid_row(); item["mean_absolute_error"] = 11.4
        mutations.append(item)
        item = valid_row(); item["mean_tardiness"] = 5.75
        mutations.append(item)
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(StabilityProtocolError):
                repair.derive_timing_fields(mutation)

    def test_target_window_is_frozen(self):
        with self.assertRaisesRegex(StabilityProtocolError, "window"):
            repair.derive_timing_fields(valid_row(), target_window=21)


class RealArtifactIntegrationTests(unittest.TestCase):
    def kwargs(self, output_dir):
        return {
            "output_dir": output_dir,
            "original_output_dir": ORIGINAL_ROOT,
            "freeze_spec_path": FREEZE_SPEC,
            "repair_dir": REPAIR_V2,
        }

    def test_complete_append_only_repair_and_exact_frozen_gate(self):
        before = repair._original_inventory(ORIGINAL_ROOT)
        with TemporaryDirectory(dir="/tmp") as temporary:
            output = Path(temporary).resolve() / "analysis"
            contract = repair.prepare(**self.kwargs(output))
            self.assertEqual(contract, repair.prepare(**self.kwargs(output)))
            report = repair.analyze(**self.kwargs(output))
            self.assertEqual(report, repair.validate(**self.kwargs(output)))
            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                sorted((repair.CONTRACT_NAME, repair.REPORT_NAME, repair.AUDIT_NAME)),
            )
        self.assertEqual(before, repair._original_inventory(ORIGINAL_ROOT))
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["strong_result"])
        self.assertEqual(report["strict_safe_complete_selected_row_count"], 144)
        self.assertEqual(report["non_dominated_individual_seed_count"], 3)
        self.assertEqual(
            report["equal_seed_aggregate"][
                "dominators_on_MAE_and_total_physical_rehandles"
            ],
            [],
        )
        metrics = report["equal_seed_aggregate"]["metrics"]
        self.assertAlmostEqual(metrics["mean_absolute_error"], 16.82378472222222)
        self.assertAlmostEqual(
            metrics["physical_rehandles_per_100_required_deliveries"],
            7.291666666666667,
        )
        self.assertNotIn(
            "vcg_constrained_v2_3_gamma1_seed10",
            contract["frozen_baseline_method_ids"],
        )
        self.assertTrue(contract["development_seed10_is_not_a_gate_comparator"])

    def test_extra_output_entry_fails_closed(self):
        with TemporaryDirectory(dir="/tmp") as temporary:
            output = Path(temporary).resolve() / "analysis"
            repair.prepare(**self.kwargs(output))
            (output / "unexpected.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(StabilityProtocolError, "file set"):
                repair.authenticate_contract(**self.kwargs(output), phase="prepared")

    def test_self_rehashed_report_semantic_tamper_is_recomputed_and_rejected(self):
        with TemporaryDirectory(dir="/tmp") as temporary:
            output = Path(temporary).resolve() / "analysis"
            repair.prepare(**self.kwargs(output))
            repair.analyze(**self.kwargs(output))
            report_path = output / repair.REPORT_NAME
            audit_path = output / repair.AUDIT_NAME
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["status"] = "failed"
            report.pop("report_sha256")
            report["report_sha256"] = repair.digest_json(report)
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
            )
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["analysis_v2_report_sha256"] = report["report_sha256"]
            audit["analysis_v2_report_raw_sha256"] = repair.sha256_file(report_path)
            audit.pop("audit_sha256")
            audit["audit_sha256"] = repair.digest_json(audit)
            audit_path.write_text(
                json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8",
            )
            with self.assertRaisesRegex(StabilityProtocolError, "recomputed"):
                repair.validate(**self.kwargs(output))

    def test_output_root_must_be_disjoint_from_original(self):
        forbidden = (
            ORIGINAL_ROOT / "forbidden-child",
            REPAIR_V2 / "forbidden-child",
            FREEZE_SPEC.parent / "forbidden-child",
        )
        for path in forbidden:
            self.assertFalse(path.exists())
            with self.subTest(path=path), self.assertRaisesRegex(
                StabilityProtocolError, "overlaps"
            ):
                repair.build_contract(
                    output_dir=path,
                    original_output_dir=ORIGINAL_ROOT,
                    freeze_spec_path=FREEZE_SPEC,
                    repair_dir=REPAIR_V2,
                )
            self.assertFalse(path.exists())

    def test_frozen_globals_restore_on_failure(self):
        originals = (
            frozen.summarize_selected_rows,
            frozen.atomic_json,
            frozen.sha256_file,
        )
        with mock.patch.object(frozen, "analyze", side_effect=RuntimeError("sentinel")):
            with self.assertRaisesRegex(RuntimeError, "sentinel"):
                repair._capture_frozen_analysis(
                    original_output_dir=ORIGINAL_ROOT,
                    freeze_spec_path=FREEZE_SPEC,
                    repair_dir=REPAIR_V2,
                )
        self.assertIs(frozen.summarize_selected_rows, originals[0])
        self.assertIs(frozen.atomic_json, originals[1])
        self.assertIs(frozen.sha256_file, originals[2])

    def test_real_selected_rows_all_match_the_derived_schema(self):
        authenticated = repair._authenticate_inputs(
            original_output_dir=ORIGINAL_ROOT,
            freeze_spec_path=FREEZE_SPEC,
            repair_dir=REPAIR_V2,
        )
        count = 0
        for selected in authenticated["selections"].values():
            self.assertIsNotNone(selected)
            rows = selected["selected_diagnostic"]["validation_ledger"]["rows"]
            for row in rows:
                enriched = repair.derive_timing_fields(deepcopy(row))
                self.assertAlmostEqual(
                    enriched["mean_tardiness"] + enriched["mean_earliness"],
                    row["mean_absolute_error"],
                )
                count += 1
        self.assertEqual(count, 144)


if __name__ == "__main__":
    unittest.main()
