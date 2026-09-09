from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).parents[1] / "vcg_v2_3_kim2020_supplement_protocol.py"
SPEC = importlib.util.spec_from_file_location("kim_supplement_protocol", MODULE_PATH)
protocol = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(protocol)
PROJECT_ROOT = Path("/home/ai_diagnosis/HRL_Management")


class ProtocolTests(unittest.TestCase):
    def test_training_ranges(self):
        self.assertEqual(protocol.training_schedule_range(0), (70000001, 70001000))
        self.assertEqual(protocol.training_schedule_range(1), (71000001, 71001000))
        self.assertEqual(protocol.training_schedule_range(2), (72000001, 72001000))

    def test_exact_180_row_grid_and_unique_fresh_rngs(self):
        rows = protocol.expected_evaluation_grid()
        self.assertEqual(len(rows), 180)
        self.assertEqual(len({row["policy_seed"] for row in rows}), 180)
        self.assertEqual(rows[0]["policy_seed"], 632000000)
        self.assertEqual(rows[-1]["policy_seed"], 632002114)
        self.assertFalse(any(622000000 <= row["policy_seed"] <= 622999999 for row in rows))

    def test_dynamic_pslap_display_label(self):
        self.assertEqual(protocol.DYNAMIC_PSLAP_METHOD_ID, "duration_aware_dynamic_pslap")
        self.assertEqual(protocol.DYNAMIC_PSLAP_DISPLAY_LABEL, "Duration-aware dynamic PSLAP")

    def test_no_final_panel_seed(self):
        used = set(protocol.FINAL_EVAL_INSTANCE_SEEDS)
        self.assertFalse(used.intersection(protocol.SEALED_INSTANCE_SEEDS))

    def test_training_command_omits_exit_width_and_resume(self):
        command = protocol.training_command(Path("/project"), Path("/output"), 1)
        self.assertNotIn("--exit-width", command)
        self.assertNotIn("--resume", command)
        self.assertEqual(command[command.index("--stochastic-rollouts") + 1], "5")
        self.assertEqual(command[command.index("--number-blocks") + 1], "8")

    def test_real_prepare_validate_and_idempotence(self):
        if MODULE_PATH.resolve() != (
            PROJECT_ROOT / "vcg_v2_3_kim2020_supplement_protocol.py"
        ).resolve():
            self.skipTest("integration test requires installed protocol path")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "supplement"
            first = protocol._write_manifest(PROJECT_ROOT, output)
            before = first.read_bytes()
            second = protocol._write_manifest(PROJECT_ROOT, output)
            self.assertEqual(first, second)
            self.assertEqual(before, second.read_bytes())
            manifest = protocol._load_json(first)
            protocol.validate_manifest(manifest, PROJECT_ROOT, output)
            protocol._validate_output_tree(output)

    def test_output_tree_injection_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "supplement"
            output.mkdir()
            (output / "supplement-run-manifest.json").write_text("{}")
            (output / "unexpected.txt").write_text("x")
            with self.assertRaises(protocol.ProtocolError):
                protocol._validate_output_tree(output)

    def test_manifest_symlink_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "supplement"
            output.mkdir()
            target = Path(temporary) / "target.json"
            target.write_text("{}")
            (output / "supplement-run-manifest.json").symlink_to(target)
            with self.assertRaises(protocol.ProtocolError):
                protocol._validate_output_tree(output)

    def test_frozen_source_drift_fails(self):
        if MODULE_PATH.resolve() != (
            PROJECT_ROOT / "vcg_v2_3_kim2020_supplement_protocol.py"
        ).resolve():
            self.skipTest("integration test requires installed protocol path")
        drifted = dict(protocol.EXPECTED_EXISTING_SOURCE_SHA256)
        first = next(iter(drifted))
        drifted[first] = "0" * 64
        with mock.patch.object(protocol, "EXPECTED_EXISTING_SOURCE_SHA256", drifted):
            with self.assertRaises(protocol.ProtocolError):
                protocol._source_hashes(PROJECT_ROOT)

    def test_run_wrapper_has_atomic_seed_reservation(self):
        wrapper = MODULE_PATH.parent / "experiments/vcg_v2_3_kim2020_supplement_85k/run.sh"
        if not wrapper.is_file():
            self.skipTest("integration test requires installed run wrapper")
        text = wrapper.read_text()
        self.assertIn('if ! mkdir "${seed_dir}"', text)
        self.assertNotIn("--resume", text)


if __name__ == "__main__":
    unittest.main()
