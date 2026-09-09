from contextlib import redirect_stdout
import io
from pathlib import Path
import random
import tempfile
import unittest

import torch

from run_vcg_v2_1_paired_policy_realization import (
    EXPECTED_SOURCE_CHECKPOINT_SHA256,
    PANEL_SEEDS,
    PairedPolicyAuditError,
    FrozenPairedPolicyRuntime,
    _tensor_state_sha256,
    _transactional_directory,
    authenticate_source,
    main,
    materialize_instances,
    soft_policy_rng_seed,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_CHECKPOINT = (
    REPOSITORY
    / "results"
    / "vcg-constrained-v2-1-development-seed0-100ep"
    / "latest.pth"
)


class FrozenPolicyRealizationProtocolTests(unittest.TestCase):
    def test_controller_rng_namespace_is_exact_and_disjoint(self):
        observed = {
            soft_policy_rng_seed(instance_index, replicate)
            for instance_index in range(3)
            for replicate in range(32)
        }
        self.assertEqual(len(observed), 96)
        self.assertEqual(soft_policy_rng_seed(0, 0), 910_000)
        self.assertEqual(soft_policy_rng_seed(0, 31), 910_031)
        self.assertEqual(soft_policy_rng_seed(1, 0), 910_100)
        self.assertEqual(soft_policy_rng_seed(2, 31), 910_231)
        with self.assertRaises(PairedPolicyAuditError):
            soft_policy_rng_seed(-1, 0)
        with self.assertRaises(PairedPolicyAuditError):
            soft_policy_rng_seed(0, 32)
        with self.assertRaises(PairedPolicyAuditError):
            soft_policy_rng_seed(True, 0)

    def test_transactional_output_rolls_back_on_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            output = parent / "audit"

            def fail(staging):
                (staging / "partial.json").write_text("partial", encoding="utf-8")
                raise RuntimeError("synthetic failure")

            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                _transactional_directory(output, fail)
            self.assertFalse(output.exists())
            self.assertEqual(list(parent.glob(".audit.tmp-*")), [])


@unittest.skipUnless(SOURCE_CHECKPOINT.is_file(), "frozen V2.1 source is absent")
class FrozenPolicyRealizationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = authenticate_source(SOURCE_CHECKPOINT)

    def test_latest_source_is_hash_and_semantics_authenticated(self):
        self.assertEqual(
            self.source.checkpoint_sha256, EXPECTED_SOURCE_CHECKPOINT_SHA256
        )
        self.assertEqual(self.source.dual_lambda, 0.147)
        self.assertEqual(
            tuple(self.source.map_schedule["within_group_temperatures"]),
            (0.01, 0.01, 0.01, 0.01),
        )
        self.assertEqual(self.source.map_schedule["group_temperature"], 0.05)
        self.assertEqual(self.source.map_schedule["policy_mode"], "map")
        self.assertEqual(
            self.source.soft_schedule["policy_mode"], "regularized_sample"
        )

    def test_frozen_clone_seeds_only_action_sampling_rng(self):
        runtime = FrozenPairedPolicyRuntime(source=self.source, device="cpu")
        nuisance_seed = 910_123
        torch.manual_seed(77_123)
        torch_state_before = torch.random.get_rng_state().clone()
        clone = runtime._new_frozen_agent(
            policy="induced_soft", policy_rng_seed=nuisance_seed
        )
        torch_state_after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(torch_state_before, torch_state_after))
        self.assertEqual(clone.seed, int(self.source.contract["model_seed"]))
        self.assertEqual(clone.audit_policy_rng_seed, nuisance_seed)
        expected_rng = random.Random(nuisance_seed)
        self.assertEqual(clone.rng.random(), expected_rng.random())
        self.assertEqual(
            _tensor_state_sha256(clone.Q_local.state_dict()),
            self.source.q_local_sha256,
        )
        self.assertFalse(clone.Q_local.training)
        self.assertFalse(any(parameter.requires_grad for parameter in clone.Q_local.parameters()))
        self.assertEqual(clone.schedule_state["policy_mode"], "regularized_sample")
        with self.assertRaisesRegex(RuntimeError, "forbids learning"):
            clone.learn()
        with self.assertRaisesRegex(RuntimeError, "forbids replay"):
            clone.remember(None)

    def test_frozen_instance_manifest_is_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            instances, manifest = materialize_instances(
                self.source, Path(temporary) / "instances"
            )
            self.assertEqual(tuple(instances), PANEL_SEEDS)
            self.assertEqual(tuple(manifest["episode_instance_seeds"]), PANEL_SEEDS)
            self.assertTrue(manifest["reused_development_panel"])
            self.assertFalse(manifest["prospective_83xxx_panel_opened"])
            for index, seed in enumerate(PANEL_SEEDS):
                record = manifest["instances"][str(seed)]
                self.assertEqual(record["instance_index"], index)
                self.assertEqual(record["instance_seed"], seed)
                self.assertEqual(record["instance_id"], instances[seed].instance_id)
                self.assertEqual(record["schedule_id"], instances[seed].schedule_id)
                self.assertEqual(len(record["serialized_sha256"]), 64)

    def test_cli_without_execute_authenticates_but_creates_no_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-exist"
            stream = io.StringIO()
            with redirect_stdout(stream):
                main(
                    [
                        "--checkpoint",
                        str(SOURCE_CHECKPOINT),
                        "--output-dir",
                        str(output),
                    ]
                )
            self.assertFalse(output.exists())
            self.assertIn('"rows_executed": 0', stream.getvalue())
            self.assertIn('"explicit_execute_required": true', stream.getvalue())


if __name__ == "__main__":
    unittest.main()
