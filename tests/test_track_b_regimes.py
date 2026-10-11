import copy
import hashlib
import json
import unittest

from example.track_b_regimes import (
    BUILTIN_REGIME_SUITES,
    MIXED_OOD_V1,
    MIXED_TRAIN_V1,
    REGIME_MANIFEST_VERSION,
    STRESS_V1,
    TrackBRegime,
    canonical_manifest,
    manifest_regimes,
)


def _manifest_digest(payload):
    unsigned = {
        key: value
        for key, value in payload.items()
        if key != "manifest_sha256"
    }
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TrackBRegimeProfileTests(unittest.TestCase):
    def test_profile_constructs_exact_environment_and_provenance(self):
        regime = TrackBRegime(
            regime_id="rectangular_test",
            arrival_rate=0.55,
            proc_mean=45,
            grid_rows=9,
            grid_cols=11,
            exit_width=2,
            number_blocks=24,
        )

        env = regime.make_env()
        provenance = regime.provenance()

        self.assertEqual(env.arrival_rate, 0.55)
        self.assertEqual(env.proc_mean, 45.0)
        self.assertEqual((env.grid_rows, env.grid_cols), (9, 11))
        self.assertEqual(tuple(env.exit_cells), ((8, 8), (8, 9)))
        self.assertEqual(len(env.blocks), 24)
        self.assertEqual(
            provenance["regime_signature"], regime.regime_signature
        )
        self.assertEqual(provenance["geometry"]["grid_rows"], 9)
        self.assertEqual(provenance["geometry"]["grid_cols"], 11)
        self.assertEqual(provenance["geometry"]["actual_exit_width"], 2)
        self.assertEqual(provenance["geometry"]["block_count"], 24)
        self.assertEqual(
            TrackBRegime.from_dict(regime.to_dict()), regime
        )

    def test_builtin_train_and_ood_profiles_are_disjoint(self):
        train_ids = {item.regime_id for item in MIXED_TRAIN_V1}
        ood_ids = {item.regime_id for item in MIXED_OOD_V1}
        train_signatures = {item.regime_signature for item in MIXED_TRAIN_V1}
        ood_signatures = {item.regime_signature for item in MIXED_OOD_V1}

        self.assertEqual(len(train_ids), len(MIXED_TRAIN_V1))
        self.assertEqual(len(ood_ids), len(MIXED_OOD_V1))
        self.assertTrue(train_ids.isdisjoint(ood_ids))
        self.assertTrue(train_signatures.isdisjoint(ood_signatures))

    def test_stress_v1_profiles_match_the_sealed_stress_panel(self):
        expected = (
            ("load_5x5_b24", 0.80, 100.0, 5, 5, 1, 24),
            ("count_5x5_b52", 0.10, 100.0, 5, 5, 1, 52),
            ("egress_5x5_b40", 0.40, 100.0, 5, 5, 1, 40),
            ("compound_5x5_b64", 0.80, 100.0, 5, 5, 1, 64),
        )
        observed = tuple(
            (
                regime.regime_id,
                regime.arrival_rate,
                regime.proc_mean,
                regime.grid_rows,
                regime.grid_cols,
                regime.exit_width,
                regime.number_blocks,
            )
            for regime in STRESS_V1
        )

        self.assertEqual(observed, expected)
        self.assertIs(BUILTIN_REGIME_SUITES["stress_v1"], STRESS_V1)
        self.assertEqual(len({item.regime_id for item in STRESS_V1}), 4)
        self.assertEqual(
            len({item.regime_signature for item in STRESS_V1}), 4
        )

    def test_stress_v1_profiles_construct_the_requested_geometries(self):
        for regime in STRESS_V1:
            with self.subTest(regime=regime.regime_id):
                env = regime.make_env()
                self.assertEqual(
                    (env.grid_rows, env.grid_cols),
                    (regime.grid_rows, regime.grid_cols),
                )
                self.assertEqual(len(env.exit_cells), regime.exit_width)
                self.assertEqual(len(env.blocks), regime.number_blocks)

    def test_profile_parser_rejects_missing_and_unknown_fields(self):
        profile = MIXED_TRAIN_V1[0].to_dict()
        missing = dict(profile)
        missing.pop("proc_mean")
        unknown = {**profile, "unversioned_knob": 1}

        with self.assertRaisesRegex(ValueError, "missing=.*proc_mean"):
            TrackBRegime.from_dict(missing)
        with self.assertRaisesRegex(ValueError, "unknown=.*unversioned_knob"):
            TrackBRegime.from_dict(unknown)


class TrackBRegimeManifestTests(unittest.TestCase):
    def test_hash_is_canonical_and_json_round_trip_is_valid(self):
        regimes = MIXED_TRAIN_V1[:2]
        first = canonical_manifest(regimes)
        second = canonical_manifest(tuple(regimes))

        self.assertEqual(first, second)
        self.assertEqual(first["manifest_version"], REGIME_MANIFEST_VERSION)
        self.assertEqual(first["manifest_sha256"], _manifest_digest(first))
        # The canonical representation must be exactly what JSON persistence
        # returns; otherwise strict validation would reject a saved manifest.
        self.assertIsInstance(
            first["regimes"][0]["geometry"]["exit_cells"], list
        )
        persisted = json.loads(json.dumps(first))
        self.assertEqual(manifest_regimes(persisted), tuple(regimes))

    def test_tampered_profile_is_rejected_even_with_recomputed_outer_hash(self):
        payload = canonical_manifest(MIXED_TRAIN_V1[:2])
        tampered = copy.deepcopy(payload)
        tampered["regimes"][0]["arrival_rate"] = 0.75
        tampered["manifest_sha256"] = _manifest_digest(tampered)

        with self.assertRaisesRegex(ValueError, "modified"):
            manifest_regimes(tampered)

    def test_tampered_geometry_is_rejected_even_with_recomputed_outer_hash(self):
        payload = canonical_manifest(MIXED_TRAIN_V1[:2])
        tampered = copy.deepcopy(payload)
        tampered["regimes"][0]["geometry"]["geometry_signature"] = "0" * 16
        tampered["manifest_sha256"] = _manifest_digest(tampered)

        with self.assertRaisesRegex(ValueError, "modified"):
            manifest_regimes(tampered)

    def test_duplicate_ids_are_rejected(self):
        first = TrackBRegime("duplicate", 0.5, 50, 10, 10, None, 40)
        second = TrackBRegime("duplicate", 0.5, 50, 10, 10, 2, 40)

        with self.assertRaisesRegex(ValueError, "ids must be unique"):
            canonical_manifest((first, second))

    def test_duplicate_environment_profiles_under_aliases_are_rejected(self):
        first = TrackBRegime("first_name", 0.5, 50, 10, 10, 2, 40)
        alias = TrackBRegime("second_name", 0.5, 50, 10, 10, 2, 40)

        self.assertEqual(first.regime_signature, alias.regime_signature)
        with self.assertRaisesRegex(ValueError, "signatures must be unique"):
            canonical_manifest((first, alias))

    def test_empty_and_unsupported_manifests_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            canonical_manifest(())
        with self.assertRaisesRegex(ValueError, "unsupported"):
            manifest_regimes(
                {
                    "manifest_version": "future_version",
                    "regimes": [],
                    "manifest_sha256": "irrelevant",
                }
            )


class TrackBRegimePairingTests(unittest.TestCase):
    def test_same_workload_seed_is_paired_across_geometry_profiles(self):
        ordinary, narrow = MIXED_TRAIN_V1[:2]
        self.assertNotEqual(
            ordinary.regime_signature, narrow.regime_signature
        )

        ordinary_instance = ordinary.make_env().sample_episode_instance(63_001)
        narrow_instance = narrow.make_env().sample_episode_instance(63_001)

        self.assertEqual(
            ordinary_instance.arrival_steps, narrow_instance.arrival_steps
        )
        self.assertEqual(
            ordinary_instance.storage_steps_needed,
            narrow_instance.storage_steps_needed,
        )
        self.assertEqual(
            ordinary_instance.schedule_id, narrow_instance.schedule_id
        )
        self.assertNotEqual(
            ordinary_instance.instance_id, narrow_instance.instance_id
        )
        self.assertNotEqual(
            ordinary_instance.exit_cells, narrow_instance.exit_cells
        )


if __name__ == "__main__":
    unittest.main()
