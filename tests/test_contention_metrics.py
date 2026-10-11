import unittest

from contention_metrics import (
    CONTENTION_METRIC_SCHEMA_VERSION,
    contention_metric_record,
    validate_contention_metric_record,
)


class ContentionMetricContractTests(unittest.TestCase):
    def test_lossless_vcg_partition_and_legacy_aliases(self):
        record = contention_metric_record(
            physical_storage_relocations=4,
            target_bound_obstruction_clearances=0,
            standalone_reconfigurations=4,
            standalone_with_direct_delivery_available=3,
            standalone_without_direct_delivery_available=1,
            directly_deliverable_self_reconfigurations=2,
        )
        with_aliases = {
            **record,
            "relocations": 4,
            "obstructive_moves": 0,
        }

        normalized = validate_contention_metric_record(
            with_aliases, require_legacy_alias=True
        )

        self.assertEqual(
            normalized["contention_metric_schema_version"],
            CONTENTION_METRIC_SCHEMA_VERSION,
        )
        self.assertEqual(normalized["physical_storage_relocations"], 4)

    def test_partition_identity_fails_closed(self):
        valid = contention_metric_record(
            physical_storage_relocations=2,
            target_bound_obstruction_clearances=1,
            standalone_reconfigurations=1,
            standalone_with_direct_delivery_available=1,
            standalone_without_direct_delivery_available=0,
            directly_deliverable_self_reconfigurations=1,
        )
        cases = (
            (
                {**valid, "physical_storage_relocations": 3},
                "must equal",
            ),
            (
                {
                    **valid,
                    "standalone_with_direct_delivery_available": 0,
                },
                "partitioned",
            ),
            (
                {
                    **valid,
                    "directly_deliverable_self_reconfigurations": 2,
                },
                "must be a subset",
            ),
            (
                {
                    **valid,
                    "relocations": 3,
                    "obstructive_moves": 1,
                },
                "exact alias",
            ),
        )
        for record, message in cases:
            with self.subTest(record=record):
                with self.assertRaisesRegex(ValueError, message):
                    validate_contention_metric_record(
                        record, require_legacy_alias="relocations" in record
                    )


if __name__ == "__main__":
    unittest.main()
