import json
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import compare_vcg_v2_3_capacity_aware_ga_repair as repair_v1
import compare_vcg_v2_3_capacity_aware_ga_repair_v2 as repair_v2
import compare_vcg_v2_3_matched_baselines as base
from PSLAP.ga_capacity_aware import (
    CapacityAwareCompleteRollingGAAssigner,
    CapacityAwareDurationAwareRollingGAAssigner,
    CapacityAwareOperationalRollingGAAssigner,
    CapacityAwareRollingGAStorageAssigner,
)
from PSLAP.ga_optimizer import GAConfig
from PSLAP.neutral_protocol import shared_candidate_mask
from tests.test_capacity_aware_rolling_ga import _make_env, _view, _yard


_ASSIGNER_BY_METHOD = {
    repair_v2.REPAIRED_2009_METHOD: CapacityAwareRollingGAStorageAssigner,
    repair_v2.REPAIRED_DURATION_METHOD:
        CapacityAwareDurationAwareRollingGAAssigner,
    repair_v2.REPAIRED_OPERATIONAL_METHOD:
        CapacityAwareOperationalRollingGAAssigner,
    repair_v2.REPAIRED_COMPLETE_METHOD:
        CapacityAwareCompleteRollingGAAssigner,
}


def _executed_source_audit(method, seed=85_000):
    env = _make_env(5)
    yard = _yard(env)
    block = _view(env)
    candidates = shared_candidate_mask(yard, block, env.pickup_cell)
    source = _ASSIGNER_BY_METHOD[method](
        env,
        GAConfig(
            population_size=base.ROLLING_POPULATION,
            generations=base.ROLLING_GENERATIONS,
            elite_count=2,
            tournament_size=3,
            crossover_rate=0.8,
            mutation_rate=0.1,
            seed=base.GA_SEED_BASE + seed,
        ),
        egress_weight=base.ROLLING_GA_EGRESS_WEIGHT,
    )
    chosen = source.preview(yard, block, env.pickup_cell, candidates)
    mask_id = repair_v2._candidate_mask_sha(candidates)[:16]
    metadata = {
        "proposal_id": f"v2-{method}",
        "block_label": block.label,
        "chosen_cell": chosen,
        "selection_time_step": env.time_steps,
        "candidate_count": len(candidates),
        "candidate_mask_id": mask_id,
    }
    source.on_preview_token_minted(**metadata)
    source.on_preview_reserved(**metadata)
    commit = {**metadata, "commit_time_step": env.time_steps + 2}
    source.validate_bound_assignment_commit(**commit)
    source.on_bound_assignment_committed(**commit)
    source_audit = source.audit()
    selector_audit = {
        "assignment_source": repair_v2.REPAIRED_METHOD_TO_SOURCE[method],
        "assignment_source_version": repair_v2.REPAIRED_METHOD_TO_VERSION[
            method
        ],
        "valid_assignment_count": 1,
        "reserved_commit_count": 1,
        "decisions": [
            {
                "proposal_id": metadata["proposal_id"],
                "valid": True,
                "block_label": block.label,
                "chosen_cell": chosen,
                "selection_time_step": env.time_steps,
                "commit_time_step": env.time_steps + 2,
                "candidate_count": len(candidates),
                "candidate_mask_id": mask_id,
            }
        ],
        "source_audit": source_audit,
    }
    return {
        "assignment_source": selector_audit["assignment_source"],
        "assignment_source_version": selector_audit[
            "assignment_source_version"
        ],
        "reservation_integrity": True,
        "delivery_count": 1,
        "selector_audit": selector_audit,
        "scheduler_audit": {
            "inbound_successes": 1,
            "reservation_bound_count": 1,
            "reservation_commit_count": 1,
            "reservation_execution_match_count": 1,
            "reservation_invalidation_count": 0,
            "bound_proposal_ids": [metadata["proposal_id"]],
            "committed_proposal_ids": [metadata["proposal_id"]],
        },
    }


class CapacityAwareRepairV2LifecycleTests(unittest.TestCase):
    def test_v1_lifecycle_and_source_identities_remain_unchanged(self):
        self.assertEqual(repair_v1.SCHEMA_VERSION, 1)
        self.assertTrue(repair_v1.PROTOCOL.endswith("_v1"))
        self.assertEqual(repair_v2.SCHEMA_VERSION, 2)
        self.assertTrue(repair_v2.PROTOCOL.endswith("_v2"))
        self.assertEqual(
            repair_v2.REPAIRED_METHOD_TO_SOURCE,
            repair_v1.REPAIRED_METHOD_TO_SOURCE,
        )
        self.assertEqual(
            repair_v2.REPAIRED_METHOD_TO_VERSION,
            repair_v1.REPAIRED_METHOD_TO_VERSION,
        )
        self.assertEqual(
            repair_v2._build_parser().parse_args([]).output_dir.name,
            repair_v2.DEFAULT_OUTPUT_DIRECTORY,
        )

    def test_execute_serialize_reload_accepts_all_four_objectives(self):
        """Exercise source execution and the exact JSON reload boundary."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for method in repair_v2.REPAIRED_METHODS:
                with self.subTest(method=method):
                    raw = _executed_source_audit(method)
                    self.assertEqual(
                        repair_v2._capacity_audit_issues(
                            raw,
                            method=method,
                            seed=85_000,
                            expected_pickup_cell=(0, 0),
                            expected_delivery_count=1,
                        ),
                        [],
                    )
                    path = root / f"{method}.json"
                    repair_v2._atomic_json(path, raw)
                    reloaded = repair_v2._load_json(path)
                    self.assertEqual(
                        repair_v2._capacity_audit_issues(
                            reloaded,
                            method=method,
                            seed=85_000,
                            expected_pickup_cell=(0, 0),
                            expected_delivery_count=1,
                        ),
                        [],
                    )

    def test_successful_full_ledger_write_then_reload_for_all_objectives(self):
        """Exercise the runner's atomic-ledger lifecycle, not just raw audit."""

        project = Path(__file__).resolve().parents[1]
        provisional = (
            project / "results/vcg-v2-3-capacity-aware-ga-repair-85k"
        )
        seed = 85_000
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for method in repair_v2.REPAIRED_METHODS:
                with self.subTest(method=method):
                    source_path = repair_v1._repair_ledger_path(
                        provisional, method, seed
                    )
                    source_ledger = json.loads(source_path.read_text())
                    row = copy.deepcopy(source_ledger["run"])
                    row["protocol"] = repair_v2.PROTOCOL
                    expected_lex = repair_v2._expected_lexicographic_fields(
                        method
                    )
                    for decision in row["capacity_aware_source_audit"][
                        "decisions"
                    ]:
                        components = decision["best_cost_components"]
                        decision["best_cost_components"] = {
                            field: components[field] for field in expected_lex
                        }
                    identity = {
                        "episode_instance_id": row["episode_instance_id"],
                        "schedule_id": row["schedule_id"],
                        "episode_instance_sha256": row[
                            "episode_instance_sha256"
                        ],
                    }
                    identities = {seed: identity}
                    input_contract = {
                        "episode_instance_sha256": identity[
                            "episode_instance_sha256"
                        ],
                        "v2_serialization_regression": True,
                    }
                    with mock.patch.object(
                        repair_v2._impl,
                        "_load_json",
                        wraps=repair_v2._impl._load_json,
                    ) as load_spy:
                        executed = repair_v2._load_or_execute_repaired(
                            output_dir=output,
                            method=method,
                            seed=seed,
                            input_contract=input_contract,
                            identities=identities,
                            execute=True,
                            executor=lambda row=row: copy.deepcopy(row),
                        )
                    self.assertEqual(load_spy.call_count, 1)
                    returned_components = executed[
                        "capacity_aware_source_audit"
                    ]["decisions"][0]["best_cost_components"]
                    self.assertEqual(
                        tuple(returned_components),
                        tuple(sorted(returned_components)),
                    )
                    reloaded = repair_v2._load_or_execute_repaired(
                        output_dir=output,
                        method=method,
                        seed=seed,
                        input_contract=input_contract,
                        identities=identities,
                        execute=False,
                        executor=lambda: self.fail("reload executed policy"),
                    )
                    self.assertEqual(executed, reloaded)

    def test_exact_key_set_and_separate_lexicographic_order_fail_closed(self):
        raw = _executed_source_audit(repair_v2.REPAIRED_DURATION_METHOD)
        record = raw["selector_audit"]["source_audit"]["decisions"][0]
        record["best_cost_components"]["unexpected"] = 0
        issues = repair_v2._capacity_audit_issues(
            raw,
            method=repair_v2.REPAIRED_DURATION_METHOD,
            seed=85_000,
            expected_pickup_cell=(0, 0),
            expected_delivery_count=1,
        )
        self.assertIn("decision[0]:cost_component_key_set", issues)

        raw = _executed_source_audit(repair_v2.REPAIRED_DURATION_METHOD)
        raw["selector_audit"]["source_audit"]["decisions"][0][
            "objective_lexicographic_fields"
        ] = tuple(reversed(repair_v2.DURATION_LEX_FIELDS))
        issues = repair_v2._capacity_audit_issues(
            raw,
            method=repair_v2.REPAIRED_DURATION_METHOD,
            seed=85_000,
            expected_pickup_cell=(0, 0),
            expected_delivery_count=1,
        )
        self.assertTrue(any("lex_fields" in issue for issue in issues), issues)

    def test_v2_contract_declares_serialization_only_lifecycle_change(self):
        root = Path(__file__).resolve().parents[1]
        sources = base.authenticate_sources(
            v23_source_dir=root
            / "results/vcg-constrained-v2-3-gamma1-ablation-seed10-200ep",
            v11_control_dir=root
            / "results/vcg-dense-v1-1-v2-2-panel-control-12instance",
            v22_source_dir=root
            / "results/vcg-constrained-v2-2-development-seed10-200ep",
        )
        matched = repair_v2.authenticate_completed_matched_comparison(
            root / "results/vcg-v2-3-matched-baselines-85k",
            sources=sources,
        )
        contract = repair_v2.build_repair_contract(
            sources=sources,
            matched=matched,
            device="cpu",
        )
        self.assertEqual(contract["schema_version"], 2)
        self.assertEqual(contract["protocol"], repair_v2.PROTOCOL)
        lifecycle = contract["artifact_validation_v2"]
        self.assertFalse(lifecycle["policy_or_assignment_source_change"])
        self.assertFalse(
            lifecycle["cost_component_mapping_order_has_semantics"]
        )
        self.assertTrue(lifecycle["exact_cost_component_key_set_required"])
        self.assertIn(
            "compare_vcg_v2_3_capacity_aware_ga_repair.py",
            contract["current_repair_source_sha256"],
        )
        self.assertIn(
            "compare_vcg_v2_3_capacity_aware_ga_repair_v2.py",
            contract["current_repair_source_sha256"],
        )
        self.assertFalse(contract["final_86xxx_panel_opened"])


if __name__ == "__main__":
    unittest.main()
