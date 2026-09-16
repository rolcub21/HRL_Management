import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

import run_vcg_objective_gamma_audit as objective_audit
from compare_vcg_dense_pareto import (
    BASELINE_METHODS,
    EVALUATION_SEEDS,
    FINAL_DIAGNOSTIC,
    FROZEN_GAMMA,
    FROZEN_MAX_STEPS,
    FROZEN_OBJECTIVE_SPEC,
    FROZEN_TOTAL_EPISODES,
    FROZEN_TRAIN_SEED_ORIGIN,
    FROZEN_TRAIN_SEED_STRIDE,
    FROZEN_VALIDATION_SEEDS,
    LATEST_DIAGNOSTIC_SEEDS,
    METHOD_VERSION,
    MODEL_SEEDS,
    SELECTED_BEST,
    TRAINING_PROTOCOL,
    _derive_exact_duplicate_rows,
    _execution_plan,
    _finish_row,
    _load_or_execute_run,
    _load_training_bundles,
)
from viability_graph_episodic_audit import (
    EpisodicViabilityGraphConfig,
    EpisodicViabilityGraphHierarchyAgent,
)


def _contract(model_seed, config):
    return {
        "training_protocol": TRAINING_PROTOCOL,
        "method_version": METHOD_VERSION,
        "trainer_schema_version": 1,
        "frozen_recipe_source": "vcg_objective_gamma_factorial_development_screen_v1",
        "objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
        "gamma": FROZEN_GAMMA,
        "model_seed": model_seed,
        "total_episodes": FROZEN_TOTAL_EPISODES,
        "train_instance_seed_base": (
            FROZEN_TRAIN_SEED_ORIGIN
            + model_seed * FROZEN_TRAIN_SEED_STRIDE
        ),
        "validation_instance_seeds": FROZEN_VALIDATION_SEEDS,
        "environment": {
            "grid_rows": 5,
            "grid_cols": 5,
            "number_blocks": 8,
            "arrival_rate": 10.0,
            "proc_mean": 80,
            "max_steps": FROZEN_MAX_STEPS,
        },
        "graph_config": config.to_dict(),
        "search_config": {
            "max_depth": None,
            "max_nodes": 20_000,
            "max_primitive_steps": None,
            "reserve_queue_cells": True,
            "search_order": "goal_directed",
        },
        "liveness_rule": {
            "max_option_steps": 10,
            "max_consecutive_defer_decisions": 16,
            "version": "bounded_observable_event_or_positive_deadline_defer_v1",
        },
        "updates_per_macro": 1,
        "epsilon_schedule": {
            "start": 0.9,
            "end": 0.05,
            "warmup_decisions": 0,
            "decay_decisions": 3_000,
        },
        "validation_every_episodes": 25,
        "checkpoint_every_episodes": 25,
        "checkpoint_selection_version": (
            "strict_completion_dense_return_mae_relocation_earlier_lexicographic_v1"
        ),
        "exact_verifier_authoritative": True,
        "viability_critic_enabled": False,
        "baseline_teacher": False,
        "baseline_policy_query": False,
        "future_schedule_visible_to_policy": False,
        "terminal_boundary_contract": (
            "completion_or_time_limit_is_terminal_no_bootstrap_at_declared_horizon_v1"
        ),
        "dual_rescore_fixed_realized_trajectory": True,
        "sealed_test_panels_opened": False,
    }


def _metadata(model_seed, contract, role, selected_episode, best_sha=None):
    return {
        "training_protocol": TRAINING_PROTOCOL,
        "method_version": METHOD_VERSION,
        "trainer_schema_version": 1,
        "trainer_resumable": role == "latest_resumable",
        "checkpoint_role": role,
        "model_seed": model_seed,
        "completed_training_episodes": FROZEN_TOTAL_EPISODES,
        "total_training_episodes": FROZEN_TOTAL_EPISODES,
        "resume_contract": contract,
        "resume_contract_sha256": objective_audit._contract_hash(contract),
        "timing_objective": "dense_piecewise_v1",
        "timing_objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
        "proper_training_confirmation_candidate": True,
        "protocol_training_complete": True,
        "selection_finalized_after_total_episodes": True,
        "selected_checkpoint_episode": selected_episode,
        "deployment_checkpoint_eligible": role == "best_deployment_finalized",
        "development_mechanism_screen_only": False,
        "performance_claim_authorized": False,
        "test_panels_opened": False,
        "future_schedule_visible_to_policy": False,
        "exact_full": True,
        "viability_critic_enabled": False,
        "best_validation_record": {
            "checkpoint_episode": selected_episode,
            "deployment_eligible": True,
            "selection_score": (1.0, 1.0, 20.0, -10.0, -20.0, -selected_episode),
            "selection_score_fields": (
                "strict_success_rate",
                "completion_rate",
                "selected_objective_return_per_block",
                "negative_mean_absolute_error",
                "negative_relocations_per_100_deliveries",
                "negative_checkpoint_episode",
            ),
        },
        "best_candidate_checkpoint_sha256": "a" * 64,
        "best_checkpoint_sha256": best_sha,
    }


def _write_bundle(root: Path, model_seed: int, selected_episode: int):
    directory = root / f"seed-{model_seed}"
    directory.mkdir()
    config = EpisodicViabilityGraphConfig(
        gamma=FROZEN_GAMMA, episode_horizon_steps=FROZEN_MAX_STEPS
    )
    contract = _contract(model_seed, config)
    best_agent = EpisodicViabilityGraphHierarchyAgent(
        config=config, seed=model_seed, device="cpu"
    )
    best = best_agent.checkpoint(
        include_replay=False,
        **_metadata(
            model_seed,
            contract,
            "best_deployment_finalized",
            selected_episode,
        ),
    )
    best_path = directory / "best.pth"
    torch.save(best, best_path)
    best_sha = objective_audit._sha256(best_path)

    if model_seed == 0:
        final_agent = best_agent
    else:
        final_agent = EpisodicViabilityGraphHierarchyAgent(
            config=config, seed=100 + model_seed, device="cpu"
        )
    latest = final_agent.checkpoint(
        include_replay=False,
        **_metadata(
            model_seed,
            contract,
            "latest_resumable",
            selected_episode,
            best_sha=best_sha,
        ),
    )
    torch.save(latest, directory / "latest.pth")
    contract_hash = objective_audit._contract_hash(contract)
    (directory / "training-contract.json").write_text(
        json.dumps(
            {
                "training_protocol": TRAINING_PROTOCOL,
                "method_version": METHOD_VERSION,
                "resume_contract_sha256": contract_hash,
                "contract": objective_audit._json_safe(contract),
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "status": "complete",
        "training_protocol": TRAINING_PROTOCOL,
        "method_version": METHOD_VERSION,
        "model_seed": model_seed,
        "completed_training_episodes": FROZEN_TOTAL_EPISODES,
        "protocol_training_complete": True,
        "selection_finalized_after_total_episodes": True,
        "deployment_checkpoint_eligible": True,
        "gamma": FROZEN_GAMMA,
        "test_panels_opened": False,
        "performance_claim_authorized": False,
        "objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
        "resume_contract": objective_audit._json_safe(contract),
        "resume_contract_sha256": contract_hash,
        "best_checkpoint_sha256": best_sha,
        "best_candidate_checkpoint_sha256": "a" * 64,
        "best_validation_record": objective_audit._json_safe(
            best["best_validation_record"]
        ),
    }
    (directory / "training-summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    return directory


def _bundles(root):
    return [
        _write_bundle(root, 0, 500),
        _write_bundle(root, 1, 450),
        _write_bundle(root, 2, 375),
    ]


class VcgDenseParetoRunnerTests(unittest.TestCase):
    def test_frozen_protocol_has_thirty_unopened_development_seed_identifiers(self):
        # This test inspects integer identifiers only; it never samples or saves
        # an EpisodeInstance and therefore does not open the panel.
        self.assertEqual(EVALUATION_SEEDS, tuple(range(80_000, 80_030)))
        self.assertEqual(len(BASELINE_METHODS), 4)
        self.assertEqual(LATEST_DIAGNOSTIC_SEEDS, (1, 2))

    def test_authenticates_three_bundles_and_omits_duplicate_seed0_execution(self):
        with TemporaryDirectory() as temporary:
            bundles = _load_training_bundles(_bundles(Path(temporary)))

        self.assertEqual(tuple(bundle.model_seed for bundle in bundles), MODEL_SEEDS)
        self.assertEqual(
            tuple(bundle.selected.policy_group for bundle in bundles),
            (SELECTED_BEST,) * 3,
        )
        self.assertIsNone(bundles[0].final_diagnostic)
        self.assertEqual(
            bundles[0].selected.deployment_policy_digest,
            bundles[0].authenticated_final.deployment_policy_digest,
        )
        self.assertEqual(
            tuple(bundle.final_diagnostic.policy_group for bundle in bundles[1:]),
            (FINAL_DIAGNOSTIC, FINAL_DIAGNOSTIC),
        )
        seed_zero_final = bundles[0].checkpoint_records[1]
        self.assertFalse(seed_zero_final["executed"])
        self.assertTrue(seed_zero_final["analysis_alias"])

    def test_bundle_hash_tampering_fails_closed(self):
        with TemporaryDirectory() as temporary:
            directories = _bundles(Path(temporary))
            path = directories[1] / "training-summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary["best_checkpoint_sha256"] = "0" * 64
            path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "best checkpoint SHA-256"):
                _load_training_bundles(directories)

    def test_frozen_contract_drift_fails_closed(self):
        with TemporaryDirectory() as temporary:
            directories = _bundles(Path(temporary))
            path = directories[2] / "training-contract.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["contract"]["environment"]["number_blocks"] = 9
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest contract"):
                _load_training_bundles(directories)

    def test_execution_plan_runs_each_baseline_once_and_five_learned_arms(self):
        with TemporaryDirectory() as temporary:
            bundles = _load_training_bundles(_bundles(Path(temporary)))
            arms = tuple(bundle.selected for bundle in bundles) + tuple(
                bundle.final_diagnostic
                for bundle in bundles
                if bundle.final_diagnostic is not None
            )
            baseline, learned = _execution_plan(arms)

        self.assertEqual(len(baseline), 4 * 30)
        self.assertEqual(len(set(baseline)), len(baseline))
        self.assertEqual(len(learned), 5 * 30)
        self.assertEqual(len(set(learned)), len(learned))

    def test_seed0_alias_completes_diagnostic_grid_without_execution(self):
        with TemporaryDirectory() as temporary:
            bundle = _load_training_bundles(_bundles(Path(temporary)))[0]
            source_rows = [
                {
                    "method_id": bundle.selected.method_id,
                    "method": bundle.selected.method_id,
                    "instance_seed": seed,
                    "instance_id": f"instance-{seed}",
                    "schedule_id": f"schedule-{seed}",
                    "mean_absolute_error": 10.0,
                    "relocations": 2,
                }
                for seed in EVALUATION_SEEDS
            ]
            aliases = _derive_exact_duplicate_rows(source_rows, bundle)

        self.assertEqual(len(aliases), 30)
        self.assertTrue(all(row["execution_reused"] for row in aliases))
        self.assertTrue(
            all(row["derived_from_exact_policy_duplicate"] for row in aliases)
        )
        self.assertTrue(
            all(row["policy_group"] == FINAL_DIAGNOSTIC for row in aliases)
        )

    def test_transactional_ledger_executes_once_and_refuses_contract_drift(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            calls = []
            contract = {
                "instance_id": "instance-a",
                "schedule_id": "schedule-a",
                "frozen": True,
            }

            def execute():
                calls.append(1)
                return {
                    "method_id": "arm-a",
                    "instance_seed": 123,
                    "instance_id": "instance-a",
                    "schedule_id": "schedule-a",
                }

            first, _ = _load_or_execute_run(
                output_dir=output,
                method_id="arm-a",
                seed=123,
                input_contract=contract,
                execute=execute,
            )
            second, _ = _load_or_execute_run(
                output_dir=output,
                method_id="arm-a",
                seed=123,
                input_contract=contract,
                execute=execute,
            )
            self.assertEqual(first, second)
            self.assertEqual(len(calls), 1)
            changed = copy.deepcopy(contract)
            changed["frozen"] = False
            with self.assertRaisesRegex(ValueError, "ledger input contract"):
                _load_or_execute_run(
                    output_dir=output,
                    method_id="arm-a",
                    seed=123,
                    input_contract=changed,
                    execute=execute,
                )

    def test_frozen_v1_projects_legacy_obstructive_alias_only_after_validation(self):
        normalized = {
            "delivery_count": 8,
            "contention_metric_schema_version": (
                "physical_storage_relocation_decomposition_v1"
            ),
            "physical_storage_relocations": 2,
            "target_bound_obstruction_clearances": 0,
            "standalone_reconfigurations": 2,
            "standalone_with_direct_delivery_available": 2,
            "standalone_without_direct_delivery_available": 0,
            "directly_deliverable_self_reconfigurations": 1,
            "relocations": 2,
            "obstructive_moves": 0,
            "legacy_rescored_return": 1.0,
            "dense_rescored_return": 2.0,
        }
        row = _finish_row(normalized, {"method_id": "synthetic"})
        self.assertEqual(row["target_bound_obstruction_clearances"], 0)
        self.assertEqual(row["obstructive_moves"], 2)
        self.assertEqual(
            row["physical_storage_relocations_per_100_deliveries"], 25.0
        )

        broken = dict(normalized, standalone_reconfigurations=1)
        with self.assertRaisesRegex(ValueError, "must equal"):
            _finish_row(broken, {"method_id": "synthetic"})


if __name__ == "__main__":
    unittest.main()
