import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from aggregate_fully_learned_v4_1_seeds import (
    BASELINE_SOURCES,
    EXPECTED_PHASE_COUNTS,
    SeedInput,
    aggregate,
    summarize_episode_runs,
)
from fully_learned_hierarchy import (
    FULLY_LEARNED_ACTION_INTERFACE,
    FULLY_LEARNED_BACKUP_VERSION,
    FULLY_LEARNED_CANDIDATE_FEATURE_VERSION,
    FULLY_LEARNED_CHECKPOINT_SCHEMA_VERSION,
    FULLY_LEARNED_CONTROLLER_ARCHITECTURE,
    FULLY_LEARNED_NETWORK_ARCHITECTURE,
    FULLY_LEARNED_REPLAY_VERSION,
)
from PSLAP.checkpoint_identity import selector_deployment_digest
from train_fully_learned_track_b import (
    CURRICULUM_CONTRACT,
    FAILURE_CONTRACT,
    FAILURE_PENALTY_CONTRACT,
    FULLY_LEARNED_TRAINER_VERSION,
    METHOD,
    MODEL_SELECTION_CONTRACT,
    RAW_MACRO_RETURN_CONTRACT,
    TRUNCATION_CONTRACT,
)


def _selector():
    return {
        "selector_architecture": "synthetic_test_selector",
        "selector_feature_version": 5,
        "config": {"test": True},
        "deployment_state_dict": {"weight": torch.tensor([1.0])},
    }


def _best_payload(seed):
    selector = _selector()
    digest = selector_deployment_digest(selector)
    return {
        "fully_learned_checkpoint_schema_version": (
            FULLY_LEARNED_CHECKPOINT_SCHEMA_VERSION
        ),
        "fully_learned_trainer_version": FULLY_LEARNED_TRAINER_VERSION,
        "curriculum_contract": CURRICULUM_CONTRACT,
        "checkpoint_kind": "deployment_best",
        "resumable": False,
        "controller_architecture": FULLY_LEARNED_CONTROLLER_ARCHITECTURE,
        "controller_action_interface": FULLY_LEARNED_ACTION_INTERFACE,
        "network_architecture": FULLY_LEARNED_NETWORK_ARCHITECTURE,
        "replay_version": FULLY_LEARNED_REPLAY_VERSION,
        "backup_version": FULLY_LEARNED_BACKUP_VERSION,
        "candidate_feature_version": FULLY_LEARNED_CANDIDATE_FEATURE_VERSION,
        "macro_return_contract": RAW_MACRO_RETURN_CONTRACT,
        "truncation_contract": TRUNCATION_CONTRACT,
        "failure_contract": FAILURE_CONTRACT,
        "failure_penalty_contract": FAILURE_PENALTY_CONTRACT,
        "model_selection_contract": MODEL_SELECTION_CONTRACT,
        "quality_gate_contract": "test_quality_gate",
        "selector_frozen_independent_replay_disabled": True,
        "selector_checkpoint": selector,
        "selector_deployment_digest": digest,
        "agent_checkpoint_state": {"training_phase": "joint"},
        "training_phase": "joint",
        "teacher_action_mixture": 0.0,
        "teacher_coefficient": 0.0,
        "accept_policy_lock": False,
        "spatial_trainable": True,
        "frozen_reg_runtime_policy_query": False,
        "common_continuation": True,
        "teacher_prior_application": "behavior_selection_only_v1",
        "td_teacher_prior": False,
        "assignment_source_independent_replay": False,
        "failure_penalty": -50.0,
        "failure_boundary_penalty": -50.0,
        "fully_learned_config": {"failure_penalty": -50.0, "test": True},
        "phase_episode_counts": dict(EXPECTED_PHASE_COUNTS),
        "completed_imitation_episodes": 50,
        "completed_temporal_episodes": 150,
        "completed_spatial_episodes": 50,
        "completed_joint_episodes": 100,
        "completed_training_episodes": 350,
        "training_seed": seed,
        "python_hash_seed": seed,
        "training_instance_seed_base": 600_000,
        "training_lambda": 0.5,
        "training_mu": 50.0,
        "geometry": {"geometry_signature": "standard-geometry"},
        "max_steps": 4000,
        "max_defer_steps": 10,
        "lookahead_margin_steps": 2.0,
        "learning_rate": 5e-5,
        "spatial_learning_rate": 5e-6,
        "batch_size": 128,
        "buffer_size": 20_000,
        "update_every": 4,
        "updates_per_macro": 2,
        "joint_updates_per_macro": 1,
        "joint_td_warmup_decisions": 2_000,
        "gamma": 0.99,
        "target_tau": 0.001,
        "grad_clip": 5.0,
        "reward_scale": 0.01,
        "replay_sampling": "mode_balanced",
        "validation_seeds": (10_000, 10_001, 10_002),
        "validation_steps": 4000,
        "eval_every": 10,
        "target_window": 20.0,
        "best_score": (1.0, 1_490.0 + seed),
        "validation": {
            "strict_method_success_rate": 1.0,
            "reservation_integrity_rate": 1.0,
            "selector_independent_replay_isolation_rate": 1.0,
            "total_invalid_assignments": 0,
            "total_fallbacks": 0,
            "total_reservation_invalidations": 0,
            "method_failures": [],
        },
    }


def _latest_payload(best):
    latest = dict(best)
    latest.update(
        {
            "checkpoint_kind": "resumable_latest",
            "resumable": True,
            "completed_joint_episodes": 250,
            "completed_training_episodes": 500,
            "quality_gate_blocked": False,
            "rng_state": {"python": "synthetic"},
        }
    )
    history = []
    episode = 0
    for phase, count in EXPECTED_PHASE_COUNTS.items():
        for _ in range(count):
            episode += 1
            history.append({"episode": episode, "phase": phase})
    latest["training_history"] = history
    latest["validation_history"] = [
        {
            "episode": 500,
            "quality_gate": {"passed": True},
            "quality_gate_enforcement": {"blocked": False},
        }
    ]
    latest["agent_checkpoint_state"] = {
        "training_phase": "joint",
        "optimizer": {},
        "replay": [],
        "rng_state": {},
    }
    return latest


def _run(seed, instance, schedule, checkpoint, episode, *, value, relocations=0):
    return {
        "return": float(value),
        "success": 1.0,
        "strict_method_success": 1.0,
        "reservation_integrity": 1.0,
        "selector_independent_replay_isolated": 1.0,
        "truncated": 0.0,
        "method_failure_reason": None,
        "steps": 1000,
        "delivery_count": 40,
        "mean_absolute_error": 4.0,
        "mean_tardiness": 2.0,
        "within_target_window_rate": 1.0,
        "obstructive_moves": relocations,
        "illegal_drops": 0,
        "selector_audit": {
            "invalid_assignment_count": 0,
            "fallback_count": 0,
        },
        "scheduler_audit": {"reservation_invalidation_count": 0},
        "method": METHOD,
        "track": "B",
        "eval_seed": seed,
        "instance_id": instance,
        "schedule_id": schedule,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_episode": episode,
    }


def _write_evaluation(path, best_path, best_episode, condition, offset=0.0):
    instances = "s" if condition == "standard" else "c"
    runs = [
        _run(
            seed,
            f"{instances}-{seed}",
            f"schedule-{seed}",
            best_path,
            best_episode,
            value=1400.0 + offset + index,
            relocations=(1 if condition == "contention" and index == 0 else 0),
        )
        for index, seed in enumerate((700, 701))
    ]
    protocol = {
        "track": "B_complete_system",
        "method": METHOD,
        "checkpoint_kind": "deployment_best",
        "strict_no_fallback": True,
        "checkpoint": str(best_path.resolve()),
        "eval_seeds": [700, 701],
        "observed_generalization_axes": (
            [] if condition == "standard" else ["exit_width"]
        ),
    }
    path.write_text(
        json.dumps(
            {"protocol": protocol, "summary": summarize_episode_runs(runs), "runs": runs}
        )
    )


def _baseline_run(geometry, source, commitment, scheduler, seed):
    condition = "standard" if geometry == "ordinary" else "contention"
    instance = ("s" if condition == "standard" else "c") + f"-{seed}"
    source_value = {
        "reg_selector_v5": 1390.0,
        "nearest_free": 900.0,
        "dynamic_pslap": 1300.0,
    }[source]
    run = _run(
        seed,
        instance,
        f"schedule-{seed}",
        Path("unused.pth"),
        0,
        value=source_value,
        relocations=(2 if source == "dynamic_pslap" and condition == "contention" else 0),
    )
    run.update(
        {
            "geometry_condition": geometry,
            "assignment_source": source,
            "assignment_commitment": commitment,
            "scheduler_variant": scheduler,
        }
    )
    return run


def _write_factorial(path):
    runs = []
    for geometry in ("ordinary", "egress_constrained"):
        for commitment in ("post_pickup_recompute", "decision_epoch_reserved"):
            for source in BASELINE_SOURCES:
                for seed in (700, 701):
                    runs.append(
                        _baseline_run(geometry, source, commitment, "duration_aware", seed)
                    )
            for seed in (700, 701):
                runs.append(
                    _baseline_run(
                        geometry,
                        "reg_selector_v5",
                        commitment,
                        "due_only",
                        seed,
                    )
                )
    path.write_text(
        json.dumps(
            {
                "protocol": {
                    "protocol_version": "track_b_reserved_hierarchy_factorial_v1",
                    "seeds": [700, 701],
                },
                "runs": runs,
            }
        )
    )


class ThreeSeedAggregationTests(unittest.TestCase):
    def _artifacts(self, root, seeds=(0, 1, 2)):
        inputs = []
        for index, seed in enumerate(seeds):
            run_dir = root / f"run-{index}-seed-{seed}"
            run_dir.mkdir()
            best = _best_payload(seed)
            latest = _latest_payload(best)
            best_path = run_dir / "best.pth"
            torch.save(best, best_path)
            torch.save(latest, run_dir / "latest.pth")
            standard = root / f"standard-{index}-seed-{seed}.json"
            contention = root / f"contention-{index}-seed-{seed}.json"
            _write_evaluation(standard, best_path, 350, "standard", offset=seed)
            _write_evaluation(contention, best_path, 350, "contention", offset=seed)
            inputs.append(SeedInput(run_dir, standard, contention))
        baseline = root / "factorial.json"
        _write_factorial(baseline)
        return inputs, baseline

    def test_authenticates_and_aggregates_training_seed_replication(self):
        with TemporaryDirectory() as directory:
            inputs, baseline = self._artifacts(Path(directory))
            result = aggregate(inputs, baseline)

        self.assertEqual(
            [item["training_seed"] for item in result["per_training_seed"]],
            [0, 1, 2],
        )
        standard_return = result["across_training_seeds"]["standard"][
            "training_seed_distribution"
        ]["mean_return"]
        self.assertEqual(standard_return["mean"], 1401.5)
        self.assertEqual(standard_return["sample_std"], 1.0)
        self.assertEqual(standard_return["range"], 2.0)
        self.assertEqual(
            result["across_training_seeds"]["contention"][
                "relocation_total_across_training_seeds"
            ],
            3,
        )
        self.assertEqual(result["baseline_factorial"]["factorial_cell_count"], 16)
        dynamic = next(
            item
            for item in result["paired_baseline_comparisons_across_training_seeds"]
            if item["condition"] == "contention"
            and item["baseline_assignment_source"] == "dynamic_pslap"
        )
        self.assertEqual(dynamic["relocation_reduction"]["mean"], 3.0)

    def test_rejects_duplicate_training_seed(self):
        with TemporaryDirectory() as directory:
            inputs, baseline = self._artifacts(Path(directory), seeds=(0, 1, 1))
            with self.assertRaisesRegex(ValueError, "distinct"):
                aggregate(inputs, baseline)

    def test_rejects_incomplete_latest_checkpoint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, baseline = self._artifacts(root)
            path = inputs[1].training_dir / "latest.pth"
            latest = torch.load(path, map_location="cpu", weights_only=False)
            latest["completed_training_episodes"] = 499
            torch.save(latest, path)
            with self.assertRaisesRegex(ValueError, "completed episodes"):
                aggregate(inputs, baseline)

    def test_rejects_evaluation_instance_mismatch_across_training_seeds(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, baseline = self._artifacts(root)
            path = inputs[2].contention_evaluation
            payload = json.loads(path.read_text())
            payload["runs"][0]["instance_id"] = "different-instance"
            payload["summary"] = summarize_episode_runs(payload["runs"])
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "ordered contention"):
                aggregate(inputs, baseline)

    def test_rejects_incomplete_baseline_factorial(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, baseline = self._artifacts(root)
            payload = json.loads(baseline.read_text())
            payload["runs"] = payload["runs"][:-1]
            baseline.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "seed order"):
                aggregate(inputs, baseline)

    def test_rejects_evaluation_bound_to_another_checkpoint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, baseline = self._artifacts(root)
            path = inputs[0].standard_evaluation
            payload = json.loads(path.read_text())
            payload["protocol"]["checkpoint"] = str(root / "other.pth")
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "checkpoint path"):
                aggregate(inputs, baseline)


if __name__ == "__main__":
    unittest.main()
