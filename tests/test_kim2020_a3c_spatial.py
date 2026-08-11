"""Behavioral tests for the Kim et al. (2020)-inspired spatial source."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from compare_kim2020_a3c_spatial import (
    LOCKED_KIM_REQUIRED_METADATA,
    checkpoint_seed_audit,
    enforce_seed_disjointness,
    paired_contrast,
    summarize_method,
    validate_locked_kim2020_checkpoint,
)
from example.Options.selector_v5 import make_track_b_assignment_option
from example.small_rooms_env import SmallRoomsEnv
from example.yard_geometry import geometry_metadata
from PSLAP.dynamic_yard import BlockView, YardSnapshot
from PSLAP.kim2020_a3c_spatial import (
    ARCHITECTURE_NAME,
    DEPLOYMENT_MAP,
    DEPLOYMENT_STOCHASTIC,
    FEATURE_VERSION,
    Kim2020A3CSpatialSource,
    Kim2020Config,
    Kim2020Observation,
    build_spatial_observation,
    kim2020_deployment_digest,
)
from PSLAP.track_a import (
    TrackAOnlinePolicy,
    TRACK_A_KIM2020_A3C_SPATIAL,
    shared_candidate_mask,
)
from train_kim2020_a3c_spatial import (
    MODEL_SELECTION_CONTRACT,
    TRAINER_VERSION,
    _validate_resume_payload,
    evaluation_selection,
    main as train_kim2020,
    model_selection_key,
    parse_args as parse_training_args,
    save_checkpoint,
    validate_cuda_determinism,
)


class Kim2020A3CSpatialTests(unittest.TestCase):
    """Keep the paper-facing claims executable at the source boundary."""

    @staticmethod
    def make_env(*, rows: int = 6, cols: int = 7) -> SmallRoomsEnv:
        return SmallRoomsEnv(
            grid_rows=rows,
            grid_cols=cols,
            choose_storage=False,
            number_blocks=4,
            arrival_rate=0.5,
            proc_mean=20,
        )

    @staticmethod
    def inbound_context(env: SmallRoomsEnv):
        inbound = next(
            block
            for block in env.blocks
            if block.position == env.pickup_cell and not block.stored
        )
        block = BlockView(
            inbound.label,
            inbound.position,
            float(inbound.get_remaining_storage_time()),
        )
        yard = YardSnapshot.from_env(env)
        candidates = shared_candidate_mask(yard, block, inbound.position)
        return yard, block, inbound.position, candidates

    @staticmethod
    def paper_grid() -> tuple[YardSnapshot, BlockView, tuple[tuple[int, int], ...]]:
        cells = frozenset((row, col) for row in range(3) for col in range(4))
        yard = YardSnapshot(
            rows=3,
            cols=4,
            traversable=cells - {(0, 3)},
            storage_cells=frozenset(
                {(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)}
            ),
            exits=((2, 3),),
            blocks=(
                BlockView("due", (0, 0), -4.0),
                BlockView("half", (0, 1), 5.0),
                BlockView("clipped", (1, 0), 30.0),
            ),
        )
        incoming = BlockView("incoming", (2, 0), 10.0)
        candidates = ((1, 1), (1, 2))
        return yard, incoming, candidates

    @staticmethod
    def tiny_config(**overrides) -> Kim2020Config:
        values = {
            "learning_rate": 1e-2,
            "weight_decay": 0.0,
            "gamma": 1.0,
            "reward_scale": 1.0,
            "entropy_coef": 0.0,
            "value_coef": 1.0,
            "grad_clip": 10.0,
        }
        values.update(overrides)
        return Kim2020Config(**values)

    @staticmethod
    def imported_workspace_config(
        configured_value: str | None,
    ) -> str:
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        if configured_value is None:
            environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            environment["CUBLAS_WORKSPACE_CONFIG"] = configured_value
        command = (
            "import os; "
            "import train_kim2020_a3c_spatial; "
            "print(os.environ['CUBLAS_WORKSPACE_CONFIG'])"
        )
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def test_trainer_sets_cublas_default_in_fresh_process(self):
        self.assertEqual(self.imported_workspace_config(None), ":4096:8")

    def test_trainer_preserves_caller_cublas_configuration(self):
        self.assertEqual(
            self.imported_workspace_config(":16:8"), ":16:8"
        )

    def test_deterministic_cuda_rejects_unsupported_cublas_configuration(self):
        with patch.dict(
            os.environ, {"CUBLAS_WORKSPACE_CONFIG": "unsupported"}
        ):
            with self.assertRaisesRegex(
                RuntimeError, "deterministic CUDA requires"
            ):
                validate_cuda_determinism(
                    device="cuda", deterministic_algorithms=True
                )
            self.assertEqual(
                validate_cuda_determinism(
                    device="cuda", deterministic_algorithms=False
                ),
                "unsupported",
            )
            self.assertEqual(
                validate_cuda_determinism(
                    device="cpu", deterministic_algorithms=True
                ),
                "unsupported",
            )

    def test_checkpoint_records_and_resume_enforces_cuda_provenance(self):
        with TemporaryDirectory() as directory:
            env = self.make_env()
            config = self.tiny_config()
            source = Kim2020A3CSpatialSource(
                env,
                config,
                seed=20,
                learning_enabled=True,
                deployment_mode=DEPLOYMENT_STOCHASTIC,
            )
            args = parse_training_args(
                [
                    "--episodes",
                    "2",
                    "--validation-seeds",
                    "10000",
                    "--device",
                    "cpu",
                    "--output-dir",
                    directory,
                ]
            )
            checkpoint_path = Path(directory) / "latest.pth"
            save_checkpoint(
                source,
                checkpoint_path,
                checkpoint_kind="latest",
                args=args,
                config=config,
                geometry={},
                completed_episodes=1,
                validation={},
                python_hash_seed="0",
                training_device="cpu",
                cublas_workspace_config=":4096:8",
            )
            payload = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            self.assertEqual(payload["training_device"], "cpu")
            self.assertEqual(
                payload["cublas_workspace_config"], ":4096:8"
            )
            self.assertEqual(
                _validate_resume_payload(
                    payload,
                    args=args,
                    config=config,
                    python_hash_seed="0",
                    training_device="cpu",
                    cublas_workspace_config=":4096:8",
                ),
                1,
            )

            with self.assertRaisesRegex(ValueError, "training_device"):
                _validate_resume_payload(
                    payload,
                    args=args,
                    config=config,
                    python_hash_seed="0",
                    training_device="cuda",
                    cublas_workspace_config=":4096:8",
                )
            with self.assertRaisesRegex(
                ValueError, "cublas_workspace_config"
            ):
                _validate_resume_payload(
                    payload,
                    args=args,
                    config=config,
                    python_hash_seed="0",
                    training_device="cpu",
                    cublas_workspace_config=":16:8",
                )

    def test_paper_relative_dwell_channel_and_masks(self):
        yard, incoming, candidates = self.paper_grid()
        observation = build_spatial_observation(yard, incoming, candidates)

        self.assertIsInstance(observation, Kim2020Observation)
        self.assertEqual(observation.spatial_features.shape[1:], (3, 4))
        relative_dwell = observation.spatial_features[0]
        occupancy = observation.spatial_features[1]
        storage = observation.spatial_features[2]

        self.assertEqual(occupancy[0, 0], 1.0)
        self.assertEqual(occupancy[0, 1], 1.0)
        self.assertEqual(occupancy[1, 0], 1.0)
        self.assertEqual(occupancy[1, 1], 0.0)
        # Negative/due remaining time is floored at zero; the upper tail is
        # clipped exactly as in the paper's [0, 2] encoding.
        np.testing.assert_allclose(
            [relative_dwell[0, 0], relative_dwell[0, 1], relative_dwell[1, 0]],
            [0.0, 0.5, 2.0],
            rtol=0,
            atol=1e-7,
        )
        self.assertEqual(storage[1, 1], 1.0)
        self.assertEqual(storage[2, 3], 0.0)

        expected_mask = np.zeros((3, 4), dtype=bool)
        expected_mask[1, 1] = True
        expected_mask[1, 2] = True
        np.testing.assert_array_equal(observation.candidate_mask, expected_mask)
        self.assertEqual(observation.candidates, candidates)

    def test_map_and_stochastic_deployment_never_escape_legal_mask(self):
        torch.manual_seed(11)
        env = self.make_env()
        env.reset(instance=env.sample_episode_instance(111))
        yard, block, source_cell, candidates = self.inbound_context(env)
        self.assertGreater(len(candidates), 1)

        for mode in (DEPLOYMENT_MAP, DEPLOYMENT_STOCHASTIC):
            source = Kim2020A3CSpatialSource(
                env,
                self.tiny_config(),
                seed=11,
                learning_enabled=False,
                deployment_mode=mode,
            )
            for _ in range(25):
                chosen = source.propose(yard, block, source_cell, candidates)
                self.assertIn(chosen, candidates)

    def test_map_preview_is_idempotent(self):
        torch.manual_seed(12)
        env = self.make_env()
        env.reset(instance=env.sample_episode_instance(112))
        yard, block, source_cell, candidates = self.inbound_context(env)
        source = Kim2020A3CSpatialSource(
            env,
            self.tiny_config(),
            seed=12,
            learning_enabled=False,
            deployment_mode=DEPLOYMENT_MAP,
        )

        before = (source.assignment_count, source.gradient_steps)
        previews = [
            source.preview(yard, block, source_cell, candidates)
            for _ in range(5)
        ]
        self.assertTrue(all(choice == previews[0] for choice in previews))
        self.assertIn(previews[0], candidates)
        self.assertEqual((source.assignment_count, source.gradient_steps), before)

    def test_stochastic_preview_is_idempotent_and_seed_reproducible(self):
        torch.manual_seed(13)
        env = self.make_env()
        env.reset(instance=env.sample_episode_instance(113))
        yard, block, source_cell, candidates = self.inbound_context(env)
        payload = Kim2020A3CSpatialSource(
            env,
            self.tiny_config(),
            seed=999,
            learning_enabled=False,
            deployment_mode=DEPLOYMENT_MAP,
        ).checkpoint()
        left = Kim2020A3CSpatialSource.from_checkpoint(
            env,
            payload,
            seed=71,
            learning_enabled=False,
            deployment_mode=DEPLOYMENT_STOCHASTIC,
        )
        right = Kim2020A3CSpatialSource.from_checkpoint(
            env,
            payload,
            seed=71,
            learning_enabled=False,
            deployment_mode=DEPLOYMENT_STOCHASTIC,
        )

        left_first = left.preview(yard, block, source_cell, candidates)
        for _ in range(7):
            self.assertEqual(
                left.preview(yard, block, source_cell, candidates), left_first
            )
        right_first = right.preview(yard, block, source_cell, candidates)
        self.assertEqual(left_first, right_first)

        # Committing the preview must not draw a second categorical sample.
        # Repeated inspection on the left consequently cannot shift the two
        # same-seed policy streams out of alignment.
        left.on_assignment_committed(
            block_label=block.label,
            chosen_cell=left_first,
            time_step=0,
        )
        right.on_assignment_committed(
            block_label=block.label,
            chosen_cell=right_first,
            time_step=0,
        )
        next_block = BlockView(
            block.label + "-next", block.position, block.remaining_time
        )
        self.assertEqual(
            left.preview(yard, next_block, source_cell, candidates),
            right.preview(yard, next_block, source_cell, candidates),
        )

    def test_checkpoint_round_trip_and_geometry_rejection(self):
        torch.manual_seed(14)
        env = self.make_env()
        env.reset(instance=env.sample_episode_instance(114))
        yard, block, source_cell, candidates = self.inbound_context(env)
        source = Kim2020A3CSpatialSource(
            env,
            self.tiny_config(),
            seed=14,
            learning_enabled=True,
            deployment_mode=DEPLOYMENT_STOCHASTIC,
        )
        payload = source.checkpoint(training_seed=14)
        restored = Kim2020A3CSpatialSource.from_checkpoint(
            env,
            payload,
            seed=140,
            learning_enabled=False,
            deployment_mode=DEPLOYMENT_MAP,
        )

        self.assertEqual(restored.feature_version, FEATURE_VERSION)
        self.assertEqual(restored.architecture_name, ARCHITECTURE_NAME)
        for name, value in source.network.state_dict().items():
            torch.testing.assert_close(
                value.cpu(), restored.network.state_dict()[name].cpu()
            )
        self.assertEqual(
            restored.preview(yard, block, source_cell, candidates),
            Kim2020A3CSpatialSource.from_checkpoint(
                env,
                payload,
                seed=999,
                learning_enabled=False,
                deployment_mode=DEPLOYMENT_MAP,
            ).preview(yard, block, source_cell, candidates),
        )

        incompatible = self.make_env(rows=7, cols=7)
        with self.assertRaisesRegex(ValueError, "geometry|Geometry"):
            Kim2020A3CSpatialSource.from_checkpoint(
                incompatible,
                payload,
                learning_enabled=False,
                deployment_mode=DEPLOYMENT_MAP,
            )

    def test_track_b_reservation_commits_the_single_previewed_cell(self):
        torch.manual_seed(140)
        env = self.make_env()
        env.reset(instance=env.sample_episode_instance(1_140))
        payload = Kim2020A3CSpatialSource(
            env,
            self.tiny_config(),
            seed=140,
            learning_enabled=False,
            deployment_mode=DEPLOYMENT_MAP,
        ).checkpoint(training_lambda=0.5, training_mu=20.0)
        option = make_track_b_assignment_option(
            env,
            TRACK_A_KIM2020_A3C_SPATIAL,
            assignment_payload=payload,
            deployment_mode=DEPLOYMENT_STOCHASTIC,
            policy_seed=8_001,
        )
        block = next(
            item
            for item in env.blocks
            if item.position == env.pickup_cell and not item.stored
        )

        first = option.preview_assignment(block)
        second = option.preview_assignment(block)
        self.assertEqual(first, second)
        self.assertEqual(option.decision_count, 0)
        committed = option.commit_preview(block, first)

        self.assertEqual(committed, first.chosen_cell)
        self.assertEqual(block.storage_location, first.chosen_cell)
        audit = option.audit()
        self.assertEqual(audit["assignment_source"], TRACK_A_KIM2020_A3C_SPATIAL)
        self.assertTrue(audit["assignment_source_learned"])
        self.assertEqual(audit["reserved_commit_count"], 1)
        self.assertEqual(audit["invalid_assignment_count"], 0)
        self.assertEqual(audit["fallback_count"], 0)

    def test_delivery_resolves_only_its_own_placement_decision(self):
        torch.manual_seed(15)
        env = self.make_env()
        env.reset(instance=env.sample_episode_instance(115))
        yard, block, source_cell, candidates = self.inbound_context(env)
        source = Kim2020A3CSpatialSource(
            env,
            self.tiny_config(),
            seed=15,
            learning_enabled=True,
            deployment_mode=DEPLOYMENT_STOCHASTIC,
        )
        source.on_episode_start()
        source.propose(yard, block, source_cell, candidates)
        source.on_step(0.0, {"stored_block": block.label})

        # This relocation belongs to an atomic retrieval whose target has no
        # placement decision in the ledger.  It must not be charged to the
        # only (and most recent) pending decision merely because it exists.
        source.on_step(999.0, {"relocated_block": "other"})
        source.on_step(-999.0, {"delivered_block": "unmatched-target"})
        self.assertEqual(len(source.completed_records), 0)
        self.assertEqual(len(source.pending_records), 1)

        # The unmatched retrieval's relocation counter is closed at delivery;
        # the subsequent zero-relocation delivery of the placed block earns 0.
        source.on_step(123.0, {"delivered_block": block.label})
        source.on_episode_end(success=True, truncated=False)
        self.assertEqual(source.completed_outcome_count, 1)
        self.assertEqual(source.censored_outcome_count, 0)
        self.assertEqual(source.last_episode_audit["unmatched_delivery_count"], 1)
        record = source.last_episode_records[0]
        self.assertEqual(record.block_label, block.label)
        self.assertEqual(record.relocation_count, 0)
        self.assertEqual(record.reward, 0.0)
        self.assertFalse(record.censored)
        self.assertAlmostEqual(source.return_history[-1], 0.0, places=7)

    def test_generator_retry_reuses_unstored_provisional_placement(self):
        torch.manual_seed(19)
        env = self.make_env()
        env.reset(instance=env.sample_episode_instance(119))
        source = Kim2020A3CSpatialSource(
            env,
            self.tiny_config(),
            seed=19,
            learning_enabled=True,
            deployment_mode=DEPLOYMENT_STOCHASTIC,
        )
        source.on_episode_start()
        policy = TrackAOnlinePolicy(
            env, TRACK_A_KIM2020_A3C_SPATIAL, source
        )
        label = next(
            block.label
            for block in env.blocks
            if block.position == env.pickup_cell and not block.stored
        )

        with patch.object(
            source, "_choose_index", wraps=source._choose_index
        ) as choose:
            # Exercise the defensive retry path after admission.  The common
            # executor now prevents the observed approach failure before this
            # point, but any later transient plan failure must remain safe.
            with patch.object(
                policy, "_plan_move_and_putdown", return_value=None
            ):
                first_generator = policy.step()
                action = next(first_generator)
                self.assertEqual(action, env.ACTION_IDS["WAIT"])
                _, reward, _, info = env.step(action)
                source.on_step(float(reward), info)
                with self.assertRaises(StopIteration):
                    next(first_generator)

                first_record = source.provisional_records[label]
                first_cell = first_record.chosen_cell
                second_generator = policy.step()
                self.assertEqual(
                    next(second_generator), env.ACTION_IDS["WAIT"]
                )

            self.assertEqual(choose.call_count, 1)
            self.assertEqual(source.assignment_count, 1)
            self.assertIs(source.provisional_records[label], first_record)
            self.assertEqual(
                policy.assignment_audit.decisions[0].executed_cell,
                first_cell,
            )
            self.assertEqual(
                policy.assignment_audit.decisions[1].executed_cell,
                first_cell,
            )

            source.on_step(0.0, {"stored_block": label})
            yard, block, source_cell, candidates = self.inbound_context(env)
            with self.assertRaisesRegex(RuntimeError, "duplicate placement"):
                source.propose(yard, block, source_cell, candidates)
            self.assertEqual(choose.call_count, 1)

    def test_truncation_censors_unknown_outcome_instead_of_imputing_zero(self):
        torch.manual_seed(16)
        env = self.make_env()
        env.reset(instance=env.sample_episode_instance(116))
        yard, block, source_cell, candidates = self.inbound_context(env)
        source = Kim2020A3CSpatialSource(
            env,
            self.tiny_config(),
            seed=16,
            learning_enabled=True,
            deployment_mode=DEPLOYMENT_STOCHASTIC,
        )
        source.on_episode_start()
        source.propose(yard, block, source_cell, candidates)
        source.on_step(0.0, {"stored_block": block.label})
        source.on_step(0.0, {"relocated_block": "other"})
        source.on_episode_end(success=False, truncated=True)

        self.assertEqual(source.completed_outcome_count, 0)
        self.assertEqual(source.censored_outcome_count, 1)
        self.assertEqual(source.gradient_steps, 0)
        self.assertEqual(len(source.last_episode_records), 1)
        record = source.last_episode_records[0]
        self.assertTrue(record.censored)
        self.assertEqual(record.censor_reason, "episode_truncated")
        self.assertIsNone(record.relocation_count)
        self.assertIsNone(record.reward)

    def test_resolved_delayed_cost_runs_actor_critic_update(self):
        torch.manual_seed(17)
        env = self.make_env()
        env.reset(instance=env.sample_episode_instance(117))
        yard, block, source_cell, candidates = self.inbound_context(env)
        source = Kim2020A3CSpatialSource(
            env,
            self.tiny_config(),
            seed=17,
            learning_enabled=True,
            deployment_mode=DEPLOYMENT_STOCHASTIC,
        )
        before = {
            name: value.detach().clone()
            for name, value in source.network.state_dict().items()
        }
        source.on_episode_start()
        source.propose(yard, block, source_cell, candidates)
        source.on_step(0.0, {"stored_block": block.label})
        source.on_step(500.0, {"relocated_block": "r1"})
        source.on_step(-500.0, {"relocated_block": "r2"})
        source.on_step(1_000.0, {"delivered_block": block.label})
        source.on_episode_end(success=True, truncated=False)

        self.assertEqual(source.gradient_steps, 1)
        self.assertEqual(source.completed_outcome_count, 1)
        self.assertTrue(np.isfinite(source.loss_history[-1]))
        self.assertTrue(
            any(
                not torch.equal(value, source.network.state_dict()[name])
                for name, value in before.items()
            )
        )
        # General environment reward is deliberately irrelevant; only the two
        # relocation events determine this delayed learning reward.
        self.assertAlmostEqual(source.return_history[-1], -2.0, places=7)

    def test_delayed_rewards_form_returns_in_placement_order(self):
        torch.manual_seed(18)
        env = self.make_env()
        env.reset(instance=env.sample_episode_instance(118))
        yard, first, source_cell, candidates = self.inbound_context(env)
        source = Kim2020A3CSpatialSource(
            env,
            self.tiny_config(gamma=0.5),
            seed=18,
            learning_enabled=True,
            deployment_mode=DEPLOYMENT_STOCHASTIC,
        )
        source.on_episode_start()
        source.propose(yard, first, source_cell, candidates)
        source.on_step(0.0, {"stored_block": first.label})

        env.time_steps += 1
        second = BlockView("second", first.position, first.remaining_time)
        source.propose(yard, second, source_cell, candidates)
        source.on_step(0.0, {"stored_block": second.label})

        # Outcomes deliberately resolve in the opposite order.  Learning must
        # restore placement order before calculating R_t.
        source.on_step(0.0, {"relocated_block": "r1"})
        source.on_step(0.0, {"relocated_block": "r2"})
        source.on_step(0.0, {"delivered_block": second.label})
        source.on_step(0.0, {"relocated_block": "r3"})
        source.on_step(0.0, {"delivered_block": first.label})
        source.on_episode_end(success=True, truncated=False)

        records = source.last_episode_records
        self.assertEqual([record.block_label for record in records], [first.label, "second"])
        self.assertEqual([record.reward for record in records], [-1.0, -2.0])
        self.assertAlmostEqual(records[0].discounted_return, -2.0, places=7)
        self.assertAlmostEqual(records[1].discounted_return, -2.0, places=7)
        self.assertTrue(source.last_episode_audit["integrity"])

    def test_comparison_censors_failed_schedules_instead_of_counting_wins(self):
        def row(method, instance_id, moves, valid):
            return {
                "comparison_method": method,
                "instance_id": instance_id,
                "schedule_id": f"schedule-{instance_id}",
                "schedule_seed": int(instance_id),
                "protocol_valid": valid,
                "strict_method_success": float(valid),
                "success": float(valid),
                "invalid_assignment_count": int(not valid),
                "fallback_count": 0,
                "illegal_drops": 0,
                "obstructive_moves": float(moves),
                "obstructive_moves_per_delivered_block": float(moves),
                "return": -float(moves),
                "mean_absolute_error": float(moves),
                "steps": float(moves),
            }

        summary = summarize_method(
            [row("kim", "1", 0, False), row("kim", "2", 4, True)]
        )
        self.assertEqual(summary["schedule_count"], 2)
        self.assertEqual(summary["eligible_schedule_count"], 1)
        self.assertEqual(summary["censored_schedule_count"], 1)
        self.assertEqual(summary["mean_obstructive_moves"], 4.0)

        contrast = paired_contrast(
            [row("kim", "1", 0, False), row("baseline", "1", 3, True)],
            "kim",
            "baseline",
        )
        self.assertEqual(contrast["eligible_schedule_count"], 0)
        self.assertEqual(contrast["censored_schedule_count"], 1)
        self.assertEqual(contrast["reference_wins"], 0)
        self.assertIsNone(
            contrast["schedule_rows"][0]["delta_obstructive_moves"]
        )

    def test_checkpoint_selection_uses_stochastic_primary_not_map_diagnostic(
        self,
    ):
        validation = {
            "map": {
                "all_strict_method_success": False,
                "strict_method_success_rate": 0.6,
                "mean_obstructive_moves": 0.2,
            },
            "stochastic": {
                "all_strict_method_success": True,
                "strict_method_success_rate": 1.0,
                "mean_obstructive_moves": 1.5,
                "mean_return": -1.5,
            },
        }
        validation["selection"] = evaluation_selection(validation)

        self.assertTrue(validation["selection"]["eligible"])
        self.assertFalse(
            validation["selection"][
                "secondary_map_all_strict_method_success"
            ]
        )
        self.assertEqual(
            validation["selection"]["contract"], MODEL_SELECTION_CONTRACT
        )

        key = model_selection_key(validation)
        validation["map"]["mean_obstructive_moves"] = 999.0
        validation["selection"] = evaluation_selection(validation)
        self.assertEqual(model_selection_key(validation), key)

        validation["stochastic"]["all_strict_method_success"] = False
        validation["stochastic"]["strict_method_success_rate"] = 0.96
        validation["selection"] = evaluation_selection(validation)
        self.assertFalse(validation["selection"]["eligible"])

    def test_final_schedule_seeds_must_be_disjoint_from_training_and_validation(self):
        payload = {
            "training_seed": 2,
            "training_instance_seed_base": 200_000,
            "completed_training_episodes": 3,
            "validation_instance_seeds": [9_001, 9_002],
        }
        audit = checkpoint_seed_audit(
            payload, [2_200_002, 9_002, 30_000]
        )
        self.assertTrue(audit["verified"])
        self.assertFalse(audit["disjoint"])
        self.assertEqual(audit["training_overlap"], [2_200_002])
        self.assertEqual(audit["validation_overlap"], [9_002])
        with self.assertRaisesRegex(ValueError, "not verified disjoint"):
            enforce_seed_disjointness({"kim2020": audit}, allow=False)
        enforce_seed_disjointness({"kim2020": audit}, allow=True)

    def test_locked_comparison_pins_checkpoint_provenance_geometry_and_digest(self):
        env = self.make_env()
        source = Kim2020A3CSpatialSource(
            env,
            self.tiny_config(),
            seed=23,
            learning_enabled=False,
        )
        valid_mode_summary = {
            "all_strict_method_success": True,
            "mean_invalid_assignment_count": 0.0,
            "mean_fallback_count": 0.0,
            "mean_retrieval_live_plan_failure_count": 0.0,
            "mean_exact_recovery_failure_count": 0.0,
        }
        payload = source.checkpoint(
            geometry=geometry_metadata(env, requested_exit_width=None),
            training_seed=0,
            completed_training_episodes=1,
            validation_instance_seeds=[91],
            validation_evaluation={
                "validation_instance_seeds": [91],
                "selection": {
                    "contract": LOCKED_KIM_REQUIRED_METADATA[
                        "model_selection_contract"
                    ],
                    "eligible": True,
                    "primary_deployment_mode": DEPLOYMENT_STOCHASTIC,
                },
                "stochastic": dict(valid_mode_summary),
                "map": dict(valid_mode_summary),
            },
            **LOCKED_KIM_REQUIRED_METADATA,
        )
        digest = kim2020_deployment_digest(payload)

        audit = validate_locked_kim2020_checkpoint(
            payload,
            env=env,
            requested_exit_width=None,
            expected_deployment_digest=digest,
        )
        self.assertTrue(audit["verified"])
        self.assertTrue(audit["deployment_digest_pinned"])
        self.assertTrue(audit["deployment_digest_match"])
        self.assertTrue(audit["geometry_contract_match"])
        self.assertTrue(audit["geometry_metadata_match"])

        latest = dict(payload, checkpoint_kind="latest")
        with self.assertRaisesRegex(ValueError, "checkpoint_kind"):
            validate_locked_kim2020_checkpoint(
                latest,
                env=env,
                requested_exit_width=None,
                expected_deployment_digest=digest,
            )

        shifted_geometry = dict(payload)
        shifted_geometry["geometry"] = {
            **payload["geometry"],
            "block_count": payload["geometry"]["block_count"] + 1,
        }
        with self.assertRaisesRegex(ValueError, "geometry_metadata"):
            validate_locked_kim2020_checkpoint(
                shifted_geometry,
                env=env,
                requested_exit_width=None,
                expected_deployment_digest=digest,
            )

        wrong_digest = "0" * 64
        if digest == wrong_digest:
            wrong_digest = "f" * 64
        with self.assertRaisesRegex(ValueError, "deployment_digest"):
            validate_locked_kim2020_checkpoint(
                payload,
                env=env,
                requested_exit_width=None,
                expected_deployment_digest=wrong_digest,
            )

        failed_validation = dict(payload)
        failed_validation["validation_evaluation"] = {
            **payload["validation_evaluation"],
            "stochastic": {
                **payload["validation_evaluation"]["stochastic"],
                "mean_exact_recovery_failure_count": 1.0,
            },
        }
        with self.assertRaisesRegex(
            ValueError,
            "validation.stochastic.mean_exact_recovery_failure_count",
        ):
            validate_locked_kim2020_checkpoint(
                failed_validation,
                env=env,
                requested_exit_width=None,
                expected_deployment_digest=digest,
            )

    def test_rejected_resume_does_not_modify_run_directory(self):
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            resume_path = output_dir / "latest.pth"
            torch.save(
                {
                    "trainer_version": TRAINER_VERSION,
                    "checkpoint_kind": "latest",
                    "training_seed": 0,
                    "python_hash_seed": "0",
                    "training_lambda": 0.5,
                    "training_mu": 50.0,
                    "training_instance_seed_base": 200_000,
                    "validation_instance_seeds": [10_000],
                    "validation_policy_seed_base": 10_000_000,
                    "stochastic_rollouts": 5,
                    "max_steps": 4000,
                    "deterministic_algorithms": True,
                    "training_device": "cpu",
                    "cublas_workspace_config": ":4096:8",
                    "config": {
                        "hidden_channels": 32,
                        "learning_rate": 1e-4,
                        "weight_decay": 0.0,
                        "gamma": 0.99,
                        "reward_scale": 1.0,
                        "entropy_coef": 0.01,
                        "value_coef": 0.5,
                        "grad_clip": 5.0,
                        "updates_per_episode": 1,
                    },
                    "completed_training_episodes": 1,
                },
                resume_path,
            )
            (output_dir / "training-history.json").write_text("[]\n")
            (output_dir / "validation-instances.json").write_text(
                '{"sentinel": "original"}\n'
            )

            def snapshot():
                return {
                    str(path.relative_to(output_dir)): path.read_bytes()
                    for path in output_dir.rglob("*")
                    if path.is_file()
                }

            before = snapshot()
            argv = [
                "--resume",
                str(resume_path),
                "--episodes",
                "2",
                "--validation-seeds",
                "10001",
                "--device",
                "cpu",
                "--output-dir",
                str(output_dir),
            ]
            with patch.dict(os.environ, {"PYTHONHASHSEED": "0"}), patch(
                "train_kim2020_a3c_spatial.seed_everything"
            ):
                with self.assertRaisesRegex(
                    ValueError, "validation_instance_seeds"
                ):
                    train_kim2020(argv)
            self.assertEqual(snapshot(), before)


if __name__ == "__main__":
    unittest.main()
