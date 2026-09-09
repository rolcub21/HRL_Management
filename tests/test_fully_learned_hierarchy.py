from collections import Counter
from dataclasses import replace
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from example.Options.AcceptStoreOption import ReservedAcceptStoreOption
from example.Options.selector_v5 import ExplicitCellAssignmentRegistry
from example.controller_options import (
    FULLY_LEARNED_RESERVED_ACTION_INTERFACE,
    build_controller_options,
)
from example.controller_observation import (
    BLOCK_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    OnlineManifestTimingObservationEncoder,
)
from example.small_rooms_env import SmallRoomsEnv
from fully_learned_hierarchy import (
    CANDIDATE_FEATURE_NAMES,
    CELL_FEATURE_SLICE,
    HISTORY_FEATURE_NAMES,
    FullyLearnedConfig,
    FullyLearnedHierarchyAgent,
    FullyLearnedInfeasible,
    FullyLearnedQNetwork,
    FullyLearnedReplayBuffer,
    FullyLearnedState,
    FullyLearnedTransition,
    masked_logmeanexp,
    mode_regularized_value,
    smdp_target,
)
from PSLAP.dynamic_yard import BlockView, YardSnapshot
from PSLAP.reg_selector_v5 import REGV5AssignmentSource, REGV5Config
from PSLAP.track_a import shared_candidate_mask


SEED = 1901


def _selector_config():
    return REGV5Config(
        block_embedding_dim=8,
        candidate_embedding_dim=8,
        context_dim=16,
        replay_size=4,
        batch_size=1,
        min_replay_size=1,
        updates_per_episode=1,
    )


def _explicit_registry_system():
    env = SmallRoomsEnv(
        choose_storage=False,
        arrival_rate=0.5,
        proc_mean=50.0,
    )
    base = env.sample_episode_instance(SEED)
    instance = replace(
        base,
        arrival_steps=(0,) + (10_000,) * (len(env.blocks) - 1),
    )
    source = REGV5AssignmentSource(
        env,
        _selector_config(),
        seed=SEED,
        learning_enabled=False,
        device="cpu",
    )
    registry = ExplicitCellAssignmentRegistry(env, source)
    build_controller_options(
        env,
        registry,
        controller_action_interface=FULLY_LEARNED_RESERVED_ACTION_INTERFACE,
        max_defer_steps=10,
    )
    state = env.reset(instance=instance)
    accept = next(
        option
        for option in env.options
        if isinstance(option, ReservedAcceptStoreOption)
    )
    return env, state, registry, accept


def _small_fully_learned_agent(env, registry, **config_overrides):
    config = FullyLearnedConfig(
        block_embedding_dim=8,
        global_embedding_dim=8,
        candidate_embedding_dim=8,
        context_dim=16,
        history_length=4,
        tau_accept=0.3,
        tau_retrieve=0.4,
        tau_defer=0.5,
        tau_mode=0.7,
    )
    if config_overrides:
        config = replace(config, **config_overrides)
    encoder = OnlineManifestTimingObservationEncoder(env)
    agent = FullyLearnedHierarchyAgent(
        env,
        encoder,
        config=config,
        spatial_network=registry.source.network,
        seed=SEED,
        device="cpu",
        batch_size=1,
        buffer_size=8,
        update_every=1,
    )
    return config, encoder, agent


def _run_option(env, state, option, limit=250):
    for _ in range(limit):
        action = option.policy(state)
        state, _, _, _ = env.step(action)
        if option.termination(state):
            return state
    raise AssertionError("reserved option did not terminate")


class RegularizedValueContractTests(unittest.TestCase):
    def test_masked_logmeanexp_is_representation_cardinality_invariant(self):
        values = torch.tensor((3.0, -1.0), dtype=torch.float64)
        duplicated = values.repeat(2)

        original = masked_logmeanexp(
            values,
            torch.ones_like(values, dtype=torch.bool),
            temperature=0.7,
        )
        repeated = masked_logmeanexp(
            duplicated,
            torch.ones_like(duplicated, dtype=torch.bool),
            temperature=0.7,
        )

        torch.testing.assert_close(original, repeated, atol=1e-12, rtol=0)

    def test_mask_excludes_invalid_values_and_empty_set_is_explicit(self):
        values = torch.tensor((2.5, 1.0e9), dtype=torch.float64)
        mask = torch.tensor((True, False))

        result = masked_logmeanexp(values, mask, temperature=0.5)

        torch.testing.assert_close(
            result,
            torch.tensor(2.5, dtype=torch.float64),
            atol=1e-12,
            rtol=0,
        )
        with self.assertRaises((ValueError, RuntimeError)):
            masked_logmeanexp(
                values,
                torch.zeros_like(mask),
                temperature=0.5,
            )

    def test_masked_logmeanexp_is_numerically_stable(self):
        values = torch.tensor((1.0e6, 1.0e6 - 1.0), dtype=torch.float64)

        result = masked_logmeanexp(
            values,
            torch.ones_like(values, dtype=torch.bool),
            temperature=0.25,
        )

        self.assertTrue(bool(torch.isfinite(result)))
        self.assertGreater(float(result), 1.0e6 - 1.0)
        self.assertLessEqual(float(result), 1.0e6)

    def test_nested_mode_value_ignores_invalid_candidate(self):
        q_values = torch.tensor((4.0, -2.0, 1.0e9), dtype=torch.float64)
        modes = torch.tensor((0, 1, 1), dtype=torch.long)
        valid = torch.tensor((True, True, False))
        temperatures = torch.tensor((0.4, 0.4), dtype=torch.float64)
        priors = torch.tensor((0.5, 0.5), dtype=torch.float64)

        actual = mode_regularized_value(
            q_values,
            valid,
            modes,
            temperatures,
            mode_temperature=0.8,
            mode_priors=priors,
        )
        expected = 0.8 * torch.logsumexp(
            torch.tensor((4.0, -2.0), dtype=torch.float64) / 0.8
            + torch.log(priors),
            dim=0,
        )

        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)

    def test_nested_mode_value_is_invariant_to_full_mode_replication(self):
        # Replicating the complete semantic representation of mode 0 must not
        # change its log-mean-exp value or the cross-mode continuation value.
        q_values = torch.tensor((3.0, -1.0, 0.5), dtype=torch.float64)
        modes = torch.tensor((0, 0, 1), dtype=torch.long)
        replicated_q = torch.tensor(
            (3.0, -1.0, 3.0, -1.0, 0.5), dtype=torch.float64
        )
        replicated_modes = torch.tensor((0, 0, 0, 0, 1), dtype=torch.long)
        temperatures = torch.tensor((0.6, 0.6), dtype=torch.float64)
        priors = torch.tensor((0.5, 0.5), dtype=torch.float64)

        original = mode_regularized_value(
            q_values,
            torch.ones_like(q_values, dtype=torch.bool),
            modes,
            temperatures,
            mode_temperature=0.9,
            mode_priors=priors,
        )
        replicated = mode_regularized_value(
            replicated_q,
            torch.ones_like(replicated_q, dtype=torch.bool),
            replicated_modes,
            temperatures,
            mode_temperature=0.9,
            mode_priors=priors,
        )

        torch.testing.assert_close(original, replicated, atol=1e-12, rtol=0)

    def test_absent_mode_prior_is_renormalized(self):
        q_values = torch.tensor((3.0, 1.0), dtype=torch.float64)
        modes = torch.tensor((0, 2), dtype=torch.long)
        temperatures = torch.tensor((0.5, 0.5, 0.5), dtype=torch.float64)
        priors = torch.tensor((0.2, 0.3, 0.5), dtype=torch.float64)

        actual = mode_regularized_value(
            q_values,
            torch.ones_like(q_values, dtype=torch.bool),
            modes,
            temperatures,
            mode_temperature=0.7,
            mode_priors=priors,
        )
        live_priors = torch.tensor((0.2, 0.5), dtype=torch.float64)
        live_priors /= live_priors.sum()
        expected = 0.7 * torch.logsumexp(
            q_values / 0.7 + torch.log(live_priors), dim=0
        )

        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


class CommonContinuationContractTests(unittest.TestCase):
    def test_smdp_target_uses_duration_discount_and_scaled_reward_once(self):
        target = smdp_target(
            reward=torch.tensor(12.0),
            duration=torch.tensor(4),
            done=torch.tensor(False),
            next_value=torch.tensor(7.0),
            gamma=0.9,
            reward_scale=0.25,
        )
        expected = 0.25 * 12.0 + (0.9**4) * 7.0

        self.assertAlmostEqual(float(target), expected, places=6)

    def test_terminal_target_never_bootstraps(self):
        target = smdp_target(
            reward=torch.tensor(12.0),
            duration=torch.tensor(400),
            done=torch.tensor(True),
            next_value=torch.tensor(1.0e9),
            gamma=0.99,
            reward_scale=0.25,
        )

        self.assertAlmostEqual(float(target), 3.0, places=6)

    def test_all_macro_modes_share_the_same_continuation_semantics(self):
        next_value = torch.tensor((5.0, 5.0, 5.0))
        targets = smdp_target(
            reward=torch.zeros(3),
            duration=torch.tensor((2, 5, 9)),
            done=torch.zeros(3, dtype=torch.bool),
            next_value=next_value,
            gamma=0.95,
        )
        expected = torch.tensor(
            tuple((0.95**duration) * 5.0 for duration in (2, 5, 9))
        )

        torch.testing.assert_close(targets, expected, atol=1e-6, rtol=0)


class ExplicitCellExecutionContractTests(unittest.TestCase):
    def test_selected_cell_is_bound_and_executed_without_reg_rescoring(self):
        env, state, registry, accept = _explicit_registry_system()
        block = env.blocks[0]
        yard = YardSnapshot.from_env(env)
        block_view = BlockView(
            block.label,
            block.position,
            float(block.get_remaining_storage_time()),
        )
        candidates = shared_candidate_mask(yard, block_view, block.position)
        self.assertGreater(len(candidates), 1)
        chosen_cell = candidates[-1]

        with patch.object(
            registry.source, "preview", wraps=registry.source.preview
        ) as reg_preview, patch.object(
            registry.source, "propose", wraps=registry.source.propose
        ) as reg_propose:
            preview = registry.preview_assignment_for_cell(block, chosen_cell)
            estimate = accept.estimate_duration_for_preview(preview)
            accept.bind_estimate(estimate)
            self.assertIsNone(block.storage_location)
            _run_option(env, state, accept)

        self.assertEqual(reg_preview.call_count, 0)
        self.assertEqual(reg_propose.call_count, 0)
        self.assertEqual(block.storage_location, chosen_cell)
        self.assertEqual(accept.episode_reservation_bound_count, 1)
        self.assertEqual(accept.episode_reservation_commit_count, 1)
        self.assertEqual(accept.episode_reservation_execution_match_count, 1)
        self.assertEqual(accept.episode_reservation_invalidation_count, 0)
        self.assertEqual(accept.last_outcome["proposal_id"], preview.proposal_id)
        self.assertTrue(accept.last_outcome["reservation_execution_match"])
        self.assertEqual(registry.decisions[0]["chosen_cell"], chosen_cell)
        self.assertEqual(registry.decisions[0]["proposal_id"], preview.proposal_id)


class ReplayCopyContractTests(unittest.TestCase):
    @staticmethod
    def _state(offset=0.0):
        return FullyLearnedState(
            global_features=np.full(
                len(GLOBAL_FEATURE_NAMES) + len(HISTORY_FEATURE_NAMES),
                offset,
                dtype=np.float32,
            ),
            block_features=np.full(
                (2, len(BLOCK_FEATURE_NAMES)), offset, dtype=np.float32
            ),
            block_mask=np.asarray((True, False), dtype=bool),
        )

    def test_replay_owns_numeric_copies_of_variable_candidate_records(self):
        state = self._state(1.0)
        next_state = self._state(2.0)
        chosen = np.full(len(CANDIDATE_FEATURE_NAMES), 3.0, np.float32)
        next_features = np.full(
            (3, len(CANDIDATE_FEATURE_NAMES)), 4.0, np.float32
        )
        next_priors = np.asarray((0.0, 5.0, 0.0), dtype=np.float32)
        next_modes = np.asarray((0, 0, 2), dtype=np.int64)
        transition = FullyLearnedTransition(
            state=state,
            candidate_features=chosen,
            candidate_prior=5.0,
            candidate_mode=0,
            reward=12.5,
            duration=7,
            next_state=next_state,
            next_candidate_features=next_features,
            next_candidate_priors=next_priors,
            next_candidate_modes=next_modes,
            done=False,
            terminal=False,
            truncated=False,
            failed=False,
        )
        replay = FullyLearnedReplayBuffer(capacity=4)

        replay.add(transition)
        state.global_features[:] = 99.0
        state.block_features[:] = 99.0
        chosen[:] = 99.0
        next_state.global_features[:] = 99.0
        next_features[:] = 99.0
        next_priors[:] = 99.0
        next_modes[:] = 99
        stored = replay.sample(1, random.Random(0))[0]

        np.testing.assert_array_equal(stored.state.global_features, 1.0)
        np.testing.assert_array_equal(stored.state.block_features, 1.0)
        np.testing.assert_array_equal(stored.candidate_features, 3.0)
        np.testing.assert_array_equal(stored.next_state.global_features, 2.0)
        np.testing.assert_array_equal(stored.next_candidate_features, 4.0)
        np.testing.assert_array_equal(
            stored.next_candidate_priors, (0.0, 5.0, 0.0)
        )
        np.testing.assert_array_equal(stored.next_candidate_modes, (0, 0, 2))
        self.assertEqual(stored.next_candidate_features.shape[0], 3)
        self.assertFalse(hasattr(stored, "option"))

    def test_replay_round_trip_copies_records_again(self):
        state = self._state(1.0)
        transition = FullyLearnedTransition(
            state=state,
            candidate_features=np.zeros(
                len(CANDIDATE_FEATURE_NAMES), dtype=np.float32
            ),
            candidate_prior=0.0,
            candidate_mode=2,
            reward=0.0,
            duration=1,
            next_state=None,
            next_candidate_features=np.empty(
                (0, len(CANDIDATE_FEATURE_NAMES)), dtype=np.float32
            ),
            next_candidate_priors=np.empty(0, dtype=np.float32),
            next_candidate_modes=np.empty(0, dtype=np.int64),
            done=True,
            terminal=True,
            truncated=False,
            failed=False,
        )
        original = FullyLearnedReplayBuffer(2)
        original.add(transition)
        restored = FullyLearnedReplayBuffer(2)

        restored.load_state_dict(original.state_dict())
        original.memory[0].state.global_features[:] = 77.0

        np.testing.assert_array_equal(
            restored.memory[0].state.global_features, 1.0
        )

    def test_mode_balanced_sample_caps_a_dominant_defer_bucket(self):
        replay = FullyLearnedReplayBuffer(100)
        next_id = 0
        for mode, count in ((0, 5), (1, 5), (2, 90)):
            for _ in range(count):
                replay.add(
                    FullyLearnedTransition(
                        state=self._state(float(next_id)),
                        candidate_features=np.zeros(
                            len(CANDIDATE_FEATURE_NAMES), np.float32
                        ),
                        candidate_prior=0.0,
                        candidate_mode=mode,
                        reward=float(next_id),
                        duration=1,
                        next_state=None,
                        next_candidate_features=np.empty(
                            (0, len(CANDIDATE_FEATURE_NAMES)), np.float32
                        ),
                        next_candidate_priors=np.empty(0, np.float32),
                        next_candidate_modes=np.empty(0, np.int64),
                        done=True,
                        terminal=True,
                        truncated=False,
                        failed=False,
                    )
                )
                next_id += 1

        sample = replay.sample_mode_balanced(12, random.Random(SEED))

        self.assertEqual(Counter(item.candidate_mode for item in sample), {
            0: 4,
            1: 4,
            2: 4,
        })
        self.assertEqual(len({item.reward for item in sample}), 12)

    def test_mode_balanced_sample_fills_scarce_modes_without_replacement(self):
        replay = FullyLearnedReplayBuffer(20)
        modes = (0, 1) + (2,) * 18
        for index, mode in enumerate(modes):
            replay.add(
                FullyLearnedTransition(
                    state=self._state(float(index)),
                    candidate_features=np.zeros(
                        len(CANDIDATE_FEATURE_NAMES), np.float32
                    ),
                    candidate_prior=0.0,
                    candidate_mode=mode,
                    reward=float(index),
                    duration=1,
                    next_state=None,
                    next_candidate_features=np.empty(
                        (0, len(CANDIDATE_FEATURE_NAMES)), np.float32
                    ),
                    next_candidate_priors=np.empty(0, np.float32),
                    next_candidate_modes=np.empty(0, np.int64),
                    done=True,
                    terminal=True,
                    truncated=False,
                    failed=False,
                )
            )

        sample = replay.sample_mode_balanced(9, random.Random(SEED))

        counts = Counter(item.candidate_mode for item in sample)
        self.assertEqual(counts, {0: 1, 1: 1, 2: 7})
        self.assertEqual(len({item.reward for item in sample}), 9)


class SharedNetworkContractTests(unittest.TestCase):
    def _network_fixture(self):
        env, _, registry, _ = _explicit_registry_system()
        config, _, _ = _small_fully_learned_agent(env, registry)
        network = FullyLearnedQNetwork(
            config, registry.source.network, seed=SEED
        )
        generator = torch.Generator().manual_seed(SEED)
        global_features = torch.randn(
            1,
            len(GLOBAL_FEATURE_NAMES) + len(HISTORY_FEATURE_NAMES),
            generator=generator,
        )
        block_features = torch.randn(
            1, 4, len(BLOCK_FEATURE_NAMES), generator=generator
        )
        block_mask = torch.tensor(((True, True, True, False),))
        candidate_features = torch.randn(
            1, 5, len(CANDIDATE_FEATURE_NAMES), generator=generator
        )
        candidate_features[..., :3] = 0.0
        candidate_features[:, :3, 0] = 1.0
        candidate_features[:, 3, 1] = 1.0
        candidate_features[:, 4, 2] = 1.0
        priors = torch.tensor(((5.0, 0.0, 0.0, 0.0, 0.0),))
        return (
            network,
            global_features,
            block_features,
            block_mask,
            candidate_features,
            priors,
        )

    def test_candidate_permutation_only_permutes_shared_scalar_scores(self):
        (
            network,
            globals_,
            blocks,
            block_mask,
            candidates,
            priors,
        ) = self._network_fixture()
        permutation = torch.tensor((4, 2, 0, 3, 1))

        q = network(globals_, blocks, block_mask, candidates, priors, 0.4)
        permuted = network(
            globals_,
            blocks,
            block_mask,
            candidates[:, permutation],
            priors[:, permutation],
            0.4,
        )

        self.assertEqual(q.shape, (1, 5))
        self.assertEqual(network.q_head.out_features, 1)
        torch.testing.assert_close(
            permuted, q[:, permutation], atol=1e-6, rtol=0
        )

    def test_block_permutation_leaves_every_candidate_score_unchanged(self):
        (
            network,
            globals_,
            blocks,
            block_mask,
            candidates,
            priors,
        ) = self._network_fixture()
        permutation = torch.tensor((2, 0, 3, 1))

        q = network(globals_, blocks, block_mask, candidates, priors)
        permuted = network(
            globals_,
            blocks[:, permutation],
            block_mask[:, permutation],
            candidates,
            priors,
        )

        torch.testing.assert_close(permuted, q, atol=1e-6, rtol=0)

    def test_spatial_encoder_gradient_control_is_explicit(self):
        (
            network,
            globals_,
            blocks,
            block_mask,
            candidates,
            priors,
        ) = self._network_fixture()
        network.train()
        network.set_spatial_trainable(False)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in network.spatial_encoder.parameters()
            )
        )
        network.zero_grad(set_to_none=True)
        network(globals_, blocks, block_mask, candidates, priors).sum().backward()
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in network.spatial_encoder.parameters()
            )
        )

        network.set_spatial_trainable(True)
        network.zero_grad(set_to_none=True)
        network(globals_, blocks, block_mask, candidates, priors).sum().backward()
        spatial_gradients = [
            parameter.grad
            for parameter in network.spatial_encoder.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(spatial_gradients)
        self.assertTrue(
            any(bool((gradient.abs() > 0).any()) for gradient in spatial_gradients)
        )


class PhaseAndCandidateContractTests(unittest.TestCase):
    def test_temporal_locks_teacher_cell_spatial_calibrates_before_joint(self):
        env, state, registry, accept = _explicit_registry_system()
        _, _, agent = _small_fully_learned_agent(env, registry)

        agent.set_training_phase("temporal")
        temporal = agent.capture_snapshot(state)
        temporal_accepts = [
            candidate for candidate in temporal.candidates
            if candidate.mode == "accept"
        ]
        self.assertEqual(len(temporal_accepts), 1)
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in agent.Q_local.spatial_encoder.parameters()
            )
        )

        agent.set_training_phase("spatial")
        spatial = agent.capture_snapshot(state)
        spatial_accepts = [
            candidate for candidate in spatial.candidates
            if candidate.mode == "accept"
        ]

        self.assertGreater(len(spatial_accepts), len(temporal_accepts))
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in agent.Q_local.spatial_encoder.parameters()
            )
        )

        agent.set_training_phase("joint")
        joint = agent.capture_snapshot(state)
        joint_accepts = [
            candidate for candidate in joint.candidates
            if candidate.mode == "accept"
        ]
        keys = [candidate.key for candidate in joint.candidates]

        self.assertGreater(len(joint_accepts), 1)
        self.assertGreater(len(joint_accepts), len(temporal_accepts))
        self.assertEqual(len(joint_accepts), len(spatial_accepts))
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(agent.effective_teacher_coefficient, 0.0)
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in agent.Q_local.spatial_encoder.parameters()
            )
        )
        self.assertEqual(registry.decision_count, 0)
        self.assertEqual(registry.decisions, [])
        self.assertIsNone(accept.active_reservation)

    def test_joint_accept_candidates_equal_the_live_shared_mask(self):
        env, state, registry, _ = _explicit_registry_system()
        _, _, agent = _small_fully_learned_agent(env, registry)
        block = env.blocks[0]
        yard = YardSnapshot.from_env(env)
        view = BlockView(
            block.label,
            block.position,
            float(block.get_remaining_storage_time()),
        )
        live_cells = set(shared_candidate_mask(yard, view, block.position))

        agent.set_training_phase("joint")
        snapshot = agent.capture_snapshot(state)
        represented = {
            candidate.chosen_cell
            for candidate in snapshot.candidates
            if candidate.mode == "accept"
        }

        self.assertEqual(represented, live_cells)
        for candidate in snapshot.candidates:
            expected_prefix = {
                "accept": "accept:",
                "retrieve": "retrieve:",
                "defer": "defer",
            }[candidate.mode]
            self.assertTrue(candidate.key.startswith(expected_prefix))

    def test_unsupervised_spatial_and_joint_snapshots_do_not_query_frozen_reg(self):
        env, state, registry, _ = _explicit_registry_system()
        _, _, agent = _small_fully_learned_agent(env, registry)

        with patch.object(
            registry.source,
            "score_candidates",
            side_effect=AssertionError("deployment snapshot queried frozen REG"),
        ) as score_call:
            for phase in ("spatial", "joint"):
                with self.subTest(phase=phase):
                    agent.set_training_phase(phase)
                    agent.set_teacher_supervision(False)
                    snapshot = agent.capture_snapshot(state)
                    self.assertGreater(
                        sum(
                            candidate.mode == "accept"
                            for candidate in snapshot.candidates
                        ),
                        1,
                    )
                    self.assertTrue(
                        all(
                            candidate.teacher_score is None
                            for candidate in snapshot.candidates
                            if candidate.mode == "accept"
                        )
                    )

        self.assertEqual(score_call.call_count, 0)

    def test_supervised_spatial_snapshot_has_complete_full_cell_reg_ranking(self):
        env, state, registry, _ = _explicit_registry_system()
        _, _, agent = _small_fully_learned_agent(env, registry)
        block = env.blocks[0]
        yard = YardSnapshot.from_env(env)
        view = BlockView(
            block.label,
            block.position,
            float(block.get_remaining_storage_time()),
        )
        live_cells = set(shared_candidate_mask(yard, view, block.position))
        agent.set_training_phase("spatial")
        agent.set_teacher_supervision(True)

        with patch.object(
            registry.source,
            "score_candidates",
            wraps=registry.source.score_candidates,
        ) as score_call:
            snapshot = agent.capture_snapshot(state)

        accepts = [
            candidate for candidate in snapshot.candidates
            if candidate.mode == "accept"
        ]
        self.assertEqual({candidate.chosen_cell for candidate in accepts}, live_cells)
        self.assertTrue(all(candidate.teacher_score is not None for candidate in accepts))
        teacher = max(
            accepts,
            key=lambda candidate: (
                float(candidate.teacher_score),
                -accepts.index(candidate),
            ),
        )
        self.assertEqual(snapshot.teacher_key, teacher.key)
        self.assertEqual(score_call.call_count, 1)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in agent.Q_local.spatial_encoder.parameters()
            )
        )

    def test_joint_accept_exploration_is_restricted_to_teacher_top_k(self):
        env, state, registry, _ = _explicit_registry_system()
        _, _, agent = _small_fully_learned_agent(
            env,
            registry,
            joint_accept_exploration_top_k=3,
        )
        agent.set_training_phase("joint")
        agent.set_teacher_supervision(True)
        snapshot = agent.capture_snapshot(state)
        accepts = [
            candidate for candidate in snapshot.candidates
            if candidate.mode == "accept"
        ]
        ranked = sorted(
            accepts,
            key=lambda candidate: (
                -float(candidate.teacher_score),
                accepts.index(candidate),
            ),
        )
        expected_keys = {candidate.key for candidate in ranked[:3]}
        self.assertGreater(len(accepts), 3)

        selected_keys = {
            accepts[agent._explore(accepts)].key
            for _ in range(100)
        }

        self.assertTrue(selected_keys)
        self.assertTrue(selected_keys <= expected_keys)
        self.assertGreater(len(selected_keys), 1)

    def test_phase_change_clears_incompatible_replay(self):
        env, _, registry, _ = _explicit_registry_system()
        _, _, agent = _small_fully_learned_agent(env, registry)
        state = ReplayCopyContractTests._state(1.0)
        transition = FullyLearnedTransition(
            state=state,
            candidate_features=np.zeros(
                len(CANDIDATE_FEATURE_NAMES), dtype=np.float32
            ),
            candidate_prior=0.0,
            candidate_mode=0,
            reward=0.0,
            duration=1,
            next_state=None,
            next_candidate_features=np.empty(
                (0, len(CANDIDATE_FEATURE_NAMES)), dtype=np.float32
            ),
            next_candidate_priors=np.empty(0, dtype=np.float32),
            next_candidate_modes=np.empty(0, dtype=np.int64),
            done=True,
            terminal=True,
            truncated=False,
            failed=False,
        )
        agent.replay.add(transition)

        agent.set_training_phase("temporal")

        self.assertEqual(len(agent.replay), 0)
        self.assertEqual(agent.replay_clears, 1)
        self.assertEqual(agent.phase_transitions[-1]["to"], "temporal")


class ImitationAndTerminalContractTests(unittest.TestCase):
    def test_behavior_cloning_is_finite_nontrivial_and_omits_teacher_prior(self):
        env, state, registry, _ = _explicit_registry_system()
        _, _, agent = _small_fully_learned_agent(env, registry)
        agent.set_training_phase("imitation")
        before = agent.Q_local.q_head.weight.detach().clone()

        with patch.object(
            agent,
            "_score_snapshot",
            wraps=agent._score_snapshot,
        ) as score_call:
            agent.select_action(state, eps=0.0, teacher_forcing=True)
            loss = agent.learn_imitation()

        self.assertIsNotNone(loss)
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(loss, 1.0e-6)
        self.assertEqual(agent.imitation_steps, 1)
        self.assertFalse(torch.equal(before, agent.Q_local.q_head.weight))
        self.assertEqual(score_call.call_count, 1)
        self.assertEqual(score_call.call_args.kwargs["teacher_coefficient"], 0.0)

    def test_spatial_listwise_distillation_uses_every_ranked_accept_cell(self):
        env, state, registry, _ = _explicit_registry_system()
        _, _, agent = _small_fully_learned_agent(env, registry)
        agent.set_training_phase("spatial")
        agent.set_teacher_supervision(True)
        snapshot = agent.capture_snapshot(state)
        accepts = [
            candidate for candidate in snapshot.candidates
            if candidate.mode == "accept"
        ]
        spatial_before = {
            name: parameter.detach().clone()
            for name, parameter in agent.Q_local.spatial_encoder.named_parameters()
        }
        q_head_before = agent.Q_local.q_head.weight.detach().clone()

        agent.select_action(state, eps=0.0, teacher_forcing=True)
        loss = agent.learn_imitation()

        self.assertGreater(len(accepts), 1)
        self.assertEqual(
            len([candidate for candidate in accepts if candidate.teacher_score is not None]),
            len(accepts),
        )
        self.assertIsNotNone(loss)
        self.assertGreater(agent.distillation_loss_history[-1], 1.0e-8)
        self.assertEqual(agent.spatial_distillation_steps, 1)
        self.assertFalse(torch.equal(q_head_before, agent.Q_local.q_head.weight))
        for name, parameter in agent.Q_local.spatial_encoder.named_parameters():
            torch.testing.assert_close(parameter, spatial_before[name])

    def test_checkpoint_restores_lr_scales_and_distillation_progress(self):
        env, _, registry, _ = _explicit_registry_system()
        _, _, agent = _small_fully_learned_agent(env, registry)
        agent.set_training_phase("spatial")
        agent.set_learning_rate_scales(main=0.2, spatial=0.1)
        agent.spatial_distillation_steps = 17
        state = agent.checkpoint_state(include_replay=False)

        restored_env, _, restored_registry, _ = _explicit_registry_system()
        _, _, restored = _small_fully_learned_agent(
            restored_env, restored_registry
        )
        restored.load_checkpoint_state(state, resumable=False)

        self.assertEqual(restored.training_phase, "spatial")
        self.assertEqual(restored.spatial_distillation_steps, 17)
        self.assertEqual(
            restored.learning_rate_scales,
            {"main": 0.2, "spatial": 0.1},
        )
        actual_lrs = {
            group["group_name"]: group["lr"]
            for group in restored.optimizer.param_groups
        }
        self.assertAlmostEqual(
            actual_lrs["main"], restored.base_learning_rates["main"] * 0.2
        )
        self.assertAlmostEqual(
            actual_lrs["spatial"], restored.base_learning_rates["spatial"] * 0.1
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in restored.Q_local.spatial_encoder.parameters()
            )
        )

    def test_temporal_behavior_retains_teacher_label_without_forcing_action(self):
        env, state, registry, _ = _explicit_registry_system()
        _, _, agent = _small_fully_learned_agent(env, registry)
        agent.set_training_phase("temporal")
        snapshot = agent.capture_snapshot(state)
        teacher_index = next(
            index
            for index, candidate in enumerate(snapshot.candidates)
            if candidate.key == snapshot.teacher_key
        )
        nonteacher_index = next(
            index for index in range(len(snapshot.candidates))
            if index != teacher_index
        )

        with patch.object(agent, "_hierarchical_map", return_value=nonteacher_index), patch.object(
            agent, "_score_snapshot", wraps=agent._score_snapshot
        ) as score_call:
            agent.select_action(
                state,
                eps=0.0,
                teacher_probability=0.0,
                retain_teacher_label=True,
            )
            loss = agent.learn_imitation(weight=0.25)

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0.0)
        self.assertEqual(agent.retention_steps, 1)
        self.assertEqual(agent.selection_source_counts["greedy"], 1)
        self.assertEqual(score_call.call_args.kwargs["teacher_coefficient"], 0.0)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in agent.Q_local.spatial_encoder.parameters()
            )
        )

    def test_teacher_mixture_precedes_epsilon_exploration(self):
        env, state, registry, _ = _explicit_registry_system()
        _, _, agent = _small_fully_learned_agent(env, registry)
        agent.set_training_phase("temporal")
        snapshot = agent.capture_snapshot(state)

        agent.select_action(
            state,
            eps=1.0,
            teacher_probability=1.0,
            retain_teacher_label=True,
        )

        self.assertEqual(agent._active_candidate.key, snapshot.teacher_key)
        self.assertEqual(
            agent.selection_source_counts["temporal_teacher_mixture"], 1
        )
        self.assertFalse(agent.decisions[-1]["explored"])

    def test_temporal_td_prediction_and_target_ignore_teacher_priors(self):
        env, state, registry, _ = _explicit_registry_system()
        _, _, agent = _small_fully_learned_agent(env, registry)
        agent.set_training_phase("temporal")
        snapshot = agent.capture_snapshot(state)
        candidate = snapshot.candidates[0]
        features = np.stack(
            [item.features for item in snapshot.candidates]
        ).astype(np.float32)
        modes = np.asarray(
            [item.mode_id for item in snapshot.candidates], dtype=np.int64
        )
        zeros = np.zeros(len(snapshot.candidates), dtype=np.float32)
        shifted = np.linspace(
            0.0, 100.0, len(snapshot.candidates), dtype=np.float32
        )
        base = FullyLearnedTransition(
            state=snapshot.state,
            candidate_features=candidate.features,
            candidate_prior=0.0,
            candidate_mode=candidate.mode_id,
            reward=2.0,
            duration=1,
            next_state=snapshot.state,
            next_candidate_features=features,
            next_candidate_priors=zeros,
            next_candidate_modes=modes,
            done=False,
            terminal=False,
            truncated=False,
            failed=False,
        )
        changed = replace(
            base,
            candidate_prior=100.0,
            next_candidate_priors=shifted,
        )

        prediction_a, target_a = agent._td_batch([base])
        prediction_b, target_b = agent._td_batch([changed])

        torch.testing.assert_close(prediction_a, prediction_b)
        torch.testing.assert_close(target_a, target_b)

    @staticmethod
    def _close_synthetic_macro(agent, state, *, env_terminal, truncated):
        snapshot = agent.capture_snapshot(state)
        candidate = snapshot.candidates[0]
        agent._active_state = snapshot.state.copy()
        agent._active_candidate = candidate
        agent._active_return = 0.0
        agent._active_discount = 1.0
        agent._active_duration = 0
        stored = agent.process_step(
            state,
            2.0,
            done=True,
            terminated=True,
            failed=False,
            env_terminal=env_terminal,
            truncated=truncated,
            store_transition=True,
        )
        if not stored:
            raise AssertionError("synthetic terminal macro was not stored")
        return agent.replay.memory[-1]

    def test_natural_terminal_and_time_limit_have_distinct_replay_flags(self):
        terminal_env, terminal_state, terminal_registry, _ = (
            _explicit_registry_system()
        )
        _, _, terminal_agent = _small_fully_learned_agent(
            terminal_env, terminal_registry
        )
        natural = self._close_synthetic_macro(
            terminal_agent,
            terminal_state,
            env_terminal=True,
            truncated=False,
        )

        truncated_env, truncated_state, truncated_registry, _ = (
            _explicit_registry_system()
        )
        _, _, truncated_agent = _small_fully_learned_agent(
            truncated_env, truncated_registry
        )
        time_limit = self._close_synthetic_macro(
            truncated_agent,
            truncated_state,
            env_terminal=False,
            truncated=True,
        )

        self.assertTrue(natural.done)
        self.assertTrue(natural.terminal)
        self.assertFalse(natural.truncated)
        self.assertFalse(natural.failed)
        self.assertTrue(time_limit.done)
        self.assertFalse(time_limit.terminal)
        self.assertTrue(time_limit.truncated)
        self.assertFalse(time_limit.failed)
        self.assertEqual(natural.boundary_penalty, 0.0)
        self.assertAlmostEqual(
            time_limit.boundary_penalty,
            truncated_agent.gamma * truncated_agent.config.failure_penalty,
        )
        _, target = truncated_agent._td_batch([time_limit])
        self.assertAlmostEqual(
            float(target.item()),
            truncated_agent.reward_scale
            * (2.0 + truncated_agent.gamma * truncated_agent.config.failure_penalty),
            places=6,
        )

    def test_failure_and_time_limit_do_not_double_boundary_penalty(self):
        env, state, registry, _ = _explicit_registry_system()
        _, _, agent = _small_fully_learned_agent(env, registry)
        snapshot = agent.capture_snapshot(state)
        agent._active_state = snapshot.state.copy()
        agent._active_candidate = snapshot.candidates[0]
        agent._active_return = 0.0
        agent._active_discount = 1.0
        agent._active_duration = 0

        agent.process_step(
            state,
            2.0,
            done=True,
            terminated=True,
            failed=True,
            env_terminal=False,
            truncated=True,
            store_transition=True,
        )

        transition = agent.replay.memory[-1]
        self.assertAlmostEqual(
            transition.boundary_penalty,
            agent.gamma * agent.config.failure_penalty,
        )

    def test_method_failure_keeps_environment_return_and_penalty_separate(self):
        env, state, registry, _ = _explicit_registry_system()
        _, _, agent = _small_fully_learned_agent(env, registry)
        snapshot = agent.capture_snapshot(state)
        agent._active_state = snapshot.state.copy()
        agent._active_candidate = snapshot.candidates[0]
        agent._active_return = 0.0
        agent._active_discount = 1.0
        agent._active_duration = 0

        stored = agent.process_step(
            state,
            2.0,
            done=True,
            terminated=True,
            failed=True,
            env_terminal=False,
            truncated=False,
            store_transition=True,
        )

        self.assertTrue(stored)
        transition = agent.replay.memory[-1]
        self.assertEqual(transition.reward, 2.0)
        self.assertAlmostEqual(
            transition.boundary_penalty,
            agent.gamma * agent.config.failure_penalty,
        )
        self.assertTrue(transition.failed)
        self.assertFalse(transition.terminal)
        self.assertFalse(transition.truncated)

    def test_infeasible_next_boundary_closes_current_macro_once(self):
        env, state, registry, _ = _explicit_registry_system()
        _, _, agent = _small_fully_learned_agent(env, registry)
        snapshot = agent.capture_snapshot(state)
        agent._active_state = snapshot.state.copy()
        agent._active_candidate = snapshot.candidates[0]
        agent._active_return = 0.0
        agent._active_discount = 1.0
        agent._active_duration = 0

        with patch.object(
            agent,
            "capture_snapshot",
            side_effect=FullyLearnedInfeasible("no next macro"),
        ):
            stored = agent.process_step(
                state,
                3.0,
                done=False,
                terminated=True,
                failed=False,
                env_terminal=False,
                truncated=False,
                store_transition=True,
            )

        self.assertTrue(stored)
        self.assertEqual(len(agent.replay), 1)
        transition = agent.replay.memory[0]
        self.assertTrue(transition.done)
        self.assertTrue(transition.failed)
        self.assertIsNone(transition.next_state)
        self.assertEqual(agent.last_boundary_failure_reason, "no next macro")
        self.assertIsNone(agent._active_candidate)
        self.assertEqual(len(agent.history), 1)
        self.assertTrue(agent.history[0]["failed"])


if __name__ == "__main__":
    unittest.main()
