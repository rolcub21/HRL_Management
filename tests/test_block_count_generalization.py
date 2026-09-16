from dataclasses import replace
from collections import Counter
import math
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from example.Options.selector_v5 import ExplicitCellAssignmentRegistry
from example.controller_observation import (
    BLOCK_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    OnlineManifestTimingObservationEncoder,
)
from example.controller_options import (
    FULLY_LEARNED_RESERVED_ACTION_INTERFACE,
    build_controller_options,
)
from example.episode_instance import EpisodeInstance
from example.small_rooms_env import SmallRoomsEnv
from example.yard_geometry import geometry_metadata, make_shipyard_env
from fully_learned_hierarchy import (
    CANDIDATE_FEATURE_NAMES,
    HISTORY_FEATURE_NAMES,
    FullyLearnedConfig,
    FullyLearnedHierarchyAgent,
    FullyLearnedReplayBuffer,
    FullyLearnedState,
    FullyLearnedTransition,
)
from PSLAP.reg_selector_v5 import REGV5AssignmentSource, REGV5Config


SEED = 56_701


def _selector_config():
    return REGV5Config(
        block_embedding_dim=8,
        candidate_embedding_dim=8,
        context_dim=16,
        replay_size=8,
        batch_size=1,
        min_replay_size=1,
        updates_per_episode=1,
    )


def _make_stack(number_blocks, *, batch_size=1):
    env, encoder, source = _make_runtime(number_blocks)
    config = FullyLearnedConfig(
        block_embedding_dim=8,
        global_embedding_dim=8,
        candidate_embedding_dim=8,
        context_dim=16,
        history_length=4,
    )
    agent = FullyLearnedHierarchyAgent(
        env,
        encoder,
        config=config,
        spatial_network=source.network,
        seed=SEED,
        device="cpu",
        batch_size=batch_size,
        buffer_size=16,
        update_every=1,
    )
    return env, encoder, agent


def _make_runtime(number_blocks):
    env = make_shipyard_env(
        arrival_rate=0.5,
        proc_mean=50,
        grid_rows=10,
        grid_cols=10,
        exit_width=3,
        number_blocks=number_blocks,
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
    encoder = OnlineManifestTimingObservationEncoder(env)
    return env, encoder, source


def _numeric_state(number_blocks=2, offset=0.0):
    return FullyLearnedState(
        global_features=np.full(
            len(GLOBAL_FEATURE_NAMES) + len(HISTORY_FEATURE_NAMES),
            offset,
            dtype=np.float32,
        ),
        block_features=np.full(
            (number_blocks, len(BLOCK_FEATURE_NAMES)),
            offset,
            dtype=np.float32,
        ),
        block_mask=np.ones(number_blocks, dtype=bool),
    )


def _terminal_transition(*, regime_id="legacy", mode=0, marker=0.0):
    candidate = np.zeros(len(CANDIDATE_FEATURE_NAMES), np.float32)
    candidate[int(mode)] = 1.0
    return FullyLearnedTransition(
        state=_numeric_state(offset=marker),
        candidate_features=candidate,
        candidate_prior=0.0,
        candidate_mode=int(mode),
        reward=float(marker),
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
        regime_id=regime_id,
    )


class BlockCountEnvironmentContracts(unittest.TestCase):
    def test_environment_uses_requested_positive_block_count(self):
        for number_blocks in (1, 7, 53):
            with self.subTest(number_blocks=number_blocks):
                env = SmallRoomsEnv(
                    choose_storage=False,
                    arrival_rate=0.5,
                    proc_mean=50,
                    number_blocks=number_blocks,
                )

                self.assertEqual(env.number_blocks, number_blocks)
                self.assertEqual(len(env.blocks), number_blocks)
                self.assertEqual(
                    [block.label for block in env.blocks],
                    [f"B{index}" for index in range(1, number_blocks + 1)],
                )

        for number_blocks in (0, -1):
            with self.subTest(number_blocks=number_blocks):
                with self.assertRaisesRegex(ValueError, "number_blocks"):
                    SmallRoomsEnv(number_blocks=number_blocks)

    def test_factory_and_geometry_metadata_preserve_block_count(self):
        env = make_shipyard_env(
            arrival_rate=0.4,
            proc_mean=60,
            grid_rows=9,
            grid_cols=11,
            exit_width=2,
            number_blocks=17,
        )
        metadata = geometry_metadata(env, requested_exit_width=2)

        self.assertEqual(env.number_blocks, 17)
        self.assertEqual(metadata["block_count"], 17)
        self.assertEqual(metadata["storage_cell_count"], len(env.storage_positions))
        self.assertAlmostEqual(
            metadata["nominal_storage_density"],
            17 / len(env.storage_positions),
        )


class BlockCountEpisodeInstanceContracts(unittest.TestCase):
    def test_sampling_reset_and_json_round_trip_keep_requested_cardinality(self):
        env = make_shipyard_env(
            arrival_rate=0.5,
            proc_mean=50,
            number_blocks=7,
        )
        instance = env.sample_episode_instance(SEED)
        restored = EpisodeInstance.from_json(instance.to_json())

        self.assertEqual(instance.number_blocks, 7)
        self.assertEqual(len(instance.arrival_steps), 7)
        self.assertEqual(len(instance.storage_steps_needed), 7)
        self.assertEqual(restored, instance)
        self.assertEqual(restored.instance_id, instance.instance_id)

        env.reset(instance=restored)
        self.assertEqual(
            [block.arrival_step for block in env.blocks],
            list(instance.arrival_steps),
        )
        self.assertEqual(
            [block.storage_steps_needed for block in env.blocks],
            list(instance.storage_steps_needed),
        )

    def test_reset_rejects_instance_with_different_block_count(self):
        seven_block_env = make_shipyard_env(
            arrival_rate=0.5,
            proc_mean=50,
            number_blocks=7,
        )
        eleven_block_env = make_shipyard_env(
            arrival_rate=0.5,
            proc_mean=50,
            number_blocks=11,
        )
        instance = seven_block_env.sample_episode_instance(SEED)

        with self.assertRaisesRegex(ValueError, "number_blocks"):
            eleven_block_env.reset(instance=instance)

    def test_block_count_participates_in_schedule_identity(self):
        source = make_shipyard_env(
            arrival_rate=0.5,
            proc_mean=50,
            number_blocks=7,
        ).sample_episode_instance(SEED)
        extended = replace(
            source,
            arrival_steps=source.arrival_steps + (source.arrival_steps[-1] + 1,),
            storage_steps_needed=source.storage_steps_needed + (50,),
        )

        self.assertEqual(source.number_blocks, 7)
        self.assertEqual(extended.number_blocks, 8)
        self.assertNotEqual(source.schedule_id, extended.schedule_id)
        self.assertNotEqual(source.instance_id, extended.instance_id)


class VariableCardinalityFullyLearnedContracts(unittest.TestCase):
    def test_snapshot_and_action_interface_follow_environment_block_count(self):
        for number_blocks in (5, 13):
            with self.subTest(number_blocks=number_blocks):
                env, encoder, agent = _make_stack(number_blocks)
                state = env.reset(instance=env.sample_episode_instance(SEED))
                agent.set_training_phase("joint")
                agent.set_teacher_supervision(False)

                observation = encoder.capture(state)
                snapshot = agent.capture_snapshot(state)

                self.assertEqual(observation.block_features.shape[0], number_blocks)
                self.assertEqual(observation.block_mask.shape, (number_blocks,))
                self.assertEqual(snapshot.state.block_features.shape[0], number_blocks)
                self.assertEqual(snapshot.state.block_mask.shape, (number_blocks,))
                self.assertGreater(len(snapshot.candidates), 1)
                retrieval_options = [
                    option
                    for option in env.options
                    if option.__class__.__name__ == "StrictRetrieveDeliverOption"
                ]
                self.assertEqual(len(retrieval_options), number_blocks)

    def test_network_ignores_extra_masked_block_rows(self):
        env, _, agent = _make_stack(5)
        state = env.reset(instance=env.sample_episode_instance(SEED))
        agent.set_training_phase("joint")
        agent.set_teacher_supervision(False)
        snapshot = agent.capture_snapshot(state)
        candidate_features, priors, _ = agent._candidate_arrays(snapshot.candidates)

        globals_ = torch.from_numpy(snapshot.state.global_features)[None]
        blocks = torch.from_numpy(snapshot.state.block_features)[None]
        mask = torch.from_numpy(snapshot.state.block_mask)[None]
        candidates = torch.from_numpy(candidate_features)[None]
        candidate_priors = torch.from_numpy(priors)[None]
        extra_rows = torch.randn(
            1, 4, len(BLOCK_FEATURE_NAMES), generator=torch.Generator().manual_seed(SEED)
        )
        padded_blocks = torch.cat((blocks, extra_rows), dim=1)
        padded_mask = torch.cat(
            (mask, torch.zeros((1, 4), dtype=torch.bool)), dim=1
        )

        with torch.no_grad():
            original = agent.Q_local(
                globals_, blocks, mask, candidates, candidate_priors, 0.0
            )
            padded = agent.Q_local(
                globals_, padded_blocks, padded_mask, candidates, candidate_priors, 0.0
            )

        torch.testing.assert_close(original, padded, atol=1e-6, rtol=0)

    def test_mixed_block_and_candidate_counts_share_one_td_batch(self):
        _, _, agent = _make_stack(5, batch_size=2)
        agent.set_training_phase("joint")

        def state(number_blocks, offset):
            return FullyLearnedState(
                global_features=np.full(
                    len(GLOBAL_FEATURE_NAMES) + len(HISTORY_FEATURE_NAMES),
                    offset,
                    dtype=np.float32,
                ),
                block_features=np.full(
                    (number_blocks, len(BLOCK_FEATURE_NAMES)),
                    offset,
                    dtype=np.float32,
                ),
                block_mask=np.ones(number_blocks, dtype=bool),
            )

        def candidates(mode_ids, offset):
            features = np.full(
                (len(mode_ids), len(CANDIDATE_FEATURE_NAMES)),
                offset,
                dtype=np.float32,
            )
            features[:, :3] = 0.0
            for row, mode_id in enumerate(mode_ids):
                features[row, mode_id] = 1.0
            return features

        transitions = []
        for current_count, next_count, mode_ids, offset in (
            (3, 7, (0, 2), 0.1),
            (11, 5, (0, 0, 1, 1, 2), 0.2),
        ):
            next_features = candidates(mode_ids, offset + 0.1)
            transitions.append(
                FullyLearnedTransition(
                    state=state(current_count, offset),
                    candidate_features=candidates((0,), offset)[0],
                    candidate_prior=0.0,
                    candidate_mode=0,
                    reward=1.0,
                    duration=2,
                    next_state=state(next_count, offset + 0.1),
                    next_candidate_features=next_features,
                    next_candidate_priors=np.zeros(len(mode_ids), np.float32),
                    next_candidate_modes=np.asarray(mode_ids, np.int64),
                    done=False,
                    terminal=False,
                    truncated=False,
                    failed=False,
                )
            )

        globals_, blocks, masks = agent._pad_states(
            [transition.state for transition in transitions]
        )
        prediction, target = agent._td_batch(transitions)

        self.assertEqual(globals_.shape[0], 2)
        self.assertEqual(blocks.shape[:2], (2, 11))
        self.assertEqual(masks.sum(axis=1).tolist(), [3, 11])
        self.assertEqual(tuple(prediction.shape), (2,))
        self.assertEqual(tuple(target.shape), (2,))
        self.assertTrue(bool(torch.isfinite(prediction).all()))
        self.assertTrue(bool(torch.isfinite(target).all()))

        for transition in transitions:
            agent.replay.add(transition)
        loss = agent.learn()
        self.assertIsNotNone(loss)
        self.assertTrue(math.isfinite(loss))


class RuntimeRebindingContracts(unittest.TestCase):
    def test_rebind_replaces_only_runtime_and_episode_local_state(self):
        original_env, _, agent = _make_stack(5)
        replacement_env, replacement_encoder, _ = _make_runtime(13)
        agent.set_training_phase("joint")
        agent.step_count = 17
        agent.decision_count = 23
        agent.gradient_steps = 5
        agent.replay.add(
            _terminal_transition(regime_id="original", mode=0, marker=1.0)
        )
        agent.history.append(
            {"mode": "defer", "duration": 2, "return": -0.1, "failed": False}
        )
        agent.decision_counts["defer"] = 3
        agent.mode_counts["defer"] = 3
        agent.decisions.append({"selected_key": "defer"})
        agent.selection_source_counts["greedy"] = 3
        identities = {
            "Q_local": id(agent.Q_local),
            "Q_target": id(agent.Q_target),
            "optimizer": id(agent.optimizer),
            "replay": id(agent.replay),
        }
        weights = {
            name: value.detach().clone()
            for name, value in agent.Q_local.state_dict().items()
        }
        rng_state = agent.rng.getstate()

        agent.bind_runtime(
            replacement_env,
            replacement_encoder,
            regime_id="dense_13",
        )

        self.assertIs(agent.env, replacement_env)
        self.assertIs(agent.observation_encoder, replacement_encoder)
        self.assertIs(agent.builder.env, replacement_env)
        self.assertIs(agent.builder.encoder, replacement_encoder)
        self.assertEqual(agent.regime_id, "dense_13")
        self.assertEqual(agent.training_phase, "joint")
        self.assertEqual(agent.step_count, 17)
        self.assertEqual(agent.decision_count, 23)
        self.assertEqual(agent.gradient_steps, 5)
        self.assertEqual(agent.rng.getstate(), rng_state)
        self.assertEqual(
            identities,
            {
                "Q_local": id(agent.Q_local),
                "Q_target": id(agent.Q_target),
                "optimizer": id(agent.optimizer),
                "replay": id(agent.replay),
            },
        )
        self.assertEqual(len(agent.replay), 1)
        self.assertEqual(agent.replay.memory[0].regime_id, "original")
        for name, value in agent.Q_local.state_dict().items():
            torch.testing.assert_close(value, weights[name], atol=0, rtol=0)

        self.assertEqual(len(agent.history), 0)
        self.assertEqual(agent.decision_counts, Counter())
        self.assertEqual(agent.mode_counts, Counter())
        self.assertEqual(agent.decisions, [])
        self.assertEqual(agent.selection_source_counts, Counter())
        state = replacement_env.reset(
            instance=replacement_env.sample_episode_instance(SEED)
        )
        snapshot = agent.capture_snapshot(state)
        self.assertEqual(snapshot.state.block_features.shape[0], 13)
        self.assertIsNot(agent.env, original_env)

    def test_rebind_rejects_mid_macro_transactionally(self):
        original_env, _, agent = _make_stack(5)
        replacement_env, replacement_encoder, _ = _make_runtime(7)
        state = original_env.reset(
            instance=original_env.sample_episode_instance(SEED)
        )
        agent.set_training_phase("joint")
        agent.set_teacher_supervision(False)
        agent.select_action(state, eps=0.0)
        original_builder = agent.builder

        with self.assertRaisesRegex(RuntimeError, "mid-macro"):
            agent.bind_runtime(
                replacement_env,
                replacement_encoder,
                regime_id="replacement",
            )

        self.assertIs(agent.env, original_env)
        self.assertIs(agent.builder, original_builder)
        self.assertEqual(agent.regime_id, "legacy")

    def test_rebind_rejects_empty_id_and_mismatched_encoder(self):
        original_env, _, agent = _make_stack(5)
        replacement_env, replacement_encoder, _ = _make_runtime(7)
        unrelated_env, unrelated_encoder, _ = _make_runtime(9)

        with self.assertRaisesRegex(ValueError, "regime_id"):
            agent.bind_runtime(
                replacement_env,
                replacement_encoder,
                regime_id="",
            )
        self.assertIs(agent.env, original_env)

        with self.assertRaisesRegex(ValueError, "observation encoder"):
            agent.bind_runtime(
                replacement_env,
                unrelated_encoder,
                regime_id="replacement",
            )
        self.assertIs(agent.env, original_env)
        self.assertIs(unrelated_encoder.env, unrelated_env)


class RegimeModeReplayContracts(unittest.TestCase):
    def test_complete_regime_mode_matrix_receives_equal_cell_quotas(self):
        replay = FullyLearnedReplayBuffer(128)
        marker = 0
        for regime_id in ("ordinary", "contention"):
            for mode in (0, 1, 2):
                for _ in range(10):
                    replay.add(
                        _terminal_transition(
                            regime_id=regime_id,
                            mode=mode,
                            marker=marker,
                        )
                    )
                    marker += 1

        sample = replay.sample_regime_mode_balanced(12, random.Random(SEED))
        counts = Counter((item.regime_id, item.candidate_mode) for item in sample)

        self.assertEqual(
            counts,
            Counter(
                {
                    (regime_id, mode): 2
                    for regime_id in ("ordinary", "contention")
                    for mode in (0, 1, 2)
                }
            ),
        )
        self.assertEqual(len({item.reward for item in sample}), 12)

    def test_regime_quota_precedes_within_regime_mode_quota(self):
        replay = FullyLearnedReplayBuffer(128)
        marker = 0
        for mode in (0, 1, 2):
            for _ in range(12):
                replay.add(
                    _terminal_transition(
                        regime_id="three_modes",
                        mode=mode,
                        marker=marker,
                    )
                )
                marker += 1
        for _ in range(12):
            replay.add(
                _terminal_transition(
                    regime_id="one_mode",
                    mode=0,
                    marker=marker,
                )
            )
            marker += 1

        sample = replay.sample_regime_mode_balanced(8, random.Random(SEED))
        regime_counts = Counter(item.regime_id for item in sample)
        cell_counts = Counter(
            (item.regime_id, item.candidate_mode) for item in sample
        )

        self.assertEqual(regime_counts, {"one_mode": 4, "three_modes": 4})
        self.assertEqual(cell_counts[("one_mode", 0)], 4)
        self.assertEqual(
            sum(cell_counts[("three_modes", mode)] for mode in (0, 1, 2)),
            4,
        )
        self.assertLessEqual(
            max(cell_counts[("three_modes", mode)] for mode in (0, 1, 2))
            - min(cell_counts[("three_modes", mode)] for mode in (0, 1, 2)),
            1,
        )

    def test_sampling_validates_size_and_uses_no_replacement(self):
        replay = FullyLearnedReplayBuffer(8)
        for marker in range(6):
            replay.add(
                _terminal_transition(
                    regime_id="r0" if marker < 3 else "r1",
                    mode=marker % 3,
                    marker=marker,
                )
            )

        with self.assertRaisesRegex(ValueError, "positive"):
            replay.sample_regime_mode_balanced(0, random.Random(SEED))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            replay.sample_regime_mode_balanced(7, random.Random(SEED))
        sample = replay.sample_regime_mode_balanced(6, random.Random(SEED))
        self.assertEqual(len(sample), 6)
        self.assertEqual(len({item.reward for item in sample}), 6)

    def test_legacy_default_and_replay_round_trip_preserve_regimes(self):
        legacy = _terminal_transition(marker=1.0)
        replay = FullyLearnedReplayBuffer(4)
        replay.add(legacy)
        replay.add(
            _terminal_transition(regime_id="dense", mode=2, marker=2.0)
        )
        state = replay.state_dict()
        restored = FullyLearnedReplayBuffer(4)

        restored.load_state_dict(state)

        self.assertEqual(legacy.regime_id, "legacy")
        self.assertEqual(
            restored.regime_mode_counts(),
            {("legacy", 0): 1, ("dense", 2): 1},
        )
        self.assertEqual(
            [item.regime_id for item in restored.memory],
            ["legacy", "dense"],
        )
        replay.memory[1] = _terminal_transition(
            regime_id="mutated", mode=1, marker=99.0
        )
        self.assertEqual(restored.memory[1].regime_id, "dense")

    def test_agent_default_is_legacy_mode_balanced_and_dispatch_is_explicit(self):
        _, _, legacy_agent = _make_stack(5, batch_size=2)
        self.assertEqual(legacy_agent.regime_id, "legacy")
        self.assertEqual(legacy_agent.replay_sampling, "mode_balanced")

        env, encoder, source = _make_runtime(5)
        configured = FullyLearnedHierarchyAgent(
            env,
            encoder,
            config=FullyLearnedConfig(
                block_embedding_dim=8,
                global_embedding_dim=8,
                candidate_embedding_dim=8,
                context_dim=16,
                history_length=4,
            ),
            spatial_network=source.network,
            seed=SEED,
            device="cpu",
            batch_size=2,
            buffer_size=8,
            update_every=1,
            replay_sampling="regime_mode_balanced",
        )
        for marker in range(2):
            configured.replay.add(
                _terminal_transition(
                    regime_id=f"r{marker}", mode=marker, marker=marker
                )
            )

        with patch.object(
            configured.replay,
            "sample_regime_mode_balanced",
            wraps=configured.replay.sample_regime_mode_balanced,
        ) as sampler:
            loss = configured.learn()

        self.assertTrue(math.isfinite(loss))
        sampler.assert_called_once_with(2, configured.rng)


if __name__ == "__main__":
    unittest.main()
