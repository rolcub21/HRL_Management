import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from example.Options.DeliverOption import DeliverOption
from example.Options.PickupRipeOption import PickupRipeOption
from example.Options.pickupOption import PickupOption
from example.Options.storeOption import StoreOption
from example.controller_observation import (
    ONLINE_SIGNED_TIMING_OBSERVATION,
    make_controller_observation_encoder,
)
from example.helper.tools import flat
from example.small_rooms_env import SmallRoomsEnv
from gated_agent import GatedModeOnlyAgent, GatedModeOnlyFoundationAgent
from primitive_option import PrimitiveOption
from train_track_b import run_training_episode


def make_agent(agent_class, *, seed=3, observation_version=None):
    env = SmallRoomsEnv(choose_storage=False, arrival_rate=0.5, proc_mean=50)
    state = env.reset(instance=env.sample_episode_instance(77))
    for action in env.get_action_space():
        env.options.add(PrimitiveOption(action, env))
    for option_class in (PickupOption, PickupRipeOption, DeliverOption, StoreOption):
        env.options.add(option_class(env))
    encoder = (
        make_controller_observation_encoder(env, observation_version)
        if observation_version is not None
        else None
    )
    state_features = encoder(state) if encoder is not None else flat(state)
    agent = agent_class(
        env=env,
        state_size=len(state_features),
        action_size=len(env.get_action_space()),
        batch_size=2,
        buffer_size=20,
        seed=seed,
        training_policy="mode_regularized",
        disable_tensorboard=True,
        device="cpu",
        verbose=False,
        **(
            {
                "state_encoder": encoder,
                "controller_observation_metadata": encoder.metadata(),
            }
            if encoder is not None
            else {}
        ),
    )
    return env, state, agent


class ReplayObjectiveTests(unittest.TestCase):
    def setUp(self):
        random.seed(4)
        np.random.seed(4)
        torch.manual_seed(4)

    def test_foundation_wait_reward_matches_environment_reward(self):
        env, state, agent = make_agent(GatedModeOnlyFoundationAgent)
        wait = next(
            option
            for option in agent.primitive_options
            if option.action == env.ACTION_IDS["WAIT"]
        )
        agent.current_option = wait
        next_state, reward, _, _ = env.step(wait.action)
        agent.process_step(
            state,
            wait.action,
            reward,
            next_state,
            done=False,
            term=True,
        )
        self.assertAlmostEqual(agent.WorkerBuffer.memory[-1].reward, reward)

    def test_legacy_wait_penalty_remains_an_explicit_ablation(self):
        env, state, agent = make_agent(GatedModeOnlyAgent)
        wait = next(
            option
            for option in agent.primitive_options
            if option.action == env.ACTION_IDS["WAIT"]
        )
        agent.current_option = wait
        next_state, reward, _, _ = env.step(wait.action)
        agent.process_step(
            state,
            wait.action,
            reward,
            next_state,
            done=False,
            term=True,
        )
        self.assertAlmostEqual(
            agent.WorkerBuffer.memory[-1].reward,
            reward - 0.5,
        )

    def test_manager_replay_keeps_immutable_option_start_observation(self):
        env, state, agent = make_agent(
            GatedModeOnlyFoundationAgent,
            observation_version=ONLINE_SIGNED_TIMING_OBSERVATION,
        )
        option = agent.manager_options[0]
        start_features = agent.encode_state(state)
        agent.current_option = option
        agent.option_start_state = state
        agent.option_start_features = start_features.copy()
        agent.option_reward_traj = []

        next_state, reward, _, _ = env.step(env.ACTION_IDS["WAIT"])
        next_features = agent.encode_state(next_state)
        agent.process_step(
            state,
            env.ACTION_IDS["WAIT"],
            reward,
            next_state,
            done=False,
            term=True,
            state_features=start_features,
            next_state_features=next_features,
        )
        np.testing.assert_array_equal(
            agent.ManagerBuffer.memory[-1].state,
            start_features,
        )
        self.assertFalse(np.array_equal(start_features, next_features))


class _OneStepOption:
    is_primitive = False

    def policy(self, state):
        return 6

    def termination(self, state):
        return False


class _OneStepEnv:
    def reset(self, instance=None):
        return (0,)

    def step(self, action):
        return (1,), -0.09, False, {}


class _RecordingSelector:
    def __init__(self):
        self.outcome = None

    def on_step(self, reward, info):
        pass

    def on_episode_end(self, *, success, truncated):
        self.outcome = (success, truncated)


class _RecordingAgent:
    def __init__(self):
        self.current_option = None
        self.epsilon = 0.0
        self.terminal_on_truncation = True
        self.close_options_on_episode_end = True
        self.step_count = 0
        self.calls = []

    def encode_state(self, state):
        return np.asarray(state, dtype=np.float32)

    def select_action(self, state, epsilon, state_features=None):
        return _OneStepOption()

    def process_step(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        term,
        **kwargs,
    ):
        self.calls.append((done, term))

    def learn(self):
        return None, None


class TruncationContractTests(unittest.TestCase):
    def test_horizon_is_terminal_and_flushes_open_option(self):
        agent = _RecordingAgent()
        selector = _RecordingSelector()
        args = SimpleNamespace(
            epsilon_warmup_decisions=0,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_decisions=1,
        )
        result = run_training_episode(
            agent,
            selector,
            _OneStepEnv(),
            instance=None,
            max_steps=1,
            exploration_state={"decisions": 0},
            args=args,
        )
        self.assertEqual(agent.calls, [(True, True)])
        self.assertIsNone(agent.current_option)
        self.assertEqual(result["success"], 0.0)
        self.assertEqual(result["truncated"], 1.0)
        self.assertEqual(selector.outcome, (False, True))


if __name__ == "__main__":
    unittest.main()
