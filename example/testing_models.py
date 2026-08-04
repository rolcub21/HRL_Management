import gc
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

from options_agent import OptionAgent
from small_rooms_env import SmallRoomsEnv
from example.Options.storeOption import StoreOption
from primitive_option import PrimitiveOption
# from ClearPathOption import ClearPathOption


def main():
    # 1) Build environment for testing
    test_env = SmallRoomsEnv(number_blocks=5)
    primitive_options = [PrimitiveOption(action, test_env) for action in test_env.get_action_space()]
    test_env.options.update(primitive_options)

    # 2) Add options needed
    #test_env.options.add(ManagerOption(test_env))
    test_env.options.add(StoreOption(test_env))
    #test_env.options.add(DeliveryOption(test_env))
    # test_env.options.add(DropOption(test_env))
    # test_env.options.add(ClearPathOption(test_env))

    # 3) Create the agent, passing test_env as both env and test_env
    agent = OptionAgent(env=test_env, test_env=test_env, epsilon=0.0)

    # 4) Load the trained model
    model_path = "/app/HRL_Management_2L/example/models/trained_model_epoch_108.pkl"
    with open(model_path, "rb") as f:
        agent_data = pickle.load(f)
    agent.q_table = agent_data["q_table"]

    # 5) Run test
    #    [CHANGED] We assume your updated test_policy returns 6 items now
    (
        avg_reward,
        delivered,
        storage_info,
        all_reward_curves,
        store_step_counts,
        deliver_step_counts
    ) = agent.test_policy(
        test_length=50000,
        test_runs=1,
        eval_number=1,
        allow_exploration=False,
        verbose_logging=False,
        episodic_eval=True
    )

    filename = "combined_experiments_rewards_test.npz"

    # Load existing data if the file exists
    if os.path.exists(filename):
        # Load the existing file (allow_pickle ensures compatibility with Python objects)
        existing_data = dict(np.load(filename, allow_pickle=True))
    else:
        existing_data = {}

    # Add or update with new experiment data (for example, exp2)
    existing_data["2L_test"] = all_reward_curves

    # Save the combined dictionary back to the same file
    np.savez(filename, **existing_data)

    print("===== TEST RESULTS =====")
    print("Average Reward:", avg_reward)
    print("Blocks Delivered:", delivered)
    print("Block Storage Info:", storage_info)

    # -------------------------------
    # 6) Plot the Reward vs. Steps
    # -------------------------------
    plt.figure(figsize=(8,4))
    for run_index, reward_curve in enumerate(all_reward_curves):
        plt.plot(range(len(reward_curve)), reward_curve, label=f"Run {run_index}")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.title("Reward over Steps (OptionAgent)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reward_over_steps.png")


if __name__ == "__main__":
    main()
