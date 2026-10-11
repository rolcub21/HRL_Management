import os
import time
import numpy as np
import matplotlib.pyplot as plt
from example.small_rooms_env import SmallRoomsEnv

# policy.py
from sub_algorithms import StoreInboundBlock, RetrieveOutboundBlock

class HeuristicPolicy:
    def __init__(self):
        self.store_algo = StoreInboundBlock()
        self.retrieve_algo = RetrieveOutboundBlock()

    def decide_sub_algorithm(self, state, env):
        # inbound check
        for blk in env.blocks:
            if blk.position == (1,5) and not blk.delivered and not blk.stored:
                return self.store_algo

        # outbound check
        for blk in env.blocks:
            if blk.stored and not blk.delivered and blk.is_storage_time_elapsed():
                return self.retrieve_algo

        return None

def run_heuristic_episode(env, policy,
                          block_storing_steps,
                          block_delivery_steps,
                          render_env=False):
    """
    Run one episode using the HeuristicPolicy.
    Records the reward at every step in a list.
    Returns:
      ( total_reward, num_steps, reward_per_step_list )
    """
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    # We'll store the *cumulative reward* at each step (you can store
    # instantaneous rewards if you prefer)
    reward_per_step = []

    while not done:
        sub_algo = policy.decide_sub_algorithm(state, env)
        if sub_algo is None:
            action = env.ACTION_IDS["WAIT"]
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            reward_per_step.append(total_reward)  # store cumulative reward
            steps += 1
            state = next_state
        else:
            # We expect (action_list, chosen_loc, block_label) from StoreInboundBlock
            # or (action_list, block_label) from RetrieveOutboundBlock
            result = sub_algo.produce_actions(state, env)

            # Distinguish the sub‐algorithm types
            if isinstance(sub_algo, StoreInboundBlock):
                # result is (action_list, chosen_loc, block_label)
                action_list, chosen_loc, block_label = result
            elif isinstance(sub_algo, RetrieveOutboundBlock):
                # result is (action_list, block_label)
                action_list, block_label = result
                chosen_loc = None
            else:
                # fallback
                action_list, chosen_loc, block_label = [], None, None

            sub_algo_step_count = len(action_list)

            for act in action_list:
                next_state, reward, done, info = env.step(act)

                # Optionally render
                
                #os.system('clear')
                #env.render()
                #time.sleep(0.5)

                total_reward += reward
                print(f"Action: {act}")
                print(f"Reward: {reward}")
                print(f"Total Reward: {total_reward}")
                reward_per_step.append(total_reward)  # store cumulative reward
                steps += 1
                state = next_state

                if done:
                    break

            # If we know which block was processed, store the sub‐algorithm step count
            if block_label:
                if isinstance(sub_algo, StoreInboundBlock):
                    # This sub‐algorithm was for storing
                    block_storing_steps[block_label] = sub_algo_step_count
                elif isinstance(sub_algo, RetrieveOutboundBlock):
                    # This sub‐algorithm was for delivering
                    block_delivery_steps[block_label] = sub_algo_step_count

    return total_reward, steps, reward_per_step


def generate_storage_heatmap(env, policy, num_trials=100):
    """
    Runs multiple trials to record the storage location chosen by the
    inbound storage sub-algorithm. The counts are aggregated in a heatmap.
    """
    heatmap = np.zeros((7, 7))
    
    for _ in range(num_trials):
        env.reset()  # Reset the environment for a fresh trial
        
        # Decide sub-alg.
        sub_algo = policy.decide_sub_algorithm(None, env)
        if sub_algo is None or not isinstance(sub_algo, StoreInboundBlock):
            continue

        # sub_algo.produce_actions returns (action_list, chosen_loc, block_label) for StoreInboundBlock
        result = sub_algo.produce_actions(None, env)
        if len(result) == 3:
            action_list, chosen_loc, block_label = result
        else:
            continue
        
        # If a location was chosen, increment the heatmap
        if chosen_loc is not None:
            r, c = chosen_loc
            heatmap[r, c] += 1

    return heatmap


if __name__ == "__main__":
    env = SmallRoomsEnv(number_blocks=7)
    policy = HeuristicPolicy()
    
    # Dictionaries for storing/delivery steps per block
    block_storing_steps = {}
    block_delivery_steps = {}

    # Run 1 or more episodes
    num_runs =1
    all_reward_curves = []  # We'll store each episode's reward sequence here

    for i in range(num_runs):
        ret, steps, reward_curve = run_heuristic_episode(env,
                                                         policy,
                                                         block_storing_steps,
                                                         block_delivery_steps,
                                                         render_env=False)
        all_reward_curves.append(reward_curve)
        print(f"Episode {i} => total reward: {ret}, steps used: {steps}")

    # Plot storing vs. delivery steps for each block
    all_blocks = sorted(set(block_storing_steps.keys()) | set(block_delivery_steps.keys()))
    x_indices = np.arange(len(all_blocks))

    storing_values = [block_storing_steps.get(b, 0) for b in all_blocks]
    delivery_values = [block_delivery_steps.get(b, 0) for b in all_blocks]

    plt.figure(figsize=(8, 4))
    plt.plot(x_indices, storing_values, marker='o', label='Storing Steps')
    plt.plot(x_indices, delivery_values, marker='s', label='Delivery Steps')
    plt.xticks(x_indices, all_blocks)
    plt.xlabel("Block Label")
    plt.ylabel("Number of Steps")
    plt.title("Steps for Storing vs. Delivery per Block")
    plt.legend()
    plt.tight_layout()
    plt.savefig("GA/results/block_steps_line_chart.png")
    plt.show()

    # -------------------------------------------------------
    # Now plot the reward curves over steps for each episode
    # -------------------------------------------------------
    plt.figure(figsize=(8, 4))
    for i, rc in enumerate(all_reward_curves):
        plt.plot(range(len(rc)), rc, label=f"Episode {i}")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.title("Reward over Steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig("GA/results/reward_curves.png")
    plt.show()
