import gc
import numpy as np
import matplotlib.pyplot as plt
import threading

from option import BaseOption
from options_agent import OptionAgent
from primitive_option import PrimitiveOption
from small_rooms_env import SmallRoomsEnv
# from small_rooms_doorway_option import DoorwayOption
# from pickupOption import PickupOption


if __name__ == "__main__":
    num_agents = 5
    n_episodes = 300000

    primitive_results = []
    option_results = []

    results = []
    for run in range(num_agents):
        # Initialize environment
        env = SmallRoomsEnv()
        # Add primitive options
        primitive_options = []
        for action in env.get_action_space():
            primitive_options.append(PrimitiveOption(action, env))
        env.options.update(primitive_options)
        # env.options.add(PickupOption())
        # Initialize and train agent
        agent = OptionAgent(env)
        result = agent.run_agent(n_episodes, 1)
        window_size = 10000
        value = agent.q_values_log_macro
        #steps_count = agent.step_counter
        smoothed_results = np.convolve(result, np.ones(window_size) / window_size, mode='valid')
        results.append(smoothed_results)
        gc.collect()
        data = results[0]


    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(np.mean(results, axis=0))
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Return over time')
    plt.grid(True)
    
    # Save the plot to a file
    output_path = '/app/HRL_Management/example/return_over_time.png'
    plt.savefig(output_path)
    #print(f"Plot saved successfully as {output_path}")
