import gc
import numpy as np
import matplotlib.pyplot as plt
import pickle

from options_agent import OptionAgent
from small_rooms_env import SmallRoomsEnv
from primitive_option import PrimitiveOption

def load_model(agent, filename):
    # Load the saved model data
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)

    # Restore the agent's parameters and q_table
    agent.q_table = model_data["q_table"]
    agent.epsilon = 0  # Set epsilon to 0 for greedy testing (no exploration)
    agent.gamma = model_data["gamma"]
    agent.macro_q_alpha = model_data["macro_q_alpha"]
    agent.intra_option_alpha = model_data["intra_option_alpha"]

def test_loaded_agent(agent, num_test_episodes=10, max_steps_per_episode=100, render_interval=0):
    """
    Test the loaded agent in the environment for a number of episodes with rendering.

    Args:
        agent: The agent to be tested.
        num_test_episodes: Number of episodes to test.
        max_steps_per_episode: Maximum number of steps per episode.
        render_interval: How often to render (e.g., render every N steps).
    """
    total_rewards = []

    for episode in range(num_test_episodes):
        state = agent.env.reset()  # Reset environment at the start of each episode
        terminal = False
        episode_reward = 0
        steps = 0

        while not terminal and steps < max_steps_per_episode:
            # Render the environment at the specified interval
            if steps % render_interval == 0:
                agent.env.render()

            # Select action using the loaded model (greedy policy)
            selected_option = agent.select_action(state, agent.executing_options, test=True)
            
            # Execute the action and step in the environment
            if isinstance(selected_option, PrimitiveOption):
                agent.executing_options.append(selected_option)
            else:
                next_state, reward, terminal, _ = agent.env.step(selected_option)
                episode_reward += reward
                state = next_state
            
            steps += 1

        # Render the final state of the episode
        agent.env.render()

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
    
    print(f"Average Reward over {num_test_episodes} episodes: {np.mean(total_rewards)}")
    return total_rewards

if __name__ == "__main__":
    # Initialize environment
    env = SmallRoomsEnv()
    
    # Add primitive options
    primitive_options = []
    for action in env.get_action_space():
        primitive_options.append(PrimitiveOption(action, env))
    env.options.update(primitive_options)
    
    # Initialize agent
    agent = OptionAgent(env)

    # Load the saved model
    load_model(agent, "trained_model.pkl")

    # Test the loaded agent with rendering
    num_test_episodes = 10
    max_steps_per_episode = 100
    render_interval = 1  # Render after every step
    test_rewards = test_loaded_agent(agent, num_test_episodes, max_steps_per_episode, render_interval)

    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(test_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Test Rewards over Episodes')
    plt.grid(True)

    # Save the plot to a file
    output_path = '/app/HRL_Management/example/test_rewards_over_episodes.png'
    plt.savefig(output_path)
    print(f"Plot saved successfully as {output_path}")
