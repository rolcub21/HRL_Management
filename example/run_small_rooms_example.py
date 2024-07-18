
import gc
import numpy as np
import matplotlib.pyplot as plt
import threading

from simpleoptions.option import BaseOption
from simpleoptions import OptionAgent, PrimitiveOption
from small_rooms_env import SmallRoomsEnv
#from small_rooms_doorway_option import DoorwayOption
#from pickupOption import PickupOption


if __name__ == "__main__":
    num_agents = 7
    n_episodes = 100000

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
        #env.options.add(PickupOption())
        # Initialize and train agent
        agent = OptionAgent(env)
        result = agent.run_agent(n_episodes, 1)
        window_size = 30000
        smoothed_results = np.convolve(result, np.ones(window_size)/window_size, mode='valid')
        results.append(smoothed_results)
        gc.collect()

    # Plot results
    plt.figure()
    plt.plot(np.mean(results, axis=0))
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Return over time')
    plt.show()

'''
if __name__ == "__main__":
    env = SmallRoomsEnv()
    env.reset()

    action_input = {
        "0": 0,  # UP
        "1": 1,  # DOWN
        "2": 2,  # LEFT
        "3": 3,  # RIGHT
        "4": 4,  # PICKUP
        "5": 5   # PUTDOWN
    }

    while True:
        action = input("Enter action: ")
        action = action_input[action]
        state, reward, terminal, _ = env.step(action)
        print(f"Action: {env.ACTION_NAMES[action]}, Reward: {reward}")
        env.render()
        if terminal:
            print("Episode finished!")
            break

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from small_rooms_env import SmallRoomsEnv
import matplotlib.pyplot as plt

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class QLearningAgent:
    def __init__(self, env, learning_rate=0.001, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01, batch_size=64):
        self.env = env
        self.state_dim = 3  # 2 for coordinates + 1 for carrying flag
        self.action_dim = len(self.env.ACTION_NAMES)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.batch_size = batch_size

        self.q_network = QNetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.memory = deque(maxlen=10000)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(list(self.env.ACTION_NAMES.keys()))
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.q_network(states)
        next_q_values = self.q_network(next_states)

        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = next_q_values.max(1)[0]

        target_q_values = rewards + (self.discount_factor * next_q_values * (1 - dones))

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_episodes):
        rewards = []
        for episode in range(num_episodes):
            state = self.env.reset()
            state = list(state)
            total_reward = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = list(next_state)

                # Intermediate rewards for each step
                reward -= 0.01

                self.memory.append((state, action, reward, next_state, done))
                self.learn()
                state = next_state
                total_reward += reward

            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)
            rewards.append(total_reward)
            print(f"Episode {episode+1}: Total Reward: {total_reward}")

        return rewards

if __name__ == "__main__":
    env = SmallRoomsEnv()
    agent = QLearningAgent(env)
    rewards = agent.train(1000)

    # Plotting the rewards
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward vs. Episode')
    plt.show()

    # Test the learned policy
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.choose_action(state)
        state, reward, done, _ = env.step(action)
        env.render()
        total_reward += reward

    print(f"Total Reward: {total_reward}")
'''