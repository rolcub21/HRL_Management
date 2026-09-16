import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
import random
from small_rooms_env import SmallRoomsEnv
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, env, gamma=0.99, epsilon=0.1, lr=1e-3, batch_size=64, memory_size=10000):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_dim = 2  # The state is a tuple (1,1) which we flatten to a size 2 array
        output_dim = len(env.get_action_space())   
        self.q_network = DQN(input_dim, output_dim).to(self.device)
        self.target_network = DQN(input_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice(self.env.get_action_space())
        else:
            state = np.array(state, dtype=np.float32).flatten()  # Convert state tuple to numpy array and flatten
            state = torch.tensor(state).unsqueeze(0).to(self.device)  # Add batch dimension and convert to tensor
            with torch.no_grad():
                q_values = self.q_network(state)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32).flatten()
        next_state = np.array(next_state, dtype=np.float32).flatten()
        self.memory.append((state, action, reward, next_state, done))

    def sample_batch(self):
        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), action, reward, np.array(next_state), done

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.sample_batch()
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        current_q = self.q_network(states).gather(1, actions)
        next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
        expected_q = rewards + (self.gamma * next_q * (1 - dones))

        loss = self.loss_fn(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def run(self, num_episodes, update_interval):
        all_rewards = []
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.select_action(state, self.epsilon)
                next_state, reward, done, _ = self.env.step(action)
                self.store_transition(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                self.train()
            all_rewards.append(total_reward)
            if episode % update_interval == 0:
                self.update_target_network()
            print(f"Episode {episode}, Total Reward: {total_reward}")
        return all_rewards


