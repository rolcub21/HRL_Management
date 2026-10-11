import random, collections, os, math, time, gc
from collections import deque
from typing import Tuple, Deque, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from small_rooms_env import SmallRoomsEnv

##############################################################################
# Hyper‑parameters
##############################################################################
GAMMA           = 0.99
LR              = 1e-3
BUFFER_SIZE     = 50_000
BATCH_SIZE      = 128
TARGET_UPDATE   = 500           # steps between target net updates
EPS_START       = 1.0
EPS_END         = 0.05
EPS_DECAY_STEPS = 25_000        # linear decay
N_EPISODES      = 2_000
MAX_STEP_PER_EP = 300
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################################
# Replay Buffer
##############################################################################
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque = deque(maxlen=capacity)
    def push(self, *transition):
        self.buffer.append(tuple(transition))
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))
    def __len__(self):
        return len(self.buffer)

##############################################################################
# Simple MLP for DQN – input: flattened discrete state, output: |A| Q‑values
##############################################################################
class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, action_dim),
        )
    def forward(self, x):
        return self.net(x)

##############################################################################
# Helper to flatten the discrete state tuple into a 1‑D numpy array
##############################################################################

def flatten_discrete_state(state: Tuple) -> np.ndarray:
    agent_idx, blocks = state
    flat = [agent_idx]
    for b in blocks:
        flat.extend(b)  # each b is already tuple of ints
    return np.array(flat, dtype=np.float32) / 10.0  # scale for stability

##############################################################################
# Training routine
##############################################################################

def train_dqn(env: SmallRoomsEnv):
    action_dim = len(env.get_action_space())
    # preview one state to know input dim
    state_dim = len(flatten_discrete_state(env.reset()))

    policy_net  = DQN(state_dim, action_dim).to(DEVICE)
    target_net  = DQN(state_dim, action_dim).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimiser = optim.Adam(policy_net.parameters(), lr=LR)
    buffer    = ReplayBuffer(BUFFER_SIZE)

    global_step = 0
    episode_returns = []

    for ep in range(N_EPISODES):
        state = env.reset()
        state_arr = flatten_discrete_state(state)
        ep_return = 0.0
        for t in range(MAX_STEP_PER_EP):
            # ε‑greedy schedule
            eps = EPS_END + max(0, (EPS_START-EPS_END)*(1 - global_step/EPS_DECAY_STEPS))
            if random.random() < eps:
                action = random.choice(env.get_action_space())
            else:
                with torch.no_grad():
                    qvals = policy_net(torch.tensor(state_arr, device=DEVICE).unsqueeze(0))
                    action = int(qvals.argmax().item())
            next_state, reward, done, _ = env.step(action)
            next_state_arr = flatten_discrete_state(next_state)
            buffer.push(state_arr, action, reward, next_state_arr, done)
            state_arr = next_state_arr
            ep_return += reward
            global_step += 1

            # Learn
            if len(buffer) >= BATCH_SIZE:
                s, a, r, s2, d = buffer.sample(BATCH_SIZE)
                s   = torch.tensor(s,  dtype=torch.float32, device=DEVICE)
                a   = torch.tensor(a,  dtype=torch.int64,   device=DEVICE).unsqueeze(1)
                r   = torch.tensor(r,  dtype=torch.float32, device=DEVICE).unsqueeze(1)
                s2  = torch.tensor(s2, dtype=torch.float32, device=DEVICE)
                d   = torch.tensor(d,  dtype=torch.float32, device=DEVICE).unsqueeze(1)

                q   = policy_net(s).gather(1, a)
                with torch.no_grad():
                    q_next = target_net(s2).max(1, keepdim=True)[0]
                    target = r + GAMMA * (1 - d) * q_next
                loss = nn.functional.mse_loss(q, target)

                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 5)
                optimiser.step()

            # Target network update
            if global_step % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break
        episode_returns.append(ep_return)
        if (ep+1) % 20 == 0:
            print(f"Ep {ep+1}/{N_EPISODES}  return={ep_return:.1f}  eps={eps:.2f}")
        gc.collect()

    return policy_net, episode_returns

##############################################################################
if __name__ == "__main__":
    env = SmallRoomsEnv()
    _, returns = train_dqn(env)
    # Simple plot
    import matplotlib.pyplot as plt
    plt.plot(returns)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('DQN on SmallRoomsEnv')
    plt.show()
