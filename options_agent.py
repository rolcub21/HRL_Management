import os
import gc
import math
import random
import pickle
import statistics
import numpy as np
from numpy.random import Generator as RNG
import threading
import os
import time
import curses
from datetime import datetime
from collections import deque, namedtuple, Counter, defaultdict
import copy
from example.Options import selector
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from option import BaseOption
from environment import BaseEnvironment
from example.small_rooms_env import SmallRoomsEnv
from example.Options.waitOption import WaitOption    # penalty for waiting
from example.Options.selector import StorageSelectOption
from example.Options.storeOption import StoreOption  # batch import completeness
from example.helper.tools import flat
from datetime import datetime
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:
        """Minimal no-op replacement used when TensorBoard is unavailable."""

        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def flush(self):
            pass

        def close(self):
            pass



####################################
# 1) NETWORK & BUFFER DEFINITIONS
####################################

class QNetwork(nn.Module):
    """
    A Dueling Q-Network that correctly handles batch normalization.
    """
    def __init__(self, state_size, action_size, seed=0, q_scale=1.0):
        super().__init__()
        torch.manual_seed(seed)
        # Shared network body
        self.fc1 = nn.Linear(state_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        # Dueling heads
        self.value_stream     = nn.Linear(32, 1)
        self.advantage_stream = nn.Linear(32, action_size)
        self.q_scale = q_scale

    def forward(self, x):
        is_single = x.dim() == 1
        if is_single:
            x = x.unsqueeze(0)
        x = self.fc1(x)
        if not is_single: x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if not is_single: x = self.bn2(x)
        x = F.relu(x)
        x = F.relu(self.fc3(x))
        V = self.value_stream(x)
        A = self.advantage_stream(x)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        Q = Q * self.q_scale
        return Q.squeeze(0) if is_single else Q

class ManagerReplayBuffer:
    """PER buffer for the manager (n-step transitions)."""
    def __init__(self, buffer_size, batch_size,
                 alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001, device=None):
        self.memory     = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
            field_names=['state','option_idx','reward','next_state','done','k'])
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[device]", self.device)
        self.alpha = alpha
        self.beta  = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = 1e-6
        self.max_priority = 1.0

    def add(self, state, option_idx, reward, next_state, done, k):
        e = self.experience(state,option_idx,reward,next_state,done,k)
        self.memory.append(e)
        self.priorities.append(float(self.max_priority))

    def sample(self):
        if len(self.memory) < self.batch_size:
            return None
        prios = np.array(self.priorities, dtype=float)
        prob_dist = prios ** self.alpha
        prob_dist /= prob_dist.sum()
        indices = np.random.choice(len(self.memory), self.batch_size,
                                   p=prob_dist, replace=True)
        exps = [self.memory[i] for i in indices]
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        weights = (len(self.memory)*prob_dist[indices]) ** -self.beta
        weights /= weights.max()
        weights = torch.from_numpy(weights).float().to(self.device).unsqueeze(1)
        states      = torch.from_numpy(np.vstack([e.state      for e in exps])).float().to(self.device)
        options     = torch.tensor([[e.option_idx] for e in exps],dtype=torch.long,device=self.device)
        rewards     = torch.from_numpy(np.vstack([e.reward     for e in exps])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in exps])).float().to(self.device)
        dones       = torch.from_numpy(np.vstack([e.done       for e in exps])).float().to(self.device)
        ks          = torch.tensor([[e.k]           for e in exps],dtype=torch.float,device=self.device)
        return states,options,rewards,next_states,dones,ks,weights,indices

    def update_priorities(self, indices, td_errors):
        new_prios = td_errors.abs().detach().cpu().numpy().flatten() + self.epsilon
        for idx,pr in zip(indices,new_prios):
            if idx < len(self.priorities):
                self.priorities[idx] = float(pr)
        self.max_priority = max(self.max_priority, float(new_prios.max()))

    def __len__(self):
        return len(self.memory)

class WorkerReplayBuffer:
    """Uniform buffer for the worker (single-step)."""
    def __init__(self, buffer_size, batch_size, device=None):
        self.memory     = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
            field_names=['state','action','reward','next_state','done'])
        #self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        if len(self.memory) < self.batch_size:
            return None
        exps = random.sample(self.memory, k=self.batch_size)

        states      = torch.from_numpy(
            np.vstack([e.state      for e in exps])
        ).float().to(self.device)
        actions     = torch.from_numpy(
            np.vstack([e.action     for e in exps])
        ).long().to(self.device)
        rewards     = torch.from_numpy(
            np.vstack([e.reward     for e in exps])
        ).float().to(self.device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in exps])
        ).float().to(self.device)
        dones       = torch.from_numpy(
            np.vstack([e.done       for e in exps]).astype(np.uint8)
        ).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


####################################
# 2) AGENT
####################################

class DQNAgent:
    def __init__(self, env:BaseEnvironment, state_size, action_size, **kwargs):
        self.env         = env
        self.state_size  = state_size
        #self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device      = torch.device("cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # buffers & nets
        self.batch_size  = kwargs.get('batch_size',256)
        self.gamma       = kwargs.get('gamma',0.99)
        self.tau         = kwargs.get('tau_soft',1e-3)
        self.update_every= kwargs.get('update_every',25)
        self.lr_mgr      = kwargs.get('lr_manager',3e-5)
        self.lr_wrk      = kwargs.get('lr_worker',1e-5)
        self.grad_clip   = kwargs.get('grad_clip',5.0)
        self.reward_clip = kwargs.get('reward_clip',100.0)
        # epsilon
        total_steps = kwargs.get('n_episodes',3000)*kwargs.get('n_steps',500)
        dp          = total_steps*0.80
        self.epsilon     = kwargs.get('epsilon',1.0)
        self.epsilon_min = kwargs.get('epsilon_min',0.05)
        self.epsilon_decay = (self.epsilon_min/self.epsilon)**(1/dp)
        # options
        #self.manager_options   = [o for o in env.options if not o.is_primitive]
        self.manager_options = sorted(
            [o for o in env.options if not o.is_primitive],
            key=lambda o: str(o)
        )
        self.primitive_options = [o for o in env.options if o.is_primitive]
        # nets
        self.Q_manager_local  = QNetwork(state_size,len(self.manager_options)).to(self.device)
        self.Q_manager_target = QNetwork(state_size,len(self.manager_options)).to(self.device)
        self.optimizer_manager= optim.Adam(self.Q_manager_local.parameters(),lr=self.lr_mgr)
        self.Q_worker_local   = QNetwork(state_size,len(self.primitive_options)).to(self.device)
        self.Q_worker_target  = QNetwork(state_size,len(self.primitive_options)).to(self.device)
        self.optimizer_worker = optim.Adam(self.Q_worker_local.parameters(),lr=self.lr_wrk)
        # sync
        self.Q_manager_target.load_state_dict(self.Q_manager_local.state_dict())
        self.Q_worker_target.load_state_dict(self.Q_worker_local.state_dict())
        self.scheduler_manager= optim.lr_scheduler.StepLR(self.optimizer_manager,step_size=2000,gamma=0.9)
        self.scheduler_worker = optim.lr_scheduler.StepLR(self.optimizer_worker, step_size=2000,gamma=0.9)
        # buffers
        buf_size = kwargs.get('buffer_size',1000000)
        self.ManagerBuffer = ManagerReplayBuffer(buf_size//10, self.batch_size, device=self.device)
        self.WorkerBuffer  = WorkerReplayBuffer(buf_size,     self.batch_size, device=self.device)

        print("agent device:", self.device)
        print("Q_manager_local:", next(self.Q_manager_local.parameters()).device)
        print("Q_worker_local:", next(self.Q_worker_local.parameters()).device)
        print("ManagerBuffer device:", self.ManagerBuffer.device)
        print("WorkerBuffer device:", self.WorkerBuffer.device)

        # bookkeeping
        self.step_count = 0
        self.option_reward_traj   = []
        self.q_log_manager=[]
        self.q_log_worker =[]
        self.action_choice_log = []

        log_root = kwargs.get("tb_logdir", "runs/hrl")
        run_name = kwargs.get("tb_run_name", "default")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.tb_run_name = run_name
        self.writer = SummaryWriter(
            log_dir=os.path.join(log_root, f"{run_name}_{stamp}")
        )



    def select_action(self, state, eps):
        s = torch.from_numpy(flat(state)).float().to(self.device)
        self.Q_manager_local.eval(); self.Q_worker_local.eval()
        with torch.no_grad():
            mgr_q = self.Q_manager_local(s)
            wrk_q = self.Q_worker_local(s)
        self.Q_manager_local.train(); self.Q_worker_local.train()

        #self.q_log_manager.append(mgr_q.cpu().tolist())
        #self.q_log_worker.append(wrk_q.cpu().tolist())

        # ε–greedy decision
        if random.random() < eps:
            opts = [o for o in self.manager_options if o.initiation(state)]
            if opts and random.random() < 0.5:
                choice = random.choice(opts)
            else:
                choice = random.choice(self.primitive_options)
        else:
            best_w, idx_w = wrk_q.max(0)
            best_m, opt_m = -1e9, None
            for i, o in enumerate(self.manager_options):
                if o.initiation(state) and mgr_q[i] > best_m:
                    best_m, opt_m = mgr_q[i], o
            choice = opt_m if (opt_m and best_m > best_w) else self.primitive_options[idx_w]

        # --- ← Insert your logging here ---
        #self.action_choice_log.append({
            #"global_step": self.step_count,
            #"type":       "option" if not choice.is_primitive else "primitive",
            #"name":       type(choice).__name__
        #})

        return choice


    def process_step(self, state, action, reward, next_state, done, term):
        # clip per-step
        reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        # WaitOption penalty
        if isinstance(self.current_option, WaitOption):
            reward -= 0.5
        s,ns = flat(state), flat(next_state)
        if self.current_option and self.current_option.is_primitive:
            self.WorkerBuffer.add(s, action, reward, ns, done)
        if self.current_option and not self.current_option.is_primitive:
            self.option_reward_traj.append(reward)
            if term:
                idx = self.manager_options.index(self.current_option)
                k   = len(self.option_reward_traj)
                Rcum= np.clip(self._discounted_return(self.option_reward_traj,self.gamma),
                              -self.reward_clip, self.reward_clip)
                s0  = flat(self.option_start_state)
                self.ManagerBuffer.add(s0, idx, Rcum, ns, done, k)
                self.current_option=None

    def learn(self):
        mgr_loss=None; wrk_loss=None
        if self.step_count % self.update_every==0:
            if len(self.WorkerBuffer)>self.batch_size: wrk_loss=self._learn_worker()
            if len(self.ManagerBuffer)>self.batch_size: mgr_loss=self._learn_manager()
        return mgr_loss,wrk_loss

    def _learn_worker(self):
        sample=self.WorkerBuffer.sample()
        if sample is None: return None
        states,acts,r,ns,d = sample
        Qn = self.Q_worker_target(ns).detach().max(1)[0].unsqueeze(1)
        Qt = r + (self.gamma*Qn*(1-d))
        Qe = self.Q_worker_local(states).gather(1,acts)
        loss=F.mse_loss(Qe,Qt)
        self.optimizer_worker.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.Q_worker_local.parameters(),self.grad_clip)
        self.optimizer_worker.step()
        self._soft_update(self.Q_worker_local,self.Q_worker_target,self.tau)
        return loss.item()


    def _learn_manager(self):
        sample=self.ManagerBuffer.sample()
        if sample is None: return None
        s,opts,rewards,ns,done,ks,weights,idxs = sample
        Qn = self.Q_manager_target(ns).detach().max(1)[0].unsqueeze(1)
        Qt = rewards + ((self.gamma**ks)*Qn*(1-done))
        Qe = self.Q_manager_local(s).gather(1,opts)
        td = Qt - Qe
        loss = (weights * F.mse_loss(Qe,Qt,reduction='none')).mean()
        self.optimizer_manager.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.Q_manager_local.parameters(),self.grad_clip)
        self.optimizer_manager.step()
        self.ManagerBuffer.update_priorities(idxs,td)
        self._soft_update(self.Q_manager_local,self.Q_manager_target,self.tau)
        return loss.item()

    def run(self, n_episodes, max_steps, npz_dir="./results/datalogs", model_dir="example/models"):
        os.makedirs(npz_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        logs = {
            "episode_returns":   [],
            "manager_losses":    [],
            "worker_losses":     [],
            "episode_avg_error": [],
            "episode_success":   [],
            #"action_choices":    [],
            #"manager_q_log":     [],
            #"worker_q_log":      [],
        }

        checkpoint_every = 500

        def get_storage_selector():
            return next(
                (
                    o for o in self.env.options
                    if type(o).__name__ in ("StorageSelectOption", "GAStorageSelectOption")
                ),
                None
            )

        for ep in range(1, n_episodes + 1):
            # --- reset env ---
            state = self.env.reset()

            # IMPORTANT: if env.reset() can rebuild options, refresh these each episode
            self.manager_options   = [o for o in self.env.options if not o.is_primitive]
            self.primitive_options = [o for o in self.env.options if o.is_primitive]

            # also refresh selector each episode (so it doesn't go stale)
            selector = get_storage_selector()
            if ep == 1:
                print("[debug] options after reset:", [type(o).__name__ for o in self.env.options])
                print("[debug] selector after reset:", selector is not None)


            ep_ret, step = 0.0, 0
            done = False
            self.current_option = None
            this_ep_errors = []
            success = False

            while not done and step < max_steps:
                # 1) pick or continue option
                if self.current_option is None:
                    a_or_o = self.select_action(state, self.epsilon)
                    self.current_option = a_or_o

                    if not a_or_o.is_primitive:
                        self.option_start_state = state
                        self.option_reward_traj = []

                prim = self.current_option.policy(state)
                nxt, rew, done, info = self.env.step(prim)

                # 2) accumulate step reward into selector pendings
                if selector is not None and hasattr(selector, "pending") and hasattr(selector, "gamma"):
                    for p in selector.pending.values():
                        p["acc"]  += p["disc"] * rew
                        p["disc"] *= selector.gamma

                # 3) notify selector on delivery events
                if (
                    selector is not None
                    and hasattr(selector, "on_delivery")
                    and info.get("delivered_block", False)
                    and "delivery_error_time" in info
                ):
                    delivered = info["delivered_block"]
                    err_t = info["delivery_error_time"]
                    selector.on_delivery(delivered, err_t)

                # 4) logging / success condition
                if "delivery_error_time" in info:
                    this_ep_errors.append(info["delivery_error_time"])


                # 5) termination + learning
                term = self.current_option.termination(nxt)
                self.step_count += 1
                self.process_step(state, prim, rew, nxt, done, term)
                if term:
                    self.current_option = None

                state = nxt
                ep_ret += rew
                step += 1

                mgr_l, wrk_l = self.learn()
                if mgr_l is not None:
                    logs["manager_losses"].append(mgr_l)
                if wrk_l is not None:
                    logs["worker_losses"].append(wrk_l)

                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # --- end of episode hooks ---
            if selector is not None and hasattr(selector, "on_episode_end"):
                selector.on_episode_end()

            episode_all_delivered = 1.0 if done else 0.0
            logs["episode_success"].append(episode_all_delivered)
            logs["episode_returns"].append(ep_ret)

            ep_avg_err = float(np.mean(this_ep_errors)) if this_ep_errors else np.nan
            logs["episode_avg_error"].append(ep_avg_err)
            
            #logs["manager_q_log"].extend(self.q_log_manager)
            #logs["worker_q_log"].extend(self.q_log_worker)

            # ---------------- TensorBoard (per-episode) ----------------
            self.writer.add_scalar(f"{self.tb_run_name}/success_all_delivered", episode_all_delivered, ep)

            # optional rolling mean (last 100)
            succ_ma100 = float(np.mean(deque(logs["episode_success"], maxlen=100)))
            self.writer.add_scalar(f"{self.tb_run_name}/success_all_delivered_ma100", succ_ma100, ep)

            self.writer.add_scalar(f"{self.tb_run_name}/return", ep_ret, ep)
            self.writer.add_scalar(f"{self.tb_run_name}/epsilon", self.epsilon, ep)

            avg_err = float(np.mean(this_ep_errors)) if this_ep_errors else np.nan
            if np.isfinite(avg_err):
                self.writer.add_scalar(f"{self.tb_run_name}/avg_delivery_error", avg_err, ep)

            # rolling mean return (last 100 episodes)
            mean_ret_100 = float(np.mean(deque(logs["episode_returns"], maxlen=100)))
            self.writer.add_scalar(f"{self.tb_run_name}/return_ma100", mean_ret_100, ep)

            # last seen losses (optional)
            if len(logs["manager_losses"]) > 0:
                self.writer.add_scalar(f"{self.tb_run_name}/manager_loss", logs["manager_losses"][-1], ep)
            if len(logs["worker_losses"]) > 0:
                self.writer.add_scalar(f"{self.tb_run_name}/worker_loss", logs["worker_losses"][-1], ep)

            # optional: make it appear quickly in TensorBoard
            if ep % 10 == 0:
                self.writer.flush()
            # ----------------------------------------------------------

            
            if ep % 50 == 0:
                np.savez_compressed(
                    os.path.join(npz_dir, f"{self.tb_run_name}_partial_ep{ep}.npz"),
                    episode_returns=np.asarray(logs["episode_returns"], dtype=np.float32),
                    episode_avg_error=np.asarray(logs["episode_avg_error"], dtype=np.float32),
                    episode_success=np.asarray(logs["episode_success"], dtype=np.float32),
                    manager_losses=np.asarray(logs["manager_losses"], dtype=np.float32),
                    worker_losses=np.asarray(logs["worker_losses"], dtype=np.float32),
                )




            # checkpoints
            if checkpoint_every and ep % checkpoint_every == 0:
                path = self.save_agent(model_dir)
                print(f"Checkpoint @ ep {ep}: {path}")

            if ep % 1 == 0 or ep == 1:
                mean_ret = np.mean(deque(logs["episode_returns"],   maxlen=100))
                recent_errs = [
                    x for x in deque(logs["episode_avg_error"], maxlen=100)
                    if x is not None and np.isfinite(x)
                ]
                mean_err = float(np.mean(recent_errs)) if recent_errs else float("nan")
                err_str = f"{mean_err:7.2f}" if np.isfinite(mean_err) else "   N/A "
                print(f"Ep {ep:4d} | AvgR {mean_ret:7.2f} | AvgErr {err_str} |  Eps {self.epsilon:.3f}")

        final_path = self.save_agent(model_dir)
        print("Final saved:", final_path)

        self.writer.flush()
        self.writer.close()

        #logs["action_choices"] = self.action_choice_log
        return logs



    def _soft_update(self, local, target, tau):
        for tp, lp in zip(target.parameters(), local.parameters()):
            tp.data.copy_(tau*lp.data + (1.0-tau)*tp.data)

    def _discounted_return(self, rewards, gamma):
        n    = len(rewards)
        exps = np.arange(n)
        gam  = np.power(gamma, exps)
        return float(np.dot(rewards, gam))
    
    def save_agent(self, out_dir="./results"):
        os.makedirs(out_dir, exist_ok=True)

        ckpt = {
            "manager_state_dict": self.Q_manager_local.state_dict(),
            "worker_state_dict":  self.Q_worker_local.state_dict(),
            "manager_opt_state":  self.optimizer_manager.state_dict(),
            "worker_opt_state":   self.optimizer_worker.state_dict(),
            "epsilon":    self.epsilon,
            "step_count": self.step_count,
        }

        # Debug: list option names
        try:
            opt_names = [type(o).__name__ for o in self.env.options]
        except TypeError:
            opt_names = [type(o).__name__ for o in list(self.env.options)]
        print("[save_agent] options in env:", opt_names)

        # Robust selector lookup (avoids isinstance import-path issues)
        selector = next((o for o in self.env.options if type(o).__name__ == "StorageSelectOption"), None)

        if selector is not None:
            ckpt["selector_state_dict"] = selector.q.state_dict()
            ckpt["selector_opt_state"]  = selector.opt.state_dict()
            ckpt["selector_call_count"] = selector.call_cnt
            ckpt["selector_eps"]        = selector.eps
            print("✓ added StorageSelectOption weights to checkpoint")
        else:
            print("[save_agent] StorageSelectOption NOT found -> selector not saved")

        fname = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f") + "_HRL.pth"
        fpath = os.path.join(out_dir, fname)
        torch.save(ckpt, fpath)
        print(f"✓ checkpoint saved to {fpath}")
        return fpath
