import os
import gc
import math
import random
import warnings
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
from example.helper.timing_metrics import summarize_delivery_timing
from datetime import datetime


class NullSummaryWriter:
    """No-op writer for evaluation and TensorBoard-free installations."""

    def add_scalar(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = NullSummaryWriter


CHECKPOINT_SCHEMA_VERSION = 3

Q_NETWORK_ARCHITECTURE = "dueling_mlp_128_64_32_v2"
LEGACY_BATCH_NORM = "legacy_batch_norm_skip_single"
LAYER_NORM = "layer_norm"
NO_NORMALIZATION = "none"
SUPPORTED_Q_NORMALIZATIONS = (
    LEGACY_BATCH_NORM,
    LAYER_NORM,
    NO_NORMALIZATION,
)
Q_NETWORK_INITIALIZATION = "isolated_explicit_seed_v1"
DUELING_ALL_OUTPUTS = "all_outputs_legacy"
DUELING_UNCENTERED = "uncentered_advantage_v1"
SUPPORTED_DUELING_CENTERING = (
    DUELING_ALL_OUTPUTS,
    DUELING_UNCENTERED,
)


def resolve_q_network_config(payload):
    """Resolve a checkpoint's critic topology without reinterpreting weights.

    Controller checkpoints created before Q-network metadata was introduced
    used BatchNorm during replay training and skipped it for rank-1 inference.
    That behavior is defective, but it must remain reproducible for internal
    ablations.  Missing metadata therefore resolves explicitly to the legacy
    topology; partially specified or unknown current metadata is rejected.
    """

    metadata_keys = {
        "q_network_architecture",
        "q_network_normalization",
        "q_network_initialization",
        "controller_initialization_seed",
        "manager_initialization_seed",
        "worker_initialization_seed",
    }
    present = metadata_keys.intersection(payload)
    if not present:
        if "q_network_dueling_centering" in payload:
            raise ValueError(
                "Incomplete Q-network checkpoint metadata: "
                "q_network_dueling_centering has no topology metadata"
            )
        return {
            "architecture": Q_NETWORK_ARCHITECTURE,
            "normalization": LEGACY_BATCH_NORM,
            "initialization": "legacy_global_seed_reset",
            "controller_seed": 0,
            "manager_seed": 0,
            "worker_seed": 0,
            "dueling_centering": DUELING_ALL_OUTPUTS,
            "legacy": True,
        }
    missing = sorted(metadata_keys.difference(payload))
    if missing:
        raise ValueError(
            "Incomplete Q-network checkpoint metadata: " + ", ".join(missing)
        )
    architecture = payload["q_network_architecture"]
    normalization = payload["q_network_normalization"]
    initialization = payload["q_network_initialization"]
    if architecture != Q_NETWORK_ARCHITECTURE:
        raise ValueError(f"Unsupported Q-network architecture: {architecture!r}")
    if normalization not in SUPPORTED_Q_NORMALIZATIONS:
        raise ValueError(f"Unsupported Q-network normalization: {normalization!r}")
    if initialization != Q_NETWORK_INITIALIZATION:
        raise ValueError(f"Unsupported Q-network initialization: {initialization!r}")
    controller_seed = int(payload["controller_initialization_seed"])
    manager_seed = int(payload["manager_initialization_seed"])
    worker_seed = int(payload["worker_initialization_seed"])
    dueling_centering = payload.get(
        "q_network_dueling_centering", DUELING_ALL_OUTPUTS
    )
    if dueling_centering not in SUPPORTED_DUELING_CENTERING:
        raise ValueError(
            "Unsupported Q-network dueling centering: "
            f"{dueling_centering!r}"
        )
    if manager_seed != controller_seed or worker_seed != controller_seed + 1:
        raise ValueError("Q-network initialization seeds are inconsistent")
    return {
        "architecture": architecture,
        "normalization": normalization,
        "initialization": initialization,
        "controller_seed": controller_seed,
        "manager_seed": manager_seed,
        "worker_seed": worker_seed,
        "dueling_centering": dueling_centering,
        "legacy": normalization == LEGACY_BATCH_NORM,
    }


def option_identifier(option):
    """Return the stable controller identity used for output-head indices."""
    explicit = getattr(option, "controller_identifier", None)
    if explicit is not None:
        return str(explicit)
    if getattr(option, "is_primitive", False):
        action = getattr(option, "action", None)
        return f"primitive:{type(action).__name__}:{action!r}"
    return f"option:{type(option).__name__}"


def canonical_manager_options(options):
    return sorted(
        (option for option in options if not option.is_primitive),
        key=option_identifier,
    )


def canonical_primitive_options(options):
    return sorted(
        (option for option in options if option.is_primitive),
        key=option_identifier,
    )


def controller_ids(options):
    return [option_identifier(option) for option in options]


def is_storage_selector(option):
    """Recognise versioned selector adapters without relying on class names."""

    return bool(getattr(option, "is_storage_selector", False)) or type(
        option
    ).__name__ in ("StorageSelectOption", "GAStorageSelectOption")


def find_storage_selector(options):
    return next((option for option in options if is_storage_selector(option)), None)


def validate_checkpoint_metadata(
    payload,
    manager_options,
    primitive_options,
    selector=None,
    *,
    allow_legacy=False,
):
    """Validate that checkpoint output indices match runtime controllers."""
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "manager_option_ids": controller_ids(manager_options),
        "primitive_action_ids": controller_ids(primitive_options),
    }
    if selector is not None:
        expected["selector_feature_version"] = getattr(
            selector, "FEATURE_VERSION", 1
        )
        expected["selector_return_definition"] = getattr(
            selector, "return_definition", None
        )
        expected["selector_gamma"] = float(getattr(selector, "gamma", 0.99))

    missing = [key for key in expected if key not in payload]
    if missing:
        message = (
            "Legacy checkpoint is missing compatibility metadata "
            f"({', '.join(missing)}). Its controller-index mapping and/or "
            "selector feature/return semantics cannot be verified."
        )
        if not allow_legacy:
            raise ValueError(
                message + " Pass allow_legacy=True only for an explicitly "
                "accepted, non-comparable legacy evaluation."
            )
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    for key, runtime_value in expected.items():
        if key not in payload:
            continue
        saved_value = payload[key]
        if isinstance(runtime_value, list):
            saved_value = list(saved_value)
        if saved_value != runtime_value:
            if key == "checkpoint_schema_version" and allow_legacy:
                warnings.warn(
                    "Loading an explicitly accepted older checkpoint schema "
                    f"({saved_value!r}; current={runtime_value!r}). Results "
                    "must remain labeled as legacy/non-comparable.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            raise ValueError(
                f"Checkpoint {key} mismatch: saved={saved_value!r}, "
                f"runtime={runtime_value!r}"
            )


def accumulate_selector_pending(selector, reward, info):
    """Accumulate one step of a delayed selector event return.

    ``full_environment`` (the default) records the environment reward stream
    verbatim. ``explicit_terminal`` removes the matching delivery component
    here so :meth:`StorageSelectOption.on_delivery` can add that same component
    once, at the same discount. Other pending assignments observe the complete
    global reward stream in both definitions.
    """
    delivered_label = info.get("delivered_block", False)
    delivery_reward = float(info.get("delivery_reward", 0.0))
    return_mode = getattr(
        selector,
        "return_mode",
        StorageSelectOption.RETURN_FULL_ENVIRONMENT,
    )

    for label, pending in list(selector.pending.items()):
        step_reward = float(reward)
        if (
            return_mode == StorageSelectOption.RETURN_EXPLICIT_TERMINAL
            and delivered_label
            and label == delivered_label
        ):
            step_reward -= delivery_reward
            pending["terminal_discount"] = pending["disc"]
        pending["acc"] += pending["disc"] * step_reward
        pending["disc"] *= selector.gamma
        pending["steps"] = pending.get("steps", 0) + 1




####################################
# 1) NETWORK & BUFFER DEFINITIONS
####################################

class QNetwork(nn.Module):
    """
    Dueling critic with an explicitly versioned normalization strategy.

    ``legacy_batch_norm_skip_single`` reproduces historical checkpoints,
    including their train/inference mismatch.  New controllers should use
    LayerNorm, whose output is independent of input rank and batch companions.
    """
    def __init__(
        self,
        state_size,
        action_size,
        seed=0,
        q_scale=1.0,
        normalization=LEGACY_BATCH_NORM,
        dueling_centering=DUELING_ALL_OUTPUTS,
    ):
        super().__init__()
        if normalization not in SUPPORTED_Q_NORMALIZATIONS:
            raise ValueError(f"Unknown Q-network normalization: {normalization!r}")
        self.normalization = normalization
        if dueling_centering not in SUPPORTED_DUELING_CENTERING:
            raise ValueError(
                "Unknown dueling centering: " f"{dueling_centering!r}"
            )
        self.dueling_centering = dueling_centering

        # Isolate initialization so constructing a critic neither resets nor
        # consumes the ambient Torch RNG used by exploration and evaluation.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        with torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(generator.get_state())
            self.fc1 = nn.Linear(state_size, 128)
            if normalization == LEGACY_BATCH_NORM:
                self.bn1 = nn.BatchNorm1d(128)
            elif normalization == LAYER_NORM:
                self.bn1 = nn.LayerNorm(128)
            else:
                self.bn1 = nn.Identity()
            self.fc2 = nn.Linear(128, 64)
            if normalization == LEGACY_BATCH_NORM:
                self.bn2 = nn.BatchNorm1d(64)
            elif normalization == LAYER_NORM:
                self.bn2 = nn.LayerNorm(64)
            else:
                self.bn2 = nn.Identity()
            self.fc3 = nn.Linear(64, 32)
            self.value_stream = nn.Linear(32, 1)
            self.advantage_stream = nn.Linear(32, action_size)
        self.q_scale = q_scale

    def forward(self, x):
        is_single = x.dim() == 1
        if is_single:
            x = x.unsqueeze(0)
        x = self.fc1(x)
        if self.normalization != LEGACY_BATCH_NORM or not is_single:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.normalization != LEGACY_BATCH_NORM or not is_single:
            x = self.bn2(x)
        x = F.relu(x)
        x = F.relu(self.fc3(x))
        V = self.value_stream(x)
        A = self.advantage_stream(x)
        if self.dueling_centering == DUELING_ALL_OUTPUTS:
            Q = V + (A - A.mean(dim=1, keepdim=True))
        else:
            # A fixed 40-block action head is usually sparse: most block jobs
            # are unavailable in a given state.  Centering over every output
            # would let those invalid controls shift all valid Q values.  The
            # uncentered form preserves a learned shared value while keeping
            # invalid outputs outside the valid-control Bellman operator.
            Q = V + A
        Q = Q * self.q_scale
        return Q.squeeze(0) if is_single else Q

class ManagerReplayBuffer:
    """PER buffer for the manager (n-step transitions)."""
    def __init__(self, buffer_size, batch_size,
                 alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001,
                 device=None, verbose=True):
        self.memory     = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
            field_names=['state','option_idx','reward','next_state','done','k'])
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if verbose:
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
        self.state_encoder = kwargs.get("state_encoder", flat)
        self.controller_observation_metadata = dict(
            kwargs.get("controller_observation_metadata", {})
        )
        self.verbose = bool(kwargs.get("verbose", True))
        #self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device      = torch.device("cpu")
        requested_device = kwargs.get("device")
        self.device = torch.device(
            requested_device
            if requested_device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        # buffers & nets
        self.batch_size  = kwargs.get('batch_size',256)
        self.gamma       = kwargs.get('gamma',0.99)
        self.tau         = kwargs.get('tau_soft',1e-3)
        self.update_every= kwargs.get('update_every',25)
        self.lr_mgr      = kwargs.get('lr_manager',3e-5)
        self.lr_wrk      = kwargs.get('lr_worker',1e-5)
        self.grad_clip   = kwargs.get('grad_clip',5.0)
        self.reward_clip = kwargs.get('reward_clip',100.0)
        self.initialization_seed = int(kwargs.get("seed", 0))
        self.q_network_normalization = kwargs.get(
            "network_normalization", LEGACY_BATCH_NORM
        )
        if self.q_network_normalization not in SUPPORTED_Q_NORMALIZATIONS:
            raise ValueError(
                "Unknown Q-network normalization: "
                f"{self.q_network_normalization!r}"
            )
        self.q_network_dueling_centering = kwargs.get(
            "dueling_centering", DUELING_ALL_OUTPUTS
        )
        if self.q_network_dueling_centering not in SUPPORTED_DUELING_CENTERING:
            raise ValueError(
                "Unknown Q-network dueling centering: "
                f"{self.q_network_dueling_centering!r}"
            )
        self.wait_training_penalty = float(
            kwargs.get("wait_training_penalty", 0.5)
        )
        self.terminal_on_truncation = bool(
            kwargs.get("terminal_on_truncation", False)
        )
        self.close_options_on_episode_end = bool(
            kwargs.get("close_options_on_episode_end", False)
        )
        # epsilon
        total_steps = kwargs.get('n_episodes',3000)*kwargs.get('n_steps',500)
        dp          = total_steps*0.80
        self.epsilon     = kwargs.get('epsilon',1.0)
        self.epsilon_min = kwargs.get('epsilon_min',0.05)
        self.epsilon_decay = (self.epsilon_min/self.epsilon)**(1/dp)
        # Stable controller-to-output mappings are part of the checkpoint ABI.
        self.manager_options = canonical_manager_options(env.options)
        self.primitive_options = canonical_primitive_options(env.options)
        learned_selectors = [
            option
            for option in self.manager_options
            if is_storage_selector(option)
            and getattr(option, "requires_agent_gamma_match", True)
            and hasattr(option, "gamma")
        ]
        for selector in learned_selectors:
            if not math.isclose(
                float(selector.gamma), float(self.gamma), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    "Selector and agent must use the same gamma: "
                    f"selector={selector.gamma}, agent={self.gamma}"
                )
        # nets
        self.Q_manager_local = QNetwork(
            state_size,
            len(self.manager_options),
            seed=self.initialization_seed,
            normalization=self.q_network_normalization,
            dueling_centering=self.q_network_dueling_centering,
        ).to(self.device)
        self.Q_manager_target = copy.deepcopy(self.Q_manager_local).to(self.device)
        self.optimizer_manager= optim.Adam(self.Q_manager_local.parameters(),lr=self.lr_mgr)
        self.Q_worker_local = QNetwork(
            state_size,
            len(self.primitive_options),
            seed=self.initialization_seed + 1,
            normalization=self.q_network_normalization,
            dueling_centering=self.q_network_dueling_centering,
        ).to(self.device)
        self.Q_worker_target = copy.deepcopy(self.Q_worker_local).to(self.device)
        if self.q_network_normalization != LEGACY_BATCH_NORM:
            self.Q_manager_target.eval()
            self.Q_worker_target.eval()
        self.optimizer_worker = optim.Adam(self.Q_worker_local.parameters(),lr=self.lr_wrk)
        # sync
        self.Q_manager_target.load_state_dict(self.Q_manager_local.state_dict())
        self.Q_worker_target.load_state_dict(self.Q_worker_local.state_dict())
        self.scheduler_manager= optim.lr_scheduler.StepLR(self.optimizer_manager,step_size=2000,gamma=0.9)
        self.scheduler_worker = optim.lr_scheduler.StepLR(self.optimizer_worker, step_size=2000,gamma=0.9)
        # buffers
        buf_size = kwargs.get('buffer_size',1000000)
        self.ManagerBuffer = ManagerReplayBuffer(
            buf_size // 10,
            self.batch_size,
            device=self.device,
            verbose=self.verbose,
        )
        self.WorkerBuffer  = WorkerReplayBuffer(buf_size,     self.batch_size, device=self.device)

        if self.verbose:
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
        if kwargs.get("disable_tensorboard", False):
            self.writer = NullSummaryWriter()
        else:
            self.writer = SummaryWriter(
                log_dir=os.path.join(log_root, f"{run_name}_{stamp}")
            )

    def checkpoint_metadata(self):
        return {
            "q_network_architecture": Q_NETWORK_ARCHITECTURE,
            "q_network_normalization": self.q_network_normalization,
            "q_network_initialization": Q_NETWORK_INITIALIZATION,
            "q_network_dueling_centering": getattr(
                self,
                "q_network_dueling_centering",
                DUELING_ALL_OUTPUTS,
            ),
            "controller_initialization_seed": self.initialization_seed,
            "manager_initialization_seed": self.initialization_seed,
            "worker_initialization_seed": self.initialization_seed + 1,
            "wait_training_penalty": self.wait_training_penalty,
            "terminal_on_truncation": self.terminal_on_truncation,
            "close_options_on_episode_end": self.close_options_on_episode_end,
            **getattr(self, "controller_observation_metadata", {}),
        }

    def encode_state(self, state):
        encoded = np.asarray(self.state_encoder(state), dtype=np.float32).reshape(-1)
        if encoded.size != self.state_size:
            raise ValueError(
                "Controller observation size changed at runtime: "
                f"expected {self.state_size}, received {encoded.size}"
            )
        return encoded



    def select_action(self, state, eps, state_features=None):
        encoded = (
            self.encode_state(state)
            if state_features is None
            else np.asarray(state_features, dtype=np.float32)
        )
        s = torch.from_numpy(encoded).float().to(self.device)
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
            valid_primitive_indices = [
                i for i, option in enumerate(self.primitive_options)
                if option.initiation(state)
            ]
            if not valid_primitive_indices:
                raise RuntimeError("No primitive action is available in the current state")
            idx_w = max(valid_primitive_indices, key=lambda i: wrk_q[i].item())
            best_w = wrk_q[idx_w]
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


    def process_step(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        term,
        *,
        state_features=None,
        next_state_features=None,
    ):
        # clip per-step
        reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        # WaitOption penalty
        is_primitive_wait = (
            self.current_option
            and self.current_option.is_primitive
            and getattr(self.current_option, "action", None)
            == SmallRoomsEnv.ACTION_IDS["WAIT"]
        )
        if isinstance(self.current_option, WaitOption) or is_primitive_wait:
            reward -= self.wait_training_penalty
        s = (
            self.encode_state(state)
            if state_features is None
            else np.asarray(state_features, dtype=np.float32)
        )
        ns = (
            self.encode_state(next_state)
            if next_state_features is None
            else np.asarray(next_state_features, dtype=np.float32)
        )
        if self.current_option and self.current_option.is_primitive:
            self.WorkerBuffer.add(s, action, reward, ns, done)
        if self.current_option and not self.current_option.is_primitive:
            self.option_reward_traj.append(reward)
            if term:
                idx = self.manager_options.index(self.current_option)
                k   = len(self.option_reward_traj)
                Rcum= np.clip(self._discounted_return(self.option_reward_traj,self.gamma),
                              -self.reward_clip, self.reward_clip)
                s0 = getattr(self, "option_start_features", None)
                if s0 is None:
                    s0 = self.encode_state(self.option_start_state)
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
            "episode_mean_signed_deviation": [],
            "episode_mean_absolute_error": [],
            "episode_mean_tardiness": [],
            "episode_mean_earliness": [],
            "episode_within_target_window_rate": [],
            "episode_tardy_delivery_rate": [],
            "episode_p90_tardiness": [],
            "episode_success":   [],
            #"action_choices":    [],
            #"manager_q_log":     [],
            #"worker_q_log":      [],
        }

        checkpoint_every = 500

        def get_storage_selector():
            return find_storage_selector(self.env.options)

        for ep in range(1, n_episodes + 1):
            # --- reset env ---
            state = self.env.reset()

            # IMPORTANT: if env.reset() can rebuild options, refresh these each episode
            self.manager_options = canonical_manager_options(self.env.options)
            self.primitive_options = canonical_primitive_options(self.env.options)
            self.env.current_episode = ep

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
                state_features = self.encode_state(state)
                # 1) pick or continue option
                if self.current_option is None:
                    a_or_o = self.select_action(
                        state, self.epsilon, state_features=state_features
                    )
                    self.current_option = a_or_o

                    if not a_or_o.is_primitive:
                        self.option_start_state = state
                        self.option_start_features = state_features.copy()
                        self.option_reward_traj = []

                prim = self.current_option.policy(state)
                nxt, rew, done, info = self.env.step(prim)
                next_state_features = self.encode_state(nxt)

                # 2) accumulate step reward into selector pendings
                selector_on_step = (
                    getattr(selector, "on_step", None)
                    if selector is not None
                    else None
                )
                if callable(selector_on_step):
                    selector_on_step(rew, info)
                elif selector is not None and hasattr(selector, "pending") and hasattr(selector, "gamma"):
                    accumulate_selector_pending(selector, rew, info)

                # 3) notify selector on delivery events
                if (
                    selector is not None
                    and hasattr(selector, "on_delivery")
                    and info.get("delivered_block", False)
                    and "delivery_error_time" in info
                ):
                    delivered = info["delivered_block"]
                    err_t = info["delivery_error_time"]
                    selector.on_delivery(
                        delivered,
                        err_t,
                        delivery_reward=info.get("delivery_reward"),
                    )

                # 4) logging / success condition
                if "delivery_error_time" in info:
                    this_ep_errors.append(info["delivery_error_time"])


                # 5) termination + learning
                term = self.current_option.termination(nxt)
                self.step_count += 1
                self.process_step(
                    state,
                    prim,
                    rew,
                    nxt,
                    done,
                    term,
                    state_features=state_features,
                    next_state_features=next_state_features,
                )
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
                if getattr(selector, "accepts_episode_outcome", False):
                    selector.on_episode_end(
                        success=bool(done),
                        truncated=bool(step >= max_steps and not done),
                    )
                else:
                    selector.on_episode_end()

            episode_all_delivered = 1.0 if done else 0.0
            logs["episode_success"].append(episode_all_delivered)
            logs["episode_returns"].append(ep_ret)

            timing = summarize_delivery_timing(
                this_ep_errors,
                getattr(self.env, "DELIVERY_TARGET_WINDOW", 20.0),
            )
            ep_avg_err = timing["mean_signed_deviation"]
            # Backward-compatible alias; this is signed bias, not an error norm.
            logs["episode_avg_error"].append(ep_avg_err)
            logs["episode_mean_signed_deviation"].append(ep_avg_err)
            logs["episode_mean_absolute_error"].append(
                timing["mean_absolute_error"]
            )
            logs["episode_mean_tardiness"].append(timing["mean_tardiness"])
            logs["episode_mean_earliness"].append(timing["mean_earliness"])
            logs["episode_within_target_window_rate"].append(
                timing["within_target_window_rate"]
            )
            logs["episode_tardy_delivery_rate"].append(
                timing["tardy_delivery_rate"]
            )
            logs["episode_p90_tardiness"].append(timing["p90_tardiness"])
            
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
                self.writer.add_scalar(
                    f"{self.tb_run_name}/mean_signed_delivery_deviation",
                    timing["mean_signed_deviation"],
                    ep,
                )
                self.writer.add_scalar(
                    f"{self.tb_run_name}/mean_absolute_timing_error",
                    timing["mean_absolute_error"],
                    ep,
                )
                self.writer.add_scalar(
                    f"{self.tb_run_name}/mean_tardiness",
                    timing["mean_tardiness"],
                    ep,
                )
                self.writer.add_scalar(
                    f"{self.tb_run_name}/mean_earliness",
                    timing["mean_earliness"],
                    ep,
                )
                self.writer.add_scalar(
                    f"{self.tb_run_name}/within_target_window_rate",
                    timing["within_target_window_rate"],
                    ep,
                )
                self.writer.add_scalar(
                    f"{self.tb_run_name}/tardy_delivery_rate",
                    timing["tardy_delivery_rate"],
                    ep,
                )

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
                    episode_mean_signed_deviation=np.asarray(logs["episode_mean_signed_deviation"], dtype=np.float32),
                    episode_mean_absolute_error=np.asarray(logs["episode_mean_absolute_error"], dtype=np.float32),
                    episode_mean_tardiness=np.asarray(logs["episode_mean_tardiness"], dtype=np.float32),
                    episode_mean_earliness=np.asarray(logs["episode_mean_earliness"], dtype=np.float32),
                    episode_within_target_window_rate=np.asarray(logs["episode_within_target_window_rate"], dtype=np.float32),
                    episode_tardy_delivery_rate=np.asarray(logs["episode_tardy_delivery_rate"], dtype=np.float32),
                    episode_p90_tardiness=np.asarray(logs["episode_p90_tardiness"], dtype=np.float32),
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
                print(f"Ep {ep:4d} | AvgR {mean_ret:7.2f} | Bias {err_str} |  Eps {self.epsilon:.3f}")

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
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "manager_option_ids": controller_ids(self.manager_options),
            "primitive_action_ids": controller_ids(self.primitive_options),
            "manager_state_dict": self.Q_manager_local.state_dict(),
            "worker_state_dict":  self.Q_worker_local.state_dict(),
            "manager_opt_state":  self.optimizer_manager.state_dict(),
            "worker_opt_state":   self.optimizer_worker.state_dict(),
            "epsilon":    self.epsilon,
            "step_count": self.step_count,
        }
        checkpoint_metadata = getattr(self, "checkpoint_metadata", None)
        if callable(checkpoint_metadata):
            ckpt.update(checkpoint_metadata())

        # Debug: list option names
        try:
            opt_names = [type(o).__name__ for o in self.env.options]
        except TypeError:
            opt_names = [type(o).__name__ for o in list(self.env.options)]
        print("[save_agent] options in env:", opt_names)

        # Robust selector lookup (avoids isinstance import-path issues)
        selector = find_storage_selector(self.env.options)

        if selector is not None and hasattr(selector, "q"):
            ckpt["selector_feature_version"] = getattr(selector, "FEATURE_VERSION", 1)
            ckpt["selector_return_definition"] = getattr(
                selector, "return_definition", None
            )
            ckpt["selector_gamma"] = float(getattr(selector, "gamma", self.gamma))
            ckpt["selector_state_dict"] = selector.q.state_dict()
            ckpt["selector_opt_state"]  = selector.opt.state_dict()
            ckpt["selector_call_count"] = selector.call_cnt
            ckpt["selector_eps"]        = selector.eps
            print("✓ added StorageSelectOption weights to checkpoint")
        elif selector is not None and callable(getattr(selector, "checkpoint", None)):
            ckpt["selector_kind"] = getattr(
                selector, "ARCHITECTURE_NAME", type(selector).__name__
            )
            ckpt["selector_feature_version"] = getattr(
                selector, "FEATURE_VERSION", None
            )
            ckpt["selector_return_definition"] = getattr(
                selector, "return_definition", None
            )
            ckpt["selector_gamma"] = float(
                getattr(selector, "gamma", self.gamma)
            )
            ckpt["selector_checkpoint"] = selector.checkpoint()
            print("✓ added versioned storage-selector checkpoint")
        else:
            print("[save_agent] trainable storage selector not found -> selector not saved")

        fname = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f") + "_HRL.pth"
        fpath = os.path.join(out_dir, fname)
        torch.save(ckpt, fpath)
        print(f"✓ checkpoint saved to {fpath}")
        return fpath
