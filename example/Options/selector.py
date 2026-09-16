# storage\_select\_option.py
import random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import deque
from option import BaseOption
from example.small_rooms_env import SmallRoomsEnv
from example.helper.tools import _astar
from typing import Tuple, List
# ───────────────────────── tiny network ──────────────────────────

class TinyQ(nn.Module):
    def __init__(self, D: int, n_out: int, hidden_1: int = 256, hidden_2: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, hidden_1),
            nn.LayerNorm(hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, n_out),
        )

    def forward(self, x):
        return self.net(x)

# ─────────────────────── Storage-Select option ───────────────────

class StorageSelectOption(BaseOption):

    # v3 removes required-duration leakage from blocks that have not arrived.
    FEATURE_VERSION = 3
    is_storage_selector = True
    RETURN_FULL_ENVIRONMENT = "full_environment"
    RETURN_EXPLICIT_TERMINAL = "explicit_terminal"
    RETURN_DEFINITION_VERSION = 1
    RETURN_MODES = frozenset(
        (RETURN_FULL_ENVIRONMENT, RETURN_EXPLICIT_TERMINAL)
    )

# -------------------------------------------------------------
    def __init__(self,
                env:           SmallRoomsEnv,
                lr:            float = 5e-4,
                buffer_size:   int   = 100_000,
                batch_size:    int   = 256,
                gamma:         float = 0.99,
                update_freq:   int   = 10,
                return_mode:   str   = RETURN_FULL_ENVIRONMENT):
        super().__init__(is_primitive=False)

        if return_mode not in self.RETURN_MODES:
            valid = ", ".join(sorted(self.RETURN_MODES))
            raise ValueError(
                f"Unknown selector return mode {return_mode!r}; expected {valid}"
            )

        # ---------- env & bookkeeping --------------------------------------
        self.env          = env
        self.cells        = list(env.storage_positions)    # fixed order
        self.n_cells      = len(self.cells)
        #self.device       = torch.device("cuda" if torch.cuda.is_available()
                                        #else "cpu")
        self.device       = torch.device("cpu")

        # ---------- static per-cell features -------------------------------
        max_d           = env.grid_rows + env.grid_cols
        self.f_door     = np.array([(max_d - env.manhattan_distance(env.door_cell, c))/max_d
                                    for c in self.cells], np.float32)
        self.f_exit     = np.array([(max_d - min(env.manhattan_distance(c, ex)
                                    for ex in env.exit_cells))/max_d
                                    for c in self.cells], np.float32)
        self.f_usage = np.zeros(self.n_cells, dtype=np.float32)
        self.MAX_B = 5
        self.MAX_T = max(1, int(getattr(env, "MAX_T", 150)))

        state_dim       = 5 * self.n_cells + 2  + self.MAX_B * 5 + 2


        
        self.q          = TinyQ(state_dim, self.n_cells).to(self.device)
        self.q_targ     = TinyQ(state_dim, self.n_cells).to(self.device)
        self.q_targ.load_state_dict(self.q.state_dict())
        self.opt        = torch.optim.Adam(self.q.parameters(), lr=lr)


        # ---------- replay -------------------------------------------------
        self.buffer       = deque(maxlen=buffer_size)
        self.batch_size   = batch_size
        self.gamma        = gamma
        self.update_freq  = update_freq
        self.return_mode  = return_mode
        self.return_definition = (
            f"selector_event_{return_mode}_v{self.RETURN_DEFINITION_VERSION}"
        )
        self.learn_calls  = 0
        self.learn_steps  = 0 
        self.learning_enabled = True
        self.soft_tau     = 0.01            # target-net soft-update factordd
        self.pending      = {}                # for the option initiation predicate
        self.fail_R       = -50.0              # penalty for failed initiation

        # ---------- ε-schedule (pure call count) ---------------------------
        self.eps         = 0.9
        self.call_cnt    = 0                # increment **each invocation**
        self.flat_calls  = 300              # keep ε high for the first N calls
        self.decay_calls = 3000            # then linear → 0.05

        # ---------- one-shot LR decay --------------------------------------
        self.lr_scaled   = False

        # ---------- option-specific state ----------------------------------
        self.block       = None             # the block we’re holding
        self.last_s      = None             # flattened state
        self.last_a      = None             # chosen cell index
        self.on_env_reset()           # recompute per-episode quantities
        self.loss_history = []



    def _block_feats(self) -> np.ndarray:
        feats: List[float] = []
        carried = next((b for b in self.env.blocks if b.carrying), None)
        remaining = [b for b in self.env.blocks if b is not carried]

        def remaining_storage_steps(block):
            required = getattr(block, "storage_steps_needed", self.MAX_T)
            if block.stored and block.stored_time_step is not None:
                elapsed = max(0, self.env.time_steps - block.stored_time_step)
                return max(0, required - elapsed)
            return required

        def information_safe_order(block):
            if block.delivered:
                return 3, 0, block.label
            if block.stored:
                return 0, remaining_storage_steps(block), block.label
            if block.position is not None:
                # Required duration is observable once the block has arrived.
                return 1, remaining_storage_steps(block), block.label
            # Unarrived blocks are ordered without consulting hidden duration.
            return 2, 0, block.label

        remaining.sort(key=information_safe_order)
        order = ([carried] if carried is not None else []) + remaining
        max_len = self.env.grid_rows + self.env.grid_cols

        def normalise_time(value):
            return float(np.clip(value / float(self.MAX_T), 0.0, 1.0))

        for b in order[:self.MAX_B]:
            if b.delivered:
                t = 3
            elif b.carrying:
                t = 1
            elif b.stored:
                t = 2
            else:
                t = 0

            # block not yet arrived / outside system
            if b.position is None:
                d_agent = 1.0
                d_stor = 1.0 if b.storage_location is None else 0.0
                d_exit = 1.0
                rem_t = 0.0
                feats += [t/3.0, d_agent, d_stor, d_exit, rem_t]
                continue

            d_agent = self.env.manhattan_distance(
                self.env.current_state, b.position
            ) / max_len

            if b.stored:
                d_stor = 0.0
            else:
                loc = b.storage_location
                d_stor = (
                    self.env.manhattan_distance(b.position, loc) / max_len
                    if loc is not None else 1.0
                )

            d_exit = self.env.compute_goal_distance(
                b.position, self.env.exit_cells
            ) / max_len

            if b.carrying:
                rem_t = normalise_time(b.storage_steps_needed)
            elif b.stored and not b.delivered:
                if b.stored_time_step is not None:
                    elapsed = min(
                        self.env.time_steps - b.stored_time_step,
                        b.storage_steps_needed
                    )
                else:
                    elapsed = 0
                rem = b.storage_steps_needed - elapsed
                rem_t = normalise_time(rem)
            else:
                rem_t = 0.0

            feats += [t/3.0, d_agent, d_stor, d_exit, rem_t]

        feats += [0.0] * (self.MAX_B * 5 - len(feats))
        return np.array(feats, dtype=np.float32)

    # ───────────────────── helper: build flat state ───────────────────────
    def _φ(self,
        mask: np.ndarray,
        f_usage: np.ndarray,
        f_cong: np.ndarray,
        agent_pos: Tuple[int,int]
    ) -> np.ndarray:
        ax, ay = agent_pos
        agent_x = ax / self.env.grid_rows
        agent_y = ay / self.env.grid_cols

        blk_feats = self._block_feats()

        stored_blocks = [
            b for b in self.env.blocks if b.stored and not b.delivered
        ]
        if stored_blocks:
            mean_exit_dist = np.mean([
                self.env.compute_goal_distance(b.position, self.env.exit_cells)
                for b in stored_blocks
            ]) / (self.env.grid_rows + self.env.grid_cols)
        else:
            mean_exit_dist = 0.0  # Default when no stored blocks

        block_count = max(1, len(self.env.blocks))
        globals = np.array([
            sum(not b.delivered for b in self.env.blocks) / block_count,
            mean_exit_dist  # Use computed or default value
        ], dtype=np.float32)

        return np.concatenate([
            mask,                # (n_cells,)
            self.f_door,         # (n_cells,)
            self.f_exit,         # (n_cells,)
            f_usage,             # (n_cells,)
            f_cong,              # (n_cells,)
            np.array([agent_x, agent_y], np.float32),  # (2,)
            blk_feats,          # (MAX_B*5,)
            globals             # (2,)
        ], axis=0)



    def intrinsic_reward(self, state, action, next_state, info):
            return 0.0

    # ───────────────────── initiation predicate ───────────────────────────
    def initiation(self, _state):
        # pick up whichever block you’re carrying that still needs a storage_cell
        self.block = next(
            (
                b for b in self.env.blocks
                if b.carrying
                and not b.stored
                and b.storage_location is None
            ),
            None
        )
        # allow this option exactly when such a block exists
        return (self.block is not None)





    # ────────────────────────── core policy ───────────────────────────────
    def policy(self, _s):

        if self.block is None:
            return SmallRoomsEnv.ACTION_IDS["WAIT"]
        # 1) build features & φ_s
        usage_counts = np.array([self.env.storage_counts[c] for c in self.cells], np.float32)
        max_u        = max(usage_counts.max(), 1.0)
        self.f_usage = 1.0 - (usage_counts / max_u)

        free = set(self.env.get_available_storage_positions())
        mask = np.array([1. if c in free else 0. for c in self.cells], np.float32)

        # ---- NEW: congestion mask ----
        occ = {b.position for b in self.env.blocks
            if (not b.delivered) and (not b.carrying)}
        # 1. if cell in occ → congested = 1.0, else 0.0
        self.f_cong = np.array([1. if cell in occ else 0.
                                for cell in self.cells],
                            dtype=np.float32)

        agent_pos    = self.env.current_state
        self.last_s  = self._φ(mask, self.f_usage, self.f_cong, agent_pos)

        # 2) pick an index
        free_idx = np.flatnonzero(mask)
        if len(free_idx) == 0:
            return SmallRoomsEnv.ACTION_IDS["WAIT"]

        if self.learning_enabled and random.random() < self.eps:
            self.last_a = int(random.choice(free_idx))
        else:
            with torch.no_grad():
                qvals     = self.q(torch.from_numpy(self.last_s)
                                .unsqueeze(0).to(self.device))[0]
            self.last_a = int(max(free_idx, key=lambda i: qvals[i].item()))

        chosen_cell = self.cells[self.last_a]
        #Immediate reward for choosing a storage location
        dist_door = len(_astar(self.env.rooms,
                            self.env.door_cell,
                            chosen_cell,
                            blocked=set()))
        # distance slot → nearest exit (Manhattan)
        dist_exit = min(self.env.manhattan_distance(chosen_cell, ex)
                        for ex in self.env.exit_cells)

        # normalise
        L              = self.env.grid_rows + self.env.grid_cols
        d_door_norm    = dist_door / L
        d_exit_norm    = dist_exit / L
        blk = self.block

        # urgency 1 = needs to be delivered soon, 0 = relaxed
        urgency = 1.0 - np.clip(blk.storage_steps_needed / float(self.MAX_T), 0.0, 1.0)

        α_exit, α_door = 5.0, 3.0          # weights
        α_path = 0.05

        #imm_cost  =  -(urgency*α_exit*d_exit_norm)   \
           #+  α_path * self._path_length_reward()

        imm_cost =  0.0
        
        #print(f"Reward info: {imm_cost}")
        # --------------------------------------------------------------------

        path_len   = min(self.env.manhattan_distance(chosen_cell, ex) for ex in self.env.exit_cells)
        self.env.store_events.append({
        "episode":    self.env.current_episode,
        "t_step":     self.env.time_steps,
        "phase":      "chosen",
        "block_label": blk.label,                # renamed
        "row":        chosen_cell[0],             # split out
        "col":        chosen_cell[1],
        "path_exit":  path_len,
        "urgency":    blk.storage_steps_needed,
        "imm_cost":   imm_cost
        })

        # 3) commit to block & pending
        
        chosen = chosen_cell
        # store the block in the chosen cell
        blk.storage_location      = chosen
        blk.storage_chosen_state  = self.last_s.copy()
        blk.storage_chosen_idx    = self.last_a
        if self.learning_enabled:
            self.pending[blk.label] = {
                'phi_s': blk.storage_chosen_state,
                'a':     blk.storage_chosen_idx,
                'imm':   imm_cost,
                'acc':   0.0,
                'disc':  1.0,
                'steps':  0,
            }
        # record storage info
        self.env.store_events.append({
            "episode":           self.env.current_episode,
            "block_label":       blk.label,
            "row":                chosen[0],
            "col":                chosen[1],
            "time_left_at_pick": blk.storage_steps_needed
        })

        # 4) no actual movement: WAIT to terminate the option
        #    learning (self._learn) and buffer-append happen in your on_delivery / on_episode_end hooks
        if self.learning_enabled:
            self._update_epsilon()
            self.learn_calls += 1
            if (
                len(self.buffer) >= self.batch_size
                and self.learn_calls % self.update_freq == 0
            ):
                self._learn()
        return SmallRoomsEnv.ACTION_IDS["WAIT"]
    
    def _path_length_reward(self) -> float:
        """
        Compute −(sum over all stored blocks of (steps to door)), 
        normalized by (n_cells × max_path_len) so it stays in [−1,0].
        """
        total = 0.0
        # worst‐case single path is grid_rows+grid_cols steps
        max_len = self.env.grid_rows + self.env.grid_cols
        for b in self.env.blocks:
            if b.storage_location is not None:
                # astar returns the full path list, so its length is num steps+1
                path = _astar(self.env.rooms,
                              b.storage_location,
                              self.env.door_cell,
                              blocked=set())
                total += len(path) - 1  
        # normalize by (n_cells * max_len)
        norm = self.n_cells * max_len
        return - total / norm


    def on_env_reset(self):
        # recompute per-episode quantities
        cells = list(self.env.storage_positions)
        if len(cells) != self.n_cells:
            raise RuntimeError(
                "Storage-cell count changed after selector construction; "
                "the selector output layer is no longer compatible"
            )
        self.cells = cells
        self.f_usage = np.zeros(self.n_cells, dtype=np.float32)
        self.f_cong = np.zeros(self.n_cells, dtype=np.float32)
        self.pending.clear()
        self.block = None
        self.last_s = None
        self.last_a = None

    def set_learning_enabled(self, enabled: bool):
        """Enable training behavior or freeze the selector for evaluation."""
        self.learning_enabled = bool(enabled)
        if not self.learning_enabled:
            self.eps = 0.0
            self.pending.clear()

    # ───────────────────────── learn from replay ──────────────────────────
    # -------------------------------------------------------------
    def on_delivery(
        self,
        block_label: str,
        error_time: float,
        delivery_reward: float | None = None,
    ):
        # 1) Pop the pending dict entry
        entry = self.pending.pop(block_label, None)
        if entry is None:
            # no option pick to credit
            return

        φ_s   = entry['phi_s']   # stored start state features
        a     = entry['a']       # chosen action index
        imm   = entry['imm']     # immediate shaping cost
        acc   = entry['acc']     # accumulated discounted transit rewards
        disc  = entry['disc']    # multiplier for the next reward after closure

        # 2) Recompute “after delivery” mask & usage exactly as in policy()
        free        = set(self.env.get_available_storage_positions())
        mask        = np.array([1. if c in free else 0. for c in self.cells], np.float32)
        mask2       = mask.copy()
        mask2[a]    = 0.0

        usage_counts = np.array([self.env.storage_counts[c] for c in self.cells], np.float32)
        max_u        = max(usage_counts.max(), 1.0)
        f_usage      = 1.0 - (usage_counts / max_u)

        # 3) Look up the block & its final storage location
        blk = next(b for b in self.env.blocks if b.label == block_label)
        storage_loc = blk.storage_location

        # 4) Build φ_s2 from the current env state
        φ_s2 = self._φ(mask2, f_usage, self.f_cong, self.env.current_state)

        # 5) Close one complete placement-to-delivery event return.  The
        # default full-environment definition has already accumulated the
        # delivery step, including its base and timing bonus.  The explicit
        # terminal definition removes that component during accumulation and
        # adds it here at exactly the same exponent.
        if self.return_mode == self.RETURN_FULL_ENVIRONMENT:
            terminal_reward = 0.0
            terminal_discount = 0.0
            R_option = imm + acc
        else:
            if delivery_reward is None:
                raise RuntimeError(
                    "explicit_terminal selector return requires the delivery "
                    "reward emitted by the environment"
                )
            if "terminal_discount" not in entry:
                raise RuntimeError(
                    "explicit_terminal selector return was closed before its "
                    "delivery step was accumulated"
                )
            terminal_reward = float(delivery_reward)
            terminal_discount = float(entry["terminal_discount"])
            R_option = imm + acc + terminal_discount * terminal_reward

        # 7) (Optional) Log everything for analysis
        self.env.store_events.append({
            "episode":     self.env.current_episode,
            "t_step":      self.env.time_steps,
            "phase":       "delivered",
            "block_label": blk.label,
            "row":         storage_loc[0],
            "col":         storage_loc[1],
            "error_t":     error_time,
            "imm_cost":    imm,
            "acc_transit": acc,
            "disc":        disc,
            "duration":    entry.get("steps", None),
            "return_definition": self.return_definition,
            "delivery_reward": float(delivery_reward or 0.0),
            "terminal_reward": terminal_reward,
            "terminal_discount": terminal_discount,
            "total_R":     R_option
        })

        # Treat each block assignment as a delayed contextual decision. Its
        # target terminates when that block is delivered, so no later storage
        # assignment is bootstrapped into this return.
        self.buffer.append((φ_s, a, R_option, φ_s2, 1.0))



    def on_episode_end(self):
        # 1) Build φ_s2 just like in on_delivery
        free        = set(self.env.get_available_storage_positions())
        mask        = np.array([1. if c in free else 0. for c in self.cells], np.float32)
        mask2       = mask.copy()
        usage_counts = np.array([self.env.storage_counts[c] for c in self.cells], np.float32)
        max_u        = max(usage_counts.max(), 1.0)
        f_usage      = 1.0 - (usage_counts / max_u)
        φ_s2 = self._φ(mask2, f_usage, self.f_cong, self.env.current_state)

        # 2) For each “still‐pending” pick, compute the total SMDP return
        for label, entry in self.pending.items():
            φ_s  = entry['phi_s']
            a    = entry['a']
            imm  = entry['imm']
            acc  = entry['acc']
            disc = entry['disc']

            # failure‐case: discount your fail_R just like you would a final reward
            R_option = imm + acc + disc * self.fail_R

            # (Optional) log for diagnostics
            self.env.store_events.append({
                "episode":     self.env.current_episode,
                "t_step":      self.env.time_steps,
                "phase":       "failed",
                "block_label": label,
                "imm_cost":    imm,
                "acc_transit": acc,
                "disc":        disc,
                "fail_R":      self.fail_R,
                "total_R":     R_option
            })

            # Episode truncation also terminates this delayed assignment.
            self.buffer.append((φ_s, a, R_option, φ_s2, 1.0))

        # 4) Clear all pending entries for the next episode
        self.pending.clear()




    def _learn(self):

        # Stop learning after 15 000 calls to policy()
        #if self.call_cnt > 15_000:
            #return
        
        
        batch = random.sample(self.buffer,
                            min(self.batch_size, len(self.buffer)))
        s, a, r, s2, d = map(np.asarray, zip(*batch))
        s, s2 = map(lambda x: torch.from_numpy(x).to(self.device), (s, s2))
        a     = torch.from_numpy(a).long().unsqueeze(1).to(self.device)
        r     = torch.from_numpy(r).float().unsqueeze(1).to(self.device)
        d     = torch.from_numpy(d).float().unsqueeze(1).to(self.device)

        if not bool(torch.all(d == 1.0)):
            raise RuntimeError(
                "Selector replay contains a non-terminal event. Placement-to-"
                "delivery records must use complete returns without bootstrap."
            )

        q     = self.q(s).gather(1, a)
        # Every replay row is a complete selector event, so its Monte Carlo
        # return is the target.  The delivery state is not asserted to be a
        # selector decision state and no Q-value is bootstrapped from it.
        y = r

        loss  = F.mse_loss(q, y)
        self.loss_history.append(loss.item())

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


        # soft target-update
        #τ = self.soft_tau
        #with torch.no_grad():
            #for θ_t, θ in zip(self.q_targ.parameters(), self.q.parameters()):
                #θ_t.data.mul_(1-τ).add_(τ*θ.data)

        # hard target-update
        if (self.learn_steps + 1) % 1000 == 0:
            self.q_targ.load_state_dict(self.q.state_dict())

        # one-shot LR decay
        self.learn_steps += 1
        if (not self.lr_scaled) and self.learn_steps >= 10_000:
            for g in self.opt.param_groups:
                g['lr'] *= 0.3
            self.lr_scaled = True


    # ───────────────────────── ε schedule (call based) ────────────────────
    def _update_epsilon(self):
        #if self.env.time_steps % 100 == 0:
            #print(f"eps: {self.eps:.3f} ({self.call_cnt})")
        self.call_cnt += 1
        if self.call_cnt <= self.flat_calls:
            self.eps = 0.90
        elif self.call_cnt <= self.flat_calls + self.decay_calls:
            f = (self.call_cnt - self.flat_calls) / self.decay_calls
            self.eps = 0.90 - f * 0.85          # 0.90 → 0.05

            
        else:
            self.eps = 0.05
        #if self.call_cnt % 200 == 0:   # or whatever cadence you like
            #print(f"[StorageSelectOption] call #{self.call_cnt:5d} → eps = {self.eps:.3f}")


    # ─────────────────────────── boiler-plate ─────────────────────────────
    def termination(self, _s) -> bool: 
        return True
    
    def __str__(self) -> str:          
        return "StorageSelectOption"
    __repr__ = __str__
    
    def __hash__(self):                
        return hash("StorageSelectOption")
 
