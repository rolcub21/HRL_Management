# storage_select_option.py
import random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import deque
from option import BaseOption
from small_rooms_env import SmallRoomsEnv
from helper.tools import _astar, _manhattan
from typing import Tuple, List


# ───────────────────────── tiny network ──────────────────────────

class TinyQ(nn.Module):
    def __init__(self, D: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_out),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────── Storage-Select option ───────────────────
class RandomSelectOption(BaseOption):
# -------------------------------------------------------------
    def __init__(self,
                env:           SmallRoomsEnv,
                lr:            float = 3e-4,
                buffer_size:   int   = 100_000,
                batch_size:    int   = 256,
                gamma:         float = 0.95,
                update_freq:   int   = 30):
        super().__init__(is_primitive=False)

        # ---------- env & bookkeeping --------------------------------------
        self.env          = env
        self.cells        = env.storage_positions          # fixed order
        self.n_cells      = len(self.cells)
        self.device       = torch.device("cuda" if torch.cuda.is_available()
                                        else "cpu")

        # ---------- static per-cell features -------------------------------
        max_d           = env.grid_rows + env.grid_cols
        self.f_door     = np.array([(max_d - _manhattan(env.door_cell, c))/max_d
                                    for c in self.cells], np.float32)
        self.f_exit     = np.array([(max_d - min(_manhattan(c, ex)
                                    for ex in env.exit_cells))/max_d
                                    for c in self.cells], np.float32)
        self.f_usage = np.zeros(self.n_cells, dtype=np.float32)
        self.MAX_B = 5
        self.MAX_T = 15

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
        self.learn_calls  = 0
        self.learn_steps  = 0 
        self.soft_tau     = 0.01            # target-net soft-update factor
        self.pending      = {}                # for the option initiation predicate
        self.fail_R       = -50.0              # penalty for failed initiation

        # ---------- ε-schedule (pure call count) ---------------------------
        self.eps         = 0.9
        self.call_cnt    = 0                # increment **each invocation**
        self.flat_calls  = 500              # keep ε high for the first N calls
        self.decay_calls = 5000            # then linear → 0.05

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
        # sort so “Block1 vs Block3” doesn’t matter
        order = sorted(
            self.env.blocks,
            key=lambda b: (b.delivered, b.stored, b.carrying)
        )
        max_len = self.env.grid_rows + self.env.grid_cols

        for b in order[:self.MAX_B]:
            # (0) type ∈ {0,1,2,3} normalized
            if   b.delivered:  t = 3
            elif b.stored:     t = 2
            elif b.carrying:   t = 1
            else:              t = 0

            # (1) agent→block
            d_agent = self.env.manhattan_distance(
                self.env.current_state, b.position
            ) / max_len

            # (2) block→storage (0 if already stored, 1 if no storage_location)
            if b.stored:
                d_stor = 0.0
            else:
                loc = b.storage_location
                d_stor = (
                    self.env.manhattan_distance(b.position, loc) / max_len
                    if loc is not None else 1.0
                )

            # (3) block→exit
            d_exit = self.env.compute_goal_distance(
                b.position, self.env.exit_cells
            ) / max_len

            # ─────── remaining storage time feature ───────
            if b.carrying:
                # use its full budget, normalized
                rem_t = b.storage_steps_needed / float(self.MAX_T)
            elif b.stored and not b.delivered:
                rem = max(
                    0,
                    b.storage_steps_needed - (self.env.time_steps - b.stored_time_step)
                )
                rem_t = rem / float(self.MAX_T)
            else:
                rem_t = 0.0


            feats += [t/3.0, d_agent, d_stor, d_exit, rem_t]

        # pad up to MAX_B*5
        feats += [0.0] * (self.MAX_B*5 - len(feats))
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

        stored_blocks = [b for b in self.env.blocks if b.stored]
        if stored_blocks:
            mean_exit_dist = np.mean([
                self.env.compute_goal_distance(b.position, self.env.exit_cells)
                for b in stored_blocks
            ]) / (self.env.grid_rows + self.env.grid_cols)
        else:
            mean_exit_dist = 0.0  # Default when no stored blocks

        globals = np.array([
            sum(not b.stored for b in self.env.blocks) / self.MAX_B,
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





    # ───────────────────── initiation predicate ───────────────────────────
    def initiation(self, _s):
        self.block = next((b for b in self.env.blocks
                        if b.carrying and not b.stored
                            and b.storage_location is None), None)
        return self.block is not None


    # ────────────────────────── core policy ───────────────────────────────
    def policy(self, _s):
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

        if random.random() < self.eps:
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
        urgency = 1.0 - blk.storage_steps_needed / float(self.MAX_T)

        α_exit, α_door = 5.0, 3.0          # weights
        α_path = 0.05

        imm_cost  =  -(urgency*α_exit*d_exit_norm)   \
           +  α_path * self._path_length_reward()
        
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
        self.pending[blk.label]   = (blk.storage_chosen_state,
                                    blk.storage_chosen_idx,
                                    imm_cost)
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
        self._update_epsilon()
        self.learn_calls += 1
        if self.learn_calls % self.update_freq == 0 and len(self.buffer) >= self.batch_size:
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
            # recompute per‐episode quantities
            self.cells   = list(self.env.storage_positions)
            self.n_cells = len(self.cells)
            self.f_cong  = np.zeros(self.n_cells, dtype=np.float32)

    # ───────────────────────── learn from replay ──────────────────────────
    # -------------------------------------------------------------
    def on_delivery(self, block_label: str, error_time: float):
        # pull the saved (φ_s, a) for this block
        if self.pending.get(block_label) is None:
            #print(f"Warning: no pending choice for block {block_label}")
            return
        φ_s, a, imm_cost = self.pending.pop(block_label)

        # recompute the “next” mask & usage exactly as in policy()
        free        = set(self.env.get_available_storage_positions())
        mask        = np.array([1. if c in free else 0. for c in self.cells], np.float32)
        mask2       = mask.copy()
        mask2[a]    = 0.0

        usage_counts = np.array([self.env.storage_counts[c] for c in self.cells], np.float32)
        max_u        = max(usage_counts.max(), 1.0)
        f_usage      = 1.0 - (usage_counts / max_u)

        # look up the actual block and its storage_location
        blk = next(b for b in self.env.blocks if b.label == block_label)
        storage_loc = blk.storage_location

        # encode φ_s2 as “after delivery” state
        φ_s2 = self._φ(mask2, f_usage, self.f_cong, self.env.current_state)

        # delayed delivery reward
        delayed_R = self.env._delivery_reward(storage_loc, error_time)
        R =  imm_cost #+ 0.05* delayed_R 
        #print(f"Info: block {block_label} delivered, R={R:.2f} (imm_cost={imm_cost:.2f}, delayed_R={0.05* delayed_R :.2f})")

        self.env.store_events.append({
        "episode":     self.env.current_episode,
        "t_step":      self.env.time_steps,
        "phase":       "delivered",
        "block_label": blk.label,                # renamed
        "row":         storage_loc[0],
        "col":         storage_loc[1],
        "error_t":     error_time,
        "delayed_R":   delayed_R,
        "total_R":     R
        })



        # now push the complete transition
        #print(f"Info: state: {self.last_s}, action: {a}, reward: {R}, next_state: {φ_s2}")
        self.buffer.append((φ_s, a, R, φ_s2, 1.0))


    def on_episode_end(self):
        # recompute φ_s2 once
        free   = set(self.env.get_available_storage_positions())
        mask   = np.array([1. if c in free else 0. for c in self.cells], np.float32)
        mask2  = mask.copy()
        usage_counts = np.array([self.env.storage_counts[c] for c in self.cells], np.float32)
        max_u        = max(usage_counts.max(), 1.0)
        f_usage      = 1.0 - (usage_counts / max_u)
        φ_s2 = self._φ(mask2, f_usage, self.f_cong, self.env.current_state)

        for label, (φ_s, a, imm_cost) in self.pending.items():
            # “failure” reward = fail_R plus whatever immediate cost you stored
            total_R = self.fail_R 
            self.buffer.append((φ_s, a, total_R, φ_s2, 0.0))

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

        q     = self.q(s).gather(1, a)
        with torch.no_grad():
            q_s2 = self.q(s2)
            # Extract mask from state features (already a tensor)
            mask2 = s2[:, :self.n_cells]  # Shape: [batch_size, n_cells]
            # Mask invalid actions by adding large negative values
            q_s2_masked = q_s2 + (mask2 - 1.0) * 1e8  # 0 becomes -1e8, 1 remains
            best = q_s2_masked.argmax(dim=1, keepdim=True)
            q_t = self.q_targ(s2).gather(1, best)
            y    = r + (1-d)*self.gamma*q_t

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
        if self.learn_steps % 1000 == 0:
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



    # ─────────────────────────── boiler-plate ─────────────────────────────
    def termination(self, _s) -> bool: return True
    def __str__(self) -> str:          return "RandomSelectOption"
    __repr__ = __str__
    def __hash__(self):                return hash("RandomSelectOption")
