from __future__ import annotations

import numpy as np

import random
import time
import heapq
from typing import Tuple, List
from math import exp
from copy import deepcopy
from environment import BaseEnvironment
from example.block_instance import Blocks   # Your Blocks class
from example.episode_instance import EpisodeInstance
from GA.helper_functions import plan_store_block, plan_retrieve_block
from collections import deque

class SmallRoomsEnv(BaseEnvironment):
    """
    SmallRoomsEnv with stochastic arrivals & processing times:

      • In-bound blocks arrive after exponentially-distributed gaps
        (mean = 1 / arrival_rate).

      • Each block’s required storage time is drawn from a Poisson( proc_mean ).
    """

    # ───────────── static action maps ─────────────
    ACTION_NAMES = {0:"UP",1:"DOWN",2:"LEFT",3:"RIGHT",4:"PICKUP",5:"PUTDOWN",6:"WAIT"}
    ACTION_IDS   = {v:k for k,v in ACTION_NAMES.items()}
    DELIVERY_BASE_REWARD = 10.0
    DELIVERY_MAX_TIMING_BONUS = 30.0
    DELIVERY_TARGET_WINDOW = 20.0

    # ─────────────────────────── ctor ────────────────────────────
    def __init__(self,
                 *,
                 options:        list = None,
                 grid_rows:      int  = 10,
                 grid_cols:      int  = 10,
                 start_state:    tuple[int,int] = (1,1),
                 door_cell:      tuple[int,int] | None = None,
                 exit_cells:     list[tuple[int,int]] | None = None,
                 pickup_cells:   tuple[int,int] | None = None,
                 choose_storage: bool = False,
                 number_blocks:  int  = 40,
                 #
                 # NEW stochastic parameters
                 arrival_rate:   float = 1.5,   # λ  (mean inter-arrival = 1/λ)
                 proc_mean:      int   = 100,   # Poisson mean for storage steps
    ):
        super().__init__()
        options = set(options or [])
        self.options        = options
        self.grid_rows      = grid_rows
        self.grid_cols      = grid_cols
        self.start_state    = start_state
        self.choose_storage = choose_storage
        self.arrival_rate   = arrival_rate
        self.proc_mean      = proc_mean
        self.MAX_T          = 150

        # ───────────── layout basics ─────────────
        if door_cell is None:
            door_cell = (0, grid_cols - 2)
        self.door_cell = door_cell

        dr, dc           = self.door_cell
        self.pickup_cell = pickup_cells or (dr + 1, dc)     # single pickup slot
        self.waiting_cell = (dr, dc)                        # queue starts here

        # ───────────── exits (Option 1: single wide gate) ─────────────
        if exit_cells is None:
            exit_row = self.grid_rows - 1  # bottom border row
            gate_width = 7                 # <-- pick 5/7/9/etc (must be odd ideally)

            exit_center_c = dc             # align gate with door column
            half = gate_width // 2

            # keep inside [1, grid_cols-2]
            c0 = max(1, exit_center_c - half)
            c1 = min(self.grid_cols - 2, exit_center_c + half)

            exit_cells = [(exit_row, c) for c in range(c0, c1 + 1)]

        self.exit_cells   = exit_cells
        self.exit_states  = self.exit_cells
        self.goal_state   = self.exit_cells

        # ───────────── grid & storage map ─────────────
        self.rooms              = self._initialise_rooms()
        self.storage_positions  = self._get_storage_positions()
        self.storage_counts     = {cell: 0 for cell in self.storage_positions}

        # ───────────── block population ─────────────
        self.number_blocks = int(number_blocks)
        if self.number_blocks <= 0:
            raise ValueError("number_blocks must be positive")
        self.blocks        = [Blocks(label=f"B{i+1}") for i in range(self.number_blocks)]

        # Fixed or uniformly random storage durations -----------------------------
        #inner_cells = (self.grid_rows - 2) * (self.grid_cols - 2)
        #min_deadline = max(1, inner_cells // self.number_blocks)
        #max_deadline = inner_cells
        #storage_deadlines = np.random.randint(low=min_deadline, high=max_deadline + 1, size=self.number_blocks)

        #for blk, deadline in zip(self.blocks, storage_deadlines):
            #blk.storage_steps_needed  = int(deadline)
            #blk.storage_steps_elapsed = 0


        # Exponential inter-arrival schedule --------------------------------------
        #inter_arrivals     = np.random.exponential(scale=1 / self.arrival_rate,
                                                   #size=self.number_blocks)
        #arrival_steps      = np.cumsum(inter_arrivals).astype(int)
        #for blk, t in zip(self.blocks, arrival_steps):
            #blk.arrival_step = int(t)        # attribute added on-the-fly

        # misc
        self.gamma                 = 0.99
        self.store_events          = []
        self.delivery_error_times  = []
        self.current_state         = None
        self.current_episode       = 0


    def sample_episode_instance(self, seed=None) -> EpisodeInstance:
        """Sample an immutable episode without mutating environment state."""

        rng = np.random if seed is None else np.random.default_rng(seed)
        storage_deadlines = rng.poisson(
            lam=self.proc_mean, size=self.number_blocks
        )
        storage_deadlines = np.maximum(storage_deadlines, 1)
        if self.arrival_rate > 0:
            inter_arrivals = rng.exponential(
                scale=1.0 / self.arrival_rate,
                size=self.number_blocks,
            )
            arrival_steps = np.cumsum(inter_arrivals).astype(int)
        else:
            arrival_steps = np.zeros(self.number_blocks, dtype=int)
        arrival_steps[0] = 0
        return EpisodeInstance(
            schema_version=EpisodeInstance.SCHEMA_VERSION,
            seed=None if seed is None else int(seed),
            arrival_rate=float(self.arrival_rate),
            proc_mean=float(self.proc_mean),
            arrival_steps=tuple(int(value) for value in arrival_steps),
            storage_steps_needed=tuple(
                int(value) for value in storage_deadlines
            ),
            grid_rows=int(self.grid_rows),
            grid_cols=int(self.grid_cols),
            start_state=tuple(self.start_state),
            door_cell=tuple(self.door_cell),
            pickup_cell=tuple(self.pickup_cell),
            waiting_cell=tuple(self.waiting_cell),
            exit_cells=tuple(self.exit_cells),
            storage_positions=tuple(self.storage_positions),
            room_rows=tuple("".join(row) for row in self.rooms),
        )

    def reset(self, instance: EpisodeInstance | None = None):
        """Reset from an exact instance or sample a backward-compatible one."""
        self.current_state  = self.start_state
        self.store_events.clear()
        self.delivery_error_times.clear()
        self.time_steps     = 0
        self.current_time   = time.time()
        self.current_episode = getattr(self, "current_episode", 0)

        # storage map may have changed
        self.storage_positions = self._get_storage_positions()
        self.storage_counts = {cell: 0 for cell in self.storage_positions}

        # re-initialise each block
        #for i, blk in enumerate(self.blocks):
            #blk.carrying            = blk.stored = blk.delivered = blk.picked = False
            #blk.reset_cleared()
            #blk.storage_location    = None
            #blk.stored_time_step    = blk.delivered_time_step = blk.delivery_error_time = None
            # place first block into pickup slot, others stay outside until arrival
            # first block at pickup, rest at waiting cell
            #blk.position = self.pickup_cell if i == 0 else self.waiting_cell

        if instance is None:
            instance = self.sample_episode_instance()
        else:
            if not isinstance(instance, EpisodeInstance):
                raise TypeError("instance must be an EpisodeInstance")
            instance.validate_for(self)
        self.current_episode_instance = instance
        storage_deadlines = instance.storage_steps_needed
        arrival_steps = instance.arrival_steps

        # re-initialise each block
        for i, blk in enumerate(self.blocks):
            blk.carrying = False
            blk.stored = False
            blk.delivered = False
            blk.picked = False
            blk.reset_cleared()
            blk.storage_location = None
            blk.stored_time_step = None
            blk.delivered_time_step = None
            blk.delivery_error_time = None

            blk.storage_steps_needed = int(storage_deadlines[i])
            blk.storage_steps_elapsed = 0
            blk.arrival_step = int(arrival_steps[i])

            # blocks start outside; they enter when arrival time is reached
            blk.position = None

        self.update_pickup_queue()


        if self.choose_storage:
            avail = self.get_available_storage_positions()
            random.shuffle(avail)
            for blk in self.blocks:
                blk.storage_location = avail.pop()

        for option in tuple(self.options):
            on_env_reset = getattr(option, "on_env_reset", None)
            if callable(on_env_reset):
                on_env_reset()

        return self.get_current_state()

    def step(self, action):
        """
        Execute the agent's action and return (next_state, reward, terminal, info).

        Rules:
        - Movement: update agent position based on the chosen action.
        - PICKUP: if a block is at the current state and not already carried/delivered, pick it up.
        - PUTDOWN: if carrying a block:
                    a. At the storage location:
                        * If the block is not yet stored, mark it as stored and reward +10 (or here +3, etc.).
                    b. At the exit:
                        * If the block is already stored, deliver it and reward +30 (or computed via a function).
                    c. Otherwise, dropping is illegal.
        - WAIT: no state change.

        """
        # Positions of the blocks
        #for block in self.blocks:
            #print(f"Block {block.label} position: {block.position}, storage_location : {block.storage_location}")
        prev_locs = {b.label: b.storage_location for b in self.blocks}
        action_name = self.ACTION_NAMES[action]

        # Fallback: if action is None, default to WAIT.
        if action is None:
            print("Warning: action is None, defaulting to WAIT.")
            action = self.ACTION_IDS["WAIT"]
            action_name = "WAIT"
            
        self.time_steps += 1
        next_state = self._get_intended_cell(self.current_state, action)
        reward = -0.09  # default per-step cost.
        info = {
            'stored_block': False,
            'relocated_block': False,
            'delivered_block': False,
            'goal_reached': False,
            'episode_instance_id': self.current_episode_instance.instance_id,
        }
        
        # Movement: update state if the intended cell is not a wall.
        if self.rooms[next_state[0]][next_state[1]] != "#":
            # Save current state as old_state for reward shaping
            old_agent_state = self.current_state  # (row, col)
            self.current_state = next_state

            for block in self.blocks:
                if block.carrying:
                    block.position = self.current_state
                    block.last_position = self.current_state
                    

                    #block.distance_to_target = self.calculate_distance_to_target(block)

        else:
            # If movement is blocked, keep old state.
            next_state = self.current_state
            old_agent_state = self.current_state

        self.current_time = time.time()
        
        # Handle PICKUP.
        if action_name == "PICKUP":
            if not any(block.carrying for block in self.blocks):
                for block in self.blocks:
                    if (self.current_state == block.position and 
                        not block.delivered and 
                        not block.carrying):
                        block.carrying = True
                        block.picked = True
                        block.position = self.current_state
                        info["picked_block"] = block.label
                        break
                        #print(f"Block {block.label} picked up at time step {self.time_steps}.")
                        #reward += 3.0  # Un-commented to add reward for picking up a block
                    
                    
                        
                        

        # Handle PUTDOWN.
        elif action_name == "PUTDOWN":
            for block in self.blocks:
                if block.carrying:
                    if not block.stored and not block.delivered:
                        if block.storage_location and self.current_state == block.storage_location:
                            block.stored = True
                            block.carrying = False
                            block.position = block.storage_location  # So it “lives” in the storage cell
                            block.stored_time_step = self.time_steps
                            info['stored_block'] = block.label
                            self.storage_counts[block.storage_location] += 1
                            #print(f"Storage count for {block.storage_location}: {self.storage_counts[block.storage_location]}")
                            reward += 5.0  # Un-commented to add reward for storing a block
                            #print(f"Block {block.label} stored at time step {self.time_steps} at {block.storage_location}.")
                            block.start_storage_timer()
                            

                    # Case B: The block is already stored => deliver it if at exit
                    elif block.stored and not block.delivered: # and block.is_storage_time_elapsed():
                        if self.current_state in self.exit_states:
                            # Compute deadline: when the block should ideally be delivered.
                            block.delivered_time_step = self.time_steps
                            deadline_time = block.stored_time_step + block.storage_steps_needed
                            # Compute error_time: positive if late, negative if early.
                            error_time = block.delivered_time_step - deadline_time  
                            block.delivery_error_time = error_time
                            
                            # Optionally, record or log the error time.
                            info['delivery_error_time'] = error_time
                            self.delivery_error_times.append((block.label, error_time, self.current_episode))
                            #reward += 30.0  # Un-commented to add reward for delivering a block
                            delivery_reward = self._delivery_reward(block.storage_location, error_time)
                            reward += delivery_reward
                            info["delivery_reward"] = delivery_reward
                            #print(f"Block {block.label} delivered at time step {self.time_steps}, with error time {error_time} and reward {reward}.")
                            
                            block.delivered = True
                            block.carrying = False
                            block.delivered_time_step = self.time_steps
                            info['delivered_block'] = block.label
                        elif (
                            block.storage_location
                            and self.current_state == block.storage_location
                        ):
                            # A PSLAP relocation preserves the original storage
                            # event and clock. It is not a second placement and
                            # must not receive another placement reward.
                            block.carrying = False
                            block.position = self.current_state
                            info['relocated_block'] = block.label
                        else:
                            block.carrying = False
                            info['illegal_drop'] = block.label

                    # If not stored or delivered => no special condition met => e.g. illegal drop
                    else:
                        block.carrying = False
                        #reward -= 5.0  # Penalty for illegal drop
                        info['illegal_drop'] = block.label
                        #break


        
        # WAIT action.
        elif action_name == "WAIT":
            next_state = self.current_state
        else:
            next_state = self.current_state

        # Increment storage timer for any stored but not yet delivered block.
        for block in self.blocks:
            if block.stored and not block.delivered:
                block.increment_storage_steps()
        
        self.block_moves(action)
        self.update_pickup_queue()
        
        # Check if terminal (e.g., if block delivered)
        if self.is_state_terminal(self.current_state):
            info['goal_reached'] = True
            #print("Goal reached!")
            #reward += self.number_blocks * 20  # Added reward for reaching the goal
            #reward += 100.0  # Added reward for reaching the goal
        # --- NEW: Potential-Based Reward Shaping ---
        # If a block is being carried, add a shaping reward based on the change in distance to the target.
        carried_block = None
        for block in self.blocks:
            if block.carrying:
                carried_block = block
                break

        if carried_block is not None:
            # Determine the target.#base
            if not carried_block.stored:
                target = carried_block.storage_location
            else:
                if isinstance(self.goal_state, (list, tuple)):
                    if len(self.goal_state) == 0:
                        target = None
                    elif len(self.goal_state) == 1:
                        target = self.goal_state[0]
                    else:
                        target = min(self.goal_state, key=lambda exit_cell: self.manhattan_distance(self.current_state, exit_cell))
                else:
                    target = self.goal_state

            if target is None:
                potential_before, potential_after = 0, 0
            else:
                potential_before = -self.manhattan_distance(old_agent_state, target)
                potential_after  = -self.manhattan_distance(self.current_state, target)
            gamma = 0.99

            shaping_reward = (gamma * potential_after) - potential_before
            shaping_scale = 0.5 # you can tune this factor
            #reward += shaping_scale * shaping_reward
            #print(f"Shaping reward: {shaping_reward}, Total reward: {reward}")

            # ——— NEW: storage_chosen flag ———
            chosen = any(
                prev_locs[b.label] is None and b.storage_location is not None
                for b in self.blocks
            )
            info['storage_chosen'] = chosen
            
        return self.get_current_state(), reward, self.is_state_terminal(self.current_state), info

    def _initialise_rooms(self):
        rooms = np.full((self.grid_rows, self.grid_cols), ".", dtype=str)
        rooms[0,:]  = rooms[-1,:] = rooms[:,0] = rooms[:,-1] = "#"
        dr, dc = self.door_cell
        rooms[dr, dc] = "."
        for r, c in self.exit_cells:
            rooms[r, c] = "."
        return rooms
   
    def distance(self, state1, state2):
        """
        Compute the Manhattan distance between two states.
        """
        return abs(state1[0] - state2[0]) + abs(state1[1] - state2[1])
    
    def distace_to_goal(self, state):
        """
        Compute the distance from the current state to the goal state.
        """
        return self.distance(state, self.goal_state)

    def get_neighbors(self, current_state):
        row, col = current_state
        neighbors = []
        if row > 0 and self.rooms[row-1][col] != "#":
            neighbors.append((row-1, col))
        if row < len(self.rooms)-1 and self.rooms[row+1][col] != "#":
            neighbors.append((row+1, col))
        if col > 0 and self.rooms[row][col-1] != "#":
            neighbors.append((row, col-1))
        if col < len(self.rooms[0])-1 and self.rooms[row][col+1] != "#":
            neighbors.append((row, col+1))
        return neighbors
    
    def _get_intended_cell(self, current_state, action):
        # Compute candidate cell based on the action.
        if self.ACTION_NAMES[action] == "UP":
            candidate = (current_state[0] - 1, current_state[1])
        elif self.ACTION_NAMES[action] == "DOWN":
            candidate = (current_state[0] + 1, current_state[1])
        elif self.ACTION_NAMES[action] == "LEFT":
            candidate = (current_state[0], current_state[1] - 1)
        elif self.ACTION_NAMES[action] == "RIGHT":
            candidate = (current_state[0], current_state[1] + 1)
        elif self.ACTION_NAMES[action] == "WAIT":
            candidate = current_state
        else:
            candidate = current_state

        # First, check boundaries.
        rows, cols = self.rooms.shape  # e.g., (5, 9)
        r, c = candidate
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return current_state

        # Check if the candidate cell is a wall.
        if self.rooms[r][c] == "#":
            return current_state

        # If the agent is carrying a block, consider blocks as obstacles.
        # (If the agent is not carrying a block, it can enter a cell with a block, which is useful for picking it up.)
        if any(block.carrying for block in self.blocks):
            if self._is_cell_occupied(candidate):
                return current_state

        return candidate
    
    def _get_storage_positions(self):
        """Define the storage positions in the grid based on the room layout."""
        storage_positions = []
        # Here we consider cells that are free in the grid (".")
        # Exclude the start state, exit states (delivery cells), and the door cell.
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                pos = (row, col)
                if self.rooms[pos] != '.':
                    continue
                if pos in self.exit_states:
                    continue
                if pos == self.door_cell:
                    continue
                if pos == self.pickup_cell:
                    continue
                storage_positions.append(pos)
        
        return storage_positions

    def _is_cell_occupied(self, position):
        """
        Returns True if the cell at 'position' is occupied by any block 
        that is not being carried and is not yet delivered.
        """
        for block in self.blocks:
            # A block that is being carried should not block movement.
            if not block.delivered and (not block.carrying) and block.position == position:
                return True
        return False
    # Get the available storage positions for the blocks
    def get_available_storage_positions(self):
        # Assuming self.storage_positions is a list of all storage positions
        occupied_positions = {block.storage_location for block in self.blocks if block.storage_location}
        available_positions = [pos for pos in self.storage_positions if pos not in occupied_positions]
        #print(f"Available storage positions from the environment: {available_positions}")
        return available_positions
    
    def compute_goal_distance(self, block_pos, goal_state):
        # If goal_state is a 2-tuple of ints, treat it as a single exit cell.
        if isinstance(goal_state, (list, tuple)):
            # Check if it represents a single coordinate:
            if len(goal_state) == 2 and isinstance(goal_state[0], int) and isinstance(goal_state[1], int):
                return self.manhattan_distance(block_pos, goal_state)
            else:
                # Otherwise, assume it’s an iterable of exit cells.
                return min(self.manhattan_distance(block_pos, exit_cell) for exit_cell in goal_state)
        else:
            # Fallback in case goal_state is already a coordinate pair.
            return self.manhattan_distance(block_pos, goal_state)

    def _delivery_reward(self, storage_loc, error_time):
        """
        Compute reward for delivering a block to storage.

        - Guaranteed base reward of 10 for any delivery
        - Sliding bonus of up to 30:
            • Max bonus for on-time delivery (error_time == 0)
            • Bonus drops to 0 as abs(error_time) → max_allowable_error
        - Early deliveries (error_time < 0) are still rewarded, but with reduced bonus
        """
        base_reward = self.DELIVERY_BASE_REWARD
        max_bonus = self.DELIVERY_MAX_TIMING_BONUS
        max_allowable_error = self.DELIVERY_TARGET_WINDOW

        try:
            path_len = min(self.manhattan_distance(storage_loc, ec)
                        for ec in self.exit_cells if ec != storage_loc)
        except:
            path_len = 0

        # Optional: distance bonus if you want to still include it separately
        max_len = self.grid_rows + self.grid_cols
        distance_bonus = 10.0 * (1 - path_len / max_len)

        # Clip error_time to avoid absurd penalties or bonuses
        bonus = max(0.0, max_bonus * (1 - abs(error_time) / max_allowable_error))
        #bonus = max_bonus

        return base_reward + bonus #+ distance_bonus

    def update_pickup_queue(self):
        """
        Queue logic with stochastic arrivals.

        1) Any block whose arrival_step has been reached and is still outside
        (position is None) is moved to the waiting cell.
        2) If pickup cell is empty, move exactly one waiting block from the
        door cell to the pickup cell.
        """

        # 1) release newly arrived blocks into the waiting queue
        for block in self.blocks:
            if (
                block.position is None
                and not block.carrying
                and not block.stored
                and not block.delivered
                and hasattr(block, "arrival_step")
                and block.arrival_step <= self.time_steps
            ):
                block.position = self.waiting_cell

        # 2) if pickup cell is occupied, do nothing
        pickup_occupied = any(
            (block.position == self.pickup_cell) and (not block.carrying) and (not block.delivered)
            for block in self.blocks
        )
        if pickup_occupied:
            return

        # 3) move one waiting block into pickup
        waiting_blocks = [
            block for block in self.blocks
            if block.position == self.waiting_cell
            and not block.carrying
            and not block.delivered
        ]

        if not waiting_blocks:
            return

        # FIFO by arrival step
        waiting_blocks.sort(key=lambda b: getattr(b, "arrival_step", 0))
        waiting_blocks[0].position = self.pickup_cell

    def block_moves(self, action):
        # Optional: additional block dynamics.
        pass

    def compute_block_status(self, block):
        """
        Status:
        [1,0,0,0,0] not yet arrived
        [0,1,0,0,0] available / uncollected
        [0,0,1,0,0] carrying
        [0,0,0,1,0] stored
        [0,0,0,0,1] delivered
        """
        if block.delivered:
            return (0, 0, 0, 0, 1)
        elif block.stored:
            return (0, 0, 0, 1, 0)
        elif block.carrying:
            return (0, 0, 1, 0, 0)
        elif block.position is None:
            return (1, 0, 0, 0, 0)
        else:
            return (0, 1, 0, 0, 0)

    def manhattan_distance(self, p1, p2):
        """
        Returns the Manhattan distance between two points p1 and p2.
        """
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def signed_remaining_storage_time(self, block):
        """Return reward-consistent signed steps until a stored deadline.

        Delivery error is defined against
        ``stored_time_step + storage_steps_needed``.  Deriving remaining time
        from that same deadline avoids the one-step offset in the auxiliary
        ``storage_steps_elapsed`` counter and preserves negative overdue time.
        ``None`` means that the storage clock has not started.
        """

        if block.stored_time_step is None or not block.stored:
            return None
        deadline = block.stored_time_step + block.storage_steps_needed
        return float(deadline - self.time_steps)

    def get_current_state(self):
        """
        Returns the current state as a tuple:
            (agent_pos, blocks_info, t_next, n_wait)

        - agent_pos: tuple (row, col) of the agent.
        - blocks_info: tuple of feature‐tuples for each block.
        - t_next: steps until the next block arrival (–1 if none remain), clipped to self.MAX_T.
        - n_wait: number of blocks still waiting at the door (not yet picked up).
        """
        # 1) agent position
        agent_pos = self.current_state

        # 2) rich features per block
        blocks_info = tuple(
            self.compute_block_features(block, agent_pos, self.goal_state)
            for block in self.blocks
        )

        # 3) time until next arrival
        future_arrivals = [
            blk.arrival_step
            for blk in self.blocks
            if hasattr(blk, "arrival_step") and blk.arrival_step > self.time_steps
        ]
        if future_arrivals:
            t_next = min(future_arrivals) - self.time_steps
            t_next = min(t_next, self.MAX_T)
        else:
            t_next = -1

        # 4) number of blocks waiting at the door cell
        n_wait = sum(
            1
            for blk in self.blocks
            if blk.position == self.waiting_cell
            and not blk.carrying
            and not blk.delivered
        )

        return (agent_pos, blocks_info)


    def compute_block_features(self, block, agent_pos, goal_state):
        """
        Computes all features for a block, including real-time pathing information.
        
        Returns a tuple containing:
            - status: One-hot tuple of the block's state.
            - block_position: The (row, col) of the block.
            - dist_agent_block: Manhattan distance from agent to block.
            - dist_block_storage: Manhattan distance from block to its storage location.
            - dist_block_goal: Manhattan distance from block to the delivery area.
            - remaining_time: Time left for storage; -1 if not applicable.
            - is_path_clear: (NEW) Binary flag, 1 if a path to the block exists, 0 otherwise.
            - path_length: (NEW) The length of the path to the block; -1 if no path.
        """
        #status = self.compute_block_status(block)
        #block_pos = block.position
        #dist_agent_block = self.manhattan_distance(agent_pos, block_pos)

        status = self.compute_block_status(block)
        block_pos = block.position

        if block_pos is None:
            return (
                status,
                (-1, -1),   # block_position placeholder
                -1,         # dist_agent_block
                -1,         # dist_block_storage
                -1,         # dist_block_goal
                -1,         # remaining_time
                0,          # is_path_clear
                -1,         # path_length
            )

        dist_agent_block = self.manhattan_distance(agent_pos, block_pos)
        
        # Compute distance from block to its storage location (if available)
        if hasattr(block, 'storage_location') and block.storage_location is not None:
            dist_block_storage = self.manhattan_distance(block_pos, block.storage_location)
        else:
            dist_block_storage = -1  # flag value
        
        # Compute distance from block to the delivery area (exit)
        dist_block_goal = self.compute_goal_distance(block_pos, goal_state)
        
        # Compute remaining storage time if the block is stored.
        if block.stored and hasattr(block, 'stored_time_step') and hasattr(block, 'storage_steps_needed'):
            remaining_time = max(0, block.storage_steps_needed - (self.time_steps - block.stored_time_step))
        else:
            remaining_time = -1  # Indicate that remaining time is not applicable.

        # --- NEW: REAL-TIME PATHING INFORMATION ---
        is_path_clear = 0  # Default to 0 (path is not clear)
        path_length = -1     # Default to -1 (no path exists)

        # We only need to check the path if the block is a potential target (not yet carried or delivered)
        if not block.carrying and not block.delivered:
            target_pos = block.position
            
            # Other non-delivered blocks are potential obstacles
            blocked_cells = {
                b.position for b in self.blocks
                if b is not block and not b.delivered
            }

            # Use your existing A* helper function to find a path
            path = self._astar(self.rooms, agent_pos, target_pos, blocked_cells)

            if path:  # If _astar returns a valid path list
                is_path_clear = 1
                path_length = len(path) - 1  # Length is the number of steps to take

        # --- END OF NEW SECTION ---
        
        return (status, block_pos, dist_agent_block, dist_block_storage, dist_block_goal, remaining_time, is_path_clear, path_length)

    def discretize_state(self):
        state = self.get_current_state()
        agent_pos, blocks_info = state
        agent_row, agent_col = agent_pos

        MAX_T = 15
        num_cols = self.rooms.shape[1]
        agent_idx = agent_row * num_cols + agent_col

        discrete_blocks = []
        for (
            status,
            block_pos,
            dist_agent_block,
            dist_block_storage,
            dist_block_goal,
            remaining_time,
            is_path_clear,
            path_length,
        ) in blocks_info:

            block_idx = -1 if block_pos == (-1, -1) else block_pos[0] * num_cols + block_pos[1]
            status_int = status.index(1) if 1 in status else -1

            def bin_distance(d):
                if d < 0:
                    return -1
                if d <= 2:
                    return 0
                if d <= 4:
                    return 1
                if d <= 6:
                    return 2
                return 3

            bin_agent_block = bin_distance(dist_agent_block)
            bin_block_storage = bin_distance(dist_block_storage)
            bin_block_goal = bin_distance(dist_block_goal)

            if remaining_time < 0:
                timer_feature = -1
            else:
                timer_feature = min(remaining_time, MAX_T)

            discrete_blocks.append((
                block_idx,
                status_int,
                bin_agent_block,
                bin_block_storage,
                bin_block_goal,
                timer_feature,
            ))

        return (agent_idx, tuple(discrete_blocks))

    def get_action_space(self):
        return [0, 1, 2, 3, 4, 5, 6]

    def get_available_actions(self, state):
        """Return only primitives that are physically meaningful right now.

        The temporal-mode controller treats this set as :math:`A(s)`.  Keeping
        impossible pickup/putdown commands or wall/occupied moves in that set
        gives a regularized policy positive probability of executing illegal
        controls and also contaminates its Bellman aggregation.
        """

        if state is None:
            agent_position = self.current_state
        elif (
            isinstance(state, tuple)
            and len(state) == 2
            and all(isinstance(value, (int, np.integer)) for value in state)
        ):
            agent_position = tuple(int(value) for value in state)
        else:
            agent_position = state[0]

        available = []
        for name in ("UP", "DOWN", "LEFT", "RIGHT"):
            action = self.ACTION_IDS[name]
            if self._get_intended_cell(agent_position, action) != agent_position:
                available.append(action)

        carried = next((block for block in self.blocks if block.carrying), None)
        if carried is None and any(
            block.position == agent_position
            and not block.delivered
            and not block.carrying
            for block in self.blocks
        ):
            available.append(self.ACTION_IDS["PICKUP"])

        if carried is not None:
            valid_inbound_drop = (
                not carried.stored
                and carried.storage_location is not None
                and agent_position == carried.storage_location
            )
            valid_delivery = carried.stored and agent_position in self.exit_cells
            valid_relocation = (
                carried.stored
                and carried.storage_location is not None
                and agent_position == carried.storage_location
            )
            if valid_inbound_drop or valid_delivery or valid_relocation:
                available.append(self.ACTION_IDS["PUTDOWN"])

        available.append(self.ACTION_IDS["WAIT"])
        return available

    def is_state_terminal(self, state):
        return all(block.delivered for block in self.blocks)

    def get_initial_states(self):
        return deepcopy([self.start_state])

    def get_successors(self, state=None, actions=None):
        if state is None:
            state = self.current_state
        if actions is None:
            actions = self.get_available_actions(state)
        successors = []
        for action in actions:
            s = self._get_intended_cell(state, action)
            if s not in successors:
                successors.append(s)
        return successors

    def render(self, mode="human"):
        render_rooms = self.rooms.copy()
        render_rooms[self.current_state] = "A"
        for block in self.blocks:
            if block.position is None:
                continue
            if not block.delivered:
                render_rooms[block.position] = block.label[0]
            else:
                render_rooms[block.position] = block.label[0].lower()
        for row in render_rooms:
            formatted_row = "  ".join("{:^3}".format(cell) for cell in row)
            print(formatted_row)
    print()

    def close(self):
        pass

    def get_state_space(self):
        return [(i, j) for i in range(self.rooms.shape[0]) for j in range(self.rooms.shape[1])]
    
    def _manhattan(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _astar(self, rooms, start: tuple[int, int], goal: tuple[int, int], blocked: set[tuple[int, int]]):
        """Classic A* that treats cells in *blocked* as obstacles."""
        open_set: list[tuple[int, tuple[int, int]]] = []
        heapq.heappush(open_set, (0, start))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g: dict[tuple[int, int], int] = {start: 0}
        f: dict[tuple[int, int], int] = {start: self._manhattan(start, goal)}
        rows, cols = rooms.shape

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                # reconstruct
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))

            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = current[0] + dr, current[1] + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                if rooms[nr, nc] == '#':
                    continue
                neigh = (nr, nc)
                if neigh in blocked and neigh != goal:
                    continue
                tentative = g[current] + 1
                if tentative < g.get(neigh, 1e9):
                    came_from[neigh] = current
                    g[neigh] = tentative
                    f[neigh] = tentative + self._manhattan(neigh, goal)
                    heapq.heappush(open_set, (f[neigh], neigh))
        return []  # no path
    
    def run_plan_only(self, plan: List[Tuple[int, int]]) -> float:
        self.reset()
        total_reward = 0.0

        for b_idx, storage_pos in enumerate(plan):
            block = self.blocks[b_idx]
            block.storage_location = storage_pos

            for a in plan_store_block(block, self):
                _, r, done, _ = self.step(a)
                total_reward += r
                if done:
                    return total_reward

            for a in plan_retrieve_block(block, self):
                _, r, done, _ = self.step(a)
                total_reward += r
                if done:
                    return total_reward

        return total_reward

    def simulate_assignment(self, assignment: List[Tuple[int, int]]) -> float:
        """
        Evaluate a full storage assignment given directly as storage positions.

        assignment:
            A list of storage cells, one per block, e.g.
            [(r1, c1), (r2, c2), ...]

        Returns:
            Total reward obtained by storing and retrieving all blocks
            under this fixed assignment.
        """
        self.reset()
        total_reward = 0.0

        if len(assignment) != len(self.blocks):
            raise ValueError(
                f"Assignment length {len(assignment)} does not match number of blocks {len(self.blocks)}"
            )

        # optional validity check: unique positions only
        if len(set(assignment)) != len(assignment):
            raise ValueError("Assignment contains duplicate storage positions.")

        for pos in assignment:
            if pos not in self.storage_positions:
                raise ValueError(f"Invalid storage position in assignment: {pos}")

        for b_idx, pos in enumerate(assignment):
            block = self.blocks[b_idx]
            block.storage_location = pos

            # store
            for a in plan_store_block(block, self):
                _, r, done, _ = self.step(a)
                total_reward += r
                if done:
                    return total_reward

            # retrieve
            for a in plan_retrieve_block(block, self):
                _, r, done, _ = self.step(a)
                total_reward += r
                if done:
                    return total_reward

        return total_reward
                
    def update_grid(self):
        """
        No‐op placeholder to keep helper_functions.plan_store_block happy.
        (If you ever add dynamic obstacles, you can rebuild any internal
        occupancy maps here.)
        """
        pass

    # In your SmallRoomsEnv class

    def plan_path_heuristic(self,
                            start: Tuple[int,int],
                            goal:  Tuple[int,int],
                            ignore_block=None,
                            extra_blocked=()) -> List[int]:
        """
        Run A* from `start` to `goal`, treating non-delivered blocks as obstacles.
        Optionally, an `ignore_block` can be specified to be excluded from the
        obstacle set. ``extra_blocked`` carries structural obstacles from an
        external planning/certification contract, such as reserved queue cells.
        """
        # Build the obstacle set, excluding the block to ignore
        blocked = {b.position for b in self.blocks
                   if (not b.delivered) and (not b.carrying) and (b is not ignore_block)}
        blocked.update(tuple(cell) for cell in extra_blocked)
        blocked.discard(tuple(start))
        if tuple(goal) in blocked:
            return []
        
        path = self._astar(self.rooms, start, goal, blocked)
        # ... (the rest of the function for converting path to actions is unchanged) ...
        if not path:
            return []
        actions = []
        for (r0,c0), (r1,c1) in zip(path, path[1:]):
            dr, dc = r1-r0, c1-c0
            if   (dr,dc)==(-1,0): actions.append(self.ACTION_IDS["UP"])
            elif (dr,dc)==( 1,0): actions.append(self.ACTION_IDS["DOWN"])
            elif (dr,dc)==( 0,-1): actions.append(self.ACTION_IDS["LEFT"])
            elif (dr,dc)==( 0, 1): actions.append(self.ACTION_IDS["RIGHT"])
            else:
                raise RuntimeError(f"Bad step { (r0,c0) }→{ (r1,c1) }")
        return actions



    def _find_env_block_by_position(self, pos: Tuple[int,int]):
        """
        Given a (r,c) from the Yard, find the matching Blocks() in self.blocks.
        """
        for b in self.blocks:
            if b.position == pos:
                return b
        raise ValueError(f"No block at position {pos} in env.blocks")
 
