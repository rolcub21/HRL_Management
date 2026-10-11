from typing import List, Tuple, Optional
import numpy as np

from GA.helper_functions import plan_store_block, plan_retrieve_block
from PSLAP.yard_logic import (
    Yard,
    find_min_obstruction_path,
    select_storage_location,
)


Cell = Tuple[int, int]


class PSLaPPolicy:
    """
    PSLAP-style heuristic policy for SmallRoomsEnv.

    Logic:
      1) If there is a due stored block, try to retrieve it.
         If blocked, relocate the obstructing blocks using the yard heuristic.
      2) Otherwise, if there is an inbound block at pickup_cell, choose a storage
         location using the yard heuristic and store it.
      3) Otherwise, WAIT.
    """

    def __init__(self, env):
        self.env = env

    # ------------------------------------------------------------------
    # Yard conversion helpers
    # ------------------------------------------------------------------
    def _build_yard(self) -> Yard:
        """
        Convert the current SmallRoomsEnv state into a Yard object.
        We use South-open exits because your environment delivers through
        bottom exit cells.
        """
        yard = Yard(self.env.grid_rows, self.env.grid_cols, open_sides=["S"])

        for b in self.env.blocks:
            if not b.delivered and b.position is not None and b.position in self.env.storage_positions:
                r, c = b.position
                yard.place_block(b, r, c)

        return yard

    def _all_yard_blocks(self, yard: Yard) -> List:
        blocks = []
        for row in yard.grid:
            for b in row:
                if b is not None:
                    blocks.append(b)
        return blocks

    def _find_env_block(self, label: str):
        for b in self.env.blocks:
            if b.label == label:
                return b
        return None

    # ------------------------------------------------------------------
    # Relocation planning
    # ------------------------------------------------------------------
    def _plan_relocation_actions(self, block, new_loc: Cell) -> List[int]:
        """
        Relocate a stored obstructing block to a new storage location.
        """
        block.stored = False
        block.storage_location = new_loc
        return plan_store_block(block, self.env)

    # ------------------------------------------------------------------
    # Main decision generator
    # ------------------------------------------------------------------
    def step(self):
        """
        Generator of primitive actions.
        """
        # ==============================================================
        # 1) RETRIEVE DUE BLOCKS
        # ==============================================================
        due_blocks = [
            b for b in self.env.blocks
            if b.stored and not b.delivered and b.is_storage_time_elapsed()
        ]

        if due_blocks:
            due = min(
                due_blocks,
                key=lambda b: b.get_remaining_storage_time()
                if hasattr(b, "get_remaining_storage_time")
                else (b.stored_time_step + b.storage_steps_needed)
            )

            #print(f"[PSLaP] due detected: {due.label} at {due.position}")

            yard = self._build_yard()
            yard_due = None
            for b in self._all_yard_blocks(yard):
                if b.label == due.label:
                    yard_due = b
                    break

            if yard_due is None:
                #print(f"[PSLaP] due block {due.label} not found in yard view -> direct retrieve")
                actions = plan_retrieve_block(due, self.env)
                #print(f"[PSLaP] retrieve plan length={len(actions)} actions={actions[:20]}")
                if not actions:
                    #print("[PSLaP] empty retrieve plan -> WAIT")
                    yield self.env.ACTION_IDS["WAIT"]
                    return
                yield from actions
                return

            blocks_due_now = [yard_due]
            path, exit_cell, obstructions = find_min_obstruction_path(
                yard, yard_due, blocks_due_now
            )

            #print(f"[PSLaP] path found={bool(path)} exit={exit_cell} obstructions={len(obstructions)}")

            # If path exists and has no obstructions, retrieve directly
            if path and not obstructions:
                actions = plan_retrieve_block(due, self.env)
                #print(f"[PSLaP] direct retrieve plan length={len(actions)} actions={actions[:20]}")
                if not actions:
                    #print("[PSLaP] empty direct retrieve plan -> WAIT")
                    yield self.env.ACTION_IDS["WAIT"]
                    return
                yield from actions
                return

            # If blocked, relocate each obstruction in sequence
            if obstructions:
                for obs in obstructions:
                    obs_env = self._find_env_block(obs.label)
                    if obs_env is None:
                        #print(f"[PSLaP] obstruction {obs.label} not found in env")
                        continue

                    #print(f"[PSLaP] relocating obstruction {obs.label} from {obs_env.position}")

                    # rebuild yard after every relocation decision
                    yard = self._build_yard()
                    all_blocks = self._all_yard_blocks(yard)

                    yard_obs = None
                    for b in all_blocks:
                        if b.label == obs_env.label:
                            yard_obs = b
                            break

                    if yard_obs is None:
                        #print(f"[PSLaP] obstruction {obs.label} not found in yard view")
                        continue

                    yard.remove_block(yard_obs)

                    new_loc = select_storage_location(yard, yard_obs, all_blocks)
                    #print(f"[PSLaP] relocation target for {obs.label} = {new_loc}")

                    if new_loc is None:
                        #print("[PSLaP] no relocation target -> WAIT")
                        yield self.env.ACTION_IDS["WAIT"]
                        return

                    actions = self._plan_relocation_actions(obs_env, new_loc)
                    #print(f"[PSLaP] relocation plan length={len(actions)} actions={actions[:20]}")

                    if not actions:
                        #print("[PSLaP] empty relocation plan -> WAIT")
                        yield self.env.ACTION_IDS["WAIT"]
                        return

                    yield from actions

                actions = plan_retrieve_block(due, self.env)
                #print(f"[PSLaP] post-relocation retrieve length={len(actions)} actions={actions[:20]}")
                if not actions:
                    #print("[PSLaP] empty post-relocation retrieve plan -> WAIT")
                    yield self.env.ACTION_IDS["WAIT"]
                    return
                yield from actions
                return

            # No feasible path
            #print("[PSLaP] no feasible retrieval path -> WAIT")
            yield self.env.ACTION_IDS["WAIT"]
            return

        # ==============================================================
        # 2) STORE INBOUND BLOCK
        # ==============================================================
        inbound = [
            b for b in self.env.blocks
            if b.position == self.env.pickup_cell
            and not b.stored
            and not b.delivered
            and not b.carrying
        ]

        if inbound:
            blk = inbound[0]
            #print(f"[PSLaP] inbound detected: {blk.label} at {blk.position}")

            yard = self._build_yard()
            all_blocks = self._all_yard_blocks(yard)

            class TempBlock:
                pass

            temp_block = TempBlock()
            temp_block.label = blk.label
            temp_block.position = None
            temp_block.stored = False

            if hasattr(blk, "get_remaining_storage_time"):
                temp_block.get_remaining_storage_time = blk.get_remaining_storage_time
            else:
                temp_block.get_remaining_storage_time = lambda: float("inf")

            target = select_storage_location(yard, temp_block, all_blocks)
            #print(f"[PSLaP] chosen storage target for {blk.label}: {target}")

            if target is None:
                #print("[PSLaP] no storage target -> WAIT")
                yield self.env.ACTION_IDS["WAIT"]
                return

            blk.storage_location = target
            actions = plan_store_block(blk, self.env)
            #print(f"[PSLaP] store plan length={len(actions)} actions={actions[:20]}")

            if not actions:
                #print("[PSLaP] empty store plan -> WAIT")
                yield self.env.ACTION_IDS["WAIT"]
                return

            yield from actions
            return

        # ==============================================================
        # 3) NOTHING TO DO
        # ==============================================================
        #print("[PSLaP] fallback WAIT")
        yield self.env.ACTION_IDS["WAIT"]


def run_pslap_episode(env, max_steps: int = 1700):
    """
    Execute one episode with the PSLAP heuristic.

    Returns:
        total_reward, steps, avg_delivery_error, success
    """
    total_reward = 0.0
    steps = 0
    delivery_errors = []

    env.reset()
    policy = PSLaPPolicy(env)
    gen = policy.step()

    while steps < max_steps and not env.is_state_terminal(env.current_state):
        try:
            act = next(gen)
        except StopIteration:
            gen = policy.step()
            act = next(gen)

        _, reward, done, info = env.step(act)
        total_reward += reward
        steps += 1

        if "delivery_error_time" in info:
            delivery_errors.append(info["delivery_error_time"])

        if done:
            break

    avg_err = float(np.mean(delivery_errors)) if delivery_errors else np.nan
    success = 1.0 if env.is_state_terminal(env.current_state) else 0.0

    #print(
        ##f"[PSLaP] Episode finished after {steps} steps | "
        #f"return={total_reward:.2f} | avg_err={avg_err:.2f} | success={success:.0f}"
    #)

    return total_reward, steps, avg_err, success