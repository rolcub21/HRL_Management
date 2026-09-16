# sub_algorithms.py

from helper_functions import plan_store_block, plan_retrieve_block
from GA.others.yard_logic import (
    find_min_obstruction_path, select_storage_location
)
from example.block_instance import Blocks

class SubAlgorithmBase:
    """
    A generic base class for a yard-management sub-algorithm.
    Subclasses implement 'produce_actions' given (state, env).
    """
    def produce_actions(self, state, env):
        raise NotImplementedError("Override produce_actions in subclasses")


class StoreInboundBlock(SubAlgorithmBase):
    """
    Closer to the ABSLAP inbound storage logic:
      - Identify inbound block
      - Use 'select_storage_location' to pick a cell
      - Return the plan of discrete actions to store the block
    """
    def produce_actions(self, state, env):
        # 1) Identify an inbound block
        inbound_block = None
        for blk in env.blocks:
            if blk.position == (1,5) and not blk.delivered and not blk.stored:
                inbound_block = blk
                break
        if inbound_block is None:
            return [], None

        # 2) Convert environment into a 'Yard', pick a storage cell
        yard = env._sync_env_to_yard()
        r, c = inbound_block.position
        yard_blk = yard.grid[r][c]
        if yard_blk is None:
            return [], None

        # Gather all yard blocks
        all_yard_blocks = []
        for row in yard.grid:
            for b in row:
                if b is not None:
                    all_yard_blocks.append(b)

        # Use yard_logic to pick a good storage location
        storage_cell = select_storage_location(yard, yard_blk, all_yard_blocks)
        if storage_cell is None:
            print(f"No feasible storage location for inbound {inbound_block.label}.")
            return [], None

        # Record the chosen location on the environment block
        inbound_block.storage_location = storage_cell
        #print(f"Inbound block {inbound_block.label} will be stored at {storage_cell}.")

        # 3) Generate a discrete plan of actions to store the block
        actions = plan_store_block(inbound_block, env)
        return actions, storage_cell, inbound_block.label




class RetrieveOutboundBlock(SubAlgorithmBase):
    """
    Closer to ABSLAP retrieval logic:
      - Identify a stored block that's done (is_storage_time_elapsed)
      - Use find_min_obstruction_path to find a path
      - If obstructions exist, relocate them with sub-plans
      - Then produce actions to pick up the block and deliver
    """

    def produce_actions(self, state, env):
        # Step 1) Find a block that is stored & not delivered & time is up
        candidate = None
        for blk in env.blocks:
            if blk.stored and not blk.delivered and blk.is_storage_time_elapsed():
                candidate = blk
                break
        if candidate is None:
            return []

        # Step 2) Use yard logic: build yard, find path with minimal obstructions
        yard = env._sync_env_to_yard()
        r, c = candidate.position
        yard_blk = yard.grid[r][c]
        if yard_blk is None:
            return []

        # In ABSLAP, "blocks_due_now" might just be [yard_blk]
        blocks_due_now = [yard_blk]
        path, exit_cell, obstructions = find_min_obstruction_path(yard, yard_blk, blocks_due_now)

        if not path:
            print("No path => retrieval canceled")
            return []

        # Step 3) For each obstruction, we attempt to relocate
        #    We'll produce sub-plans for each obstruction
        action_list = []
        for obs in obstructions:
            # find env block
            obs_env_block = env._find_env_block_by_position(obs.location)
            if not obs_env_block:
                continue

            # remove from yard so we can find a new location
            yard.remove_block(obs)
            all_yard_blocks = []
            for row in yard.grid:
                for b in row:
                    if b is not None:
                        all_yard_blocks.append(b)

            new_loc = select_storage_location(yard, obs, all_yard_blocks)
            if not new_loc:
                print("No relocation => retrieval canceled for", candidate.label)
                return []

            # produce sub-plan to relocate obs_env_block from current pos to new_loc
            relocate_actions = env._plan_relocate_block(obs_env_block, new_loc)
            action_list.extend(relocate_actions)

            # place in yard
            yard.place_block(obs, new_loc[0], new_loc[1])
            obs.location = new_loc

        # Step 4) Now produce the normal plan to pick up candidate and deliver it
        deliver_actions = plan_retrieve_block(candidate, env)
        action_list.extend(deliver_actions)

        return action_list, candidate.label
