def plan_store_block(block, env):
    actions = []
    env.update_grid()  # Ensure the grid reflects current obstacles.
    
    start_pos = env.current_state
    block_pos = block.position
    target_pos = block.storage_location

    if not target_pos:
        return actions

    # 1) Move from env.current_state to block.position using A*:
    path_to_block = env.plan_path_heuristic(start_pos, block_pos)
    #print(f"Path to block: {path_to_block}")
    actions.extend(path_to_block)

    # 2) PICKUP
    actions.append(env.ACTION_IDS["PICKUP"])

    env.update_grid()  # Update grid again, if necessary.

    # 3) Move from block.position to target storage location using A*:
    path_to_storage = env.plan_path_heuristic(block_pos, target_pos)
    #print(f"Path to storage: {path_to_storage}")
    actions.extend(path_to_storage)

    # 4) PUTDOWN
    actions.append(env.ACTION_IDS["PUTDOWN"])

    # Do not update block.position here if you want the PUTDOWN action handler to do it.
    return actions




def plan_retrieve_block(block, env):
    """
    Return a sequence of actions for retrieving a block:
      1) Move from env.current_state to block.position
      2) Pick up the block
      3) Move from block.position to an exit cell
      4) Put down the block

    This returns a list of action IDs.
    """
    actions = []
    if block is None:
        return actions

    # 1) Path from env.current_state to block.position
    forklift_pos = env.current_state
    block_pos = block.position
    if block_pos is None:
        return actions

    path_to_block = env.plan_path_heuristic(forklift_pos, block_pos)
    actions.extend(path_to_block)

    # 2) PICKUP the block
    actions.append(env.ACTION_IDS["PICKUP"])

    # 3) Move from block.position to an exit cell.
    if env.exit_states:
        # Choose the first exit (or pick the closest exit)
        exit_cell = env.exit_states[0]
        path_to_exit = env.plan_path_heuristic(block_pos, exit_cell)
        actions.extend(path_to_exit)

        # 4) PUTDOWN the block
        actions.append(env.ACTION_IDS["PUTDOWN"])

    return actions
