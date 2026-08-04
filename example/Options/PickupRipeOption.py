from option import BaseOption
from small_rooms_env import SmallRoomsEnv
from helper.tools import _astar, _action_between

class PickupRipeOption(BaseOption):
    """
    Navigates to a ripe, stored block and picks it up. This option acts as the
    bridge between a block finishing storage and the DeliverOption.
    """
    def __init__(self, env: SmallRoomsEnv, timeout: int = 25):
        super().__init__(is_primitive=False)
        self.env = env
        self.timeout = timeout
        self.reset_internal()

    def reset_internal(self):
        """Resets the internal state of the option."""
        self.target_block = None
        self.target_pos = None
        self.path = []
        self.steps_taken = 0
        self.failed = False

    def initiation(self, state):
        """
        Start this option iff at least one stored block is already (or will
        soon be) late *after we include the full travel time* needed to
        fetch and deliver it.
        """
        # ───────── reset internal ─────────
        self.reset_internal()
        agent_pos = state[0]

        # cannot start while carrying something
        if any(b.carrying for b in self.env.blocks):
            return False

        ripe_candidates = []          # (block, total_travel, slack) triples

        for b in self.env.blocks:
            # only consider stored–undelivered–not-carried blocks
            if not (b.stored and not b.delivered and not b.carrying):
                continue

            # ----- 1) timing information -----
            #elapsed = self.env.time_steps - b.arrival_step
            # elapsed time since the block was stored
            if b.stored_time_step is None:
                continue  # block not yet stored properly
            elapsed = self.env.time_steps - b.stored_time_step


            # time from block → closest exit
            # 1a) agent → block via A*
            path_ab = _astar(self.env.rooms, agent_pos, b.position,
                            {blk.position for blk in self.env.blocks
                            if blk is not b and not blk.delivered})
            if not path_ab:
                continue  # if you can’t even reach it, skip

            t_ab = len(path_ab) - 1

            # 1b) block → exit via A*
            # pick whichever exit gives the shortest A* path
            best = 1e9
            for ex in self.env.exit_cells:
                path_be = _astar(self.env.rooms, b.position, ex,
                                {blk.position for blk in self.env.blocks
                                if blk is not b and not blk.delivered})
                if path_be:
                    best = min(best, len(path_be)-1)
            t_be = best

            total_travel = t_ab + t_be

            # slack = how many steps we *can still wait* until the deadline
            slack = (b.storage_steps_needed - elapsed) - total_travel

            # consider the block “ripe” if slack ≤ 10   (i.e. we must leave *now*)
            # You can relax this inequality if you want earlier pick-ups.
            if slack <= 20:
                ripe_candidates.append((b, total_travel, slack))

        # nothing urgent → option not available
        if not ripe_candidates:
            return False

        # ───────── choose the *most urgent* block ─────────
        # tie-breaker: smaller total_travel if two have same slack
        self.target_block, _, _ = min(
            ripe_candidates,
            key=lambda tup: (tup[2], tup[1])     # (slack, travel)
        )
        self.target_pos = self.target_block.position

        # ───────── plan path ─────────
        blocked_cells = {
            blk.position for blk in self.env.blocks
            if blk is not self.target_block and not blk.delivered
        }
        path = _astar(self.env.rooms, agent_pos, self.target_pos, blocked_cells)

        if not path:            # no path → fail initiation
            self.failed = True
            return False

        self.path = path[1:]    # skip current cell
        return True


    def policy(self, state):
        self.steps_taken += 1
        agent_pos = state[0]

        if self.failed:
            return SmallRoomsEnv.ACTION_IDS['WAIT']

        # If we have arrived at the target block's location, PICKUP.
        if agent_pos == self.target_pos:
            return SmallRoomsEnv.ACTION_IDS['PICKUP']

        # If we have a path, follow it.
        if self.path:
            next_cell = self.path.pop(0)
            return _action_between(agent_pos, next_cell)

        # If path is empty but not at target, something went wrong.
        self.failed = True
        return SmallRoomsEnv.ACTION_IDS['WAIT']

    def termination(self, state):
        # Success condition: The target block is now being carried.
        if self.target_block and self.target_block.carrying:
            return True

        # Failure condition: Timeout or internal failure.
        if self.steps_taken >= self.timeout or self.failed:
            return True
            
        return False

    def __str__(self):
        return "PickupRipeOption"

    __repr__ = __str__

    def __hash__(self):
        return hash("PickupRipeOption")