from option import BaseOption
from example.small_rooms_env import SmallRoomsEnv
from example.helper.tools import _astar, _action_between

class PickupOption(BaseOption):
    """Navigate to a free block at the pickup cell via A* and execute PICKUP exactly once."""

    def __init__(self, env: SmallRoomsEnv, timeout: int = 15):
        super().__init__(is_primitive=False)
        self.env = env
        self.timeout = timeout
        self.reset_internal()

    def reset_internal(self):
        self.block = None       # block to pick up
        self.path = []          # planned A* path (list of cells)
        self.steps = 0          # step counter

    def initiation(self, state):
        self.reset_internal()
        agent_pos = state[0]

        # 1) don’t start if already carrying
        if any(b.carrying for b in self.env.blocks):
            return False

        pickup_cell = self.env.pickup_cell

        # 2) prefer a block currently at pickup cell
        candidate = next(
            (
                b for b in self.env.blocks
                if b.position == pickup_cell
                and b.position is not None
                and not b.stored
                and not b.delivered
            ),
            None
        )

        if candidate is None:
            return False

        # 4) plan A* path to that block
        blocked = {
            b.position for b in self.env.blocks
            if b is not candidate
            and b.position is not None
            and not b.delivered
        }

        full_path = _astar(self.env.rooms, agent_pos, candidate.position, blocked)
        if not full_path:
            return False

        self.block = candidate
        self.path = full_path[1:]
        return True


    def policy(self, state):
        self.steps += 1
        agent_pos = state[0]

        if self.block is None or self.block.position is None:
            self.steps = self.timeout
            return SmallRoomsEnv.ACTION_IDS["WAIT"]

        if not self.block.carrying and agent_pos == self.block.position:
            return SmallRoomsEnv.ACTION_IDS["PICKUP"]

        blocked = {
            b.position for b in self.env.blocks
            if b is not self.block and b.position is not None
        }

        if not self.path or self.path[0] in blocked:
            new_path = _astar(self.env.rooms, agent_pos, self.block.position, blocked)
            if not new_path:
                self.steps = self.timeout
                return SmallRoomsEnv.ACTION_IDS["WAIT"]
            self.path = new_path[1:]

        next_cell = self.path.pop(0)
        return _action_between(agent_pos, next_cell)

    def termination(self, state):
        # succeed once carrying, or when timeout expires
        carrying = self.block is not None and self.block.carrying
        done = carrying or (self.steps >= self.timeout)
        if done:
            self.reset_internal()
        return done

    def __str__(self):
        return "PickupOption"
    __repr__ = __str__

    def __hash__(self):
        return hash("PickupOption")
