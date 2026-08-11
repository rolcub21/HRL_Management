from option import BaseOption
from example.small_rooms_env import SmallRoomsEnv
from example.helper.tools import _astar, _action_between

class StoreOption(BaseOption):
    """
    Navigates to an already-carried block’s assigned storage location.
    """
    def __init__(self, env: SmallRoomsEnv, timeout: int = 15):
        super().__init__(is_primitive=False)
        self.env = env
        self.timeout = timeout
        self.reset_internal()

    def reset_internal(self):
        self.phase = None
        self.block = None
        self.path = []
        self.cursor = 0
        self.steps = 0
        self.planning_failed = False

    def initiation(self, state):
        self.reset_internal()
        # Only start when already carrying a block that needs storing
        carrying = next(
            (b for b in self.env.blocks
             if b.carrying and not b.stored and b.storage_location is not None),
            None
        )
        if carrying:
            occupied = {
                b.position for b in self.env.blocks
                if b is not carrying and not b.delivered
            }
            if carrying.storage_location in occupied:
                self.reset_internal()
                return False
            self.block = carrying
            self.phase = 'store'
            return True
        return False

    def policy(self, state):
        agent_pos = state[0]
        self.steps += 1

        # A zero-length route is a valid plan: store immediately.
        if self.phase == 'store' and agent_pos == self.block.storage_location:
            return SmallRoomsEnv.ACTION_IDS['PUTDOWN']

        # Plan if needed
        if (not self.path or self.cursor >= len(self.path)) and not self.planning_failed:
            if not self._plan_path(agent_pos):
                self.planning_failed = True

        if self.planning_failed:
            return SmallRoomsEnv.ACTION_IDS['WAIT']

        # Follow planned path
        if self.cursor < len(self.path):
            next_pos = self.path[self.cursor]
            self.cursor += 1
            return _action_between(agent_pos, next_pos)

        return SmallRoomsEnv.ACTION_IDS['WAIT']

    def termination(self, state):
        if self.block and self.block.stored:
            self.reset_internal()
            return True
        if self.planning_failed or self.steps >= self.timeout:
            self.reset_internal()
            return True
        if self.phase == 'store' and self.block:
            occupied = {b.position for b in self.env.blocks if b is not self.block}
            if self.block.storage_location in occupied:
                self.reset_internal()
                return True
        return False

    def _plan_path(self, agent_pos):
        start = agent_pos
        target = self.block.storage_location
        blocked = {
            b.position for b in self.env.blocks
            if b is not self.block and not b.delivered
        }
        raw = _astar(self.env.rooms, start, target, blocked)
        if not raw:
            self.path = []
            return False
        self.path = raw[1:]
        self.cursor = 0
        return True

    def __str__(self):
        return 'StoreOption'

    __repr__ = __str__

    def __hash__(self):
        return hash('StoreOption')
