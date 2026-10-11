import random
from option import BaseOption
from example.small_rooms_env import SmallRoomsEnv
from example.helper.tools import _astar, _action_between

class DeliverOption(BaseOption):
    """
    A simplified option that ONLY delivers a block that is already being carried
    and is ripe for delivery. It does not handle picking up blocks from the world.
    """
    def __init__(self, env: SmallRoomsEnv, timeout: int = 20):
        super().__init__(is_primitive=False)
        self.env = env
        self.timeout = timeout
        self.reset_internal()

    def reset_internal(self):
        """Resets the internal state of the option."""
        self.block_to_deliver = None
        self.path = []
        self.steps_taken = 0
        self.planning_failed = False

    def initiation(self, state):
        agent_pos = state[0]
        
        # This option can ONLY start if the agent is already carrying a ripe block.
        carrying_ripe_block = next(
            (b for b in self.env.blocks
             if b.carrying and b.stored and not b.delivered),
            None
        )

        if not carrying_ripe_block:
            # If not carrying a ripe block, this option cannot be initiated.
            return False

        # If we are carrying a ripe block, plan the path to the nearest exit.
        self.reset_internal()
        self.block_to_deliver = carrying_ripe_block
        
        if self._plan_path(agent_pos):
            return True  # Initiation is successful if a path can be found.
        else:
            # If no path to an exit can be found, fail the initiation.
            self.planning_failed = True
            return False

    def policy(self, state):
        agent_pos = state[0]
        self.steps_taken += 1

        if self.planning_failed:
            return SmallRoomsEnv.ACTION_IDS['WAIT']

        # If we have arrived at an exit, the policy is to put down the block.
        if agent_pos in self.env.exit_cells:
            return SmallRoomsEnv.ACTION_IDS['PUTDOWN']

        # If we have a path, follow it.
        if self.path:
            next_pos = self.path.pop(0) # Get and remove the next step
            return _action_between(agent_pos, next_pos)

        # Dynamic occupancy may invalidate a previous route. Replan rather
        # than waiting until timeout while another exit remains reachable.
        if self._plan_path(agent_pos) and self.path:
            next_pos = self.path.pop(0)
            return _action_between(agent_pos, next_pos)

        self.planning_failed = True
        return SmallRoomsEnv.ACTION_IDS['WAIT']

    def termination(self, state):
        # Success condition: The block has been delivered.
        if self.block_to_deliver and self.block_to_deliver.delivered:
            self.reset_internal()
            return True

        # Failure condition: We have run out of time or planning failed.
        if self.steps_taken >= self.timeout or self.planning_failed:
            self.reset_internal()
            return True
            
        return False

    def _plan_path(self, agent_pos):
        """Plans a path to the nearest exit cell."""
        start = agent_pos
        
        # Blocked cells are other non-delivered blocks.
        blocked = {
            b.position for b in self.env.blocks
            if b is not self.block_to_deliver and not b.delivered
        }
        
        candidates = []
        for target in self.env.exit_cells:
            if target in blocked:
                continue
            raw_path = _astar(self.env.rooms, start, target, blocked)
            if raw_path:
                candidates.append(raw_path)

        if not candidates:
            self.path = []
            return False  # Planning failed

        raw_path = min(candidates, key=len)
        self.path = raw_path[1:]  # Exclude the agent's starting position
        return True

    def __str__(self): return 'DeliverOption'
    __repr__ = __str__
    def __hash__(self): return hash('DeliverOption')
