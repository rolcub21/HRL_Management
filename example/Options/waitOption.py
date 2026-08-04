from option import BaseOption
from small_rooms_env import SmallRoomsEnv

class WaitOption(BaseOption):
    """A no-op option: execute WAIT exactly once, then terminate."""
    def __init__(self, env: SmallRoomsEnv):
        super().__init__(is_primitive=False)
        self.env = env

    def initiation(self, state):
        # Always allowed
        return True

    def policy(self, state):
        # Do nothing but WAIT
        return SmallRoomsEnv.ACTION_IDS["WAIT"]

    def termination(self, state):
        # Always terminate immediately after one step
        return True
    
    def intrinsic_reward(self, state, action, next_state, info):
        return 0.0

    def __str__(self):
        return "WaitOption"
    __repr__ = __str__
    def __hash__(self):
        return hash("WaitOption")
