import pickle
from option import BaseOption
from example.small_rooms_env import SmallRoomsEnv


class GAStorageSelectOption(BaseOption):
    is_storage_selector = True
    def __init__(self, env: SmallRoomsEnv, assignment_path: str):
        super().__init__(is_primitive=False)
        self.env = env
        self.assignment_path = assignment_path
        self.assignment = [tuple(position) for position in self._load_assignment()]

        if len(self.assignment) != len(self.env.blocks):
            raise ValueError(
                f"GA assignment length {len(self.assignment)} does not match "
                f"number of env blocks {len(self.env.blocks)}"
            )

        for pos in self.assignment:
            if pos not in self.env.storage_positions:
                raise ValueError(f"Invalid GA storage position: {pos}")
        if len(set(self.assignment)) != len(self.assignment):
            raise ValueError("GA assignment contains duplicate storage positions")

        self.block = None

        # compatibility with options_agent.py
        self.pending = {}
        self.gamma = 0.99

    def _load_assignment(self):
        with open(self.assignment_path, "rb") as f:
            return pickle.load(f)

    def initiation(self, _state):
        self.block = next(
            (
                b for b in self.env.blocks
                if b.carrying and not b.stored and b.storage_location is None
            ),
            None
        )
        return self.block is not None

    def policy(self, _state):
        if self.block is None:
            return SmallRoomsEnv.ACTION_IDS["WAIT"]

        block_idx = int(self.block.label[1:]) - 1
        chosen_cell = self.assignment[block_idx]
        if chosen_cell not in self.env.get_available_storage_positions():
            raise RuntimeError(
                f"Assigned GA storage position is unavailable: {chosen_cell}"
            )
        self.block.storage_location = chosen_cell

        self.env.store_events.append({
            "episode": getattr(self.env, "current_episode", 0),
            "t_step": getattr(self.env, "time_steps", 0),
            "phase": "ga_chosen",
            "block_label": self.block.label,
            "row": chosen_cell[0],
            "col": chosen_cell[1],
        })

        return SmallRoomsEnv.ACTION_IDS["WAIT"]

    def intrinsic_reward(self, state, action, next_state, info):
        return 0.0

    def termination(self, _state):
        return True

    def on_env_reset(self):
        pass

    def on_delivery(self, block_label, error_time, delivery_reward=None):
        pass

    def on_episode_end(self):
        self.pending.clear()

    def __str__(self):
        return "GAStorageSelectOption"

    __repr__ = __str__

    def __hash__(self):
        return hash("GAStorageSelectOption")
