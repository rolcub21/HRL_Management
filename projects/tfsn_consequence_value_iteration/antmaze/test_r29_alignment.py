import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r29_alignment as r29,
)
from scripts.antmaze.current_tfsn.constants import COMMAND_STEPS, STATE_DIM


class R29AlignmentTest(unittest.TestCase):
    def test_protocol_is_reward_free(self):
        protocol = json.loads(r29.PROTOCOL.read_text())
        reference = r29.validate_protocol(protocol)
        self.assertEqual(reference, r29.METHOD_REFERENCE)
        self.assertFalse(protocol["hidden_goals_accessed"])
        self.assertFalse(protocol["task_reward_used"])
        self.assertEqual(protocol["oracle_inputs"], [])

    def test_shard_validation_recomputes_connection_label(self):
        commands = 2
        states = np.zeros((commands, COMMAND_STEPS + 1, STATE_DIM), np.float32)
        states[1, 0] = states[0, -1]
        command_xy = np.zeros((commands, 2), np.float32)
        arrays = {
            "states": states,
            "commands": command_xy,
            "task_id": np.zeros(commands, np.int16),
            "episode_id": np.zeros(commands, np.int16),
            "command_index": np.arange(commands, dtype=np.int16),
            "reset_seed": np.ones(commands, np.int64),
            "task_type": np.zeros(commands, np.int8),
            "selected_connection_weight": np.ones(commands, np.float32),
            "selected_score": np.ones(commands, np.float32),
            "selected_command_distance": np.zeros(commands, np.float32),
            "selected_kind": np.asarray(["model_bridge"] * commands),
            "actual_connection_within_sigma": np.ones(commands, np.uint8),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task_0000.npz"
            np.savez_compressed(path, **arrays)
            loaded = r29.validate_shard(path, 0, commands)
        self.assertEqual(loaded["states"].shape, states.shape)


if __name__ == "__main__":
    unittest.main()
