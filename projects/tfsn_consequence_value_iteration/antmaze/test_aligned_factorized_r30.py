import unittest

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze.aligned_factorized_consequence_r30 import (
    load_protocol,
    split_tasks,
)


class R30AlignedConsequenceTest(unittest.TestCase):
    def test_protocol_and_whole_task_split(self):
        protocol = load_protocol()
        trained, held = split_tasks(protocol, 80, "full")
        self.assertEqual(len(trained), 64)
        self.assertEqual(len(held), 16)
        self.assertFalse(set(trained.tolist()) & set(held.tolist()))
        self.assertEqual(set(trained.tolist()) | set(held.tolist()), set(range(80)))

    def test_split_is_deterministic(self):
        protocol = load_protocol()
        first = split_tasks(protocol, 80, "full")
        second = split_tasks(protocol, 80, "full")
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])


if __name__ == "__main__":
    unittest.main()

