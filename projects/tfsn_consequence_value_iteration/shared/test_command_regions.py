"""Contract tests for shared command-scale regions."""

import unittest

import numpy as np

from .command_regions import (
    CommandRegionCover,
    RoadmapRegionUnreachableError,
    movement_weighted_command_reach,
)


class CommandRegionTest(unittest.TestCase):
    def test_scale_is_weighted_median_of_achieved_displacement(self):
        start = np.zeros((3, 2), np.float32)
        command = np.asarray([[0, 0], [2, 0], [4, 0]], np.float32)
        endpoint = np.asarray([[0, 0], [2, 0], [4, 0]], np.float32)
        self.assertEqual(
            movement_weighted_command_reach(start, command, endpoint, 0.5),
            4.0,
        )

    def test_cover_is_deterministic_and_mass_preserving(self):
        points = np.asarray([[2, 0], [0, 0], [0.5, 0], [3, 0]], np.float32)
        first = CommandRegionCover.build(points, 1.0)
        second = CommandRegionCover.build(points[::-1], 1.0)
        np.testing.assert_array_equal(first.candidate_xy, second.candidate_xy)
        np.testing.assert_array_equal(first.region_index, second.region_index)
        mass = first.masses(first.candidate_xy, np.full(4, 0.25))
        self.assertAlmostEqual(float(mass.sum()), 1.0)
        self.assertEqual(first.region_count, 2)

    def test_mass_rejects_silent_candidate_reordering(self):
        points = np.asarray([[0, 0], [1, 0], [2, 0]], np.float32)
        cover = CommandRegionCover.build(points, 1.0)
        with self.assertRaisesRegex(ValueError, "not exactly aligned"):
            cover.masses(cover.candidate_xy[::-1], np.full(3, 1.0 / 3.0))

    def test_roadmap_assignment_is_directed_and_reliability_weighted(self):
        points = np.asarray(
            [[0, 0], [0, 1], [10, 0], [10, 1]], np.float32
        )
        cover = CommandRegionCover.build_roadmap(
            points,
            1.1,
            source_candidate=np.asarray([0, 3, 0, 3]),
            target_candidate=np.asarray([1, 1, 2, 2]),
            edge_reliability=np.asarray([0.1, 1.0, 0.1, 1.0]),
        )
        np.testing.assert_array_equal(cover.center_candidate_index, [0, 3])
        # Candidate 1 is geometrically beside centre 0, but retry cost is one
        # command from centre 3 versus ten from centre 0.
        self.assertEqual(int(cover.region_index[1]), 1)
        self.assertEqual(int(cover.region_index[2]), 1)

    def test_roadmap_assignment_has_no_euclidean_fallback(self):
        points = np.asarray([[0, 0], [0.5, 0], [5, 0]], np.float32)
        with self.assertRaises(RoadmapRegionUnreachableError) as caught:
            CommandRegionCover.build_roadmap(
                points,
                1.0,
                source_candidate=np.asarray([], np.int32),
                target_candidate=np.asarray([], np.int32),
                edge_reliability=np.asarray([], np.float32),
            )
        self.assertEqual(caught.exception.unreachable_count, 1)
        self.assertEqual(caught.exception.directed_source_component_count, 3)

    def test_sigma_alias_attaches_only_within_declared_resolution(self):
        points = np.asarray([[0, 0], [0.5, 0], [5, 0]], np.float32)
        cover = CommandRegionCover.build_roadmap(
            points,
            1.0,
            source_candidate=np.asarray([], np.int32),
            target_candidate=np.asarray([], np.int32),
            edge_reliability=np.asarray([], np.float32),
            alias_radius=0.6,
            allow_unassigned=True,
        )
        diagnostic = cover.diagnostic()
        self.assertEqual(
            diagnostic["command_region_pre_alias_unassigned_candidate_count"], 1
        )
        self.assertEqual(
            diagnostic["command_region_alias_attachment_count"], 1
        )
        self.assertEqual(
            diagnostic["command_region_unassigned_candidate_count"], 0
        )

    def test_remaining_gap_is_excluded_from_regions_but_keeps_mass(self):
        points = np.asarray([[0, 0], [0.5, 0], [5, 0]], np.float32)
        cover = CommandRegionCover.build_roadmap(
            points,
            1.0,
            source_candidate=np.asarray([], np.int32),
            target_candidate=np.asarray([], np.int32),
            edge_reliability=np.asarray([], np.float32),
            alias_radius=0.1,
            allow_unassigned=True,
        )
        probability = np.full(3, 1.0 / 3.0)
        self.assertAlmostEqual(
            cover.unassigned_mass(cover.candidate_xy, probability), 1.0 / 3.0
        )
        self.assertAlmostEqual(
            float(cover.masses(cover.candidate_xy, probability).sum()),
            2.0 / 3.0,
        )

    def test_alias_attachments_do_not_chain(self):
        points = np.asarray(
            [[0, 0], [0.09, 0], [0.18, 0], [10, 0]], np.float32
        )
        cover = CommandRegionCover.build_roadmap(
            points,
            1.0,
            source_candidate=np.asarray([], np.int32),
            target_candidate=np.asarray([], np.int32),
            edge_reliability=np.asarray([], np.float32),
            alias_radius=0.1,
            allow_unassigned=True,
        )
        diagnostic = cover.diagnostic()
        self.assertEqual(
            diagnostic["command_region_pre_alias_unassigned_candidate_count"], 2
        )
        self.assertEqual(
            diagnostic["command_region_alias_attachment_count"], 1
        )
        self.assertEqual(
            diagnostic["command_region_unassigned_candidate_count"], 1
        )


if __name__ == "__main__":
    unittest.main()
