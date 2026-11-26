"""Data-derived command-scale regions shared by continuous domains."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree


class RoadmapRegionUnreachableError(RuntimeError):
    """Raised when directed roadmap costs cannot assign every candidate."""

    def __init__(
        self,
        unreachable_count,
        candidate_count,
        center_count,
        directed_source_component_count,
    ):
        self.unreachable_count = int(unreachable_count)
        self.candidate_count = int(candidate_count)
        self.center_count = int(center_count)
        self.directed_source_component_count = int(
            directed_source_component_count
        )
        super().__init__(
            "directed roadmap cannot assign every command-region candidate: "
            f"{self.unreachable_count}/{self.candidate_count} candidates are "
            f"not reachable outward from any of {self.center_count} "
            "data-derived centres; the directed roadmap has "
            f"{self.directed_source_component_count} source strongly connected "
            "components and therefore requires at least that many centres for "
            "complete outward reachability; no Euclidean fallback is permitted"
        )


def collapsed_candidate_roadmap_links(
    node_xy: np.ndarray,
    recorded_source_node: np.ndarray,
    recorded_target_node: np.ndarray,
    recorded_reliability: np.ndarray,
    bridge_target_node: np.ndarray,
    bridge_reliability: np.ndarray,
):
    """Collapse node-level recorded and bridge links onto unique xy support."""

    node = np.asarray(node_xy, np.float64).reshape(-1, 2)
    if not len(node) or not np.isfinite(node).all():
        raise ValueError("roadmap node xy must be finite and nonempty")
    candidate, node_candidate = np.unique(node, axis=0, return_inverse=True)
    recorded_source = np.asarray(recorded_source_node, np.int64).reshape(-1)
    recorded_target = np.asarray(recorded_target_node, np.int64).reshape(-1)
    recorded_weight = np.asarray(recorded_reliability, np.float64).reshape(-1)
    if not (
        len(recorded_source) == len(recorded_target) == len(recorded_weight)
    ):
        raise ValueError("recorded roadmap link arrays must be aligned")
    bridge_target = np.asarray(bridge_target_node, np.int64)
    bridge_weight = np.asarray(bridge_reliability, np.float64)
    if bridge_target.shape != bridge_weight.shape:
        raise ValueError("bridge targets and reliabilities must be aligned")
    if bridge_target.ndim != 2 or bridge_target.shape[0] != len(node):
        raise ValueError("bridge links must have one row per roadmap node")
    for name, index in (
        ("recorded source", recorded_source),
        ("recorded target", recorded_target),
        ("bridge target", bridge_target.reshape(-1)),
    ):
        if np.any(index < 0) or np.any(index >= len(node)):
            raise ValueError(f"{name} exceeds roadmap node support")
    bridge_source = np.repeat(node_candidate, bridge_target.shape[1])
    return (
        candidate,
        np.concatenate((bridge_source, node_candidate[recorded_source])),
        np.concatenate((
            node_candidate[bridge_target.reshape(-1)],
            node_candidate[recorded_target],
        )),
        np.concatenate((bridge_weight.reshape(-1), recorded_weight)),
    )


def movement_weighted_command_reach(
    start_xy: np.ndarray,
    command_xy: np.ndarray,
    endpoint_xy: np.ndarray,
    sigma: float,
) -> float:
    """Median achieved displacement under the shared continuous reach measure."""

    start = np.asarray(start_xy, np.float64).reshape(-1, 2)
    command = np.asarray(command_xy, np.float64).reshape(-1, 2)
    endpoint = np.asarray(endpoint_xy, np.float64).reshape(-1, 2)
    if not (len(start) == len(command) == len(endpoint)) or not len(start):
        raise ValueError("recorded command rows must be nonempty and aligned")
    if not np.isfinite(sigma) or float(sigma) <= 0.0:
        raise ValueError("landing sigma must be finite and positive")
    if not all(np.isfinite(value).all() for value in (start, command, endpoint)):
        raise ValueError("recorded command rows must be finite")
    requested = np.linalg.norm(command - start, axis=1)
    achieved = np.linalg.norm(endpoint - start, axis=1)
    denominator = 2.0 * float(sigma) ** 2
    weight = (
        (1.0 - np.exp(-np.square(requested) / denominator))
        * (1.0 - np.exp(-np.square(achieved) / denominator))
    )
    total = float(weight.sum())
    if not total > 0.0:
        raise ValueError("recorded commands contain no movement-weighted reach")
    order = np.argsort(achieved, kind="stable")
    cumulative = np.cumsum(weight[order])
    selected = min(
        int(np.searchsorted(cumulative, 0.5 * total, side="left")),
        len(order) - 1,
    )
    reach = float(achieved[order[selected]])
    if not np.isfinite(reach) or reach <= 0.0:
        raise ValueError("data-derived command reach must be positive")
    return reach


@dataclass
class CommandRegionCover:
    """Deterministic command-scale centres with roadmap-cost assignment."""

    candidate_xy: np.ndarray
    cover_radius: float
    region_index: np.ndarray
    center_candidate_index: np.ndarray
    roadmap_cost: np.ndarray
    _tree: cKDTree = field(init=False, repr=False)

    def __post_init__(self):
        self.candidate_xy = np.asarray(self.candidate_xy, np.float64).reshape(-1, 2)
        self.region_index = np.asarray(self.region_index, np.int32).reshape(-1)
        self.center_candidate_index = np.asarray(
            self.center_candidate_index, np.int32
        ).reshape(-1)
        self.roadmap_cost = np.asarray(self.roadmap_cost, np.float64).reshape(-1)
        if not (
            len(self.candidate_xy)
            == len(self.region_index)
            == len(self.roadmap_cost)
        ):
            raise ValueError("command-region cover arrays are not aligned")
        self._tree = cKDTree(self.candidate_xy)

    @staticmethod
    def _unique_sorted(candidate_xy):
        candidate = np.asarray(candidate_xy, np.float64).reshape(-1, 2)
        unique = np.unique(candidate, axis=0)
        if not len(unique) or not np.isfinite(unique).all():
            raise ValueError("candidate support must be finite and nonempty")
        return unique

    @classmethod
    def farthest_point_centers(cls, candidate_xy, cover_radius):
        """Choose a seed-free Euclidean k-center cover at the reach scale."""
        candidate = cls._unique_sorted(candidate_xy)
        if not np.isfinite(cover_radius) or float(cover_radius) <= 0.0:
            raise ValueError("command-region cover radius must be positive")
        mean = candidate.mean(axis=0)
        first = int(np.argmin(np.linalg.norm(candidate - mean[None], axis=1)))
        centers = [first]
        nearest = np.linalg.norm(candidate - candidate[first][None], axis=1)
        while float(nearest.max()) > float(cover_radius):
            center = int(np.argmax(nearest))
            centers.append(center)
            nearest = np.minimum(
                nearest,
                np.linalg.norm(candidate - candidate[center][None], axis=1),
            )
        return candidate, np.asarray(centers, np.int32), nearest

    @staticmethod
    def _maximum_reliability_graph(
        candidate_count, source_candidate, target_candidate, edge_reliability
    ):
        source = np.asarray(source_candidate, np.int64).reshape(-1)
        target = np.asarray(target_candidate, np.int64).reshape(-1)
        reliability = np.asarray(edge_reliability, np.float64).reshape(-1)
        if not (len(source) == len(target) == len(reliability)):
            raise ValueError("roadmap link arrays must be aligned")
        if np.any(source < 0) or np.any(source >= candidate_count):
            raise ValueError("roadmap link source exceeds candidate support")
        if np.any(target < 0) or np.any(target >= candidate_count):
            raise ValueError("roadmap link target exceeds candidate support")
        if np.any(reliability < 0.0) or np.any(reliability > 1.0):
            raise ValueError("roadmap link reliability must lie in [0,1]")
        keep = np.isfinite(reliability) & (reliability > 0.0)
        source, target, reliability = (
            source[keep], target[keep], reliability[keep]
        )
        if not len(source):
            return csr_matrix((candidate_count, candidate_count), dtype=np.float64)
        # Several bridge proposals can produce the same directed link. The
        # planner can select the most reliable one, so its Bellman-equivalent
        # link reliability is max(w), and its retry cost is 1/max(w).
        key = source * int(candidate_count) + target
        order = np.argsort(key, kind="stable")
        key, reliability = key[order], reliability[order]
        begin = np.r_[0, 1 + np.flatnonzero(key[1:] != key[:-1])]
        unique_key = key[begin]
        maximum = np.maximum.reduceat(reliability, begin)
        return csr_matrix(
            (
                1.0 / maximum,
                (
                    unique_key // int(candidate_count),
                    unique_key % int(candidate_count),
                ),
            ),
            shape=(candidate_count, candidate_count),
        )

    @classmethod
    def build_roadmap(
        cls,
        candidate_xy,
        cover_radius,
        source_candidate,
        target_candidate,
        edge_reliability,
        alias_radius=None,
        allow_unassigned=False,
    ):
        """Select centres geometrically and assign by directed retry cost.

        Exact continuous source states can lack an incoming recorded edge even
        when the executor has landed at the same controllable place.  When an
        alias radius is supplied, each candidate unreachable in the original
        graph gets at most one zero-command attachment from its nearest
        originally reachable candidate within that radius.  Attachments do
        not chain, so dense samples cannot percolate across a region.
        """
        candidate, centers, nearest = cls.farthest_point_centers(
            candidate_xy, cover_radius
        )
        graph = cls._maximum_reliability_graph(
            len(candidate), source_candidate, target_candidate, edge_reliability
        )
        cost, _predecessor, source = dijkstra(
            graph,
            directed=True,
            indices=centers,
            return_predecessors=True,
            min_only=True,
        )
        pre_alias_unreachable = source < 0
        alias_attachment_count = 0
        if alias_radius is not None:
            if not np.isfinite(alias_radius) or float(alias_radius) <= 0.0:
                raise ValueError("command-region alias radius must be positive")
            reachable_index = np.flatnonzero(~pre_alias_unreachable)
            unreachable_index = np.flatnonzero(pre_alias_unreachable)
            if len(unreachable_index):
                reachable_tree = cKDTree(candidate[reachable_index])
                distance_value, local = reachable_tree.query(
                    candidate[unreachable_index],
                    k=1,
                    distance_upper_bound=float(alias_radius),
                    workers=-1,
                )
                attach = np.isfinite(distance_value)
                alias_target = unreachable_index[attach]
                alias_source = reachable_index[np.asarray(local[attach], np.int64)]
                alias_attachment_count = int(len(alias_target))
                if alias_attachment_count:
                    stored = graph.tocoo()
                    graph = csr_matrix(
                        (
                            np.concatenate((
                                stored.data,
                                np.zeros(alias_attachment_count, np.float64),
                            )),
                            (
                                np.concatenate((stored.row, alias_source)),
                                np.concatenate((stored.col, alias_target)),
                            ),
                        ),
                        shape=graph.shape,
                    )
                    cost, _predecessor, source = dijkstra(
                        graph,
                        directed=True,
                        indices=centers,
                        return_predecessors=True,
                        min_only=True,
                    )
        unreachable = source < 0
        if np.any(unreachable) and not allow_unassigned:
            component_count, component = connected_components(
                graph, directed=True, connection="strong"
            )
            stored = graph.tocoo()
            row, column = stored.row, stored.col
            cross = component[row] != component[column]
            has_incoming = np.zeros(component_count, dtype=bool)
            has_incoming[component[column[cross]]] = True
            source_component_count = int((~has_incoming).sum())
            raise RoadmapRegionUnreachableError(
                int(unreachable.sum()),
                len(candidate),
                len(centers),
                source_component_count,
            )
        center_to_region = np.full(len(candidate), -1, np.int32)
        center_to_region[centers] = np.arange(len(centers), dtype=np.int32)
        region = np.full(len(candidate), -1, np.int32)
        assigned = ~unreachable
        region[assigned] = center_to_region[
            np.asarray(source[assigned], np.int64)
        ]
        if np.any(region[assigned] < 0):
            raise RuntimeError("Dijkstra returned an unknown command-region source")
        value = cls(
            candidate_xy=candidate,
            cover_radius=float(cover_radius),
            region_index=region,
            center_candidate_index=centers,
            roadmap_cost=cost,
        )
        value._euclidean_center_distance_max = float(nearest.max())
        value._roadmap_edge_count = int(graph.nnz)
        value._pre_alias_unassigned_count = int(pre_alias_unreachable.sum())
        value._alias_radius = (
            None if alias_radius is None else float(alias_radius)
        )
        value._alias_attachment_count = alias_attachment_count
        return value

    @classmethod
    def build(cls, candidate_xy: np.ndarray, cover_radius: float):
        """Euclidean-only helper retained for unit tests, not R13 deployment."""
        candidate = np.unique(
            np.asarray(candidate_xy, np.float64).reshape(-1, 2), axis=0
        )
        if not len(candidate) or not np.isfinite(candidate).all():
            raise ValueError("candidate support must be finite and nonempty")
        if not np.isfinite(cover_radius) or float(cover_radius) <= 0.0:
            raise ValueError("command-region cover radius must be positive")
        tree = cKDTree(candidate)
        region = np.full(len(candidate), -1, np.int32)
        centers = []
        # np.unique supplies lexicographic order. The first uncovered point is
        # therefore a deterministic center with no random seed or grid origin.
        for center in range(len(candidate)):
            if region[center] >= 0:
                continue
            neighbors = np.asarray(
                tree.query_ball_point(candidate[center], float(cover_radius)),
                np.int64,
            )
            uncovered = neighbors[region[neighbors] < 0]
            region[uncovered] = len(centers)
            centers.append(center)
        if np.any(region < 0):
            raise RuntimeError("greedy command-region cover left candidates unassigned")
        return cls(
            candidate_xy=candidate,
            cover_radius=float(cover_radius),
            region_index=region,
            center_candidate_index=np.asarray(centers, np.int32),
            roadmap_cost=np.zeros(len(candidate), np.float64),
        )

    @property
    def region_count(self):
        return int(len(self.center_candidate_index))

    def assign(self, points_xy: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xy, np.float64).reshape(-1, 2)
        nearest = self._tree.query(points, k=1, workers=-1)[1]
        return self.region_index[np.asarray(nearest, np.int64)]

    def masses(
        self, probability_candidate_xy: np.ndarray,
        candidate_probability: np.ndarray,
    ) -> np.ndarray:
        probability = self._aligned_probability(
            probability_candidate_xy, candidate_probability
        )
        assigned = self.region_index >= 0
        return np.bincount(
            self.region_index[assigned],
            weights=probability[assigned],
            minlength=self.region_count,
        ).astype(np.float64)

    def unassigned_mass(
        self,
        probability_candidate_xy: np.ndarray,
        candidate_probability: np.ndarray,
    ) -> float:
        probability = self._aligned_probability(
            probability_candidate_xy, candidate_probability
        )
        return float(probability[self.region_index < 0].sum())

    def _aligned_probability(
        self,
        probability_candidate_xy: np.ndarray,
        candidate_probability: np.ndarray,
    ) -> np.ndarray:
        coordinates = np.asarray(
            probability_candidate_xy, np.float64
        ).reshape(-1, 2)
        if not np.array_equal(coordinates, self.candidate_xy):
            raise ValueError(
                "candidate probability coordinates are not exactly aligned "
                "with command-region support"
            )
        probability = np.asarray(candidate_probability, np.float64).reshape(-1)
        if len(probability) != len(self.candidate_xy):
            raise ValueError("candidate probabilities do not match region support")
        if np.any(probability < 0.0) or not np.isfinite(probability).all():
            raise ValueError("candidate probabilities must be finite and nonnegative")
        return probability

    def diagnostic(self):
        assigned = self.region_index >= 0
        size = np.bincount(
            self.region_index[assigned], minlength=self.region_count
        )
        finite_cost = self.roadmap_cost[np.isfinite(self.roadmap_cost)]
        return {
            "command_region_scale_rule": (
                "movement_weighted_median_recorded_achieved_displacement"
            ),
            "command_region_cover_rule": (
                "farthest_point_centres_then_directed_roadmap_retry_cost_voronoi"
            ),
            "command_region_cover_radius_m": self.cover_radius,
            "command_region_cover_radius_semantics": (
                "centre_count_until_every_candidate_is_within_one_reach_of_a_centre;_"
                "regions_can_span_two_reaches"
            ),
            "command_region_count": self.region_count,
            "command_region_candidate_count": int(len(self.candidate_xy)),
            "command_region_pre_alias_unassigned_candidate_count": int(
                getattr(self, "_pre_alias_unassigned_count", 0)
            ),
            "command_region_alias_radius_m": getattr(
                self, "_alias_radius", None
            ),
            "command_region_alias_rule": (
                "one_non_chaining_nearest_originally_reachable_attachment_"
                "within_sigma"
            ),
            "command_region_alias_attachment_count": int(
                getattr(self, "_alias_attachment_count", 0)
            ),
            "command_region_unassigned_candidate_count": int((~assigned).sum()),
            "command_region_unassigned_candidate_fraction": float(
                (~assigned).mean()
            ),
            "command_region_size_min": int(size.min()),
            "command_region_size_median": float(np.median(size)),
            "command_region_size_max": int(size.max()),
            "command_region_roadmap_cost_rule": "directed_shortest_path_sum_1_over_w",
            "command_region_roadmap_cost_max": (
                float(finite_cost.max()) if len(finite_cost) else None
            ),
            "command_region_roadmap_edge_count": int(
                getattr(self, "_roadmap_edge_count", 0)
            ),
        }
