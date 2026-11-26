"""PointMaze adapter for command-scale posterior-region acquisition."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.shared.command_regions import (
    CommandRegionCover,
    collapsed_candidate_roadmap_links,
    movement_weighted_command_reach,
)
from projects.tfsn_consequence_value_iteration.shared.latent_detection import (
    LatentRadiusPosterior,
    PosteriorFalsifiedError,
)

from .factorial_planner_r12 import FactorialPointPlanner
from .planner import load_planner


HERE = Path(__file__).resolve().parent
REGION_MANIFEST = HERE.parent / "shared" / "method_manifest_command_regions_r13.json"
FACTORIAL_MANIFEST = HERE / "method_manifest_factorial_r12.json"


class CommandRegionMemory:
    def __init__(self, graph, targets, weight, sigma):
        self.graph = graph
        self.sigma = float(sigma)
        start = graph.nodes[graph.source_index, :2]
        endpoint = graph.nodes[graph.endpoint_index, :2]
        reach = movement_weighted_command_reach(
            start, graph.commands, endpoint, self.sigma
        )
        candidate, source, target, reliability = collapsed_candidate_roadmap_links(
            graph.nodes[:, :2],
            graph.source_index,
            graph.endpoint_index,
            graph.edge_reliability,
            targets,
            weight,
        )
        self.regions = CommandRegionCover.build_roadmap(
            candidate,
            reach,
            source,
            target,
            reliability,
            alias_radius=self.sigma,
            allow_unassigned=True,
        )
        self.reset()

    def reset(self):
        self.posterior = LatentRadiusPosterior(
            self.regions.candidate_xy, self.sigma
        )

    def observe(self, xy, reward):
        self.posterior.update(xy, reward)

    def region_mass(self, posterior=None):
        value = self.posterior if posterior is None else posterior
        return self.regions.masses(
            value.goal_xy, value.goal_probability
        )

    def unassigned_mass(self, posterior=None):
        value = self.posterior if posterior is None else posterior
        return self.regions.unassigned_mass(
            value.goal_xy, value.goal_probability
        )

    def diagnostic(self):
        summary = self.posterior.summary()
        return {
            "observation_memory_factor": "latent_radius",
            "passage_prediction_factor": "command_region",
            "candidate_goal_source": "unique_recorded_roadmap_state_xy",
            "candidate_goal_count": int(len(self.posterior.goal_xy)),
            "posterior_exhausted": bool(self.posterior.exhausted),
            "posterior_exhaustion_behavior": "fatal_model_falsification",
            "rho_min_data_derived_m": summary.rho_min,
            "rho_max_data_derived_m": summary.rho_max,
            "feasible_goal_candidates": summary.feasible_candidate_count,
            "posterior_marginal_evidence": summary.marginal_evidence,
            "unassigned_candidate_posterior_mass": self.unassigned_mass(),
            **self.regions.diagnostic(),
        }


class CommandRegionPointPlanner(FactorialPointPlanner):
    """Hold R12 search and execution fixed while replacing passage gain."""

    def __init__(self, base):
        self.base = base
        self.graph = base.graph
        self.sigma = base.sigma
        self.horizon = base.horizon
        settings = json.loads(FACTORIAL_MANIFEST.read_text())[
            "categories"
        ]["global_computational"]
        self.beam_width = int(settings["consumptive_lookahead_beam_width"])
        self.lookahead_depth = int(settings["consumptive_lookahead_depth"])
        if self.lookahead_depth != 2:
            raise ValueError("command-region screen requires shared depth two")
        self.factorial = CommandRegionMemory(
            self.graph, self.base.targets, self.base.weight, self.sigma
        )
        self.observation_level = "latent_radius"
        self.passage_level = "command_region"
        self.node_region = self.factorial.regions.assign(
            self.graph.nodes[:, :2]
        ).astype(np.int32)

    def _immediate(self, state, posterior):
        command, target, execution, kind = self._candidates(state)
        connection_member = self.base._member_connection(state, command)
        connection = (connection_member * execution[:, None]).min(axis=1)
        region_mass = self.factorial.region_mass(posterior)
        target_region = self.node_region[target]
        targetable = target_region >= 0
        if not np.any(targetable):
            raise RuntimeError(
                "no proposed command targets an assigned command region"
            )
        command = command[targetable]
        target = target[targetable]
        connection = connection[targetable]
        kind = kind[targetable]
        target_region = target_region[targetable]
        region_gain = np.asarray(region_mass, np.float64)[target_region].astype(
            np.float32
        )
        target_xy = self.graph.nodes[target, :2]
        local_gain = posterior.contact_probability(
            target_xy[:, None, :]
        ).astype(np.float32)
        local_score = connection * local_gain
        # Region selection and node selection are distinct. Within each
        # currently proposed region, retain the node with maximum exact local
        # posterior mass times success probability. Region mass then scores
        # which of those region representatives to pursue.
        representative = []
        for region in np.unique(target_region):
            rows = np.flatnonzero(target_region == region)
            best_value = float(local_score[rows].max())
            tied = rows[local_score[rows] == best_value]
            if len(tied) > 1:
                best_connection = float(connection[tied].max())
                tied = tied[connection[tied] == best_connection]
            representative.append(int(tied[0]))
        representative = np.asarray(representative, np.int32)
        score = connection[representative] * region_gain[representative]
        return {
            "command": command[representative],
            "target": target[representative],
            "target_region": target_region[representative],
            "connection": connection[representative],
            "region_gain": region_gain[representative],
            "local_gain": local_gain[representative],
            "local_score": local_score[representative],
            "score": score,
            "kind": kind[representative],
            "raw_candidate_count": int(len(targetable)),
            "unassigned_node_candidate_count": int((~targetable).sum()),
        }

    def choose(self, state, command_index, fields):
        if fields.contacted:
            return self.base.choose(state, command_index, fields)
        self.factorial.posterior.assert_not_exhausted(
            "PointMaze command-region acquisition decision"
        )
        remaining = self.horizon - int(command_index)
        first = self._immediate(state, self.factorial.posterior)
        beam_count = min(self.beam_width, len(first["score"]))
        beam = np.argpartition(first["score"], -beam_count)[-beam_count:]
        beam = beam[np.argsort(first["score"][beam])[::-1]]
        consumed_score = np.zeros(beam_count, np.float32)
        consumed_tail = np.zeros(beam_count, np.float32)
        for local, candidate in enumerate(beam):
            region_gain = float(
                np.clip(first["region_gain"][candidate], 0.0, 1.0)
            )
            local_gain = float(
                np.clip(first["local_gain"][candidate], 0.0, 1.0)
            )
            tail = 0.0
            local_no_contact = max(0.0, 1.0 - local_gain)
            if remaining > 1 and local_no_contact > 0.0:
                target_xy = self.graph.nodes[first["target"][candidate], :2]
                try:
                    consumed = self.factorial.posterior.imagined_no_contact(
                        target_xy[None]
                    )
                    second_state = self.graph.nodes[first["target"][candidate]]
                    second = self._immediate(second_state, consumed)
                    tail = float(np.max(second["score"]))
                except PosteriorFalsifiedError:
                    tail = 0.0
            # Region mass is route salience, not a claim that one landing
            # consumes the full region. Only the exact node-local contact
            # probability weights the no-contact branch.
            consumed_score[local] = float(first["connection"][candidate]) * (
                region_gain + local_no_contact * tail
            )
            consumed_tail[local] = tail
        local_winner = int(np.argmax(consumed_score))
        winner = int(beam[local_winner])
        diagnostic = {
            "phase": "acquisition",
            "remaining": int(remaining),
            "selected_score": float(consumed_score[local_winner]),
            "selected_kind": str(first["kind"][winner]),
            "selected_connection_weight": float(first["connection"][winner]),
            "selected_target_node_index": int(first["target"][winner]),
            "selected_target_node_xy": self.graph.nodes[
                first["target"][winner], :2
            ].astype(float).tolist(),
            "selected_region_index": int(first["target_region"][winner]),
            "selected_region_posterior_mass": float(
                first["region_gain"][winner]
            ),
            "selected_node_local_contact_mass": float(
                first["local_gain"][winner]
            ),
            "selected_node_local_mass_times_connection": float(
                first["local_score"][winner]
            ),
            "selected_consumed_tail_value": float(consumed_tail[local_winner]),
            "candidate_count": int(len(first["score"])),
            "raw_node_candidate_count": first["raw_candidate_count"],
            "unassigned_node_candidate_count": first[
                "unassigned_node_candidate_count"
            ],
            "candidate_score_maximum": float(np.max(first["score"])),
            "candidate_score_median": float(np.median(first["score"])),
            "consumptive_lookahead_depth_global": self.lookahead_depth,
            "consumptive_lookahead_beam_width_global": self.beam_width,
            "acquisition_backup": (
                "R12_local_depth_two_top_four_regional_route_salience_"
                "with_node_local_consumption"
            ),
            "advance_prefilter": "observation_neutral_maximum_reliability",
            "passage_geometry_used_for_acquisition": False,
            **self.factorial.diagnostic(),
        }
        return first["command"][winner].copy(), diagnostic


def load_command_region_planner(
    *, models, roadmap, planner, device, seed,
    observation_level, passage_level, passage_artifact=None,
):
    if observation_level != "latent_radius" or passage_level != "command_region":
        raise ValueError("R13 requires latent_radius__command_region")
    if passage_artifact is not None:
        raise ValueError("command-region acquisition does not use a passage artifact")
    base = load_planner(
        models, roadmap, planner, device, seed,
        advance_weighting=True, consumptive_lookahead=True,
    )
    return CommandRegionPointPlanner(base)
