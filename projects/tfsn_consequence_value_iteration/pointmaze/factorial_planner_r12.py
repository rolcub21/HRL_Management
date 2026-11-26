"""Isolated PointMaze adapter for the latent-memory x passage 2x2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.shared.factorial_acquisition import (
    FactorialAcquisition,
)
from projects.tfsn_consequence_value_iteration.shared.latent_detection import (
    PosteriorFalsifiedError,
)
from projects.tfsn_consequence_value_iteration.shared.learned_passage import (
    LearnedPassageEnsemble,
)

from .planner import load_planner
from .roadmap import alias_kernel


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "method_manifest_factorial_r12.json"


class FactorialPointPlanner:
    """Local depth-two acquisition adapter; frozen R11 handles return."""

    def __init__(self, base, observation_level, passage_level, passage_artifact):
        self.base = base
        self.graph = base.graph
        self.sigma = base.sigma
        self.horizon = base.horizon
        manifest = json.loads(MANIFEST.read_text())
        settings = manifest["categories"]["global_computational"]
        self.beam_width = int(settings["consumptive_lookahead_beam_width"])
        self.lookahead_depth = int(settings["consumptive_lookahead_depth"])
        if self.lookahead_depth != 2:
            raise ValueError("PointMaze factorial screen requires depth two")
        learned = (
            LearnedPassageEnsemble(
                passage_artifact, base.consequence.device,
                manifest_path=MANIFEST,
            )
            if passage_level == "shared_learned" else None
        )
        if learned is not None and learned.state_dim != 4:
            raise ValueError("passage artifact is not the PointMaze adapter artifact")
        self.factorial = FactorialAcquisition(
            recorded_goal_xy=self.graph.nodes[:, :2],
            sigma=self.sigma,
            observation_level=observation_level,
            passage_level=passage_level,
            learned_passage=learned,
            matched_passage=self._matched_passage,
        )
        self.observation_level = observation_level
        self.passage_level = passage_level

    def _matched_passage(self, start_xy, endpoint_xy):
        start = np.asarray(start_xy, np.float32).reshape(-1, 2)
        endpoint = np.asarray(endpoint_xy, np.float32).reshape(-1, 2)
        if len(start) != len(endpoint):
            raise ValueError("passage starts and endpoints must align")
        # Frozen R11 accepts one live start and a batch of endpoints.  The
        # factorial scorer carries an explicit start for every candidate.
        # Group exact repeated starts and call R11 unchanged for each group.
        unique_start, inverse = np.unique(start, axis=0, return_inverse=True)
        middle = np.empty(
            (len(start), self.base.passages.relative.shape[1], 2),
            np.float32,
        )
        for group, source in enumerate(unique_start):
            row = np.flatnonzero(inverse == group)
            middle[row] = self.base.passages.live_passage_samples(
                source, endpoint[row]
            )
        return np.concatenate(
            (start[:, None], middle, endpoint[:, None]), axis=1
        ).astype(np.float32)

    def reset_task_memory(self):
        self.base.reset_task_memory()
        self.factorial.reset()

    def begin_episode(self):
        self.base.begin_episode()

    def observe_transition(self, start_state, command, endpoint_state):
        self.base.observe_transition(start_state, command, endpoint_state)

    def observe_detection_consequence(self, xy, reward):
        self.factorial.observe(xy, reward)

    def _neutral_advance(self, node_index):
        node_index = np.asarray(node_index, np.int32).reshape(-1)
        model_target = self.base.targets[node_index]
        model_weight = self.base.weight[node_index]
        choice = np.argmax(model_weight, axis=1)
        row = np.arange(len(node_index))
        target = model_target[row, choice].astype(np.int32)
        command = self.graph.nodes[target, :2].copy()
        reliability = model_weight[row, choice].astype(np.float32)
        kind = np.full(len(node_index), "model_reliability_prefilter", object)
        for output, node in enumerate(node_index):
            begin = self.base.outgoing_offsets[node]
            stop = self.base.outgoing_offsets[node + 1]
            if begin == stop:
                continue
            edges = self.base.outgoing_order[begin:stop]
            local = int(np.argmax(self.graph.edge_reliability[edges]))
            edge = int(edges[local])
            value = float(self.graph.edge_reliability[edge])
            if value <= float(reliability[output]):
                continue
            target[output] = self.graph.endpoint_index[edge]
            command[output] = self.graph.commands[edge]
            reliability[output] = value
            kind[output] = "recorded_reliability_prefilter"
        return command, target, reliability, kind

    def _candidates(self, state):
        state = np.asarray(state, np.float32).reshape(4)
        distance, local = self.base.reach_query.query(state[:2], workers=-1)
        bridge_target = self.base.landmarks[np.asarray(local, np.int64)]
        bridge_command = self.graph.nodes[bridge_target, :2]
        association = alias_kernel(distance, self.sigma).astype(np.float32)
        advance_command, advance_target, advance_reliability, advance_kind = (
            self._neutral_advance(bridge_target)
        )
        advance_distance = np.linalg.norm(
            advance_command - self.graph.nodes[bridge_target, :2], axis=1
        )
        advance_factor = 1.0 - alias_kernel(advance_distance, self.sigma)
        command = np.concatenate((bridge_command, advance_command)).astype(np.float32)
        target = np.concatenate((bridge_target, advance_target)).astype(np.int32)
        execution = np.concatenate((
            1.0 - association,
            association * advance_factor * advance_reliability,
        )).astype(np.float32)
        kind = np.concatenate((
            np.full(len(bridge_command), "model_bridge", object),
            advance_kind,
        ))
        return command, target, execution, kind

    def _immediate(self, state, posterior):
        command, target, execution, kind = self._candidates(state)
        source = np.repeat(
            np.asarray(state, np.float32).reshape(1, 4), len(command), axis=0
        )
        connection_member = self.base._member_connection(state, command)
        connection = (connection_member * execution[:, None]).min(axis=1)
        passage = self.factorial.passages(
            source, command, self.graph.nodes[target, :2]
        )
        passage_member_gain = np.stack([
            posterior.contact_probability(value) for value in passage
        ]).astype(np.float32)
        gain = passage_member_gain.min(axis=0)
        score = connection * gain
        return {
            "command": command,
            "target": target,
            "execution": execution,
            "connection": connection,
            "gain": gain,
            "passage_member_gain": passage_member_gain,
            "passage": passage,
            "score": score,
            "kind": kind,
        }

    def choose(self, state, command_index, fields):
        if fields.contacted:
            return self.base.choose(state, command_index, fields)
        self.factorial.posterior.assert_not_exhausted(
            "PointMaze acquisition decision"
        )
        remaining = self.horizon - int(command_index)
        first = self._immediate(state, self.factorial.posterior)
        beam_count = min(self.beam_width, len(first["score"]))
        beam = np.argpartition(first["score"], -beam_count)[-beam_count:]
        beam = beam[np.argsort(first["score"][beam])[::-1]]
        consumed_score = np.zeros(beam_count, np.float32)
        consumed_tail = np.zeros(beam_count, np.float32)
        for local, candidate in enumerate(beam):
            paths = first["passage"][:, candidate]
            gains = first["passage_member_gain"][:, candidate]
            pessimistic = int(np.argmin(gains))
            path = paths[pessimistic]
            gain = float(gains[pessimistic])
            tail = 0.0
            if remaining > 1:
                try:
                    consumed = self.factorial.posterior.imagined_no_contact(path)
                    second_state = self.graph.nodes[first["target"][candidate]]
                    second = self._immediate(second_state, consumed)
                    tail = float(np.max(second["score"]))
                except PosteriorFalsifiedError:
                    # The no-contact branch has zero model probability.
                    tail = 0.0
            consumed_score[local] = float(first["connection"][candidate]) * (
                gain + (1.0 - gain) * tail
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
            "selected_first_passage_gain": float(first["gain"][winner]),
            "selected_consumed_tail_value": float(consumed_tail[local_winner]),
            "candidate_count": int(len(first["score"])),
            "candidate_score_maximum": float(np.max(first["score"])),
            "candidate_score_median": float(np.median(first["score"])),
            "consumptive_lookahead_depth_global": self.lookahead_depth,
            "consumptive_lookahead_beam_width_global": self.beam_width,
            "acquisition_backup": "local_depth_two_top_four_no_R11_field_leakage",
            "advance_prefilter": "observation_neutral_maximum_reliability",
            **self.factorial.diagnostic(),
        }
        return first["command"][winner].copy(), diagnostic


def load_factorial_planner(
    *, models, roadmap, planner, device, seed,
    observation_level, passage_level, passage_artifact=None,
):
    base = load_planner(
        models, roadmap, planner, device, seed,
        advance_weighting=True, consumptive_lookahead=True,
    )
    return FactorialPointPlanner(
        base, observation_level, passage_level, passage_artifact
    )
