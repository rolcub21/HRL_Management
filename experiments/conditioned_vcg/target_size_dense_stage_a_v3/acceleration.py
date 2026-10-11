"""Semantic-equivalent no-gradient deduplication for episodic TD continuation."""

from __future__ import annotations

import torch

from PSLAP.viability import RecoveryState
from PSLAP.yard_graph import encode_recovery_state, pad_yard_graphs
from viability_graph_hierarchy import (
    ACTION_FEATURE_DIM,
    double_dqn_regularized_continuation,
)
from viability_graph_episodic_audit import episodic_common_smdp_target


IMPLEMENTATION = "identity_state_dedup_no_grad_continuation_v1"


def _deduplicated_no_grad_scores(records, network, audit=None):
    if torch.is_grad_enabled():
        raise RuntimeError("accelerated continuation scoring requires no_grad")
    records = tuple(records)
    if not records:
        raise ValueError("cannot score an empty continuation frontier")
    current = tuple(record.current_state for record in records)
    successor = tuple(record.successor_state for record in records)
    states = current + successor
    unique = []
    identity_to_index = {}
    inverse = []
    for state in states:
        if not isinstance(state, RecoveryState):
            raise TypeError("candidate state must be a RecoveryState")
        identity = id(state)
        index = identity_to_index.get(identity)
        if index is None:
            index = len(unique)
            unique.append(state)
            identity_to_index[identity] = index
        elif unique[index] is not state:
            raise RuntimeError("Python object identity collision")
        inverse.append(index)

    parameter = next(network.parameters())
    features = torch.as_tensor(
        tuple(record.action_features for record in records),
        dtype=parameter.dtype,
        device=parameter.device,
    )
    if features.shape != (len(records), ACTION_FEATURE_DIM):
        raise RuntimeError("action feature schema changed")
    graphs = tuple(
        encode_recovery_state(state, timing_scale=network.config.timing_scale)
        for state in unique
    )
    unique_embeddings = network.graph_encoder(pad_yard_graphs(graphs))
    gather = torch.as_tensor(inverse, dtype=torch.long, device=parameter.device)
    embeddings = unique_embeddings.index_select(0, gather)
    count = len(records)
    current_embedding = embeddings[:count]
    successor_embedding = embeddings[count:]
    action_embedding = network.action_encoder(features)
    scores = network.q_head(
        torch.cat(
            (
                current_embedding,
                successor_embedding,
                successor_embedding - current_embedding,
                action_embedding,
            ),
            dim=-1,
        )
    ).squeeze(-1)
    if audit is not None:
        audit["calls"] = int(audit.get("calls", 0)) + 1
        audit["state_slots"] = int(audit.get("state_slots", 0)) + len(states)
        audit["unique_state_identities"] = int(
            audit.get("unique_state_identities", 0)
        ) + len(unique)
    return scores


def accelerated_episodic_td_batch(self, transitions):
    """Native prediction path plus deduplicated no-gradient continuation."""

    transitions = tuple(transitions)
    prediction = self._score_records(
        tuple(item.chosen for item in transitions),
        network=self.Q_local,
        grad=True,
    )
    next_values = torch.zeros(
        len(transitions), dtype=prediction.dtype, device=self.device
    )
    records = [
        record
        for transition in transitions
        if not transition.done
        for record in transition.next_candidates
    ]
    if records:
        stats = getattr(self, "_td_dedup_audit", None)
        if stats is None:
            stats = {
                "implementation": IMPLEMENTATION,
                "calls": 0,
                "state_slots": 0,
                "unique_state_identities": 0,
            }
            self._td_dedup_audit = stats
        with torch.no_grad():
            online = _deduplicated_no_grad_scores(
                records, self.Q_local, audit=stats
            )
            target = _deduplicated_no_grad_scores(
                records, self.Q_target, audit=stats
            )
            offset = 0
            for index, transition in enumerate(transitions):
                if transition.done:
                    continue
                count = len(transition.next_candidates)
                modes = torch.as_tensor(
                    [record.mode_id for record in transition.next_candidates],
                    dtype=torch.long,
                    device=self.device,
                )
                next_values[index] = double_dqn_regularized_continuation(
                    online[offset : offset + count],
                    target[offset : offset + count],
                    modes,
                    self.within_temperatures,
                    self.config.tau_mode,
                )
                offset += count

    rewards = torch.as_tensor(
        [item.reward for item in transitions],
        dtype=prediction.dtype,
        device=self.device,
    )
    durations = torch.as_tensor(
        [item.duration for item in transitions],
        dtype=torch.long,
        device=self.device,
    )
    dones = torch.as_tensor(
        [item.done for item in transitions],
        dtype=torch.bool,
        device=self.device,
    )
    targets = episodic_common_smdp_target(
        rewards,
        durations,
        dones,
        next_values,
        gamma=self.config.gamma,
        reward_scale=self.config.reward_scale,
    )
    return prediction, targets.detach()


def audit_payload(agent) -> dict:
    stats = dict(getattr(agent, "_td_dedup_audit", {}))
    slots = int(stats.get("state_slots", 0))
    unique = int(stats.get("unique_state_identities", 0))
    stats.update({
        "implementation": IMPLEMENTATION,
        "scope": "current_process_since_start_or_resume",
        "selection_scoring_native": True,
        "chosen_action_gradient_path_native": True,
        "state_identity_reduction_fraction": (
            None if slots == 0 else 1.0 - unique / slots
        ),
    })
    return stats
