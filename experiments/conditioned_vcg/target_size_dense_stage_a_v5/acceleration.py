"""Exact TD backup with bounded immutable graph-input tensor caching."""

from __future__ import annotations

from collections import OrderedDict

import torch

from PSLAP.viability import RecoveryState
from PSLAP.yard_graph import encode_recovery_state
from experiments.conditioned_vcg.target_size_dense_stage_a_v4 import acceleration as v4
from viability_graph_hierarchy import (
    ACTION_FEATURE_DIM,
    double_dqn_regularized_continuation,
)
from viability_graph_episodic_audit import episodic_common_smdp_target


IMPLEMENTATION = (
    "identity_graph_input_tensor_cache_plus_frozen_target_score_cache_v1"
)
MAXIMUM_GRAPH_CACHE_ENTRIES = 100_000


def _graph_stats(agent):
    value = getattr(agent, "_graph_input_cache_audit", None)
    if value is None:
        value = {
            "implementation": IMPLEMENTATION,
            "requests": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "batch_calls": 0,
            "state_slots": 0,
            "unique_state_identities": 0,
            "shared_topology_batches": 0,
        }
        agent._graph_input_cache_audit = value
    return value


def _cache(agent, network):
    parameter = next(network.parameters())
    signature = (
        str(parameter.dtype),
        float(network.config.timing_scale),
    )
    if getattr(agent, "_graph_input_cache_signature", None) != signature:
        agent._graph_input_cache = OrderedDict()
        agent._graph_topology_cache = {}
        agent._graph_input_cache_signature = signature
    return agent._graph_input_cache, agent._graph_topology_cache


def _encoded_input(agent, state: RecoveryState, network):
    if not isinstance(state, RecoveryState):
        raise TypeError("candidate state must be a RecoveryState")
    cache, topologies = _cache(agent, network)
    identity = id(state)
    entry = cache.get(identity)
    stats = _graph_stats(agent)
    stats["requests"] += 1
    if entry is not None and entry[0] is state:
        stats["hits"] += 1
        cache.move_to_end(identity)
        return entry[1]

    stats["misses"] += 1
    graph = encode_recovery_state(
        state, timing_scale=network.config.timing_scale
    )
    parameter = next(network.parameters())
    features = torch.tensor(
        graph.node_features,
        dtype=parameter.dtype,
        device="cpu",
    )
    topology_key = (state.rows, state.cols, state.traversable)
    topology = topologies.get(topology_key)
    if topology is None:
        topology = torch.tensor(
            graph.edge_index, dtype=torch.long, device="cpu"
        ).reshape(2, -1)
        topologies[topology_key] = topology
    encoded = (features, topology)
    cache[identity] = (state, encoded)
    cache.move_to_end(identity)
    if len(cache) > MAXIMUM_GRAPH_CACHE_ENTRIES:
        cache.popitem(last=False)
        stats["evictions"] += 1
    return encoded


def _batched_graph_tensors(agent, states, network):
    states = tuple(states)
    if not states:
        raise ValueError("at least one recovery state is required")
    entries = tuple(_encoded_input(agent, state, network) for state in states)
    features = tuple(entry[0] for entry in entries)
    topologies = tuple(entry[1] for entry in entries)
    node_counts = tuple(value.shape[0] for value in features)
    maximum = max(node_counts)
    feature_dim = features[0].shape[1]
    if any(value.shape[1] != feature_dim for value in features):
        raise RuntimeError("cached graph feature schema changed")

    if all(count == maximum for count in node_counts):
        node_features = torch.stack(features)
        node_mask = torch.ones(
            (len(features), maximum), dtype=torch.bool, device="cpu"
        )
    else:
        node_features = features[0].new_zeros(
            (len(features), maximum, feature_dim)
        )
        node_mask = torch.zeros(
            (len(features), maximum), dtype=torch.bool, device="cpu"
        )
        for index, value in enumerate(features):
            count = value.shape[0]
            node_features[index, :count].copy_(value)
            node_mask[index, :count] = True

    shared_topology = all(value is topologies[0] for value in topologies)
    if shared_topology:
        base = topologies[0]
        offsets = (
            torch.arange(len(states), dtype=torch.long)
            .mul_(maximum)
            .reshape(-1, 1, 1)
        )
        edge_index = (
            base.unsqueeze(0).add(offsets).permute(1, 0, 2).reshape(2, -1)
        )
        _graph_stats(agent)["shared_topology_batches"] += 1
    else:
        edge_index = torch.cat(tuple(
            value + index * maximum
            for index, value in enumerate(topologies)
        ), dim=1)

    parameter = next(network.parameters())
    return (
        node_features.to(device=parameter.device, non_blocking=False),
        node_mask.to(device=parameter.device, non_blocking=False),
        edge_index.to(device=parameter.device, non_blocking=False),
    )


def _deduplicated_cached_scores(agent, records, network, audit=None):
    if torch.is_grad_enabled():
        raise RuntimeError("cached continuation scoring requires no_grad")
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
    action_features = torch.as_tensor(
        tuple(record.action_features for record in records),
        dtype=parameter.dtype,
        device=parameter.device,
    )
    if action_features.shape != (len(records), ACTION_FEATURE_DIM):
        raise RuntimeError("action feature schema changed")
    graph_tensors = _batched_graph_tensors(agent, unique, network)
    unique_embeddings = network.graph_encoder(*graph_tensors)
    gather = torch.as_tensor(
        inverse, dtype=torch.long, device=parameter.device
    )
    embeddings = unique_embeddings.index_select(0, gather)
    count = len(records)
    current_embedding = embeddings[:count]
    successor_embedding = embeddings[count:]
    action_embedding = network.action_encoder(action_features)
    scores = network.q_head(torch.cat((
        current_embedding,
        successor_embedding,
        successor_embedding - current_embedding,
        action_embedding,
    ), dim=-1)).squeeze(-1)

    stats = _graph_stats(agent)
    stats["batch_calls"] += 1
    stats["state_slots"] += len(states)
    stats["unique_state_identities"] += len(unique)
    if audit is not None:
        audit["calls"] = int(audit.get("calls", 0)) + 1
        audit["state_slots"] = int(audit.get("state_slots", 0)) + len(states)
        audit["unique_state_identities"] = int(
            audit.get("unique_state_identities", 0)
        ) + len(unique)
    return scores


def _target_scores(agent, records):
    records = tuple(records)
    stats = v4._stats(agent)
    generation = agent.gradient_steps // agent.config.target_update_every
    if getattr(agent, "_target_score_cache_generation", None) != generation:
        agent._target_score_cache = {}
        agent._target_score_cache_generation = generation
        stats["invalidations"] += 1
    cache = agent._target_score_cache
    identities = [id(record) for record in records]
    missing = []
    seen = set()
    hits = 0
    for identity, record in zip(identities, records):
        entry = cache.get(identity)
        if entry is not None and entry[0] is record:
            hits += 1
        elif identity not in seen:
            missing.append(record)
            seen.add(identity)
    if missing and len(cache) + len(missing) > v4.MAXIMUM_CACHE_ENTRIES:
        cache.clear()
        stats["capacity_clears"] += 1
        missing = []
        seen.clear()
        hits = 0
        for identity, record in zip(identities, records):
            if identity not in seen:
                missing.append(record)
                seen.add(identity)
    if missing:
        with torch.no_grad():
            values = _deduplicated_cached_scores(
                agent, missing, agent.Q_target
            ).detach().cpu().tolist()
        for record, value in zip(missing, values):
            cache[id(record)] = (record, float(value))
    stats["requests"] += len(records)
    stats["hits"] += hits
    stats["misses"] += len(records) - hits
    parameter = next(agent.Q_target.parameters())
    return torch.as_tensor(
        [cache[id(record)][1] for record in records],
        dtype=parameter.dtype,
        device=parameter.device,
    )


def accelerated_cached_tensor_td_batch(self, transitions):
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
        target_stats = v4._stats(self)
        local_dedup = {
            "calls": 0, "state_slots": 0, "unique_state_identities": 0
        }
        with torch.no_grad():
            online = _deduplicated_cached_scores(
                self, records, self.Q_local, audit=local_dedup
            )
            target = _target_scores(self, records)
            target_stats["dedup_calls"] += local_dedup["calls"]
            target_stats["state_slots"] += local_dedup["state_slots"]
            target_stats["unique_state_identities"] += local_dedup[
                "unique_state_identities"
            ]
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
                    online[offset:offset + count],
                    target[offset:offset + count],
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


def audit_payload(agent):
    target = v4.audit_payload(agent)
    graph = dict(_graph_stats(agent))
    requests = int(graph["requests"])
    slots = int(graph["state_slots"])
    unique = int(graph["unique_state_identities"])
    graph.update({
        "scope": "current_process_since_start_or_resume",
        "maximum_cache_entries": MAXIMUM_GRAPH_CACHE_ENTRIES,
        "current_cache_entries": len(
            getattr(agent, "_graph_input_cache", {})
        ),
        "topology_cache_entries": len(
            getattr(agent, "_graph_topology_cache", {})
        ),
        "hit_rate": None if requests == 0 else graph["hits"] / requests,
        "identity_reduction_fraction": (
            None if slots == 0 else 1.0 - unique / slots
        ),
        "checkpointed": False,
        "learned_embeddings_cached": False,
    })
    return {
        "implementation": IMPLEMENTATION,
        "graph_input_tensor_cache": graph,
        "frozen_target_score_cache": target,
        "selection_scoring_native": True,
        "chosen_action_gradient_path_native": True,
    }
