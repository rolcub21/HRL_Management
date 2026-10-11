"""No-gradient state deduplication plus bounded frozen-target score caching."""

from __future__ import annotations

import torch

from viability_graph_hierarchy import double_dqn_regularized_continuation
from viability_graph_episodic_audit import episodic_common_smdp_target
from experiments.conditioned_vcg.target_size_dense_stage_a_v3 import acceleration as v3


IMPLEMENTATION = "identity_state_dedup_plus_frozen_target_score_cache_v1"
MAXIMUM_CACHE_ENTRIES = 1_000_000


def _stats(agent):
    value = getattr(agent, "_target_score_cache_audit", None)
    if value is None:
        value = {
            "implementation": IMPLEMENTATION,
            "requests": 0,
            "hits": 0,
            "misses": 0,
            "invalidations": 0,
            "capacity_clears": 0,
            "dedup_calls": 0,
            "state_slots": 0,
            "unique_state_identities": 0,
        }
        agent._target_score_cache_audit = value
    return value


def _target_scores(agent, records):
    records = tuple(records)
    stats = _stats(agent)
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
    if missing and len(cache) + len(missing) > MAXIMUM_CACHE_ENTRIES:
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
            values = v3._deduplicated_no_grad_scores(
                missing, agent.Q_target
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


def accelerated_cached_td_batch(self, transitions):
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
        stats = _stats(self)
        local_dedup = {"calls": 0, "state_slots": 0, "unique_state_identities": 0}
        with torch.no_grad():
            online = v3._deduplicated_no_grad_scores(
                records, self.Q_local, audit=local_dedup
            )
            target = _target_scores(self, records)
            stats["dedup_calls"] += local_dedup["calls"]
            stats["state_slots"] += local_dedup["state_slots"]
            stats["unique_state_identities"] += local_dedup["unique_state_identities"]
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


def audit_payload(agent):
    stats = dict(_stats(agent))
    requests = int(stats["requests"])
    slots = int(stats["state_slots"])
    unique = int(stats["unique_state_identities"])
    stats.update({
        "scope": "current_process_since_start_or_resume",
        "maximum_cache_entries": MAXIMUM_CACHE_ENTRIES,
        "current_cache_entries": len(getattr(agent, "_target_score_cache", {})),
        "hit_rate": None if requests == 0 else stats["hits"] / requests,
        "online_state_identity_reduction_fraction": (
            None if slots == 0 else 1.0 - unique / slots
        ),
        "selection_scoring_native": True,
        "chosen_action_gradient_path_native": True,
    })
    return stats
