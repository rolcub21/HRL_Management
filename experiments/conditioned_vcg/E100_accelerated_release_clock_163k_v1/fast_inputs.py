"""Exact, weight-independent graph-input cache for VCG Q scoring.

Each RecoveryState is encoded once. Static geometry (9 feature columns and the
edge list) is shared per geometry template; the 6 state-dependent columns are
stored per state on the network device. Scaled times are computed with the same
Python float64 arithmetic as ``encode_recovery_state`` before the float32 cast,
so assembled inputs are bit-identical to the reference padded batch. All 8x8
graphs have the same node count, so no padding is needed; mixed node counts
fall back to the reference encoder.
"""
from contextlib import contextmanager
from unittest.mock import patch
import math
import torch
from PSLAP.viability import RecoveryState
from PSLAP.yard_graph import encode_recovery_state, NODE_FEATURE_INDEX as I, pad_yard_graphs
from viability_graph_hierarchy import ViabilityGraphQNetwork, ACTION_FEATURE_DIM
from experiments.conditioned_vcg.target_size_dense_stage_a_v3 import acceleration as v3

DYNAMIC = ('agent', 'occupied', 'occupant_time_known', 'occupant_remaining_time_scaled',
           'occupant_abs_remaining_time_scaled', 'occupant_overdue')
DYNAMIC_COLUMNS = [I[name] for name in DYNAMIC]
IMPLEMENTATION = 'e100_exact_device_graph_input_cache_v1'


class InputCache:
    def __init__(self, device, dtype, timing_scale, capacity=4_000_000, chunk=65536):
        if not math.isfinite(timing_scale) or timing_scale <= 0:
            raise ValueError('Invalid timing scale')
        self.device, self.dtype, self.scale = torch.device(device), dtype, float(timing_scale)
        self.capacity, self.chunk = capacity, chunk
        self.templates = {}        # geometry key -> template index
        self.static = []           # per template: [N, F] tensor with dynamic columns zeroed
        self.edges = []            # per template: [2, E] long tensor
        self.slots = {}            # id(state) -> (state, slot); holding state prevents id reuse
        self.dynamic = torch.zeros((0, 0, len(DYNAMIC)), dtype=dtype, device=self.device)
        self.template_of = torch.zeros(0, dtype=torch.long, device=self.device)
        self.size = 0
        self.stats = dict(implementation=IMPLEMENTATION, hits=0, misses=0, clears=0, templates=0)

    def _template(self, s):
        key = (s.rows, s.cols, s.traversable, s.storage_cells, s.exits, s.fixed_obstacles,
               s.reserved_cells, s.pickup_cells, s.wait_cells)
        index = self.templates.get(key)
        if index is None:
            # Reference encoder validates geometry and fixes feature/edge order.
            graph = encode_recovery_state(s, timing_scale=self.scale)
            features = torch.tensor(graph.node_features, dtype=self.dtype)
            features[:, DYNAMIC_COLUMNS] = 0
            index = len(self.static)
            self.templates[key] = index
            self.static.append(features.to(self.device))
            self.edges.append(torch.tensor(graph.edge_index, dtype=torch.long).reshape(2, -1).to(self.device))
            self.stats['templates'] = len(self.static)
        return index

    def _dynamic_rows(self, s):
        cells = s.rows * s.cols
        row = [[0.0] * len(DYNAMIC) for _ in range(cells)]
        r, c = s.agent_position
        if not (0 <= r < s.rows and 0 <= c < s.cols):
            raise ValueError('agent_position out of bounds')
        row[r * s.cols + c][0] = 1.0
        for (r, c), block in s.occupancy().items():
            if not (0 <= r < s.rows and 0 <= c < s.cols):
                raise ValueError('block position out of bounds')
            values = row[r * s.cols + c]
            values[1] = 1.0
            try:
                remaining = float(getattr(block, 'remaining_time', None))
            except (TypeError, ValueError):
                remaining = math.nan
            if math.isfinite(remaining):
                scaled = remaining / self.scale
                values[2:6] = [1.0, scaled, abs(scaled), float(remaining <= 0.0)]
        return row

    def _grow(self, needed, cells):
        if self.dynamic.shape[1] not in (0, cells):
            raise ValueError('mixed node counts')
        if needed <= self.dynamic.shape[0]:
            return
        size = max(needed, min(self.capacity, max(2 * self.dynamic.shape[0], self.chunk)))
        dynamic = torch.zeros((size, cells, len(DYNAMIC)), dtype=self.dtype, device=self.device)
        template = torch.zeros(size, dtype=torch.long, device=self.device)
        if self.size:
            dynamic[:self.size] = self.dynamic[:self.size]
            template[:self.size] = self.template_of[:self.size]
        self.dynamic, self.template_of = dynamic, template

    def clear(self):
        self.slots.clear(); self.size = 0; self.stats['clears'] += 1

    def slots_for(self, states):
        missing, seen = [], set()
        for s in states:
            entry = self.slots.get(id(s))
            if entry is None or entry[0] is not s:
                if id(s) not in seen:
                    seen.add(id(s)); missing.append(s)
        self.stats['misses'] += len(missing); self.stats['hits'] += len(states) - len(missing)
        if missing:
            if self.size + len(missing) > self.capacity:
                self.clear()
                return self.slots_for(states)
            for s in missing:
                if not isinstance(s, RecoveryState):
                    raise TypeError('candidate state must be a RecoveryState')
            cells = missing[0].rows * missing[0].cols
            if any(s.rows * s.cols != cells for s in missing):
                raise ValueError('mixed node counts')
            self._grow(self.size + len(missing), cells)
            templates = [self._template(s) for s in missing]
            rows = torch.tensor([self._dynamic_rows(s) for s in missing], dtype=self.dtype)
            start = self.size
            self.dynamic[start:start + len(missing)] = rows.to(self.device)
            self.template_of[start:start + len(missing)] = torch.tensor(templates, device=self.device)
            for offset, s in enumerate(missing):
                self.slots[id(s)] = (s, start + offset)
            self.size += len(missing)
        return torch.tensor([self.slots[id(s)][1] for s in states], dtype=torch.long, device=self.device)

    def batch(self, states):
        """Return (features [B,N,F], mask [B,N], edge_index [2,E]) matching the reference."""
        slots = self.slots_for(states)
        template = self.template_of[slots]
        static = torch.stack(self.static)                       # [T, N, F]
        features = static.index_select(0, template).clone()
        features[:, :, DYNAMIC_COLUMNS] = self.dynamic.index_select(0, slots)
        batch, nodes = features.shape[:2]
        mask = torch.ones((batch, nodes), dtype=torch.bool, device=self.device)
        offsets = torch.arange(batch, device=self.device) * nodes
        if len(self.edges) == 1:
            edges = (self.edges[0].unsqueeze(1) + offsets.view(1, -1, 1)).reshape(2, -1)
        else:
            # Graph-major order, as in pad_yard_graphs.
            parts = [self.edges[t] + o for t, o in zip(template.tolist(), offsets.tolist())]
            edges = torch.cat(parts, dim=1)
        return features, mask, edges


_REGISTRY = {}


def _cache_for(network):
    # Inputs are weight-independent, so online and target networks share one cache.
    parameter = next(network.parameters())
    key = (str(parameter.device), parameter.dtype, float(network.config.timing_scale))
    cache = _REGISTRY.get(key)
    if cache is None:
        cache = _REGISTRY[key] = InputCache(parameter.device, parameter.dtype, network.config.timing_scale)
    return cache


def _embed(network, states):
    states = tuple(states)
    if states and all(s.rows == states[0].rows and s.cols == states[0].cols for s in states):
        return network.graph_encoder(*_cache_for(network).batch(states))
    graphs = tuple(encode_recovery_state(s, timing_scale=network.config.timing_scale) for s in states)
    return network.graph_encoder(pad_yard_graphs(graphs))


def _q(network, current_embedding, successor_embedding, action_features):
    return network.q_head(torch.cat((current_embedding, successor_embedding,
                                     successor_embedding - current_embedding,
                                     network.action_encoder(action_features)), dim=-1)).squeeze(-1)


def fast_forward(self, current_states, successor_states, action_features):
    current_states, successor_states = tuple(current_states), tuple(successor_states)
    if not current_states:
        raise ValueError('Q network requires at least one candidate state')
    if len(current_states) != len(successor_states):
        raise ValueError('current and successor state counts must match')
    parameter = next(self.parameters())
    action_features = torch.as_tensor(action_features, dtype=parameter.dtype, device=parameter.device)
    if action_features.ndim == 1:
        action_features = action_features.unsqueeze(0)
    if action_features.shape != (len(current_states), ACTION_FEATURE_DIM):
        raise ValueError(f'action features must have shape [{len(current_states)}, {ACTION_FEATURE_DIM}]')
    embedding = _embed(self, current_states + successor_states)
    n = len(current_states)
    return _q(self, embedding[:n], embedding[n:], action_features)


def fast_deduplicated_no_grad_scores(records, network, audit=None):
    if torch.is_grad_enabled():
        raise RuntimeError('accelerated continuation scoring requires no_grad')
    records = tuple(records)
    if not records:
        raise ValueError('cannot score an empty continuation frontier')
    states = tuple(r.current_state for r in records) + tuple(r.successor_state for r in records)
    unique, index, inverse = [], {}, []
    for s in states:
        i = index.get(id(s))
        if i is None:
            i = index[id(s)] = len(unique); unique.append(s)
        inverse.append(i)
    parameter = next(network.parameters())
    features = torch.as_tensor(tuple(r.action_features for r in records), dtype=parameter.dtype, device=parameter.device)
    if features.shape != (len(records), ACTION_FEATURE_DIM):
        raise RuntimeError('action feature schema changed')
    embedding = _embed(network, unique).index_select(0, torch.tensor(inverse, device=parameter.device))
    n = len(records)
    if audit is not None:
        audit['calls'] = int(audit.get('calls', 0)) + 1
        audit['state_slots'] = int(audit.get('state_slots', 0)) + len(states)
        audit['unique_state_identities'] = int(audit.get('unique_state_identities', 0)) + len(unique)
    return _q(network, embedding[:n], embedding[n:], features)


@contextmanager
def enabled():
    with patch.object(v3, '_deduplicated_no_grad_scores', fast_deduplicated_no_grad_scores), \
         patch.object(ViabilityGraphQNetwork, 'forward', fast_forward):
        yield _REGISTRY


def cache_stats():
    return {f'{k[0]}|{k[1]}|{k[2]}': dict(c.stats, entries=c.size) for k, c in _REGISTRY.items()}
