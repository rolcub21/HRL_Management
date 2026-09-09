"""Policy-independent graph observations for recovery states.

The graph in this module is a representation of physical state, not a
storage policy.  It deliberately contains no urgency ordering, assignment
score, hand-written route-preservation rule, or baseline recommendation.

Every in-bounds grid cell is represented by one node.  Walls therefore remain
observable as isolated nodes with ``traversable == 0``; sparse directed edges
connect orthogonally adjacent traversable cells.  Dynamic obstacles and
reservations are node attributes rather than silently changing the static
yard topology.  Encoding an already constructed counterfactual
:class:`~PSLAP.viability.RecoveryState` consequently uses exactly the same
code path as encoding the current state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence, Union

import torch
from torch import Tensor, nn

from PSLAP.dynamic_yard import Cell
from PSLAP.viability import RecoveryState


NODE_FEATURE_NAMES = (
    # Geometry.  Coordinates are normalized only by the rectangular extent;
    # they do not encode a preferred door, route, or storage ranking.
    "row_fraction",
    "column_fraction",
    "traversable",
    "storage",
    "exit",
    "pickup",
    "wait",
    # Current/counterfactual physical state.
    "agent",
    "occupied",
    "fixed_obstacle",
    "reserved",
    # Occupant timing is explicitly masked when it is unavailable.
    "occupant_time_known",
    "occupant_remaining_time_scaled",
    "occupant_abs_remaining_time_scaled",
    "occupant_overdue",
)

NODE_FEATURE_INDEX = {
    name: index for index, name in enumerate(NODE_FEATURE_NAMES)
}
NODE_FEATURE_DIM = len(NODE_FEATURE_NAMES)


def _coerce_cells(
    cells: Iterable[Cell] | Cell,
    *,
    name: str,
) -> frozenset[Cell]:
    """Accept either one ``(row, column)`` cell or an iterable of cells."""

    values = tuple(cells)
    if len(values) == 2 and all(isinstance(value, int) for value in values):
        values = (values,)

    normalized = []
    for value in values:
        try:
            row, column = value
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain (row, column) cells") from error
        if not isinstance(row, int) or not isinstance(column, int):
            raise ValueError(f"{name} coordinates must be integers")
        normalized.append((row, column))
    return frozenset(normalized)


def _validate_in_bounds(
    cells: Iterable[Cell],
    *,
    rows: int,
    cols: int,
    name: str,
) -> None:
    invalid = tuple(
        sorted(
            cell
            for cell in cells
            if not (0 <= cell[0] < rows and 0 <= cell[1] < cols)
        )
    )
    if invalid:
        raise ValueError(f"{name} contains out-of-bounds cells: {invalid}")


@dataclass(frozen=True)
class YardGraph:
    """Immutable variable-size cell graph represented by plain Python data.

    ``edge_index`` follows the common sparse COO convention: its first tuple
    contains source indices and its second tuple contains destination indices.
    Both directions of every traversable undirected grid edge are present.
    """

    cells: tuple[Cell, ...]
    node_features: tuple[tuple[float, ...], ...]
    edge_index: tuple[tuple[int, ...], tuple[int, ...]]
    feature_names: tuple[str, ...] = NODE_FEATURE_NAMES

    def __post_init__(self) -> None:
        if len(self.cells) != len(self.node_features):
            raise ValueError("cells and node_features must have equal lengths")
        if len(self.cells) != len(set(self.cells)):
            raise ValueError("graph cells must be unique")
        if not self.feature_names:
            raise ValueError("feature_names cannot be empty")
        if any(
            len(features) != len(self.feature_names)
            for features in self.node_features
        ):
            raise ValueError("every node must use the declared feature schema")
        if len(self.edge_index) != 2:
            raise ValueError("edge_index must have shape [2, edge_count]")
        sources, destinations = self.edge_index
        if len(sources) != len(destinations):
            raise ValueError("edge source and destination counts must match")
        if any(
            index < 0 or index >= len(self.cells)
            for index in sources + destinations
        ):
            raise ValueError("edge_index contains an invalid node index")

    @property
    def node_count(self) -> int:
        return len(self.cells)

    @property
    def edge_count(self) -> int:
        return len(self.edge_index[0])

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        """Return sparse edges as source/destination pairs."""

        return tuple(zip(*self.edge_index))

    def index(self, cell: Cell) -> int:
        """Return the deterministic row-major node index for ``cell``."""

        try:
            return self.cells.index(cell)
        except ValueError as error:
            raise KeyError(cell) from error

    def features_for(self, cell: Cell) -> dict[str, float]:
        """Return a named copy of one node's features for audits/tests."""

        features = self.node_features[self.index(cell)]
        return dict(zip(self.feature_names, features))


def encode_recovery_state(
    state: RecoveryState,
    *,
    pickup_cells: Optional[Iterable[Cell] | Cell] = None,
    wait_cells: Optional[Iterable[Cell] | Cell] = None,
    timing_scale: float = 1.0,
) -> YardGraph:
    """Encode a current or counterfactual recovery state as a sparse graph.

    ``timing_scale`` is an explicit units conversion only.  No state-relative
    ranking or normalization is performed, so two identical remaining times
    have identical features across differently populated yards.  Missing,
    non-numeric, or non-finite timing values set ``occupant_time_known`` to
    zero and leave all timing values at zero.
    """

    if not isinstance(state, RecoveryState):
        raise TypeError("encode_recovery_state requires a RecoveryState")
    if (
        isinstance(timing_scale, bool)
        or not isinstance(timing_scale, (int, float))
        or not math.isfinite(float(timing_scale))
        or timing_scale <= 0
    ):
        raise ValueError("timing_scale must be a positive finite number")
    timing_scale = float(timing_scale)

    pickup = (
        state.pickup_cells
        if pickup_cells is None
        else _coerce_cells(pickup_cells, name="pickup_cells")
    )
    wait = (
        state.wait_cells
        if wait_cells is None
        else _coerce_cells(wait_cells, name="wait_cells")
    )
    in_bounds_sets = (
        (state.traversable, "traversable"),
        (state.storage_cells, "storage_cells"),
        (state.exits, "exits"),
        (state.fixed_obstacles, "fixed_obstacles"),
        (state.reserved_cells, "reserved_cells"),
        (pickup, "pickup_cells"),
        (wait, "wait_cells"),
        ((state.agent_position,), "agent_position"),
        ((block.position for block in state.blocks), "block positions"),
    )
    for values, name in in_bounds_sets:
        _validate_in_bounds(
            values,
            rows=state.rows,
            cols=state.cols,
            name=name,
        )

    cells = tuple(
        (row, column)
        for row in range(state.rows)
        for column in range(state.cols)
    )
    cell_index = {cell: index for index, cell in enumerate(cells)}
    occupancy = state.occupancy()
    exits = frozenset(state.exits)
    node_features = []

    for row, column in cells:
        cell = (row, column)
        features = [0.0] * NODE_FEATURE_DIM
        features[NODE_FEATURE_INDEX["row_fraction"]] = row / max(
            state.rows - 1, 1
        )
        features[NODE_FEATURE_INDEX["column_fraction"]] = column / max(
            state.cols - 1, 1
        )
        features[NODE_FEATURE_INDEX["traversable"]] = float(
            cell in state.traversable
        )
        features[NODE_FEATURE_INDEX["storage"]] = float(
            cell in state.storage_cells
        )
        features[NODE_FEATURE_INDEX["exit"]] = float(cell in exits)
        features[NODE_FEATURE_INDEX["pickup"]] = float(cell in pickup)
        features[NODE_FEATURE_INDEX["wait"]] = float(cell in wait)
        features[NODE_FEATURE_INDEX["agent"]] = float(
            cell == state.agent_position
        )
        features[NODE_FEATURE_INDEX["occupied"]] = float(cell in occupancy)
        features[NODE_FEATURE_INDEX["fixed_obstacle"]] = float(
            cell in state.fixed_obstacles
        )
        features[NODE_FEATURE_INDEX["reserved"]] = float(
            cell in state.reserved_cells
        )

        occupant = occupancy.get(cell)
        remaining = getattr(occupant, "remaining_time", None)
        try:
            remaining = float(remaining)
        except (TypeError, ValueError):
            remaining = math.nan
        if math.isfinite(remaining):
            scaled = remaining / timing_scale
            features[NODE_FEATURE_INDEX["occupant_time_known"]] = 1.0
            features[
                NODE_FEATURE_INDEX["occupant_remaining_time_scaled"]
            ] = scaled
            features[
                NODE_FEATURE_INDEX["occupant_abs_remaining_time_scaled"]
            ] = abs(scaled)
            features[NODE_FEATURE_INDEX["occupant_overdue"]] = float(
                remaining <= 0.0
            )
        node_features.append(tuple(features))

    sources = []
    destinations = []
    # Add each undirected edge once, then explicitly materialize both sparse
    # directions.  Iteration order makes graph construction deterministic.
    for row, column in sorted(state.traversable):
        source = cell_index[(row, column)]
        for neighbor in ((row + 1, column), (row, column + 1)):
            if neighbor not in state.traversable:
                continue
            destination = cell_index[neighbor]
            sources.extend((source, destination))
            destinations.extend((destination, source))

    return YardGraph(
        cells=cells,
        node_features=tuple(node_features),
        edge_index=(tuple(sources), tuple(destinations)),
    )


# A short neutral alias is useful at call sites that already name the state.
encode_yard_graph = encode_recovery_state


@dataclass(frozen=True)
class PaddedYardGraphBatch:
    """Pure-Python padded nodes plus one flattened sparse edge list.

    Sparse indices use ``batch_index * max_nodes + local_node_index``.  Padded
    nodes have ``node_mask == False`` and no incident edges.  Keeping this
    container free of tensors makes construction deterministic, serializable,
    and usable in data-loader workers without choosing a device.
    """

    node_features: tuple[tuple[tuple[float, ...], ...], ...]
    node_mask: tuple[tuple[bool, ...], ...]
    cells: tuple[tuple[Optional[Cell], ...], ...]
    edge_index: tuple[tuple[int, ...], tuple[int, ...]]
    node_counts: tuple[int, ...]
    feature_names: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        return len(self.node_features)

    @property
    def max_nodes(self) -> int:
        return len(self.node_features[0]) if self.node_features else 0

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)


def pad_yard_graphs(
    graphs: Sequence[YardGraph],
    *,
    pad_value: float = 0.0,
) -> PaddedYardGraphBatch:
    """Pad variable-size node tables while retaining sparse adjacency."""

    graphs = tuple(graphs)
    if not graphs:
        raise ValueError("at least one YardGraph is required")
    if not all(isinstance(graph, YardGraph) for graph in graphs):
        raise TypeError("pad_yard_graphs accepts only YardGraph instances")
    if any(graph.feature_names != graphs[0].feature_names for graph in graphs):
        raise ValueError("all graphs must use the same feature schema")
    if (
        isinstance(pad_value, bool)
        or not isinstance(pad_value, (int, float))
        or not math.isfinite(float(pad_value))
    ):
        raise ValueError("pad_value must be a finite number")

    maximum = max(graph.node_count for graph in graphs)
    feature_dim = graphs[0].feature_dim
    padding_feature = tuple(float(pad_value) for _ in range(feature_dim))
    padded_features = []
    masks = []
    padded_cells = []
    sources = []
    destinations = []

    for batch_index, graph in enumerate(graphs):
        padding_count = maximum - graph.node_count
        padded_features.append(
            graph.node_features + (padding_feature,) * padding_count
        )
        masks.append(
            (True,) * graph.node_count + (False,) * padding_count
        )
        padded_cells.append(graph.cells + (None,) * padding_count)
        offset = batch_index * maximum
        sources.extend(offset + source for source in graph.edge_index[0])
        destinations.extend(
            offset + destination for destination in graph.edge_index[1]
        )

    return PaddedYardGraphBatch(
        node_features=tuple(padded_features),
        node_mask=tuple(masks),
        cells=tuple(padded_cells),
        edge_index=(tuple(sources), tuple(destinations)),
        node_counts=tuple(graph.node_count for graph in graphs),
        feature_names=graphs[0].feature_names,
    )


def padded_batch_tensors(
    batch: PaddedYardGraphBatch,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor, Tensor]:
    """Materialize a pure batch as features, mask, and sparse edge tensors."""

    if not isinstance(batch, PaddedYardGraphBatch):
        raise TypeError("padded_batch_tensors requires PaddedYardGraphBatch")
    features = torch.tensor(
        batch.node_features,
        dtype=dtype,
        device=device,
    )
    mask = torch.tensor(batch.node_mask, dtype=torch.bool, device=device)
    edge_index = torch.tensor(
        batch.edge_index,
        dtype=torch.long,
        device=device,
    ).reshape(2, -1)
    return features, mask, edge_index


class _SparseMessageLayer(nn.Module):
    """Dependency-free sparse aggregation using ``index_add_``."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # The degree/count channel prevents a mean aggregate from erasing the
        # distinction between a leaf, corridor cell, and high-degree cell.
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden: Tensor,
        valid_mask: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        node_count = hidden.shape[0]
        aggregate = torch.zeros_like(hidden)
        degree = hidden.new_zeros((node_count, 1))
        if edge_index.numel():
            sources, destinations = edge_index
            messages = self.message(hidden.index_select(0, sources))
            aggregate.index_add_(0, destinations, messages)
            degree.index_add_(
                0,
                destinations,
                hidden.new_ones((destinations.numel(), 1)),
            )
        mean_message = aggregate / degree.clamp_min(1.0)
        update = self.update(
            torch.cat((hidden, mean_message, torch.log1p(degree)), dim=-1)
        )
        hidden = self.normalization(hidden + update)
        return hidden * valid_mask.unsqueeze(-1).to(hidden.dtype)


class YardGraphEncoder(nn.Module):
    """Compact sparse message-passing encoder with count-aware pooling.

    Graph pooling concatenates the masked node sum, masked node mean, and
    logarithmic valid-node count.  It is therefore invariant to padding while
    retaining explicit yard/cardinality information that pure mean pooling
    would discard.
    """

    def __init__(
        self,
        input_dim: int = NODE_FEATURE_DIM,
        hidden_dim: int = 64,
        output_dim: int = 64,
        message_passing_steps: int = 2,
    ) -> None:
        super().__init__()
        for value, name in (
            (input_dim, "input_dim"),
            (hidden_dim, "hidden_dim"),
            (output_dim, "output_dim"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(message_passing_steps, bool)
            or not isinstance(message_passing_steps, int)
            or message_passing_steps < 0
        ):
            raise ValueError(
                "message_passing_steps must be a non-negative integer"
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.message_layers = nn.ModuleList(
            _SparseMessageLayer(hidden_dim)
            for _ in range(message_passing_steps)
        )
        self.pool_projection = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, output_dim),
            nn.SiLU(),
            nn.LayerNorm(output_dim),
        )

    def _prepare_inputs(
        self,
        batch_or_features: PaddedYardGraphBatch | Tensor,
        node_mask: Optional[Tensor],
        edge_index: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        parameter = next(self.parameters())
        if isinstance(batch_or_features, PaddedYardGraphBatch):
            if node_mask is not None or edge_index is not None:
                raise ValueError(
                    "node_mask/edge_index must be omitted for a padded batch"
                )
            features, node_mask, edge_index = padded_batch_tensors(
                batch_or_features,
                device=parameter.device,
                dtype=parameter.dtype,
            )
        elif torch.is_tensor(batch_or_features):
            features = batch_or_features.to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            if features.ndim == 2:
                features = features.unsqueeze(0)
            if features.ndim != 3:
                raise ValueError("node_features must have shape [B, N, F]")
            if node_mask is None:
                node_mask = torch.ones(
                    features.shape[:2],
                    dtype=torch.bool,
                    device=features.device,
                )
            else:
                node_mask = node_mask.to(
                    device=features.device,
                    dtype=torch.bool,
                )
            if edge_index is None:
                edge_index = torch.empty(
                    (2, 0), dtype=torch.long, device=features.device
                )
            else:
                edge_index = edge_index.to(
                    device=features.device,
                    dtype=torch.long,
                )
        else:
            raise TypeError(
                "encoder input must be PaddedYardGraphBatch or Tensor"
            )

        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected {self.input_dim} node features, "
                f"received {features.shape[-1]}"
            )
        if node_mask.shape != features.shape[:2]:
            raise ValueError("node_mask must have shape [B, N]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        flattened_nodes = features.shape[0] * features.shape[1]
        if edge_index.numel() and (
            int(edge_index.min()) < 0
            or int(edge_index.max()) >= flattened_nodes
        ):
            raise ValueError("edge_index contains an invalid flattened index")
        if not torch.all(node_mask.any(dim=1)):
            raise ValueError("every graph must contain at least one valid node")
        return features, node_mask, edge_index

    def encode_nodes(
        self,
        batch_or_features: PaddedYardGraphBatch | Tensor,
        node_mask: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Return padded node embeddings and their validity mask."""

        features, node_mask, edge_index = self._prepare_inputs(
            batch_or_features, node_mask, edge_index
        )
        batch_size, maximum, _ = features.shape
        hidden = self.input_projection(features.reshape(-1, self.input_dim))
        valid = node_mask.reshape(-1)
        hidden = hidden * valid.unsqueeze(-1).to(hidden.dtype)
        for layer in self.message_layers:
            hidden = layer(hidden, valid, edge_index)
        return hidden.reshape(batch_size, maximum, self.hidden_dim), node_mask

    def forward(
        self,
        batch_or_features: PaddedYardGraphBatch | Tensor,
        node_mask: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
    ) -> Tensor:
        hidden, valid = self.encode_nodes(
            batch_or_features, node_mask, edge_index
        )
        weights = valid.unsqueeze(-1).to(hidden.dtype)
        node_sum = (hidden * weights).sum(dim=1)
        count = weights.sum(dim=1).clamp_min(1.0)
        node_mean = node_sum / count
        pooled = torch.cat((node_sum, node_mean, torch.log1p(count)), dim=-1)
        return self.pool_projection(pooled)


# Explicit semantic alias for code that treats this as a recovery component.
RecoveryGraphEncoder = YardGraphEncoder


__all__ = [
    "NODE_FEATURE_DIM",
    "NODE_FEATURE_INDEX",
    "NODE_FEATURE_NAMES",
    "PaddedYardGraphBatch",
    "RecoveryGraphEncoder",
    "YardGraph",
    "YardGraphEncoder",
    "encode_recovery_state",
    "encode_yard_graph",
    "pad_yard_graphs",
    "padded_batch_tensors",
]
