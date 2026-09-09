"""Counterfactual admission certification from the recovery transition model.

The functions in this module do not rank cells and never query an assignment
policy.  They convert each already-physical ``Accept(block, cell)`` candidate
into its post-decision recovery state and classify that state with
``PSLAP.viability``.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, MutableMapping, Optional

from PSLAP.dynamic_yard import BlockView, Cell, YardSnapshot
from PSLAP.viability import (
    RECOVERY_SEARCH_ORDERS,
    RecoverabilityCertificate,
    RecoveryState,
    ViabilityStatus,
    analyze_recoverability,
)
from PSLAP.viability_prioritizer import (
    RecoveryStatePrioritizer,
    StatePriorityBatch,
    validate_priority_batch,
)


ROBUST_QUEUE_OBSTACLE_CONTRACT = (
    "online_no_future_schedule_pickup_and_waiting_cells_reserved_as_"
    "worst_case_fixed_obstacles_v1"
)
CURRENT_LIVE_OBSTACLE_CONTRACT = (
    "online_no_future_schedule_current_live_nonstored_obstacles_v1"
)
POST_ACCEPT_CERTIFICATION_CONTRACT = (
    "physical_accept_successor_closed_admission_recoverability_v1"
)


@dataclass(frozen=True)
class ViabilitySearchConfig:
    """Policy-independent resource and disturbance contract for certification."""

    max_depth: Optional[int] = None
    max_nodes: Optional[int] = 100_000
    max_primitive_steps: Optional[int] = None
    reserve_queue_cells: bool = True
    search_order: str = "goal_directed"

    def __post_init__(self) -> None:
        for name, value, allow_zero in (
            ("max_depth", self.max_depth, True),
            ("max_nodes", self.max_nodes, False),
            ("max_primitive_steps", self.max_primitive_steps, True),
        ):
            if value is None:
                continue
            minimum = 0 if allow_zero else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be a {qualifier} integer or None")
        if self.search_order not in RECOVERY_SEARCH_ORDERS:
            raise ValueError(
                f"search_order must be one of {RECOVERY_SEARCH_ORDERS!r}"
            )

    @property
    def fixed_obstacle_contract(self) -> str:
        return (
            ROBUST_QUEUE_OBSTACLE_CONTRACT
            if self.reserve_queue_cells
            else CURRENT_LIVE_OBSTACLE_CONTRACT
        )


@dataclass(frozen=True)
class CandidateCertificate:
    cell: Cell
    recovery_state: RecoveryState
    certificate: RecoverabilityCertificate
    cache_hit: bool = False

    @property
    def status(self) -> ViabilityStatus:
        return self.certificate.status


@dataclass(frozen=True)
class CandidateViabilityReport:
    """Complete, order-preserving classification of one physical cell set."""

    candidates: tuple[CandidateCertificate, ...]
    analysis_seconds: float
    fixed_obstacle_contract: str
    certification_contract: str = POST_ACCEPT_CERTIFICATION_CONTRACT
    exact_analysis_seconds: float = 0.0
    priority_batch: Optional[StatePriorityBatch] = None

    def cells_with_status(self, status: ViabilityStatus) -> tuple[Cell, ...]:
        return tuple(item.cell for item in self.candidates if item.status is status)

    @property
    def safe_cells(self) -> tuple[Cell, ...]:
        return self.cells_with_status(ViabilityStatus.SAFE)

    @property
    def unsafe_cells(self) -> tuple[Cell, ...]:
        return self.cells_with_status(ViabilityStatus.UNSAFE)

    @property
    def unknown_cells(self) -> tuple[Cell, ...]:
        return self.cells_with_status(ViabilityStatus.UNKNOWN)

    def audit_dict(self) -> dict:
        certificates = tuple(item.certificate for item in self.candidates)
        return {
            "certification_contract": self.certification_contract,
            "fixed_obstacle_contract": self.fixed_obstacle_contract,
            "physical_accept_count": len(self.candidates),
            "safe_accept_count": len(self.safe_cells),
            "unsafe_accept_count": len(self.unsafe_cells),
            "unknown_accept_count": len(self.unknown_cells),
            "rejected_accept_count": len(self.unsafe_cells)
            + len(self.unknown_cells),
            "cache_hits": sum(item.cache_hit for item in self.candidates),
            "cache_misses": sum(not item.cache_hit for item in self.candidates),
            "explored_nodes": sum(item.explored_nodes for item in certificates),
            "generated_states": sum(
                item.generated_states for item in certificates
            ),
            "exhaustive_certificates": sum(
                item.exhaustive for item in certificates
            ),
            "analysis_seconds": float(self.analysis_seconds),
            "exact_analysis_seconds": float(self.exact_analysis_seconds),
            "certification_order": (
                "canonical_exact_full_v1"
                if self.priority_batch is None
                else self.priority_batch.protocol
            ),
            "priority_states_scored": (
                0 if self.priority_batch is None else len(self.priority_batch.entries)
            ),
            "priority_pass_count": (
                0
                if self.priority_batch is None
                else self.priority_batch.priority_pass_count
            ),
            "priority_inference_seconds": (
                0.0
                if self.priority_batch is None
                else float(self.priority_batch.inference_seconds)
            ),
            "priority_checkpoint_sha256": (
                None
                if self.priority_batch is None
                else self.priority_batch.checkpoint_sha256
            ),
            "complete_frontier_exactly_verified": True,
        }


CertificateCacheKey = tuple[
    RecoveryState,
    Optional[int],
    Optional[int],
    Optional[int],
    str,
]


def online_fixed_obstacles(
    env,
    *,
    exclude_labels: Iterable[str] = (),
    reserve_queue_cells: bool = True,
) -> frozenset[Cell]:
    """Return online-visible nonstored obstacles plus an optional queue envelope.

    Reserving pickup/waiting cells is a structural worst-case assumption.  It
    uses no future arrival times or processing durations and prevents an
    in-macro queue promotion from silently invalidating a geometric proof.
    """

    excluded = frozenset(str(label) for label in exclude_labels)
    cells = {
        tuple(block.position)
        for block in env.blocks
        if (
            block.label not in excluded
            and block.position is not None
            and not block.stored
            and not block.delivered
            and not block.carrying
        )
    }
    if reserve_queue_cells:
        cells.update((tuple(env.pickup_cell), tuple(env.waiting_cell)))
    traversable = {
        (row, col)
        for row in range(env.grid_rows)
        for col in range(env.grid_cols)
        if env.rooms[row, col] != "#"
    }
    return frozenset(cell for cell in cells if cell in traversable)


def post_accept_recovery_state(
    env,
    yard: YardSnapshot,
    inbound: BlockView,
    cell: Cell,
    *,
    reserve_queue_cells: bool = True,
    reserved_cells: Iterable[Cell] = (),
) -> RecoveryState:
    """Construct the strict macro-boundary successor of ``Accept(inbound, cell)``.

    Physical executability of the Accept macro remains the caller's
    responsibility.  This function certifies the counterfactual state after a
    successful putdown and therefore places the agent and inbound block at the
    selected cell.
    """

    cell = tuple(cell)
    if cell not in yard.storage_cells:
        raise ValueError("post-accept cell must be a storage cell")
    if cell in yard.occupancy():
        raise ValueError("post-accept cell must be unoccupied")
    post_yard = yard.with_block(inbound, cell)
    fixed = online_fixed_obstacles(
        env,
        exclude_labels=(inbound.label,),
        reserve_queue_cells=reserve_queue_cells,
    )
    return RecoveryState.from_yard_snapshot(
        post_yard,
        agent_position=cell,
        fixed_obstacles=fixed,
        reserved_cells=reserved_cells,
        pickup_cells=(tuple(env.pickup_cell),),
        wait_cells=(tuple(env.waiting_cell),),
    )


def certify_post_accept_candidates(
    env,
    yard: YardSnapshot,
    inbound: BlockView,
    candidates: Iterable[Cell],
    *,
    config: ViabilitySearchConfig = ViabilitySearchConfig(),
    reserved_cells: Iterable[Cell] = (),
    cache: Optional[
        MutableMapping[CertificateCacheKey, RecoverabilityCertificate]
    ] = None,
    state_prioritizer: Optional[RecoveryStatePrioritizer] = None,
) -> CandidateViabilityReport:
    """Classify every physical Accept cell under exact-verifier authority.

    A supplied learned prioritizer may permute *uncached* exact checks.  Every
    cell is still certified, and results are restored to the caller's original
    order before the report is constructed.
    """

    cells = tuple(tuple(cell) for cell in candidates)
    if len(cells) != len(set(cells)):
        raise ValueError("physical accept candidates must be unique")
    started = perf_counter()
    states = tuple(
        post_accept_recovery_state(
            env,
            yard,
            inbound,
            cell,
            reserve_queue_cells=config.reserve_queue_cells,
            reserved_cells=reserved_cells,
        )
        for cell in cells
    )
    results: list[Optional[CandidateCertificate]] = [None] * len(cells)
    missing_indices = []
    for index, (cell, state) in enumerate(zip(cells, states)):
        key = (
            state,
            config.max_depth,
            config.max_nodes,
            config.max_primitive_steps,
            config.search_order,
        )
        cached = None if cache is None else cache.get(key)
        if cached is None:
            missing_indices.append(index)
        else:
            results[index] = CandidateCertificate(
                cell=cell,
                recovery_state=state,
                certificate=cached,
                cache_hit=True,
            )

    priority_batch = None
    ordered_missing = tuple(missing_indices)
    if state_prioritizer is not None and missing_indices:
        priority_request = tuple(
            (
                f"accept:{inbound.label}:{cells[index][0]}:{cells[index][1]}",
                states[index],
            )
            for index in missing_indices
        )
        priority_batch = validate_priority_batch(
            state_prioritizer.prioritize(priority_request),
            priority_request,
            prioritizer=state_prioritizer,
        )
        ordered_missing = tuple(
            missing_indices[index] for index in priority_batch.ordered_indices
        )

    exact_seconds = 0.0
    for index in ordered_missing:
        cell = cells[index]
        state = states[index]
        key = (
            state,
            config.max_depth,
            config.max_nodes,
            config.max_primitive_steps,
            config.search_order,
        )
        exact_started = perf_counter()
        certificate = analyze_recoverability(
            state,
            max_depth=config.max_depth,
            max_nodes=config.max_nodes,
            max_primitive_steps=config.max_primitive_steps,
            search_order=config.search_order,
        )
        exact_seconds += perf_counter() - exact_started
        if cache is not None:
            cache[key] = certificate
        results[index] = CandidateCertificate(
            cell=cell,
            recovery_state=state,
            certificate=certificate,
            cache_hit=False,
        )
    if any(item is None for item in results):  # pragma: no cover - defensive
        raise RuntimeError("post-accept certification left an unresolved state")
    return CandidateViabilityReport(
        candidates=tuple(results),  # type: ignore[arg-type]
        analysis_seconds=perf_counter() - started,
        fixed_obstacle_contract=config.fixed_obstacle_contract,
        exact_analysis_seconds=exact_seconds,
        priority_batch=priority_batch,
    )


__all__ = [
    "CURRENT_LIVE_OBSTACLE_CONTRACT",
    "POST_ACCEPT_CERTIFICATION_CONTRACT",
    "ROBUST_QUEUE_OBSTACLE_CONTRACT",
    "CandidateCertificate",
    "CandidateViabilityReport",
    "ViabilitySearchConfig",
    "certify_post_accept_candidates",
    "online_fixed_obstacles",
    "post_accept_recovery_state",
]
