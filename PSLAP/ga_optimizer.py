"""Schedule-level genetic search for a Park--Seo-inspired PSLAP baseline.

The paper establishes the obstruction-minimization objective and finite known
schedule. Exact source code and every evolutionary-operator setting are not
publicly available, so operator choices in this module are explicit reproduction
choices rather than claims about the authors' original implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
import random
from typing import Callable, Iterable, Optional

from PSLAP.dynamic_yard import (
    BlockView,
    Cell,
    YardSnapshot,
    find_min_obstruction_path,
    shortest_clear_path,
)


Chromosome = tuple[Cell, ...]


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 30
    generations: int = 30
    elite_count: int = 2
    tournament_size: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    seed: int = 0

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.generations < 1:
            raise ValueError("generations must be positive")
        if not 1 <= self.elite_count < self.population_size:
            raise ValueError("elite_count must be in [1, population_size)")
        if not 2 <= self.tournament_size <= self.population_size:
            raise ValueError("invalid tournament_size")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1]")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1]")


@dataclass(frozen=True)
class AssignmentCost:
    infeasible_events: int
    obstructive_moves: int
    route_steps: int

    @property
    def scalar(self) -> int:
        """Lexicographic minimization encoded as one deterministic integer."""

        return (
            self.infeasible_events * 10**9
            + self.obstructive_moves * 10**6
            + self.route_steps
        )


@dataclass(frozen=True)
class DurationAwareAssignmentCost:
    """Lexicographic online cost for the improved rolling-GA ablation.

    Deadline ranks weight obstruction and egress duration, while inbound
    travel remains an unweighted secondary operating cost. This is deliberately
    a separately named extension, not an attribution to Park--Seo (2009).
    """

    infeasible_events: int
    priority_obstruction_cost: int
    weighted_route_steps: int
    raw_obstructive_moves: int
    raw_route_steps: int

    @property
    def scalar(self) -> int:
        return (
            self.infeasible_events * 10**15
            + self.priority_obstruction_cost * 10**10
            + self.weighted_route_steps * 10**4
            + self.raw_obstructive_moves * 10**2
            + self.raw_route_steps
        )


@dataclass(frozen=True)
class GAResult:
    best_assignment: Chromosome
    best_cost: AssignmentCost
    best_cost_history: tuple[int, ...]
    mean_cost_history: tuple[float, ...]
    config: GAConfig


def _minimum_obstruction_route(
    yard: YardSnapshot,
    starts: Iterable[Cell],
    goals: Iterable[Cell],
    occupancy: dict[Cell, str],
    *,
    ignore_label: Optional[str] = None,
) -> Optional[tuple[int, int]]:
    """Return minimum distinct blockers, then path length, across endpoints."""

    goal_set = frozenset(goals)
    queue = []
    serial = 0
    best: dict[Cell, tuple[int, int]] = {}
    for start in starts:
        if start not in yard.traversable:
            continue
        best[start] = (0, 0)
        heappush(queue, (0, 0, serial, start, frozenset()))
        serial += 1

    while queue:
        count, distance, _, cell, seen_labels = heappop(queue)
        if best.get(cell) != (count, distance):
            continue
        if cell in goal_set:
            return count, distance
        row, col = cell
        for neighbor in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if neighbor not in yard.traversable:
                continue
            labels = seen_labels
            occupant = occupancy.get(neighbor)
            if occupant is not None and occupant != ignore_label:
                labels = seen_labels | {occupant}
            cost = (len(labels), distance + 1)
            if cost >= best.get(neighbor, (10**9, 10**9)):
                continue
            best[neighbor] = cost
            heappush(queue, (cost[0], cost[1], serial, neighbor, labels))
            serial += 1
    return None


def evaluate_schedule_assignment(env, chromosome: Chromosome) -> AssignmentCost:
    """Evaluate overlapping inventory without executing or mutating the env.

    Outbound events precede inbound events at the same discrete time, matching
    the paper's period-level operating order. Obstructing blocks are counted and
    conceptually restored, so only scheduled inbound/outbound events alter the
    static occupancy used by this fitness model.
    """

    if len(chromosome) != len(env.blocks):
        raise ValueError("chromosome length must equal the number of blocks")
    if len(set(chromosome)) != len(chromosome):
        raise ValueError("chromosome storage cells must be unique")
    if any(cell not in env.storage_positions for cell in chromosome):
        raise ValueError("chromosome contains a non-storage cell")

    yard = YardSnapshot.from_env(env)
    events = []
    for index, (block, cell) in enumerate(zip(env.blocks, chromosome)):
        arrival = int(block.arrival_step)
        departure = arrival + int(block.storage_steps_needed)
        events.append((arrival, 1, index, "inbound", block.label, cell))
        events.append((departure, 0, index, "outbound", block.label, cell))
    events.sort()

    occupancy: dict[Cell, str] = {}
    infeasible = 0
    obstructions = 0
    distance = 0
    for _, _, _, event, label, cell in events:
        if event == "outbound":
            if occupancy.get(cell) != label:
                infeasible += 1
                continue
            route = _minimum_obstruction_route(
                yard,
                (cell,),
                yard.exits,
                occupancy,
                ignore_label=label,
            )
            if route is None:
                infeasible += 1
            else:
                obstructions += route[0]
                distance += route[1]
            occupancy.pop(cell, None)
            continue

        if cell in occupancy:
            infeasible += 1
            continue
        route = _minimum_obstruction_route(
            yard,
            (env.pickup_cell,),
            (cell,),
            occupancy,
        )
        if route is None:
            infeasible += 1
            continue
        obstructions += route[0]
        distance += route[1]
        occupancy[cell] = label

    return AssignmentCost(infeasible, obstructions, distance)


def evaluate_rolling_assignment(
    yard: YardSnapshot,
    pending_blocks: tuple[BlockView, ...],
    chromosome: Chromosome,
    *,
    source: Cell,
) -> AssignmentCost:
    """Score assignments using only inventory known at the current epoch.

    Existing stored blocks remain fixed. Arrived-but-unassigned blocks are
    placed in FIFO order, then every known block's minimum-obstruction egress is
    evaluated. No future arrival or duration is accepted by this interface.
    """

    if len(chromosome) != len(pending_blocks):
        raise ValueError("chromosome length must match known pending blocks")
    if len(set(chromosome)) != len(chromosome):
        raise ValueError("chromosome storage cells must be unique")
    if any(cell not in yard.storage_cells for cell in chromosome):
        raise ValueError("chromosome contains a non-storage cell")

    working = yard
    infeasible = 0
    distance = 0
    for block, cell in zip(pending_blocks, chromosome):
        if working.occupancy().get(cell) is not None:
            infeasible += 1
            continue
        route = shortest_clear_path(working, source, cell)
        if route is None:
            infeasible += 1
            continue
        distance += len(route) - 1
        working = working.with_block(block, cell)

    obstructions = 0
    for block in working.blocks:
        route = find_min_obstruction_path(working, block.label)
        if route is None:
            infeasible += 1
            continue
        obstructions += route.obstruction_count
        distance += len(route.cells) - 1

    return AssignmentCost(infeasible, obstructions, distance)


def evaluate_duration_aware_rolling_assignment(
    yard: YardSnapshot,
    pending_blocks: tuple[BlockView, ...],
    chromosome: Chromosome,
    *,
    source: Cell,
    egress_weight: int = 4,
) -> DurationAwareAssignmentCost:
    """Score an online assignment by urgency-weighted retrieval exposure.

    Only currently stored and arrived-pending blocks are visible. Earlier
    remaining times receive larger integer ranks. The egress multiplier breaks
    the common inbound-plus-outbound distance tie that otherwise makes a cell
    near the pickup look equivalent to a cell near the delivery gate.
    """

    if not isinstance(egress_weight, int) or egress_weight < 1:
        raise ValueError("egress_weight must be a positive integer")
    if len(chromosome) != len(pending_blocks):
        raise ValueError("chromosome length must match known pending blocks")
    if len(set(chromosome)) != len(chromosome):
        raise ValueError("chromosome storage cells must be unique")
    if any(cell not in yard.storage_cells for cell in chromosome):
        raise ValueError("chromosome contains a non-storage cell")

    working = yard
    infeasible = 0
    inbound_steps = 0
    for block, cell in zip(pending_blocks, chromosome):
        if working.occupancy().get(cell) is not None:
            infeasible += 1
            continue
        route = shortest_clear_path(working, source, cell)
        if route is None:
            infeasible += 1
            continue
        inbound_steps += len(route) - 1
        working = working.with_block(block, cell)

    ordered = sorted(
        working.blocks, key=lambda item: (item.remaining_time, item.label)
    )
    block_count = len(ordered)
    priority_obstructions = 0
    weighted_egress = 0
    raw_obstructions = 0
    raw_egress = 0
    for rank, block in enumerate(ordered):
        route = find_min_obstruction_path(working, block.label)
        if route is None:
            infeasible += 1
            continue
        priority = block_count - rank
        egress_steps = len(route.cells) - 1
        priority_obstructions += priority * route.obstruction_count
        weighted_egress += priority * egress_steps
        raw_obstructions += route.obstruction_count
        raw_egress += egress_steps

    return DurationAwareAssignmentCost(
        infeasible_events=infeasible,
        priority_obstruction_cost=priority_obstructions,
        weighted_route_steps=inbound_steps + egress_weight * weighted_egress,
        raw_obstructive_moves=raw_obstructions,
        raw_route_steps=inbound_steps + raw_egress,
    )


def evaluate_operational_rolling_assignment(
    yard: YardSnapshot,
    pending_blocks: tuple[BlockView, ...],
    chromosome: Chromosome,
    *,
    source: Cell,
    egress_weight: int = 4,
) -> DurationAwareAssignmentCost:
    """Protect all known access while emphasizing assigned-block egress.

    This v2 extension avoids letting the largely fixed route lengths of the
    whole stored inventory dominate the cell choice for the current inbound
    block. Existing blocks still contribute urgency-weighted obstruction risk
    and unweighted route cost.
    """

    if not isinstance(egress_weight, int) or egress_weight < 1:
        raise ValueError("egress_weight must be a positive integer")
    if len(chromosome) != len(pending_blocks):
        raise ValueError("chromosome length must match known pending blocks")
    if len(set(chromosome)) != len(chromosome):
        raise ValueError("chromosome storage cells must be unique")
    if any(cell not in yard.storage_cells for cell in chromosome):
        raise ValueError("chromosome contains a non-storage cell")

    working = yard
    infeasible = 0
    inbound_steps = 0
    for block, cell in zip(pending_blocks, chromosome):
        if working.occupancy().get(cell) is not None:
            infeasible += 1
            continue
        route = shortest_clear_path(working, source, cell)
        if route is None:
            infeasible += 1
            continue
        inbound_steps += len(route) - 1
        working = working.with_block(block, cell)

    ordered = sorted(
        working.blocks, key=lambda item: (item.remaining_time, item.label)
    )
    pending_labels = {block.label for block in pending_blocks}
    pending_order = {
        block.label: len(pending_blocks) - rank
        for rank, block in enumerate(
            sorted(
                pending_blocks,
                key=lambda item: (item.remaining_time, item.label),
            )
        )
    }
    block_count = len(ordered)
    priority_obstructions = 0
    weighted_egress = 0
    raw_obstructions = 0
    raw_egress = 0
    for rank, block in enumerate(ordered):
        route = find_min_obstruction_path(working, block.label)
        if route is None:
            infeasible += 1
            continue
        urgency_priority = block_count - rank
        egress_steps = len(route.cells) - 1
        priority_obstructions += urgency_priority * route.obstruction_count
        multiplier = (
            egress_weight * pending_order[block.label]
            if block.label in pending_labels
            else 1
        )
        weighted_egress += multiplier * egress_steps
        raw_obstructions += route.obstruction_count
        raw_egress += egress_steps

    return DurationAwareAssignmentCost(
        infeasible_events=infeasible,
        priority_obstruction_cost=priority_obstructions,
        weighted_route_steps=inbound_steps + weighted_egress,
        raw_obstructive_moves=raw_obstructions,
        raw_route_steps=inbound_steps + raw_egress,
    )


def _repair(
    genes: list[Cell], storage_cells: tuple[Cell, ...], rng: random.Random
) -> Chromosome:
    used = set()
    missing = [cell for cell in storage_cells if cell not in genes]
    rng.shuffle(missing)
    repaired = []
    for gene in genes:
        if gene in used:
            gene = missing.pop()
        repaired.append(gene)
        used.add(gene)
    return tuple(repaired)


def _crossover(
    first: Chromosome,
    second: Chromosome,
    storage_cells: tuple[Cell, ...],
    rng: random.Random,
) -> tuple[Chromosome, Chromosome]:
    if len(first) < 2:
        return first, second
    cut = rng.randrange(1, len(first))
    child_a = _repair(list(first[:cut] + second[cut:]), storage_cells, rng)
    child_b = _repair(list(second[:cut] + first[cut:]), storage_cells, rng)
    return child_a, child_b


def _mutate(
    chromosome: Chromosome,
    storage_cells: tuple[Cell, ...],
    mutation_rate: float,
    rng: random.Random,
) -> Chromosome:
    genes = list(chromosome)
    if rng.random() >= mutation_rate or not genes:
        return chromosome
    if len(genes) > 1 and rng.random() < 0.5:
        first, second = rng.sample(range(len(genes)), 2)
        genes[first], genes[second] = genes[second], genes[first]
    else:
        unused = [cell for cell in storage_cells if cell not in genes]
        if unused:
            genes[rng.randrange(len(genes))] = rng.choice(unused)
    return tuple(genes)


def _run_assignment_ga(
    storage_cells: tuple[Cell, ...],
    chromosome_length: int,
    evaluate: Callable[[Chromosome], AssignmentCost],
    config: GAConfig,
) -> GAResult:
    config.validate()
    if chromosome_length < 1:
        raise ValueError("chromosome_length must be positive")
    if len(storage_cells) < chromosome_length:
        raise ValueError("not enough storage cells for a unique assignment")

    rng = random.Random(config.seed)
    population = [
        tuple(rng.sample(storage_cells, chromosome_length))
        for _ in range(config.population_size)
    ]
    cost_cache: dict[Chromosome, AssignmentCost] = {}

    def cost(chromosome: Chromosome) -> AssignmentCost:
        if chromosome not in cost_cache:
            cost_cache[chromosome] = evaluate(chromosome)
        return cost_cache[chromosome]

    best_history = []
    mean_history = []
    for _ in range(config.generations):
        ranked = sorted(population, key=lambda item: cost(item).scalar)
        scalar_costs = [cost(item).scalar for item in ranked]
        best_history.append(scalar_costs[0])
        mean_history.append(sum(scalar_costs) / len(scalar_costs))
        next_population = list(ranked[: config.elite_count])

        def tournament() -> Chromosome:
            competitors = rng.sample(population, config.tournament_size)
            return min(competitors, key=lambda item: cost(item).scalar)

        while len(next_population) < config.population_size:
            parent_a = tournament()
            parent_b = tournament()
            if rng.random() < config.crossover_rate:
                child_a, child_b = _crossover(
                    parent_a, parent_b, storage_cells, rng
                )
            else:
                child_a, child_b = parent_a, parent_b
            next_population.append(
                _mutate(child_a, storage_cells, config.mutation_rate, rng)
            )
            if len(next_population) < config.population_size:
                next_population.append(
                    _mutate(child_b, storage_cells, config.mutation_rate, rng)
                )
        population = next_population

    best = min(population, key=lambda item: cost(item).scalar)
    return GAResult(
        best_assignment=best,
        best_cost=cost(best),
        best_cost_history=tuple(best_history),
        mean_cost_history=tuple(mean_history),
        config=config,
    )


def _run_singleton_complete_search(
    storage_cells: tuple[Cell, ...],
    evaluate: Callable[[Chromosome], object],
    config: GAConfig,
) -> GAResult:
    """Complete a one-gene search that a GA can otherwise sample sparsely."""

    config.validate()
    if not storage_cells:
        raise ValueError("singleton search requires at least one storage cell")
    population = [(cell,) for cell in storage_cells]
    costs = {chromosome: evaluate(chromosome) for chromosome in population}
    best = min(population, key=lambda item: costs[item].scalar)
    scalars = [cost.scalar for cost in costs.values()]
    return GAResult(
        best_assignment=best,
        best_cost=costs[best],
        best_cost_history=(int(costs[best].scalar),),
        mean_cost_history=(float(sum(scalars) / len(scalars)),),
        config=config,
    )


def optimize_schedule_assignment(
    env, config: GAConfig = GAConfig()
) -> GAResult:
    """Run deterministic elitist tournament GA on the current known schedule."""

    storage_cells = tuple(sorted(env.storage_positions))
    return _run_assignment_ga(
        storage_cells,
        len(env.blocks),
        lambda chromosome: evaluate_schedule_assignment(env, chromosome),
        config,
    )


def optimize_rolling_assignment(
    yard: YardSnapshot,
    pending_blocks: Iterable[BlockView],
    *,
    source: Cell,
    config: GAConfig = GAConfig(),
    storage_cells: Optional[Iterable[Cell]] = None,
) -> GAResult:
    """Optimize only blocks observable at the current online decision epoch.

    ``storage_cells`` lets a neutral evaluator impose its shared admissibility
    mask without changing the GA fitness.  The default retains the historical
    rolling-GA behavior of searching every currently free storage cell.
    """

    pending_blocks = tuple(pending_blocks)
    occupied = frozenset(yard.occupancy())
    free_cells = yard.storage_cells - occupied
    if storage_cells is None:
        storage_cells = tuple(sorted(free_cells))
    else:
        storage_cells = tuple(sorted(set(storage_cells)))
        if any(cell not in free_cells for cell in storage_cells):
            raise ValueError(
                "rolling-GA search domain contains an occupied or non-storage cell"
            )
    return _run_assignment_ga(
        storage_cells,
        len(pending_blocks),
        lambda chromosome: evaluate_rolling_assignment(
            yard,
            pending_blocks,
            chromosome,
            source=source,
        ),
        config,
    )


def optimize_duration_aware_rolling_assignment(
    yard: YardSnapshot,
    pending_blocks: Iterable[BlockView],
    *,
    source: Cell,
    config: GAConfig = GAConfig(),
    storage_cells: Optional[Iterable[Cell]] = None,
    egress_weight: int = 4,
    exhaustive_singleton: bool = False,
) -> GAResult:
    """Run the separately identified urgency/egress-aware rolling GA."""

    pending_blocks = tuple(pending_blocks)
    occupied = frozenset(yard.occupancy())
    free_cells = yard.storage_cells - occupied
    if storage_cells is None:
        storage_cells = tuple(sorted(free_cells))
    else:
        storage_cells = tuple(sorted(set(storage_cells)))
        if any(cell not in free_cells for cell in storage_cells):
            raise ValueError(
                "rolling-GA search domain contains an occupied or non-storage cell"
            )
    evaluate = lambda chromosome: evaluate_duration_aware_rolling_assignment(
            yard,
            pending_blocks,
            chromosome,
            source=source,
            egress_weight=egress_weight,
        )
    if exhaustive_singleton and len(pending_blocks) == 1:
        return _run_singleton_complete_search(storage_cells, evaluate, config)
    return _run_assignment_ga(
        storage_cells, len(pending_blocks), evaluate, config
    )


def optimize_operational_rolling_assignment(
    yard: YardSnapshot,
    pending_blocks: Iterable[BlockView],
    *,
    source: Cell,
    config: GAConfig = GAConfig(),
    storage_cells: Optional[Iterable[Cell]] = None,
    egress_weight: int = 4,
) -> GAResult:
    """Run the v2 current-assignment-focused rolling GA extension."""

    pending_blocks = tuple(pending_blocks)
    occupied = frozenset(yard.occupancy())
    free_cells = yard.storage_cells - occupied
    if storage_cells is None:
        storage_cells = tuple(sorted(free_cells))
    else:
        storage_cells = tuple(sorted(set(storage_cells)))
        if any(cell not in free_cells for cell in storage_cells):
            raise ValueError(
                "rolling-GA search domain contains an occupied or non-storage cell"
            )
    return _run_assignment_ga(
        storage_cells,
        len(pending_blocks),
        lambda chromosome: evaluate_operational_rolling_assignment(
            yard,
            pending_blocks,
            chromosome,
            source=source,
            egress_weight=egress_weight,
        ),
        config,
    )


__all__ = [
    "AssignmentCost",
    "Chromosome",
    "GAConfig",
    "GAResult",
    "DurationAwareAssignmentCost",
    "evaluate_duration_aware_rolling_assignment",
    "evaluate_operational_rolling_assignment",
    "evaluate_schedule_assignment",
    "evaluate_rolling_assignment",
    "optimize_rolling_assignment",
    "optimize_duration_aware_rolling_assignment",
    "optimize_operational_rolling_assignment",
    "optimize_schedule_assignment",
]
