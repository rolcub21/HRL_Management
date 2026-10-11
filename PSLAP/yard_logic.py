import math
from collections import deque
from heapq import heappush, heappop
from typing import List, Tuple, Optional

from example.block_instance import Blocks

Cell = Tuple[int, int]


class Yard:
    """
    Minimal yard representation derived from SmallRoomsEnv.
    grid[r][c] contains either a Blocks instance or None.
    """
    def __init__(self, m: int, n: int, open_sides: List[str]):
        self.m = m
        self.n = n
        self.open_sides = open_sides
        self.grid = [[None for _ in range(n)] for __ in range(m)]

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.m and 0 <= c < self.n

    def is_empty(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and self.grid[r][c] is None

    def place_block(self, block: Blocks, r: int, c: int):
        self.grid[r][c] = block
        block.position = (r, c)

    def remove_block(self, block: Blocks):
        if block.position is not None:
            r, c = block.position
            if self.grid[r][c] == block:
                self.grid[r][c] = None
            block.position = None

    def get_open_exits(self) -> List[Cell]:
        exits = []
        if "S" in self.open_sides:
            for col in range(1, self.n - 1):
                exits.append((self.m - 1, col))
        return exits


def find_min_obstruction_path(
    yard: Yard,
    block: Blocks,
    blocks_due_now: List[Blocks]
) -> Tuple[List[Cell], Optional[Cell], List[Blocks]]:
    start = block.position
    if start is None:
        return [], None, []

    exits = set(yard.get_open_exits())
    if start in exits:
        return [start], start, []

    visited = {}
    pq = []
    heappush(pq, (0, 0, [start], [], start))
    visited[start] = 0

    while pq:
        obstruction_count, dist, path, obstr_list, (r, c) = heappop(pq)

        if (r, c) in exits:
            return path, (r, c), obstr_list

        for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if not yard.in_bounds(nr, nc):
                continue

            occupant = yard.grid[nr][nc]
            next_count = obstruction_count
            next_obstr_list = obstr_list

            if occupant is not None and occupant not in blocks_due_now and occupant != block:
                next_count += 1
                if occupant not in obstr_list:
                    next_obstr_list = obstr_list + [occupant]

            if (nr, nc) not in visited or visited[(nr, nc)] > next_count:
                visited[(nr, nc)] = next_count
                heappush(
                    pq,
                    (next_count, dist + 1, path + [(nr, nc)], next_obstr_list, (nr, nc))
                )

    return [], None, []


def can_reach_any_exit(yard: Yard, block: Blocks) -> bool:
    if block.position is None:
        return True

    exits = set(yard.get_open_exits())
    start = block.position
    if start in exits:
        return True

    visited = {start}
    queue = deque([start])

    while queue:
        r, c = queue.popleft()
        if (r, c) in exits:
            return True

        for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if yard.in_bounds(nr, nc):
                occupant = yard.grid[nr][nc]
                if occupant is None or occupant == block:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

    return False


def causes_blockage_when_placed(
    yard: Yard,
    candidate: Cell,
    block: Blocks,
    all_blocks: List[Blocks]
) -> bool:
    r, c = candidate
    old_loc = block.position

    yard.grid[r][c] = block
    block.position = (r, c)

    for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
        if yard.in_bounds(nr, nc):
            neighbor = yard.grid[nr][nc]
            if neighbor is not None and neighbor != block:
                if neighbor.get_remaining_storage_time() < block.get_remaining_storage_time():
                    if not can_reach_any_exit(yard, neighbor):
                        yard.grid[r][c] = None
                        block.position = old_loc
                        return True

    yard.grid[r][c] = None
    block.position = old_loc
    return False


def step_1_find_accessible_candidates(yard: Yard) -> List[Cell]:
    m, n = yard.m, yard.n
    raw_candidates = set()

    def try_add(r, c):
        if yard.in_bounds(r, c) and yard.grid[r][c] is None:
            raw_candidates.add((r, c))

    # W -> E
    for row in range(m):
        row_empty = all(yard.grid[row][col] is None for col in range(n))
        if row_empty:
            if "W" in yard.open_sides and "E" in yard.open_sides:
                try_add(row, n // 2)
            else:
                try_add(row, n - 1)
        else:
            for col in range(n):
                if yard.grid[row][col] is not None:
                    if col > 0 and yard.grid[row][col - 1] is None:
                        try_add(row, col - 1)
                    break

    # E -> W
    for row in range(m):
        row_empty = all(yard.grid[row][col] is None for col in range(n))
        if row_empty:
            if "W" in yard.open_sides and "E" in yard.open_sides:
                try_add(row, n // 2)
            else:
                try_add(row, 0)
        else:
            for col in reversed(range(n)):
                if yard.grid[row][col] is not None:
                    if col < n - 1 and yard.grid[row][col + 1] is None:
                        try_add(row, col + 1)
                    break

    # N -> S
    for col in range(n):
        col_empty = all(yard.grid[row][col] is None for row in range(m))
        if col_empty:
            if "N" in yard.open_sides and "S" in yard.open_sides:
                try_add(m // 2, col)
            else:
                try_add(m - 1, col)
        else:
            for row in range(m):
                if yard.grid[row][col] is not None:
                    if row > 0 and yard.grid[row - 1][col] is None:
                        try_add(row - 1, col)
                    break

    # S -> N
    for col in range(n):
        col_empty = all(yard.grid[row][col] is None for row in range(m))
        if col_empty:
            if "N" in yard.open_sides and "S" in yard.open_sides:
                try_add(m // 2, col)
            else:
                try_add(0, col)
        else:
            for row in reversed(range(m)):
                if yard.grid[row][col] is not None:
                    if row < m - 1 and yard.grid[row + 1][col] is None:
                        try_add(row + 1, col)
                    break

    # keep only valid inner storage area
    candidates = [#
        (r, c)
        for (r, c) in raw_candidates
        if 1 <= r <= (m - 2) and 1 <= c <= (n - 2)
    ]

    # ── FALLBACK: empty yard produces no heuristic hits ──
    if not candidates:
        #print(f"[DEBUG step1] Heuristic scan empty. raw={len(raw_candidates)} "
              #f"(all filtered by inner bounds). Falling back to all empty inner cells.")
        for r in range(1, m - 1):
            for c in range(1, n - 1):
                if yard.grid[r][c] is None:
                    candidates.append((r, c))

    #print(f"[DEBUG step1] Returning {len(candidates)} candidates")
    return candidates


def _pick_by_exit_distance(yard: Yard, candidates: List[Cell]) -> Optional[Cell]:
    exits = yard.get_open_exits()
    if not exits:
        return None

    best_loc = None
    best_dist = math.inf
    for r, c in candidates:
        d = min(abs(r - ex[0]) + abs(c - ex[1]) for ex in exits)
        if d < best_dist:
            best_dist = d
            best_loc = (r, c)
    return best_loc


def select_storage_location(
    yard: Yard,
    block,
    all_blocks: List[Blocks]
) -> Optional[Cell]:
    candidates = step_1_find_accessible_candidates(yard)

    # -------- fallback 1: any empty inner cell --------
    if not candidates:
        #print(f"[DEBUG select] No candidates from step1, trying inner-cell fallback")
        for r in range(1, yard.m - 1):
            for c in range(1, yard.n - 1):
                if yard.grid[r][c] is None:
                    candidates.append((r, c))

    if not candidates:
        print(f"[DEBUG select] No candidates even after fallback. "
              f"yard.m={yard.m}, yard.n={yard.n}")
        #print(f"[DEBUG select] Empty inner cells: "
              #f"{sum(1 for r in range(1,yard.m-1) for c in range(1,yard.n-1) if yard.grid[r][c] is None)}")
        return None

    valid_candidates = []
    for cand in candidates:
        if not causes_blockage_when_placed(yard, cand, block, all_blocks):
            valid_candidates.append(cand)

    final_candidates = valid_candidates if valid_candidates else candidates

    #print(f"[DEBUG select] candidates={len(candidates)}, "
          #f"valid={len(valid_candidates)}, final={len(final_candidates)}")

    if not final_candidates:
        return None

    # inbound block
    if not getattr(block, "stored", False):
        result = _pick_by_exit_distance(yard, final_candidates)
        #print(f"[DEBUG select] inbound -> picked {result}")
        return result#

    # relocation of stored block
    if getattr(block, "position", None) is not None:
        br, bc = block.position
        best_loc = None
        best_dist = math.inf
        for r, c in final_candidates:
            d = abs(r - br) + abs(c - bc)
            if d < best_dist:
                best_dist = d
                best_loc = (r, c)
        #print(f"[DEBUG select] relocation -> picked {best_loc}")
        return best_loc

    #print(f"[DEBUG select] fallback -> picked {final_candidates[0]}")
    return final_candidates[0]

def retrieve_outbound_block(
    yard: Yard,
    block: Blocks,
    all_blocks: List[Blocks],
    blocks_due_now: List[Blocks]
) -> bool:
    path, exit_cell, obstructions = find_min_obstruction_path(yard, block, blocks_due_now)
    if not path:
        return False

    original_locs = {}
    for obs in obstructions:
        original_locs[obs] = obs.position
        yard.remove_block(obs)

        new_loc = select_storage_location(yard, obs, all_blocks)
        if new_loc is None:
            yard.place_block(obs, original_locs[obs][0], original_locs[obs][1])
            for obs2 in obstructions:
                if obs2 in original_locs and obs2.position != original_locs[obs2]:
                    yard.remove_block(obs2)
                    yard.place_block(obs2, original_locs[obs2][0], original_locs[obs2][1])
            return False

        yard.place_block(obs, new_loc[0], new_loc[1])

    yard.remove_block(block)
    return True