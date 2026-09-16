############################################################
# yard_logic.py
#
# Helper module with all yard logic from the ABSLAP approach,
# but adapted to use *your* Blocks class directly (no custom Block).
#
# Main features:
#  1) find_min_obstruction_path
#  2) find_accessible_locations
#  3) causes_blockage_when_placed
#  4) select_storage_location
#  5) retrieve_outbound_block
############################################################

import math
import time
import random
from typing import List, Tuple, Optional
from collections import deque
from heapq import heappush, heappop

# Import your environment's "Blocks" class
from example.block_instance import Blocks

class Yard:
    """
    Minimal yard class with a 2D grid and 'open_sides' that define exits.
    yard.grid[r][c] = a Blocks instance or None.
    """
    def __init__(self, m: int, n: int, open_sides: List[str]):
        self.m = m
        self.n = n
        self.open_sides = open_sides
        # 2D array [m rows][n cols], each cell: None or Blocks.
        self.grid = [[None for _ in range(n)] for __ in range(m)]

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.m and 0 <= c < self.n

    def is_empty(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and (self.grid[r][c] is None)

    def place_block(self, block: 'Blocks', r: int, c: int):
        """
        Place 'block' in yard.grid[r][c]. Assumes yard[r][c] is empty.
        Also updates block.position.
        """
        self.grid[r][c] = block
        block.position = (r, c)

    def remove_block(self, block: 'Blocks'):
        """
        Remove 'block' from wherever it is in yard (if anywhere).
        Clears yard[r][c] and sets block.position to None.
        """
        if block.position is not None:
            r, c = block.position
            if self.grid[r][c] == block:
                self.grid[r][c] = None
            block.position = None

    def get_open_exits(self) -> List[Tuple[int, int]]:
        """
        Return a list of exit (row, col) cells along the boundary according to open_sides.
        For our environment, we want only the South side (row = m-1) and only cells that
        are not on the extreme boundaries (i.e. columns 1 through n-2).
        """
        exits = []
        if 'S' in self.open_sides:
            for col in range(1, self.n - 1):
                exits.append((self.m - 1, col))
        # If needed, similar logic can be added for other directions.
        return exits



############################################################
# 1) OBSTRUCTIVE BLOCK SELECTION SUB-ALGORITHM (Paper §3.1)
############################################################

def find_min_obstruction_path(
    yard: Yard,
    block: Blocks,
    blocks_due_now: List[Blocks]
) -> Tuple[List[Tuple[int,int]], Optional[Tuple[int,int]], List[Blocks]]:
    """
    Return (best_path, exit_cell, obstructions_on_that_path).
    We do a BFS with a priority: (#obstructions, distance).
      - #obstructions = how many occupant blocks are not in blocks_due_now
      - distance = BFS path length

    If the block is already on an exit, we return ([start], start, []).
    If no path is found, we return ([], None, []).
    """
    start = block.position
    if not start:
        return [], None, []
    exits = set(yard.get_open_exits())

    if start in exits:
        # already at an exit
        return [start], start, []

    visited = {}
    pq = []
    # state in PQ: (obstr_count, dist, path, obstruct_list, (r,c))
    heappush(pq, (0, 0, [start], [], start))
    visited[start] = 0

    best_path = []
    best_exit = None
    best_obstructions = []
    found_path = False

    while pq:
        obstruction_count, dist, path, obstr_list, (r, c) = heappop(pq)

        if (r,c) in exits:
            best_path = path
            best_exit = (r,c)
            best_obstructions = obstr_list
            found_path = True
            break

        # check neighbors
        for (nr,nc) in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
            if yard.in_bounds(nr,nc):
                occupant = yard.grid[nr][nc]
                next_count = obstruction_count
                next_obstr_list = obstr_list

                if occupant is not None and occupant not in blocks_due_now and occupant != block:
                    next_count += 1
                    if occupant not in obstr_list:
                        next_obstr_list = obstr_list + [occupant]

                if (nr,nc) not in visited or visited[(nr,nc)] > next_count:
                    visited[(nr,nc)] = next_count
                    new_path = path + [(nr,nc)]
                    heappush(pq, (next_count, dist+1, new_path, next_obstr_list, (nr,nc)))

    if not found_path:
        return [], None, []
    return best_path, best_exit, best_obstructions


############################################################
# 2) STORAGE LOCATION DETERMINATION SUB-ALGORITHM (§3.2)
############################################################

def can_reach_any_exit(yard: Yard, block: Blocks) -> bool:
    """
    BFS to see if 'block' can reach any open exit, ignoring occupant blocks except 'block' itself.
    """
    if block.position is None:
        return True
    exits = set(yard.get_open_exits())
    if block.position in exits:
        return True

    visited = set([block.position])
    queue = deque([block.position])
    while queue:
        r,c = queue.popleft()
        if (r,c) in exits:
            return True
        for (nr,nc) in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
            if yard.in_bounds(nr,nc):
                occupant = yard.grid[nr][nc]
                # pass if occupant is None or occupant==block
                if occupant is None or occupant == block:
                    if (nr,nc) not in visited:
                        visited.add((nr,nc))
                        queue.append((nr,nc))
    return False


def causes_blockage_when_placed(
    yard: Yard,
    candidate: Tuple[int,int],
    block: Blocks,
    all_blocks: List[Blocks]
) -> bool:
    """
    Step 2: "Screen out obstructive candidate locations."
    If placing 'block' at candidate prevents a neighbor with *earlier departure*
    from reaching an exit, we discard that candidate.

    We'll treat "earlier departure" as "neighbor has fewer remaining steps."
    """
    (r,c) = candidate
    old_loc = block.position
    # place temporarily
    yard.grid[r][c] = block
    block.position = (r,c)

    # For neighbors that have fewer steps left => must remain unblocked
    for (nr,nc) in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
        if yard.in_bounds(nr,nc):
            neighbor = yard.grid[nr][nc]
            if neighbor is not None and neighbor != block:
                # If neighbor is "earlier," i.e. fewer steps left:
                if neighbor.get_remaining_storage_time() < block.get_remaining_storage_time():
                    if not can_reach_any_exit(yard, neighbor):
                        # revert
                        yard.grid[r][c] = None
                        block.position = old_loc
                        return True

    # revert
    yard.grid[r][c] = None
    block.position = old_loc
    return False


def step_1_find_accessible_candidates(yard: Yard) -> List[Tuple[int,int]]:
    """
    Step 1: W->E, E->W, N->S, S->N scanning for candidate empty cells.
    We skip the outer boundary row=0, col=0, row=m-1, col=n-1, etc. by default.
    The scanning logic is based on the paper's step 1.1 and 1.2.
    """
    m, n = yard.m, yard.n
    result = set()

    def try_add(r,c):
        if yard.in_bounds(r,c) and yard.grid[r][c] is None:
            # skip outer boundary
            if 1 <= r <= (m-2) and 1 <= c <= (n-2):
                result.add((r,c))

    # 1.1 W->E
    for row in range(m):
        row_empty = all(yard.grid[row][col] is None for col in range(n))
        if row_empty:
            if 'W' in yard.open_sides and 'E' in yard.open_sides:
                mid = n // 2
                try_add(row, mid)
            else:
                try_add(row, n-1)
        else:
            occupant_idx = None
            for col in range(n):
                if yard.grid[row][col] is not None:
                    occupant_idx = col
                    break
            try_add(row, occupant_idx-1)


    # 1.1 E->W
    for row in range(m):
        row_empty = all(yard.grid[row][col] is None for col in range(n))
        if row_empty:
            if 'W' in yard.open_sides and 'E' in yard.open_sides:
                mid = n // 2
                try_add(row, mid)

            else:
                try_add(row, 0)

        else:
            occupant_idx = None
            for col in reversed(range(n)):
                if yard.grid[row][col] is not None:
                    occupant_idx = col
                    break
            if occupant_idx is not None and occupant_idx<(n-1):
                try_add(row, occupant_idx+1)


    # 1.2 N->S
    for col in range(n):
        col_cells = [yard.grid[r][col] for r in range(m)]
        all_empty = all(x is None for x in col_cells)
        if all_empty:
            if 'N' in yard.open_sides and 'S' in yard.open_sides:
                mid = m // 2
                try_add(mid, col)

            else:
                try_add(m-1, col)

        else:
            occupant_idx = None
            for row in range(m):
                if col_cells[row] is not None:
                    occupant_idx = row
                    break
            if occupant_idx is not None and occupant_idx>0:
                if col_cells[occupant_idx-1] is None:
                    result.add((occupant_idx-1, col))

    # 1.2 S->N
    for col in range(n):
        col_cells = [yard.grid[r][col] for r in range(m)]
        all_empty = all(x is None for x in col_cells)
        if all_empty:
            if 'N' in yard.open_sides and 'S' in yard.open_sides:
                mid = m // 2
                try_add(mid, col)

            else:
                try_add(0, col)

        else:
            occupant_idx = None
            for row in reversed(range(m)):
                if col_cells[row] is not None:
                    occupant_idx = row
                    break
            if occupant_idx is not None and occupant_idx<(m-1):
                try_add(occupant_idx+1, col)


    # The paper says "If accessible candidate crosses over another => remove it."
    # We'll skip that or do minimal.

    return list(result)


def _pick_by_exit_distance(yard: Yard, candidates: List[Tuple[int,int]]) -> Optional[Tuple[int,int]]:
    """
    Among 'candidates', pick the one that yields minimal manhattan distance to an exit.
    """
    best_loc = None
    best_d = math.inf
    exits = yard.get_open_exits()
    #print(f"Exits: {exits}")
    if not exits:
        return None

    for (r,c) in candidates:
        d = min(abs(r - ex[0]) + abs(c - ex[1]) for ex in exits)
        if d < best_d:
            best_d = d
            best_loc = (r,c)
    return best_loc


def select_storage_location(
    yard: Yard,
    block: Blocks,
    all_blocks: List[Blocks]
) -> Optional[Tuple[int,int]]:
    """
    Step 1: find_accessible_locations
    Step 2: remove those that cause blockage (causes_blockage_when_placed)
    Step 3: pick location that yields minimal movement distance from block.loc or from exit if inbound
    """
    # Step 1
    candidates = step_1_find_accessible_candidates(yard)
    #print(f"Found {len(candidates)} accessible locations for {block}")
    if not candidates:
        return None

    # Step 2
    valid_candidates = []
    for (r,c) in candidates:
        if not causes_blockage_when_placed(yard, (r,c), block, all_blocks):
            valid_candidates.append((r,c))

    final_candidates = valid_candidates if valid_candidates else candidates
    #print(f"Found {len(final_candidates)} non-obstructive locations for {block}")
    #for (r,c) in final_candidates:
        #print(f"  ({r},{c})")
    if not final_candidates:
        return None

    # Step 3
    if block.stored is False:
        # inbound => location near exit
        best_location = _pick_by_exit_distance(yard, final_candidates)
        #print(f"Selected location near exit: {best_location}")
        return best_location
    
    else:
        # obstructive => location near block
        (br, bc) = block.position
        best_loc = None
        best_dist = math.inf
        for (r,c) in final_candidates:
            dist = abs(r - br) + abs(c - bc)
            if dist < best_dist:
                best_dist = dist
                best_loc = (r,c)
        return best_loc


###########################################################
#  3) OPTIONAL HELPER: RETRIEVE_OUTBOUND_BLOCK (no temp)
###########################################################

def retrieve_outbound_block(
    yard: Yard,
    block: Blocks,
    all_blocks: List[Blocks],
    blocks_due_now: List[Blocks]
) -> bool:
    """
    1) find path with minimal obstructions
    2) relocate obstructions
    3) remove block from yard
    """
    path, exit_cell, obstructions = find_min_obstruction_path(yard, block, blocks_due_now)
    if not path:
        print(f"No path found for {block}, retrieval canceled.")
        return False

    original_locs = {}
    for obs in obstructions:
        original_locs[obs] = obs.position
        yard.remove_block(obs)
        new_loc = select_storage_location(yard, obs, all_blocks)
        if new_loc is None:
            # revert
            yard.place_block(obs, original_locs[obs][0], original_locs[obs][1])
            # revert prior
            for obs2 in obstructions:
                if obs2 in original_locs and obs2.position != original_locs[obs2]:
                    yard.remove_block(obs2)
                    yard.place_block(obs2, original_locs[obs2][0], original_locs[obs2][1])
            print(f"No storage found for obstructive {obs} => retrieval canceled.")
            return False
        else:
            yard.place_block(obs, new_loc[0], new_loc[1])
            print(f"Relocated {obs} => {new_loc}")

    yard.remove_block(block)
    print(f"Successfully retrieved {block} from yard.")
    return True
