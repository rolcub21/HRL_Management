import math
import time
import random
from collections import deque
from typing import List, Tuple, Optional
from heapq import heappush, heappop

class Block:
    def __init__(self, block_id:int, arrival_time:int, stay_duration:int):
        self.block_id = block_id
        self.arrival_time = arrival_time
        self.departure_time = arrival_time + stay_duration
        self.location: Optional[Tuple[int,int]] = None

    def __repr__(self):
        return f"B{self.block_id}(Arr={self.arrival_time},Dep={self.departure_time},Loc={self.location})"


class Yard:
    """
    A 2D yard of size m x n with certain open sides (N, S, W, E).
    No 'temp_stockyard' here; if we can't find space for an obstructive block,
    we report it and cancel the retrieval.
    """
    def __init__(self, m:int, n:int, open_sides:List[str]):
        self.m = m
        self.n = n
        self.open_sides = open_sides
        self.grid = [[None for _ in range(n)] for __ in range(m)]

    def in_bounds(self, r:int, c:int) -> bool:
        return 0 <= r < self.m and 0 <= c < self.n

    def is_empty(self, r:int, c:int) -> bool:
        return self.in_bounds(r, c) and (self.grid[r][c] is None)

    def place_block(self, block:Block, r:int, c:int):
        """Place a block in (r,c). Assumes yard[r][c] is empty and in bounds."""
        self.grid[r][c] = block
        block.location = (r, c)

    def remove_block(self, block:Block):
        """Remove a block from its current location (if any)."""
        if block.location is not None:
            r, c = block.location
            if self.grid[r][c] == block:
                self.grid[r][c] = None
            block.location = None

    def get_open_exits(self) -> List[Tuple[int,int]]:
        """
        Return a list of all (row, col) cells that are 'exits' because they lie
        on an open boundary side: top row if 'N' open, bottom if 'S', left if 'W', right if 'E'.
        """
        exits = []
        if 'N' in self.open_sides:
            for col in range(self.n):
                exits.append((0, col))
        if 'S' in self.open_sides:
            for col in range(self.n):
                exits.append((self.m - 1, col))
        if 'W' in self.open_sides:
            for row in range(self.m):
                exits.append((row, 0))
        if 'E' in self.open_sides:
            for row in range(self.m):
                exits.append((row, self.n - 1))
        return exits

    def display(self):
        print("Yard layout:")
        for r in range(self.m):
            row_str = ""
            for c in range(self.n):
                if self.grid[r][c] is None:
                    row_str += "[    ] "
                else:
                    row_str += f"[B{self.grid[r][c].block_id:2d}] "
            print(row_str)
        print()


############################################################
#   OBSTRUCTIVE BLOCK SELECTION (Section 3.1 in the paper) #
############################################################

def find_min_obstruction_path(
    yard: Yard,
    block: Block,
    blocks_due_now: List[Block]
) -> Tuple[List[Tuple[int,int]], Optional[Tuple[int,int]], List[Block]]:
    """
    Find a path from the block's location to one of the yard's open exits that:
      - Minimizes the number of obstructive blocks (not in blocks_due_now),
      - Then (among ties) has the shortest travel distance.

    Returns: (best_path, exit_cell, list_of_obstructions_on_that_path).

    If no path is found, returns ([], None, []).
    """
    start = block.location
    if not start:
        return [], None, []

    exits = set(yard.get_open_exits())
    if start in exits:
        # Already at an exit
        return [start], start, []

    # We'll do a multi-criteria BFS using a min-heap:
    # Priority: (#obstructions, distance), then path, then occupant set
    visited = dict()  # (r,c) -> minimal obstruction_count found so far
    pq = []
    # state: (obstruction_count, distance, path, obstruct_blocks, (r,c))
    heappush(pq, (0, 0, [start], [], start))
    visited[start] = 0

    best_path = []
    best_exit = None
    best_obs_blocks = []
    found_any = False

    while pq:
        obstruction_count, dist, path, obstr_list, (r, c) = heappop(pq)
        if (r, c) in exits:
            # Reached an exit => best by definition of our min-heap
            found_any = True
            best_path = path
            best_exit = (r, c)
            best_obs_blocks = obstr_list
            break

        # Explore neighbors
        for (nr, nc) in [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]:
            if yard.in_bounds(nr, nc):
                occupant = yard.grid[nr][nc]
                next_obstr_count = obstruction_count
                next_obstr_list = obstr_list

                # occupant is an obstruction if:
                #   occupant is not None,
                #   occupant is not the block we're retrieving,
                #   occupant not in the blocks_due_now
                if occupant is not None and occupant not in blocks_due_now and occupant != block:
                    # That occupant must be removed
                    next_obstr_count += 1
                    if occupant not in obstr_list:
                        next_obstr_list = obstr_list + [occupant]

                # If we haven't visited (nr,nc) or we found a cheaper #obstructions:
                if (nr, nc) not in visited or visited[(nr, nc)] > next_obstr_count:
                    visited[(nr, nc)] = next_obstr_count
                    new_path = path + [(nr,nc)]
                    heappush(pq, (next_obstr_count, dist+1, new_path, next_obstr_list, (nr,nc)))

    if not found_any:
        return [], None, []

    return best_path, best_exit, best_obs_blocks


###############################################
#   STORAGE LOCATION DETERMINATION (Section 3.2)
###############################################

def can_reach_any_exit(yard:Yard, block:Block) -> bool:
    """
    Checks if 'block' can reach an open exit in the current yard configuration.
    BFS: can pass only through empty cells or its own cell.
    """
    if block.location is None:
        # Not in yard => trivially no blocking issues
        return True

    exits = set(yard.get_open_exits())
    start = block.location
    if start in exits:
        return True

    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        r, c = queue.popleft()
        if (r,c) in exits:
            return True
        for (nr,nc) in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
            if yard.in_bounds(nr, nc):
                occupant = yard.grid[nr][nc]
                # can move if occupant is None or occupant == block
                if occupant is None or occupant == block:
                    if (nr,nc) not in visited:
                        visited.add((nr,nc))
                        queue.append((nr,nc))

    return False

def causes_blockage_when_placed(
    yard: Yard,
    candidate: Tuple[int,int],
    block: Block,
    all_blocks: List[Block]
) -> bool:
    """
    Step 2: "Screen out obstructive candidate locations."
    If placing 'block' at 'candidate' makes some neighbor with earlier departure
    unable to reach an exit, we must eliminate that candidate.
    """
    r, c = candidate
    yard.grid[r][c] = block  # temporarily place
    old_loc = block.location
    block.location = (r, c)

    # Check neighbors with an earlier departure time
    for (nr, nc) in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
        if yard.in_bounds(nr, nc):
            neighbor = yard.grid[nr][nc]
            if neighbor is not None and neighbor != block:
                if neighbor.departure_time < block.departure_time:
                    # If neighbor can't reach an exit now, it's a problem
                    if not can_reach_any_exit(yard, neighbor):
                        # revert
                        yard.grid[r][c] = None
                        block.location = old_loc
                        return True

    # revert
    yard.grid[r][c] = None
    block.location = old_loc
    return False


def find_accessible_locations(yard:Yard) -> List[Tuple[int,int]]:
    """
    Implements Step 1: "Search for accessible candidate locations" of the paper,
    combining W->E, E->W, N->S, S->N directional logic.
    """
    m, n = yard.m, yard.n
    cands = set()

    def try_add(r, c):
        if yard.in_bounds(r, c) and yard.grid[r][c] is None:
            cands.add((r,c))

    # Step 1.1: W->E
    for row in range(m):
        row_empty = all(yard.grid[row][col] is None for col in range(n))
        if row_empty:
            # If row is fully empty:
            # - If both W & E open => middle cell
            # - Else last cell (W->E)
            if 'W' in yard.open_sides and 'E' in yard.open_sides:
                mid = n // 2
                try_add(row, mid)
            else:
                try_add(row, n-1)
        else:
            # find first occupant from left
            for col in range(n):
                if yard.grid[row][col] is not None:
                    if col > 0 and yard.grid[row][col-1] is None:
                        try_add(row, col-1)
                    break

    # Step 1.1: E->W
    for row in range(m):
        row_empty = all(yard.grid[row][col] is None for col in range(n))
        if row_empty:
            if 'W' in yard.open_sides and 'E' in yard.open_sides:
                mid = n // 2
                try_add(row, mid)
            else:
                try_add(row, 0)
        else:
            # find first occupant from the right
            for col in reversed(range(n)):
                if yard.grid[row][col] is not None:
                    if col < n-1 and yard.grid[row][col+1] is None:
                        try_add(row, col+1)
                    break

    # Step 1.2: N->S
    for col in range(n):
        col_empty = all(yard.grid[row][col] is None for row in range(m))
        if col_empty:
            if 'N' in yard.open_sides and 'S' in yard.open_sides:
                mid = m // 2
                try_add(mid, col)
            else:
                try_add(m-1, col)
        else:
            # find first occupant from top
            for row in range(m):
                if yard.grid[row][col] is not None:
                    if row > 0 and yard.grid[row-1][col] is None:
                        try_add(row-1, col)
                    break

    # Step 1.2: S->N
    for col in range(n):
        col_empty = all(yard.grid[row][col] is None for row in range(m))
        if col_empty:
            if 'N' in yard.open_sides and 'S' in yard.open_sides:
                mid = m // 2
                try_add(mid, col)
            else:
                try_add(0, col)
        else:
            # find first occupant from bottom
            for row in reversed(range(m)):
                if yard.grid[row][col] is not None:
                    if row < m-1 and yard.grid[row+1][col] is None:
                        try_add(row+1, col)
                    break

    return list(cands)


def select_storage_location(
    yard:Yard,
    block:Block,
    all_blocks:List[Block]
) -> Optional[Tuple[int,int]]:
    """
    Step 1: find accessible locations,
    Step 2: screen them out if they block earlier departures,
    Step 3: pick location that minimizes distance.

    - If block.location is None (inbound), we pick location that is min distance to an exit.
    - If block is obstructive (in yard), pick location that is min distance from current block's location.
    """
    # Step 1
    all_candidates = find_accessible_locations(yard)
    if not all_candidates:
        return None  # no accessible location found

    # Step 2
    valid_candidates = []
    for (r,c) in all_candidates:
        if not causes_blockage_when_placed(yard, (r,c), block, all_blocks):
            valid_candidates.append((r,c))

    # If all were eliminated, revert to original set
    final_candidates = valid_candidates if valid_candidates else all_candidates
    if not final_candidates:
        return None

    # Step 3
    if block.location is None:
        # inbound => pick location with smallest distance to an exit
        best_loc = None
        best_dist = math.inf
        exits = yard.get_open_exits()
        for (r,c) in final_candidates:
            dist_to_exit = min(abs(r-ex[0]) + abs(c-ex[1]) for ex in exits)
            if dist_to_exit < best_dist:
                best_dist = dist_to_exit
                best_loc = (r,c)
        return best_loc
    else:
        # obstructive => pick location closest to block's current location
        br, bc = block.location
        best_loc = None
        best_dist = math.inf
        for (r,c) in final_candidates:
            dist = abs(r - br) + abs(c - bc)
            if dist < best_dist:
                best_dist = dist
                best_loc = (r,c)
        return best_loc


#######################################################
#   MAIN SIMULATION DRIVER (no temp stockyard version)
#######################################################

def run_ABSLAP_heuristic_no_temp(
    m=5, n=5, open_sides=('N'), horizon=50,
    inbound_rate=0.8, max_stay=30
):
    yard = Yard(m, n, list(open_sides))
    all_blocks: List[Block] = []
    sim_time = 0
    block_counter = 1

    total_obstructive_moves = 0

    while sim_time <= horizon:
        print(f"\n===== TIME {sim_time} =====")

        # 1. Retrieve all blocks due now
        blocks_due_now = [
            b for b in all_blocks
            if b.departure_time == sim_time and b.location is not None
        ]
        blocks_due_now.sort(key=lambda x: x.block_id)

        for outbound_block in blocks_due_now:
            print(f"  Retrieving outbound {outbound_block}")

            # 3.1: find path with minimal obstructions
            path, chosen_exit, obstructions = find_min_obstruction_path(yard, outbound_block, blocks_due_now)
            if not path:
                print(f"    No path found => retrieval canceled.")
                continue

            # Try relocating each obstruction
            # If any obstructive block can't be relocated => cancel retrieval
            revert_needed = False
            original_locations = {}  # keep track to revert if needed

            for obs_block in obstructions:
                total_obstructive_moves += 1
                original_locations[obs_block] = obs_block.location

                yard.remove_block(obs_block)
                new_loc = select_storage_location(yard, obs_block, all_blocks)
                if new_loc is not None:
                    yard.place_block(obs_block, new_loc[0], new_loc[1])
                    print(f"    Obstructive {obs_block} => relocated to {new_loc}")
                else:
                    # revert
                    yard.place_block(obs_block, original_locations[obs_block][0], original_locations[obs_block][1])
                    print(f"    No feasible location found for obstructive {obs_block} => retrieval canceled.")
                    revert_needed = True
                    break

            if revert_needed:
                # put any already-moved blocks back
                for obb in obstructions:
                    if obb in original_locations:
                        oldr, oldc = original_locations[obb]
                        if obb.location != (oldr, oldc):
                            yard.remove_block(obb)
                            yard.place_block(obb, oldr, oldc)
                # do not remove the outbound block
                continue

            # If we successfully relocated all obstructive blocks, remove outbound
            yard.remove_block(outbound_block)
            print(f"    Outbound block {outbound_block.block_id} retrieved successfully.")

        # 2. Possibly inbound arrival
        if random.random() < inbound_rate:
            stay = random.randint(10, max_stay)
            new_block = Block(block_counter, sim_time, stay)
            block_counter += 1
            all_blocks.append(new_block)
            print(f"  Inbound arrives: {new_block}")

            new_loc = select_storage_location(yard, new_block, all_blocks)
            if new_loc is not None:
                yard.place_block(new_block, new_loc[0], new_loc[1])
                print(f"    => assigned to {new_loc}")
            else:
                print("    => NO feasible location found. Inbound blocked.")

        yard.display()
        time.sleep(0.3)
        sim_time += 1

    print("=== SIMULATION COMPLETE ===")
    print(f"Total blocks created: {len(all_blocks)}")
    delivered = [b for b in all_blocks if b.location is None and b.departure_time <= horizon]
    print(f"Total blocks actually delivered (departed on schedule): {len(delivered)}")
    print(f"Total obstructive moves counted: {total_obstructive_moves}")
    print()


if __name__ == "__main__":
    run_ABSLAP_heuristic_no_temp()
