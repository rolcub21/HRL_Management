from __future__ import annotations

import heapq
import numpy as np
from option import BaseOption
from example.small_rooms_env import SmallRoomsEnv
MAX_TIMER = 15

__all__ = [
    "PickupOption",
    "StoreOption",
    "DeliverOption",
]

###############################################################################
# Helper functions shared by StoreOption / DeliverOption
###############################################################################



import heapq

def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _astar(
    rooms,
    start: tuple[int, int] | None,
    goal: tuple[int, int] | None,
    blocked: set[tuple[int, int]] | None = None,
):
    """Classic A* that treats cells in `blocked` as obstacles."""
    if start is None or goal is None:
        return []

    blocked = blocked or set()
    blocked = {cell for cell in blocked if cell is not None}

    rows, cols = rooms.shape

    open_set: list[tuple[int, tuple[int, int]]] = []
    heapq.heappush(open_set, (0, start))

    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g: dict[tuple[int, int], int] = {start: 0}
    f: dict[tuple[int, int], int] = {start: _manhattan(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = current[0] + dr, current[1] + dc

            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if rooms[nr, nc] == "#":
                continue

            neigh = (nr, nc)

            if neigh in blocked and neigh != goal:
                continue

            tentative = g[current] + 1
            if tentative < g.get(neigh, float("inf")):
                came_from[neigh] = current
                g[neigh] = tentative
                f[neigh] = tentative + _manhattan(neigh, goal)
                heapq.heappush(open_set, (f[neigh], neigh))

    return []


def diverse_density_path(
    rooms: np.ndarray,
    door: tuple[int,int],
    exits: list[tuple[int,int]],
    u: tuple[int,int],
    occ_cells: set[tuple[int,int]]
) -> list[tuple[int,int]]:
    """Return the door→u→exit path or [] if any segment is blocked."""
    # first leg: only static walls
    p1 = _astar(rooms, door,      u, blocked=set())
    if not p1:
        return []

    # choose nearest exit
    ex, d = min(
        ((ex, len(_astar(rooms, u, ex, blocked=occ_cells)))
         for ex in exits),
        key=lambda pair: pair[1],
        default=(None, None)
    )
    if ex is None:
        return []

    # second leg: respect dynamic blockers
    p2 = _astar(rooms, u, ex, blocked=occ_cells)
    if not p2:
        return []

    full = p1[-1:] + p2
    return full

def diverse_density_selector(
    rooms: np.ndarray,
    G: list[tuple[int,int]],
    door: tuple[int,int],
    exits: list[tuple[int,int]],
    occ_cells: set[tuple[int,int]],
    T: int
) -> list[tuple[int,int]]:
    """
    Repeatedly pick one cell u maximizing |PATH(u) ∩ C| and
    refine C ← C ∩ PATH(u).  Debug prints included.
    """
    C = set(G)              # initial concept = all cells
    chosen_cells: list[tuple[int,int]] = []

    #print(f"[DD Selector] Starting with concept size |C|={len(C)}, occ_cells={occ_cells}")

    for round_idx in range(1, T+1):
        best_u = None
        best_score = -1
        best_path: list[tuple[int,int]] = []

        #print(f"\n[DD Selector] Round {round_idx} — evaluating {len(G)-len(chosen_cells)} candidates")

        for u in G:
            if u in chosen_cells:
                continue

            P = diverse_density_path(rooms, door, exits, u, occ_cells)
            if not P:
                #print(f"  [skip] u={u} → no valid door→u→exit path")
                continue

            score = len(set(P) & C)
            #print(f"  [score] u={u}, |PATH ∩ C|={score}, PATH={P}")

            if score > best_score:
                best_score, best_u, best_path = score, u, P

        if best_u is None:
            #print("[DD Selector] No feasible candidate found; terminating early.")
            break

        # accept best_u
        chosen_cells.append(best_u)
        old_C_size = len(C)
        C &= set(best_path)
        occ_cells.add(best_u)

        #print(f"\n[DD Selector] Round {round_idx} → selected u={best_u} with score={best_score}")
        #print(f"               chosen path: {best_path}")
        #print(f"               concept shrinks: |C| {old_C_size} → {len(C)}")
        #print(f"               mark u={best_u} occupied; occ_cells now size {len(occ_cells)}")

    #print(f"\n[DD Selector] Finished: chosen_cells={chosen_cells}")
    return chosen_cells





def _action_between(curr: tuple[int, int], nxt: tuple[int, int]):
    if nxt[0] < curr[0]:
        return SmallRoomsEnv.ACTION_IDS["UP"]
    if nxt[0] > curr[0]:
        return SmallRoomsEnv.ACTION_IDS["DOWN"]
    if nxt[1] < curr[1]:
        return SmallRoomsEnv.ACTION_IDS["LEFT"]
    if nxt[1] > curr[1]:
        return SmallRoomsEnv.ACTION_IDS["RIGHT"]
    return SmallRoomsEnv.ACTION_IDS["WAIT"]

class StateFlattener:
    def __init__(self, grid_rows, grid_cols, max_dist_storage, max_dist_exit, max_timer=15):
        self.R, self.C  = grid_rows, grid_cols
        self.Ds, self.De = max_dist_storage, max_dist_exit
        self.T          = max_timer

    def __call__(self, state):
        """
        Flatten the raw state into a feature vector.
        Now expects state = (agent_pos, blocks_info, t_next, n_wait)
        """
        # Unpack all four elements
        #(ar, ac), blocks, t_next, n_wait = state
        (ar, ac), blocks = state

        # Start with normalized agent position
        feats = [
            ar / (self.R - 1),
            ac / (self.C - 1)
        ]

        # Loop over each block’s rich feature tuple
        for (
            status,
            (br, bc),
            d_ab, d_bs, d_bg,
            timer,
            is_path_clear,
            path_len
        ) in blocks:
            # One-hot status
            feats.extend(status)

            # Block position (normalized)
            feats.append(br / (self.R - 1))
            feats.append(bc / (self.C - 1))

            # Distances (treat –1 as “no value”)
            feats.append(d_ab / (self.R + self.C))                  # agent→block
            feats.append(1.0 if d_bs < 0 else d_bs / self.Ds)       # block→storage
            feats.append(d_bg / self.De)                            # block→goal

            # Remaining-time feature (clipped and normalized)
            feats.append(-1.0 if timer < 0 else min(timer, self.T) / self.T)

            # Real-time pathing features
            feats.append(is_path_clear)                             # already 0 or 1
            feats.append(
                -1.0 if path_len < 0 else path_len / (self.R + self.C)
            )

        # ─── NEW ─── Add the two timer features ────────────────────────
        # Time until next arrival (clipped & normalized)
        #feats.append(-1.0 if t_next < 0 else min(t_next, self.T) / self.T)
        # Number of blocks waiting (normalized by total blocks)
        #feats.append(n_wait / len(blocks))
        # ────────────────────────────────────────────────────────────────

        return np.asarray(feats, dtype=np.float32)


    
env = SmallRoomsEnv()
flat = StateFlattener(grid_rows = env.grid_rows,
                      grid_cols = env.grid_cols,
                      max_dist_storage = env.grid_rows + env.grid_cols,
                      max_dist_exit = env.grid_rows + env.grid_cols)     # create **once**

#vec  = flat(env.get_current_state())
