"""Independent FIFO/nominal-stay dispatcher: no contender or certificate code.

This scheduler deliberately uses the primitive simulator only. External due
dates are assigned AFTER dispatch; the dispatcher cannot optimize those dates.
"""
from collections import deque
from dataclasses import replace
import random
import numpy as np

from example.small_rooms_env import SmallRoomsEnv

MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))
FILLS = (0.0, 0.4, 0.7)
INCOMING = 36
HORIZON = 10000  # Reference construction cap, NOT a contender task horizon.


def make_env(instance=None, total=36):
    env = SmallRoomsEnv(grid_rows=8, grid_cols=8, start_state=(1, 1),
                        door_cell=(0, 6), pickup_cells=(1, 6),
                        exit_cells=[(7, 4), (7, 5), (7, 6)],
                        number_blocks=total if instance is None else instance.number_blocks,
                        choose_storage=False, proc_mean=80,
                        arrival_rate=1/30 if instance is None else instance.arrival_rate)
    if instance is not None:
        env.reset(instance=instance)
    return env


def generate(seed, fill, gap, incoming=INCOMING):
    if fill not in FILLS or gap < 1 or incoming < 1:
        raise ValueError("unsupported fill, gap or incoming count")
    rng = random.Random(seed)
    probe = make_env()
    count = int(len(probe.storage_positions) * fill)
    # Explicit corridor-conditioned initial distribution, not arbitrary packing.
    # Left column and bottom interior row remain traversable independently of
    # the occupied pickup. At ~70% all 24 other cells are filled. Round DOWN:
    # 24/35=68.6%, rather than making the reserved passage infeasible at 25.
    cells = [(r, c) for r in range(1, 6) for c in range(2, 7)
             if (r, c) != probe.pickup_cell]
    rng.shuffle(cells)
    initial = cells[:count]
    env = make_env(total=count + incoming)
    base = env.sample_episode_instance(seed=seed)
    arrivals = (0,) * count + tuple(i * gap for i in range(incoming))
    durations = tuple(max(1, int(x)) for x in
                      np.random.default_rng(seed).poisson(80, len(arrivals)))
    instance = replace(base, arrival_rate=1/gap, arrival_steps=arrivals,
                       storage_steps_needed=durations)
    return instance, initial


def restore(instance, initial):
    env = make_env(instance)
    for index, block in enumerate(env.blocks):
        block.position = tuple(initial[index]) if index < len(initial) else None
        block.storage_location = block.position
        block.stored = index < len(initial)
        block.picked = block.stored
        block.carrying = block.delivered = False
        block.stored_time_step = 0 if block.stored else None
        block.storage_steps_elapsed = 0
    env.storage_counts = {cell: 0 for cell in env.storage_positions}
    for cell in initial:
        env.storage_counts[tuple(cell)] += 1
    env.update_pickup_queue()
    return env


def path(env, start, goal, source=None):
    """Own BFS. The block being approached is enterable only at its source."""
    occupied = {b.position for b in env.blocks if b.position is not None
                and not b.carrying and not b.delivered}
    occupied.discard(source)
    # Queue/pickup never used as a through-route, even before future arrivals.
    occupied.add(env.waiting_cell)
    if goal != env.pickup_cell:
        occupied.add(env.pickup_cell)
    queue = deque([start])
    previous = {start: None}
    while queue:
        here = queue.popleft()
        if here == goal:
            actions = []
            while previous[here] is not None:
                here, action = previous[here]
                actions.append(action)
            return list(reversed(actions))
        for action, (dr, dc) in enumerate(MOVES):
            nxt = (here[0]+dr, here[1]+dc)
            if (0 <= nxt[0] < 8 and 0 <= nxt[1] < 8
                    and env.rooms[nxt] != '#' and nxt not in occupied
                    and nxt not in previous):
                previous[nxt] = (here, action)
                queue.append(nxt)
    return None


def dispatch(instance, initial, horizon=HORIZON):
    env = restore(instance, initial)
    trace = []
    work = {b.label: 0 for b in env.blocks}
    first_storage = {b.label: 0 for b in env.blocks[:len(initial)]}
    deliveries = {}
    while len(deliveries) < len(env.blocks):
        if env.time_steps >= horizon:
            raise RuntimeError("independent reference exceeded construction horizon")
        options = []
        for block in env.blocks:
            if not block.stored or block.delivered:
                continue
            approach = path(env, env.current_state, block.position, block.position)
            if approach is None:
                continue
            routes = [path(env, block.position, gate, block.position)
                      for gate in env.exit_cells]
            routes = [route for route in routes if route is not None]
            if not routes:
                continue
            route = min(routes, key=lambda x: (len(x), x))
            actions = approach + [4] + route + [5]
            nominal = block.stored_time_step + block.storage_steps_needed
            options.append((nominal, block.label, block, actions))
        ready = [x for x in options if env.time_steps + len(x[3]) >= x[0]]
        chosen = min(ready, key=lambda x: (x[0], x[1])) if ready else None
        kind = 'deliver'
        if chosen is None:
            inbound = next((b for b in env.blocks if not b.stored and
                            not b.delivered and b.position == env.pickup_cell), None)
            if inbound is not None:
                approach = path(env, env.current_state, env.pickup_cell, env.pickup_cell)
                if approach is not None:
                    placements = []
                    for cell in env.storage_positions:
                        route = path(env, env.pickup_cell, cell, env.pickup_cell)
                        if route is not None:
                            placements.append((len(route), cell, route))
                    if placements:
                        # Deepest reachable cell, lexical tie break, independent
                        # of all learned merits, witnesses and baseline rules.
                        _, cell, route = min(placements, key=lambda x: (-x[0], x[1]))
                        inbound.storage_location = cell
                        chosen = (0, inbound.label, inbound, approach+[4]+route+[5])
                        kind = 'accept'
        if chosen is None:
            env.step(6)
            trace.append({'action': 6, 'job': None, 'kind': 'wait', 'target': None})
            continue
        _, label, block, actions = chosen
        for action in actions:
            if env.time_steps >= horizon:
                raise RuntimeError("reference macro crossed construction horizon")
            target = list(block.storage_location) if kind == 'accept' else None
            _, _, _, info = env.step(action)
            if info.get('illegal_drop'):
                raise RuntimeError("reference attempted illegal drop")
            trace.append({'action': action, 'job': label, 'kind': kind, 'target': target})
            work[label] += 1
            if info.get('stored_block'):
                first_storage[label] = env.time_steps
            if info.get('delivered_block'):
                deliveries[label] = env.time_steps
    return {'trace': trace, 'work_by_job': work,
            'first_storage': first_storage, 'deliveries': deliveries,
            'completion_time': env.time_steps}
