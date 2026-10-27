"""Primitive reference replay checks, independent of dispatcher routing/choices."""
from example.small_rooms_env import SmallRoomsEnv


def verify(instance, initial, result, due_dates):
    env = SmallRoomsEnv(grid_rows=8, grid_cols=8, start_state=(1, 1),
                        door_cell=(0, 6), pickup_cells=(1, 6),
                        exit_cells=[(7, 4), (7, 5), (7, 6)],
                        number_blocks=instance.number_blocks, proc_mean=80,
                        arrival_rate=instance.arrival_rate, choose_storage=False)
    env.reset(instance=instance)
    if len(set(map(tuple, initial))) != len(initial):
        raise ValueError('duplicate initial placement')
    for index, block in enumerate(env.blocks):
        cell = tuple(initial[index]) if index < len(initial) else None
        if cell is not None and (cell not in env.storage_positions or cell == env.start_state):
            raise ValueError('invalid initial placement')
        block.position = block.storage_location = cell
        block.stored = block.picked = cell is not None
        block.carrying = block.delivered = False
        block.stored_time_step = 0 if cell is not None else None
        block.storage_steps_elapsed = 0
    env.storage_counts = {cell: int(cell in map(tuple, initial))
                          for cell in env.storage_positions}
    env.update_pickup_queue()
    blocks = {b.label: b for b in env.blocks}
    first = {b.label: 0 for b in env.blocks[:len(initial)]}
    delivered = {}
    work = {b.label: 0 for b in env.blocks}
    offsets = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    trace = result['trace']
    if len(trace) > 10000:
        raise ValueError('reference exceeds construction horizon')
    active_macro = None
    for index, record in enumerate(trace):
        action, label, kind = record['action'], record['job'], record['kind']
        if action not in range(7):
            raise ValueError('unknown primitive action')
        held = [b for b in env.blocks if b.carrying]
        if len(held) > 1:
            raise ValueError('multiple carried blocks')
        if action == 6:
            if label is not None or kind != 'wait' or held:
                raise ValueError('invalid waiting record')
        elif label not in blocks or kind not in ('accept', 'deliver'):
            raise ValueError('invalid work attribution')
        if action == 6:
            if active_macro is not None:
                raise ValueError('wait interrupts reference macro')
        else:
            identity = (label, kind, tuple(record['target']) if kind == 'accept' else None)
            if active_macro is None:
                active_macro = identity
            elif active_macro != identity:
                raise ValueError('reference macro identity/target changed')
            if held and held[0].label != label:
                raise ValueError('carried work attributed to a different job')
        if action in offsets:
            dr, dc = offsets[action]
            cell = (env.current_state[0]+dr, env.current_state[1]+dc)
            if not (0 <= cell[0] < 8 and 0 <= cell[1] < 8) or env.rooms[cell] == '#':
                raise ValueError('wall/boundary crossing')
            obstructors = [b for b in env.blocks if b.position == cell
                           and not b.carrying and not b.delivered]
            if obstructors:
                # Entering an occupied source is permitted solely to pick it
                # up on the very next primitive, never as a transit shortcut.
                nxt = trace[index+1] if index+1 < len(trace) else {}
                if (held or len(obstructors) != 1 or obstructors[0].label != label
                        or nxt.get('action') != 4 or nxt.get('job') != label):
                    raise ValueError('block collision or occupied transit')
        if action == 4:
            block = blocks[label]
            if held or block.delivered or block.position != env.current_state:
                raise ValueError('invalid pickup')
            if (kind == 'accept') == block.stored:
                raise ValueError('incorrect pickup kind')
        if action == 5:
            if len(held) != 1 or held[0].label != label:
                raise ValueError('invalid putdown identity')
            if kind == 'accept':
                cell = tuple(record['target'])
                if cell not in env.storage_positions or cell != env.current_state:
                    raise ValueError('invalid admission target')
                if any(b.position == cell and not b.carrying and not b.delivered
                       for b in env.blocks):
                    raise ValueError('occupied admission target')
                held[0].storage_location = cell
            elif env.current_state not in env.exit_cells:
                raise ValueError('delivery outside exit')
        _, _, _, info = env.step(action)
        if info.get('illegal_drop'):
            raise ValueError('simulator rejected drop')
        if action != 6:
            work[label] += 1
        if info.get('stored_block'):
            first[info['stored_block']] = env.time_steps
        if info.get('delivered_block'):
            delivered[info['delivered_block']] = env.time_steps
        if action == 5:
            active_macro = None
    if set(delivered) != set(blocks) or any(b.carrying for b in env.blocks):
        raise ValueError('incomplete reference')
    if (first != result['first_storage'] or delivered != result['deliveries']
            or work != result['work_by_job'] or env.time_steps != result['completion_time']):
        raise ValueError('replayed reference records disagree')
    if set(due_dates) != set(blocks) or any(
            type(due_dates[k]) is not int or due_dates[k] < blocks[k].arrival_step
            or abs(delivered[k]-due_dates[k]) > 20 for k in blocks):
        raise ValueError('reference violates constructed due window')
    # Geometry-only lower bound: ignores empty travel and all obstructions.
    lower_bound = 0
    for index, b in enumerate(env.blocks):
        if index < len(initial):
            cell = tuple(initial[index])
            lower_bound += 2 + min(abs(cell[0]-g[0])+abs(cell[1]-g[1])
                                   for g in env.exit_cells)
        else:
            lower_bound += min(4 + abs(1-c[0])+abs(6-c[1])
                               + min(abs(c[0]-g[0])+abs(c[1]-g[1]) for g in env.exit_cells)
                               for c in env.storage_positions)
    initial_labels = {b.label for b in env.blocks[:len(initial)]}
    incoming = [b for b in env.blocks if b.label not in initial_labels]
    return {'strict_complete': True, 'jobs': len(blocks), 'incoming_jobs': len(incoming),
            'initial_jobs': len(initial), 'equipment_work': sum(work.values()),
            'incoming_work': sum(work[b.label] for b in incoming),
            'initial_clearance_work': sum(work[k] for k in initial_labels),
            'work_per_incoming_job': sum(work[b.label] for b in incoming)/len(incoming),
            'mean_admission_wait': sum(first[b.label]-b.arrival_step for b in incoming)/len(incoming),
            'completion_time': env.time_steps, 'rehandles': 0,
            'transport_work_lower_bound': lower_bound,
            'reference_work_over_lower_bound': sum(work.values())/lower_bound}
