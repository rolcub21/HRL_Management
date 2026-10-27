"""Cost predictions using released jobs and bound paths only."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ReleasedJob:
    label: str
    due: int


def components(now, duration, jobs, kind, target=None, gamma=1.):
    if duration <= 0 or not 0 <= gamma <= 1:
        raise ValueError('Invalid macro duration or discount')
    # A delivery occurs on the last primitive step: it is undelivered at the
    # beginning of that step, just as in ReleaseETCostEnv.step.
    late=sum(gamma**k * sum(now+k >= job.due+20 for job in jobs)
             for k in range(duration))
    early=0.
    if kind=='deliver':
        job=next(j for j in jobs if j.label==target)
        early=gamma**(duration-1)*max(job.due-(now+duration)-20,0)
    work=0. if kind=='defer' else sum(gamma**k for k in range(duration))
    rehandle=gamma**(duration-1) if kind=='reconfigure' else 0.
    return (float(late),float(early),float(work),float(rehandle))


def predict(env,candidate,gamma=1.):
    kind=candidate.action_type.value
    # Never inspect the schedule or duration of an unreleased job.
    jobs=tuple(ReleasedJob(str(b.label),int(b.arrival_step+b.storage_steps_needed))
               for b in env.blocks if b.arrival_step<=env.time_steps and not b.delivered)
    if kind=='defer':
        duration=candidate.horizon_steps
    elif candidate.recovery_action is not None:
        duration=candidate.recovery_action.steps
    else:
        block=next(b for b in env.blocks if b.label==candidate.target_label)
        approach=env.plan_path_heuristic(env.current_state,block.position,ignore_block=block)
        transport=env.plan_path_heuristic(block.position,candidate.destination,ignore_block=block)
        if (env.current_state!=block.position and not approach
                or block.position!=candidate.destination and not transport):
            raise ValueError('Candidate has no predicted executable route')
        duration=len(approach)+len(transport)+2
    return dict(duration=duration,components=components(env.time_steps,duration,jobs,kind,
                candidate.target_label,gamma),
                assumption='no exogenous interruption or path repair during macro',
                event_sensitive=(kind=='defer'))


def compare(prediction,actual_duration,actual_components,success):
    error=max(abs(a-b) for a,b in zip(prediction['components'],actual_components))
    return dict(matches=bool(success and prediction['duration']==actual_duration and error<1e-8),
                duration_error=actual_duration-prediction['duration'],max_component_error=error)
