"""Trajectory rescoring only: no policy inputs, fitting, or action changes."""
from dataclasses import dataclass
import math

WAIT_WEIGHT=0.5
TOLERANCE=20.
PRIMARY_WEIGHTS=(2.,1.,.1,10.)


@dataclass(frozen=True)
class Job:
    label: str
    release: float
    due: float
    first_storage: float | None
    delivery: float | None
    initially_stored: bool=False


def finite_nonnegative(value,name):
    value=float(value)
    if not math.isfinite(value) or value<0: raise ValueError(f'Invalid {name}')
    return value


def rescore(jobs,*,observation_end,equipment_work,rehandles,strict_complete):
    """Compute J and J_W from exactly the same observed physical trajectory.

    Admission wait means release -> first completed storage, consistent with
    existing reporting. It includes movement in the admission macro; it is not
    pure stationary time at the pickup. Initially stored jobs contribute zero
    wait within this observation window. Failed episodes retain censored wait
    and partial observed cost, but are not eligible for completed-cost ranking.
    """
    end=finite_nonnegative(observation_end,'observation end')
    if not isinstance(strict_complete,bool): raise ValueError('Strict completion must be a boolean')
    work=finite_nonnegative(equipment_work,'equipment work')
    relocations=finite_nonnegative(rehandles,'rehandles')
    jobs=tuple(jobs)
    if not jobs or len({j.label for j in jobs})!=len(jobs):
        raise ValueError('Need nonempty uniquely labeled job manifest')
    late=early=wait=0.
    censored=not_released=0
    records=[]
    for job in jobs:
        if not isinstance(job.initially_stored,bool): raise ValueError('Initial-inventory marker must be a boolean')
        if not math.isfinite(job.release) or not math.isfinite(job.due) or job.due<job.release:
            raise ValueError('Invalid release or external due date')
        if not job.initially_stored and job.release<0:
            raise ValueError('Negative release requires an explicit initial-inventory convention')
        stored=job.first_storage
        delivered=job.delivery
        if stored is not None and (not math.isfinite(stored) or stored>end):
            raise ValueError('Storage timestamp is unobserved or invalid')
        if delivered is not None and (not math.isfinite(delivered) or delivered<0 or delivered>end):
            raise ValueError('Delivery timestamp is unobserved or invalid')
        if job.initially_stored:
            if stored is None or stored>0: raise ValueError('Initial inventory must be stored by time zero')
        elif stored is not None and stored<job.release:
            raise ValueError('Admission precedes release')
        if delivered is not None and (stored is None or delivered<stored or delivered<job.release):
            raise ValueError('Delivery must follow release and first storage')
        if job.release>end:
            if job.initially_stored or stored is not None or delivered is not None:
                raise ValueError('Unreleased job has physical events')
            not_released+=1
            records.append(dict(label=job.label,wait=0.,wait_status='not_released'))
            continue
        stop=end if delivered is None else delivered
        # Count only tardiness accrued during observation, not prehistory.
        late+=max(stop-max(0.,job.release,job.due+TOLERANCE),0.)
        if delivered is not None: early+=max(job.due-delivered-TOLERANCE,0.)
        if job.initially_stored:
            value=0.; status='initial_inventory_excluded'
        elif stored is None:
            value=end-job.release; status='right_censored'; censored+=1
        else:
            value=stored-job.release; status='observed'
        wait+=value
        records.append(dict(label=job.label,wait=value,wait_status=status))
    physical_complete=all(j.delivery is not None for j in jobs)
    if strict_complete and not physical_complete:
        raise ValueError('Strict completion claimed with missing jobs')
    primary=2*late+early+.1*work+10*relocations
    return dict(J=primary,J_W=primary+WAIT_WEIGHT*wait,total_admission_wait=wait,
        wait_cost=WAIT_WEIGHT*wait,wait_weight=WAIT_WEIGHT,
        components=dict(tardiness=late,earliness=early,equipment_work=work,rehandles=relocations),
        wait_records=records,right_censored_jobs=censored,not_released_jobs=not_released,
        strict_complete=bool(strict_complete),cost_ranking_eligible=bool(strict_complete and physical_complete),
        sensitivity_only=True,retraining=False,trajectory_changed=False)
