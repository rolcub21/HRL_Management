"""Bounded first-disagreement diagnostic; never changes any learned weights."""
import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path
from unittest.mock import patch

import torch
import example.Options.selector_v5 as selector_module
import example.urgency_scheduler as scheduler_module
from example.controller_options import build_controller_options
from example.Options.AcceptStoreOption import ReservedAcceptStoreOption
from example.Options.RetrieveDeliverOption import StrictRetrieveDeliverOption
from example.Options.StrategicDeferOption import StrategicDeferOption
from PSLAP.track_a import TRACK_A_DYNAMIC
from PSLAP.retrieval_context import retrieval_planning_context
from PSLAP.viability_candidates import ViabilityMode
from experiments.conditioned_vcg.E110_guarded_deliver_accept_counterfactual_8x8_174k_v1 import run as provenance
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v2 import run as source
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v2.acceleration import SharedProgressAgent,cache_session
from experiments.conditioned_vcg.E126_release_clock_informed_baselines_190k_v1 import run as informed,clock

ROOT=source.p.ROOT
OUTPUT=ROOT/'results/vcg-conditioned-e127-same-state-ranking-diagnostic-191k-v1'
PROTOCOL='vcg_conditioned_e127_same_state_ranking_diagnostic_191k_v1'
ARM='A_outflow_bounded'
SEED=source.p.VALIDATION_SEEDS[0]
CATEGORIES=('admission_cell','retrieval_timing')
CHECKPOINT=source.p.DEFAULT_OUTPUT/ARM/'seed-0/candidate-episode-120.pth'
read,write,require=source.read,source.write,source.require
sha256,digest=informed.ref.sha256,informed.ref.digest


@lru_cache(maxsize=1)
def contract():
    parent=informed.contract()
    paths=[Path(__file__),Path(__file__).with_name('run.sh'),
           ROOT/'tests/test_vcg_e127_same_state_ranking.py']
    payload=dict(protocol=PROTOCOL,source_contract_sha256=parent['contract_sha256'],
        checkpoint_sha256=sha256(CHECKPOINT),checkpoint_episode=120,arm=ARM,model_seed=0,
        instance_seed=SEED,device='cpu',categories=list(CATEGORIES),maximum_episode_runs=3,
        intervention='first available disagreement per category; one forced certified macro, then native guarded continuation',
        baseline_query='release-informed Dynamic PSLAP on a physical clone, fresh scheduler memory at each boundary',
        caveat='a same-state baseline-inspired proposal, not a full alternative baseline history',
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in paths},
        training_runs=0,paper_evidence=False,independent_test_opened=False)
    return {**payload,'contract_sha256':digest(payload)}


def prepare(output):
    expected=contract()
    path=output/'contract.json'
    if path.exists(): require(read(path)==expected,'E127 contract changed')
    else:
        require(not output.exists() or not any(output.iterdir()),'Need empty contracted output')
        write(path,expected)


def baseline_proposal(env):
    """Never run options or alter the native yard while asking the heuristic."""
    before=provenance._environment_digest(env)
    shadow=deepcopy(env)
    shadow.options=set()
    shadow._retrieval_planning_context_cache=None
    shadow.e126_clock_calls=Counter()
    shadow.e126_clock_samples=[]
    with ExitStack() as contexts:
        def make_view(kind):
            return lambda *args,**kwargs:clock.planning_view(shadow,kind,*args,**kwargs)
        contexts.enter_context(patch.object(selector_module,'BlockView',make_view('assignment_view')))
        contexts.enter_context(patch.object(clock.accept,'BlockView',make_view('accept_mask_view')))
        contexts.enter_context(patch.object(scheduler_module.DurationAwareAtomicScheduler,'_projected_plan',clock.projected_plan))
        selector=selector_module.make_track_b_assignment_option(shadow,TRACK_A_DYNAMIC,seed=0,device='cpu')
        build_controller_options(shadow,selector,
            controller_action_interface=scheduler_module.RESERVED_CELL_ACTION_INTERFACE,max_defer_steps=10)
        scheduler=informed.baseline.adapter.WaitEnabledReconfigurationScheduler(shadow,margin_steps=0.)
        option=scheduler.select_action(shadow.get_current_state())
        reason=scheduler.decisions[-1]['reason']
        if isinstance(option,ReservedAcceptStoreOption):
            estimate=option.active_reservation
            require(estimate is not None,'Baseline Accept lacks bound destination')
            proposal=dict(action_type='accept',target_label=estimate.block_label,
                          destination=list(estimate.preview_storage_cell),reason=reason)
        elif isinstance(option,StrictRetrieveDeliverOption):
            plan=retrieval_planning_context(shadow,relocation_selector=scheduler.relocation_selector).plan(option.target_label)
            require(plan is not None,'Baseline retrieval plan absent')
            if plan.relocations:
                move=plan.relocations[0]
                proposal=dict(action_type='reconfigure',target_label=move.block_label,
                              destination=list(move.target),reason=reason,mapped_first_retrieval_leg=True)
            else:
                proposal=dict(action_type='deliver',target_label=option.target_label,
                              destination=list(plan.exit_cell),reason=reason,estimated_steps=plan.estimated_steps)
        elif isinstance(option,StrategicDeferOption):
            proposal=dict(action_type='defer',target_label=None,destination=None,reason=reason,
                          comparison='intent only; Defer stopping conditions may differ')
        else:
            proposal=dict(action_type='reconfigure',target_label=option.target_label,
                          destination=list(option.destination),reason=reason)
    require(provenance._environment_digest(env)==before,'Shadow query changed native physical state')
    return proposal


def matches(candidate,proposal):
    return (candidate.action_type.value==proposal['action_type']
        and candidate.target_label==proposal['target_label']
        and (proposal['destination'] is None or tuple(candidate.destination or ())==tuple(proposal['destination'])))


def attribution(proposal,native,safe_match,admissible_match,forced,q_native=None,q_alternative=None):
    if matches(native,proposal): return 'same_choice'
    if not safe_match: return 'absent_from_exact_safe_frontier_not_proven_unsafe'
    if not admissible_match: return 'guard_or_interface_restriction'
    if forced: return 'guard_forced'
    if q_alternative>q_native+1e-7: return 'hierarchy_rejects_higher_merit_alternative'
    if q_alternative<q_native-1e-7: return 'critic_ranks_native_higher'
    return 'merit_tie_or_numerical_tolerance'


def force(agent,decision,snapshot,index):
    require(not decision.liveness_forced,'Cannot override a committed/forced guard choice')
    prepared=decision.prepared_snapshot
    candidate=snapshot.candidates[prepared.source_indices[index]]
    record=prepared.records[index]
    require(candidate.key==record.key,'Force mapping differs')
    revised=replace(decision,candidate=candidate,record=record,selection_source='e127_single_certified_intervention',
        explored=False,liveness_forced=False,exact_rank_progress=bool(snapshot.audit.recovery_rank_exact
        and candidate.mode is ViabilityMode.RECOVER and candidate.rank_delta is not None and candidate.rank_delta>0))
    for counts,old,new in ((agent.mode_decisions,decision.candidate.mode.value,candidate.mode.value),
        (agent.action_decisions,decision.candidate.action_type.value,candidate.action_type.value),
        (agent.selection_sources,decision.selection_source,revised.selection_source)):
        counts[old]-=1
        counts[new]+=1
    if agent.decision_log:
        agent.decision_log[-1].update(selected_key=candidate.key,selected_mode=candidate.mode.value,
            selected_action_type=candidate.action_type.value,selection_source=revised.selection_source,
            explored=False,liveness_forced=False,exact_rank_progress=revised.exact_rank_progress)
    return revised


def run_episode(output,fork=None):
    prepare(output)
    agent=SharedProgressAgent.from_checkpoint(torch.load(CHECKPOINT,map_location='cpu',weights_only=False),
                                             device='cpu',resumable=False,seed=0)
    agent.Q_local.eval().requires_grad_(False)
    agent.Q_target.eval().requires_grad_(False)
    before=source.original.replay.parameter_digest(agent)
    before_updates=agent.gradient_steps
    before_replay=len(agent.replay)
    native_select=agent.select
    rows=[]
    captures={}
    keys=[]
    interventions=0
    def select(snapshot,*,training=False,epsilon=0.):
        nonlocal interventions
        require(not training and epsilon==0.,'Frozen evaluation only')
        decision=native_select(snapshot,training=False,epsilon=0.)
        index=len(keys)
        env=agent.context_environment
        if fork is not None and index<=fork['decision_index']:
            require(keys==fork['prefix_keys'][:index],'Counterfactual prefix keys differ')
        # Branches query only at the fork; reference queries every native boundary.
        inspect=fork is None or index==fork['decision_index']
        if inspect:
            proposal=baseline_proposal(env)
            safe=[c for c in snapshot.candidates if matches(c,proposal)]
            bank=decision.prepared_snapshot
            found=[i for i,source_index in enumerate(bank.source_indices)
                   if matches(snapshot.candidates[source_index],proposal)]
            alternative=min(found,key=lambda i:bank.records[i].key) if found else None
            q_native=decision.q_values[next(i for i,r in enumerate(bank.records) if r.key==decision.record.key)]
            q_alternative=decision.q_values[alternative] if alternative is not None else None
            label=attribution(proposal,decision.candidate,bool(safe),alternative is not None,
                              decision.liveness_forced,q_native,q_alternative)
            row=dict(decision_index=index,time_steps=env.time_steps,
                native_key=decision.candidate.key,native_action=decision.candidate.action_type.value,
                native_destination=decision.candidate.destination,baseline_proposal=proposal,attribution=label,
                safe_match_keys=[c.key for c in safe],admissible_match_keys=[bank.records[i].key for i in found],
                q_native=q_native,q_alternative=q_alternative,mode_values=decision.mode_values,
                native_q_global_max=max(decision.q_values),forced=decision.liveness_forced,
                candidate_count=len(snapshot.candidates),admissible_count=len(bank.records))
            rows.append(row)
            category=('admission_cell' if proposal['action_type']=='accept' and row['native_action']=='accept'
                      else 'retrieval_timing' if proposal['action_type']=='deliver' and row['native_action']!='deliver' else None)
            eligible=(alternative is not None and not decision.liveness_forced and not matches(decision.candidate,proposal))
            if (fork is None and category in CATEGORIES and category not in captures and eligible) or fork is not None:
                identity=dict(**row,category=category,prefix_keys=list(keys),state_sha256=provenance._environment_digest(env),
                    guard_sha256=provenance._guard_digest(agent),candidate_keys=[r.key for r in bank.records],
                    q_values=list(decision.q_values),alternative_key=bank.records[alternative].key if alternative is not None else None)
                if fork is None:
                    captures[category]=identity
                else:
                    require(eligible,'Expected available disagreement disappeared')
                    for k in ('native_key','state_sha256','guard_sha256','candidate_keys','alternative_key','q_values'):
                        require(identity[k]==fork[k],f'Counterfactual fork differs: {k}')
                    decision=force(agent,decision,snapshot,alternative)
                    interventions+=1
            if index%10==0:
                print(f'E127 {"branch" if fork else "reference"} | decision={index} step={env.time_steps} '
                      f'delivered={sum(b.delivered for b in env.blocks)}/36 attribution={label}',flush=True)
        keys.append(decision.candidate.key)
        return decision
    with cache_session(),source.implementation(),patch.object(agent,'select',select),torch.no_grad():
        run=source.original.episode(agent,ARM,SEED,False)
    require(source.original.replay.parameter_digest(agent)==before and agent.gradient_steps==before_updates
            and len(agent.replay)==before_replay,
            'Frozen diagnostic changed weights or performed learning')
    require(interventions==(1 if fork else 0),'Incorrect number of interventions')
    return dict(run=run,rows=rows,captures=captures,executed_keys=keys,interventions=interventions,
                checkpoint_unchanged=sha256(CHECKPOINT)==contract()['checkpoint_sha256'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run-all','analyze'))
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output=args.output.resolve()
    prepare(output)
    if args.command=='run-all':
        native=output/'reference.json'
        if not native.exists(): write(native,run_episode(output))
        reference=read(native)
        require(reference['run']['strict_method_success'] and reference['checkpoint_unchanged'],
                'Reference must complete strictly with unchanged source before branching')
        for category,fork in reference['captures'].items():
            path=output/f'branch-{category}.json'
            if not path.exists(): write(path,run_episode(output,fork))
    reference=read(output/'reference.json')
    branches=[]
    for category,fork in reference['captures'].items():
        path=output/f'branch-{category}.json'
        if path.exists():
            row=read(path)
            branches.append(dict(category=category,decision_index=fork['decision_index'],time_steps=fork['time_steps'],
                attribution=fork['attribution'],native_key=fork['native_key'],alternative_key=fork['alternative_key'],
                reference_J=reference['run']['operator_cost'],branch_J=row['run']['operator_cost'],
                delta_J=row['run']['operator_cost']-reference['run']['operator_cost'],
                strict_complete=row['run']['strict_method_success'],interventions=row['interventions'],
                branch_service=row['run']['service_metrics']))
    golden=next(v for v in informed.ref.source_data()[1] if v['checkpoint_episode']==120)['runs'][0]
    report=dict(protocol=PROTOCOL,contract_sha256=contract()['contract_sha256'],status='complete',
        source_checkpoint=120,instance_seed=SEED,reference_J=reference['run']['operator_cost'],
        source_validation_J=golden['operator_cost'],
        source_outcome_matches=(reference['run']['operator_cost']==golden['operator_cost']
                               and reference['run']['steps']==golden['steps']),
        reference_strict_complete=reference['run']['strict_method_success'],
        attribution_counts=dict(Counter(r['attribution'] for r in reference['rows'])),
        source_stage_counts=dict(Counter(r['native_action'] for r in reference['rows'])),branches=branches,
        checkpoint_unchanged=sha256(CHECKPOINT)==contract()['checkpoint_sha256'],training_runs=0,
        episode_runs=1+len(branches),paper_evidence=False,independent_test_opened=False,
        caveat='one instance, first disagreement per category, single certified intervention; fixed continuation, not global optimality')
    write(output/'e127-report.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
