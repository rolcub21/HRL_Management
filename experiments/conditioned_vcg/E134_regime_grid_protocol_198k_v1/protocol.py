"""Prospective choices, frozen before any grid test or contender-quality run."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUTPUT=ROOT/'results/vcg-conditioned-e134-regime-grid-protocol-198k-v1'
PROTOCOL='vcg_conditioned_e134_regime_grid_protocol_198k_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def specification():
    return dict(protocol=PROTOCOL,status='prospective_design_and_rescorer_only',yard='8x8',
        primary=dict(formula='J=2*T+E+0.1*L+10*R',tolerance=20,clock='external release-fixed due date',
            L='equipment primitive actions excluding WAIT',completion='constraint, never a priced failure penalty',
            used_for=['training','development primary comparator selection','primary test claims']),
        wait_sensitivity=dict(formula='J_W=J+0.5*sum_i(W_i)',weight_per_job_step=.5,
            W='first completed storage time minus release time, for incoming jobs only',
            interpretation='includes travel/handling during admission; not pure stationary pickup waiting',
            mean_wait_denominator='incoming jobs only; initial stored jobs must not dilute the mean',
            initial_inventory='zero in-window admission wait; no prehistory charge',
            interrupted='observed wait until horizon for arrived/unadmitted jobs, explicitly right-censored',
            unreleased='zero observed wait; keep in completion denominator',
            applied_to='every contender and every fixed test trajectory; no rescoring-driven policy/checkpoint selection',
            use='evaluation-only secondary sensitivity; no retraining or fifth head',
            comparator='same comparator frozen on primary development J; do not select a new test winner on J_W'),
        regimes=[dict(initial_fill=f,nominal_offered_load=l) for f in (0.,.4,.7) for l in (.6,.85)],
        initial_inventory='derive rounded counts from actual legal capacity; record achieved fill',
        incoming_jobs='same incoming-stream length across all cells; initial jobs are additional workload',
        load_calibration=dict(source='reserved generator/reference calibration data; no contender results',
            definition='reference equipment work attributable to incoming jobs per incoming job / arrival interval',
            initial_inventory='report initial-clearance work separately, plus finite-episode total work; do not call total burden steady utilization',
            procedure='bounded iterative recalibration by fill/load because interval changes may change work; freeze intervals and stopping rule before contender evaluation',
            achieved='report achieved method utilization separately; do not assume work increases monotonically with fill'),
        due_generation=dict(formula='d_i=C_i_reference+Uniform[0,20]',
            witness='independent full-information reference primitive schedule; hidden from contenders',
            independence='no VCG certificate/witness/guard; no contender scheduler or retrieval-plan reuse',
            verifier='separate primitive trace replay validates legal moves, completion, fixed horizon and timing window',
            caveat='proves simulator feasibility, not absence of reference-induced distribution bias',
            reference_quality=['equipment work','rehandles','admission wait','completion time','transport-work lower bound']),
        methods=['residual_vcg','informed_dynamic_pslap','informed_ga','certified_dynamic_pslap'],
        learned=dict(training_seeds=[0,1,2],policy='one mixture-trained model per seed covering all six regimes',
            architecture='four-head E131 residual design; no optional auxiliary wait head',stage_b='deferred'),
        interfaces=dict(all='same information, executable macro library, objective and horizon',
            certified_pair='identical actual certificate, witness guard and Defer restrictions, not just similarly named configurations',
            interpretation='plain and certified methods may have different eligibility filtering; do not claim identical eligible sets',
            ga='fixed compute budget before development comparisons'),
        tests=dict(instances_per_cell=20,separate_from=['calibration','training','development'],
            sealed=True,generation_or_execution_authorized=False),
        analysis=dict(noninferiority_margin_fraction=.05,noninferiority_cells='two zero-fill cells',
            completion='all methods/all cells, failures and timeouts in denominator',
            costs='common-completion subset with sample size and exclusion coverage explicitly reported',
            baseline_selection='development-only by cell; frozen before test',
            multiplicity='predeclare hypothesis families before execution; Holm across six primary regime contrasts',
            reporting='every cell and every training seed, including losses; separate J and J_W and ranking reversals',
            uncertainty='condition on trained models for paired instance intervals; do not count repeated seeds/checkpoints as independent instances'),
        remaining_execution_gates=['independent reference implementation and primitive verifier',
            'calibration tolerance/iteration budget and measured intervals',
            'incoming-stream length, horizon and explicit initial-state generator',
            'certificate/guard/Defer and information-access alignment',
            'mixture training budget, exploration, sampling and checkpoint rule',
            'GA compute budget and development comparator selection rule',
            'disjoint seed manifests and complete statistical analysis specification'],
        training_launch_authorized=False,test_panel_opened=False)


def contract():
    value=specification()
    paths=[Path(__file__),Path(__file__).with_name('objective.py'),Path(__file__).with_name('run.py'),
           ROOT/'tests/test_vcg_e134_wait_sensitivity.py']
    value['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return {**value,'contract_sha256':digest(value)}
