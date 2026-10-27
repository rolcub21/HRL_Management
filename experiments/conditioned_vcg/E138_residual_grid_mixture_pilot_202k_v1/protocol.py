from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT/'results/vcg-conditioned-e138-residual-grid-mixture-pilot-202k-v1-repair1'
MAX_EPISODE = 60
TRAIN_BASE = 21000000
DEV_BASE = 22000000
EPSILON_DECISIONS = 6000


def epsilon(decisions):
    return max(.05, .9-(.9-.05)*max(0, decisions)/EPSILON_DECISIONS)


def regime(number):
    if not 1 <= number <= MAX_EPISODE:
        raise ValueError('episode outside one-seed pilot budget')
    block, index = divmod(number-1, 6)
    order = list(range(6))
    random.Random(138000+block).shuffle(order)
    return order[index]


def specification():
    return dict(protocol='vcg_conditioned_e138_residual_grid_mixture_pilot_202k_v1',
        implementation='checkpoint_metadata_collision_repair1',
        seed=0, initialization='scratch; never initialize from old checkpoints',
        maximum_episode=60, mixture='one deterministic shuffled permutation of six regimes per six episodes',
        training_seeds=[TRAIN_BASE+i for i in range(1,61)],
        greedy_development_seeds=[DEV_BASE+i for i in range(6)],
        test_seeds=None, test_panel_opened=False, stage_b_authorized=False,
        objective='J=2*T+E+0.1*L+10*R, external release-fixed due dates, tolerance 20',
        wait_cost='evaluation-only J_W=J+0.5*sum admission wait; no training wait head',
        learner='E131 signed four-head residual; shared selected successor for all Double-DQN heads',
        epsilon=dict(start=.9, floor=.05, decay_training_decisions=6000),
        validation_every=6, development_instances_per_regime=1,
        checkpoint_rule='save every completed episode and every scheduled greedy endpoint; '
                        'report last three endpoints 48/54/60, do not select best for paper',
        runtime=dict(training_episode_wall_seconds=1200, greedy_episode_wall_seconds=300,
                     horizon=6000, decisions_per_episode=1000, certificate_max_nodes=2000),
        stopping_rule='stop on any incomplete training or greedy episode, timeout, nonfinite loss, '
                      'or absolute TD error >1e6; no automatic retries or budget changes',
        checkpoint_recovery='rollback incomplete episode to last completed checkpoint including replay/optimizer/RNG',
        claims='development pilot only; empirical completion cap, no proved uniform completion bound; '
               'baseline macro mismatch remains unresolved and is not a training blocker',
        existing_e137_certified_timeout='retained, explicitly not a blocker to this bounded pilot')
