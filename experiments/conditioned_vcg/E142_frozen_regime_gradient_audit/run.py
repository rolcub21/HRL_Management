"""Frozen, balanced replay diagnostic. Never calls learn or optimizer.step."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'results/vcg-conditioned-e142-frozen-regime-gradient-audit'


def replay_labels(history, memory):
    """Reconstruct FIFO suffix only when episode/terminal accounting agrees."""
    labels, terminals = [], []
    for row in history:
        count = int(row['decisions'])
        if count < 1 or not row['strict_complete']:
            raise ValueError('History must contain completed nonempty episodes')
        labels.extend([int(row['regime'])] * count)
        terminals.extend([False] * (count - 1) + [True])
    if not memory or len(memory) > len(labels):
        raise ValueError('Replay cannot be explained by completed history')
    if [bool(t.done) for t in memory] != terminals[-len(memory):]:
        raise ValueError('Replay terminal boundaries disagree with FIFO history')
    return labels[-len(memory):]


def digest(agent):
    h = hashlib.sha256()
    for net in (agent.Q_local, agent.Q_target):
        for key, value in net.state_dict().items():
            h.update(key.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def gradient(loss, parameters):
    values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten().cpu()
                      for p, g in zip(parameters, values)])


def distribution(values):
    values = values.detach().float().cpu().flatten()
    return dict(mean=float(values.mean()), p50=float(values.quantile(.5)),
                p90=float(values.quantile(.9)), maximum=float(values.max()))


def report(result):
    lines = ['# Frozen regime replay audit', '',
             f"Seed {result['model_seed']}; checkpoint episode {result['completed_episodes']}; "
             f"status: {result['status']}.", '',
             'No training or optimizer updates. Gradients below are before clipping.', '',
             '| Regime | Episodes | Replay records | Native sampler share | Mean TD loss | Mean gradient norm |',
             '|---|---:|---:|---:|---:|---:|']
    names = ['empty / .60', 'empty / .85', '40% / .60', '40% / .85', '~70% / .60', '~70% / .85']
    sampled = {int(k): v for k, v in result['mode_balanced_sample_counts'].items()}
    episodes = {int(k): v for k, v in result['episode_counts'].items()}
    replay = {int(k): v for k, v in result['replay_counts'].items()}
    for regime, name in enumerate(names):
        rows = [b for b in result['batches'] if b['regime'] == regime]
        mean = lambda key: sum(b[key] for b in rows)/len(rows) if rows else float('nan')
        lines.append(f"| {name} | {episodes.get(regime, 0)} | "
                     f"{replay.get(regime, 0)} | "
                     f"{100*sampled.get(regime, 0)/sum(sampled.values()):.1f}% | "
                     f"{mean('td_loss'):.2f} | {mean('preclip_gradient_norm'):.2f} |")
    lines += ['', '## Shared mean-gradient alignment', '',
              '| Regimes (zero-based) | Cosine |', '|---|---:|']
    for row in result['gradient_cosines']:
        value = row['shared_mean_gradient_cosine']
        lines.append(f"| {row['a']} vs {row['b']} | {value:.3f} |" if value is not None
                     else f"| {row['a']} vs {row['b']} | undefined (zero norm) |")
    lines += ['', 'Small frozen batches are diagnostic, not evidence of historical gradient domination.',
              'Different sampled actions/frontier sizes remain within each regime; no causal claim.',
              'Full per-head errors, saturation, anchor gradients and invariance checks: result.json.']
    return '\n'.join(lines)+'\n'


def source(seed):
    if seed == 0:
        return ROOT/'results/vcg-conditioned-e138-residual-grid-mixture-pilot-202k-v1-repair1/seed-0/latest.pth'
    return ROOT/f'results/vcg-conditioned-e139-grid-mixture-three-seeds-203k-v1/seed-{seed}/latest.pth'


def audit(args):
    from experiments.conditioned_vcg.E138_residual_grid_mixture_pilot_202k_v1.learner import MixtureAgent
    from experiments.conditioned_vcg.E139_grid_mixture_three_seeds_203k_v1.run import agent_class
    from experiments.conditioned_vcg.E115_release_et_cost_vcg_8x8_179k_v1.model import (
        COMPONENT_NAMES, COMPONENT_LOSS_WEIGHTS, ANCHOR_LOSS_WEIGHT)
    from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v2.acceleration import cache_session

    torch.set_num_threads(args.threads)
    path = source(args.model_seed)
    identity = file_sha(path)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    cls = MixtureAgent if args.model_seed == 0 else agent_class(args.model_seed)
    agent = cls.from_checkpoint(payload, device=args.device, resumable=True)
    memory = list(agent.replay.memory)
    history = payload['training_history']
    labels = replay_labels(history, memory)
    if agent.transition_count != sum(r['decisions'] for r in history):
        raise ValueError('Transition count differs from completed history')
    before = digest(agent)
    counters = (agent.gradient_steps, agent.target_updates, agent.decision_count,
                agent.transition_count, len(agent.replay), agent.rng.getstate())
    named = list(agent.Q_local.named_parameters())
    parameters = [p for _, p in named]
    head_names = ('service_head', 'wait_head', 'step_head', 'rehandle_head')
    shared_mask = torch.cat([torch.full((p.numel(),), not any(h in n for h in head_names),
                                       dtype=torch.bool) for n, p in named])
    rng = random.Random(args.audit_seed)
    # Estimate the actual mode-balanced sampler, not just uniform replay shares.
    by_id = {id(t): r for t, r in zip(memory, labels)}
    sampled = Counter()
    for _ in range(200):
        sampled.update(by_id[id(t)] for t in agent.replay.sample_mode_balanced(
            min(agent.config.batch_size, len(memory)), rng))
    result = dict(status='running', model_seed=args.model_seed,
        checkpoint=str(path), checkpoint_sha256=identity,
        completed_episodes=len(history), no_optimizer_updates=True, test_panel_opened=False,
        scope='current frozen checkpoint/replay; not historical gradient attribution or a learning curve',
        sampling=dict(batch_size=args.batch_size, batches_per_regime=args.batches,
                      audit_seed=args.audit_seed, order='shuffled regimes each round'),
        episode_counts=dict(Counter(r['regime'] for r in history)),
        replay_counts=dict(Counter(labels)), mode_balanced_sample_counts=dict(sampled),
        episode_costs={r: [h['metrics']['J'] for h in history if h['regime'] == r]
                       for r in range(6)}, batches=[], gradient_cosines=[])
    out = args.output/f'seed-{args.model_seed}'
    if out.exists():
        raise ValueError('Output already exists; use a new --output for a new audit')
    out.mkdir(parents=True)
    def save():
        tmp = out/'result.json.tmp'
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        tmp.replace(out/'result.json')
    save()
    started = time.monotonic()
    means = {r: [] for r in range(6)}
    try:
        with cache_session():
            for repeat in range(args.batches):
                regimes = list(range(6))
                rng.shuffle(regimes)
                for regime in regimes:
                    pool = [i for i, r in enumerate(labels) if r == regime]
                    if len(pool) < args.batch_size:
                        raise ValueError(f'Regime {regime} has too few retained records')
                    indices = rng.sample(pool, args.batch_size)
                    prediction, target, anchor, pairs = agent._td_batch([memory[i] for i in indices])
                    losses = [F.smooth_l1_loss(prediction[:, i], target[:, i],
                              beta=agent.config.huber_delta) for i in range(4)]
                    td = sum(w*l for w, l in zip(COMPONENT_LOSS_WEIGHTS, losses))
                    total = td + ANCHOR_LOSS_WEIGHT*anchor
                    g = gradient(total, parameters)
                    means[regime].append(g[shared_mask])
                    per_head = {}
                    for i, name in enumerate(COMPONENT_NAMES):
                        error = (prediction[:, i]-target[:, i]).detach()
                        hg = gradient(COMPONENT_LOSS_WEIGHTS[i]*losses[i], parameters)
                        per_head[name] = dict(abs_td=distribution(error.abs()),
                            abs_target=distribution(target[:, i].abs()),
                            huber_loss=float(losses[i].detach()),
                            saturated_fraction=float((error.abs() >= agent.config.huber_delta).float().mean()),
                            weighted_gradient_norm=float(hg.norm()),
                            shared_gradient_norm=float(hg[shared_mask].norm()))
                    ag = gradient(ANCHOR_LOSS_WEIGHT*anchor, parameters) if anchor.requires_grad else torch.zeros_like(g)
                    result['batches'].append(dict(regime=regime, repeat=repeat,
                        replay_indices=indices, candidate_records=sum(len(memory[i].next_candidates) for i in indices),
                        heads=per_head, td_loss=float(td.detach()), anchor_loss=float(anchor.detach()),
                        anchor_pairs=int(pairs), anchor_gradient_norm=float(ag.norm()),
                        total_loss=float(total.detach()), preclip_gradient_norm=float(g.norm()),
                        shared_gradient_norm=float(g[shared_mask].norm()),
                        grad_clip=agent.config.grad_clip))
                    save()
                    print(f'E142 seed={args.model_seed} batch={repeat+1}/{args.batches} '
                          f'regime={regime} loss={float(total.detach()):.3f} '
                          f'gradient={float(g.norm()):.3f} elapsed={time.monotonic()-started:.1f}s', flush=True)
                    del prediction, target, anchor, losses, td, total
        vectors = {r: torch.stack(v).mean(0) for r, v in means.items()}
        for a in range(6):
            for b in range(a+1, 6):
                denominator = float(vectors[a].norm()*vectors[b].norm())
                result['gradient_cosines'].append(dict(a=a, b=b,
                    shared_mean_gradient_cosine=float(torch.dot(vectors[a], vectors[b]))/denominator
                    if denominator else None))
        result['status'] = 'complete'
    finally:
        result['parameters_unchanged'] = digest(agent) == before
        result['counters_replay_length_rng_unchanged'] = counters == (
            agent.gradient_steps, agent.target_updates, agent.decision_count,
            agent.transition_count, len(agent.replay), agent.rng.getstate())
        result['checkpoint_file_unchanged'] = file_sha(path) == identity
        result['wall_seconds'] = time.monotonic()-started
        if not all(result[k] for k in ('parameters_unchanged',
                'counters_replay_length_rng_unchanged', 'checkpoint_file_unchanged')):
            result['status'] = 'invariance_failure'
        save()
        (out/'REPORT.md').write_text(report(result))
    print(json.dumps({k: result[k] for k in ('status', 'completed_episodes', 'wall_seconds')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-seed', type=int, choices=(0,1,2), default=0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--batches', type=int, default=3)
    parser.add_argument('--audit-seed', type=int, default=142)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if min(args.batch_size, args.batches, args.threads) < 1:
        parser.error('batch size, batches and threads must be positive')
    audit(args)


if __name__ == '__main__':
    main()
