"""Validation-selected posterior probes with incremental-memory controls and fresh test layouts."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from analysis.evaluate_belief_decoding import collect, load_cached_rollouts, episode_weights, paired_interval
from analysis.evaluate_belief_consistency import history_features, random_memory


CANDIDATES = [('linear', .001), ('linear', .1), ('mlp', .001), ('mlp', .1)]


def subset(data, mask):
    return {k: v if k == 'cells' else v[mask] for k, v in data.items()}


def region_targets(belief, cells, size):
    """Four fixed world-coordinate quadrants, not privileged goal labels."""
    region = (cells[:, 0] >= size / 2).astype(int) * 2 + (cells[:, 1] >= size / 2).astype(int)
    return np.stack([belief[:, region == i].sum(-1) for i in range(4)], axis=1)


def features(data, random, size):
    basic = np.c_[data['observation'], data['pose'], data['timestep'] / 400.]
    history = np.c_[basic, history_features(data, size)]
    return dict(observation=basic, history=history,
                history_memory=np.c_[history, data['hidden']], history_random=np.c_[history, random])


def model(input_dim, output_dim, kind, budget):
    if kind == 'linear':
        return nn.Linear(input_dim, output_dim)
    width = max(1, round((budget - output_dim) / (input_dim + output_dim + 1)))
    return nn.Sequential(nn.Linear(input_dim, width), nn.ReLU(), nn.Linear(width, output_dim))


def fit_candidate(x, target, data, args, seed, kind, decay):
    """Only training/validation records affect normalization, fitting or selection."""
    train = data['split'] == 0
    valid = (data['split'] == 1) & (data['belief'].max(-1) < 1)
    if not train.any() or not valid.any():
        raise ValueError('Need training and unresolved validation records.')
    mean, scale = x[train].mean(0), x[train].std(0).clip(1e-5)
    xt = torch.tensor((x[train] - mean) / scale, dtype=torch.float32)
    xv = torch.tensor((x[valid] - mean) / scale, dtype=torch.float32)
    yt, yv = torch.tensor(target[train], dtype=torch.float32), torch.tensor(target[valid], dtype=torch.float32)
    wt = torch.tensor(episode_weights(data['episode'][train]), dtype=torch.float32)
    wt /= wt.mean()
    wv = torch.tensor(episode_weights(data['episode'][valid]), dtype=torch.float32)
    torch.manual_seed(seed)
    net = model(x.shape[1], target.shape[1], kind, args.parameter_budget)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=decay)
    best, state, selected, history = float('inf'), None, 0, []
    entropy = -(yv * yv.clamp_min(1e-30).log()).sum(-1)
    for epoch in range(1, args.epochs + 1):
        net.train()
        for indices in torch.randperm(len(xt)).split(args.batch_size):
            ce = -(yt[indices] * net(xt[indices]).log_softmax(-1)).sum(-1)
            loss = (ce * wt[indices]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            kl = -(yv * net(xv).log_softmax(-1)).sum(-1) - entropy
            score = float((kl * wv).sum() / wv.sum())
        history.append(score)
        if score < best:
            best, state, selected = score, copy.deepcopy(net.state_dict()), epoch
        if args.patience and epoch - selected >= args.patience:
            break
    return dict(kind=kind, decay=decay, input_dim=x.shape[1], output_dim=target.shape[1],
                parameter_budget=args.parameter_budget, parameters=sum(p.numel() for p in net.parameters()),
                mean=torch.tensor(mean, dtype=torch.float32), scale=torch.tensor(scale, dtype=torch.float32), state_dict=state,
                validation_kl=best, selected_epoch=selected, epochs_run=len(history), history=history)


@torch.no_grad()
def predict(fitted, x, target):
    net = model(fitted['input_dim'], fitted['output_dim'], fitted['kind'], fitted['parameter_budget'])
    net.load_state_dict(fitted['state_dict'])
    net.eval()
    x = (torch.tensor(x, dtype=torch.float32) - fitted['mean']) / fitted['scale']
    y = torch.tensor(target, dtype=torch.float32)
    logp = net(x).log_softmax(-1)
    kl = (y * (y.clamp_min(1e-30).log() - logp)).sum(-1)
    return kl.numpy(), logp.exp().numpy()


def known_layouts(root):
    """Available previous rollout layouts, plus reconstructed forced-history resets."""
    layouts, sources = set(), []
    for path in sorted(root.glob('**/*rollouts.npz')):
        with np.load(path, allow_pickle=False) as saved:
            if 'layout' in saved.files:
                layouts.update(map(tuple, np.unique(saved['layout'], axis=0)))
                sources.append(str(path))
    # Forced-history artifacts lack layout arrays; recover their initial conditions.
    from environment import make_partial_obs_learning_env
    resets = set()
    for path in root.glob('**/matched_traces.npz'):
        config = json.loads(path.with_name('summary.json').read_text())['config']
        resets.update((config['size'], config['view_size'], seed)
                      for seed in range(config['seed'], config['seed'] + config['episodes']))
    envs = {}
    try:
        for size, view, seed in sorted(resets):
            env = envs.setdefault((size, view), None)
            if env is None:
                env = envs[size, view] = make_partial_obs_learning_env(size=size, num_distractors=0, agent_view_size=view)
            env.reset(seed=seed)
            base = env.unwrapped
            goal = next((x,y) for x in range(1,size-1) for y in range(1,size-1)
                        if (obj := base.grid.get(x,y)) is not None and obj.type == 'goal')
            layouts.add((*goal, *base.agent_pos, base.agent_dir))
    finally:
        for env in envs.values():
            env.close()
    return layouts, sources


def fresh_subset(data, excluded, episodes):
    chosen, used = [], set(excluded)
    for episode in np.unique(data['episode']):
        layout = tuple(data['layout'][data['episode'] == episode][0])
        if layout not in used:
            chosen.append(episode)
            used.add(layout)
        if len(chosen) == episodes:
            break
    if len(chosen) < episodes:
        raise ValueError('Insufficient fresh layouts. No test scores computed; increase collection budget in a new run.')
    selected = subset(data, np.isin(data['episode'], chosen))
    selected['split'][:] = 2
    return selected


def run(args):
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError('Choose a new output directory.')
    source = Path(args.rollouts)
    metadata = json.loads(source.with_name('summary.json').read_text())
    config = SimpleNamespace(**metadata['config'])
    config.cached_rollouts = str(source)
    config.split_seed = metadata['config'].get('split_seed', metadata['config']['probe_seed'])
    old = load_cached_rollouts(config)
    dev = subset(old, old['split'] != 2)
    torch.set_num_threads(1)
    excluded, sources = known_layouts(Path(args.artifacts_root))
    # Freeze exclusion and decision rules before fitting or collecting new test episodes.
    excluded.update(map(tuple, old['layout']))
    out.mkdir(parents=True)
    protocol = dict(config=vars(args), checkpoint_sha256=metadata['checkpoint_sha256'],
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    candidates=CANDIDATES, targets=['full', 'quadrants'],
                    primary='Full posterior KL; quadrants are secondary.',
                    rule='For each target: average per-step KL over fitting seeds, then bootstrap fresh layouts. Signal requires positive lower 95% bounds versus both history and history+random, and positive mean gains for every fitting seed. If any winning memory/control fit selects in final 10% of epoch budget, mark signal budget-limited. Cross-policy replication must be assessed separately.',
                    exclusions=sources, excluded_layouts=len(excluded),
                    limitation='Fresh relative to available analysis artifacts, not guaranteed unseen during IQ training. Selection performed on previously used development layouts. No causal claim.')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    np.save(out / 'excluded_layouts.npy', np.array(sorted(excluded)))
    targets = dict(full=dev['belief'], quadrants=region_targets(dev['belief'], dev['cells'], config.size))
    selection = {}
    for seed in args.seeds:
        print(f'Development: seed {seed}', flush=True)
        inputs = features(dev, random_memory(dev, config, seed), config.size)
        for target_name, target in targets.items():
            for name, x in inputs.items():
                print(f'  Selecting {target_name}/{name}', flush=True)
                fits = [fit_candidate(x, target, dev, args, seed, kind, decay) for kind, decay in CANDIDATES]
                winner = min(fits, key=lambda f:f['validation_kl'])
                key = f'seed{seed}_{target_name}_{name}'
                torch.save(winner, out / f'{key}.pt')
                selection[key] = [{k:v for k,v in f.items() if k not in ('mean','scale','state_dict')} for f in fits]
    (out / 'selection.json').write_text(json.dumps(selection, indent=2, allow_nan=False))
    print('All choices frozen. Collecting fresh test histories.', flush=True)
    fresh_args = SimpleNamespace(**vars(config))
    fresh_args.episodes, fresh_args.rollout_seed = args.test_episodes + 100, args.test_seed
    fresh = fresh_subset(collect(fresh_args), excluded, args.test_episodes)
    unresolved = fresh['belief'].max(-1) < 1
    if not unresolved.any():
        raise ValueError('No unresolved fresh test records.')
    np.savez_compressed(out / 'fresh_rollouts.npz', **fresh)
    results, predictions, per_seed = {}, {}, {}
    weights = episode_weights(fresh['episode'][unresolved])
    for seed in args.seeds:
        inputs = features(fresh, random_memory(fresh, config, seed), config.size)
        for target_name, target in dict(full=fresh['belief'], quadrants=region_targets(fresh['belief'], fresh['cells'], config.size)).items():
            for name, x in inputs.items():
                key = f'seed{seed}_{target_name}_{name}'
                fitted = torch.load(out / f'{key}.pt', weights_only=True)
                kl, prediction = predict(fitted, x, target)
                per_seed[key], predictions[key] = kl, prediction
                results[key] = dict(test_kl=float(np.average(kl[unresolved], weights=weights)),
                                    kind=fitted['kind'], decay=fitted['decay'], parameters=fitted['parameters'],
                                    selected_epoch=fitted['selected_epoch'],
                                    selected_near_budget=fitted['selected_epoch'] >= .9*args.epochs)
    comparisons = {}
    for target in targets:
        item = {}
        for control in ('history', 'history_random'):
            deltas = [per_seed[f'seed{seed}_{target}_{control}'] - per_seed[f'seed{seed}_{target}_history_memory'] for seed in args.seeds]
            effect = paired_interval(np.mean(deltas, axis=0), fresh, unresolved, 0)
            effect['per_seed_mean'] = [float(np.average(d[unresolved], weights=weights)) for d in deltas]
            item[control] = effect
        signal = all(v['ci95'][0] > 0 and min(v['per_seed_mean']) > 0 for v in item.values())
        budget = any(results[f'seed{seed}_{target}_{name}']['selected_near_budget'] for seed in args.seeds for name in ('history','history_random','history_memory'))
        item['decision'] = 'budget_limited_signal' if signal and budget else ('signal_detected' if signal else 'not_established')
        comparisons[target] = item
    report = dict(protocol=protocol, comparisons=comparisons, models=results,
                  test=dict(episodes=len(np.unique(fresh['episode'])), unresolved_episodes=len(np.unique(fresh['episode'][unresolved])),
                            unresolved_steps=int(unresolved.sum()), rollout_sha256=hashlib.sha256((out/'fresh_rollouts.npz').read_bytes()).hexdigest()),
                  interpretation='Predictive information beyond controls, not causal use. Intervals resample layouts conditional on fitted models; seed averaging does not create independent samples.')
    (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    np.savez_compressed(out / 'predictions.npz', **predictions)
    plot(report, out)
    print(json.dumps(comparisons, indent=2), flush=True)


def plot(report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax, target in zip(axes, ('full','quadrants')):
        for i, name in enumerate(('observation','history','history_random','history_memory')):
            values = [v['test_kl'] for k,v in report['models'].items() if k.endswith(f'_{target}_{name}')]
            ax.scatter([i]*len(values),values)
        ax.set_xticks(range(4),['Observation','History','History + random','History + memory'], rotation=20)
        ax.set_title(target + ': ' + report['comparisons'][target]['decision'])
        ax.set_ylabel('Fresh-test posterior KL (nats); lower is better')
    fig.savefig(out / 'decoder_comparison.png',dpi=180)
    fig.savefig(out / 'decoder_comparison.pdf')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollouts',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--artifacts-root',default='artifacts')
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--test-seed',type=int,default=700000)
    p.add_argument('--test-episodes',type=int,default=200)
    p.add_argument('--epochs',type=int,default=300)
    p.add_argument('--patience',type=int,default=40)
    p.add_argument('--parameter-budget',type=int,default=20000)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--lr',type=float,default=.001)
    args=p.parse_args()
    if min(args.test_episodes,args.epochs,args.parameter_budget,args.batch_size)<1 or args.patience<0 or not np.isfinite(args.lr) or args.lr<=0:
        p.error('Invalid fitting settings.')
    if len(set(args.seeds))!=len(args.seeds) or min(args.seeds)<0:
        p.error('Fitting seeds must be unique and nonnegative.')
    run(args)


if __name__ == '__main__':
    main()
