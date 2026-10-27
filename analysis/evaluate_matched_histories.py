"""Controlled, equal-duration histories at identical observations and physical states.

Forced navigation loops test history sensitivity; they do not isolate a causal
belief representation or constitute a reward-optimal belief planner.
"""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from analysis.discrete_goal_belief import DiscreteGoalBelief, select_action
from analysis.policy_io import add_policy_args, load_policy, representation
from environment import make_partial_obs_learning_env


def histories(distance, control=False):
    if control:
        return [0] * (4 * (distance + 1)), [1] * (4 * (distance + 1))
    return ([2] * distance + [0]) * 4, ([2] * distance + [1]) * 4


@torch.no_grad()
def replay(env, initial_obs, actions, agent):
    """Consume each observation once, including the shared endpoint observation."""
    env = deepcopy(env)
    try:
        belief = DiscreteGoalBelief.from_env(env)
        obs, hidden = initial_obs.copy(), None
        observations, positions = [], []
        for step in range(len(actions) + 1):
            belief.observe(env)
            hidden = representation(agent, obs, hidden)
            observations.append(np.asarray(obs).copy())
            positions.append((*env.unwrapped.agent_pos, env.unwrapped.agent_dir))
            if step == len(actions):
                q = agent.q_net(hidden)[0]
                # Temperature is passed by the caller; save Q without changing it.
                scores, _ = belief.action_scores(env)
                return dict(obs=np.asarray(obs).copy(), pose=positions[-1],
                            timestep=env.unwrapped.step_count,
                            hidden=hidden[0].numpy().copy(), q=q.numpy().copy(),
                            candidates=tuple(sorted(belief.candidates)),
                            ig=np.array([scores[a] for a in range(3)]),
                            observations=np.asarray(observations), positions=positions)
            obs, _, terminated, truncated, _ = env.step(actions[step])
            if terminated or truncated:
                return None
    finally:
        env.close()


def matched(a, b, initial_pose, initial_obs):
    if a is None or b is None:
        return False
    return (a['pose'] == b['pose'] == initial_pose
            and a['timestep'] == b['timestep']
            and np.array_equal(a['obs'], b['obs'])
            and np.array_equal(a['obs'], initial_obs))


def policy_probabilities(q, temperature):
    z = np.asarray(q, dtype=np.float64) / temperature
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()


def pair_metrics(a, b, temperature):
    pa, pb = (policy_probabilities(x['q'], temperature) for x in (a, b))
    aa, ab = int(np.argmax(a['q'])), int(np.argmax(b['q']))
    ca, cb = set(a['candidates']), set(b['candidates'])
    belief_tv = .5 * sum(abs((1 / len(ca) if x in ca else 0) -
                            (1 / len(cb) if x in cb else 0)) for x in ca | cb)
    ia, ta = select_action(dict(enumerate(a['ig'])))
    ib, tb = select_action(dict(enumerate(b['ig'])))
    eligible = len(ta) == len(tb) == 1 and ia != ib and min(len(ca), len(cb)) > 1
    return dict(candidates_a=len(ca), candidates_b=len(cb),
                unresolved=len(ca) > 1 and len(cb) > 1,
                same_belief=ca == cb, belief_tv=belief_tv,
                identical_observation_history=np.array_equal(a['observations'], b['observations']),
                policy_tv=float(.5 * np.abs(pa - pb).sum()),
                action_a=aa, action_b=ab, action_changed=aa != ab,
                outside_navigation_a=aa > 2, outside_navigation_b=ab > 2,
                ig_action_a=ia, ig_action_b=ib,
                unique_ig_a=len(ta) == 1, unique_ig_b=len(tb) == 1,
                different_unique_ig=eligible,
                both_actions_match_ig=(aa == ia and ab == ib) if eligible else None,
                own_ig_support_shift=float(.5 * ((pa[ia] - pb[ia]) + (pb[ib] - pa[ib]))) if eligible else None)


def summarize(rows, seed):
    """Seed-cluster intervals; several distances share each reset layout."""
    result = {}
    rng = np.random.default_rng(seed)
    for name, subset in [
        ('different_belief', [r for r in rows if r['kind'] == 'loop' and r['unresolved'] and not r['same_belief']]),
        ('same_belief_control', [r for r in rows if r['kind'] == 'rotation' and r['unresolved'] and r['same_belief'] and not r['identical_observation_history']]),
        ('different_unique_ig', [r for r in rows if r['kind'] == 'loop' and r['different_unique_ig']]),
    ]:
        seeds = sorted({r['reset_seed'] for r in subset})
        item = dict(pairs=len(subset), reset_seeds=len(seeds))
        keys = ['policy_tv', 'action_changed']
        if name == 'different_unique_ig':
            keys += ['both_actions_match_ig', 'own_ig_support_shift']
        for key in keys:
            if not subset:
                item[key] = dict(mean=None, ci95=None)
                continue
            # Equal reset-seed weighting; all distances within a seed kept together.
            values = np.array([np.mean([r[key] for r in subset if r['reset_seed'] == s]) for s in seeds])
            draws = rng.integers(len(seeds), size=(2000, len(seeds)))
            interval = np.quantile(values[draws].mean(1), [.025, .975]).tolist() if len(seeds) > 1 else None
            item[key] = dict(mean=float(values.mean()), ci95=interval)
        result[name] = item
    return result


def run(args):
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(f'Choose a new output directory: {out}')
    torch.set_num_threads(1)
    env = make_partial_obs_learning_env(size=args.size, num_distractors=0,
                                       agent_view_size=args.view_size, max_steps=args.max_steps)
    agent = load_policy(args, env.observation_space.shape[0])
    if not hasattr(agent, 'encoder'):
        raise ValueError('Requires a recurrent policy.')
    agent.encoder.requires_grad_(False)
    agent.q_net.requires_grad_(False)
    out.mkdir(parents=True)
    rows, excluded, traces = [], {'terminated_or_truncated': 0, 'endpoint_mismatch': 0}, {}
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            obs, _ = env.reset(seed=seed)
            endpoint = deepcopy(env)
            endpoint_obs = obs.copy()
            endpoint_done = False
            for _ in range(args.approach_steps):
                endpoint_obs, _, terminated, truncated, _ = endpoint.step(2)
                if terminated or truncated:
                    endpoint_done = True
                    break
            initial_pose = (*endpoint.unwrapped.agent_pos, endpoint.unwrapped.agent_dir)
            endpoint.close()
            for distance in args.distances:
                for kind in ('loop', 'rotation'):
                    actions_a, actions_b = histories(distance, kind == 'rotation')
                    actions_a += [2] * args.approach_steps
                    actions_b += [2] * args.approach_steps
                    if endpoint_done:
                        excluded['terminated_or_truncated'] += 1
                        continue
                    a = replay(env, obs, actions_a, agent)
                    b = replay(env, obs, actions_b, agent)
                    if a is None or b is None:
                        excluded['terminated_or_truncated'] += 1
                        continue
                    if not matched(a, b, initial_pose, endpoint_obs):
                        excluded['endpoint_mismatch'] += 1
                        continue
                    metrics = pair_metrics(a, b, args.init_temp)
                    row = dict(pair_id=len(rows), reset_seed=seed, distance=distance, kind=kind,
                               history_steps=len(actions_a), **metrics)
                    rows.append(row)
                    # Every accepted pair is auditable; no outcome-selected examples.
                    for label, value, actions in [('a', a, actions_a), ('b', b, actions_b)]:
                        prefix = f"pair{row['pair_id']}_{label}_"
                        for key in ('observations', 'hidden', 'q', 'ig', 'candidates', 'positions'):
                            traces[prefix + key] = np.asarray(value[key])
                        traces[prefix + 'actions'] = np.asarray(actions)
            if (episode + 1) % 50 == 0:
                print(f'{episode + 1}/{args.episodes} reset layouts; {len(rows)} matched pairs', flush=True)
    finally:
        env.close()
    summary = dict(config=vars(args),
                   checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                   attempted_pairs=args.episodes * len(args.distances) * 2,
                   matched_pairs=len(rows), exclusions=excluded,
                   groups=summarize(rows, args.bootstrap_seed),
                   interpretation='Controlled forced histories, not on-policy trajectories. Same endpoint observation, physical state and elapsed time. IG is one-step information gain, not reward-optimal action value. History dependence does not isolate a causal belief representation. Rotation controls match duration but differ in motion and coverage from loops; no direct causal difference-of-groups claim.',
                   weighting='Equal reset-seed means within each group; percentile bootstrap clusters distances by reset seed. Fixed policy, exploratory intervals.')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    np.savez_compressed(out / 'matched_traces.npz', **traces)
    if rows:
        with (out / 'pairs.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plot(summary, out)
    print(json.dumps(summary['groups'], indent=2))


def plot(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, key in zip(axes[:2], ('policy_tv', 'action_changed')):
        for i, group in enumerate(('different_belief', 'same_belief_control')):
            result = summary['groups'][group][key]
            if result['mean'] is not None:
                ax.bar(i, result['mean'], color=f'C{i}')
                if result['ci95']:
                    ax.plot([i, i], result['ci95'], color='black')
        ax.set(xticks=[0,1], xticklabels=['Different belief', 'Same-belief rotations'], title=key, ylim=(0,1))
    result = summary['groups']['different_unique_ig']['own_ig_support_shift']
    axes[2].axhline(0, color='gray', linestyle='--')
    if result['mean'] is not None:
        axes[2].plot(0, result['mean'], 'o')
        if result['ci95']:
            axes[2].plot([0,0], result['ci95'])
    axes[2].set(xticks=[], title='IG-directed probability shift')
    fig.suptitle('Matched forced histories; 95% reset-cluster intervals')
    fig.savefig(out / 'matched_histories.png', dpi=180)
    fig.savefig(out / 'matched_histories.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_policy_args(parser)
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--view-size', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=400)
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--seed', type=int, default=300000)
    parser.add_argument('--bootstrap-seed', type=int, default=0)
    parser.add_argument('--distances', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--approach-steps', type=int, default=1)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.approach_steps < 0 or any(d < 1 for d in args.distances) or len(set(args.distances)) != len(args.distances):
        parser.error('Episodes and distances must be positive; distances must be distinct.')
    run(args)


if __name__ == '__main__':
    main()
