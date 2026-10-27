"""Find exact matched endpoints on-policy, before any hidden-state intervention.

Candidate selection never uses policy actions or reference agreement. Development
uses split 0/1; split 2 remains reserved for a later preregistered intervention.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from analysis.belief_reward_planner import BeliefRewardPlanner, maximizers
from analysis.evaluate_belief_decoding import collect
from analysis.policy_io import add_policy_args, load_policy


def select_pairs(data, allowed, max_pairs, seed, same_belief=False):
    """Exact obs/pose/time matching; episode-disjoint pairs within each group."""
    groups = defaultdict(list)
    starts = {}
    for i, episode in enumerate(data['episode']):
        starts.setdefault(int(episode), i)
        if allowed[i] and data['belief'][i].max() < 1:
            key = (data['observation'][i].tobytes(), data['pose'][i].tobytes(), int(data['timestep'][i]))
            groups[key].append(i)
    rng = np.random.default_rng(seed)
    groups = [v for v in groups.values() if len(v) > 1]
    rng.shuffle(groups)
    selected, used = [], set()
    for group in groups:
        rng.shuffle(group)
        accepted = False
        for offset, a in enumerate(group):
            ea = int(data['episode'][a])
            if ea in used:
                continue
            for b in group[offset + 1:]:
                eb = int(data['episode'][b])
                if ea == eb or eb in used:
                    continue
                if np.array_equal(data['belief'][a], data['belief'][b]) != same_belief:
                    continue
                # Identical observation sequences are not distinct memory evidence.
                if np.array_equal(data['observation'][starts[ea]:a+1],
                                  data['observation'][starts[eb]:b+1]):
                    continue
                selected.append((a, b))
                used.update((ea, eb))
                accepted = True
                break
            if accepted:
                break
        if len(selected) >= max_pairs:
            break
    return selected


def stable_winner(qs, margin):
    winners = [maximizers(q) for q in qs]
    if any(len(w) != 1 for w in winners) or len({w[0] for w in winners}) != 1:
        return None
    if any(np.sort(q)[-1] - np.sort(q)[-2] < margin for q in qs):
        return None
    return winners[0][0]


def metric(rows, key):
    values = np.array([r[key] for r in rows if r[key] is not None], dtype=float)
    if not len(values):
        return dict(mean=None, ci95=None, pairs=0)
    draws = np.random.default_rng(0).integers(len(values), size=(2000, len(values)))
    return dict(mean=float(values.mean()), pairs=len(values),
                ci95=np.quantile(values[draws].mean(1), [.025,.975]).tolist() if len(values)>1 else None)


def run(args):
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(f'Choose a new output directory: {out}')
    torch.set_num_threads(1)
    with np.load(args.exclude_rollouts, allow_pickle=False) as source:
        old_layouts = {tuple(v) for v in source['layout']}
    data = collect(args)
    novel = np.array([tuple(v) not in old_layouts for v in data['layout']])
    allowed = novel & (data['split'] != 2)
    groups = {name: select_pairs(data, allowed, args.max_pairs, args.match_seed, same)
              for name, same in [('different_belief', False), ('same_belief', True)]}
    # Hash the full proposed pair list before computing reference or IQ scores.
    selection = {name: [[int(a), int(b)] for a,b in pairs] for name,pairs in groups.items()}
    out.mkdir(parents=True)
    np.savez_compressed(out / 'rollouts.npz', **data)
    (out / 'selected_pairs.json').write_text(json.dumps(selection, indent=2))
    agent = load_policy(args, data['observation'].shape[1])
    planner = BeliefRewardPlanner(args.size, args.view_size, args.max_steps, args.gamma)
    rows, references = [], {}
    for kind, pairs in groups.items():
        for index, (a,b) in enumerate(pairs):
            qa, qb = [], []
            pose = (int(round(data['pose'][a,0]*(args.size-1))),
                    int(round(data['pose'][a,1]*(args.size-1))),
                    int(np.argmax(data['pose'][a,2:])))
            for state, dest in ((a,qa),(b,qb)):
                for horizon in args.horizons:
                    dest.append(planner.q_values(pose, data['cells'][data['belief'][state]>0],
                                                 int(data['timestep'][state]), horizon))
            wa, wb = stable_winner(qa,args.min_margin), stable_winner(qb,args.min_margin)
            eligible = kind == 'different_belief' and wa is not None and wb is not None and wa != wb
            with torch.no_grad():
                q = agent.q_net(torch.tensor(data['hidden'][[a,b]],dtype=torch.float32))
                probs = (q / args.init_temp).softmax(-1).numpy()
                actions = q.argmax(-1).numpy()
            row = dict(group=kind, pair_id=index, row_a=a, row_b=b,
                       episode_a=int(data['episode'][a]), episode_b=int(data['episode'][b]),
                       reset_seed_a=args.rollout_seed+int(data['episode'][a]),
                       reset_seed_b=args.rollout_seed+int(data['episode'][b]),
                       timestep=int(data['timestep'][a]),
                       stable_action_a=wa, stable_action_b=wb, intervention_candidate=eligible,
                       iq_action_a=int(actions[0]), iq_action_b=int(actions[1]),
                       action_changed=bool(actions[0]!=actions[1]),
                       policy_tv=float(.5*np.abs(probs[0]-probs[1]).sum()),
                       both_match_reference=bool(actions[0]==wa and actions[1]==wb) if eligible else None,
                       own_reward_support_shift=float(.5*(probs[0,wa]-probs[1,wa]+probs[1,wb]-probs[0,wb])) if eligible else None)
            for horizon, va, vb in zip(args.horizons,qa,qb):
                row[f'winners_a_h{horizon}'] = json.dumps(maximizers(va))
                row[f'winners_b_h{horizon}'] = json.dumps(maximizers(vb))
            rows.append(row)
            references[f'{kind}_{index}_a'] = np.asarray(qa)
            references[f'{kind}_{index}_b'] = np.asarray(qb)
            if (index+1)%20 == 0:
                print(f'{kind}: planned {index+1}/{len(pairs)} pairs',flush=True)
    np.savez_compressed(out/'reference_values.npz',**references)
    if rows:
        with (out/'pairs.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = dict(config=vars(args),checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                   selection_sha256=hashlib.sha256(json.dumps(selection,sort_keys=True).encode()).hexdigest(),
                   counts=dict(collected_episodes=args.episodes,
                               novel_development_episodes=len(np.unique(data['episode'][allowed])),
                               reserved_novel_episodes=len(np.unique(data['episode'][novel & (data['split']==2)])),
                               excluded_previously_seen_episodes=len(np.unique(data['episode'][~novel]))),
                   groups={kind:{key:metric([r for r in rows if r['group']==kind],key)
                                 for key in ('intervention_candidate','action_changed','policy_tv',
                                             'both_match_reference','own_reward_support_shift')}
                           for kind in groups},
                   interpretation='Natural on-policy histories; exact current observation, pose and timestep matched. Distinct episodes, disjoint within each group. Layouts previously used for belief probes excluded. Split 2 not queried by planner or matching. This is development data for intervention design, not causal evidence. Stability at tested finite horizons is not proof of optimality. Pair bootstrap conditions on selected pairs and fixed policy; groups can reuse episodes, so do not interpret separate intervals as a causal between-group contrast.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    eligible=[r for r in rows if r['intervention_candidate']]
    (out/'intervention_candidates.json').write_text(json.dumps(eligible,indent=2))
    plot(summary,out)
    print(json.dumps(dict(counts=summary['counts'],groups=summary['groups']),indent=2))


def plot(summary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,key,title in zip(axes[:2],('action_changed','policy_tv'),('Actual action changes','Action-distribution TV')):
        for i,group in enumerate(('different_belief','same_belief')):
            result=summary['groups'][group][key]
            if result['mean'] is not None:
                ax.bar(i,result['mean'],color=f'C{i}')
                if result['ci95']: ax.plot([i,i],result['ci95'],color='black')
        ax.set(xticks=[0,1],xticklabels=['Different belief','Same belief'],ylim=(0,1),title=title)
    result=summary['groups']['different_belief']['own_reward_support_shift']
    axes[2].axhline(0,color='gray',linestyle='--')
    if result['mean'] is not None:
        axes[2].plot(0,result['mean'],'o')
        if result['ci95']: axes[2].plot([0,0],result['ci95'])
    else: axes[2].text(.5,.5,'No stable different-winner pairs',transform=axes[2].transAxes,ha='center')
    axes[2].set(xticks=[],title=f"Stable reward support shift (n={result['pairs']})")
    fig.suptitle('Natural matched histories: development cohort, no memory intervention')
    fig.savefig(out/'natural_histories.png',dpi=180)
    fig.savefig(out/'natural_histories.pdf')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    add_policy_args(parser)
    parser.add_argument('--exclude-rollouts',required=True)
    parser.add_argument('--size',type=int,default=10)
    parser.add_argument('--view-size',type=int,default=3)
    parser.add_argument('--max-steps',type=int,default=400)
    parser.add_argument('--episodes',type=int,default=600)
    parser.add_argument('--rollout-seed',type=int,default=400000)
    parser.add_argument('--split-seed',type=int,default=0)
    parser.add_argument('--match-seed',type=int,default=0)
    parser.add_argument('--horizons',type=int,nargs='+',default=[5,7])
    parser.add_argument('--min-margin',type=float,default=.0001)
    parser.add_argument('--max-pairs',type=int,default=100)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    if (args.episodes<10 or args.max_pairs<1 or len(set(args.horizons))<2
            or len(set(args.horizons))!=len(args.horizons) or min(args.horizons)<1
            or not np.isfinite(args.min_margin) or args.min_margin<0):
        parser.error('Need >=10 episodes, positive max-pairs, >=2 distinct positive horizons, nonnegative margin.')
    run(args)


if __name__=='__main__':
    main()
