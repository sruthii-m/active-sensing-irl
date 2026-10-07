"""Expert action likelihood on separately held-out episodes."""
import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from analysis.policy_io import add_policy_args, load_policy


def load_expert(path, kind):
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    model = (RecurrentPPO if kind == 'recurrent' else PPO).load(path, device='cpu')
    model.policy.set_training_mode(False)
    return model


@torch.no_grad()
def expert_log_probs(expert, states):
    """Replay one observation sequence, resetting the expert LSTM each episode."""
    policy = expert.policy
    if tuple(states.shape[1:]) != tuple(expert.observation_space.shape):
        raise ValueError('Expert observation shape differs from recorded observations.')
    hidden = None
    if hasattr(policy, 'lstm_actor'):
        lstm = policy.lstm_actor
        hidden = tuple(torch.zeros(lstm.num_layers, 1, lstm.hidden_size) for _ in range(2))
    result = []
    for t, obs in enumerate(states):
        if hidden is None:
            dist = policy.get_distribution(obs[None])
        else:
            dist, hidden = policy.get_distribution(obs[None], hidden, torch.tensor([float(t == 0)]))
        # Categorical.logits are normalized log probabilities, stable even for
        # extremely small masses. Do not log rounded/underflowed probabilities.
        result.append(dist.distribution.logits[0])
    return torch.stack(result)


@torch.no_grad()
def evaluate(agent, data, expert=None, step_rows=None):
    episodes = []
    for states, next_states, actions in zip(data['states'], data['next_states'], data['actions'], strict=True):
        states = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        next_states = torch.as_tensor(np.asarray(next_states), dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.long).reshape(-1)
        if len(states) == 0 or len(actions) != len(states):
            raise ValueError('Each episode must contain matching nonempty states/actions.')
        h = agent.encode_episode(states, next_states)[:-1] if hasattr(agent, 'encoder') else states
        q = agent.q_net(h)
        logp = torch.log_softmax(q / agent.alpha, dim=-1)
        ll = logp.gather(1, actions[:, None]).squeeze(1)
        episode = dict(steps=len(actions), log_likelihood_sum=ll.sum().item(),
                             mean_log_likelihood=ll.mean().item(),
                             correct=int((q.argmax(-1) == actions).sum()),
                             entropy_sum=float(-(logp.exp() * logp).sum()))
        episode['success'] = (bool(np.sum(data['rewards'][len(episodes)]) > 0)
                              if 'rewards' in data else None)
        if expert is not None:
            expert_lp = expert_log_probs(expert, states)
            if expert_lp.shape != logp.shape:
                raise ValueError('Expert and IQ action spaces differ.')
            expert_ll = expert_lp.gather(1, actions[:, None]).squeeze(1)
            gap = expert_ll - ll
            kl = (expert_lp.exp() * (expert_lp - logp)).sum(-1).clamp_min(0)
            agree = expert_lp.argmax(-1) == logp.argmax(-1)
            episode.update(expert_log_likelihood_sum=expert_ll.sum().item(),
                           expert_mean_log_likelihood=expert_ll.mean().item(),
                           mean_likelihood_gap=gap.mean().item(),
                           mean_kl_expert_to_iq=kl.mean().item(),
                           greedy_policy_agreement=agree.float().mean().item())
            if step_rows is not None:
                cumulative = gap.cumsum(0)
                for t in range(len(actions)):
                    row = dict(episode=len(episodes), timestep=t, success=episode['success'],
                               action=int(actions[t]), iq_log_likelihood=float(ll[t]),
                               expert_log_likelihood=float(expert_ll[t]), likelihood_gap=float(gap[t]),
                               cumulative_likelihood_gap=float(cumulative[t]),
                               kl_expert_to_iq=float(kl[t]), greedy_agreement=bool(agree[t]),
                               expert_argmax=int(expert_lp[t].argmax()), iq_argmax=int(logp[t].argmax()))
                    for a in range(logp.shape[1]):
                        row[f'expert_prob_{a}'] = float(expert_lp[t, a].exp())
                        row[f'iq_prob_{a}'] = float(logp[t, a].exp())
                    step_rows.append(row)
        episodes.append(episode)
    steps = sum(e['steps'] for e in episodes)
    if not steps:
        raise ValueError('No held-out steps.')
    mean = sum(e['log_likelihood_sum'] for e in episodes) / steps
    result = dict(episodes=episodes, steps=steps, mean_per_step_log_likelihood=mean,
                negative_log_likelihood=-mean,
                episode_mean_log_likelihood=float(np.mean([e['mean_log_likelihood'] for e in episodes])),
                expert_top1_accuracy=sum(e['correct'] for e in episodes) / steps,
                mean_policy_entropy_nats=sum(e['entropy_sum'] for e in episodes) / steps)
    if expert is not None:
        def summarize(items):
            n = sum(e['steps'] for e in items)
            keys = ('expert_mean_log_likelihood', 'mean_likelihood_gap',
                    'mean_kl_expert_to_iq', 'greedy_policy_agreement')
            return dict(episodes=len(items), steps=n,
                        iq_mean_log_likelihood=sum(e['log_likelihood_sum'] for e in items) / n if n else None,
                        **{k: sum(e[k] * e['steps'] for e in items) / n if n else None for k in keys})
        result['paired'] = summarize(episodes)
        result['paired_by_outcome'] = {label: summarize([e for e in episodes if e['success'] is value])
                                      for label, value in [('success', True), ('failure', False), ('unknown', None)]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_policy_args(parser)
    parser.add_argument('--heldout-demos', required=True, help='Trusted pickle, excluded from training by episode/seed.')
    parser.add_argument('--output', default='artifacts/analysis/likelihood.json')
    parser.add_argument('--expert-model', help='Saved PPO/RecurrentPPO expert zip for paired comparison.')
    parser.add_argument('--expert-kind', choices=['recurrent', 'ppo'], default='recurrent')
    args = parser.parse_args()
    with open(args.heldout_demos, 'rb') as f:
        data = pickle.load(f)
    agent = load_policy(args, np.asarray(data['states'][0]).shape[1])
    expert = load_expert(args.expert_model, args.expert_kind) if args.expert_model else None
    rows = []
    result = dict(config=vars(args), dataset_selection=data.get('selection', 'unknown'),
                  **evaluate(agent, data, expert, rows))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with path.with_suffix('.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k != 'episodes'}, indent=2))


if __name__ == '__main__':
    main()
