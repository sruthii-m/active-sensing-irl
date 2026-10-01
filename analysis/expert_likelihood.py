"""Expert action likelihood on separately held-out episodes."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from analysis.policy_io import add_policy_args, load_policy


@torch.no_grad()
def evaluate(agent, data):
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
        episodes.append(dict(steps=len(actions), log_likelihood_sum=ll.sum().item(),
                             mean_log_likelihood=ll.mean().item(),
                             correct=int((q.argmax(-1) == actions).sum()),
                             entropy_sum=float(-(logp.exp() * logp).sum())))
    steps = sum(e['steps'] for e in episodes)
    if not steps:
        raise ValueError('No held-out steps.')
    mean = sum(e['log_likelihood_sum'] for e in episodes) / steps
    return dict(episodes=episodes, steps=steps, mean_per_step_log_likelihood=mean,
                negative_log_likelihood=-mean,
                episode_mean_log_likelihood=float(np.mean([e['mean_log_likelihood'] for e in episodes])),
                expert_top1_accuracy=sum(e['correct'] for e in episodes) / steps,
                mean_policy_entropy_nats=sum(e['entropy_sum'] for e in episodes) / steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_policy_args(parser)
    parser.add_argument('--heldout-demos', required=True, help='Trusted pickle, excluded from training by episode/seed.')
    parser.add_argument('--output', default='artifacts/analysis/likelihood.json')
    args = parser.parse_args()
    with open(args.heldout_demos, 'rb') as f:
        data = pickle.load(f)
    agent = load_policy(args, np.asarray(data['states'][0]).shape[1])
    result = dict(config=vars(args), **evaluate(agent, data))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k != 'episodes'}, indent=2))


if __name__ == '__main__':
    main()
