"""Consistent replay adapters for versioned and legacy IQ checkpoints."""
import torch
from imitation.iq_learn_agent import OfflineSoftQAgent, make_args
from imitation.recurrent_iq_learn_agent import RecurrentOfflineSoftQAgent


def add_policy_args(parser):
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--init-temp', type=float, required=True,
                        help='Training temperature; legacy checkpoints do not store it.')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--grid-shape', type=int, nargs=3)


def load_policy(args, obs_dim):
    if args.init_temp <= 0:
        raise ValueError('Temperature must be positive.')
    weights = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = make_args(gamma=args.gamma, init_temp=args.init_temp)
    if 'encoder' in weights:
        agent = RecurrentOfflineSoftQAgent.from_checkpoint(
            args.checkpoint, obs_dim=obs_dim, args=config,
            grid_shape=tuple(args.grid_shape) if args.grid_shape else None)
    else:
        stored = weights.get('config', {})
        q_weights = weights['q_net'] if stored else weights
        agent = OfflineSoftQAgent(obs_dim, q_weights['fc3.weight'].shape[0], config,
                                  terminal_semantics=stored.get('terminal_semantics', 'recorded'))
        agent.load(args.checkpoint)
    agent.q_net.eval()
    if hasattr(agent, 'encoder'):
        agent.encoder.eval()
    return agent


@torch.no_grad()
def representation(agent, obs, hidden=None, previous_action=None):
    tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).reshape(1, -1)
    return agent.encoder(tensor, hidden, previous_action)[0] if hasattr(agent, 'encoder') else tensor
