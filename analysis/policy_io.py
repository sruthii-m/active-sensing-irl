"""Analysis-only adapters for legacy IQ checkpoints (weights only)."""
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
        enc = weights['encoder']
        agent = RecurrentOfflineSoftQAgent(
            obs_dim, weights['q_net']['fc3.weight'].shape[0], config,
            latent_dim=enc['gru.weight_ih'].shape[1],
            rnn_hidden_dim=enc['gru.weight_hh'].shape[1],
            grid_shape=tuple(args.grid_shape) if args.grid_shape else None)
    else:
        agent = OfflineSoftQAgent(obs_dim, weights['fc3.weight'].shape[0], config)
    agent.load(args.checkpoint)
    agent.q_net.eval()
    if hasattr(agent, 'encoder'):
        agent.encoder.eval()
    return agent


@torch.no_grad()
def representation(agent, obs, hidden=None):
    tensor = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
    return agent.encoder(tensor, hidden)[0] if hasattr(agent, 'encoder') else tensor
