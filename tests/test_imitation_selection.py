"""Held-out likelihood selection must not depend on goal rewards."""
import contextlib
import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from imitation import train_iq_recurrent as trainer


def episode(value, length=2):
    states = torch.full((length, 3), float(value))
    return (states, states + 1, torch.zeros(length, 1, dtype=torch.long),
            torch.zeros(length, 1), torch.arange(length).float())


class SelectionTests(unittest.TestCase):
    def test_split_groups_duplicates_and_is_reproducible(self):
        episodes = [episode(i) for i in range(10)] + [episode(0)]
        train, valid = trainer.validation_split(episodes, .2, 4)
        self.assertEqual((train, valid), trainer.validation_split(episodes, .2, 4))
        self.assertEqual(set(train) | set(valid), set(range(11)))
        self.assertFalse(set(train) & set(valid))
        self.assertEqual(0 in train, 10 in train)
        with self.assertRaises(ValueError):
            trainer.validation_split([episode(0), episode(0)], .2, 0)

    def test_best_checkpoint_and_meaningful_progress_are_separate(self):
        stop = trainer.LikelihoodStopping(.1, 2)
        self.assertEqual(stop.observe(1.), (True, False))
        self.assertEqual(stop.observe(.95), (True, False))
        self.assertEqual(stop.observe(.94), (True, True))
        self.assertEqual(stop.best, .94)
        self.assertEqual(stop.observe(.8), (True, False))
        with self.assertRaises(ValueError):
            stop.observe(float('nan'))

    def test_likelihood_weights_episodes_and_restores_modes(self):
        class Agent:
            encoder = torch.nn.Identity()
            q_net = torch.nn.Identity()
            alpha = .5

            def encode_episode(self, states, next_states):
                self_test.assertFalse(torch.is_grad_enabled())
                self_test.assertFalse(self.encoder.training)
                return torch.cat([states, next_states[-1:]])

        self_test = self
        agent = Agent()
        agent.q_net.eval()
        short = list(episode(0, 1))
        long = list(episode(0, 3))
        short[0][:, 0] = 1
        long[0][:, 1] = 1
        result = trainer.evaluate_imitation(agent, [short, long])
        nll_short = -torch.log_softmax(short[0] / .5, -1)[0, 0].item()
        nll_long = -torch.log_softmax(long[0] / .5, -1)[0, 0].item()
        self.assertAlmostEqual(result['episode_mean_nll'], (nll_short + nll_long)/2)
        self.assertAlmostEqual(result['step_mean_nll'], (nll_short + 3*nll_long)/4, places=6)
        self.assertEqual(result['action_accuracy'], .25)
        self.assertTrue(agent.encoder.training)
        self.assertFalse(agent.q_net.training)

    def test_training_holds_out_data_and_evaluates_final_step(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            episodes = [episode(i) for i in range(5)]
            data = {key: [ep[index].numpy().reshape(-1) if index >= 2 else ep[index].numpy()
                          for ep in episodes]
                    for key, index in [('states', 0), ('next_states', 1), ('actions', 2),
                                       ('dones', 3), ('rewards', 4)]}
            demo = root / 'demos.pkl'
            demo.write_bytes(pickle.dumps(data))
            checkpoint = root / 'iq.pt'
            seen = []
            original = trainer.RecurrentOfflineSoftQAgent.update

            def update(agent, batch, **kwargs):
                seen.extend(trainer.episode_fingerprint(ep) for ep in batch)
                return original(agent, batch, **kwargs)

            argv = ['train', '--env', 'partial_obs', '--demo-path', str(demo),
                    '--save-path', str(checkpoint), '--checkpoint-dir', str(root / 'checkpoints'), '--max-steps', '2', '--eval-freq', '10',
                    '--latent-dim', '4', '--rnn-hidden-dim', '4', '--device', 'cpu']
            with patch('sys.argv', argv), patch.object(trainer, 'evaluate_policy_success_rate', return_value=1.) as success, patch.object(trainer.RecurrentOfflineSoftQAgent, 'update', update), contextlib.redirect_stdout(io.StringIO()):
                trainer.main()
            success.assert_called_once()
            result = json.loads(Path(str(checkpoint) + '.selection.json').read_text())
            self.assertEqual([x['step'] for x in result['evaluations']], [0, 2])
            manifest = json.loads((root / 'checkpoints' / 'manifest.json').read_text())
            self.assertEqual(manifest['evaluations'], result['evaluations'])
            for record in manifest['evaluations']:
                self.assertTrue(Path(record['checkpoint']).is_file())
            self.assertEqual(result['best_validation_nll'], min(x['episode_mean_nll'] for x in result['evaluations']))
            valid = [episodes[i] for i in result['validation_indices']]
            self.assertFalse(set(seen) & {trainer.episode_fingerprint(ep) for ep in valid})
            agent = success.call_args.args[0]
            measured = trainer.evaluate_imitation(agent, valid)['episode_mean_nll']
            self.assertAlmostEqual(measured, result['best_validation_nll'])
            saved = torch.load(checkpoint, weights_only=False)
            for key, value in agent.q_net.state_dict().items():
                torch.testing.assert_close(value, saved['q_net'][key])


if __name__ == '__main__':
    unittest.main()
