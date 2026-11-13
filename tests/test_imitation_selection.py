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

from analysis.expert_likelihood import evaluate as evaluate_likelihood
from imitation import selection
from imitation import train_iq_full_state as full_trainer
from imitation import train_iq_recurrent as trainer


def episode(value, length=2):
    states = torch.full((length, 3), float(value))
    return (states, states + 1, torch.zeros(length, 1, dtype=torch.long),
            torch.zeros(length, 1), torch.arange(length).float())


def write_demos(path, episodes, **metadata):
    data = {key: [ep[index].numpy().reshape(-1) if index >= 2 else ep[index].numpy()
                  for ep in episodes]
            for key, index in [('states', 0), ('next_states', 1), ('actions', 2),
                               ('dones', 3), ('rewards', 4)]}
    data.update(metadata)
    path.write_bytes(pickle.dumps(data))
    return data


def provenance(seed, count=2, **overrides):
    result = dict(seed_start=seed, attempts=count, selection='all',
                  expert_model='expert.zip', deterministic=True,
                  env_kwargs=dict(size=10, num_distractors=0, agent_view_size=3))
    result.update(overrides)
    return result


class ExplicitValidationProvenanceTests(unittest.TestCase):
    def test_disjoint_generation_retains_repeated_trajectories_and_audits_counts(self):
        with tempfile.TemporaryDirectory() as root:
            train_path, valid_path = Path(root) / 'train.pkl', Path(root) / 'valid.pkl'
            write_demos(train_path, [episode(0), episode(1), episode(1)], **provenance(100000, 3))
            write_demos(valid_path, [episode(1), episode(1), episode(2)], **provenance(102500, 3))
            audit = {}
            with self.assertWarnsRegex(UserWarning, '2/3 validation episodes'):
                train, valid, ti, vi = selection.load_training_validation(
                    str(train_path), str(valid_path), .2, 0, 'cpu', split_audit=audit)
            self.assertEqual((len(train), len(valid)), (3, 3))
            self.assertEqual((ti, vi), ([0, 1, 2], [0, 1, 2]))
            self.assertEqual(audit['shared_trajectory_groups'], 1)
            self.assertEqual(audit['train_episodes_matching_validation'], 2)
            self.assertEqual(audit['validation_episodes_matching_training'], 2)
            self.assertEqual(audit['provenance']['train_seed_range'], [100000, 100002])
            self.assertEqual(audit['provenance']['validation_seed_range'], [102500, 102502])
            self.assertEqual(audit['provenance']['status'], 'compatible_disjoint_seed_ranges')
            self.assertTrue(audit['repeated_trajectories_allowed'])
            self.assertEqual(audit['layout_disjointness'], 'not_established')
            json.dumps(audit, allow_nan=False)

    def test_same_file_and_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            train_path = Path(root) / 'train.pkl'
            write_demos(train_path, [episode(0), episode(1)], **provenance(100))
            alias = Path(root) / 'alias.pkl'
            alias.symlink_to(train_path)
            for valid_path in (train_path, alias):
                with self.subTest(path=valid_path), self.assertRaisesRegex(ValueError, 'same file'):
                    selection.load_training_validation(str(train_path), str(valid_path), .2, 0, 'cpu')

    def test_overlapping_reset_ranges_fail_even_without_matching_trajectories(self):
        with tempfile.TemporaryDirectory() as root:
            train_path, valid_path = Path(root) / 'train.pkl', Path(root) / 'valid.pkl'
            write_demos(train_path, [episode(0), episode(1)], **provenance(100))
            for seed in (99, 100, 101):
                write_demos(valid_path, [episode(2), episode(3)], **provenance(seed))
                with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, 'seed ranges overlap'):
                    selection.load_training_validation(str(train_path), str(valid_path), .2, 0, 'cpu')

    def test_adjacent_reset_ranges_are_disjoint(self):
        with tempfile.TemporaryDirectory() as root:
            train_path, valid_path = Path(root) / 'train.pkl', Path(root) / 'valid.pkl'
            write_demos(train_path, [episode(0), episode(1)], **provenance(100))
            write_demos(valid_path, [episode(2), episode(3)], **provenance(102))
            audit = {}
            selection.load_training_validation(str(train_path), str(valid_path), .2, 0, 'cpu', split_audit=audit)
            self.assertTrue(audit['provenance']['seed_ranges_disjoint'])
            self.assertEqual(audit['shared_trajectory_groups'], 0)

    def test_copied_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            train_path, valid_path = Path(root) / 'train.pkl', Path(root) / 'valid.pkl'
            write_demos(train_path, [episode(0), episode(1)], **provenance(100))
            valid_path.write_bytes(train_path.read_bytes())
            with self.assertRaisesRegex(ValueError, 'seed ranges overlap'):
                selection.load_training_validation(str(train_path), str(valid_path), .2, 0, 'cpu')

    def test_filtered_attempt_ranges_include_discarded_episodes(self):
        with tempfile.TemporaryDirectory() as root:
            train_path, valid_path = Path(root) / 'train.pkl', Path(root) / 'valid.pkl'
            write_demos(train_path, [episode(0), episode(1)],
                        **provenance(100, 10, selection='successful_filtered'))
            write_demos(valid_path, [episode(2), episode(3)],
                        **provenance(105, selection='successful_filtered'))
            with self.assertRaisesRegex(ValueError, 'seed ranges overlap'):
                selection.load_training_validation(str(train_path), str(valid_path), .2, 0, 'cpu')

    def test_incomplete_provenance_cannot_exempt_repeated_trajectories(self):
        with tempfile.TemporaryDirectory() as root:
            train_path, valid_path = Path(root) / 'train.pkl', Path(root) / 'valid.pkl'
            write_demos(train_path, [episode(0), episode(1)], **provenance(100))
            for key in provenance(200):
                metadata = provenance(200)
                del metadata[key]
                write_demos(valid_path, [episode(1), episode(2)], **metadata)
                with self.subTest(missing=key), self.assertRaisesRegex(ValueError, 'identical trajectories'):
                    selection.load_training_validation(str(train_path), str(valid_path), .2, 0, 'cpu')

    def test_incompatible_generation_conditions_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            train_path, valid_path = Path(root) / 'train.pkl', Path(root) / 'valid.pkl'
            write_demos(train_path, [episode(0), episode(1)], **provenance(100))
            for key, value in [('env_kwargs', dict(size=8)), ('expert_model', 'other.zip'),
                               ('deterministic', False), ('selection', 'successful_filtered')]:
                write_demos(valid_path, [episode(2), episode(3)], **provenance(200, **{key: value}))
                with self.subTest(field=key), self.assertRaisesRegex(ValueError, key + ' differs'):
                    selection.load_training_validation(str(train_path), str(valid_path), .2, 0, 'cpu')

    def test_invalid_seed_and_attempt_metadata_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            train_path, valid_path = Path(root) / 'train.pkl', Path(root) / 'valid.pkl'
            write_demos(train_path, [episode(0), episode(1)], **provenance(100))
            for key, value in [('seed_start', -1), ('seed_start', True), ('seed_start', 200.),
                               ('attempts', 0), ('attempts', 1), ('attempts', 3), ('attempts', False)]:
                write_demos(valid_path, [episode(1), episode(2)], **provenance(200, **{key: value}))
                with self.subTest(field=key, value=value), self.assertRaisesRegex(ValueError, 'Invalid demonstration provenance'):
                    selection.load_training_validation(str(train_path), str(valid_path), .2, 0, 'cpu')

    def test_legacy_distinct_files_still_work_but_provenance_is_unverified(self):
        with tempfile.TemporaryDirectory() as root:
            train_path, valid_path = Path(root) / 'train.pkl', Path(root) / 'valid.pkl'
            write_demos(train_path, [episode(0), episode(1)])
            write_demos(valid_path, [episode(2), episode(3)])
            audit = {}
            selection.load_training_validation(str(train_path), str(valid_path), .2, 0, 'cpu', split_audit=audit)
            self.assertEqual(audit['provenance']['status'], 'unverified')
            self.assertFalse(audit['repeated_trajectories_allowed'])

    def test_both_trainers_record_explicit_split_audit_without_changing_selection(self):
        for module in (trainer, full_trainer):
            with self.subTest(trainer=module.__name__), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                train_path, valid_path = root / 'train.pkl', root / 'valid.pkl'
                write_demos(train_path, [episode(0), episode(1)], **provenance(100))
                write_demos(valid_path, [episode(1), episode(2)], **provenance(200))
                checkpoint = root / 'iq.pt'
                argv = ['train', '--demo-path', str(train_path), '--validation-demo-path', str(valid_path),
                        '--save-path', str(checkpoint), '--max-steps', '1', '--eval-freq', '1', '--init-temp', '.5']
                if module is trainer:
                    argv += ['--env', 'partial_obs', '--device', 'cpu', '--latent-dim', '4',
                             '--rnn-hidden-dim', '4', '--checkpoint-dir', str(root / 'checkpoints')]
                with patch('sys.argv', argv), patch.object(module, 'evaluate_policy_success_rate', return_value=0.), contextlib.redirect_stdout(io.StringIO()), self.assertWarns(UserWarning):
                    module.main()
                metadata = json.loads(Path(str(checkpoint) + '.selection.json').read_text())
                self.assertEqual(metadata['split_audit']['shared_trajectory_groups'], 1)
                self.assertEqual(metadata['split_audit']['validation_episodes_matching_training'], 1)
                self.assertEqual(metadata['metric'], 'episode_mean_nll')
                self.assertEqual(metadata['best_validation_nll'], min(r['episode_mean_nll'] for r in metadata['evaluations']))
                if module is trainer:
                    manifest = json.loads((root / 'checkpoints' / 'manifest.json').read_text())
                    self.assertEqual(manifest['split_audit'], metadata['split_audit'])


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

            def encode_episode(self, states, next_states, actions):
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
            self.assertEqual(result['split_audit']['strategy'], 'grouped_trajectory_holdout')
            self.assertEqual(result['split_audit']['shared_trajectory_groups'], 0)
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


class FullStateSelectionTests(unittest.TestCase):
    def test_feedforward_likelihood_matches_final_evaluator_and_restores_mode(self):
        class Agent:
            q_net = torch.nn.Identity()
            alpha = .5

        agent = Agent()
        short, long = episode(0, 1), episode(0, 3)
        short[0][:, 0] = 1
        long[0][:, 1] = 1
        with tempfile.TemporaryDirectory() as root:
            data = write_demos(Path(root) / 'demos.pkl', [short, long])
        for training in (True, False):
            agent.q_net.train(training)
            result = full_trainer.evaluate_imitation(agent, [short, long])
            self.assertEqual(agent.q_net.training, training)
        reported = evaluate_likelihood(agent, data)
        self.assertAlmostEqual(result['episode_mean_nll'], -reported['episode_mean_log_likelihood'])
        self.assertAlmostEqual(result['step_mean_nll'], reported['negative_log_likelihood'])
        self.assertAlmostEqual(result['action_accuracy'], reported['expert_top1_accuracy'])
        self.assertNotEqual(result['episode_mean_nll'], result['step_mean_nll'])

    def test_explicit_validation_rejects_overlap_even_when_rewards_differ(self):
        with tempfile.TemporaryDirectory() as root:
            train_path, valid_path = Path(root) / 'train.pkl', Path(root) / 'valid.pkl'
            write_demos(train_path, [episode(0), episode(1)])
            duplicate = list(episode(1))
            duplicate[-1] = torch.full_like(duplicate[-1], 100.)
            write_demos(valid_path, [duplicate])
            with self.assertRaisesRegex(ValueError, 'identical trajectories'):
                selection.load_training_validation(str(train_path), str(valid_path), .2, 0, 'cpu')

    def test_full_state_training_restores_best_and_is_independent_of_rewards_and_success(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            train_path, valid_path = root / 'train.pkl', root / 'valid.pkl'
            train_episodes = [episode(i) for i in range(4)]
            valid_episodes = [episode(10), episode(11, 3)]
            manifests, checkpoints = [], []
            original = full_trainer.OfflineSoftQAgent.update
            for run in (0, 1):
                # Keep observations/actions fixed while changing all reward labels.
                if run:
                    for ep in train_episodes + valid_episodes:
                        ep[-1].fill_(100.)
                write_demos(train_path, train_episodes)
                write_demos(valid_path, valid_episodes)
                checkpoint = root / f'iq_{run}.pt'
                seen = []

                def update(agent, obs, next_obs, actions, dones, **kwargs):
                    seen.extend(obs[:, 0].tolist())
                    result = original(agent, obs, next_obs, actions, dones, **kwargs)
                    # First update has the best likelihood; later updates deteriorate.
                    with torch.no_grad():
                        agent.q_net.fc3.weight.zero_()
                        agent.q_net.fc3.bias.zero_()
                        agent.q_net.fc3.bias[0] = [2., 1., -1.][kwargs['step'] - 1]
                    return result

                def success(agent, **kwargs):
                    # Task success must be measured only after checkpoint restoration.
                    self.assertEqual(agent.q_net.fc3.bias[0].item(), 2.)
                    return float(run)

                argv = ['train', '--demo-path', str(train_path), '--validation-demo-path', str(valid_path),
                        '--save-path', str(checkpoint), '--steps', '3', '--eval-freq', '1',
                        '--patience', '0', '--batch-size', '8', '--init-temp', '.5']
                with patch('sys.argv', argv), patch.object(full_trainer.OfflineSoftQAgent, 'update', update), patch.object(full_trainer, 'evaluate_policy_success_rate', side_effect=success) as task_success, contextlib.redirect_stdout(io.StringIO()):
                    full_trainer.main()
                task_success.assert_called_once()
                self.assertTrue(seen)
                self.assertTrue(set(seen) <= {0., 1., 2., 3.})
                metadata = json.loads(Path(str(checkpoint) + '.selection.json').read_text())
                self.assertEqual([r['step'] for r in metadata['evaluations']], [0, 1, 2, 3])
                self.assertEqual(metadata['train_indices'], [0, 1, 2, 3])
                self.assertEqual(metadata['validation_indices'], [0, 1])
                self.assertEqual(metadata['best_step'], 1)
                self.assertEqual(metadata['final_training_step'], 3)
                self.assertEqual(metadata['final_success_rate'], float(run))
                self.assertEqual(metadata['best_validation_nll'], min(r['episode_mean_nll'] for r in metadata['evaluations']))
                agent = task_success.call_args.args[0]
                measured = full_trainer.evaluate_imitation(agent, valid_episodes)['episode_mean_nll']
                self.assertAlmostEqual(measured, metadata['best_validation_nll'])
                saved = torch.load(checkpoint, weights_only=True)['q_net']
                for key, value in agent.q_net.state_dict().items():
                    torch.testing.assert_close(saved[key], value)
                    torch.testing.assert_close(agent.target_net.state_dict()[key], value)
                checkpoints.append(saved)
                manifests.append(metadata)
            self.assertEqual(manifests[0]['evaluations'], manifests[1]['evaluations'])
            self.assertEqual(manifests[0]['best_step'], manifests[1]['best_step'])
            for key in checkpoints[0]:
                torch.testing.assert_close(checkpoints[0][key], checkpoints[1][key])

    def test_grouped_holdout_and_final_step_are_used_by_full_state_entrypoint(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            episodes = [episode(i) for i in range(5)] + [episode(0)]
            demo, checkpoint = root / 'demos.pkl', root / 'iq.pt'
            write_demos(demo, episodes)
            seen = []
            original = full_trainer.OfflineSoftQAgent.update

            def update(agent, obs, *args, **kwargs):
                seen.extend(obs[:, 0].tolist())
                return original(agent, obs, *args, **kwargs)

            argv = ['train', '--demo-path', str(demo), '--save-path', str(checkpoint),
                    '--max-steps', '2', '--eval-freq', '10', '--init-temp', '.5']
            with patch('sys.argv', argv), patch.object(full_trainer.OfflineSoftQAgent, 'update', update), patch.object(full_trainer, 'evaluate_policy_success_rate', return_value=0.), contextlib.redirect_stdout(io.StringIO()):
                full_trainer.main()
            result = json.loads(Path(str(checkpoint) + '.selection.json').read_text())
            self.assertEqual([r['step'] for r in result['evaluations']], [0, 2])
            train, valid = result['train_indices'], result['validation_indices']
            self.assertEqual(0 in train, 5 in train)
            self.assertEqual(set(train) | set(valid), set(range(6)))
            held_values = {episodes[i][0][0, 0].item() for i in valid}
            self.assertFalse(set(seen) & held_values)


if __name__ == '__main__':
    unittest.main()
