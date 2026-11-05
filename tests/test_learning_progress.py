"""Scientific and end-to-end checks for reward-planner learning diagnostics."""
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from analysis import evaluate_belief_alignment, evaluate_learning_progress
from analysis.alignment_metrics import reward_action_alignment, summarize_reward_alignment
from analysis.belief_reward_planner import BeliefRewardPlanner
from environment import make_partial_obs_learning_env
from imitation.recurrent_iq_learn_agent import RecurrentOfflineSoftQAgent, make_args
from imitation.selection import evaluate_imitation, load_episodes


def reward_row(values, action, episode=0, resolved=False, horizon=3):
    metrics = reward_action_alignment(values, action)
    return dict(episode=episode, resolved=resolved,
                **{f'reward_h{horizon}_{key}': value for key, value in metrics.items()})


def small_agent(action=3):
    torch.manual_seed(0)
    agent = RecurrentOfflineSoftQAgent(28, 7, make_args(init_temp=.5, gamma=.9),
                                     latent_dim=4, rnn_hidden_dim=4)
    with torch.no_grad():
        agent.q_net.fc3.weight.zero_()
        agent.q_net.fc3.bias.zero_()
        agent.q_net.fc3.bias[action] = 1.
    return agent


class RewardAlignmentTests(unittest.TestCase):
    def test_regret_uses_actual_action_and_all_seven_reference_values(self):
        values = [0., 0., .9, .1, .1, .1, .1]
        for action in range(3, 7):
            result = reward_action_alignment(values, action)
            self.assertFalse(result['strict_agreement_unique_reward'])
            self.assertFalse(result['q_in_reward_maximizers'])
            self.assertAlmostEqual(result['action_regret'], .8)
        chosen = reward_action_alignment(values, 2)
        self.assertTrue(chosen['strict_agreement_unique_reward'])
        self.assertEqual(chosen['action_regret'], 0.)

    def test_ties_have_no_arbitrary_agreement_and_keep_exact_regret(self):
        result = reward_action_alignment([1., 1., 0., 0., 0., 0., 0.], 1)
        self.assertEqual(result['maximizers'], (0, 1))
        self.assertIsNone(result['strict_agreement_unique_reward'])
        self.assertTrue(result['q_in_reward_maximizers'])
        self.assertEqual(result['action_regret'], 0.)
        nearly_tied = reward_action_alignment([1., 1. - 5e-11, 0., 0., 0., 0., 0.], 1)
        self.assertTrue(nearly_tied['tie'])
        self.assertGreater(nearly_tied['action_regret'], 0.)
        self.assertFalse(reward_action_alignment([1., 1. - 2e-10, 0., 0., 0., 0., 0.], 1)['tie'])

    def test_denominators_separate_unique_informative_and_all_tied_states(self):
        rows = [reward_row([1., 0., 0., 0., 0., 0., 0.], 0),
                reward_row([1., 0., 0., 0., 0., 0., 0.], 3),
                reward_row([1., 1., 0., 0., 0., 0., 0.], 1, episode=1),
                reward_row([0.] * 7, 6, episode=2),
                reward_row([1., 0., 0., 0., 0., 0., 0.], 0, episode=3, resolved=True)]
        summary = summarize_reward_alignment(rows, [3])['3']
        unresolved = summary['unresolved']
        self.assertEqual(unresolved['steps'], 4)
        self.assertEqual(unresolved['episodes'], 3)
        self.assertEqual(unresolved['unique_reward_steps'], 2)
        self.assertEqual(unresolved['unique_reward_episodes'], 1)
        self.assertEqual(unresolved['reward_tie_steps'], 2)
        self.assertEqual(unresolved['all_reward_tied_steps'], 1)
        self.assertEqual(unresolved['informative_reward_steps'], 3)
        self.assertEqual(unresolved['informative_reward_episodes'], 2)
        self.assertEqual(unresolved['strict_agreement_unique_reward'], .5)
        self.assertAlmostEqual(unresolved['q_in_reward_maximizers'], 2 / 3)
        self.assertAlmostEqual(unresolved['mean_action_regret'], .25)
        self.assertAlmostEqual(unresolved['mean_action_regret_informative'], 1 / 3)
        self.assertEqual(summary['resolved']['strict_agreement_unique_reward'], 1.)
        empty = summarize_reward_alignment([], [3])['3']['unresolved']
        self.assertEqual(empty['steps'], 0)
        self.assertIsNone(empty['strict_agreement_unique_reward'])
        self.assertIsNone(empty['mean_action_regret'])
        all_tied = summarize_reward_alignment([reward_row([0.] * 7, 3)], [3])['3']['unresolved']
        self.assertEqual(all_tied['mean_action_regret'], 0.)
        self.assertIsNone(all_tied['mean_action_regret_informative'])
        self.assertIsNone(all_tied['q_in_reward_maximizers'])

    def test_short_horizon_zero_values_do_not_count_as_unique_agreement(self):
        planner = BeliefRewardPlanner(size=6, view_size=3, max_steps=40, gamma=.9)
        pose, candidates = (1, 1, 0), [(4, 1)]
        short = reward_action_alignment(planner.q_values(pose, candidates, 0, 1), 2)
        longer = reward_action_alignment(planner.q_values(pose, candidates, 0, 3), 2)
        self.assertTrue(short['all_tied'])
        self.assertIsNone(short['strict_agreement_unique_reward'])
        self.assertTrue(longer['strict_agreement_unique_reward'])


class LearningProgressTests(unittest.TestCase):
    def test_commands_use_recorded_environment_budget_and_requested_horizons(self):
        config = dict(size=6, view_size=3, gamma=.9, init_temp=.5, grid_shape=None,
                      max_steps=200000, env_max_steps=4)
        _, alignment = evaluate_learning_progress.evaluation_commands(
            config, 'policy.pt', 'valid.pkl', Path('out'), 2, 100, reward_horizons=[1, 3])
        self.assertEqual(alignment[alignment.index('--reward-horizons') + 1:alignment.index('--output')], ['1', '3'])
        self.assertEqual(alignment[alignment.index('--max-steps') + 1], '4')
        self.assertEqual(alignment[alignment.index('--gamma') + 1], '0.9')
        del config['env_max_steps']
        _, default = evaluate_learning_progress.evaluation_commands(
            config, 'policy.pt', 'valid.pkl', Path('out'), 2, 100)
        self.assertNotIn('--max-steps', default)
        self.assertEqual(default[default.index('--reward-horizons') + 1:default.index('--output')], ['3', '5'])

    def test_reward_planning_preserves_rollouts_and_agrees_with_independent_queries(self):
        agent = small_agent()
        with tempfile.TemporaryDirectory() as root:
            outputs, summaries = [], []
            for name, extra in [('legacy', []), ('reward', ['--reward-horizons', '1', '3'])]:
                output = Path(root) / f'{name}.csv'
                argv = ['alignment', '--checkpoint', 'unused.pt', '--init-temp', '.5',
                        '--gamma', '.9', '--size', '6', '--view-size', '3', '--max-steps', '4',
                        '--episodes', '2', '--seed', '100', '--output', str(output), *extra]
                with patch('sys.argv', argv), patch.object(evaluate_belief_alignment, 'load_policy', return_value=agent), contextlib.redirect_stdout(io.StringIO()):
                    evaluate_belief_alignment.main()
                with output.open() as stream:
                    outputs.append(list(csv.DictReader(stream)))
                summaries.append(json.loads(output.with_suffix('.json').read_text()))
            self.assertNotIn('reward_planning', summaries[0])
            for key in ('overall', 'resolved', 'unresolved', 'success_rate'):
                self.assertEqual(summaries[0][key], summaries[1][key])
            for legacy, reward in zip(*outputs, strict=True):
                self.assertEqual(legacy, {key: reward[key] for key in legacy})
            planner = BeliefRewardPlanner(size=6, view_size=3, max_steps=4, gamma=.9)
            self.assertEqual(summaries[1]['reward_planning']['max_steps'], 4)
            for row in outputs[1]:
                pose = tuple(int(row[k]) for k in ('agent_x', 'agent_y', 'direction'))
                candidates = json.loads(row['belief_candidates'])
                for horizon in (1, 3):
                    expected = planner.q_values(pose, candidates, int(row['timestep']), horizon)
                    actual = [float(row[f'reward_h{horizon}_q_{a}']) for a in range(7)]
                    np.testing.assert_allclose(actual, expected)
                    regret = expected.max() - expected[int(row['executed_action'])]
                    self.assertAlmostEqual(float(row[f'reward_h{horizon}_action_regret']), regret)

    def test_learning_progress_runs_real_evaluators_and_writes_reward_curves(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            env = make_partial_obs_learning_env(size=6, num_distractors=0, agent_view_size=3, max_steps=4)
            try:
                obs, _ = env.reset(seed=1234)
                states, next_states = [], []
                for _ in range(2):
                    states.append(obs)
                    obs, *_ = env.step(0)
                    next_states.append(obs)
            finally:
                env.close()
            data = dict(states=[np.array(states)], next_states=[np.array(next_states)],
                        actions=[np.zeros(2, dtype=np.int64)], rewards=[np.zeros(2)],
                        dones=[np.zeros(2, dtype=bool)])
            demos = root / 'validation.pkl'
            demos.write_bytes(pickle.dumps(data))
            config = dict(env='partial_obs', num_distractors=0, size=6, view_size=3,
                          max_steps=200000, env_max_steps=4, gamma=.9, init_temp=.5, grid_shape=None,
                          validation_demo_path=str(demos))
            records = []
            for step, action in enumerate((3, 0)):
                agent = small_agent(action)
                checkpoint = root / f'policy{step}.pt'
                agent.save(str(checkpoint))
                metrics = evaluate_imitation(agent, load_episodes(str(demos), 'cpu'))
                records.append(dict(step=step, checkpoint=str(checkpoint), **metrics))
            manifest = root / 'manifest.json'
            metadata = dict(config=config, evaluations=records,
                            validation_sha256=hashlib.sha256(demos.read_bytes()).hexdigest())
            manifest.write_text(json.dumps(metadata))
            argv = ['progress', '--manifest', str(manifest), '--output-dir', str(root / 'curves'),
                    '--episodes', '1', '--seed', '100', '--reward-horizons', '1', '3']
            cache = root / 'cache'
            cache.mkdir()
            with patch('sys.argv', argv), patch.dict(os.environ, {'MPLCONFIGDIR': str(root / 'mpl'), 'XDG_CACHE_HOME': str(cache), 'MPLBACKEND': 'Agg', 'OMP_NUM_THREADS': '1'}), contextlib.redirect_stdout(io.StringIO()):
                evaluate_learning_progress.main()
            with (root / 'curves' / 'curves.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row['step'] for row in rows], ['0', '1'])
            self.assertTrue((root / 'curves' / 'curves.png').is_file())
            recorded = json.loads((root / 'curves' / 'config.json').read_text())
            self.assertEqual(recorded['arguments']['reward_horizons'], [1, 3])
            for row, record in zip(rows, records, strict=True):
                self.assertAlmostEqual(float(row['validation_episode_nll']), record['episode_mean_nll'])
                report = json.loads((root / 'curves' / f"step_{record['step']:09d}" / 'alignment.json').read_text())
                for horizon in (1, 3):
                    group = report['reward_planning']['by_horizon'][str(horizon)]['unresolved']
                    for curve, metric in [('strict_reward_agreement', 'strict_agreement_unique_reward'),
                                          ('reward_action_regret', 'mean_action_regret'),
                                          ('reward_action_regret_informative', 'mean_action_regret_informative')]:
                        value = row[f'{curve}_h{horizon}']
                        if group[metric] is None:
                            self.assertEqual(value, '')
                        else:
                            self.assertAlmostEqual(float(value), group[metric])
                    self.assertEqual(int(row[f'unique_reward_steps_h{horizon}']), group['unique_reward_steps'])
                    self.assertEqual(int(row[f'all_reward_tied_steps_h{horizon}']), group['all_reward_tied_steps'])
            self.assertEqual(json.loads(manifest.read_text()), metadata)


if __name__ == '__main__':
    unittest.main()
