"""Causal replay, rollout, checkpoint, and finite-horizon regression tests."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from analysis.expert_likelihood import evaluate
from analysis.policy_io import load_policy, representation
from analysis.reward_maps import counterfactual_rewards
from imitation.episode_contract import task_terminal, terminal_masks
from imitation.iq_learn_agent import OfflineSoftQAgent, make_args
from imitation.recurrent_iq_learn_agent import RecurrentOfflineSoftQAgent
from imitation.selection import evaluate_imitation


def agent(**kwargs):
    torch.manual_seed(19)
    return RecurrentOfflineSoftQAgent(4, 3, make_args(init_temp=.5),
                                     latent_dim=5, rnn_hidden_dim=6, **kwargs)


def episode(length=4):
    frames = torch.arange((length + 1) * 4, dtype=torch.float32).reshape(-1, 4) / 20
    return frames[:-1], frames[1:], torch.arange(length).remainder(3)[:, None], torch.zeros(length, 1)


class RecurrentContractTests(unittest.TestCase):
    def test_batch_sequence_and_inference_match_with_unequal_lengths(self):
        policy = agent()
        episodes = [episode(4), episode(1), episode(2)]
        current, following, actions, dones = policy._encode_batch(episodes)
        histories = [policy.encode_episode(s, n, a) for s, n, a, _ in episodes]
        torch.testing.assert_close(current, torch.cat([h[:-1] for h in histories]))
        torch.testing.assert_close(following, torch.cat([h[1:] for h in histories]))
        for ep, expected in zip(episodes, histories):
            states, next_states, acts, _ = ep
            hidden = None
            for t, obs in enumerate(torch.cat((states, next_states[-1:]))):
                hidden = representation(policy, obs, hidden, -1 if t == 0 else acts[t - 1])
                torch.testing.assert_close(hidden[0], expected[t])

    def test_current_action_cannot_leak_into_current_hidden_or_q(self):
        policy = agent()
        states, next_states, actions, _ = episode()
        altered = actions.clone()
        altered[1] = (altered[1] + 1) % 3
        original = policy.encode_episode(states, next_states, actions)
        changed = policy.encode_episode(states, next_states, altered)
        torch.testing.assert_close(original[:2], changed[:2], rtol=0, atol=0)
        torch.testing.assert_close(policy.q_values(original[:2]), policy.q_values(changed[:2]), rtol=0, atol=0)
        self.assertGreater(float((original[2] - changed[2]).abs().max()), 1e-5)
        final_changed = actions.clone()
        final_changed[-1] = (final_changed[-1] + 1) % 3
        final = policy.encode_episode(states, next_states, final_changed)
        torch.testing.assert_close(original[:-1], final[:-1], rtol=0, atol=0)
        self.assertGreater(float((original[-1] - final[-1]).abs().max()), 1e-5)

    def test_reset_is_distinct_and_missing_or_invalid_actions_fail(self):
        policy = agent()
        obs = episode()[0][:1]
        reset, _ = policy.encoder(obs, None)
        action_zero, _ = policy.encoder(obs, None, 0)
        self.assertFalse(torch.equal(reset, action_zero))
        with self.assertRaisesRegex(ValueError, 'previous action'):
            policy.encoder(obs, reset)
        for token in (-2, 3, .5, True):
            with self.subTest(token=token), self.assertRaises(ValueError):
                policy.encoder(obs, reset, token)
        with self.assertRaisesRegex(ValueError, 'one action'):
            policy.encode_episode(*episode()[:2])

    def test_rollout_tracks_the_executed_action_and_resets_each_episode(self):
        policy = agent()
        states = episode()[0]
        for deterministic in (True, False):
            policy.reset_hidden()
            hidden, previous = None, -1
            for obs in states:
                expected = representation(policy, obs, hidden, previous)
                action = policy.act(obs, deterministic=deterministic)
                torch.testing.assert_close(policy._hidden, expected)
                self.assertEqual(policy._previous_action, action)
                hidden, previous = expected, action
            policy.reset_hidden()
            policy.act(states[0])
            torch.testing.assert_close(policy._hidden, representation(policy, states[0]))

    def test_training_and_likelihood_evaluators_use_same_causal_features(self):
        policy = agent()
        ep = episode()
        states, next_states, actions, _ = ep
        data = dict(states=[states.numpy()], next_states=[next_states.numpy()],
                    actions=[actions.flatten().numpy()])
        selection = evaluate_imitation(policy, [ep])
        measured = evaluate(policy, data)
        self.assertAlmostEqual(selection['episode_mean_nll'], -measured['episode_mean_log_likelihood'], places=6)
        before = policy.encoder.gru.weight_ih.detach().clone()
        losses = policy.update([ep], step=1, target_update_freq=1)
        self.assertTrue(all(np.isfinite(v) for v in losses.values()))
        self.assertGreater(float((policy.encoder.gru.weight_ih[:, 5:] - before[:, 5:]).abs().max()), 0)
        self.assertTrue(all(p.grad is None for p in policy.target_net.parameters()))

    def test_new_and_legacy_checkpoints_roundtrip_without_changing_protocol(self):
        with tempfile.TemporaryDirectory() as root:
            for conditioned in (True, False):
                original = agent(condition_on_action=conditioned)
                path = Path(root) / f'{conditioned}.pt'
                original.save(path)
                args = SimpleNamespace(checkpoint=path, init_temp=.5, gamma=.99, grid_shape=None)
                restored = load_policy(args, 4)
                self.assertEqual(restored.config['condition_on_action'], conditioned)
                s, n, a, _ = episode()
                torch.testing.assert_close(original.encode_episode(s, n, a), restored.encode_episode(s, n, a))
                if not conditioned:
                    torch.save(dict(encoder=original.encoder.state_dict(), q_net=original.q_net.state_dict()), path)
                    legacy = load_policy(args, 4)
                    torch.testing.assert_close(original.encode_episode(s, n), legacy.encode_episode(s, n))
                    with self.assertRaisesRegex(ValueError, 'Legacy'):
                        agent(condition_on_action=False, terminal_semantics='intrinsic_horizon').load(path)
                    # Explicitly verify the original observation-only recurrence.
                    hidden = None
                    for obs, expected in zip(s, legacy.encode_episode(s, n)):
                        latent = original.encoder._featurize(obs[None])
                        hidden = original.encoder.gru(latent, hidden)
                        torch.testing.assert_close(hidden[0], expected)
                else:
                    args.init_temp = .1
                    with self.assertRaisesRegex(ValueError, 'metadata'):
                        load_policy(args, 4)
                    with self.assertRaisesRegex(ValueError, 'configuration'):
                        agent(condition_on_action=False).load(path)

    def test_cnn_checkpoint_infers_observation_latent_width_separately_from_action_tokens(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'cnn.pt'
            original = RecurrentOfflineSoftQAgent(28, 7, make_args(init_temp=.5), latent_dim=4,
                                                  rnn_hidden_dim=6, grid_shape=(3, 3, 3))
            original.save(path)
            args = SimpleNamespace(checkpoint=path, init_temp=.5, gamma=.99, grid_shape=None)
            restored = load_policy(args, 28)
            self.assertEqual(restored.encoder.gru.input_size, 4 + 8)
            obs = torch.zeros(28)
            torch.testing.assert_close(representation(original, obs), representation(restored, obs))
            args.grid_shape = [7, 7, 3]
            with self.assertRaisesRegex(ValueError, 'Grid shape'):
                load_policy(args, 28)

    def test_full_state_checkpoint_preserves_terminal_protocol(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'full.pt'
            original = OfflineSoftQAgent(4, 3, make_args(init_temp=.5), terminal_semantics='intrinsic_horizon')
            original.save(path)
            args = SimpleNamespace(checkpoint=path, init_temp=.5, gamma=.99, grid_shape=None)
            restored = load_policy(args, 4)
            self.assertEqual(restored.config['terminal_semantics'], 'intrinsic_horizon')
            torch.testing.assert_close(original.q_values(episode()[0]), restored.q_values(episode()[0]))
            torch.save(original.q_net.state_dict(), path)
            self.assertEqual(load_policy(args, 4).config['terminal_semantics'], 'recorded')


class TerminalContractTests(unittest.TestCase):
    def test_deadline_goal_and_collection_cutoff_are_distinct(self):
        self.assertTrue(task_terminal(False, True, 400, 400))
        self.assertTrue(task_terminal(True, False, 15, 400))
        self.assertFalse(task_terminal(False, True, 15, 400))
        data = dict(dones=[[False] * 4, [False] * 2, [False, True]], task_horizon=4)
        before, audit = deepcopy(data), {}
        masks = terminal_masks(data, 'intrinsic_horizon', audit=audit)
        self.assertEqual([bool(m[-1]) for m in masks], [True, False, True])
        self.assertEqual(audit['changed_transitions'], 1)
        self.assertEqual(data, before)
        self.assertFalse(terminal_masks(data)[0][-1])
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            terminal_masks(data, 'intrinsic_horizon', horizon=5)
        with self.assertRaisesRegex(ValueError, 'reset'):
            terminal_masks({**data, 'starts_at_reset': False}, 'intrinsic_horizon')

    def test_raw_flags_override_conflated_collection_done(self):
        data = dict(dones=[[False, True]], terminated=[[False, False]],
                    truncated=[[False, True]], task_horizon=4, starts_at_reset=True)
        masks = terminal_masks(data, 'intrinsic_horizon')
        self.assertFalse(masks[0][-1])

    def test_recording_new_demo_masks_preserves_their_intrinsic_semantics(self):
        data = dict(dones=[[False, True]], terminal_semantics='intrinsic_horizon', task_horizon=2)
        audit = {}
        terminal_masks(data, 'recorded', audit=audit)
        self.assertEqual(audit['effective_semantics'], 'intrinsic_horizon')
        self.assertEqual(audit['changed_transitions'], 0)

    def test_counterfactuals_feed_branch_action_and_use_checkpoint_terminal_mask(self):
        class Environment:
            action_space = SimpleNamespace(n=3)
            max_steps = 4
            step_count = 3
            render_mode = None

            @property
            def unwrapped(self):
                return self

            def step(self, action):
                self.step_count += 1
                return np.ones(4), 0., False, self.step_count >= self.max_steps, {}

            def close(self):
                pass

        env = Environment()
        for mode in ('intrinsic_horizon', 'recorded'):
            policy = agent(terminal_semantics=mode)
            hidden = representation(policy, np.zeros(4))
            values = counterfactual_rewards(policy, env, hidden)
            q = policy.q_values(hidden)[0].detach().numpy()
            if mode == 'intrinsic_horizon':
                np.testing.assert_allclose(values, q)
            else:
                expected = [q[a] - policy.gamma * float(policy.getV(representation(policy, np.ones(4), hidden, a)))
                            for a in range(3)]
                np.testing.assert_allclose(values, expected, rtol=1e-5, atol=1e-7)
            self.assertEqual(env.step_count, 3)


if __name__ == '__main__':
    unittest.main()
