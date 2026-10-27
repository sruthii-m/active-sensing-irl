"""Scientific invariants for frozen-policy posterior decoding."""
import unittest
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from analysis.evaluate_belief_decoding import (
    distribution_scores, episode_weights, layout_split, paired_interval, load_cached_rollouts,
    fit_probe,
)


class BeliefDecodingTest(unittest.TestCase):
    def test_test_targets_cannot_change_fit_or_epoch_selection(self):
        features = np.arange(18, dtype=np.float32).reshape(6, 3) / 18
        data = dict(split=np.array([0, 0, 1, 1, 2, 2]), episode=np.arange(6),
                    belief=np.tile([.5, .5], (6, 1)))
        args = SimpleNamespace(probe_seed=0, width=4, lr=.001, epochs=3,
                               batch_size=2, patience=0)
        first = fit_probe(features, data, args)[0]
        data['belief'][4:] = [1., 0.]
        second = fit_probe(features, data, args)[0]
        self.assertEqual(first['history'], second['history'])
        self.assertEqual(first['selected_epoch'], second['selected_epoch'])
        for key in first['state_dict']:
            torch.testing.assert_close(first['state_dict'][key], second['state_dict'][key])

    def test_identical_posteriors_have_zero_kl_including_zero_mass(self):
        target = torch.tensor([[.5, 0., .5], [0., 1., 0.]])
        ce, kl = distribution_scores(target.clamp_min(1e-30).log(), target)
        torch.testing.assert_close(kl, torch.zeros(2))
        self.assertAlmostEqual(ce[0].item(), np.log(2), places=6)
        self.assertEqual(ce[1].item(), 0.)

    def test_layouts_cannot_leak_across_splits(self):
        layouts = np.repeat(np.arange(20)[:, None], 7, axis=0)
        split, ids = layout_split(layouts, 42)
        self.assertEqual(set(split), {0, 1, 2})
        for layout in np.unique(ids):
            self.assertEqual(len(np.unique(split[ids == layout])), 1)
        np.testing.assert_array_equal(split, layout_split(layouts, 42)[0])

    def test_long_episodes_do_not_dominate(self):
        episodes = np.array([0, 1, 1, 1])
        values = np.array([4., 0., 0., 0.])
        self.assertAlmostEqual(np.average(values, weights=episode_weights(episodes)), 2.)

    def test_paired_bootstrap_preserves_difference_direction(self):
        data = dict(episode=np.repeat(np.arange(10), 3),
                    layout_id=np.repeat(np.arange(10), 3))
        result = paired_interval(np.full(30, .25), data, np.ones(30, dtype=bool), 0)
        self.assertEqual(result['mean'], .25)
        self.assertEqual(result['ci95'], [.25, .25])

    def test_cached_splits_stay_fixed_across_initializations_and_reject_wrong_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'policy.pt'
            checkpoint.write_bytes(b'checkpoint fingerprint fixture')
            config = dict(size=10, view_size=3, max_steps=None, episodes=3,
                          rollout_seed=200000, grid_shape=[3, 3, 3], init_temp=.001,
                          gamma=.99, probe_seed=0)
            (root / 'summary.json').write_text(json.dumps(dict(config=config,
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())))
            cached = root / 'rollouts.npz'
            np.savez(cached, split=[0, 1, 2], episode=[0, 1, 2], layout_id=[0, 1, 2])
            args = SimpleNamespace(**{**config, 'probe_seed': 9}, split_seed=0,
                                   checkpoint=str(checkpoint), cached_rollouts=str(cached))
            np.testing.assert_array_equal(load_cached_rollouts(args)['split'], [0, 1, 2])
            checkpoint.write_bytes(b'wrong checkpoint')
            with self.assertRaisesRegex(ValueError, 'different checkpoint'):
                load_cached_rollouts(args)

    def test_cached_layout_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'policy.pt'
            checkpoint.write_bytes(b'checkpoint')
            config = dict(size=10, view_size=3, max_steps=None, episodes=3,
                          rollout_seed=200000, grid_shape=[3, 3, 3], init_temp=.001,
                          gamma=.99, probe_seed=0)
            (root / 'summary.json').write_text(json.dumps(dict(config=config,
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())))
            cached = root / 'rollouts.npz'
            np.savez(cached, split=[0, 1, 2], episode=[0, 1, 2], layout_id=[0, 0, 2])
            args = SimpleNamespace(**config, split_seed=0, checkpoint=str(checkpoint),
                                   cached_rollouts=str(cached))
            with self.assertRaisesRegex(ValueError, 'leak layout_id'):
                load_cached_rollouts(args)


if __name__ == '__main__':
    unittest.main()
