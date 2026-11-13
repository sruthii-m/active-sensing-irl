"""Ensure expert failures and long successes survive default demo generation."""
import contextlib
import io
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from imitation import generate_full_state_demos as full
from imitation import generate_partial_obs_demos as partial


class Model:
    def predict(self, obs, **kwargs):
        return 0, None


class Environment:
    def __init__(self):
        self.seeds = []
        self.closed = False
        self.unwrapped = self
        self.max_steps = 3

    def reset(self, seed):
        self.seeds.append(seed)
        self.step_count = 0
        return np.zeros(3), {}

    def step(self, action):
        self.step_count += 1
        end = self.step_count == 3
        success = len(self.seeds) % 2 == 0
        return np.ones(3), float(end and success), end and success, end and not success, {}

    def close(self):
        self.closed = True


class DemoSelectionTests(unittest.TestCase):
    def test_default_keeps_failures_and_long_episodes_in_both_generators(self):
        for module, loader, loader_name, factory in [(full, full.PPO, 'load', 'make_full_obs_learning_env'),
                                        (partial, partial, 'load_model', 'make_partial_obs_learning_env')]:
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as root:
                env = Environment()
                path = Path(root) / 'demos.pkl'
                kwargs = {'max_length': 1} if module is partial else {}
                with patch.object(loader, loader_name, return_value=Model()), patch.object(module, factory, return_value=env), contextlib.redirect_stdout(io.StringIO()):
                    module.generate('expert', {}, 2, str(path), seed=42, **kwargs)
                with path.open('rb') as stream:
                    data = pickle.load(stream)
                self.assertEqual(data['selection'], 'all')
                self.assertEqual(data['attempts'], 2)
                self.assertEqual(env.seeds, [42, 43])
                self.assertEqual([r.sum() for r in data['rewards']], [0., 1.])
                self.assertEqual(data['lengths'].tolist(), [3, 3])
                self.assertEqual(data['episode_seeds'], [42, 43])
                self.assertEqual(data['task_horizon'], 3)
                self.assertTrue(all(d[-1] for d in data['task_terminal']))
                self.assertTrue(data['truncated'][0][-1])
                self.assertFalse(data['terminated'][0][-1])
                self.assertTrue(env.closed)


if __name__ == '__main__':
    unittest.main()
