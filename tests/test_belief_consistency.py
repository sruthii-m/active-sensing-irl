import unittest
from types import SimpleNamespace
import numpy as np
import torch
from analysis.evaluate_belief_consistency import history_features, update_errors
from analysis.evaluate_belief_decoding import fit_probe


class ConsistencyTests(unittest.TestCase):
    def test_action_history_excludes_target_and_resets(self):
        data = dict(episode=np.array([0, 0, 1]), timestep=np.array([0, 1, 0]),
                    action=np.array([2, 1, 0]), pose=np.tile([.5, .5], (3, 1)))
        first = history_features(data, 3)
        self.assertEqual(first[0, :28].sum(), 0)
        self.assertEqual(first[1, 2], 1)
        self.assertEqual(first[2, :28].sum(), 0)
        data['action'][1] = 6
        np.testing.assert_array_equal(first, history_features(data, 3))

    def test_updates_exclude_boundaries_and_resolved_priors(self):
        data = dict(episode=np.array([0, 0, 0, 1, 1]),
                    timestep=np.array([0, 1, 2, 0, 1]),
                    belief=np.array([[.5, .5], [1., 0], [1., 0], [.5, .5], [.5, .5]]))
        error, eligible, changed = update_errors(data['belief'], data)
        np.testing.assert_array_equal(eligible, [False, True, False, False, True])
        self.assertTrue(changed[1])
        self.assertFalse(changed[4])
        np.testing.assert_array_equal(error, np.zeros(5))
        error, _, _ = update_errors(np.full((5, 2), .5), data)
        self.assertEqual(error[1], 1.)

    def test_action_fit_does_not_use_test_labels(self):
        data = dict(split=np.array([0, 0, 1, 1, 2, 2]), episode=np.arange(6),
                    belief=np.eye(2)[[0, 1, 0, 1, 0, 1]])
        args = SimpleNamespace(probe_seed=0, width=4, lr=.001, epochs=3,
                               batch_size=2, patience=0)
        x = np.arange(18, dtype=np.float32).reshape(6, 3)
        first = fit_probe(x, data, args, np.ones(6, bool))[0]
        data['belief'][4:] = data['belief'][4:, ::-1]
        second = fit_probe(x, data, args, np.ones(6, bool))[0]
        self.assertEqual(first['selected_epoch'], second['selected_epoch'])
        for key in first['state_dict']:
            torch.testing.assert_close(first['state_dict'][key], second['state_dict'][key])


if __name__ == '__main__':
    unittest.main()
