import unittest
import numpy as np
from analysis.select_belief_rank import select_rank


class BeliefRankTest(unittest.TestCase):
    def test_known_linear_belief_selects_nonzero_rank(self):
        rng = np.random.default_rng(4)
        belief = rng.dirichlet(np.ones(6), size=300)
        hidden = np.c_[belief, rng.normal(size=(300, 6))]
        data = dict(hidden=hidden, belief=belief, pose=rng.normal(size=(300, 6)),
                    timestep=np.tile(np.arange(10), 30), episode=np.repeat(np.arange(30), 10),
                    split=np.repeat([0, 1, 2], 100))
        result, _ = select_rank(data, [0, 2, 4, 5], .01)
        self.assertEqual(result['selected_rank'], 5)
        self.assertLess(result['curve'][-1]['validation_mse'], result['curve'][0]['validation_mse'] / 100)

    def test_test_data_and_actions_cannot_change_selection(self):
        rng = np.random.default_rng(0)
        data = dict(hidden=rng.normal(size=(120, 12)), belief=rng.dirichlet(np.ones(6), size=120),
                    pose=rng.normal(size=(120, 6)), timestep=np.tile(np.arange(10), 12),
                    episode=np.repeat(np.arange(12), 10), split=np.repeat([0, 1, 2], 40),
                    action=rng.integers(7, size=120))
        first, _ = select_rank(data, [0, 2, 4], .01)
        for key in ('hidden', 'belief', 'pose'):
            data[key][80:] *= 1000
        data['action'][:] = 6
        second, _ = select_rank(data, [0, 2, 4], .01)
        self.assertEqual(first, second)
        expected = min(first['curve'], key=lambda r:(r['validation_mse'],r['rank']))['rank']
        self.assertEqual(first['selected_rank'], expected)


if __name__ == '__main__':
    unittest.main()
