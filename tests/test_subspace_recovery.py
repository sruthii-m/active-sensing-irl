import unittest
import numpy as np
from analysis.validate_subspace_recovery import make_network, oracle_transfer, probabilities, heldout_pairs


class SubspaceRecoveryTest(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(4)
        return dict(belief=rng.dirichlet(np.ones(10), size=90), split=np.repeat([0, 1, 2], 30),
                    episode=np.repeat(np.arange(9), 10))

    def test_oracle_replaces_belief_without_changing_nuisance(self):
        network = make_network(self.data(), 8, 32, 0)
        source, donor = network['hidden'][60:65], network['hidden'][70:75]
        edited = oracle_transfer(source, donor, network['rotation'], 8)
        np.testing.assert_allclose(probabilities(edited, network['rotation'], network['action_weights']),
                                   probabilities(donor, network['rotation'], network['action_weights']), atol=1e-12)
        np.testing.assert_allclose(probabilities(edited, network['rotation'], network['noise_weights'], 8),
                                   probabilities(source, network['rotation'], network['noise_weights'], 8), atol=1e-12)

    def test_heldout_posteriors_do_not_change_network_construction(self):
        data = self.data()
        first = make_network(data, 8, 32, 0)
        data['belief'][30:] = np.roll(data['belief'][30:], 1, axis=1)
        second = make_network(data, 8, 32, 0)
        for key in ('rotation', 'components', 'scales', 'posterior_mean', 'action_weights', 'noise_weights'):
            np.testing.assert_allclose(first[key], second[key])
        np.testing.assert_allclose(first['hidden'][:30], second['hidden'][:30])

    def test_pairing_keeps_train_out_and_uses_different_episodes(self):
        data = self.data()
        source, donor = heldout_pairs(data)
        self.assertTrue(np.all(data['split'][source] == 2))
        self.assertTrue(np.all(data['split'][donor] == 2))
        self.assertTrue(np.all(data['episode'][source] != data['episode'][donor]))
        self.assertEqual(len(set(data['episode'][source])), len(source))


if __name__ == '__main__':
    unittest.main()
