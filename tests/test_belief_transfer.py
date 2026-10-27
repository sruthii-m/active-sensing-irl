import unittest
import numpy as np
from analysis.transfer_belief_subspace import fit_subspace, donor_delta, cluster_summary


class BeliefTransferTest(unittest.TestCase):
    def test_fit_uses_training_records_only(self):
        rng = np.random.default_rng(42)
        data = dict(hidden=rng.normal(size=(80, 12)), belief=rng.dirichlet(np.ones(6), size=80),
                    pose=rng.normal(size=(80, 6)), timestep=np.arange(80),
                    split=np.repeat([0, 1], 40), episode=np.repeat(np.arange(8), 10))
        first = fit_subspace(data, rank=3)
        for key in ('hidden', 'belief', 'pose'):
            data[key][40:] *= 500
        second = fit_subspace(data, rank=3)
        for key in first:
            np.testing.assert_allclose(first[key], second[key])

    def test_transfer_changes_only_selected_subspace(self):
        rng = np.random.default_rng(0)
        basis = np.linalg.qr(rng.normal(size=(12, 3)))[0]
        a, b = rng.normal(size=(2, 12))
        edited = a + donor_delta(a, b, basis)
        np.testing.assert_allclose(edited @ basis, b @ basis, atol=1e-12)
        projector = basis @ basis.T
        np.testing.assert_allclose(edited @ (np.eye(12)-projector), a @ (np.eye(12)-projector), atol=1e-12)
        np.testing.assert_allclose(donor_delta(a, a, basis), 0)

    def test_bootstrap_clusters_directions_and_distances(self):
        rows = [dict(pair_id=0, reset_seed=0, score=1), dict(pair_id=0, reset_seed=0, score=1),
                dict(pair_id=1, reset_seed=1, score=0), dict(pair_id=2, reset_seed=1, score=0)]
        result = cluster_summary(rows, 'score')
        self.assertEqual(result['mean'], .5)
        self.assertEqual(result['reset_seeds'], 2)
        self.assertEqual(result['pairs'], 3)


if __name__ == '__main__':
    unittest.main()
