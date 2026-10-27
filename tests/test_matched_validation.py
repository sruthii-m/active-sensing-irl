import unittest
import numpy as np
from analysis.evaluate_matched_histories import pair_metrics
from analysis.validate_matched_histories import reconstructed_pair


class MatchedValidationTest(unittest.TestCase):
    def test_explicit_control_changes_in_predicted_direction(self):
        a = dict(q=np.array([1., 0, 0, 0, 0, 0, 0]), ig=np.array([1., 0, 0]),
                 candidates=((1, 1), (1, 2)), observations=np.array([[0], [1]]))
        b = dict(q=np.array([0., 1, 0, 0, 0, 0, 0]), ig=np.array([0., 1, 0]),
                 candidates=((2, 1), (2, 2)), observations=np.array([[2], [1]]))
        r = pair_metrics(a, b, .25)
        self.assertTrue(r['different_unique_ig'])
        self.assertTrue(r['both_actions_match_ig'])
        self.assertGreater(r['own_ig_support_shift'], 0)
        # Actual non-navigation argmax must fail strict agreement.
        a['q'][3] = 2
        self.assertFalse(pair_metrics(a, b, .25)['both_actions_match_ig'])
        self.assertTrue(pair_metrics(a, b, .25)['different_unique_ig'])

    def test_reject_mismatched_endpoints(self):
        traces = {}
        for label in ('a', 'b'):
            for key, value in dict(observations=np.array([[0], [1]]),
                                   positions=np.array([[1, 1, 0], [1, 1, 0]]),
                                   candidates=np.array([[2, 1], [2, 2]]),
                                   q=np.zeros(7), ig=np.zeros(3)).items():
                traces[f'pair0_{label}_{key}'] = value
        reconstructed_pair(traces, 0)
        traces['pair0_b_positions'][-1, -1] = 1
        with self.assertRaises(ValueError):
            reconstructed_pair(traces, 0)


if __name__ == '__main__':
    unittest.main()
