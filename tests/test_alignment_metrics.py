"""Regression checks for actual-policy versus navigation-only alignment."""
import unittest

from analysis.alignment_metrics import action_alignment, summarize


ACTIONS = (0, 1, 2)


def row(q, scores=None, resolved=False):
    scores = scores if scores is not None else {0: 3., 1: 2., 2: 1.}
    best = max(scores.values())
    ties = tuple(a for a in ACTIONS if scores[a] == best)
    return dict(resolved=resolved, candidates=1 if resolved else 4,
                **action_alignment(q, scores, ties[0], ties, ACTIONS))


class AlignmentMetricsTest(unittest.TestCase):
    def test_each_non_navigation_argmax_is_strict_disagreement(self):
        for action in range(3, 7):
            with self.subTest(action=action):
                q = [3., 2., 1., 0., 0., 0., 0.]
                q[action] = 4.
                result = row(q)
                self.assertEqual(result['executed_action'], action)
                self.assertEqual(result['iq_argmax'], action)
                self.assertEqual(result['iq_argmax_all'], action)
                self.assertEqual(result['iq_argmax_navigation'], 0)
                self.assertTrue(result['policy_argmax_outside_navigation'])
                self.assertFalse(result['agreement'])
                self.assertFalse(result['q_in_ig_maximizers'])
                self.assertTrue(result['agreement_navigation_conditional'])
                self.assertTrue(result['q_in_ig_maximizers_navigation_conditional'])
                self.assertAlmostEqual(result['spearman_navigation'], 1.)

    def test_navigation_match_and_mismatch(self):
        for q, expected in [([3, 2, 1, 0, 0, 0, 0], True),
                            ([2, 3, 1, 0, 0, 0, 0], False)]:
            result = row(q)
            self.assertFalse(result['policy_argmax_outside_navigation'])
            self.assertEqual(result['agreement'], expected)
            self.assertEqual(result['agreement_navigation_conditional'], expected)

    def test_ig_ties_distinguish_exact_agreement_from_membership(self):
        result = row([1, 3, 2, 0, 0, 0, 0], {0: 2., 1: 2., 2: 0.})
        self.assertFalse(result['agreement'])
        self.assertTrue(result['q_in_ig_maximizers'])
        self.assertTrue(result['ig_tie'])
        self.assertFalse(result['ig_all_tied'])

    def test_q_ties_use_actual_numpy_argmax(self):
        result = row([3, 3, 1, 3, 0, 0, 0])
        self.assertEqual(result['executed_action'], 0)
        self.assertTrue(result['agreement'])
        self.assertTrue(result['q_tie'])

    def test_summary_denominators_and_belief_groups(self):
        rows = [row([3, 2, 1, 0, 0, 0, 0]),
                row([3, 2, 1, 4, 0, 0, 0]),
                row([1, 3, 2, 0, 0, 0, 0], {0: 2., 1: 2., 2: 0.}),
                row([3, 2, 1, 4, 0, 0, 0], {0: 0., 1: 0., 2: 0.}, resolved=True)]
        result = summarize(rows)
        overall = result['overall']
        self.assertEqual(overall['steps'], 4)
        self.assertEqual(overall['agreement'], .25)
        self.assertEqual(overall['agreement_navigation_conditional'], .75)
        self.assertEqual(overall['policy_argmax_outside_navigation_fraction'], .5)
        self.assertEqual(overall['unique_ig_steps'], 2)
        self.assertEqual(overall['agreement_unique_ig'], .5)
        self.assertEqual(overall['agreement_unique_ig_navigation_conditional'], 1.)
        self.assertEqual(overall['informative_steps'], 3)
        self.assertAlmostEqual(overall['q_in_ig_maximizers'], 2 / 3)
        self.assertEqual(overall['q_in_ig_maximizers_navigation_conditional'], 1.)
        self.assertEqual(overall['rank_steps_navigation'], 3)
        self.assertEqual(result['resolved']['policy_argmax_outside_navigation_fraction'], 1.)
        self.assertIsNone(result['resolved']['agreement_unique_ig'])
        self.assertIsNone(result['resolved']['q_in_ig_maximizers'])
        self.assertIsNone(result['resolved']['mean_spearman_navigation'])
        self.assertEqual(result['by_candidate_count']['4'], result['unresolved'])

    def test_empty_summary(self):
        overall = summarize([])['overall']
        self.assertEqual(overall['steps'], 0)
        self.assertIsNone(overall['agreement'])
        self.assertIsNone(overall['agreement_navigation_conditional'])
        self.assertIsNone(overall['policy_argmax_outside_navigation_fraction'])


if __name__ == '__main__':
    unittest.main()
