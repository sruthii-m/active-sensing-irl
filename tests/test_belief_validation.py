import unittest
import numpy as np
from analysis.belief_reward_planner import BeliefRewardPlanner
from analysis.validate_belief_consistency import budget_width, information_policy, shuffled_beliefs


class BeliefValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.planner = BeliefRewardPlanner(size=6, view_size=3)

    def test_parameter_budget_is_nearest_feasible_width(self):
        for inputs in (32, 64, 200, 300):
            width = budget_width(inputs, 7, 20000)
            count = width * (inputs + 8) + 7
            self.assertLessEqual(abs(count - 20000), (inputs + 8) / 2)
        with self.assertRaises(ValueError):
            budget_width(300, 7, 10)

    def test_placebo_never_crosses_splits(self):
        data = dict(split=np.repeat(np.arange(3), 10), belief=np.arange(60).reshape(30, 2))
        actual = shuffled_beliefs(data, 42)
        for split in range(3):
            mask = data['split'] == split
            np.testing.assert_array_equal(np.sort(actual[mask], axis=0), data['belief'][mask])
        np.testing.assert_array_equal(actual, shuffled_beliefs(data, 42))
        self.assertFalse(np.array_equal(actual, data['belief']))

    def test_information_gain_matches_uniform_enumeration(self):
        planner = self.planner
        b = np.ones((1, len(planner.cells))) / len(planner.cells)
        pose = np.array([[2/5, 2/5, 1, 0, 0, 0]])
        probs, scores = information_policy(b, pose, planner.cells, planner)
        n = len(planner.cells)
        for action in range(7):
            nxt = planner.transition((2, 2, 0), action)
            revealed = set(planner.visibility[nxt])
            if action == 2 and nxt[:2] != (2, 2):
                revealed.add(planner.index[nxt[:2]])
            remaining = n - len(revealed)
            expected = np.log(n) - (remaining / n * np.log(remaining) if remaining else 0)
            self.assertAlmostEqual(scores[0, action], expected)
        self.assertAlmostEqual(probs.sum(), 1.)
        self.assertTrue(np.all(probs > 0))

    def test_policy_changes_with_belief_at_fixed_pose(self):
        planner = self.planner
        rng = np.random.default_rng(0)
        beliefs = rng.dirichlet(np.ones(len(planner.cells)), size=20)
        poses = np.tile([2/5, 2/5, 1, 0, 0, 0], (20, 1))
        probs, _ = information_policy(beliefs, poses, planner.cells, planner)
        self.assertGreater(np.abs(probs[1:] - probs[0]).max(), .05)
        # A resolved belief has zero entropy regardless of the observation geometry.
        resolved = np.eye(len(planner.cells))[:1]
        uniform, scores = information_policy(resolved, poses[:1], planner.cells, planner)
        np.testing.assert_allclose(scores, 0.)
        np.testing.assert_allclose(uniform, 1/7)


if __name__ == '__main__':
    unittest.main()
