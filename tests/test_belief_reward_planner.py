import unittest
from copy import deepcopy

import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal

from environment.forage_env import make_forage_env
from analysis.belief_reward_planner import BeliefRewardPlanner, maximizers


def simulator_q(envs, horizon, gamma):
    """Independent small belief tree, branching on actual simulator observations."""
    values = []
    for action in range(4):
        immediate, branches = 0., {}
        for original in envs:
            env = deepcopy(original)
            obs, reward, terminated, truncated, _ = env.step(action)
            immediate += reward
            if not (terminated or truncated):
                key = (obs['image'].tobytes(), obs['direction'])
                branches.setdefault(key, []).append(env)
        continuation = 0.
        if horizon > 1:
            continuation = sum(len(group) * max(simulator_q(group, horizon - 1, gamma))
                               for group in branches.values())
        values.append((immediate + gamma * continuation) / len(envs))
    return np.array(values[:3] + [values[3]] * 4)


class BeliefRewardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.planner = BeliefRewardPlanner(size=5, view_size=3, max_steps=40, gamma=.9)

    def test_immediate_goal_entry_and_final_timestep(self):
        p = self.planner
        q = p.q_values((2,2,0), [(3,2)], 39, 5)
        self.assertAlmostEqual(q[2], .1)
        np.testing.assert_allclose(q[[0,1,3,4,5,6]], 0)
        np.testing.assert_allclose(p.q_values((2,2,0), [(3,2)], 40, 5), 0)

    def test_known_goal_discount_and_horizon(self):
        p = self.planner
        # Two forward moves, goal reward at elapsed time 2.
        self.assertAlmostEqual(p.q_values((1,2,0), [(3,2)], 0, 2)[2], .9 * (1 - .9 * 2 / 40))
        np.testing.assert_allclose(p.q_values((1,2,0), [(3,2)], 0, 1), 0)
        self.assertEqual(maximizers(p.q_values((1,2,0), [(3,2)], 0, 2)), (2,))

    def test_expected_reward_and_noop_equivalence(self):
        p = self.planner
        q = p.q_values((2,2,0), [(3,2),(1,1)], 0, 1)
        self.assertAlmostEqual(q[2], .5 * (1 - .9 / 40))
        np.testing.assert_array_equal(q[3:], [q[3]] * 4)

    def test_matches_independent_simulator_belief_tree(self):
        p = self.planner
        pose, goals = (2,2,0), [(1,1),(1,2)]
        envs = []
        for goal in goals:
            env = make_forage_env(size=5, agent_view_size=3, num_distractors=0, max_steps=40)
            env.reset(seed=0)
            env.grid = Grid(5,5)
            env.grid.wall_rect(0,0,5,5)
            env.grid.set(*goal, Goal())
            env.agent_pos, env.agent_dir = pose[:2], pose[2]
            env.step_count = 4
            envs.append(env)
        try:
            reference = simulator_q(envs, 4, .9)
            self.assertGreater(reference.max(), 0)
            self.assertGreater(np.ptp(reference), 0)
            np.testing.assert_allclose(p.q_values(pose, goals, 4, 4), reference, atol=1e-12)
        finally:
            for env in envs:
                env.close()

    def test_longer_horizon_cannot_reduce_expected_optimal_value(self):
        p = self.planner
        values = [p.q_values((2,2,0), [(1,1),(1,3)], 0, h).max() for h in range(1,6)]
        self.assertTrue(np.all(np.diff(values) >= -1e-12))


if __name__ == '__main__':
    unittest.main()
