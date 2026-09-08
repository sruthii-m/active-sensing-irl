"""Deterministic checks for pre-registered active-sensing analysis."""

from math import isclose
import unittest

from perception_irl.analysis.metrics import (
    fixation_time_goal_vs_distractors,
    path_length_vs_shortest_path,
    visited_state_entropy,
)
from perception_irl.analysis.entropy_score_baseline import (
    categorical_entropy,
    entropy_score,
    select_entropy_action,
)
from perception_irl.analysis.training_curves import _moving_average, _series_points


class MetricsTest(unittest.TestCase):
    def test_visited_state_entropy_handles_revisits_and_empty_paths(self):
        self.assertEqual(visited_state_entropy([]), 0.0)
        self.assertTrue(isclose(visited_state_entropy([(1, 1), (1, 1), (2, 1), (2, 1)]), 1.0))

    def test_path_length_ratio_counts_executed_transitions(self):
        self.assertEqual(path_length_vs_shortest_path({"positions": [(1, 1), (1, 1), (2, 1)]}, 2), 1.0)
        self.assertEqual(path_length_vs_shortest_path([(1, 1)], 0), 1.0)

    def test_fixation_summary_separates_goal_and_distractors(self):
        result = fixation_time_goal_vs_distractors(
            [(1, 1), (2, 1), (3, 1), (2, 1)],
            goal_region=(3, 1, 3, 1),
            distractor_regions={"red_ball": {(2, 1)}},
        )
        self.assertEqual(result["goal"]["time_to_first_fixation"], 2)
        self.assertEqual(result["goal"]["dwell_steps"], 1)
        self.assertEqual(result["distractors"]["red_ball"]["dwell_steps"], 2)
        self.assertEqual(result["any_distractor"]["time_to_first_fixation"], 1)

    def test_entropy_baseline_prefers_the_action_that_resolves_uncertainty(self):
        belief = {"goal_left": 0.5, "goal_right": 0.5}

        def update(_, action):
            if action == "look":
                return [(0.5, {"goal_left": 1.0}), (0.5, {"goal_right": 1.0})]
            return [(1.0, belief)]

        self.assertTrue(isclose(categorical_entropy(belief), 1.0))
        self.assertTrue(isclose(entropy_score(belief, "look", update), 1.0))
        self.assertTrue(isclose(entropy_score(belief, "move", update), 0.0))
        action, scores = select_entropy_action(belief, ["move", "look"], update)
        self.assertEqual(action, "look")
        self.assertGreater(scores["look"], scores["move"])

    def test_training_curve_formats_and_smoothing_are_normalized(self):
        self.assertEqual(_series_points([(10, 0.1), (20, 0.3)], None), ([10.0, 20.0], [0.1, 0.3]))
        self.assertEqual(_series_points({"steps": [10, 20], "values": [1, 3]}, None), ([10.0, 20.0], [1.0, 3.0]))
        self.assertEqual(_moving_average([1.0, 3.0, 5.0], 2), [1.0, 2.0, 4.0])


if __name__ == "__main__":
    unittest.main()
