import unittest

import numpy as np

import monthly5_walkforward_filter


class Monthly5WalkforwardFilterTests(unittest.TestCase):
    def test_candidate_score_does_not_receive_target_month(self):
        history = np.array([[0.06, 0.06], [-0.01, -0.01]])
        scores = monthly5_walkforward_filter.candidate_scores(history, "balanced")
        self.assertGreater(scores[0], scores[1])

    def test_validation_selection_ignores_holdout(self):
        candidates = [
            {
                "name": "validation",
                "validation": {
                    "months_ge_5": 2,
                    "months_ge_0": 2,
                    "min_month_pct": -1.0,
                    "max_drawdown_pct": -2.0,
                    "avg_month_pct": 1.0,
                },
                "training": {
                    "months_ge_5": 1,
                    "months_ge_0": 1,
                    "min_month_pct": -1.0,
                    "max_drawdown_pct": -2.0,
                    "avg_month_pct": 1.0,
                },
                "holdout_diagnostic": {"months_ge_5": 0},
            },
            {
                "name": "holdout",
                "validation": {
                    "months_ge_5": 1,
                    "months_ge_0": 2,
                    "min_month_pct": -1.0,
                    "max_drawdown_pct": -2.0,
                    "avg_month_pct": 1.0,
                },
                "training": {
                    "months_ge_5": 100,
                    "months_ge_0": 100,
                    "min_month_pct": 10.0,
                    "max_drawdown_pct": 0.0,
                    "avg_month_pct": 10.0,
                },
                "holdout_diagnostic": {"months_ge_5": 100},
            },
        ]
        winner = monthly5_walkforward_filter.select_validation_winner(candidates)
        self.assertEqual(winner["name"], "validation")


if __name__ == "__main__":
    unittest.main()
