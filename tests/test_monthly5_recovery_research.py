import unittest

import monthly5_recovery_research


class Monthly5RecoveryResearchTests(unittest.TestCase):
    def test_recovery_grid_keeps_leverage_cap(self):
        self.assertLessEqual(monthly5_recovery_research.BASE_CONFIG["leverage"], 5)
        self.assertTrue(
            all(0.0 <= row["monthly_recovery_scale"] <= 0.5 for row in monthly5_recovery_research.recovery_configs())
        )

    def test_winner_selection_uses_development_only(self):
        candidates = [
            {
                "name": "development_winner",
                "development": {
                    "months_ge_5": 10,
                    "months_ge_0": 11,
                    "min_month_pct": -5.0,
                    "max_drawdown_pct": -10.0,
                    "avg_month_pct": 2.0,
                },
                "holdout": {"months_ge_5": 0},
            },
            {
                "name": "holdout_winner",
                "development": {
                    "months_ge_5": 9,
                    "months_ge_0": 11,
                    "min_month_pct": -5.0,
                    "max_drawdown_pct": -10.0,
                    "avg_month_pct": 2.0,
                },
                "holdout": {"months_ge_5": 100},
            },
        ]
        winner = monthly5_recovery_research.select_development_winner(candidates)
        self.assertEqual(winner["name"], "development_winner")


if __name__ == "__main__":
    unittest.main()
