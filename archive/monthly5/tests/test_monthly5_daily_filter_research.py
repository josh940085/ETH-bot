import unittest

import monthly5_daily_filter_research


class Monthly5DailyFilterResearchTests(unittest.TestCase):
    def test_filter_grid_has_unfiltered_control(self):
        self.assertTrue(
            any(row["name"] == "none" for row in monthly5_daily_filter_research.FILTER_CONFIGS)
        )

    def test_selection_uses_development_only(self):
        candidates = [
            {
                "name": "development",
                "development": {
                    "months_ge_5": 2,
                    "months_ge_0": 2,
                    "min_month_pct": -1.0,
                    "max_drawdown_pct": -2.0,
                    "avg_month_pct": 1.0,
                },
                "holdout_diagnostic": {"months_ge_5": 0},
            },
            {
                "name": "holdout",
                "development": {
                    "months_ge_5": 1,
                    "months_ge_0": 2,
                    "min_month_pct": -1.0,
                    "max_drawdown_pct": -2.0,
                    "avg_month_pct": 1.0,
                },
                "holdout_diagnostic": {"months_ge_5": 100},
            },
        ]
        self.assertEqual(
            monthly5_daily_filter_research.select_development_winner(candidates)["name"],
            "development",
        )


if __name__ == "__main__":
    unittest.main()
