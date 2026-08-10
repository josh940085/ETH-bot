import unittest

import numpy as np
import pandas as pd

import monthly5_risk_profile_walkforward as walkforward


class Monthly5RiskProfileWalkforwardTests(unittest.TestCase):
    def test_profile_switch_closes_and_reopens_position(self):
        index = pd.date_range("2026-01-01T00:05:00Z", periods=3, freq="5min")
        frame = pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0],
                "high": [100.0, 100.0, 100.0],
                "low": [100.0, 100.0, 100.0],
                "close": [100.0, 100.0, 100.0],
            },
            index=index,
        )
        _, turnover, actual = walkforward.simulate_dynamic_risk_path(
            frame,
            np.ones(3),
            np.array([0, 1, 1]),
        )
        self.assertEqual(turnover.tolist(), [1.0, 2.0, 0.0])
        self.assertEqual(actual.tolist(), [1.0, 1.0, 1.0])

    def test_monthly_selector_cannot_see_target_month(self):
        index = pd.date_range("2025-01-01", periods=4, freq="MS", tz="UTC")
        frame = pd.DataFrame(index=index)
        first = np.array([1.10, 1.10, 0.50, 1.0])
        second = np.array([0.90, 0.90, 2.00, 1.0])
        selected, _ = walkforward.select_monthly_profiles(
            frame,
            [first, second],
            lookback_months=2,
            score_mode="balanced",
        )
        self.assertEqual(selected[2], 0)


if __name__ == "__main__":
    unittest.main()
