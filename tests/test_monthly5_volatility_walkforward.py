import unittest

import numpy as np
import pandas as pd

import monthly5_risk_profile_walkforward as risk_walkforward


class Monthly5VolatilityWalkforwardTests(unittest.TestCase):
    def test_custom_profile_names_are_preserved(self):
        index = pd.date_range("2025-01-01", periods=3, freq="MS", tz="UTC")
        frame = pd.DataFrame(index=index)
        profiles = ({"name": "first"}, {"name": "second"})
        _, selections = risk_walkforward.select_monthly_profiles(
            frame,
            [np.array([1.10, 1.10, 1.0]), np.array([0.90, 0.90, 1.0])],
            lookback_months=2,
            score_mode="balanced",
            profiles=profiles,
        )
        self.assertEqual(selections[-1]["profile"], "first")


if __name__ == "__main__":
    unittest.main()
