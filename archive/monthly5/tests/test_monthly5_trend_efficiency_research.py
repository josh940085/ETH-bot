import unittest

import pandas as pd

import monthly5_trend_efficiency_research as efficiency


class Monthly5TrendEfficiencyResearchTests(unittest.TestCase):
    def test_efficiency_distinguishes_directional_and_choppy_paths(self):
        directional = efficiency.efficiency_ratio(
            pd.Series([100.0, 101.0, 102.0, 103.0]), 3
        )
        choppy = efficiency.efficiency_ratio(
            pd.Series([100.0, 101.0, 100.0, 101.0]), 3
        )
        self.assertAlmostEqual(directional.iloc[-1], 1.0)
        self.assertAlmostEqual(choppy.iloc[-1], 1.0 / 3.0)

    def test_baseline_allows_all_entries(self):
        frame = pd.DataFrame(
            index=pd.date_range("2024-01-01", periods=4, freq="5min", tz="UTC")
        )
        allowed = efficiency.build_entry_allowed(frame, efficiency.BASELINE_CONFIG)
        self.assertEqual(allowed.tolist(), [True, True, True, True])


if __name__ == "__main__":
    unittest.main()
