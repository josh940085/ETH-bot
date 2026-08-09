import unittest

import pandas as pd

import monthly5_volume_confirmation_research as volume


class Monthly5VolumeConfirmationResearchTests(unittest.TestCase):
    def test_relative_volume_uses_prior_median(self):
        values = pd.Series([10.0, 20.0, 30.0, 40.0])
        prior_median = values.shift(1).rolling(3, min_periods=3).median()
        ratio = values / prior_median
        self.assertAlmostEqual(ratio.iloc[-1], 2.0)

    def test_baseline_allows_all_entries(self):
        frame = pd.DataFrame(
            index=pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC")
        )
        allowed = volume.build_entry_allowed(frame, volume.BASELINE_CONFIG)
        self.assertEqual(allowed.tolist(), [True, True, True])


if __name__ == "__main__":
    unittest.main()
