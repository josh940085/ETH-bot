import unittest

import numpy as np
import pandas as pd

import monthly5_entry_quality_research as quality
import monthly5_intramonth_recovery_research as account


class Monthly5EntryQualityResearchTests(unittest.TestCase):
    def test_quality_gate_blocks_overheated_long_and_short(self):
        index = pd.date_range("2026-01-01T00:05:00Z", periods=2, freq="5min")
        frame = pd.DataFrame({"close": [100.0, 100.0]}, index=index)
        features = pd.DataFrame(
            {"rsi": [80.0, 20.0], "distance_atr": [3.0, -3.0]},
            index=index,
        )
        allowed = quality.build_entry_allowed(
            frame,
            np.array([1.0, -1.0]),
            features,
            max_rsi=70.0,
            max_extension_atr=2.0,
        )
        self.assertEqual(allowed.tolist(), [False, False])

    def test_gate_does_not_close_existing_position(self):
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
        _, _, _, _, positions = account.simulate_account_path(
            frame,
            np.ones(3),
            np.zeros(3, dtype="int32"),
            np.ones(3),
            quality.BASELINE_CONFIG,
            entry_allowed=np.array([True, False, False]),
        )
        self.assertEqual(positions.tolist(), [1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
