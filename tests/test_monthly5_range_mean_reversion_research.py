import unittest
from unittest import mock

import numpy as np
import pandas as pd

import monthly5_range_mean_reversion_research as range_research


class Monthly5RangeMeanReversionResearchTests(unittest.TestCase):
    def test_range_signal_enters_extreme_and_exits_near_mean(self):
        frame = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.0] * 5,
                "volume": [1.0] * 5,
            },
            index=pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC"),
        )
        config = {
            "enabled": True,
            "entry_atr": 0.5,
            "exit_atr": 0.2,
        }
        values = pd.Series([-0.6, -0.4, -0.1, 0.7, 0.1], index=frame.index)
        with mock.patch.object(
            range_research, "completed_4h_distance", return_value=values
        ):
            desired = range_research.build_range_desired(
                frame, np.ones(len(frame), dtype="bool"), config
            )
        self.assertEqual(desired.tolist(), [1.0, 1.0, 0.0, -1.0, 0.0])

    def test_range_path_only_fills_flat_primary(self):
        desired, strategy_ids = range_research.combine_paths(
            np.array([1.0, 0.0, -1.0, 0.0]),
            np.array([0, 0, 1, 1]),
            np.array([-1.0, 1.0, 1.0, -1.0]),
        )
        self.assertEqual(desired.tolist(), [1.0, 1.0, -1.0, -1.0])
        self.assertEqual(strategy_ids.tolist(), [0, 10, 1, 10])


if __name__ == "__main__":
    unittest.main()
