import unittest

import numpy as np
import pandas as pd

import monthly5_strong_trend_leverage_research as strong


class Monthly5StrongTrendLeverageResearchTests(unittest.TestCase):
    def test_only_directionally_strong_bars_get_strong_strategy_id(self):
        frame = pd.DataFrame(
            {
                "open": [100.0] * 4,
                "high": [101.0] * 4,
                "low": [99.0] * 4,
                "close": [100.0] * 4,
                "volume": [1.0] * 4,
            },
            index=pd.date_range("2024-01-01", periods=4, freq="5min", tz="UTC"),
        )
        strength = pd.DataFrame(
            {
                "distance_atr": [1.2, -1.2, 1.2, -1.2],
                "slope_atr": [0.2, -0.2, -0.2, 0.2],
            },
            index=frame.index,
        )
        ids, mask = strong.build_strategy_ids(
            frame,
            np.array([1.0, -1.0, 1.0, -1.0]),
            np.array([0, 1, 0, 1], dtype="int32"),
            {
                "enabled": True,
                "distance_atr": 1.0,
                "slope_atr": 0.1,
                "strong_leverage": 2.0,
            },
            strength,
        )
        self.assertEqual(mask.tolist(), [True, True, False, False])
        self.assertEqual(ids.tolist(), [20, 21, 0, 1])

    def test_risk_profiles_never_exceed_requested_leverage(self):
        profiles = strong.risk_profiles(
            {
                "enabled": True,
                "distance_atr": 1.0,
                "slope_atr": 0.1,
                "strong_leverage": 2.0,
            }
        )
        self.assertEqual({row["leverage"] for row in profiles.values()}, {2.0})


if __name__ == "__main__":
    unittest.main()
