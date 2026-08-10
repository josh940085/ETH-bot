import unittest

import numpy as np
import pandas as pd

import monthly5_volatility_regime_research as volatility


class Monthly5VolatilityRegimeResearchTests(unittest.TestCase):
    def test_atr_classifier_finds_completed_uptrend(self):
        index = pd.date_range("2026-01-01T00:00:00Z", periods=50, freq="4h")
        close = np.linspace(100.0, 150.0, len(index))
        frame = pd.DataFrame(
            {
                "open": close - 0.2,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.ones(len(index)),
            },
            index=index,
        )
        labels = volatility.classify_4h_frame(
            frame,
            {"mode": "atr", "distance_atr": 0.3, "slope_atr": 0.05},
        )
        self.assertEqual(labels.iloc[-1], "up")

    def test_completed_label_is_available_next_five_minute_bar(self):
        index = pd.date_range("2026-01-01T00:00:00Z", periods=50 * 48, freq="5min")
        close = np.linspace(100.0, 150.0, len(index))
        frame = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "volume": np.ones(len(index)),
            },
            index=index,
        )
        labels = volatility.classify_completed_4h(
            frame,
            {"mode": "fixed", "distance": 0.006, "slope": 0.001},
        )
        self.assertTrue(all(timestamp.minute == 5 for timestamp in labels.index))


if __name__ == "__main__":
    unittest.main()
