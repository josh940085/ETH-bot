import unittest

import pandas as pd

import monthly5_regime_hysteresis_research as hysteresis


class Monthly5RegimeHysteresisResearchTests(unittest.TestCase):
    def test_opposite_regime_requires_confirmation_before_flip(self):
        index = pd.date_range("2026-01-01T00:05:00Z", periods=5, freq="5min")
        frame = pd.DataFrame({"close": [100.0] * 5}, index=index)
        labels = pd.Series(
            ["up", "up", "down", "down"],
            index=index[[0, 1, 2, 3]],
        )
        position, _ = hysteresis.build_hysteresis_position(
            frame,
            labels,
            confirmation_bars=2,
            range_grace_bars=0,
        )
        self.assertEqual(position.tolist(), [0.0, 1.0, 1.0, -1.0, -1.0])

    def test_short_range_keeps_trend_until_grace_expires(self):
        index = pd.date_range("2026-01-01T00:05:00Z", periods=4, freq="5min")
        frame = pd.DataFrame({"close": [100.0] * 4}, index=index)
        labels = pd.Series(["up", "range"], index=index[[0, 1]])
        position, _ = hysteresis.build_hysteresis_position(
            frame,
            labels,
            confirmation_bars=1,
            range_grace_bars=2,
        )
        self.assertEqual(position.tolist(), [1.0, 1.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
