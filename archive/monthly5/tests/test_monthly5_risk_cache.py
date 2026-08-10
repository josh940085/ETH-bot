import unittest

import numpy as np
import pandas as pd

import monthly5_risk_cache
import monthly5_selector_cache


class Monthly5RiskCacheTests(unittest.TestCase):
    def _frame(self, rows):
        return pd.DataFrame(
            rows,
            columns=("open", "high", "low", "close"),
            index=pd.date_range("2026-01-01T00:05:00Z", periods=len(rows), freq="5min"),
        )

    def test_short_flat_signal_never_goes_long(self):
        close = np.linspace(120.0, 80.0, 900)
        frame = pd.DataFrame(
            {"open": close, "high": close + 0.1, "low": close - 0.1, "close": close},
            index=pd.date_range("2026-01-01", periods=len(close), freq="5min", tz="UTC"),
        )
        signal = monthly5_selector_cache.build_signal(frame, "mom480_sf")
        self.assertTrue((signal <= 0.0).all())
        self.assertTrue((signal.iloc[482:] == -1.0).all())

    def test_same_bar_stop_wins_over_target(self):
        frame = self._frame(
            [
                (100.0, 100.0, 100.0, 100.0),
                (100.0, 103.0, 97.0, 102.0),
            ]
        )
        pnl, turnover, actual = monthly5_risk_cache.simulate_trade_risk_path(
            frame,
            np.array([0.0, 1.0]),
            stop_pct=0.02,
            target_pct=0.02,
            cooldown_bars=12,
        )
        self.assertAlmostEqual(pnl[1], -0.02)
        self.assertEqual(turnover[1], 2.0)
        self.assertEqual(actual[1], 1.0)

    def test_stop_exit_enters_cooldown(self):
        frame = self._frame(
            [
                (100.0, 100.0, 100.0, 100.0),
                (100.0, 100.0, 97.0, 98.0),
                (98.0, 100.0, 97.0, 99.0),
                (99.0, 101.0, 98.0, 100.0),
            ]
        )
        _, turnover, actual = monthly5_risk_cache.simulate_trade_risk_path(
            frame,
            np.ones(4),
            stop_pct=0.02,
            target_pct=None,
            cooldown_bars=2,
        )
        self.assertEqual(turnover[1], 1.0)
        self.assertEqual(actual[2], 0.0)
        self.assertEqual(actual[3], 0.0)

    def test_gap_through_long_stop_fills_at_worse_open(self):
        frame = self._frame(
            [
                (100.0, 100.0, 100.0, 100.0),
                (95.0, 96.0, 94.0, 95.0),
            ]
        )
        pnl, _, _ = monthly5_risk_cache.simulate_trade_risk_path(
            frame,
            np.array([0.0, 1.0]),
            stop_pct=0.02,
            target_pct=None,
            cooldown_bars=12,
        )
        self.assertAlmostEqual(pnl[1], -0.05)


if __name__ == "__main__":
    unittest.main()
