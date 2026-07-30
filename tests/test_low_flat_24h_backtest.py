import datetime as dt
import unittest

import pandas as pd

import backtest


def _frame(rows):
    index = pd.date_range("2026-01-01", periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame(rows, index=index)


class LowFlat24hBacktestTests(unittest.TestCase):
    def test_candidate_sets_low_flat_mode_and_24h_max_hold(self):
        frame_5m = _frame(
            [
                {"open": 100.0 + i * 0.1, "high": 100.3 + i * 0.1, "low": 99.8 + i * 0.1, "close": 100.1 + i * 0.1, "volume": 1.0}
                for i in range(80)
            ]
        )
        frame_15m = frame_5m.resample("15min", label="right", closed="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        decision = {
            "final": "觀望",
            "score": 0.51,
            "atr": 0.5,
            "rsi_15m": 50.0,
            "host_opening_logic": {"reasons": ["baseline_wait"]},
        }

        result = backtest._build_low_flat_24h_candidate(
            decision,
            {"5m": frame_5m, "15m": frame_15m},
            float(frame_5m["close"].iloc[-1]),
            frame_5m.index[-1],
            0.0,
        )

        self.assertIsNotNone(result)
        final, _score, candidate = result
        self.assertFalse(final.startswith("觀望"))
        self.assertEqual(candidate["max_hold_sec"], 24 * 3600.0)
        self.assertEqual(candidate["primary_indicator"], "low_flat_24h")
        self.assertEqual(candidate["host_opening_logic"]["mode"], "low_flat_24h")

    def test_time_exposure_reports_trades_over_24h(self):
        start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
        end = dt.datetime(2026, 1, 3, tzinfo=dt.timezone.utc)
        trades = [
            {
                "opened_at": "2026-01-01T00:00:00+00:00",
                "closed_at": "2026-01-01T12:00:00+00:00",
            },
            {
                "opened_at": "2026-01-01T13:00:00+00:00",
                "closed_at": "2026-01-02T14:00:01+00:00",
            },
        ]

        exposure = backtest._summarize_time_exposure(trades, start, end)

        self.assertEqual(exposure["trades_over_24h"], 1)
        self.assertGreater(exposure["holding_time_pct"], 0.0)
        self.assertLess(exposure["flat_time_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
