import os
import unittest

import pandas as pd

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import backtest


class BacktestQualitySignalTests(unittest.TestCase):
    def test_quality_signal_keeps_probe_size_and_disables_scaling(self):
        decision = {
            "position_size": 0.05,
            "daily_anchor_quality_signal": True,
            "max_position_size": 0.05,
            "host_opening_logic": {"mode": "trend_pullback_long", "confidence": 0.8},
            "regime": "bull_trend",
            "candlestick_turn_count": 0,
            "candlestick_turn_confidence": 0.0,
            "candlestick_turning": {"direction": "neutral"},
            "features": {},
            "tp": 1950.0,
            "sl": 1900.0,
        }

        trade = backtest._build_open_trade(
            pd.Timestamp("2026-07-21T06:00:00Z"),
            "long",
            "📈 多週期趨勢續強做多",
            1930.0,
            0.74,
            decision,
        )

        self.assertEqual(trade["size"], 0.05)
        self.assertEqual(trade["max_size"], 0.05)


if __name__ == "__main__":
    unittest.main()
