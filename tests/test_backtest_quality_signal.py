import os
import unittest

import pandas as pd

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import backtest


class BacktestQualitySignalTests(unittest.TestCase):
    def test_inverted_long_signal_opens_short_with_mirrored_protection(self):
        direction, signal, decision = backtest._invert_entry_signal(
            "long",
            "📈 多週期趨勢續強做多",
            2000.0,
            {"tp": 2060.0, "sl": 1970.0},
        )

        self.assertEqual(direction, "short")
        self.assertIn("做空（反向原訊號", signal)
        self.assertEqual(decision["tp"], 1940.0)
        self.assertEqual(decision["sl"], 2030.0)
        self.assertEqual(decision["original_signal_direction"], "long")

    def test_inverted_short_signal_opens_long_with_mirrored_protection(self):
        direction, signal, decision = backtest._invert_entry_signal(
            "short",
            "📉 多週期趨勢轉弱做空",
            2000.0,
            {"tp": 1920.0, "sl": 2040.0},
        )

        self.assertEqual(direction, "long")
        self.assertIn("做多（反向原訊號", signal)
        self.assertEqual(decision["tp"], 2080.0)
        self.assertEqual(decision["sl"], 1960.0)
        self.assertEqual(decision["original_signal_direction"], "short")

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
        self.assertEqual(trade["entry_tp"], 1950.0)
        self.assertEqual(trade["entry_sl"], 1900.0)


if __name__ == "__main__":
    unittest.main()
