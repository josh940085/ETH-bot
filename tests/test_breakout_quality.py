import os
import unittest
from unittest import mock

import pandas as pd

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import eth


def _frame(rows):
    return pd.DataFrame(rows, columns=["high", "low", "close"])


class BreakoutQualityTests(unittest.TestCase):
    def test_repeated_tests_do_not_double_count_same_5m_and_15m_event(self):
        rows = [(101.0, 99.9, 100.1)] * 4
        result = eth.analyze_repeated_level_tests(
            100.0,
            _frame(rows),
            _frame(rows),
            support=100.0,
            resistance=110.0,
            atr=0.1,
        )
        self.assertEqual(result["support_5m_tests"], 4)
        self.assertEqual(result["support_15m_tests"], 4)
        self.assertEqual(result["support_tests"], 4)

    def test_repeated_pressure_uses_structural_level_for_attempt_and_close(self):
        df_5m = _frame(
            [
                (100.0, 98.0, 99.0),
                (100.2, 98.5, 99.5),
                (100.4, 99.0, 100.0),
                (100.6, 99.5, 100.2),
                (106.0, 104.5, 105.8),
                (106.2, 105.0, 105.9),
                (106.3, 105.2, 106.0),
            ]
        )
        result = eth._build_breakout_reference(
            price=106.0,
            df_5m=df_5m,
            atr=0.5,
            recent_support=95.0,
            recent_resistance=105.0,
            repeated_support_tests=0,
            repeated_resistance_tests=2,
            sr_analysis={},
        )
        self.assertEqual(result["attempt"], 1)
        self.assertEqual(result["resistance_level"], 105.0)
        self.assertTrue(result["close_confirmed"])

    def test_range_breakout_can_qualify_from_close_without_extreme_volume(self):
        reference = {
            "attempt": 1,
            "close_confirmed": True,
            "sweep_high": False,
            "sweep_low": False,
        }
        result = eth._score_breakout_quality(
            reference,
            regime="range",
            htf=-1,
            mid_trend=-1,
            volume_ratio=1.0,
            buy_pressure=True,
            sell_pressure=False,
            taker_buy_ratio=0.50,
            macro_bias=0.0,
            derivatives_pressure=0.0,
        )
        self.assertEqual(result["required_score"], 4.0)
        self.assertFalse(result["confirmed"])

        result = eth._score_breakout_quality(
            reference,
            regime="range",
            htf=1,
            mid_trend=-1,
            volume_ratio=1.2,
            buy_pressure=True,
            sell_pressure=False,
            taker_buy_ratio=0.50,
            macro_bias=0.0,
            derivatives_pressure=0.0,
        )
        self.assertTrue(result["confirmed"])

    def test_strong_live_confluence_can_start_before_candle_close(self):
        reference = {
            "attempt": -1,
            "close_confirmed": False,
            "sweep_high": False,
            "sweep_low": False,
        }
        result = eth._score_breakout_quality(
            reference,
            regime="bear_trend",
            htf=-1,
            mid_trend=-1,
            volume_ratio=1.25,
            buy_pressure=False,
            sell_pressure=True,
            taker_buy_ratio=0.45,
            macro_bias=-0.3,
            derivatives_pressure=-0.2,
        )
        self.assertFalse(reference["close_confirmed"])
        self.assertTrue(result["confirmed"])

    def test_breakout_retest_uses_actual_level_and_then_reclaims(self):
        pending = {
            "direction": "long",
            "price": 100.30,
            "breakout_level": 100.0,
            "score": 0.7,
            "ts": 100.0,
            "require_pullback": True,
        }
        env = {
            "TRADE_ENTRY_CONFIRM_MIN_WAIT_SEC": "15",
            "TRADE_ENTRY_CONFIRM_PULLBACK_MIN_RATE": "0.0006",
            "TRADE_ENTRY_CONFIRM_RECLAIM_TOLERANCE_RATE": "0.0002",
        }
        with mock.patch.dict(os.environ, env):
            confirmed, message = eth._evaluate_pending_entry_confirmation(
                pending,
                "long",
                100.05,
                0.7,
                "candle",
                120.0,
                require_pullback=True,
            )
            self.assertFalse(confirmed)
            self.assertIn("100.00", message)
            confirmed, message = eth._evaluate_pending_entry_confirmation(
                pending,
                "long",
                100.08,
                0.7,
                "candle",
                122.0,
                require_pullback=True,
            )
        self.assertTrue(confirmed, message)


if __name__ == "__main__":
    unittest.main()
