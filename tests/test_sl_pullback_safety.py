import os
import unittest
from unittest.mock import patch

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import eth


class StopLossAndPullbackSafetyTests(unittest.TestCase):
    def setUp(self):
        self.panel_state = dict(eth.POSITION_PANEL_STATE)

    def tearDown(self):
        eth.POSITION_PANEL_STATE.clear()
        eth.POSITION_PANEL_STATE.update(self.panel_state)

    @patch("eth.time.time", side_effect=[1000.0, 1040.0])
    def test_partial_stop_fills_are_one_loss(self, _time):
        eth.POSITION_PANEL_STATE["close_hits"] = []

        self.assertTrue(eth.record_position_close("SL", 1930.76))
        self.assertFalse(eth.record_position_close("SL", 1930.28))

        self.assertEqual(
            eth.POSITION_PANEL_STATE["close_hits"],
            [{"reason": "SL", "price": 1930.28, "ts": 1040}],
        )
        self.assertEqual(eth._recent_tp_sl_stats(5), {"total": 1, "tp": 0, "sl": 1})

    def test_existing_duplicate_stop_hits_are_collapsed(self):
        eth.POSITION_PANEL_STATE["close_hits"] = [
            {"reason": "SL", "price": 1930.28, "ts": 1040},
            {"reason": "SL", "price": 1930.76, "ts": 1000},
            {"reason": "TP", "price": 1960.0, "ts": 500},
        ]

        self.assertEqual(eth._recent_tp_sl_stats(5), {"total": 2, "tp": 1, "sl": 1})

    def test_duplicate_sl_followup_lessons_are_collapsed(self):
        reviews = [
            {"direction": "long", "close_price": 1930.76, "close_ts": 1000.0},
            {"direction": "long", "close_price": 1930.28, "close_ts": 1040.0},
            {"direction": "short", "close_price": 1940.0, "close_ts": 1050.0},
        ]

        deduped = eth._dedupe_sl_followup_reviews(reviews, window_sec=300)

        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["close_price"], 1930.28)
        self.assertEqual(deduped[1]["direction"], "short")

    def test_reclaim_signal_requires_pullback(self):
        self.assertTrue(
            eth._entry_confirmation_requires_pullback(
                "long",
                {
                    "final": "📈 多週期趨勢續強做多",
                    "multitimeframe_bull_reclaim": {"applied": True},
                },
            )
        )
        self.assertFalse(
            eth._entry_confirmation_requires_pullback(
                "long",
                {"final": "🚀 做多", "multitimeframe_bull_reclaim": {"applied": False}},
            )
        )
        self.assertTrue(
            eth._entry_confirmation_requires_pullback(
                "short",
                {"final": "🚀 做空", "breakout": -1, "regime": "bear_trend"},
            )
        )

    def test_breakout_waits_for_pullback_and_reclaim(self):
        pending = {
            "direction": "long",
            "price": 2000.0,
            "score": 0.75,
            "ts": 1000.0,
            "candle_id": 1,
            "require_pullback": True,
        }
        env = {
            "TRADE_ENTRY_CONFIRM_MIN_WAIT_SEC": "15",
            "TRADE_ENTRY_CONFIRM_PULLBACK_MIN_RATE": "0.0006",
            "TRADE_ENTRY_CONFIRM_RECLAIM_TOLERANCE_RATE": "0.0002",
        }
        with patch.dict(eth.os.environ, env, clear=False):
            confirmed, reason = eth._evaluate_pending_entry_confirmation(
                pending, "long", 2000.2, 0.75, 1, 1020.0, require_pullback=True
            )
            self.assertFalse(confirmed)
            self.assertIn("等待多單回踩", reason)

            confirmed, reason = eth._evaluate_pending_entry_confirmation(
                pending, "long", 1998.7, 0.75, 1, 1030.0, require_pullback=True
            )
            self.assertFalse(confirmed)
            self.assertTrue(pending["pullback_seen"])
            self.assertIn("重新站回", reason)

            confirmed, reason = eth._evaluate_pending_entry_confirmation(
                pending, "long", 1999.7, 0.75, 1, 1040.0, require_pullback=True
            )
            self.assertTrue(confirmed)
            self.assertIn("延遲確認通過", reason)

    def test_ordinary_signal_accepts_shorter_confirmation_and_moderate_move(self):
        pending = {
            "direction": "long",
            "price": 2000.0,
            "score": 0.70,
            "ts": 1000.0,
            "candle_id": 1,
        }
        env = {
            "TRADE_ENTRY_CONFIRM_MIN_WAIT_SEC": "10",
            "TRADE_ENTRY_CONFIRM_MAX_CHASE_RATE": "0.004",
            "TRADE_ENTRY_CONFIRM_MAX_REVERSAL_RATE": "0.0045",
            "TRADE_ENTRY_CONFIRM_REQUIRE_NEW_5M": "0",
        }

        with patch.dict(eth.os.environ, env, clear=False):
            confirmed, reason = eth._evaluate_pending_entry_confirmation(
                pending,
                "long",
                2007.0,
                0.70,
                1,
                1010.0,
            )

        self.assertTrue(confirmed)
        self.assertIn("延遲確認通過 10s", reason)


if __name__ == "__main__":
    unittest.main()
