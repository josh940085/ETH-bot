import os
import unittest
from unittest.mock import patch

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import eth


class StopLossAndPullbackSafetyTests(unittest.TestCase):
    def setUp(self):
        self.panel_state = dict(eth.POSITION_PANEL_STATE)
        self.active_trade_state = dict(eth.active_trade)

    def tearDown(self):
        eth.POSITION_PANEL_STATE.clear()
        eth.POSITION_PANEL_STATE.update(self.panel_state)
        eth.active_trade.clear()
        eth.active_trade.update(self.active_trade_state)

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

    def test_post_sl_review_rechecks_qualified_opposite_direction(self):
        review = eth._build_post_sl_opposite_review(
            "short",
            {
                "htf": 1,
                "mid_trend": 1,
                "breakout": 1,
                "macro_bias": 1.5,
                "derivatives_pressure": 0.05,
                "event_risk": 0,
                "ai_long_prob": 0.72,
                "host_opening_logic": {
                    "direction": "long",
                    "confidence": 0.80,
                },
            },
        )

        self.assertEqual(review["opposite_direction"], "long")
        self.assertTrue(review["ready_for_fresh_evaluation"])
        self.assertTrue(review["requires_normal_entry_validation"])
        self.assertIn("正常進場與RR/TP/SL確認", review["summary"])

    def test_post_sl_review_does_not_force_unconfirmed_reversal(self):
        review = eth._build_post_sl_opposite_review(
            "short",
            {
                "htf": -1,
                "mid_trend": 1,
                "breakout": 0,
                "macro_bias": 1.5,
                "derivatives_pressure": -0.10,
                "event_risk": 0,
                "ai_long_prob": 0.36,
                "host_opening_logic": {"direction": "neutral", "confidence": 0.30},
            },
        )

        self.assertFalse(review["ready_for_fresh_evaluation"])
        self.assertIn("4H未轉多", review["missing_conditions"])
        self.assertIn("反向突破未確認", review["missing_conditions"])
        self.assertIn("AI做多機率 0.36<0.65", review["missing_conditions"])

    def test_stop_loss_review_persists_opposite_direction_analysis(self):
        review = eth._review_stop_loss_event(
            "short",
            100.0,
            98.0,
            101.0,
            101.0,
            101.1,
            99.9,
            1.0,
            {
                "htf": 1,
                "mid_trend": 1,
                "breakout": 1,
                "macro_bias": 1.0,
                "derivatives_pressure": 0.0,
                "event_risk": 0,
                "ai_long_prob": 0.70,
                "host_opening_logic": {"direction": "long", "confidence": 0.75},
            },
        )

        self.assertEqual(review["opposite_direction_review"]["opposite_direction"], "long")
        self.assertTrue(review["opposite_direction_review"]["ready_for_fresh_evaluation"])

    def test_binance_confirmed_stop_loss_records_review(self):
        eth.active_trade.update(
            {
                "open": True,
                "direction": "long",
                "entry": 100.0,
                "avg_entry": 100.0,
                "tp": 103.0,
                "sl": 99.0,
                "size": 0.2,
                "open_time": 1000.0,
            }
        )
        eth.POSITION_PANEL_STATE.update(
            {
                "binance_mark_price": 99.0,
                "strategy_score": 0.55,
                "strategy_ai_prob": 0.55,
                "strategy_ai_long_prob": 0.52,
                "strategy_ai_short_prob": 0.68,
                "strategy_context": {
                    "htf": -1,
                    "mid_trend": -1,
                    "macro_bias": -0.5,
                    "derivatives_pressure": 0.0,
                    "event_risk": 0,
                    "support_hits": 0,
                    "resistance_hits": 1,
                    "net_edge_rate_est": 0.0,
                    "risk_rate": 0.01,
                },
                "strategy_host_logic": {
                    "direction": "short",
                    "confidence": 0.72,
                },
            }
        )

        with (
            patch.object(eth, "_binance_futures_signed_get", return_value=[]),
            patch.object(eth, "_binance_futures_open_algo_orders", return_value=[]),
            patch.object(eth, "_load_pending_training_sample_state", return_value=None),
            patch.object(eth, "record_sl_review") as record_review,
            patch.object(eth, "sync_position_panel"),
            patch.object(eth, "_send_trade_notification"),
            patch.object(eth.time, "time", return_value=2000.0),
        ):
            ok, _msg = eth.sync_active_trade_from_binance(send_notice=False)

        self.assertTrue(ok)
        self.assertFalse(eth.active_trade["open"])
        review = eth.POSITION_PANEL_STATE["last_sl_review"]
        self.assertEqual(review["direction"], "long")
        self.assertEqual(review["opposite_direction_review"]["opposite_direction"], "short")
        self.assertTrue(review["opposite_direction_review"]["requires_normal_entry_validation"])
        record_review.assert_called_once()

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

    def test_strong_directional_breakout_can_skip_retest(self):
        common = {
            "breakout_confirmed": True,
            "breakout_quality_score": 6.8,
            "breakout_quality_required": 3.0,
        }
        self.assertFalse(
            eth._entry_confirmation_requires_pullback(
                "short",
                {
                    **common,
                    "final": "🚀 做空",
                    "breakout": -1,
                    "regime": "bear_trend",
                    "repeated_support_tests": 7,
                },
            )
        )
        self.assertFalse(
            eth._entry_confirmation_requires_pullback(
                "long",
                {
                    **common,
                    "final": "🚀 做多",
                    "breakout": 1,
                    "regime": "bull_trend",
                    "repeated_resistance_tests": 7,
                },
            )
        )

    def test_ordinary_directional_breakout_still_requires_retest(self):
        self.assertTrue(
            eth._entry_confirmation_requires_pullback(
                "short",
                {
                    "final": "🚀 做空",
                    "breakout": -1,
                    "breakout_confirmed": True,
                    "breakout_quality_score": 4.0,
                    "breakout_quality_required": 3.0,
                    "regime": "bear_trend",
                    "repeated_support_tests": 7,
                },
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
