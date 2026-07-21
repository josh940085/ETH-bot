import os
import unittest
from unittest.mock import patch

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import eth


class StrategyExecutionSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.panel_state = dict(eth.POSITION_PANEL_STATE)
        self.active_trade = dict(eth.active_trade)

    def tearDown(self):
        eth.POSITION_PANEL_STATE.clear()
        eth.POSITION_PANEL_STATE.update(self.panel_state)
        eth.active_trade.clear()
        eth.active_trade.update(self.active_trade)

    def test_validated_mark_price_is_used_for_strategy(self):
        price = eth._validated_strategy_mark_price({"price": 1871.80})
        self.assertEqual(price, 1871.80)

    def test_strategy_rejects_invalid_mark_price(self):
        with self.assertRaisesRegex(RuntimeError, "標記價格無效"):
            eth._validated_strategy_mark_price({"price": 0.0})

    def test_multitimeframe_bull_reclaim_allows_sustained_breakout(self):
        result = eth._assess_multitimeframe_bull_reclaim(
            price=1928.52,
            higher_timeframe={
                "fifteen_min_trend": 1,
                "one_hour_trend": 1,
                "four_hour_trend": 1,
                "fifteen_min_window_change_pct": 4.0309,
                "one_hour_window_change_pct": 8.0873,
                "four_hour_window_change_pct": 10.9183,
                "daily_trend": -1,
                "weekly_trend": -1,
                "fifteen_min_resistance": 1925.04,
                "one_hour_resistance": 1924.77,
            },
            market_profile={"phase": "range_base"},
            regime="bull_trend",
            htf=1,
            mid_trend=1,
            breakout=0,
            sweep_high=False,
            macro_bias=1.5,
            derivatives_pressure=0.0,
            event_risk=0,
            rsi_15m=69.09,
            ema50_deviation_15m=0.00977,
            ai_long_prob=0.6526,
            candlestick_turn_direction="neutral",
            candlestick_turn_count=1,
            candlestick_turn_confidence=0.0,
        )

        self.assertTrue(result["applied"])
        self.assertEqual(result["reclaim_level"], 1925.04)
        self.assertLessEqual(result["max_position_size"], 0.05)

    def test_multitimeframe_bull_reclaim_allows_live_confirmed_breakout(self):
        result = eth._assess_multitimeframe_bull_reclaim(
            price=1936.0,
            higher_timeframe={
                "fifteen_min_trend": 1,
                "one_hour_trend": 1,
                "four_hour_trend": 1,
                "fifteen_min_window_change_pct": 3.9611,
                "one_hour_window_change_pct": 8.6726,
                "four_hour_window_change_pct": 10.9183,
                "daily_trend": -1,
                "weekly_trend": -1,
                "fifteen_min_resistance": 1928.816,
                "one_hour_resistance": 1924.9924,
            },
            market_profile={"phase": "range_base"},
            regime="bull_trend",
            htf=1,
            mid_trend=1,
            breakout=1,
            sweep_high=False,
            macro_bias=1.5,
            derivatives_pressure=0.0,
            event_risk=0,
            rsi_15m=72.7,
            ema50_deviation_15m=0.00997,
            ai_long_prob=0.8301,
            candlestick_turn_direction="neutral",
            candlestick_turn_count=1,
            candlestick_turn_confidence=0.0,
        )

        self.assertTrue(result["applied"])
        self.assertEqual(result["min_change_pct"]["15m"], 3.5)

    def test_multitimeframe_bull_reclaim_keeps_stricter_15m_move_without_breakout(self):
        result = eth._assess_multitimeframe_bull_reclaim(
            price=1936.0,
            higher_timeframe={
                "fifteen_min_trend": 1,
                "one_hour_trend": 1,
                "four_hour_trend": 1,
                "fifteen_min_window_change_pct": 3.9611,
                "one_hour_window_change_pct": 8.6726,
                "four_hour_window_change_pct": 10.9183,
                "daily_trend": -1,
                "weekly_trend": -1,
                "fifteen_min_resistance": 1928.816,
                "one_hour_resistance": 1924.9924,
            },
            market_profile={"phase": "range_base"},
            regime="bull_trend",
            htf=1,
            mid_trend=1,
            breakout=0,
            sweep_high=False,
            macro_bias=1.5,
            derivatives_pressure=0.0,
            event_risk=0,
            rsi_15m=72.7,
            ema50_deviation_15m=0.00997,
            ai_long_prob=0.8301,
            candlestick_turn_direction="neutral",
            candlestick_turn_count=1,
            candlestick_turn_confidence=0.0,
        )

        self.assertFalse(result["applied"])

    def test_multitimeframe_bull_reclaim_allows_sustained_high_rsi_reclaim(self):
        result = eth._assess_multitimeframe_bull_reclaim(
            price=1938.5,
            higher_timeframe={
                "fifteen_min_trend": 1,
                "one_hour_trend": 1,
                "four_hour_trend": 1,
                "fifteen_min_window_change_pct": 4.0986,
                "one_hour_window_change_pct": 8.6726,
                "four_hour_window_change_pct": 11.7373,
                "daily_trend": -1,
                "weekly_trend": -1,
                "fifteen_min_resistance": 1931.522,
                "one_hour_resistance": 1924.9924,
            },
            market_profile={"phase": "range_base"},
            regime="bull_trend_strong",
            htf=1,
            mid_trend=1,
            breakout=0,
            sweep_high=False,
            macro_bias=1.5,
            derivatives_pressure=0.0,
            event_risk=0,
            rsi_15m=73.87,
            ema50_deviation_15m=0.01077,
            ai_long_prob=0.8301,
            candlestick_turn_direction="neutral",
            candlestick_turn_count=1,
            candlestick_turn_confidence=0.0,
        )

        self.assertTrue(result["applied"])

    def test_multitimeframe_bull_reclaim_still_blocks_unreclaimed_pressure(self):
        result = eth._assess_multitimeframe_bull_reclaim(
            price=1924.90,
            higher_timeframe={
                "fifteen_min_trend": 1,
                "one_hour_trend": 1,
                "four_hour_trend": 1,
                "fifteen_min_window_change_pct": 4.0309,
                "one_hour_window_change_pct": 8.0873,
                "four_hour_window_change_pct": 10.9183,
                "daily_trend": -1,
                "weekly_trend": -1,
                "fifteen_min_resistance": 1925.04,
                "one_hour_resistance": 1924.77,
            },
            market_profile={"phase": "range_base"},
            regime="bull_trend",
            htf=1,
            mid_trend=1,
            breakout=0,
            sweep_high=False,
            macro_bias=1.5,
            derivatives_pressure=0.0,
            event_risk=0,
            rsi_15m=69.09,
            ema50_deviation_15m=0.00977,
            ai_long_prob=0.6526,
            candlestick_turn_direction="neutral",
            candlestick_turn_count=1,
            candlestick_turn_confidence=0.0,
        )

        self.assertFalse(result["applied"])

    def test_multitimeframe_bull_reclaim_is_not_used_in_broad_bull_market(self):
        result = eth._assess_multitimeframe_bull_reclaim(
            price=1928.52,
            higher_timeframe={
                "fifteen_min_trend": 1,
                "one_hour_trend": 1,
                "four_hour_trend": 1,
                "daily_trend": 1,
                "weekly_trend": 1,
                "fifteen_min_window_change_pct": 4.0309,
                "one_hour_window_change_pct": 8.0873,
                "four_hour_window_change_pct": 10.9183,
                "fifteen_min_resistance": 1925.04,
                "one_hour_resistance": 1924.77,
            },
            market_profile={"phase": "range_base"},
            regime="bull_trend",
            htf=1,
            mid_trend=1,
            breakout=0,
            sweep_high=False,
            macro_bias=1.5,
            derivatives_pressure=0.0,
            event_risk=0,
            rsi_15m=69.09,
            ema50_deviation_15m=0.00977,
            ai_long_prob=0.6526,
            candlestick_turn_direction="neutral",
            candlestick_turn_count=1,
            candlestick_turn_confidence=0.0,
        )

        self.assertFalse(result["applied"])

    def test_daily_anchor_keeps_bull_reclaim_small_but_does_not_wait(self):
        decision = {
            "market_profile": {"phase": "range_base"},
            "risk_rate": 0.01,
            "net_edge_rate_est": 0.003,
            "position_size": 0.15,
            "multitimeframe_bull_reclaim": {
                "applied": True,
                "max_position_size": 0.05,
            },
        }

        should_wait = eth._daily_anchor_guard_should_wait(
            "📈 多週期趨勢續強做多",
            0.74,
            decision,
        )

        self.assertFalse(should_wait)
        self.assertEqual(decision["position_size"], 0.05)
        self.assertEqual(decision["max_position_size"], 0.05)
        self.assertTrue(decision["daily_anchor_quality_signal"])

    def test_real_order_priority_is_enabled_by_default(self):
        with patch.dict(eth.os.environ, {}, clear=True):
            self.assertTrue(eth._real_order_priority_enabled())

    def test_panel_marks_binance_as_authoritative_for_real_position(self):
        eth.active_trade.update(
            {
                "open": True,
                "direction": "long",
                "entry": 1800.0,
                "avg_entry": 1800.0,
                "tp": 1830.0,
                "sl": 1780.0,
                "size": 0.03,
                "position_qty": 0.01,
                "open_time": 1990.0,
            }
        )
        with (
            patch.object(eth, "_get_follow_mode_enabled", return_value=True),
            patch.object(eth, "_is_real_copy_enabled", return_value=True),
            patch.object(eth, "_refresh_position_panel_account_state"),
            patch.object(eth, "_write_json_atomic"),
            patch.object(eth, "_queue_panel_realtime_publish"),
        ):
            eth.sync_position_panel(1810.0)

        self.assertEqual(eth.POSITION_PANEL_STATE["execution_priority"], "real_order")
        self.assertEqual(eth.POSITION_PANEL_STATE["execution_mode"], "real")
        self.assertEqual(eth.POSITION_PANEL_STATE["position_source"], "binance")

    @patch("eth.time.time", return_value=2000.0)
    def test_only_actual_open_sets_long_or_short_signal(self, _time):
        eth.active_trade["direction"] = "long"
        eth.POSITION_PANEL_STATE["binance_mark_price_ts"] = 1999
        decision = {"final": "做多", "score": 0.7, "ai_prob": 0.72, "regime": "bull_trend"}

        eth._update_panel_execution_snapshot(
            decision, 1871.8, "pending_confirmation", reason="等待確認", actual_open=False
        )
        self.assertEqual(eth.POSITION_PANEL_STATE["strategy_signal"], "wait")
        self.assertFalse(eth.POSITION_PANEL_STATE["strategy_actual_open"])

        eth._update_panel_execution_snapshot(
            decision, 1871.8, "opened", reason="Binance 實際開單成功", actual_open=True
        )
        self.assertEqual(eth.POSITION_PANEL_STATE["strategy_signal"], "long")
        self.assertTrue(eth.POSITION_PANEL_STATE["strategy_actual_open"])
        self.assertEqual(eth.POSITION_PANEL_STATE["strategy_evaluated_ts"], 2000)


if __name__ == "__main__":
    unittest.main()
