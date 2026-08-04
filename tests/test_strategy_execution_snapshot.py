import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import eth


class StrategyExecutionSnapshotTests(unittest.TestCase):
    def test_telegram_control_keyboard_opens_external_panel_without_crypto_mini_app(self):
        with patch.object(
            eth,
            "_build_position_panel_external_url",
            return_value="https://example.com/panel?panel_session=v2.test",
        ):
            keyboard = eth._build_control_panel_keyboard(chat_id=123456)

        buttons = [
            button
            for row in keyboard.get("inline_keyboard", [])
            for button in row
        ]
        self.assertTrue(buttons)
        self.assertIn(eth.POSITION_PANEL_BUTTON_TEXT, [button.get("text") for button in buttons])
        panel_button = next(button for button in buttons if button.get("text") == eth.POSITION_PANEL_BUTTON_TEXT)
        self.assertEqual(panel_button["url"], "https://example.com/panel?panel_session=v2.test")
        self.assertIn("toggle_follow", [button.get("callback_data") for button in buttons])
        self.assertIn("manual_close", [button.get("callback_data") for button in buttons])
        self.assertFalse(any("web_app" in button for button in buttons))

    def test_position_panel_button_sends_fresh_private_panel(self):
        with patch.object(eth, "send_control_panel") as send_panel:
            reply = eth.handle_ai_command(
                eth.POSITION_PANEL_BUTTON_TEXT,
                {"chat_id": 123456, "user_id": 123456, "chat_type": "private"},
            )

        self.assertIsNone(reply)
        send_panel.assert_called_once_with(123456)

    def test_control_panel_text_includes_external_position_panel_link(self):
        with patch.object(eth, "_refresh_position_panel_account_state"), patch.object(
            eth, "_build_position_panel_external_url", return_value="https://example.com/panel?panel_session=v2.test"
        ):
            text = eth._build_control_panel_text(chat_id=123456)

        self.assertIn("📊 倉位面板", text)
        self.assertIn("外部 HTTPS 連結", text)
        self.assertIn("https://example.com/panel?panel_session=v2.test", text)

    def test_position_panel_external_url_uses_session_and_realtime_urls(self):
        with patch.object(eth, "_resolve_private_chat_id_for_controls", return_value="123456"), patch.object(
            eth, "_current_panel_realtime_urls",
            return_value=("https://panel.example.com/api/panel/state", "wss://panel.example.com/ws/panel", ""),
        ), patch.object(eth, "_create_panel_session", return_value="v2.session"):
            url = eth._build_position_panel_external_url(chat_id=123456)

        self.assertTrue(url.startswith(eth.MINI_APP_URL))
        self.assertIn("panel_session=v2.session", url)
        self.assertIn("state_url=https%3A%2F%2Fpanel.example.com%2Fapi%2Fpanel%2Fstate", url)
        self.assertIn("ws_url=wss%3A%2F%2Fpanel.example.com%2Fws%2Fpanel", url)

    def test_telegram_sanitize_preserves_panel_url_query_separators(self):
        safe_text, fallback_text = eth._sanitize_telegram_text(
            "https://example.com/?panel_session=v2.test&state_url=https%3A%2F%2Fpanel.example.com"
        )

        self.assertIn("&state_url=", safe_text)
        self.assertNotIn("andstate_url=", safe_text)
        self.assertEqual(safe_text, fallback_text)

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

    def test_fresh_binance_websocket_price_can_replace_unavailable_mark_price(self):
        with (
            patch.object(eth, "WS_PRICE", 1871.80),
            patch.object(eth, "WS_PRICE_TS", 1999.0),
            patch.object(eth.time, "time", return_value=2000.0),
        ):
            payload = eth._binance_ws_strategy_price_payload(
                "ETHUSDT",
                reference_price=1872.0,
            )

        self.assertTrue(payload["validated"])
        self.assertTrue(payload["fallback"])
        self.assertEqual(payload["source"], "binance_agg_trade_fallback")
        self.assertEqual(payload["price"], 1871.80)

    def test_unavailable_panel_uses_fresh_binance_websocket_instead_of_blocking(self):
        with (
            patch.object(eth.HTTP_SESSION, "get", side_effect=RuntimeError("panel unavailable")),
            patch.object(eth, "WS_PRICE", 1871.80),
            patch.object(eth, "WS_PRICE_TS", 1999.0),
            patch.object(eth.time, "time", return_value=2000.0),
        ):
            payload = eth._fetch_strategy_mark_price(
                "ETHUSDT",
                reference_price=1872.0,
            )

        self.assertEqual(payload["source"], "binance_agg_trade_fallback")
        self.assertTrue(payload["validated"])

    def test_unavailable_panel_and_stale_websocket_use_external_kline(self):
        with (
            patch.object(eth.HTTP_SESSION, "get", side_effect=RuntimeError("panel unavailable")),
            patch.object(eth, "WS_PRICE", 1871.80),
            patch.object(eth, "WS_PRICE_TS", 1990.0),
            patch.object(eth.time, "time", return_value=2000.0),
        ):
            payload = eth._fetch_strategy_mark_price(
                "ETHUSDT",
                reference_price=1872.0,
            )

        self.assertEqual(payload["source"], "external_kline_fallback")
        self.assertTrue(payload["validated"])
        self.assertEqual(payload["price"], 1872.0)

    def test_stale_binance_websocket_price_is_rejected(self):
        with (
            patch.object(eth, "WS_PRICE", 1871.80),
            patch.object(eth, "WS_PRICE_TS", 1990.0),
            patch.object(eth.time, "time", return_value=2000.0),
        ):
            with self.assertRaisesRegex(RuntimeError, "過期"):
                eth._binance_ws_strategy_price_payload(
                    "ETHUSDT",
                    reference_price=1872.0,
                )

    def test_divergent_binance_websocket_price_is_rejected(self):
        with (
            patch.object(eth, "WS_PRICE", 1900.0),
            patch.object(eth, "WS_PRICE_TS", 1999.0),
            patch.object(eth.time, "time", return_value=2000.0),
        ):
            with self.assertRaisesRegex(RuntimeError, "價差過大"):
                eth._binance_ws_strategy_price_payload(
                    "ETHUSDT",
                    reference_price=1872.0,
                )

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
            market_profile={"phase": "bear"},
            regime="bull_trend",
            htf=1,
            mid_trend=1,
            breakout=0,
            sweep_high=False,
            macro_bias=1.5,
            derivatives_pressure=-0.1443,
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
        self.assertEqual(result["diagnostics"]["min_derivatives_pressure"], -0.15)

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
            "market_profile": {"phase": "bear"},
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

    def test_historical_quality_guard_requires_structure_for_support_reclaim(self):
        self.assertFalse(
            eth._historical_support_reclaim_quality_ok(
                direction="long",
                breakout_confirmed=False,
                support_hits=1,
            )
        )
        self.assertTrue(
            eth._historical_support_reclaim_quality_ok(
                direction="long",
                breakout_confirmed=False,
                support_hits=2,
            )
        )
        self.assertTrue(
            eth._historical_support_reclaim_quality_ok(
                direction="long",
                breakout_confirmed=True,
                support_hits=0,
            )
        )

    def test_historical_quality_guard_does_not_change_short_entries(self):
        self.assertTrue(
            eth._historical_support_reclaim_quality_ok(
                direction="short",
                breakout_confirmed=False,
                support_hits=0,
            )
        )

    def test_daily_anchor_allows_quality_long_in_bear_profile(self):
        decision = {
            "market_profile": {"phase": "bear"},
            "risk_rate": 0.008,
            "net_edge_rate_est": 0.004,
            "position_size": 0.08,
            "host_opening_logic": {
                "direction": "long",
                "mode": "breakout_after_pressure_tests",
                "confidence": 0.72,
                "range_pos": 0.24,
            },
            "host_logic_applied": True,
            "htf": -1,
            "mid_trend": 1,
            "buy_pressure": True,
            "volume_spike": True,
            "event_risk": 0,
        }

        should_wait = eth._daily_anchor_guard_should_wait(
            "↩️ 反彈做多",
            0.78,
            decision,
        )

        self.assertFalse(should_wait)
        self.assertEqual(decision["position_size"], 0.02)
        self.assertEqual(decision["max_position_size"], 0.02)
        self.assertEqual(decision["general_entry_relaxation"], "bear_tested_breakout_long")

    def test_daily_anchor_still_waits_for_weak_bear_profile_long(self):
        decision = {
            "market_profile": {"phase": "bear"},
            "risk_rate": 0.008,
            "net_edge_rate_est": 0.0004,
            "position_size": 0.02,
            "host_opening_logic": {
                "direction": "long",
                "mode": "breakout_after_pressure_tests",
                "confidence": 0.72,
                "range_pos": 0.50,
            },
            "host_logic_applied": True,
            "htf": -1,
            "mid_trend": -1,
            "event_risk": 0,
        }

        should_wait = eth._daily_anchor_guard_should_wait(
            "🚀 做多",
            0.78,
            decision,
        )

        self.assertTrue(should_wait)

    def test_daily_anchor_keeps_event_risk_block_for_bear_breakout_long(self):
        decision = {
            "market_profile": {"phase": "bear"},
            "risk_rate": 0.02,
            "net_edge_rate_est": 0.045,
            "position_size": 0.02,
            "host_opening_logic": {
                "direction": "long",
                "mode": "breakout_after_pressure_tests",
                "confidence": 0.88,
                "range_pos": 0.72,
            },
            "host_logic_applied": True,
            "event_risk": 1,
            "news_bias": 0,
        }

        should_wait = eth._daily_anchor_guard_should_wait(
            "🚀 做多",
            0.90,
            decision,
        )

        self.assertTrue(should_wait)
        self.assertNotIn("general_entry_relaxation", decision)

    def test_daily_anchor_allows_tested_range_break_with_small_size(self):
        decision = {
            "market_profile": {"phase": "range_base"},
            "risk_rate": 0.011,
            "net_edge_rate_est": 0.021,
            "position_size": 0.15,
            "max_position_size": 0.20,
            "host_opening_logic": {
                "mode": "breakout_after_pressure_tests",
                "confidence": 0.88,
                "range_pos": 0.82,
            },
            "htf": 1,
            "mid_trend": 1,
            "resistance_hits": 2,
            "news_bias": 0,
            "event_risk": 0,
        }

        with patch.dict(
            eth.os.environ,
            {"DAILY_MIN_ANCHOR_RANGE_TESTED_BREAK_ENABLED": "1"},
            clear=False,
        ):
            should_wait = eth._daily_anchor_guard_should_wait(
                "🚀 做多",
                0.8959,
                decision,
            )

        self.assertFalse(should_wait)
        self.assertEqual(decision["position_size"], 0.02)
        self.assertEqual(decision["max_position_size"], 0.02)
        self.assertTrue(decision["daily_anchor_quality_signal"])
        self.assertEqual(decision["general_entry_relaxation"], "range_tested_break")

    def test_daily_anchor_blocks_high_pressure_wait_short_breakout_long(self):
        decision = {
            "market_profile": {"phase": "range_base"},
            "risk_rate": 0.011,
            "net_edge_rate_est": 0.021,
            "position_size": 0.15,
            "max_position_size": 0.20,
            "host_opening_logic": {
                "mode": "breakout_after_pressure_tests",
                "confidence": 0.88,
                "range_pos": 0.82,
            },
            "learned_entry_logic": {
                "direction": "long",
                "long_setup": 4.48,
                "short_setup": 1.00,
                "reasons": [
                    "高位靠近壓力，優先等空方確認",
                    "壓力連續測試後放量突破，偏多",
                ],
            },
            "htf": 1,
            "mid_trend": 1,
            "resistance_hits": 4,
            "repeated_resistance_tests": 4,
            "breakout_quality_score": 6.8,
            "breakout_quality_required": 3.5,
            "news_bias": 0,
            "event_risk": 0,
        }

        with patch.dict(
            eth.os.environ,
            {"DAILY_MIN_ANCHOR_RANGE_TESTED_BREAK_ENABLED": "1"},
            clear=False,
        ):
            should_wait = eth._daily_anchor_guard_should_wait(
                "🚀 做多",
                0.90,
                decision,
            )

        self.assertTrue(should_wait)
        self.assertEqual(
            decision["daily_anchor_block_reason"],
            "high_pressure_wait_short_confirmation",
        )
        self.assertNotIn("general_entry_relaxation", decision)

    def test_daily_anchor_allows_extreme_quality_high_pressure_breakout_long(self):
        decision = {
            "market_profile": {"phase": "range_base"},
            "risk_rate": 0.011,
            "net_edge_rate_est": 0.021,
            "position_size": 0.15,
            "max_position_size": 0.20,
            "host_opening_logic": {
                "mode": "breakout_after_pressure_tests",
                "confidence": 0.88,
                "range_pos": 0.82,
            },
            "learned_entry_logic": {
                "reasons": ["高位靠近壓力，優先等空方確認"],
            },
            "htf": 1,
            "mid_trend": 1,
            "resistance_hits": 4,
            "repeated_resistance_tests": 4,
            "breakout_quality_score": 7.1,
            "breakout_quality_required": 3.5,
            "news_bias": 0,
            "event_risk": 0,
        }

        with patch.dict(
            eth.os.environ,
            {"DAILY_MIN_ANCHOR_RANGE_TESTED_BREAK_ENABLED": "1"},
            clear=False,
        ):
            should_wait = eth._daily_anchor_guard_should_wait(
                "🚀 做多",
                0.90,
                decision,
            )

        self.assertFalse(should_wait)
        self.assertEqual(decision["position_size"], 0.02)
        self.assertEqual(decision["general_entry_relaxation"], "range_tested_break")

    def test_daily_anchor_keeps_event_risk_block_for_tested_range_break(self):
        decision = {
            "market_profile": {"phase": "range_base"},
            "risk_rate": 0.011,
            "net_edge_rate_est": 0.021,
            "position_size": 0.15,
            "max_position_size": 0.20,
            "host_opening_logic": {
                "mode": "breakout_after_pressure_tests",
                "confidence": 0.88,
                "range_pos": 0.82,
            },
            "htf": 1,
            "mid_trend": 1,
            "resistance_hits": 2,
            "news_bias": 0,
            "event_risk": 1,
        }

        with patch.dict(
            eth.os.environ,
            {"DAILY_MIN_ANCHOR_RANGE_TESTED_BREAK_ENABLED": "1"},
            clear=False,
        ):
            should_wait = eth._daily_anchor_guard_should_wait(
                "🚀 做多",
                0.8959,
                decision,
            )

        self.assertTrue(should_wait)
        self.assertNotIn("general_entry_relaxation", decision)

    def test_real_order_priority_is_enabled_by_default(self):
        with patch.dict(eth.os.environ, {}, clear=True):
            self.assertTrue(eth._real_order_priority_enabled())

    def test_real_position_close_waits_for_binance_confirmation(self):
        with (
            patch.object(eth, "_get_follow_mode_enabled", return_value=True),
            patch.object(eth, "_is_real_copy_enabled", return_value=True),
            patch.dict(eth.os.environ, {"REAL_ORDER_PRIORITY_ENABLED": "1"}, clear=False),
        ):
            self.assertTrue(eth._binance_close_confirmation_required())

    def test_local_tp_sl_hit_detection_is_direction_aware(self):
        long_trade = {
            "direction": "long",
            "entry": 1800.0,
            "tp": 1830.0,
            "sl": 1780.0,
        }
        short_trade = {
            "direction": "short",
            "entry": 1900.0,
            "tp": 1875.0,
            "sl": 1925.0,
        }

        self.assertEqual(
            eth._local_tp_sl_hits(long_trade, 1825.0, 1831.0, 1810.0),
            (True, False),
        )
        self.assertEqual(
            eth._local_tp_sl_hits(short_trade, 1880.0, 1890.0, 1874.0),
            (True, False),
        )

    def test_copy_trade_fallback_preserves_small_signal_ratio(self):
        with (
            patch.object(eth, "_get_binance_available_balance", return_value=0.0),
        ):
            qty = eth._calc_copy_trade_qty(0.05, leverage=5, asset_price=70_000.0)
            buffered_qty = eth._calc_copy_trade_qty_with_buffer(
                0.05,
                leverage=5,
                asset_price=70_000.0,
                extra_buffer_ratio=0.8,
                enforce_min=False,
            )

        self.assertEqual(qty, eth.COPY_TRADE_MIN_QTY)
        self.assertEqual(buffered_qty, 0.0)

    def test_position_state_drops_flat_previous_symbol_trade_history(self):
        old_path = eth.POSITION_PANEL_FILE
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                eth.POSITION_PANEL_FILE = Path(tmpdir) / "position.json"
                eth.POSITION_PANEL_FILE.write_text(
                    json.dumps(
                        {
                            "pair": "ETHUSDT",
                            "open": False,
                            "last_close_reason": "SL",
                            "last_close_price": 1900,
                            "close_hits": [{"reason": "SL", "price": 1900, "ts": 1}],
                            "daily_trade_date": "2026-07-30",
                            "daily_trade_opened": True,
                        }
                    ),
                    encoding="utf-8",
                )
                state = eth._load_position_panel_state()

            self.assertEqual(state["last_close_reason"], "")
            self.assertEqual(state["last_close_price"], 0.0)
            self.assertEqual(state["close_hits"], [])
            self.assertFalse(state["daily_trade_opened"])
        finally:
            eth.POSITION_PANEL_FILE = old_path

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

    def test_panel_snapshot_publishes_breakout_quality(self):
        breakout = {
            "attempt": 1,
            "confirmed": False,
            "quality_score": 2.5,
            "required_score": 3.5,
            "resistance_level": 1940.75,
        }
        eth.POSITION_PANEL_STATE["strategy_breakout"] = breakout

        with (
            patch.object(eth, "_refresh_position_panel_account_state"),
            patch.object(eth, "_write_json_atomic") as write_snapshot,
            patch.object(eth, "_queue_panel_realtime_publish"),
        ):
            eth.sync_position_panel(1938.0)

        payload = write_snapshot.call_args.args[1]
        self.assertEqual(payload["strategy_breakout"], breakout)

    def test_panel_snapshot_publishes_monthly5_shadow_state(self):
        shadow_state = {
            "strategy_id": "monthly5_postlock_hourly_v0",
            "shadow_only": True,
            "mode": "post_lock",
            "suggested_exposure_scale": 0.15,
            "market_selection": {
                "market_bias": "bullish",
                "selected_plan": "post_lock_low_exposure",
                "shadow_action": "reduced_exposure",
            },
        }
        eth.POSITION_PANEL_STATE["monthly5_execution_guard"] = {
            "allowed": True,
            "adjusted_size": 0.15,
            "reason_code": "allowed",
        }

        with (
            patch.object(eth, "_refresh_position_panel_account_state"),
            patch.object(eth, "_update_monthly5_shadow_panel_state", return_value=shadow_state),
            patch.object(eth, "_write_json_atomic") as write_snapshot,
            patch.object(eth, "_queue_panel_realtime_publish"),
        ):
            eth.sync_position_panel(1938.0)

        payload = write_snapshot.call_args.args[1]
        self.assertEqual(payload["monthly5_shadow"], shadow_state)
        self.assertEqual(
            payload["monthly5_shadow"]["market_selection"]["selected_plan"],
            "post_lock_low_exposure",
        )
        self.assertEqual(payload["monthly5_execution_guard"]["adjusted_size"], 0.15)

    def test_monthly5_position_guard_reduces_local_position_to_cap(self):
        eth.active_trade.update(
            {
                "open": True,
                "direction": "long",
                "entry": 1900.0,
                "avg_entry": 1900.0,
                "tp": 1950.0,
                "sl": 1870.0,
                "size": 0.5,
                "max_size": 0.8,
                "min_size": 0.1,
                "monthly5_position_guard_ts": 0.0,
            }
        )
        shadow_state = {
            "mode": "post_lock",
            "market_selection": {
                "selected_plan": "post_lock_low_exposure",
                "shadow_action": "reduced_exposure",
                "exposure_cap": 0.15,
            },
        }

        with (
            patch.object(eth, "_update_monthly5_shadow_panel_state", return_value=shadow_state),
            patch.object(eth, "_get_follow_mode_enabled", return_value=False),
            patch.object(eth, "_is_real_copy_enabled", return_value=False),
            patch.object(eth, "sync_position_panel"),
            patch.object(eth, "send_telegram"),
            patch.object(eth.time, "time", return_value=2000.0),
        ):
            adjusted = eth.manage_monthly5_position_guard(1910.0)

        self.assertTrue(adjusted)
        self.assertEqual(eth.active_trade["size"], 0.15)
        self.assertEqual(eth.POSITION_PANEL_STATE["monthly5_position_guard"]["action"], "reduce_to_cap")

    def test_monthly5_position_guard_closes_local_position_on_risk_off(self):
        eth.active_trade.update(
            {
                "open": True,
                "direction": "short",
                "entry": 1900.0,
                "avg_entry": 1900.0,
                "tp": 1850.0,
                "sl": 1930.0,
                "size": 0.5,
                "position_qty": 0.01,
                "monthly5_position_guard_ts": 0.0,
            }
        )
        shadow_state = {
            "mode": "intraday_stop",
            "market_selection": {
                "selected_plan": "risk_off",
                "shadow_action": "risk_off",
                "exposure_cap": 0.0,
            },
        }

        with (
            patch.object(eth, "_update_monthly5_shadow_panel_state", return_value=shadow_state),
            patch.object(eth, "_get_follow_mode_enabled", return_value=False),
            patch.object(eth, "_is_real_copy_enabled", return_value=False),
            patch.object(eth, "record_position_close"),
            patch.object(eth, "sync_position_panel"),
            patch.object(eth, "send_telegram"),
            patch.object(eth.time, "time", return_value=2000.0),
        ):
            adjusted = eth.manage_monthly5_position_guard(1910.0)

        self.assertTrue(adjusted)
        self.assertFalse(eth.active_trade["open"])
        self.assertEqual(eth.active_trade["size"], 0.0)

    @patch("eth.time.time", return_value=2000.0)
    def test_only_actual_open_sets_long_or_short_signal(self, _time):
        eth.active_trade["direction"] = "long"
        eth.POSITION_PANEL_STATE["binance_mark_price_ts"] = 1999
        decision = {
            "final": "做多",
            "score": 0.7,
            "ai_prob": 0.72,
            "regime": "bull_trend",
            "host_opening_logic": {
                "direction": "long",
                "mode": "trend_pullback_long",
                "confidence": 0.71,
                "edge": 1.4,
                "range_pos": 0.52,
                "reasons": ["4H/1H方向偏多"],
            },
            "macro_indicator_alignment": {
                "score": 1.3,
                "min_score": 1.15,
                "aligned": 4,
                "against": 1,
                "hard_block": False,
                "reasons": ["大中週期趨勢同向"],
            },
            "htf": 1,
            "mid_trend": 1,
            "derivatives_pressure": 0.22,
            "event_risk": 0,
            "net_edge_rate_est": 0.003,
            "risk_rate": 0.01,
        }

        eth._update_panel_execution_snapshot(
            decision, 1871.8, "pending_confirmation", reason="等待確認", actual_open=False
        )
        self.assertEqual(eth.POSITION_PANEL_STATE["strategy_signal"], "wait")
        self.assertFalse(eth.POSITION_PANEL_STATE["strategy_actual_open"])
        self.assertEqual(
            eth.POSITION_PANEL_STATE["strategy_host_logic"]["mode"],
            "trend_pullback_long",
        )
        self.assertEqual(eth.POSITION_PANEL_STATE["strategy_macro_alignment"]["score"], 1.3)
        self.assertEqual(eth.POSITION_PANEL_STATE["strategy_context"]["htf"], 1)

        eth._update_panel_execution_snapshot(
            decision, 1871.8, "opened", reason="Binance 實際開單成功", actual_open=True
        )
        self.assertEqual(eth.POSITION_PANEL_STATE["strategy_signal"], "long")
        self.assertTrue(eth.POSITION_PANEL_STATE["strategy_actual_open"])
        self.assertEqual(eth.POSITION_PANEL_STATE["strategy_evaluated_ts"], 2000)


if __name__ == "__main__":
    unittest.main()
