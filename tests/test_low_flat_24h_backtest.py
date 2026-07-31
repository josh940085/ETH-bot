import datetime as dt
import os
import unittest
from unittest.mock import patch

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
            "news_bias": 0.0,
            "event_risk": 0,
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

        trade = backtest._build_open_trade(frame_5m.index[-1], "long", final, 108.0, _score, candidate)
        self.assertEqual(trade["size"], 0.02)

    def test_low_flat_candidate_can_use_exchange_minimum_size(self):
        decision = {
            "tp": 101.0,
            "sl": 99.0,
            "position_size": 0.001,
            "host_opening_logic": {"mode": "low_flat_24h"},
        }

        trade = backtest._build_open_trade(
            dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
            "long",
            "做多（低空倉24h候選）",
            100.0,
            0.56,
            decision,
        )

        self.assertEqual(trade["size"], 0.001)

    def test_low_flat_candidate_does_not_scale_position(self):
        decision = {
            "tp": 101.0,
            "sl": 99.0,
            "position_size": 0.001,
            "host_opening_logic": {"mode": "low_flat_24h"},
        }
        trade = backtest._build_open_trade(
            dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
            "long",
            "做多（低空倉24h候選）",
            100.0,
            0.56,
            decision,
        )

        with (
            patch.object(backtest.eth, "maybe_lock_profit_after_reversal", return_value=False),
            patch.object(backtest.eth, "maybe_activate_auto_break_even", return_value=False),
            patch.object(backtest.eth, "maybe_shrink_tp_after_hold", return_value=False),
            patch.object(backtest.eth, "_ensure_minimum_net_profit_tp", return_value=(101.0, 0.0)),
            patch.object(backtest.eth, "manage_position_scaling") as scaling,
        ):
            updated = backtest._apply_trade_management(
                trade,
                100.5,
                1.0,
                dt.datetime(2026, 1, 1, 1, tzinfo=dt.timezone.utc),
                favorable_price=100.8,
            )

        scaling.assert_not_called()
        self.assertEqual(updated["size"], 0.001)

    def test_candidate_rejects_adverse_news_direction(self):
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
            "news_bias": -0.7,
            "event_risk": 0,
            "host_opening_logic": {"reasons": ["baseline_wait"]},
        }

        result = backtest._build_low_flat_24h_candidate(
            decision,
            {"5m": frame_5m, "15m": frame_15m},
            float(frame_5m["close"].iloc[-1]),
            frame_5m.index[-1],
            0.0,
        )

        self.assertIsNone(result)

    def test_candidate_rejects_high_event_risk_by_default(self):
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
            "news_bias": 0.0,
            "event_risk": 1,
            "host_opening_logic": {"reasons": ["baseline_wait"]},
        }

        with patch.dict(os.environ, {"BACKTEST_LOW_FLAT_MAX_EVENT_RISK": "0"}):
            result = backtest._build_low_flat_24h_candidate(
                decision,
                {"5m": frame_5m, "15m": frame_15m},
                float(frame_5m["close"].iloc[-1]),
                frame_5m.index[-1],
                0.0,
            )

        self.assertIsNone(result)

    def test_low_flat_news_gate_rejects_non_daily_event_risk(self):
        self.assertFalse(
            backtest._low_flat_news_allows_entry(
                {"event_risk": 1, "news_bias": 0.0},
                "long",
                daily_min_forced=False,
            )
        )

    def test_low_flat_news_gate_allows_daily_min_event_risk(self):
        self.assertTrue(
            backtest._low_flat_news_allows_entry(
                {"event_risk": 1, "news_bias": -1.0},
                "long",
                daily_min_forced=True,
            )
        )

    def test_low_flat_strict_quality_requires_rr_direction_macro_and_edge(self):
        quality = {
            "rr_at_entry": 2.1,
            "event_risk": 1,
            "ai_long_prob": 0.66,
            "ai_short_prob": 0.4,
            "macro_bias": 0.2,
            "net_edge_rate_est_pct": 0.25,
            "total_trade_cost_rate_est_pct": 0.1,
        }
        self.assertTrue(backtest._low_flat_strict_quality_allows_entry(quality, "long"))

        weak_rr = dict(quality, rr_at_entry=1.9)
        self.assertFalse(backtest._low_flat_strict_quality_allows_entry(weak_rr, "long"))

        weak_prob = dict(quality, ai_long_prob=0.64)
        self.assertFalse(backtest._low_flat_strict_quality_allows_entry(weak_prob, "long"))

        macro_against = dict(quality, macro_bias=-0.1)
        self.assertFalse(backtest._low_flat_strict_quality_allows_entry(macro_against, "long"))

        fee_eats_edge = dict(quality, net_edge_rate_est_pct=0.05)
        self.assertFalse(backtest._low_flat_strict_quality_allows_entry(fee_eats_edge, "long"))

    def test_low_flat_strict_quality_supports_short_direction(self):
        quality = {
            "rr_at_entry": 2.0,
            "event_risk": 1,
            "ai_long_prob": 0.4,
            "ai_short_prob": 0.58,
            "macro_bias": -0.2,
            "net_edge_rate_est_pct": 0.2,
            "total_trade_cost_rate_est_pct": 0.1,
        }

        self.assertTrue(backtest._low_flat_strict_quality_allows_entry(quality, "short"))
        self.assertFalse(
            backtest._low_flat_strict_quality_allows_entry(dict(quality, macro_bias=0.1), "short")
        )

    def test_forced_max_hold_caps_existing_trade_limit(self):
        trade = {"max_hold_sec": 48 * 3600.0}

        capped = backtest._apply_forced_max_hold(trade, force_max_hold_hours=24.0)

        self.assertEqual(capped["max_hold_sec"], 24 * 3600.0)
        self.assertEqual(trade["max_hold_sec"], 48 * 3600.0)

    def test_forced_max_hold_keeps_shorter_existing_trade_limit(self):
        trade = {"max_hold_sec": 6 * 3600.0}

        capped = backtest._apply_forced_max_hold(trade, force_max_hold_hours=24.0)

        self.assertEqual(capped["max_hold_sec"], 6 * 3600.0)

    def test_low_flat_quality_size_allows_only_bounded_high_reward_setups(self):
        quality = {
            "risk_rate": 0.012,
            "reward_rate": 0.035,
            "breakout_quality_score": 2.0,
        }
        weak = {
            "risk_rate": 0.02,
            "reward_rate": 0.035,
            "breakout_quality_score": 2.0,
        }

        self.assertEqual(backtest._low_flat_quality_size(quality, default_size=0.001), 0.03)
        self.assertEqual(backtest._low_flat_quality_size(weak, default_size=0.001), 0.001)

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
