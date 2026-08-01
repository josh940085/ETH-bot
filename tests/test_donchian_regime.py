import os
import unittest
from unittest.mock import patch

import pandas as pd

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import eth


def _hourly_frame():
    idx = pd.date_range("2026-01-01", periods=90, freq="1h", tz="UTC")
    rows = []
    for i in range(90):
        close = 100.0 + i * 0.03
        rows.append(
            {
                "open": close - 0.1,
                "high": close + 0.4,
                "low": close - 0.4,
                "close": close,
                "volume": 1.0,
            }
        )
    return pd.DataFrame(rows, index=idx)


def _daily_frame(bull=True):
    idx = pd.date_range("2025-11-01", periods=70, freq="1D", tz="UTC")
    rows = []
    for i in range(70):
        close = 90.0 + i if bull else 160.0 - i
        rows.append({"open": close, "high": close + 2.0, "low": close - 2.0, "close": close, "volume": 1.0})
    return pd.DataFrame(rows, index=idx)


def _fake_breakout_hourly_frame():
    idx = pd.date_range("2026-01-01", periods=220, freq="1h", tz="UTC")
    rows = []
    spike = 106.0
    for i in range(220):
        if i >= 80 and i % 20 == 0:
            close = spike
            spike += 1.0
        elif i >= 80 and i % 20 in {1, 2, 3, 4, 5, 6}:
            close = 100.0 + (i % 3) * 0.2
        else:
            close = 100.0 + ((i % 10) - 5) * 0.15
        rows.append({"open": close, "high": close + 0.2, "low": close - 0.2, "close": close, "volume": 1.0})
    return pd.DataFrame(rows, index=idx)


def _trend_hourly_frame():
    idx = pd.date_range("2026-01-01", periods=220, freq="1h", tz="UTC")
    rows = []
    for i in range(220):
        close = 100.0 + i * 0.12
        rows.append({"open": close - 0.05, "high": close + 0.2, "low": close - 0.1, "close": close, "volume": 1.0})
    return pd.DataFrame(rows, index=idx)


class DonchianRegimeTests(unittest.TestCase):
    def test_bull_daily_regime_and_donchian_breakout_builds_long(self):
        df_1h = _hourly_frame()
        completed_high = float(df_1h.iloc[:-1]["high"].tail(72).max())

        with patch.dict(os.environ, {"TRADE_DONCHIAN_REGIME_ENABLED": "1"}):
            plan = eth._build_donchian_regime_plan(
                price=completed_high + 1.0,
                df_1h=df_1h,
                df_1d=_daily_frame(bull=True),
                news_bias=0.0,
                event_risk=0,
            )

        self.assertIsNotNone(plan)
        self.assertEqual(plan["direction"], "long")
        self.assertEqual(plan["host_opening_logic"]["mode"], "don72_ls_ma10_50")
        self.assertGreater(plan["tp"], completed_high + 1.0)
        self.assertLess(plan["sl"], completed_high + 1.0)

    def test_bear_daily_regime_and_donchian_breakdown_builds_short(self):
        df_1h = _hourly_frame()
        completed_low = float(df_1h.iloc[:-1]["low"].tail(72).min())

        with patch.dict(os.environ, {"TRADE_DONCHIAN_REGIME_ENABLED": "1"}):
            plan = eth._build_donchian_regime_plan(
                price=completed_low - 1.0,
                df_1h=df_1h,
                df_1d=_daily_frame(bull=False),
                news_bias=0.0,
                event_risk=0,
            )

        self.assertIsNotNone(plan)
        self.assertEqual(plan["direction"], "short")
        self.assertLess(plan["tp"], completed_low - 1.0)
        self.assertGreater(plan["sl"], completed_low - 1.0)

    def test_news_against_direction_blocks_candidate(self):
        df_1h = _hourly_frame()
        completed_high = float(df_1h.iloc[:-1]["high"].tail(72).max())

        with patch.dict(os.environ, {"TRADE_DONCHIAN_REGIME_ENABLED": "1"}):
            plan = eth._build_donchian_regime_plan(
                price=completed_high + 1.0,
                df_1h=df_1h,
                df_1d=_daily_frame(bull=True),
                news_bias=-0.8,
                event_risk=0,
            )

        self.assertIsNone(plan)

    def test_candidate_allows_scanned_tv_point_six_size(self):
        df_1h = _hourly_frame()
        completed_high = float(df_1h.iloc[:-1]["high"].tail(72).max())

        with patch.dict(
            os.environ,
            {
                "TRADE_DONCHIAN_REGIME_ENABLED": "1",
                "TRADE_DONCHIAN_REGIME_SIZE_RATIO": "0.60",
                "TRADE_INITIAL_MAX_OPEN_SIZE_RATIO": "0.80",
                "TRADE_MAX_OPEN_SIZE_RATIO": "0.80",
            },
        ):
            plan = eth._build_donchian_regime_plan(
                price=completed_high + 1.0,
                df_1h=df_1h,
                df_1d=_daily_frame(bull=True),
                news_bias=0.0,
                event_risk=0,
            )

        self.assertIsNotNone(plan)
        self.assertEqual(plan["position_size"], 0.60)

    def test_candidate_defaults_to_full_available_asset_ratio(self):
        df_1h = _hourly_frame()
        completed_high = float(df_1h.iloc[:-1]["high"].tail(72).max())

        with patch.dict(
            os.environ,
            {
                "TRADE_DONCHIAN_REGIME_ENABLED": "1",
                "TRADE_DONCHIAN_REGIME_SIZE_RATIO": "1.00",
                "TRADE_INITIAL_MAX_OPEN_SIZE_RATIO": "0.55",
                "TRADE_MAX_OPEN_SIZE_RATIO": "0.80",
            },
        ):
            plan = eth._build_donchian_regime_plan(
                price=completed_high + 1.0,
                df_1h=df_1h,
                df_1d=_daily_frame(bull=True),
                news_bias=0.0,
                event_risk=0,
            )

        self.assertIsNotNone(plan)
        self.assertEqual(plan["position_size"], 1.00)

    def test_recent_fake_breakout_market_state_blocks_donchian_entry(self):
        df_1h = _fake_breakout_hourly_frame()
        completed_high = float(df_1h.iloc[:-1]["high"].tail(72).max())

        with patch.dict(
            os.environ,
            {
                "TRADE_DONCHIAN_REGIME_ENABLED": "1",
                "TRADE_DONCHIAN_MARKET_STATE_FILTER_ENABLED": "1",
            },
        ):
            state = eth._score_recent_donchian_market_state(
                df_1h,
                direction="long",
                current_price=completed_high + 1.0,
            )
            plan = eth._build_donchian_regime_plan(
                price=completed_high + 1.0,
                df_1h=df_1h,
                df_1d=_daily_frame(bull=True),
                news_bias=0.0,
                event_risk=0,
            )

        self.assertEqual(state["action"], "block")
        self.assertGreaterEqual(state["fake_rate"], 0.70)
        self.assertIsNone(plan)

    def test_recent_trend_market_state_keeps_full_donchian_size(self):
        df_1h = _trend_hourly_frame()
        completed_high = float(df_1h.iloc[:-1]["high"].tail(72).max())

        with patch.dict(
            os.environ,
            {
                "TRADE_DONCHIAN_REGIME_ENABLED": "1",
                "TRADE_DONCHIAN_REGIME_SIZE_RATIO": "1.00",
                "TRADE_DONCHIAN_MARKET_STATE_FILTER_ENABLED": "1",
            },
        ):
            state = eth._score_recent_donchian_market_state(
                df_1h,
                direction="long",
                current_price=completed_high + 1.0,
            )
            plan = eth._build_donchian_regime_plan(
                price=completed_high + 1.0,
                df_1h=df_1h,
                df_1d=_daily_frame(bull=True),
                news_bias=0.0,
                event_risk=0,
            )

        self.assertEqual(state["action"], "allow")
        self.assertIsNotNone(plan)
        self.assertEqual(plan["position_size"], 1.00)
        self.assertEqual(plan["market_state_filter"]["action"], "allow")

    def test_full_asset_ratio_qty_uses_actual_available_balance_with_minimum(self):
        with (
            patch.object(eth, "_get_binance_available_balance", return_value=100.0),
            patch.object(eth, "WS_PRICE", 100_000.0),
            patch.dict(
                os.environ,
                {
                    "COPY_TRADE_BALANCE_USAGE_CAP": "1.00",
                    "COPY_TRADE_NOTIONAL_SAFETY": "1.00",
                    "COPY_TRADE_LEVERAGE": "5",
                },
            ),
        ):
            qty = eth._calc_copy_trade_qty(1.0, leverage=5, asset_price=100_000.0)

        self.assertEqual(qty, 0.005)

        with (
            patch.object(eth, "_get_binance_available_balance", return_value=1.0),
            patch.object(eth, "WS_PRICE", 100_000.0),
            patch.dict(
                os.environ,
                {
                    "COPY_TRADE_BALANCE_USAGE_CAP": "1.00",
                    "COPY_TRADE_NOTIONAL_SAFETY": "1.00",
                    "COPY_TRADE_LEVERAGE": "5",
                },
            ),
        ):
            min_qty = eth._calc_copy_trade_qty(1.0, leverage=5, asset_price=100_000.0)

        self.assertEqual(min_qty, 0.001)

    def test_disabled_candidate_returns_none(self):
        df_1h = _hourly_frame()
        completed_high = float(df_1h.iloc[:-1]["high"].tail(72).max())

        with patch.dict(os.environ, {"TRADE_DONCHIAN_REGIME_ENABLED": "0"}):
            plan = eth._build_donchian_regime_plan(
                price=completed_high + 1.0,
                df_1h=df_1h,
                df_1d=_daily_frame(bull=True),
                news_bias=0.0,
                event_risk=0,
            )

        self.assertIsNone(plan)


if __name__ == "__main__":
    unittest.main()
