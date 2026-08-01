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
