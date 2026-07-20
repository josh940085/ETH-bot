import os
import unittest
from unittest import mock

import pandas as pd

import eth


class PreviousMonthAdrTpSlTests(unittest.TestCase):
    def _daily_frame(self):
        index = pd.date_range("2026-06-01", periods=30, freq="1D", tz="UTC")
        return pd.DataFrame(
            {
                "high": [105.0] * 30,
                "low": [95.0] * 30,
                "close": [100.0] * 30,
            },
            index=index,
        )

    def test_uses_only_previous_complete_month(self):
        frame = self._daily_frame()
        july = pd.DataFrame(
            {"high": [140.0], "low": [80.0], "close": [110.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2026-07-01", tz="UTC")]),
        )
        result = eth._previous_month_average_daily_range(
            pd.concat([frame, july]),
            reference_ts=pd.Timestamp("2026-07-20", tz="UTC"),
        )
        self.assertEqual(result["month"], "2026-06")
        self.assertEqual(result["days"], 30)
        self.assertAlmostEqual(result["value"], 10.0)

    def test_live_range_index_uses_time_column(self):
        frame = self._daily_frame().reset_index(drop=True)
        frame["time"] = [
            int(ts.timestamp() * 1000)
            for ts in pd.date_range("2026-06-01", periods=30, freq="1D", tz="UTC")
        ]
        result = eth._previous_month_average_daily_range(
            frame,
            reference_ts=pd.Timestamp("2026-07-20", tz="UTC").timestamp() * 1000,
        )
        self.assertEqual(result["month"], "2026-06")
        self.assertEqual(result["days"], 30)
        self.assertAlmostEqual(result["value"], 10.0)

    def test_long_plan_uses_adr_and_preserves_minimum_rr(self):
        with mock.patch.dict(
            os.environ,
            {
                "TRADE_PREV_MONTH_ADR_ENABLED": "1",
                "TRADE_PREV_MONTH_ADR_MODE": "exact",
                "TRADE_PREV_MONTH_ADR_STOP_RATIO": "0.18",
                "TRADE_PREV_MONTH_ADR_TARGET_RATIO": "0.33",
            },
            clear=False,
        ):
            sl, tp, meta = eth._apply_previous_month_adr_trade_plan(
                "做多",
                100.0,
                99.0,
                102.0,
                self._daily_frame(),
                reference_ts=pd.Timestamp("2026-07-20", tz="UTC"),
                min_rr=1.8,
            )
        self.assertAlmostEqual(sl, 98.2)
        self.assertAlmostEqual(tp, 103.3)
        self.assertGreaterEqual(meta["planned_rr"], 1.8)
        self.assertTrue(meta["applied"])

    def test_short_plan_falls_back_when_previous_month_is_incomplete(self):
        frame = self._daily_frame().iloc[:10]
        sl, tp, meta = eth._apply_previous_month_adr_trade_plan(
            "做空",
            100.0,
            101.0,
            98.0,
            frame,
            reference_ts=pd.Timestamp("2026-07-20", tz="UTC"),
        )
        self.assertEqual((sl, tp), (101.0, 98.0))
        self.assertEqual(meta, {})

    def test_daily_min_plan_uses_adr_bounds_without_replacing_valid_structure(self):
        index_15m = pd.date_range("2026-07-19 19:00", periods=20, freq="15min", tz="UTC")
        frame_15m = pd.DataFrame(
            {"high": [101.0] * 20, "low": [99.0] * 20},
            index=index_15m,
        )
        index_5m = pd.date_range("2026-07-19 23:30", periods=6, freq="5min", tz="UTC")
        frame_5m = pd.DataFrame(
            {"high": [101.0] * 6, "low": [99.0] * 6},
            index=index_5m,
        )
        with mock.patch.dict(
            os.environ,
            {
                "TRADE_PREV_MONTH_ADR_DAILY_MIN_ENABLED": "1",
                "TRADE_PREV_MONTH_ADR_MODE": "bounded",
            },
            clear=False,
        ):
            plan = eth._build_daily_min_trade_plan(
                100.0,
                1.0,
                frame_15m,
                frame_5m,
                1,
                1,
                df_1d=self._daily_frame(),
            )
        self.assertTrue(plan["previous_month_adr"]["applied"])
        self.assertEqual(plan["previous_month_adr"]["mode"], "bounded")
        self.assertAlmostEqual(abs(plan["sl"] - 100.0), 1.2)
        self.assertAlmostEqual(abs(plan["tp"] - 100.0), 1.8)

    def test_general_trade_adr_is_opt_in(self):
        self.assertFalse(eth._is_truthy(os.getenv("TRADE_PREV_MONTH_ADR_GENERAL_ENABLED", "0")))


if __name__ == "__main__":
    unittest.main()
