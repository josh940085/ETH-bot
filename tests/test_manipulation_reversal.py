import os
import unittest
from unittest.mock import patch

import pandas as pd

import eth


def _frame(rows):
    index = pd.date_range("2026-01-01", periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame(rows, index=index)


class ManipulationReversalTests(unittest.TestCase):
    def _base_frames(self):
        frame_5m = _frame(
            [
                {"open": 100.0, "high": 100.4, "low": 99.6, "close": 100.1, "volume": 1.0},
                {"open": 100.1, "high": 100.5, "low": 99.7, "close": 100.2, "volume": 1.0},
                {"open": 100.2, "high": 100.6, "low": 99.8, "close": 100.3, "volume": 1.0},
                {"open": 100.3, "high": 100.7, "low": 99.9, "close": 100.4, "volume": 1.0},
                {"open": 100.4, "high": 100.8, "low": 100.0, "close": 100.5, "volume": 1.0},
                {"open": 100.5, "high": 100.9, "low": 100.1, "close": 100.6, "volume": 1.0},
                {"open": 100.6, "high": 101.0, "low": 100.2, "close": 100.7, "volume": 1.0},
                {"open": 100.7, "high": 101.1, "low": 100.3, "close": 100.8, "volume": 1.0},
                {"open": 100.8, "high": 101.2, "low": 100.4, "close": 100.9, "volume": 1.0},
                {"open": 100.9, "high": 102.4, "low": 100.5, "close": 100.7, "volume": 2.0},
                {"open": 100.7, "high": 101.0, "low": 99.2, "close": 100.4, "volume": 2.0},
                {"open": 100.4, "high": 100.8, "low": 98.8, "close": 100.2, "volume": 2.0},
            ]
        )
        frame_15m = frame_5m.resample("15min", label="right", closed="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        return frame_5m, frame_15m

    def test_upside_fake_breakout_builds_small_short_reversal(self):
        frame_5m, frame_15m = self._base_frames()
        with patch.dict(
            os.environ,
            {
                "TRADE_MANIPULATION_REVERSAL_ENABLED": "1",
                "TRADE_MANIPULATION_REVERSAL_MAX_EVENT_RISK": "0",
            },
        ):
            plan = eth._build_manipulation_reversal_plan(
                price=100.2,
                atr=0.8,
                df_5m=frame_5m,
                df_15m=frame_15m,
                fake_breakout=True,
                breakout=0,
                breakout_attempt=1,
                breakout_quality={"score": 1.5, "required_score": 3.5},
                sweep_high=True,
                sweep_low=False,
                absorption=True,
                buy_pressure=False,
                sell_pressure=True,
                btc_change=-0.2,
                news_bias=0.0,
                event_risk=0,
                derivatives_flow={},
                score=0.58,
                regime="range",
            )

        self.assertIsNotNone(plan)
        self.assertEqual(plan["direction"], "short")
        self.assertEqual(plan["host_opening_logic"]["mode"], "manipulation_reversal")
        self.assertLess(plan["tp"], 100.2)
        self.assertGreater(plan["sl"], 100.2)
        self.assertLessEqual(plan["position_size"], 0.02)

    def test_downside_fake_breakout_builds_small_long_reversal(self):
        frame_5m, frame_15m = self._base_frames()
        plan = eth._build_manipulation_reversal_plan(
            price=100.2,
            atr=0.8,
            df_5m=frame_5m,
            df_15m=frame_15m,
            fake_breakout=True,
            breakout=0,
            breakout_attempt=-1,
            breakout_quality={"score": 1.5, "required_score": 3.5},
            sweep_high=False,
            sweep_low=True,
            absorption=True,
            buy_pressure=True,
            sell_pressure=False,
            btc_change=0.2,
            news_bias=0.0,
            event_risk=0,
            derivatives_flow={},
            score=0.42,
            regime="range",
        )

        self.assertIsNotNone(plan)
        self.assertEqual(plan["direction"], "long")
        self.assertGreater(plan["tp"], 100.2)
        self.assertLess(plan["sl"], 100.2)

    def test_quality_failed_breakout_attempt_can_reverse_without_fake_flag(self):
        frame_5m, frame_15m = self._base_frames()
        plan = eth._build_manipulation_reversal_plan(
            price=100.2,
            atr=0.8,
            df_5m=frame_5m,
            df_15m=frame_15m,
            fake_breakout=False,
            breakout=0,
            breakout_attempt=-1,
            breakout_quality={"score": 2.0, "required_score": 3.0},
            sweep_high=False,
            sweep_low=False,
            absorption=False,
            buy_pressure=True,
            sell_pressure=True,
            btc_change=0.0,
            news_bias=0.0,
            event_risk=0,
            derivatives_flow={},
            score=0.26,
            regime="range",
        )

        self.assertIsNotNone(plan)
        self.assertEqual(plan["direction"], "long")
        self.assertIn("breakdown_quality_failed", plan["evidence"])

    def test_event_risk_blocks_manipulation_reversal(self):
        frame_5m, frame_15m = self._base_frames()
        plan = eth._build_manipulation_reversal_plan(
            price=100.2,
            atr=0.8,
            df_5m=frame_5m,
            df_15m=frame_15m,
            fake_breakout=True,
            breakout=0,
            breakout_attempt=1,
            breakout_quality={"score": 1.5, "required_score": 3.5},
            sweep_high=True,
            sweep_low=False,
            absorption=True,
            buy_pressure=False,
            sell_pressure=True,
            btc_change=-0.2,
            news_bias=0.0,
            event_risk=1,
            derivatives_flow={},
            score=0.58,
            regime="range",
        )

        self.assertIsNone(plan)

    def test_strong_news_against_reversal_blocks_entry(self):
        frame_5m, frame_15m = self._base_frames()
        plan = eth._build_manipulation_reversal_plan(
            price=100.2,
            atr=0.8,
            df_5m=frame_5m,
            df_15m=frame_15m,
            fake_breakout=True,
            breakout=0,
            breakout_attempt=1,
            breakout_quality={"score": 1.5, "required_score": 3.5},
            sweep_high=True,
            sweep_low=False,
            absorption=True,
            buy_pressure=False,
            sell_pressure=True,
            btc_change=-0.2,
            news_bias=0.7,
            event_risk=0,
            derivatives_flow={},
            score=0.58,
            regime="range",
        )

        self.assertIsNone(plan)


if __name__ == "__main__":
    unittest.main()
