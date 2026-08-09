import unittest

import numpy as np
import pandas as pd

import monthly5_intramonth_recovery_research as account


class Monthly5LeverageResearchTests(unittest.TestCase):
    def _frame(self, closes):
        values = np.asarray(closes, dtype="float64")
        return pd.DataFrame(
            {"open": values, "high": values, "low": values, "close": values},
            index=pd.date_range("2026-01-01", periods=len(values), freq="5min", tz="UTC"),
        )

    def test_leverage_scales_pnl_and_exposure(self):
        frame = self._frame([100.0, 101.0])
        desired = np.ones(len(frame))
        factors, _, _, _, positions = account.simulate_account_path(
            frame,
            desired,
            np.zeros(len(frame), dtype="int32"),
            desired,
            {"mode": "none", "trigger": None, "scale": 0.0, "exit": 0.0},
            leverage=2.0,
            round_trip_fee=0.0,
        )
        self.assertAlmostEqual(factors[1], 1.02)
        self.assertEqual(positions[1], 2.0)

    def test_leverage_is_capped_at_five(self):
        frame = self._frame([100.0])
        desired = np.ones(len(frame))
        with self.assertRaises(ValueError):
            account.simulate_account_path(
                frame,
                desired,
                np.zeros(len(frame), dtype="int32"),
                desired,
                {"mode": "none", "trigger": None, "scale": 0.0, "exit": 0.0},
                leverage=5.1,
            )

    def test_strategy_change_charges_each_leverage_leg(self):
        frame = self._frame([100.0, 100.0])
        desired = np.ones(len(frame))
        factors, _, _, _, positions = account.simulate_account_path(
            frame,
            desired,
            np.array([0, 10], dtype="int32"),
            desired,
            {"mode": "none", "trigger": None, "scale": 0.0, "exit": 0.0},
            risk_profiles={
                10: {
                    "stop_pct": 0.03,
                    "target_pct": 0.06,
                    "cooldown_bars": 12,
                    "leverage": 2.0,
                }
            },
            round_trip_fee=0.001,
        )
        self.assertAlmostEqual(factors[1], 1.0 - 0.0015)
        self.assertEqual(positions[1], 2.0)


if __name__ == "__main__":
    unittest.main()
