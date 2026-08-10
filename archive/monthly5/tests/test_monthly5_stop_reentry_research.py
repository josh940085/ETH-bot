import unittest

import numpy as np
import pandas as pd

import monthly5_intramonth_recovery_research as account


class Monthly5StopReentryResearchTests(unittest.TestCase):
    def _frame(self):
        index = pd.date_range("2026-01-01", periods=6, freq="5min", tz="UTC")
        frame = pd.DataFrame(
            {
                "open": [100.0] * 6,
                "high": [100.0] * 6,
                "low": [100.0, 98.0, 100.0, 100.0, 100.0, 100.0],
                "close": [100.0] * 6,
            },
            index=index,
        )
        return frame

    def test_signal_reset_blocks_same_trend_until_it_clears(self):
        frame = self._frame()
        desired = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        _, _, _, _, positions = account.simulate_account_path(
            frame,
            desired,
            np.zeros(len(frame), dtype="int32"),
            desired,
            {"mode": "none", "trigger": None, "scale": 0.0, "exit": 0.0},
            risk_profiles={
                0: {"stop_pct": 0.01, "target_pct": 0.03, "cooldown_bars": 1}
            },
            stop_reentry_policy="signal_reset",
            round_trip_fee=0.0,
        )
        self.assertEqual(positions.tolist(), [1.0, 1.0, 0.0, 0.0, 1.0, 1.0])

    def test_stop_wait_is_distinct_from_target_cooldown(self):
        frame = self._frame()
        desired = np.ones(len(frame))
        _, _, _, _, positions = account.simulate_account_path(
            frame,
            desired,
            np.zeros(len(frame), dtype="int32"),
            desired,
            {"mode": "none", "trigger": None, "scale": 0.0, "exit": 0.0},
            risk_profiles={
                0: {"stop_pct": 0.01, "target_pct": 0.03, "cooldown_bars": 1}
            },
            stop_cooldown_bars=3,
            round_trip_fee=0.0,
        )
        self.assertEqual(positions.tolist(), [1.0, 1.0, 0.0, 0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
