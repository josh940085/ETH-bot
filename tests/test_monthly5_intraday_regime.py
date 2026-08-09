import unittest

import numpy as np
import pandas as pd

import monthly5_intraday_regime


class Monthly5IntradayRegimeTests(unittest.TestCase):
    def test_completed_4h_label_is_delayed_one_5m_bar(self):
        index = pd.date_range("2026-01-01", periods=400, freq="5min", tz="UTC")
        close = np.linspace(100.0, 130.0, len(index))
        frame = pd.DataFrame(
            {"open": close, "high": close + 0.1, "low": close - 0.1, "close": close, "volume": 1.0},
            index=index,
        )
        labels = monthly5_intraday_regime.classify_completed_4h(
            frame, distance_threshold=0.003, slope_threshold=0.0005
        )
        self.assertTrue(all(timestamp.minute == 5 for timestamp in labels.index))

    def test_regime_position_respects_up_and_down_direction(self):
        index = pd.date_range("2026-01-01T04:05:00Z", periods=4, freq="5min")
        frame = pd.DataFrame({"close": [100.0] * 4}, index=index)
        labels = pd.Series(["up", "down"], index=[index[0], index[2]])
        rsi = pd.Series([50.0], index=[index[0]])
        position, _ = monthly5_intraday_regime.build_regime_position(
            frame, labels, rsi, range_mode="flat", rsi_low=0.0, rsi_high=0.0
        )
        self.assertEqual(position.tolist(), [1.0, 1.0, -1.0, -1.0])

    def test_strict_daily_alignment_filters_conflicting_4h_direction(self):
        index = pd.date_range("2026-01-01T04:05:00Z", periods=4, freq="5min")
        frame = pd.DataFrame({"close": [100.0] * 4}, index=index)
        labels = pd.Series(["up", "down"], index=[index[0], index[2]])
        daily = pd.Series(["down", "down"], index=[index[0], index[2]])
        position, _ = monthly5_intraday_regime.build_regime_position(
            frame,
            labels,
            pd.Series([50.0], index=[index[0]]),
            range_mode="flat",
            rsi_low=0.0,
            rsi_high=0.0,
            daily_trend=daily,
            daily_alignment_mode="strict",
        )
        self.assertEqual(position.tolist(), [0.0, 0.0, -1.0, -1.0])

    def test_monthly_floor_turns_exposure_off_after_rollback(self):
        index = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
        frame = pd.DataFrame(index=index)
        factors, scales = monthly5_intraday_regime.apply_monthly_lock(
            frame,
            pnl=np.array([0.06, -0.10, 0.10, 0.10]),
            turnover=np.zeros(4),
            actual=np.ones(4),
            leverage=1,
            lock_scale=0.15,
            lock_trigger=0.05,
            lock_floor=0.05,
            monthly_stop=-0.50,
        )
        self.assertEqual(scales.tolist(), [1.0, 0.15, 0.0, 0.0])
        self.assertLess(factors[2], 1.0)
        self.assertAlmostEqual(factors[3], 1.0)

    def test_monthly_recovery_uses_low_scale_until_exit(self):
        index = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
        frame = pd.DataFrame(index=index)
        _, scales = monthly5_intraday_regime.apply_monthly_lock(
            frame,
            pnl=np.array([-0.09, 0.10, 0.10, 0.01]),
            turnover=np.zeros(4),
            actual=np.zeros(4),
            leverage=1,
            lock_scale=0.15,
            daily_stop=None,
            monthly_stop=-0.08,
            monthly_recovery_scale=0.5,
            recovery_exit=-0.03,
        )
        self.assertEqual(scales.tolist(), [1.0, 0.5, 0.5, 1.0])

    def test_zero_recovery_scale_preserves_monthly_hard_stop(self):
        index = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
        frame = pd.DataFrame(index=index)
        _, scales = monthly5_intraday_regime.apply_monthly_lock(
            frame,
            pnl=np.array([-0.09, 0.10, 0.10]),
            turnover=np.zeros(3),
            actual=np.zeros(3),
            leverage=1,
            lock_scale=0.15,
            daily_stop=None,
            monthly_stop=-0.08,
            monthly_recovery_scale=0.0,
        )
        self.assertEqual(scales.tolist(), [1.0, 0.0, 0.0])

    def test_alternate_recovery_path_is_used_after_trigger(self):
        index = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
        frame = pd.DataFrame(index=index)
        factors, scales = monthly5_intraday_regime.apply_monthly_lock(
            frame,
            pnl=np.array([-0.09, -0.10, -0.10]),
            turnover=np.zeros(3),
            actual=np.zeros(3),
            leverage=1,
            lock_scale=0.15,
            daily_stop=None,
            monthly_stop=-0.08,
            monthly_recovery_scale=0.5,
            recovery_exit=0.0,
            recovery_pnl=np.array([0.0, 0.10, 0.10]),
            recovery_turnover=np.zeros(3),
            recovery_actual=np.zeros(3),
        )
        self.assertEqual(scales.tolist(), [1.0, 0.5, 0.5])
        self.assertGreater(factors[1], 1.0)


if __name__ == "__main__":
    unittest.main()
