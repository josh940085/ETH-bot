import unittest

import numpy as np
import pandas as pd

import monthly5_experimental_candidate_shadow as shadow


def _bars_4h(n, start="2024-01-01", freq="4h"):
    return pd.date_range(start, periods=n, freq=freq, tz="UTC")


def _synthetic_frame(n, *, trend_pct_per_bar=0.0, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.001, size=n)
    returns = trend_pct_per_bar + noise
    close = 100.0 * np.cumprod(1.0 + returns)
    high = close * 1.002
    low = close * 0.998
    open_ = np.concatenate([[100.0], close[:-1]])
    volume = np.full(n, 1000.0)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=_bars_4h(n),
    )


class BuildPositionSeriesTests(unittest.TestCase):
    def test_confirms_after_required_consecutive_bars(self):
        labels = pd.Series(["up", "up", "up", "up"], index=range(4))
        position = shadow._build_position_series(labels, confirmation_bars=2, range_grace_bars=0)
        self.assertEqual(list(position), [0.0, 1.0, 1.0, 1.0])

    def test_zero_range_grace_flattens_on_first_range_bar(self):
        labels = pd.Series(["up", "up", "range"], index=range(3))
        position = shadow._build_position_series(labels, confirmation_bars=2, range_grace_bars=0)
        self.assertEqual(list(position), [0.0, 1.0, 0.0])

    def test_range_grace_delays_flattening(self):
        labels = pd.Series(["up", "range", "range", "range"], index=range(4))
        position = shadow._build_position_series(labels, confirmation_bars=1, range_grace_bars=2)
        self.assertEqual(list(position), [1.0, 1.0, 1.0, 0.0])

    def test_unknown_label_resets_immediately(self):
        labels = pd.Series(["up", "unknown", "up"], index=range(3))
        position = shadow._build_position_series(labels, confirmation_bars=1, range_grace_bars=6)
        self.assertEqual(list(position), [1.0, 0.0, 1.0])

    def test_direction_flip_requires_new_confirmation(self):
        labels = pd.Series(["up", "down", "down"], index=range(3))
        position = shadow._build_position_series(labels, confirmation_bars=2, range_grace_bars=0)
        self.assertEqual(list(position), [0.0, 0.0, -1.0])


class RecoveryHysteresisTests(unittest.TestCase):
    def test_arms_once_trigger_breached(self):
        self.assertFalse(shadow._next_recovery_active(False, -5.9))
        self.assertTrue(shadow._next_recovery_active(False, -6.0))
        self.assertTrue(shadow._next_recovery_active(False, -9.0))

    def test_stays_active_until_exit_level(self):
        self.assertTrue(shadow._next_recovery_active(True, -1.5))
        self.assertFalse(shadow._next_recovery_active(True, -1.0))
        self.assertFalse(shadow._next_recovery_active(True, 2.0))


class PaperSummaryTests(unittest.TestCase):
    def _rows(self, closes, signals):
        return [
            {
                "usable": True,
                "bar_close_ts_ms": idx * 4 * 3600 * 1000,
                "close": close,
                "candidate_signal": signal,
                "baseline_signal": signal,
            }
            for idx, (close, signal) in enumerate(zip(closes, signals))
        ]

    def test_long_signal_captures_upside(self):
        rows = self._rows([100.0, 110.0, 121.0], [1.0, 1.0, 0.0])
        summary = shadow._paper_summary(rows, "candidate_signal")
        self.assertAlmostEqual(summary["return_pct"], 20.0, places=4)
        self.assertEqual(summary["rows"], 2)

    def test_flat_signal_contributes_zero_and_counts_as_flat(self):
        rows = self._rows([100.0, 110.0], [0.0])
        summary = shadow._paper_summary(rows, "candidate_signal")
        self.assertEqual(summary["return_pct"], 0.0)
        self.assertEqual(summary["flat_time_pct"], 100.0)

    def test_month_to_date_only_sums_current_month_intervals(self):
        rows = [
            {
                "usable": True,
                "bar_close_ts_ms": int(pd.Timestamp("2026-07-31T20:00:00Z").timestamp() * 1000),
                "close": 100.0,
                "candidate_signal": 1.0,
            },
            {
                "usable": True,
                "bar_close_ts_ms": int(pd.Timestamp("2026-08-01T00:00:00Z").timestamp() * 1000),
                "close": 110.0,
                "candidate_signal": 1.0,
            },
            {
                "usable": True,
                "bar_close_ts_ms": int(pd.Timestamp("2026-08-01T04:00:00Z").timestamp() * 1000),
                "close": 121.0,
                "candidate_signal": 1.0,
            },
        ]
        # Both intervals resolve (their end timestamp falls) in August, even
        # though the first interval's signal was decided on the last July
        # bar - month-to-date is keyed by when the return realizes, not by
        # which bar produced the signal.
        mtd = shadow._month_to_date_return_pct(rows, "candidate_signal", "2026-08")
        self.assertAlmostEqual(mtd, 20.0, places=4)


class BuildLiveProbeTests(unittest.TestCase):
    def test_insufficient_history_is_not_usable(self):
        frame = _synthetic_frame(10)
        probe = shadow.build_live_probe(frame)
        self.assertFalse(probe["usable"])
        self.assertIn("completed_4h_history_insufficient", probe["blocking_reasons"])

    def test_missing_columns_is_not_usable(self):
        frame = _synthetic_frame(400).drop(columns=["volume"])
        probe = shadow.build_live_probe(frame)
        self.assertFalse(probe["usable"])
        self.assertIn("required_ohlcv_missing", probe["blocking_reasons"])

    def test_sufficient_history_produces_usable_probe(self):
        frame = _synthetic_frame(400, trend_pct_per_bar=0.0015, seed=1)
        probe = shadow.build_live_probe(frame)
        self.assertTrue(probe["usable"])
        self.assertIn(probe["walkforward_selected_config"], {c["name"] for c in shadow.PRIMARY_CONFIGS})
        self.assertIn(probe["candidate_signal"], {-1.0, 0.0, 1.0})
        self.assertEqual(probe["shadow_only"], True)
        self.assertEqual(probe["execution_allowed"], False)

    def test_low_volume_forces_flat_candidate_signal(self):
        frame = _synthetic_frame(400, trend_pct_per_bar=0.0015, seed=1)
        frame = frame.copy()
        # build_live_probe drops the frame's final (forming) bar, so the
        # last *completed* bar used in the volume gate is index -2.
        frame.iloc[-2, frame.columns.get_loc("volume")] = 0.0001
        probe = shadow.build_live_probe(frame)
        self.assertTrue(probe["usable"])
        self.assertFalse(probe["volume_allowed"])
        self.assertEqual(probe["candidate_signal"], 0.0)

    def test_recovery_active_uses_recovery_signal_source(self):
        frame = _synthetic_frame(400, trend_pct_per_bar=0.0015, seed=1)
        probe_normal = shadow.build_live_probe(frame, recovery_active=False)
        probe_recovery = shadow.build_live_probe(frame, recovery_active=True)
        self.assertEqual(probe_normal["recovery_active_applied"], False)
        self.assertEqual(probe_recovery["recovery_active_applied"], True)
        if probe_recovery["volume_allowed"]:
            self.assertEqual(probe_recovery["candidate_signal"], probe_recovery["recovery_signal"])
        if probe_normal["volume_allowed"]:
            self.assertEqual(probe_normal["candidate_signal"], probe_normal["walkforward_primary_signal"])


class SelectWalkforwardConfigTests(unittest.TestCase):
    def test_returns_one_of_the_candidate_configs(self):
        frame = _synthetic_frame(500, trend_pct_per_bar=0.001, seed=2)
        result = shadow.select_walkforward_config(frame)
        names = {c["name"] for c in shadow.PRIMARY_CONFIGS}
        self.assertIn(result["selected_config_name"], names)
        self.assertGreaterEqual(result["trailing_months_used"], 0)

    def test_short_history_is_low_confidence(self):
        frame = _synthetic_frame(35, seed=3)
        result = shadow.select_walkforward_config(frame)
        self.assertTrue(result["low_confidence"])


class UpdateStateBlockersTests(unittest.TestCase):
    def test_new_trial_has_span_and_row_blockers(self):
        probe = {
            "usable": True,
            "bar_close_ts_ms": int(pd.Timestamp("2026-08-10T00:00:00Z").timestamp() * 1000),
            "close": 100.0,
            "candidate_signal": 1.0,
            "baseline_signal": 1.0,
        }
        rows = [probe]
        state = shadow.update_state({}, rows, probe, now_ts=pd.Timestamp("2026-08-10T00:00:00Z").timestamp())
        self.assertFalse(state["promotion_ready"])
        self.assertIn("forward_span_lt_min_days", state["promotion_blockers"])
        self.assertIn("forward_rows_insufficient", state["promotion_blockers"])
        self.assertEqual(state["methodology_caveats"], shadow.METHODOLOGY_CAVEATS)

    def test_volume_component_blocker_reflects_research_validity_flag(self):
        probe = {
            "usable": True,
            "bar_close_ts_ms": int(pd.Timestamp("2026-08-10T00:00:00Z").timestamp() * 1000),
            "close": 100.0,
            "candidate_signal": 0.0,
            "baseline_signal": 0.0,
        }
        state = shadow.update_state({}, [probe], probe, now_ts=pd.Timestamp("2026-08-10T00:00:00Z").timestamp())
        self.assertEqual(state["volume_component_research_valid"], False)
        self.assertIn("volume_component_research_invalidated_forward_pending", state["promotion_blockers"])


if __name__ == "__main__":
    unittest.main()
