import unittest

import monthly5_shadow
import verify_monthly5_readiness


class VerifyMonthly5ReadinessTests(unittest.TestCase):
    def _spec(self, avg_flat_time_pct=None):
        spec = {
            "strategy_id": monthly5_shadow.STRATEGY_ID,
            "backtest_evidence": {
                "candidate_name": monthly5_shadow.SELECTED_CANDIDATE,
            },
        }
        if avg_flat_time_pct is not None:
            spec["backtest_evidence"]["avg_flat_time_pct"] = avg_flat_time_pct
        return spec

    def _row(
        self,
        ts,
        *,
        action="evaluate_long",
        plan="normal_long_selector",
        position_open=False,
        mark_price=100.0,
        exposure_cap=0.35,
        max_leverage=5,
    ):
        return {
            "schema_version": 1,
            "strategy_id": monthly5_shadow.STRATEGY_ID,
            "selected_candidate": monthly5_shadow.SELECTED_CANDIDATE,
            "shadow_only": True,
            "updated_ts": ts,
            "month_key": "2026-08",
            "day_key": "2026-08-05",
            "mode": "normal",
            "max_leverage": max_leverage,
            "exposure_cap": exposure_cap,
            "selected_plan": plan,
            "shadow_action": action,
            "market_bias": "bullish",
            "market_state": "chop",
            "guard_allowed": True,
            "position_open": position_open,
            "position_notional": 100.0 if position_open else 0.0,
            "mark_price": mark_price,
        }

    def test_collecting_when_history_is_valid_but_small(self):
        rows = [self._row(1000, action="wait", plan="normal_wait")]

        report = verify_monthly5_readiness._history_readiness(
            rows,
            self._spec(),
            min_records=5,
            min_span_hours=1.0,
            max_age_sec=None,
        )

        self.assertEqual(report["status"], "collecting")
        self.assertFalse(report["ready"])
        self.assertEqual(report["failures"], [])
        self.assertTrue(any("sample count" in item for item in report["warnings"]))

    def test_ready_when_history_has_enough_span_and_evaluation_samples(self):
        rows = [self._row(1000 + idx * 900) for idx in range(5)]

        report = verify_monthly5_readiness._history_readiness(
            rows,
            self._spec(),
            min_records=5,
            min_span_hours=1.0,
            max_age_sec=None,
        )

        self.assertEqual(report["status"], "ready")
        self.assertTrue(report["ready"])
        self.assertEqual(report["evaluate_rows"], 5)

    def test_collecting_when_weighted_shadow_flat_time_exceeds_backtest_cap(self):
        rows = [
            self._row(1000, action="wait", plan="normal_wait", position_open=False),
            self._row(1900, action="evaluate_long", plan="normal_long_selector", position_open=True),
            self._row(2800, action="evaluate_long", plan="normal_long_selector", position_open=False),
            self._row(3700, action="evaluate_long", plan="normal_long_selector", position_open=False),
            self._row(4600, action="evaluate_long", plan="normal_long_selector", position_open=False),
        ]

        report = monthly5_shadow.build_readiness_report(
            rows,
            strategy_id=monthly5_shadow.STRATEGY_ID,
            selected_candidate=monthly5_shadow.SELECTED_CANDIDATE,
            min_records=5,
            min_span_hours=1.0,
            max_age_sec=None,
            max_flat_time_pct=20.0,
            now_ts=1000 + 4 * 900,
        )

        self.assertEqual(report["status"], "collecting")
        self.assertFalse(report["ready"])
        self.assertEqual(report["flat_sample_pct"], 80.0)
        self.assertEqual(report["shadow_flat_time_pct"], 25.0)
        self.assertTrue(any("shadow flat time pct high" in item for item in report["warnings"]))

    def test_flat_time_is_weighted_by_timestamp_span(self):
        rows = [
            self._row(1000, position_open=True),
            self._row(1060, position_open=False),
            self._row(1600, position_open=False),
        ]

        report = monthly5_shadow.build_readiness_report(
            rows,
            strategy_id=monthly5_shadow.STRATEGY_ID,
            selected_candidate=monthly5_shadow.SELECTED_CANDIDATE,
            min_records=3,
            min_span_hours=0.1,
            max_age_sec=None,
            max_flat_time_pct=90.0,
            now_ts=1600,
        )

        self.assertAlmostEqual(report["flat_sample_pct"], 66.6667, places=3)
        self.assertAlmostEqual(report["flat_time_pct"], 90.0, places=3)

    def test_actual_flat_time_does_not_block_shadow_readiness(self):
        rows = [self._row(1000 + idx * 900, position_open=False) for idx in range(5)]

        report = monthly5_shadow.build_readiness_report(
            rows,
            strategy_id=monthly5_shadow.STRATEGY_ID,
            selected_candidate=monthly5_shadow.SELECTED_CANDIDATE,
            min_records=5,
            min_span_hours=1.0,
            max_age_sec=None,
            max_flat_time_pct=50.0,
            now_ts=1000 + 4 * 900,
        )

        self.assertEqual(report["status"], "ready")
        self.assertTrue(report["ready"])
        self.assertEqual(report["actual_flat_time_pct"], 100.0)
        self.assertEqual(report["shadow_flat_time_pct"], 0.0)

    def test_shadow_paper_return_tracks_direction_exposure_and_leverage(self):
        rows = [
            self._row(1000, action="evaluate_long", plan="normal_long_selector", mark_price=100.0, exposure_cap=0.5, max_leverage=2),
            self._row(1900, action="wait", plan="normal_wait", mark_price=110.0),
            self._row(2800, action="evaluate_short", plan="normal_short_selector", mark_price=105.0, exposure_cap=0.2, max_leverage=5),
            self._row(3700, action="wait", plan="normal_wait", mark_price=100.0),
        ]

        report = monthly5_shadow.build_readiness_report(
            rows,
            strategy_id=monthly5_shadow.STRATEGY_ID,
            selected_candidate=monthly5_shadow.SELECTED_CANDIDATE,
            min_records=4,
            min_span_hours=0.5,
            max_age_sec=None,
            now_ts=3700,
        )

        self.assertEqual(report["shadow_paper_intervals"], 2)
        self.assertEqual(report["shadow_paper_long_intervals"], 1)
        self.assertEqual(report["shadow_paper_short_intervals"], 1)
        self.assertAlmostEqual(report["shadow_paper_return_pct"], 14.7619, places=4)
        self.assertEqual(report["shadow_paper_win_rate_pct"], 100.0)
        self.assertTrue(report["shadow_monthly_target_met"])
        self.assertGreater(report["shadow_projected_monthly_return_pct"], 5.0)

    def test_shadow_monthly_projection_warns_after_enough_span_when_below_target(self):
        rows = [
            self._row(1000, action="evaluate_long", plan="normal_long_selector", mark_price=100.0, exposure_cap=0.1, max_leverage=1),
            self._row(1000 + 24 * 3600, action="wait", plan="normal_wait", mark_price=100.1),
        ]

        report = monthly5_shadow.build_readiness_report(
            rows,
            strategy_id=monthly5_shadow.STRATEGY_ID,
            selected_candidate=monthly5_shadow.SELECTED_CANDIDATE,
            min_records=2,
            min_span_hours=24.0,
            max_age_sec=None,
            now_ts=1000 + 24 * 3600,
        )

        self.assertFalse(report["shadow_monthly_target_met"])
        self.assertAlmostEqual(report["shadow_projected_monthly_return_pct"], 0.3, places=4)
        self.assertTrue(any("shadow monthly projection below target" in item for item in report["warnings"]))

    def test_invalid_when_history_breaks_shadow_safety(self):
        row = self._row(1000)
        row["max_leverage"] = 6

        report = verify_monthly5_readiness._history_readiness(
            [row],
            self._spec(),
            min_records=1,
            min_span_hours=0.0,
            max_age_sec=None,
        )

        self.assertEqual(report["status"], "invalid")
        self.assertTrue(any("unsafe shadow rows" in item for item in report["failures"]))


if __name__ == "__main__":
    unittest.main()
