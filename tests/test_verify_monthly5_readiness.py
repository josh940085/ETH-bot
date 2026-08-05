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
        self.assertTrue(report["shadow_monthly_projection_valid"])
        self.assertTrue(report["shadow_monthly_target_met"])
        self.assertTrue(report["promotion_ready"])
        self.assertGreater(report["shadow_projected_monthly_return_pct"], 5.0)

    def test_shadow_monthly_target_requires_projection_span(self):
        rows = [
            self._row(1000, action="evaluate_long", plan="normal_long_selector", mark_price=100.0, exposure_cap=1.0, max_leverage=5),
            self._row(1900, action="wait", plan="normal_wait", mark_price=110.0),
        ]

        report = monthly5_shadow.build_readiness_report(
            rows,
            strategy_id=monthly5_shadow.STRATEGY_ID,
            selected_candidate=monthly5_shadow.SELECTED_CANDIDATE,
            min_records=2,
            min_span_hours=24.0,
            max_age_sec=None,
            now_ts=1900,
        )

        self.assertFalse(report["shadow_monthly_projection_valid"])
        self.assertFalse(report["shadow_monthly_target_met"])
        self.assertFalse(report["promotion_ready"])
        self.assertGreater(report["shadow_projected_monthly_return_pct"], 5.0)

    def test_promotion_ready_requires_readiness_and_monthly_target(self):
        rows = [
            self._row(1000, action="evaluate_long", plan="normal_long_selector", mark_price=100.0, exposure_cap=0.1, max_leverage=5),
            self._row(1000 + 24 * 3600, action="wait", plan="normal_wait", mark_price=110.0),
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

        self.assertTrue(report["ready"])
        self.assertTrue(report["shadow_monthly_projection_valid"])
        self.assertTrue(report["shadow_monthly_target_met"])
        self.assertTrue(report["shadow_rolling_monthly_projection_valid"])
        self.assertTrue(report["shadow_rolling_monthly_target_met"])
        self.assertTrue(report["promotion_ready"])

    def test_promotion_ready_requires_rolling_24h_target(self):
        rows = [
            self._row(1000, action="evaluate_long", plan="normal_long_selector", mark_price=100.0, exposure_cap=0.1, max_leverage=5),
            self._row(1000 + 24 * 3600, action="evaluate_long", plan="normal_long_selector", mark_price=120.0, exposure_cap=0.1, max_leverage=5),
            self._row(1000 + 48 * 3600, action="wait", plan="normal_wait", mark_price=119.0),
        ]

        report = monthly5_shadow.build_readiness_report(
            rows,
            strategy_id=monthly5_shadow.STRATEGY_ID,
            selected_candidate=monthly5_shadow.SELECTED_CANDIDATE,
            min_records=3,
            min_span_hours=24.0,
            max_age_sec=None,
            now_ts=1000 + 48 * 3600,
        )

        self.assertTrue(report["ready"])
        self.assertTrue(report["shadow_monthly_target_met"])
        self.assertTrue(report["shadow_rolling_monthly_projection_valid"])
        self.assertFalse(report["shadow_rolling_monthly_target_met"])
        self.assertFalse(report["promotion_ready"])
        self.assertTrue(any("shadow rolling 24h projection below target" in item for item in report["warnings"]))

    def test_grouped_paper_return_flags_underperforming_plan(self):
        rows = [
            self._row(1000 + idx * 60, action="evaluate_long", plan="normal_long_selector", mark_price=100.0 - idx, exposure_cap=0.5, max_leverage=1)
            for idx in range(13)
        ]

        report = monthly5_shadow.build_readiness_report(
            rows,
            strategy_id=monthly5_shadow.STRATEGY_ID,
            selected_candidate=monthly5_shadow.SELECTED_CANDIDATE,
            min_records=13,
            min_span_hours=0.1,
            max_age_sec=None,
            now_ts=1000 + 12 * 60,
        )

        self.assertGreaterEqual(report["shadow_underperforming_plan_count"], 1)
        self.assertIn(
            "normal_long_selector|evaluate_long|bullish|chop",
            report["shadow_underperforming_plan_keys"],
        )
        weakest = report["shadow_grouped_paper_returns"][0]
        self.assertEqual(weakest["selected_plan"], "normal_long_selector")
        self.assertLess(weakest["return_pct"], 0.0)

    def test_active_underperforming_plan_expires_after_rolling_window(self):
        weak_rows = [
            self._row(1000 + idx * 60, action="evaluate_long", plan="normal_long_selector", mark_price=100.0 - idx, exposure_cap=0.5, max_leverage=1)
            for idx in range(13)
        ]
        later_rows = [
            self._row(1000 + 48 * 3600 + idx * 60, action="wait", plan="normal_wait", mark_price=90.0)
            for idx in range(2)
        ]

        report = monthly5_shadow.build_readiness_report(
            weak_rows + later_rows,
            strategy_id=monthly5_shadow.STRATEGY_ID,
            selected_candidate=monthly5_shadow.SELECTED_CANDIDATE,
            min_records=15,
            min_span_hours=0.1,
            max_age_sec=None,
            now_ts=1000 + 48 * 3600 + 60,
        )

        self.assertGreaterEqual(report["shadow_underperforming_plan_count"], 1)
        self.assertEqual(report["shadow_active_underperforming_plan_count"], 0)
        self.assertEqual(report["shadow_active_underperforming_plan_keys"], [])

    def test_suppressed_recovering_plan_removes_active_underperforming_key(self):
        rows = []
        for idx in range(13):
            row = self._row(
                1000 + idx * 60,
                action="evaluate_long",
                plan="normal_long_selector",
                mark_price=100.0 - idx,
                exposure_cap=0.5,
                max_leverage=1,
            )
            rows.append(row)
        base_ts = 1000 + 3600
        for idx in range(7):
            row = self._row(
                base_ts + idx * 60,
                action="wait",
                plan="underperforming_wait",
                mark_price=90.0 + idx,
                exposure_cap=0.0,
                max_leverage=1,
            )
            row["suppressed_plan"] = "normal_long_selector"
            row["suppressed_action"] = "evaluate_long"
            row["suppressed_key"] = "normal_long_selector|evaluate_long|bullish|chop"
            row["suppressed_exposure_cap"] = 0.5
            rows.append(row)

        report = monthly5_shadow.build_readiness_report(
            rows,
            strategy_id=monthly5_shadow.STRATEGY_ID,
            selected_candidate=monthly5_shadow.SELECTED_CANDIDATE,
            min_records=20,
            min_span_hours=0.1,
            max_age_sec=None,
            now_ts=base_ts + 6 * 60,
        )

        self.assertIn(
            "normal_long_selector|evaluate_long|bullish|chop",
            report["shadow_suppressed_recovering_plan_keys"],
        )
        self.assertEqual(report["shadow_active_underperforming_plan_count"], 0)
        self.assertGreaterEqual(report["shadow_suppressed_observed_intervals"], 6)
        self.assertEqual(report["shadow_suppressed_recovery_remaining_intervals"], 0)

    def test_suppressed_recovery_reports_remaining_intervals_before_probe(self):
        rows = []
        base_ts = 1000
        for idx in range(4):
            row = self._row(
                base_ts + idx * 60,
                action="wait",
                plan="underperforming_wait",
                mark_price=90.0 + idx,
                exposure_cap=0.0,
                max_leverage=1,
            )
            row["suppressed_plan"] = "normal_long_selector"
            row["suppressed_action"] = "evaluate_long"
            row["suppressed_key"] = "normal_long_selector|evaluate_long|bullish|chop"
            row["suppressed_exposure_cap"] = 0.5
            rows.append(row)

        report = monthly5_shadow.build_readiness_report(
            rows,
            strategy_id=monthly5_shadow.STRATEGY_ID,
            selected_candidate=monthly5_shadow.SELECTED_CANDIDATE,
            min_records=4,
            min_span_hours=0.01,
            max_age_sec=None,
            now_ts=base_ts + 3 * 60,
        )

        self.assertEqual(report["shadow_suppressed_observed_intervals"], 3)
        self.assertEqual(
            report["shadow_suppressed_recovery_remaining_intervals"],
            monthly5_shadow.SUPPRESSED_RECOVERY_MIN_INTERVALS - 3,
        )
        self.assertEqual(report["shadow_recovery_probe_state"], "collecting")

    def test_recovery_probe_success_and_failure_keys(self):
        success_rows = []
        for idx in range(7):
            row = self._row(
                1000 + idx * 60,
                action="evaluate_long",
                plan="normal_long_selector",
                mark_price=100.0 + idx,
                exposure_cap=monthly5_shadow.RECOVERY_PROBE_EXPOSURE_CAP,
                max_leverage=1,
            )
            row["recovery_probe"] = True
            row["recovery_probe_key"] = "normal_long_selector|evaluate_long|bullish|chop"
            success_rows.append(row)

        success = monthly5_shadow.build_readiness_report(
            success_rows,
            strategy_id=monthly5_shadow.STRATEGY_ID,
            selected_candidate=monthly5_shadow.SELECTED_CANDIDATE,
            min_records=7,
            min_span_hours=0.01,
            max_age_sec=None,
            now_ts=1000 + 6 * 60,
        )

        self.assertIn(
            "normal_long_selector|evaluate_long|bullish|chop",
            success["shadow_recovery_probe_success_keys"],
        )
        self.assertEqual(success["shadow_recovery_probe_state"], "probe_success")
        self.assertEqual(success["shadow_recovery_probe_failed_count"], 0)

        failure_rows = []
        for idx in range(7):
            row = self._row(
                2000 + idx * 60,
                action="evaluate_long",
                plan="normal_long_selector",
                mark_price=100.0 - idx,
                exposure_cap=monthly5_shadow.RECOVERY_PROBE_EXPOSURE_CAP,
                max_leverage=1,
            )
            row["recovery_probe"] = True
            row["recovery_probe_key"] = "normal_long_selector|evaluate_long|bullish|chop"
            failure_rows.append(row)

        failure = monthly5_shadow.build_readiness_report(
            failure_rows,
            strategy_id=monthly5_shadow.STRATEGY_ID,
            selected_candidate=monthly5_shadow.SELECTED_CANDIDATE,
            min_records=7,
            min_span_hours=0.01,
            max_age_sec=None,
            now_ts=2000 + 6 * 60,
        )

        self.assertIn(
            "normal_long_selector|evaluate_long|bullish|chop",
            failure["shadow_recovery_probe_failed_keys"],
        )
        self.assertEqual(failure["shadow_recovery_probe_state"], "probe_failed")
        self.assertIn(
            "normal_long_selector|evaluate_long|bullish|chop",
            failure["shadow_active_underperforming_plan_keys"],
        )

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
