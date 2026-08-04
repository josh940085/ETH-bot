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

    def _row(self, ts, *, action="evaluate_long", plan="normal_long_selector", position_open=False):
        return {
            "schema_version": 1,
            "strategy_id": monthly5_shadow.STRATEGY_ID,
            "selected_candidate": monthly5_shadow.SELECTED_CANDIDATE,
            "shadow_only": True,
            "updated_ts": ts,
            "month_key": "2026-08",
            "day_key": "2026-08-05",
            "mode": "normal",
            "max_leverage": 5,
            "exposure_cap": 0.35,
            "selected_plan": plan,
            "shadow_action": action,
            "market_bias": "bullish",
            "market_state": "chop",
            "guard_allowed": True,
            "position_open": position_open,
            "position_notional": 100.0 if position_open else 0.0,
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

    def test_collecting_when_flat_time_exceeds_backtest_cap(self):
        rows = [self._row(1000 + idx * 900, position_open=(idx == 0)) for idx in range(5)]

        report = verify_monthly5_readiness._history_readiness(
            rows,
            self._spec(avg_flat_time_pct=50.0),
            min_records=5,
            min_span_hours=1.0,
            max_age_sec=None,
        )

        self.assertEqual(report["status"], "collecting")
        self.assertFalse(report["ready"])
        self.assertEqual(report["flat_sample_pct"], 80.0)
        self.assertTrue(any("flat sample pct high" in item for item in report["warnings"]))

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
