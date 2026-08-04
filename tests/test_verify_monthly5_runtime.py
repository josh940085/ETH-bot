import json
import tempfile
import unittest
from pathlib import Path

import monthly5_shadow
import verify_monthly5_runtime


class VerifyMonthly5RuntimeTests(unittest.TestCase):
    def _spec(self, summary_path):
        return {
            "strategy_id": monthly5_shadow.STRATEGY_ID,
            "objective": {
                "monthly_return_floor_pct": 5.0,
                "max_leverage": 5,
            },
            "policy": {
                "risk": {
                    "intraday_stop_pct": monthly5_shadow.INTRADAY_STOP_PCT,
                    "monthly_recovery_trigger_pct": monthly5_shadow.MONTHLY_RECOVERY_TRIGGER_PCT,
                    "monthly_lock_pct": monthly5_shadow.MONTHLY_LOCK_PCT,
                    "post_lock_exposure_scale": monthly5_shadow.POST_LOCK_EXPOSURE_SCALE,
                }
            },
            "backtest_evidence": {
                "source_summary": str(summary_path),
                "candidate_name": monthly5_shadow.SELECTED_CANDIDATE,
                "months_ge_5": 79,
                "months_ge_0": 80,
                "complete_months": 79,
                "complete_months_ge_5": 79,
                "incomplete_month": "2026-08",
                "worst_intramonth_pnl_pct": -10.32,
                "avg_flat_time_pct": 41.65,
            },
        }

    def _summary(self):
        return {
            "top": [
                {
                    "name": monthly5_shadow.SELECTED_CANDIDATE,
                    "months_ge_5": 79,
                    "months_ge_0": 80,
                    "worst_intramonth_pnl_pct": -10.32,
                    "avg_flat_time_pct": 41.65,
                    "failed_months": [{"month": "2026-08", "return_pct": 0.497}],
                }
            ]
        }

    def _shadow(self):
        return {
            "strategy_id": monthly5_shadow.STRATEGY_ID,
            "selected_candidate": monthly5_shadow.SELECTED_CANDIDATE,
            "shadow_only": True,
            "max_leverage": 5,
            "monthly_lock_pct": monthly5_shadow.MONTHLY_LOCK_PCT,
            "monthly_recovery_trigger_pct": monthly5_shadow.MONTHLY_RECOVERY_TRIGGER_PCT,
            "intraday_stop_pct": monthly5_shadow.INTRADAY_STOP_PCT,
            "post_lock_exposure_scale": monthly5_shadow.POST_LOCK_EXPOSURE_SCALE,
            "recovery_exposure_scale": monthly5_shadow.RECOVERY_EXPOSURE_SCALE,
            "mode": "normal",
            "updated_ts": 1000,
            "market_selection": {
                "selected_plan": "normal_long_selector",
                "shadow_action": "evaluate_long",
                "exposure_cap": 0.35,
                "max_leverage": 5,
            },
        }

    def test_runtime_verifier_accepts_matching_spec_summary_and_shadow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(self._summary()), encoding="utf-8")
            spec = self._spec(summary_path)
            shadow = self._shadow()

            failures = []
            failures.extend(verify_monthly5_runtime._verify_spec_and_summary(spec))
            failures.extend(verify_monthly5_runtime._verify_shadow_state("shadow", shadow, spec, None))
            failures.extend(verify_monthly5_runtime._verify_guard_scenarios())
            failures.extend(verify_monthly5_runtime._verify_end_to_end_scenarios())

        self.assertEqual(failures, [])

    def test_runtime_verifier_end_to_end_scenarios_pass(self):
        self.assertEqual(verify_monthly5_runtime._verify_end_to_end_scenarios(), [])

    def test_runtime_verifier_rejects_candidate_drift(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(self._summary()), encoding="utf-8")
            spec = self._spec(summary_path)
            spec["backtest_evidence"]["candidate_name"] = "wrong"

            failures = verify_monthly5_runtime._verify_spec_and_summary(spec)

        self.assertTrue(any("candidate" in failure for failure in failures))

    def test_runtime_verifier_rejects_missing_market_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_path = root / "summary.json"
            spec = self._spec(summary_path)
            shadow = self._shadow()
            shadow.pop("market_selection")

            failures = verify_monthly5_runtime._verify_shadow_state("shadow", shadow, spec, None)

        self.assertIn("shadow missing market_selection", failures)


if __name__ == "__main__":
    unittest.main()
