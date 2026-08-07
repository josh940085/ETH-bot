import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import monthly5_research_selector
import monthly5_shadow
import verify_monthly5_runtime


class VerifyMonthly5RuntimeTests(unittest.TestCase):
    def _spec(self, summary_path, monthly_path):
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
                "source_monthly": str(monthly_path),
                "candidate_name": monthly5_shadow.SELECTED_CANDIDATE,
                "period_start": "2020-01",
                "period_end": "2020-03",
                "complete_months_end": "2020-02",
                "months": 3,
                "months_ge_5": 2,
                "months_ge_0": 3,
                "complete_months": 2,
                "complete_months_ge_5": 2,
                "incomplete_month": "2020-03",
                "worst_intramonth_pnl_pct": -4.0,
                "avg_flat_time_pct": 20.0,
            },
        }

    def _summary(self):
        return {
            "top": [
                {
                    "name": monthly5_shadow.SELECTED_CANDIDATE,
                    "months_ge_5": 2,
                    "months_ge_0": 3,
                    "worst_intramonth_pnl_pct": -4.0,
                    "avg_flat_time_pct": 20.0,
                    "failed_months": [{"month": "2020-03", "return_pct": 1.0}],
                }
            ]
        }

    def _monthly(self):
        return {
            monthly5_shadow.SELECTED_CANDIDATE: [
                {
                    "month": "2020-01",
                    "return_pct": 5.0,
                    "flat_time_pct": 10.0,
                    "min_intramonth_pnl_pct": -1.0,
                    "top_pick": "mom72_ls|lev4|stopNone|target0.05|redlev0.5",
                },
                {
                    "month": "2020-02",
                    "return_pct": 8.0,
                    "flat_time_pct": 20.0,
                    "min_intramonth_pnl_pct": -4.0,
                    "top_pick": "mom48_lf|lev5|stopNone|target0.08|redlev1.0",
                },
                {
                    "month": "2020-03",
                    "return_pct": 1.0,
                    "flat_time_pct": 30.0,
                    "min_intramonth_pnl_pct": 0.0,
                    "top_pick": "mom120_lf|lev4|stopNone|target0.05|redlev0.5",
                },
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
            "promotion_ready": False,
            "promotion_blockers": ["sample_span"],
            "market_selection": {
                "selector_policy_version": monthly5_shadow.SELECTOR_POLICY_VERSION,
                "selector_source": monthly5_shadow.RESEARCH_SELECTOR_SOURCE,
                "selector_policy_source": monthly5_shadow.RESEARCH_SELECTOR_SOURCE,
                "selector_alignment": "live_similar_day",
                "selector_key": "mom120_lf|lev4|stopNone|target0.05|redlev0.5",
                "selector_primary_direction": "long",
                "selected_plan": "normal_long_selector",
                "shadow_action": "evaluate_long",
                "exposure_cap": 0.35,
                "max_leverage": 4,
            },
        }

    def _live_selector_input(self):
        return {
            "selector_source": monthly5_shadow.RESEARCH_SELECTOR_SOURCE,
            "input_source": "live_daily_kline",
            "feature_set": "short_market_state",
            "usable": True,
            "blocking_reasons": [],
            "daily_rows": monthly5_research_selector.REQUIRED_LIVE_DAILY_ROWS,
            "required_daily_rows": monthly5_research_selector.REQUIRED_LIVE_DAILY_ROWS,
            "latest_daily_key": "2026-08-02",
            "missing_columns": [],
            "cache_available": True,
            "cache_feature_count": monthly5_research_selector.EXPECTED_SHORT_MARKET_STATE_FEATURES,
            "expected_feature_count": monthly5_research_selector.EXPECTED_SHORT_MARKET_STATE_FEATURES,
            "cache_candidate_count": 220,
            "cache_day_count": 2407,
            "cache_latest_day": "2026-08-03",
        }

    def _live_selector_decision(self):
        return {
            "selector_source": monthly5_shadow.RESEARCH_SELECTOR_SOURCE,
            "usable": True,
            "blocking_reasons": [],
            "feature_set": "short_market_state",
            "selected_key": "mom120_lf|lev4|stopNone|target0.05|redlev0.5",
            "primary_direction": "long",
            "max_leverage": 4,
            "selected_q25_return_pct": 5.5,
            "selected_hit_rate": 0.75,
        }

    def _history_row(self, shadow=None):
        shadow = shadow or self._shadow()
        return monthly5_shadow.build_history_record(
            shadow,
            {
                "allowed": True,
                "reason_code": "allowed",
                "adjusted_size": 0.0,
            },
        )

    def test_runtime_verifier_accepts_matching_spec_summary_and_shadow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_path = root / "summary.json"
            monthly_path = root / "monthly.json"
            summary_path.write_text(json.dumps(self._summary()), encoding="utf-8")
            monthly_path.write_text(json.dumps(self._monthly()), encoding="utf-8")
            spec = self._spec(summary_path, monthly_path)
            shadow = self._shadow()

            failures = []
            failures.extend(verify_monthly5_runtime._verify_spec_and_summary(spec))
            failures.extend(verify_monthly5_runtime._verify_shadow_state("shadow", shadow, spec, None))
            failures.extend(verify_monthly5_runtime._verify_shadow_history("history", [self._history_row(shadow)], shadow, spec, None))
            failures.extend(verify_monthly5_runtime._verify_guard_scenarios())
            failures.extend(verify_monthly5_runtime._verify_promotion_gate({"monthly5_shadow": shadow}))
            failures.extend(verify_monthly5_runtime._verify_end_to_end_scenarios())

        self.assertEqual(failures, [])

    def test_runtime_verifier_end_to_end_scenarios_pass(self):
        self.assertEqual(verify_monthly5_runtime._verify_end_to_end_scenarios(), [])

    def test_runtime_verifier_accepts_matching_shadow_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_path = root / "summary.json"
            monthly_path = root / "monthly.json"
            spec = self._spec(summary_path, monthly_path)
            shadow = self._shadow()

            failures = verify_monthly5_runtime._verify_shadow_history(
                "history",
                [self._history_row(shadow)],
                shadow,
                spec,
                None,
            )

        self.assertEqual(failures, [])

    def test_runtime_verifier_rejects_shadow_history_mode_drift(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_path = root / "summary.json"
            monthly_path = root / "monthly.json"
            spec = self._spec(summary_path, monthly_path)
            shadow = self._shadow()
            row = self._history_row(shadow)
            row["mode"] = "post_lock"

            failures = verify_monthly5_runtime._verify_shadow_history("history", [row], shadow, spec, None)

        self.assertTrue(any("mode does not match" in failure for failure in failures))

    def test_runtime_verifier_rejects_candidate_drift(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_path = root / "summary.json"
            monthly_path = root / "monthly.json"
            summary_path.write_text(json.dumps(self._summary()), encoding="utf-8")
            monthly_path.write_text(json.dumps(self._monthly()), encoding="utf-8")
            spec = self._spec(summary_path, monthly_path)
            spec["backtest_evidence"]["candidate_name"] = "wrong"

            failures = verify_monthly5_runtime._verify_spec_and_summary(spec)

        self.assertTrue(any("candidate" in failure for failure in failures))

    def test_runtime_verifier_rejects_missing_market_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_path = root / "summary.json"
            monthly_path = root / "monthly.json"
            spec = self._spec(summary_path, monthly_path)
            shadow = self._shadow()
            shadow.pop("market_selection")

            failures = verify_monthly5_runtime._verify_shadow_state("shadow", shadow, spec, None)

        self.assertIn("shadow missing market_selection", failures)

    def test_runtime_verifier_accepts_live_selector_input_probe(self):
        spec = self._spec("summary.json", "monthly.json")
        position = {
            "monthly5_shadow": self._shadow(),
            "monthly5_live_selector_input": self._live_selector_input(),
            "monthly5_live_selector_decision": self._live_selector_decision(),
        }
        probe = {
            "artifact_available": True,
            "selected_candidate": monthly5_shadow.SELECTED_CANDIDATE,
            "selector_source": monthly5_shadow.RESEARCH_SELECTOR_SOURCE,
            "max_leverage": 4,
            "primary_direction": "long",
            "top_pick": "mom120_lf|lev4|stopNone|target0.05|redlev0.5",
        }

        with unittest.mock.patch.object(
            verify_monthly5_runtime.monthly5_research_selector,
            "build_research_selector_probe",
            return_value=probe,
        ):
            failures = verify_monthly5_runtime._verify_research_selector_artifact(position, spec)

        self.assertEqual(failures, [])

    def test_runtime_verifier_rejects_unusable_live_selector_input_probe(self):
        spec = self._spec("summary.json", "monthly.json")
        live_input = self._live_selector_input()
        live_input["usable"] = False
        live_input["blocking_reasons"] = ["daily_warmup_insufficient"]
        position = {
            "monthly5_shadow": self._shadow(),
            "monthly5_live_selector_input": live_input,
            "monthly5_live_selector_decision": self._live_selector_decision(),
        }
        probe = {
            "artifact_available": True,
            "selected_candidate": monthly5_shadow.SELECTED_CANDIDATE,
            "selector_source": monthly5_shadow.RESEARCH_SELECTOR_SOURCE,
            "max_leverage": 4,
            "primary_direction": "long",
            "top_pick": "mom120_lf|lev4|stopNone|target0.05|redlev0.5",
        }

        with unittest.mock.patch.object(
            verify_monthly5_runtime.monthly5_research_selector,
            "build_research_selector_probe",
            return_value=probe,
        ):
            failures = verify_monthly5_runtime._verify_research_selector_artifact(position, spec)

        self.assertIn("monthly5 live selector input not usable", failures)

    def test_runtime_verifier_rejects_unusable_live_selector_decision(self):
        spec = self._spec("summary.json", "monthly.json")
        live_decision = self._live_selector_decision()
        live_decision["usable"] = False
        position = {
            "monthly5_shadow": self._shadow(),
            "monthly5_live_selector_input": self._live_selector_input(),
            "monthly5_live_selector_decision": live_decision,
        }
        probe = {
            "artifact_available": True,
            "selected_candidate": monthly5_shadow.SELECTED_CANDIDATE,
            "selector_source": monthly5_shadow.RESEARCH_SELECTOR_SOURCE,
            "max_leverage": 4,
            "primary_direction": "long",
            "top_pick": "mom120_lf|lev4|stopNone|target0.05|redlev0.5",
        }

        with unittest.mock.patch.object(
            verify_monthly5_runtime.monthly5_research_selector,
            "build_research_selector_probe",
            return_value=probe,
        ):
            failures = verify_monthly5_runtime._verify_research_selector_artifact(position, spec)

        self.assertIn("monthly5 live selector decision not usable", failures)

    def test_runtime_verifier_rejects_live_selector_key_mismatch(self):
        spec = self._spec("summary.json", "monthly.json")
        shadow = self._shadow()
        shadow["market_selection"]["selector_key"] = "mom48_lf|lev5|stopNone|target0.05|redlev0.5"
        position = {
            "monthly5_shadow": shadow,
            "monthly5_live_selector_input": self._live_selector_input(),
            "monthly5_live_selector_decision": self._live_selector_decision(),
        }
        probe = {
            "artifact_available": True,
            "selected_candidate": monthly5_shadow.SELECTED_CANDIDATE,
            "selector_source": monthly5_shadow.RESEARCH_SELECTOR_SOURCE,
            "max_leverage": 4,
            "primary_direction": "long",
            "top_pick": "mom120_lf|lev4|stopNone|target0.05|redlev0.5",
        }

        with unittest.mock.patch.object(
            verify_monthly5_runtime.monthly5_research_selector,
            "build_research_selector_probe",
            return_value=probe,
        ):
            failures = verify_monthly5_runtime._verify_research_selector_artifact(position, spec)

        self.assertIn("monthly5 market selection selector key mismatch", failures)

    def test_runtime_verifier_rejects_short_action_against_long_flat_selector(self):
        spec = self._spec("summary.json", "monthly.json")
        shadow = self._shadow()
        shadow["market_selection"]["selected_plan"] = "normal_short_selector"
        shadow["market_selection"]["shadow_action"] = "evaluate_short"
        position = {
            "monthly5_shadow": shadow,
            "monthly5_live_selector_input": self._live_selector_input(),
            "monthly5_live_selector_decision": self._live_selector_decision(),
        }
        probe = {
            "artifact_available": True,
            "selected_candidate": monthly5_shadow.SELECTED_CANDIDATE,
            "selector_source": monthly5_shadow.RESEARCH_SELECTOR_SOURCE,
            "max_leverage": 4,
            "primary_direction": "long",
            "top_pick": "mom120_lf|lev4|stopNone|target0.05|redlev0.5",
        }

        with unittest.mock.patch.object(
            verify_monthly5_runtime.monthly5_research_selector,
            "build_research_selector_probe",
            return_value=probe,
        ):
            failures = verify_monthly5_runtime._verify_research_selector_artifact(position, spec)

        self.assertIn("monthly5 market selection action not allowed by live selector direction", failures)

    def test_runtime_verifier_rejects_selection_leverage_above_live_selector_key(self):
        spec = self._spec("summary.json", "monthly.json")
        shadow = self._shadow()
        shadow["market_selection"]["max_leverage"] = 5
        position = {
            "monthly5_shadow": shadow,
            "monthly5_live_selector_input": self._live_selector_input(),
            "monthly5_live_selector_decision": self._live_selector_decision(),
        }
        probe = {
            "artifact_available": True,
            "selected_candidate": monthly5_shadow.SELECTED_CANDIDATE,
            "selector_source": monthly5_shadow.RESEARCH_SELECTOR_SOURCE,
            "max_leverage": 4,
            "primary_direction": "long",
            "top_pick": "mom120_lf|lev4|stopNone|target0.05|redlev0.5",
        }

        with unittest.mock.patch.object(
            verify_monthly5_runtime.monthly5_research_selector,
            "build_research_selector_probe",
            return_value=probe,
        ):
            failures = verify_monthly5_runtime._verify_research_selector_artifact(position, spec)

        self.assertIn("monthly5 market selection leverage exceeds live selector key", failures)

    def test_runtime_verifier_rejects_disabled_promotion_gate(self):
        position = {"monthly5_shadow": self._shadow()}

        with unittest.mock.patch.dict(
            verify_monthly5_runtime.os.environ,
            {"MONTHLY5_SIGNAL_OVERRIDE_REQUIRE_PROMOTION_READY": "0"},
        ):
            failures = verify_monthly5_runtime._verify_promotion_gate(position)

        self.assertIn("monthly5 signal override promotion gate disabled", failures)

    def test_runtime_verifier_rejects_override_applied_before_promotion_ready(self):
        position = {
            "monthly5_shadow": self._shadow(),
            "monthly5_signal_override": {
                "applied": True,
                "reason": "monthly5_market_selection",
            },
        }

        failures = verify_monthly5_runtime._verify_promotion_gate(position)

        self.assertIn("monthly5 signal override applied before promotion_ready", failures)


if __name__ == "__main__":
    unittest.main()
