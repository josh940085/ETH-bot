import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import maintenance


class StrategyStrictnessCheckTests(unittest.TestCase):
    def _run_check(self, summary, env=None):
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "backtest_latest_summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with (
                patch.object(maintenance, "data_path", return_value=summary_path),
                patch.dict(os.environ, env or {}, clear=False),
            ):
                return maintenance._check_strategy_strictness()

    def test_passes_when_general_signals_are_not_hidden_by_daily_minimum(self):
        result = self._run_check(
            {
                "trades": 9,
                "trade_day_coverage": {
                    "calendar_days": 8,
                    "trade_days": 8,
                    "daily_min_trades": 5,
                },
            }
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["general_trades"], 4)
        self.assertEqual(result["required_general_trades"], 2)

    def test_flags_strategy_that_only_trades_from_daily_minimum(self):
        with (
            patch.object(
                maintenance,
                "_check_monthly5_strictness_cover",
                return_value={"covered": False, "reason": "runtime_failed"},
            ),
            self.assertRaisesRegex(RuntimeError, "策略可能過嚴.*一般策略單"),
        ):
            self._run_check(
                {
                    "trades": 7,
                    "trade_day_coverage": {
                        "calendar_days": 7,
                        "trade_days": 7,
                        "daily_min_trades": 7,
                    },
                }
            )

    def test_monthly5_cover_allows_low_general_trade_count(self):
        with patch.object(
            maintenance,
            "_check_monthly5_strictness_cover",
            return_value={
                "covered": True,
                "candidate_detail": "PASS monthly5_candidate",
                "runtime_detail": "PASS monthly5_runtime",
            },
        ):
            result = self._run_check(
                {
                    "trades": 7,
                    "trade_day_coverage": {
                        "calendar_days": 7,
                        "trade_days": 7,
                        "daily_min_trades": 7,
                    },
                },
                env={"MAINTENANCE_STRICTNESS_MONTHLY5_COVER": "1"},
            )

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["monthly5_cover"])
        self.assertEqual(result["general_trades"], 0)
        self.assertIn("一般策略偏少", result["detail"])

    def test_monthly5_cover_requires_candidate_and_runtime_health(self):
        completed = subprocess.CompletedProcess(
            args=["verify"],
            returncode=1,
            stdout="FAIL monthly5_runtime stale",
        )
        with patch.object(maintenance, "_run_command", return_value=completed):
            cover = maintenance._check_monthly5_strictness_cover()

        self.assertFalse(cover["covered"])
        self.assertEqual(cover["reason"], "candidate_failed")

    def test_monthly5_cover_fails_when_runtime_is_unhealthy(self):
        candidate_ok = subprocess.CompletedProcess(
            args=["verify_monthly5_candidate.py"],
            returncode=0,
            stdout="PASS monthly5_candidate",
        )
        runtime_failed = subprocess.CompletedProcess(
            args=["verify_monthly5_runtime.py"],
            returncode=1,
            stdout="FAIL monthly5_runtime stale",
        )
        with patch.object(maintenance, "_run_command", side_effect=[candidate_ok, runtime_failed]):
            cover = maintenance._check_monthly5_strictness_cover()

        self.assertFalse(cover["covered"])
        self.assertEqual(cover["reason"], "runtime_failed")

    def test_flags_low_trade_day_coverage(self):
        with (
            patch.object(
                maintenance,
                "_check_monthly5_strictness_cover",
                return_value={"covered": False, "reason": "runtime_failed"},
            ),
            self.assertRaisesRegex(RuntimeError, "策略可能過嚴.*交易日覆蓋率"),
        ):
            self._run_check(
                {
                    "trades": 4,
                    "trade_day_coverage": {
                        "calendar_days": 7,
                        "trade_days": 4,
                        "daily_min_trades": 2,
                    },
                }
            )

    def test_short_sample_is_reported_without_false_alarm(self):
        result = self._run_check(
            {
                "trades": 1,
                "trade_day_coverage": {
                    "calendar_days": 2,
                    "trade_days": 1,
                    "daily_min_trades": 1,
                },
            }
        )

        self.assertFalse(result["sample_sufficient"])


if __name__ == "__main__":
    unittest.main()
