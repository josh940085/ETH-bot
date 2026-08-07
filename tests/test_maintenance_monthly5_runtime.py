import unittest
from types import SimpleNamespace
from unittest.mock import patch

import maintenance


class Monthly5MaintenanceTests(unittest.TestCase):
    def test_monthly5_candidate_check_wraps_verifier_pass(self):
        command_result = SimpleNamespace(
            returncode=0,
            stdout=(
                "PASS postlock_scale0.15_floor_pdaystopNone months_ge_5=79 "
                "complete_months_ge_5=79/79 complete_hit_rate_pct=100.0 max_leverage_used=5\n"
            ),
        )

        with patch.object(maintenance, "_run_command", return_value=command_result) as run_command:
            result = maintenance._check_monthly5_candidate()

        self.assertEqual(result["status"], "ok")
        self.assertIn("complete_months_ge_5=79/79", result["detail"])
        self.assertIn("verify_monthly5_candidate.py", run_command.call_args.args[0])

    def test_monthly5_candidate_check_raises_on_verifier_failure(self):
        command_result = SimpleNamespace(
            returncode=1,
            stdout="FAIL complete monthly rows below 5.0%\n",
        )

        with patch.object(maintenance, "_run_command", return_value=command_result):
            with self.assertRaisesRegex(RuntimeError, "complete monthly rows below"):
                maintenance._check_monthly5_candidate()

    def test_monthly5_runtime_check_wraps_verifier_pass(self):
        command_result = SimpleNamespace(
            returncode=0,
            stdout="PASS monthly5_runtime strategy_id=monthly5_postlock_hourly_v0 mode=normal selected_plan=normal_long_selector exposure_cap=0.35 history=ok e2e=ok\n",
        )

        with patch.object(maintenance, "_run_command", return_value=command_result) as run_command:
            result = maintenance._check_monthly5_runtime()

        self.assertEqual(result["status"], "ok")
        self.assertIn("PASS monthly5_runtime", result["detail"])
        self.assertIn("verify_monthly5_runtime.py", run_command.call_args.args[0])

    def test_monthly5_runtime_check_can_require_history(self):
        command_result = SimpleNamespace(
            returncode=0,
            stdout="PASS monthly5_runtime strategy_id=monthly5_postlock_hourly_v0 mode=normal selected_plan=normal_long_selector exposure_cap=0.35 history=ok e2e=ok\n",
        )

        with (
            patch.dict(maintenance.os.environ, {"MONTHLY5_REQUIRE_HISTORY": "1"}),
            patch.object(maintenance, "_run_command", return_value=command_result) as run_command,
        ):
            maintenance._check_monthly5_runtime()

        self.assertIn("--require-history", run_command.call_args.args[0])

    def test_monthly5_runtime_check_raises_on_verifier_failure(self):
        command_result = SimpleNamespace(
            returncode=1,
            stdout="FAIL shadow_file post-lock exposure drift\n",
        )

        with patch.object(maintenance, "_run_command", return_value=command_result):
            with self.assertRaisesRegex(RuntimeError, "post-lock exposure drift"):
                maintenance._check_monthly5_runtime()

    def test_monthly5_account_check_wraps_verifier_pass(self):
        command_result = SimpleNamespace(
            returncode=0,
            stdout='PASS monthly5_account {"symbol":"BTCUSDT","open_count":0,"positions":[]}\n',
        )

        with patch.object(maintenance, "_run_command", return_value=command_result) as run_command:
            result = maintenance._check_monthly5_account()

        self.assertEqual(result["status"], "ok")
        self.assertIn('"open_count":0', result["detail"])
        self.assertIn("verify_monthly5_account.py", run_command.call_args.args[0])

    def test_monthly5_account_check_raises_on_open_position(self):
        command_result = SimpleNamespace(
            returncode=1,
            stdout='FAIL monthly5_account {"symbol":"BTCUSDT","open_count":1}\n',
        )

        with patch.object(maintenance, "_run_command", return_value=command_result):
            with self.assertRaisesRegex(RuntimeError, "open_count"):
                maintenance._check_monthly5_account()

    def test_monthly5_readiness_check_wraps_collecting_status(self):
        command_result = SimpleNamespace(
            returncode=0,
            stdout="PASS monthly5_readiness status=collecting ready=false rows=2 span_hours=0.04 evaluate_rows=1 risk_rows=0\nWARN sample count collecting\n",
        )

        with patch.object(maintenance, "_run_command", return_value=command_result) as run_command:
            result = maintenance._check_monthly5_readiness()

        self.assertEqual(result["status"], "ok")
        self.assertIn("status=collecting", result["detail"])
        self.assertIn("verify_monthly5_readiness.py", run_command.call_args.args[0])
        self.assertEqual(
            run_command.call_args.args[0][
                run_command.call_args.args[0].index("--min-span-hours") + 1
            ],
            "8.0",
        )

    def test_monthly5_readiness_check_can_require_ready(self):
        command_result = SimpleNamespace(
            returncode=0,
            stdout="PASS monthly5_readiness status=ready ready=true rows=48 span_hours=24.0 evaluate_rows=20 risk_rows=1\n",
        )

        with (
            patch.dict(maintenance.os.environ, {"MONTHLY5_REQUIRE_READY": "1"}),
            patch.object(maintenance, "_run_command", return_value=command_result) as run_command,
        ):
            maintenance._check_monthly5_readiness()

        self.assertIn("--require-ready", run_command.call_args.args[0])

    def test_monthly5_readiness_check_can_require_promotion_ready(self):
        command_result = SimpleNamespace(
            returncode=0,
            stdout="PASS monthly5_readiness status=ready ready=true promotion_ready=true rows=48 span_hours=24.0\n",
        )

        with (
            patch.dict(maintenance.os.environ, {"MONTHLY5_REQUIRE_PROMOTION_READY": "1"}),
            patch.object(maintenance, "_run_command", return_value=command_result) as run_command,
        ):
            maintenance._check_monthly5_readiness()

        self.assertIn("--require-promotion-ready", run_command.call_args.args[0])

    def test_monthly5_readiness_failure_keeps_promotion_blocker_detail(self):
        command_result = SimpleNamespace(
            returncode=1,
            stdout=(
                "PASS monthly5_readiness status=collecting ready=false promotion_ready=false\n"
                "BLOCKER code=sample_span label=樣本時間 current=11.2 target=24.0 remaining_hours=12.8\n"
            ),
        )

        with patch.object(maintenance, "_run_command", return_value=command_result):
            with self.assertRaisesRegex(RuntimeError, "remaining_hours=12.8"):
                maintenance._check_monthly5_readiness()


if __name__ == "__main__":
    unittest.main()
