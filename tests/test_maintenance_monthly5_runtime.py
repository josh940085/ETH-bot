import unittest
from types import SimpleNamespace
from unittest.mock import patch

import maintenance


class Monthly5MaintenanceTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
