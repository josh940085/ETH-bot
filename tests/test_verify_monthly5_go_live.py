import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import verify_monthly5_go_live


class VerifyMonthly5GoLiveTests(unittest.TestCase):
    def test_go_live_passes_when_all_steps_pass(self):
        results = [
            SimpleNamespace(returncode=0, stdout="PASS candidate\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="PASS bias\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="PASS execution evidence\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="PASS runtime\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="PASS account open_count=0\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="PASS readiness promotion_ready=true\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="PASS activation status=actionable\n", stderr=""),
        ]

        with (
            patch.object(verify_monthly5_go_live.subprocess, "run", side_effect=results),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            code = verify_monthly5_go_live.main([])

        self.assertEqual(code, 0)
        self.assertIn("PASS monthly5_go_live promotion_ready=true live_gate=ready", stdout.getvalue())

    def test_go_live_fails_and_preserves_promotion_blockers(self):
        results = [
            SimpleNamespace(returncode=0, stdout="PASS candidate\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="PASS bias\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="PASS execution evidence\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="PASS runtime promotion_ready=false\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="PASS account open_count=0\n", stderr=""),
            SimpleNamespace(
                returncode=1,
                stdout=(
                    "PASS readiness promotion_ready=false\n"
                    "BLOCKER code=sample_span remaining_hours=6.7\n"
                ),
                stderr="",
            ),
            SimpleNamespace(returncode=0, stdout="PASS activation status=pending_promotion\n", stderr=""),
        ]

        with (
            patch.object(verify_monthly5_go_live.subprocess, "run", side_effect=results),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            code = verify_monthly5_go_live.main([])

        output = stdout.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("FAIL monthly5_go_live_step=promotion", output)
        self.assertIn("BLOCKER code=sample_span remaining_hours=6.7", output)
        self.assertIn("FAIL monthly5_go_live failed_steps=promotion", output)

    def test_go_live_fails_when_account_is_not_flat(self):
        results = [
            SimpleNamespace(returncode=0, stdout="PASS candidate\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="PASS bias\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="PASS execution evidence\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="PASS runtime\n", stderr=""),
            SimpleNamespace(
                returncode=1,
                stdout='FAIL monthly5_account {"symbol":"BTCUSDT","open_count":1}\n',
                stderr="",
            ),
            SimpleNamespace(returncode=0, stdout="PASS readiness promotion_ready=true\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="PASS activation status=actionable\n", stderr=""),
        ]

        with (
            patch.object(verify_monthly5_go_live.subprocess, "run", side_effect=results),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            code = verify_monthly5_go_live.main([])

        output = stdout.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("FAIL monthly5_go_live_step=account", output)
        self.assertIn("FAIL monthly5_go_live failed_steps=account", output)

    def test_go_live_fails_when_activation_is_blocked_after_promotion(self):
        results = [
            SimpleNamespace(returncode=0, stdout="PASS candidate\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="PASS bias\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="PASS execution evidence\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="PASS runtime\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="PASS account open_count=0\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="PASS readiness promotion_ready=true\n", stderr=""),
            SimpleNamespace(
                returncode=1,
                stdout=(
                    "FAIL monthly5_activation status=blocked\n"
                    "BLOCKER selected_plan not actionable: mixed_bias_long_probe\n"
                ),
                stderr="",
            ),
        ]

        with (
            patch.object(verify_monthly5_go_live.subprocess, "run", side_effect=results),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            code = verify_monthly5_go_live.main([])

        output = stdout.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("FAIL monthly5_go_live_step=activation", output)
        self.assertIn("BLOCKER selected_plan not actionable", output)

    def test_go_live_forwards_research_evidence_paths(self):
        results = [
            SimpleNamespace(returncode=0, stdout="PASS\n", stderr="")
            for _ in range(7)
        ]
        with patch.object(
            verify_monthly5_go_live.subprocess,
            "run",
            side_effect=results,
        ) as run:
            code = verify_monthly5_go_live.main(
                [
                    "--strategy-source",
                    "strategy.py",
                    "--prefix-replay-report",
                    "prefix.json",
                    "--trade-evidence",
                    "fills.csv",
                ]
            )

        self.assertEqual(code, 0)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn("--strategy-source", commands[1])
        self.assertIn("strategy.py", commands[1])
        self.assertIn("--prefix-replay-report", commands[1])
        self.assertIn("prefix.json", commands[1])
        self.assertIn("--trade-evidence", commands[2])
        self.assertIn("fills.csv", commands[2])


if __name__ == "__main__":
    unittest.main()
