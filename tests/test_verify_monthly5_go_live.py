import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import verify_monthly5_go_live


class VerifyMonthly5GoLiveTests(unittest.TestCase):
    def test_go_live_passes_when_all_steps_pass(self):
        results = [
            SimpleNamespace(returncode=0, stdout="PASS candidate\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="PASS runtime\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="PASS readiness promotion_ready=true\n", stderr=""),
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
            SimpleNamespace(returncode=0, stdout="PASS runtime promotion_ready=false\n", stderr=""),
            SimpleNamespace(
                returncode=1,
                stdout=(
                    "PASS readiness promotion_ready=false\n"
                    "BLOCKER code=sample_span remaining_hours=6.7\n"
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
        self.assertIn("FAIL monthly5_go_live_step=promotion", output)
        self.assertIn("BLOCKER code=sample_span remaining_hours=6.7", output)
        self.assertIn("FAIL monthly5_go_live failed_steps=promotion", output)


if __name__ == "__main__":
    unittest.main()
