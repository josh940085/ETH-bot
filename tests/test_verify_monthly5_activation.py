import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import io

import verify_monthly5_activation


class VerifyMonthly5ActivationTests(unittest.TestCase):
    def _position(self, *, promotion_ready=True, selected_plan="normal_long_selector", reason_codes=None, override=None):
        return {
            "strategy_signal": "wait",
            "strategy_execution_status": "waiting",
            "monthly5_signal_override": override if override is not None else {"applied": True, "reason": "monthly5_wait_override"},
            "monthly5_readiness": {"promotion_ready": promotion_ready},
            "monthly5_shadow": {
                "promotion_ready": promotion_ready,
                "market_selection": {
                    "selected_plan": selected_plan,
                    "shadow_action": "evaluate_long",
                    "reason_codes": list(reason_codes or []),
                },
            },
        }

    def test_activation_pending_when_promotion_is_not_ready(self):
        failures = verify_monthly5_activation._activation_failures(
            self._position(
                promotion_ready=False,
                selected_plan="mixed_bias_long_probe",
                reason_codes=["mixed_bias_shadow_probe"],
                override={"applied": False, "reason": "monthly5_promotion_not_ready"},
            )
        )

        self.assertEqual(failures, [])

    def test_activation_rejects_probe_after_promotion_ready(self):
        position = self._position(
            selected_plan="mixed_bias_long_probe",
            reason_codes=["mixed_bias_shadow_probe"],
            override={"applied": False, "reason": "micro_probe_requires_host_signal"},
        )

        failures = verify_monthly5_activation._activation_failures(position)

        self.assertIn("selected_plan not actionable: mixed_bias_long_probe", failures)
        self.assertIn("probe reason_codes active: mixed_bias_shadow_probe", failures)
        self.assertIn("waiting state missing applied monthly5 override: micro_probe_requires_host_signal", failures)

    def test_activation_accepts_applied_normal_selector(self):
        failures = verify_monthly5_activation._activation_failures(self._position())

        self.assertEqual(failures, [])

    def test_main_fails_with_blocker_detail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "position.json"
            path.write_text(
                json.dumps(
                    self._position(
                        selected_plan="profile_quality_shadow_probe",
                        reason_codes=["profile_quality_shadow_probe"],
                        override={"applied": False, "reason": "micro_probe_requires_host_signal"},
                    )
                ),
                encoding="utf-8",
            )

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                code = verify_monthly5_activation.main(["--position", str(path)])

        self.assertEqual(code, 1)
        self.assertIn("FAIL monthly5_activation", stdout.getvalue())
        self.assertIn("profile_quality_shadow_probe", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
