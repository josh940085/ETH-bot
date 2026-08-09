import unittest
from unittest import mock

import research_tool_status


class ResearchToolStatusTests(unittest.TestCase):
    def test_status_keeps_tardis_excluded(self):
        with mock.patch.object(
            research_tool_status,
            "_probe_package",
            return_value={"installed": True, "version": "test"},
        ):
            status = research_tool_status.build_status()

        self.assertTrue(status["ready"])
        self.assertFalse(status["excluded"]["tardis"]["installed"])
        self.assertEqual(
            status["excluded"]["tardis"]["reason"],
            "explicitly_excluded_by_user",
        )
        self.assertTrue(status["research_dependencies_isolated_from_live"])
        self.assertTrue(
            status["tools"]["freqtrade_validation"][
                "full_freqtrade_runtime_installed"
            ]
        )
        self.assertFalse(
            status["tools"]["market_regime_drift"]["live_control_enabled"]
        )


if __name__ == "__main__":
    unittest.main()
