import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import runtime_python


class RuntimePythonTests(unittest.TestCase):
    def test_accepts_python_312(self):
        report = runtime_python.runtime_report(
            SimpleNamespace(major=3, minor=12, micro=13)
        )

        self.assertTrue(report["supported"])
        self.assertEqual(report["required"], "3.12")

    def test_rejects_python_311(self):
        report = runtime_python.runtime_report(
            SimpleNamespace(major=3, minor=11, micro=15)
        )

        self.assertFalse(report["supported"])

    def test_current_test_runtime_is_python_312(self):
        self.assertEqual(sys.version_info[:2], (3, 12))

    def test_maintenance_check_reports_supported_runtime(self):
        import maintenance

        result = maintenance._check_python_runtime()

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["supported"])


if __name__ == "__main__":
    unittest.main()
