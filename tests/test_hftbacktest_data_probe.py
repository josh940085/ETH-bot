import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import hftbacktest_data_probe


class HftbacktestDataProbeTests(unittest.TestCase):
    def test_requires_trade_and_l2_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "capture.jsonl"
            path.write_text(json.dumps({"type": "trade"}) + "\n", encoding="utf-8")
            with mock.patch.object(
                hftbacktest_data_probe.importlib.metadata,
                "version",
                return_value="2.4.4",
            ):
                report = hftbacktest_data_probe.probe(path)

        self.assertFalse(report["ready"])
        self.assertIn("l2_book_updates_missing", report["blockers"])

    def test_accepts_complete_capture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "capture.jsonl"
            path.write_text(
                json.dumps({"type": "trade"})
                + "\n"
                + json.dumps({"type": "l2_book"})
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                hftbacktest_data_probe.importlib.metadata,
                "version",
                return_value="2.4.4",
            ):
                report = hftbacktest_data_probe.probe(path)

        self.assertTrue(report["ready"])


if __name__ == "__main__":
    unittest.main()
