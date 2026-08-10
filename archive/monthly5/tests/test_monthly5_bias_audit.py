import json
import tempfile
import unittest
from pathlib import Path

import monthly5_bias_audit


class Monthly5BiasAuditTests(unittest.TestCase):
    def test_missing_strategy_source_and_prefix_replay_block_promotion(self):
        audit = monthly5_bias_audit.build_audit()

        self.assertFalse(audit["passed"])
        self.assertIn("strategy_source_missing", audit["blockers"])
        self.assertIn("prefix_replay_missing", audit["blockers"])

    def test_negative_shift_is_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "strategy.py"
            source.write_text("future = close.shift(-1)\n", encoding="utf-8")
            audit = monthly5_bias_audit.build_audit(strategy_source=source)

        self.assertIn("negative_shift_detected", audit["blockers"])

    def test_stable_prefix_and_recursive_report_passes_clean_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "strategy.py"
            report = Path(tmpdir) / "prefix.json"
            source.write_text("past = close.shift(1)\n", encoding="utf-8")
            report.write_text(
                json.dumps({"prefix_stable": True, "recursive_stable": True}),
                encoding="utf-8",
            )
            audit = monthly5_bias_audit.build_audit(
                strategy_source=source,
                prefix_replay_report=report,
            )

        self.assertTrue(audit["passed"])


if __name__ == "__main__":
    unittest.main()
