import json
import os
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
        with self.assertRaisesRegex(RuntimeError, "策略可能過嚴.*一般策略單"):
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

    def test_flags_low_trade_day_coverage(self):
        with self.assertRaisesRegex(RuntimeError, "策略可能過嚴.*交易日覆蓋率"):
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
