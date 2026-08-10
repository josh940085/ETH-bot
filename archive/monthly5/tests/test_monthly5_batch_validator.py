import csv
import json
import tempfile
import unittest
from pathlib import Path

import monthly5_batch_validator


def _rows(start="2023-01", periods=24, *, return_pct=6.0, flat_time_pct=20.0, leverage=4):
    months = __import__("pandas").period_range(start, periods=periods, freq="M")
    return [
        {
            "month": month.strftime("%Y-%m"),
            "return_pct": return_pct,
            "flat_time_pct": flat_time_pct,
            "top_pick": f"mom120_lf|lev{leverage}|stopNone|target0.05|redlev0.5",
        }
        for month in months
    ]


class Monthly5BatchValidatorTests(unittest.TestCase):
    def _trade_evidence(self, root, candidate="qualified"):
        path = Path(root) / "trades.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "entry_time",
                    "exit_time",
                    "entry_fill_time",
                    "exit_fill_time",
                    "side",
                    "quantity",
                    "pnl",
                    "fee",
                    "slippage",
                    "data_source",
                    "candidate",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "entry_time": "2023-01-01",
                    "exit_time": "2023-01-02",
                    "entry_fill_time": "2023-01-01T00:00:00Z",
                    "exit_fill_time": "2023-01-02T00:00:00Z",
                    "side": "long",
                    "quantity": "0.001",
                    "pnl": "1.0",
                    "fee": "0.1",
                    "slippage": "0.01",
                    "data_source": "binance_public_data_usdm_aggTrades",
                    "candidate": candidate,
                }
            )
            for month in __import__("pandas").period_range("2023-02", "2024-12", freq="M"):
                writer.writerow(
                    {
                        "entry_time": f"{month.strftime('%Y-%m')}-01T00:00:00Z",
                        "exit_time": f"{month.strftime('%Y-%m')}-02T00:00:00Z",
                        "entry_fill_time": f"{month.strftime('%Y-%m')}-01T00:00:01Z",
                        "exit_fill_time": f"{month.strftime('%Y-%m')}-02T00:00:01Z",
                        "side": "long",
                        "quantity": "0.001",
                        "pnl": "1.0",
                        "fee": "0.1",
                        "slippage": "0.01",
                        "data_source": "binance_public_data_usdm_aggTrades",
                        "candidate": candidate,
                    }
                )
        return path

    def test_candidate_is_deployment_candidate_with_holdout_and_cost_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = monthly5_batch_validator.validate_candidate(
                "qualified",
                _rows(),
                start="2023-01",
                complete_through="2024-12",
                holdout_start="2024-01",
                trade_evidence=self._trade_evidence(tmpdir),
            )

        self.assertEqual(result["verdict"], "deployment_candidate")
        self.assertTrue(result["deployment_ready"])
        self.assertEqual(result["holdout"]["months_ge_target"], 12)

    def test_missing_trade_evidence_keeps_metric_winner_research_only(self):
        result = monthly5_batch_validator.validate_candidate(
            "research",
            _rows(),
            start="2023-01",
            complete_through="2024-12",
            holdout_start="2024-01",
        )

        self.assertEqual(result["verdict"], "research_only")
        self.assertIn("trade_evidence_missing", result["evidence_blockers"])

    def test_exact_floor_saturation_is_an_integrity_blocker(self):
        result = monthly5_batch_validator.validate_candidate(
            "clamped",
            _rows(return_pct=5.0),
            start="2023-01",
            complete_through="2024-12",
            holdout_start="2024-01",
        )

        self.assertEqual(result["verdict"], "research_only")
        self.assertIn("exact_target_floor_saturation", result["evidence_blockers"])

    def test_missing_month_and_leverage_violation_reject_candidate(self):
        rows = _rows(leverage=5)
        rows.pop(3)
        rows[-1]["top_pick"] = "mom120_lf|lev6|stopNone|target0.05|redlev0.5"
        result = monthly5_batch_validator.validate_candidate(
            "invalid",
            rows,
            start="2023-01",
            complete_through="2024-12",
            holdout_start="2024-01",
            max_leverage=5,
        )

        self.assertEqual(result["verdict"], "rejected")
        self.assertIn("month_coverage_incomplete", result["metric_blockers"])
        self.assertIn("max_leverage_exceeded", result["metric_blockers"])

    def test_batch_report_ranks_metric_qualified_candidate_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monthly_path = Path(tmpdir) / "monthly.json"
            monthly_path.write_text(
                json.dumps(
                    {
                        "rejected": _rows(return_pct=2.0),
                        "qualified": _rows(return_pct=6.0),
                    }
                ),
                encoding="utf-8",
            )
            report = monthly5_batch_validator.build_batch_report(
                monthly_path,
                start="2023-01",
                complete_through="2024-12",
                holdout_start="2024-01",
            )

        self.assertEqual(report["candidate_count"], 2)
        self.assertEqual(report["metric_qualified_count"], 1)
        self.assertEqual(report["results"][0]["candidate"], "qualified")


if __name__ == "__main__":
    unittest.main()
