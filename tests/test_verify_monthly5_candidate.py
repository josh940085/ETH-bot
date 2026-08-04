import json
import tempfile
import unittest
from pathlib import Path

import monthly5_shadow
import verify_monthly5_candidate


class VerifyMonthly5CandidateTests(unittest.TestCase):
    def _write_inputs(self, root: Path, *, monthly_rows=None):
        summary_path = root / "summary.json"
        monthly_path = root / "monthly.json"
        rows = monthly_rows or [
            {
                "month": "2020-01",
                "return_pct": 5.0,
                "flat_time_pct": 10.0,
                "min_intramonth_pnl_pct": -1.0,
                "top_pick": "mom72_ls|lev4|stopNone|target0.05|redlev0.5",
            },
            {
                "month": "2020-02",
                "return_pct": 8.0,
                "flat_time_pct": 20.0,
                "min_intramonth_pnl_pct": -4.0,
                "top_pick": "mom48_lf|lev5|stopNone|target0.08|redlev1.0",
            },
            {
                "month": "2020-03",
                "return_pct": 1.0,
                "flat_time_pct": 30.0,
                "min_intramonth_pnl_pct": 0.0,
                "top_pick": "mom120_lf|lev4|stopNone|target0.05|redlev0.5",
            },
        ]
        summary = {
            "top": [
                {
                    "name": monthly5_shadow.SELECTED_CANDIDATE,
                    "months_ge_5": 2,
                    "months_ge_0": 3,
                    "worst_intramonth_pnl_pct": -4.0,
                    "avg_flat_time_pct": 20.0,
                    "failed_months": [{"month": "2020-03", "return_pct": 1.0}],
                }
            ]
        }
        spec = {
            "objective": {
                "monthly_return_floor_pct": 5.0,
                "max_leverage": 5,
            },
            "backtest_evidence": {
                "source_summary": str(summary_path),
                "source_monthly": str(monthly_path),
                "candidate_name": monthly5_shadow.SELECTED_CANDIDATE,
                "period_start": "2020-01",
                "period_end": "2020-03",
                "complete_months_end": "2020-02",
                "months": 3,
                "months_ge_5": 2,
                "months_ge_0": 3,
                "complete_months": 2,
                "complete_months_ge_5": 2,
                "incomplete_month": "2020-03",
                "worst_intramonth_pnl_pct": -4.0,
                "avg_flat_time_pct": 20.0,
            },
        }
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        monthly_path.write_text(
            json.dumps({monthly5_shadow.SELECTED_CANDIDATE: rows}),
            encoding="utf-8",
        )
        return summary, spec

    def test_candidate_verifier_accepts_monthly_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary, spec = self._write_inputs(Path(tmpdir))

            failures = verify_monthly5_candidate._failures(summary, spec)

        self.assertEqual(failures, [])

    def test_candidate_verifier_rejects_complete_month_below_floor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                {
                    "month": "2020-01",
                    "return_pct": 4.9,
                    "flat_time_pct": 10.0,
                    "min_intramonth_pnl_pct": -1.0,
                    "top_pick": "mom72_ls|lev4|stopNone|target0.05|redlev0.5",
                },
                {
                    "month": "2020-02",
                    "return_pct": 8.0,
                    "flat_time_pct": 20.0,
                    "min_intramonth_pnl_pct": -4.0,
                    "top_pick": "mom48_lf|lev5|stopNone|target0.08|redlev1.0",
                },
                {
                    "month": "2020-03",
                    "return_pct": 1.0,
                    "flat_time_pct": 30.0,
                    "min_intramonth_pnl_pct": 0.0,
                    "top_pick": "mom120_lf|lev4|stopNone|target0.05|redlev0.5",
                },
            ]
            summary, spec = self._write_inputs(Path(tmpdir), monthly_rows=rows)

            failures = verify_monthly5_candidate._failures(summary, spec)

        self.assertTrue(any("complete monthly rows below" in failure for failure in failures))

    def test_candidate_verifier_rejects_leverage_above_cap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                {
                    "month": "2020-01",
                    "return_pct": 5.0,
                    "flat_time_pct": 10.0,
                    "min_intramonth_pnl_pct": -1.0,
                    "top_pick": "mom72_ls|lev6|stopNone|target0.05|redlev0.5",
                },
                {
                    "month": "2020-02",
                    "return_pct": 8.0,
                    "flat_time_pct": 20.0,
                    "min_intramonth_pnl_pct": -4.0,
                    "top_pick": "mom48_lf|lev5|stopNone|target0.08|redlev1.0",
                },
                {
                    "month": "2020-03",
                    "return_pct": 1.0,
                    "flat_time_pct": 30.0,
                    "min_intramonth_pnl_pct": 0.0,
                    "top_pick": "mom120_lf|lev4|stopNone|target0.05|redlev0.5",
                },
            ]
            summary, spec = self._write_inputs(Path(tmpdir), monthly_rows=rows)

            failures = verify_monthly5_candidate._failures(summary, spec)

        self.assertTrue(any("exceed leverage cap" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
