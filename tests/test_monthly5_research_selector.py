import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import monthly5_research_selector
import monthly5_shadow


class Monthly5ResearchSelectorTests(unittest.TestCase):
    def test_parse_top_pick_extracts_direction_and_risk(self):
        parsed = monthly5_research_selector.parse_top_pick(
            "mom120_lf|lev4|stopNone|target0.05|redlev0.5"
        )

        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["primary_direction"], "long")
        self.assertEqual(parsed["direction_label"], "long_flat")
        self.assertEqual(parsed["max_leverage"], 4)
        self.assertEqual(parsed["target_pct"], 5.0)
        self.assertIsNone(parsed["stop_pct"])
        self.assertEqual(parsed["recovery_exposure_scale"], 0.5)

    def test_build_probe_reads_current_month_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            monthly_path = root / "monthly.json"
            spec_path = root / "spec.json"
            monthly_path.write_text(
                json.dumps(
                    {
                        monthly5_shadow.SELECTED_CANDIDATE: [
                            {
                                "month": "2026-08",
                                "return_pct": 0.497,
                                "flat_time_pct": 90.28,
                                "lock_hit_day": None,
                                "recovery_used": False,
                                "top_pick": "mom120_lf|lev4|stopNone|target0.05|redlev0.5",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            spec_path.write_text(
                json.dumps(
                    {
                        "backtest_evidence": {
                            "candidate_name": monthly5_shadow.SELECTED_CANDIDATE,
                            "source_monthly": str(monthly_path),
                        }
                    }
                ),
                encoding="utf-8",
            )

            probe = monthly5_research_selector.build_research_selector_probe(
                spec_path,
                now_ts=1785927000,
            )

        self.assertTrue(probe["artifact_available"])
        self.assertFalse(probe["stale"])
        self.assertEqual(probe["selector_source"], monthly5_shadow.RESEARCH_SELECTOR_SOURCE)
        self.assertEqual(probe["month"], "2026-08")
        self.assertEqual(probe["top_pick"], "mom120_lf|lev4|stopNone|target0.05|redlev0.5")
        self.assertEqual(probe["primary_direction"], "long")
        self.assertEqual(probe["max_leverage"], 4)

    def test_live_selector_input_probe_accepts_warm_daily_frame_and_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "selector_cache.npz"
            np.savez(
                cache_path,
                Xday=np.zeros((2, monthly5_research_selector.EXPECTED_SHORT_MARKET_STATE_FEATURES), dtype="float32"),
                keys=np.array(["mom120_lf|lev4|stopNone|target0.05|redlev0.5"]),
                days=np.array(["2026-08-01", "2026-08-02"]),
            )
            frame = pd.DataFrame(
                {
                    "high": [101.0] * monthly5_research_selector.REQUIRED_LIVE_DAILY_ROWS,
                    "low": [99.0] * monthly5_research_selector.REQUIRED_LIVE_DAILY_ROWS,
                    "close": [100.0] * monthly5_research_selector.REQUIRED_LIVE_DAILY_ROWS,
                },
                index=pd.date_range("2025-08-03", periods=monthly5_research_selector.REQUIRED_LIVE_DAILY_ROWS, freq="1D", tz="UTC"),
            )

            probe = monthly5_research_selector.build_live_selector_input_probe(
                frame,
                cache_path=cache_path,
            )

        self.assertTrue(probe["usable"])
        self.assertEqual(probe["selector_source"], monthly5_shadow.RESEARCH_SELECTOR_SOURCE)
        self.assertEqual(probe["cache_feature_count"], monthly5_research_selector.EXPECTED_SHORT_MARKET_STATE_FEATURES)
        self.assertEqual(probe["daily_rows"], monthly5_research_selector.REQUIRED_LIVE_DAILY_ROWS)
        self.assertEqual(probe["latest_daily_key"], "2026-08-02")

    def test_live_selector_input_probe_reports_missing_warmup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "selector_cache.npz"
            np.savez(
                cache_path,
                Xday=np.zeros((2, monthly5_research_selector.EXPECTED_SHORT_MARKET_STATE_FEATURES), dtype="float32"),
                keys=np.array(["mom120_lf|lev4|stopNone|target0.05|redlev0.5"]),
                days=np.array(["2026-08-01", "2026-08-02"]),
            )
            frame = pd.DataFrame(
                {"high": [101.0], "low": [99.0], "close": [100.0]},
                index=pd.date_range("2026-08-02", periods=1, freq="1D", tz="UTC"),
            )

            probe = monthly5_research_selector.build_live_selector_input_probe(
                frame,
                cache_path=cache_path,
            )

        self.assertFalse(probe["usable"])
        self.assertIn("daily_warmup_insufficient", probe["blocking_reasons"])


if __name__ == "__main__":
    unittest.main()
