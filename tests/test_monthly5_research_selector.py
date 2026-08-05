import json
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
