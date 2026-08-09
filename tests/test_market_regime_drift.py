import json
import tempfile
import unittest
from pathlib import Path

import market_regime_drift


class ThresholdDetector:
    def __init__(self):
        self.drift_detected = False

    def update(self, value):
        self.drift_detected = value >= 5.0


class MarketRegimeDriftTests(unittest.TestCase):
    def test_load_shadow_rows_filters_invalid_and_old_policy_rows(self):
        rows = [
            {"shadow_only": True, "selector_policy_version": 7, "updated_ts": 1, "mark_price": 10},
            {"shadow_only": False, "selector_policy_version": 8, "updated_ts": 2, "mark_price": 10},
            {"shadow_only": True, "selector_policy_version": 8, "updated_ts": 3, "mark_price": 10},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            loaded = market_regime_drift.load_shadow_rows(path)

        self.assertEqual([row["updated_ts"] for row in loaded], [3])

    def test_report_is_shadow_only_and_detects_recent_feature_shift(self):
        rows = []
        for index, bull_score in enumerate((1.0, 1.0, 6.0), start=1):
            rows.append(
                {
                    "updated_ts": index,
                    "market_bias": "bullish",
                    "market_state": "trend",
                    "features": {
                        "price_return_bps": 0.0,
                        "score_margin": bull_score,
                        "score_intensity": bull_score,
                        "selector_hit_rate": 0.5,
                        "selector_q25_return_pct": 0.0,
                        "monthly_pnl_pct": 0.0,
                    },
                }
            )

        report = market_regime_drift.analyze_feature_rows(
            rows,
            detector_factory=lambda _feature: ThresholdDetector(),
            min_rows=2,
            recent_window_rows=2,
        )

        self.assertTrue(report["ready"])
        self.assertTrue(report["drift_detected"])
        self.assertTrue(report["shadow_only"])
        self.assertFalse(report["live_control_enabled"])
        self.assertFalse(report["promotion_eligible"])
        self.assertEqual(report["observation_status"], "recent_drift_observed")

    def test_short_history_is_not_ready(self):
        report = market_regime_drift.analyze_feature_rows(
            [],
            detector_factory=lambda _feature: ThresholdDetector(),
            min_rows=48,
        )
        self.assertFalse(report["ready"])
        self.assertEqual(report["observation_status"], "insufficient_history")


if __name__ == "__main__":
    unittest.main()
