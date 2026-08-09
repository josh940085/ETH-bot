import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import monthly5_volume_forward_shadow as shadow
import verify_monthly5_volume_forward as verifier


class Monthly5VolumeForwardShadowTests(unittest.TestCase):
    def _frame(self, periods, freq):
        index = pd.date_range("2026-01-01", periods=periods, freq=freq, tz="UTC")
        close = pd.Series(range(periods), dtype="float64") + 100.0
        return pd.DataFrame(
            {
                "time": (index.as_unit("ns").astype("int64") // 1_000_000).to_numpy(),
                "open": close.to_numpy(),
                "high": (close + 1.0).to_numpy(),
                "low": (close - 1.0).to_numpy(),
                "close": close.to_numpy(),
                "volume": [10.0] * periods,
            }
        )

    def test_live_probe_uses_completed_frames_and_volume_gate(self):
        probe = shadow.build_live_probe(self._frame(40, "4h"), self._frame(4, "5min"))
        self.assertTrue(probe["usable"])
        self.assertEqual(probe["relative_volume_4h"], 1.0)
        self.assertTrue(probe["volume_allowed"])
        self.assertEqual(probe["candidate_signal"], probe["baseline_signal"])

    def test_files_remain_shadow_only_and_deduplicate_bar(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            history_path = Path(directory) / "history.jsonl"
            probe = shadow.build_live_probe(self._frame(40, "4h"), self._frame(4, "5min"))
            now_ts = probe["bar_close_ts_ms"] / 1000
            state = shadow.update_files(state_path, history_path, probe, now_ts=now_ts)
            shadow.update_files(state_path, history_path, probe, now_ts=now_ts)
            rows = shadow.load_history(history_path)
            self.assertEqual(len(rows), 1)
            self.assertFalse(state["execution_allowed"])
            self.assertIn("forward_span_lt_30d", state["promotion_blockers"])
            self.assertIn("forward_rows_insufficient", state["promotion_blockers"])
            self.assertIn("forward_month_target_unproven", state["promotion_blockers"])
            self.assertEqual(verifier.verify(state, rows, 1_000_000_000.0), [])

    def test_backfill_is_causal_and_starts_after_frozen_research(self):
        index = pd.date_range("2026-07-20 00:04:59.999", periods=6000, freq="5min", tz="UTC")
        close = 100.0 + np.linspace(0.0, 20.0, len(index))
        frame = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.full(len(index), 10.0),
            },
            index=index,
        )
        probes = shadow.build_backfill_probes(
            frame,
            now_ts=pd.Timestamp("2026-08-09 00:02:00", tz="UTC").timestamp(),
        )
        self.assertTrue(probes)
        self.assertGreaterEqual(probes[0]["bar_close_ts_ms"], 1785801600000)
        self.assertLessEqual(probes[-1]["bar_close_ts_ms"], 1786233600000)
        self.assertTrue(all(row["execution_allowed"] is False for row in probes))
        self.assertTrue(all(row["market_regime_4h"] in {"up", "down", "range", "unknown"} for row in probes))

    def test_merge_history_sorts_deduplicates_and_disables_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            shadow.merge_history(
                path,
                [
                    {"bar_close_ts_ms": 20, "execution_allowed": True},
                    {"bar_close_ts_ms": 10},
                    {"bar_close_ts_ms": 20, "close": 2.0},
                ],
            )
            rows = shadow.load_history(path)
            self.assertEqual([row["bar_close_ts_ms"] for row in rows], [10, 20])
            self.assertEqual(rows[-1]["close"], 2.0)
            self.assertTrue(all(row["shadow_only"] is True for row in rows))
            self.assertTrue(all(row["execution_allowed"] is False for row in rows))


if __name__ == "__main__":
    unittest.main()
