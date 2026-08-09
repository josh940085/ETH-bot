import tempfile
import unittest
from pathlib import Path

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
            self.assertEqual(verifier.verify(state, rows, 1_000_000_000.0), [])


if __name__ == "__main__":
    unittest.main()
