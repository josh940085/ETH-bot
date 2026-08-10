import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import market_data_sources
import market_history
import market_history_multi_source as mhms


def _row(open_ms, close="100"):
    return [open_ms, "99", "101", "98", close, "10", open_ms + 299999, "0", 0, "0", "0", "0"]


def _binance_frame(missing_months=None):
    frame = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [10.0]},
        index=pd.DatetimeIndex(["2023-06-01T00:04:59.999Z"]),
    )
    frame.attrs["kline_source"] = "binance_history_um:1m"
    if missing_months:
        frame.attrs["kline_missing_months"] = list(missing_months)
    return frame


def _gap_day_frame():
    return pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]},
        index=pd.DatetimeIndex(["2023-06-02T00:04:59.999Z"], name="close_time"),
    )


class GapFillDayTests(unittest.TestCase):
    def test_prefers_coinbase_over_twelve_data(self):
        with (
            mock.patch.object(
                market_data_sources, "fetch_coinbase_klines", return_value=[_row(1685577600000)]
            ) as coinbase,
            mock.patch.object(market_data_sources, "fetch_twelve_data_klines") as twelve,
        ):
            frame, source = mhms.fetch_gap_fill_day("BTCUSDT", "2023-06-01")
        self.assertEqual(source, "coinbase")
        self.assertEqual(len(frame), 1)
        coinbase.assert_called_once()
        twelve.assert_not_called()

    def test_falls_back_to_twelve_data_when_coinbase_fails(self):
        with (
            mock.patch.object(market_data_sources, "fetch_coinbase_klines", side_effect=RuntimeError("boom")),
            mock.patch.object(
                market_data_sources, "fetch_twelve_data_klines", return_value=[_row(1685577600000)]
            ) as twelve,
        ):
            frame, source = mhms.fetch_gap_fill_day("BTCUSDT", "2023-06-01")
        self.assertEqual(source, "twelve_data")
        self.assertEqual(len(frame), 1)
        twelve.assert_called_once()

    def test_returns_none_when_both_sources_fail(self):
        with (
            mock.patch.object(market_data_sources, "fetch_coinbase_klines", side_effect=RuntimeError("a")),
            mock.patch.object(market_data_sources, "fetch_twelve_data_klines", side_effect=RuntimeError("b")),
        ):
            frame, detail = mhms.fetch_gap_fill_day("BTCUSDT", "2023-06-01")
        self.assertIsNone(frame)
        self.assertIn("coinbase", detail)
        self.assertIn("twelve_data", detail)


class BuildHistoryWithGapFillTests(unittest.TestCase):
    def test_returns_archive_frame_unchanged_when_no_gaps(self):
        with mock.patch.object(market_history, "fetch_klines_from_binance_history", return_value=_binance_frame()):
            frame = mhms.build_history_with_gap_fill("BTCUSDT", 0, 1000)
        self.assertEqual(frame.attrs["kline_source"], "binance_history_um:1m")

    def test_fills_gap_and_tags_combined_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resume_path = Path(tmpdir) / "progress.json"
            with (
                mock.patch.object(
                    market_history,
                    "fetch_klines_from_binance_history",
                    return_value=_binance_frame(missing_months=["2023-06-02"]),
                ),
                mock.patch.object(mhms, "fetch_gap_fill_day", return_value=(_gap_day_frame(), "coinbase")),
                mock.patch.object(mhms.time, "sleep"),
            ):
                frame = mhms.build_history_with_gap_fill(
                    "BTCUSDT", 0, 1000, request_gap_sec=0.01, resume_path=resume_path
                )
            self.assertEqual(len(frame), 2)
            self.assertEqual(frame.attrs["kline_source"], "binance_history_um:1m+coinbase")
            self.assertNotIn("kline_missing_months", frame.attrs)
            progress = json.loads(resume_path.read_text(encoding="utf-8"))
            self.assertEqual(progress["completed_days"], ["2023-06-02"])
            self.assertEqual(progress["day_sources"]["2023-06-02"], "coinbase")
            cache_path = mhms._day_cache_path("BTCUSDT", mhms.GAP_FILL_INTERVAL, "2023-06-02", resume_path)
            self.assertTrue(cache_path.exists(), "expected the gap-filled day to be cached to disk")

    def test_resume_reconstructs_frame_from_cached_day_without_refetching(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resume_path = Path(tmpdir) / "progress.json"
            resume_path.write_text(
                json.dumps({"completed_days": ["2023-06-02"], "day_sources": {"2023-06-02": "coinbase"}}),
                encoding="utf-8",
            )
            cache_path = mhms._day_cache_path("BTCUSDT", mhms.GAP_FILL_INTERVAL, "2023-06-02", resume_path)
            mhms._save_day_frame(_gap_day_frame(), cache_path)

            with (
                mock.patch.object(
                    market_history,
                    "fetch_klines_from_binance_history",
                    return_value=_binance_frame(missing_months=["2023-06-02"]),
                ),
                mock.patch.object(mhms, "fetch_gap_fill_day") as gap_fill,
            ):
                frame = mhms.build_history_with_gap_fill("BTCUSDT", 0, 1000, resume_path=resume_path)

            gap_fill.assert_not_called()
            self.assertEqual(len(frame), 2)
            self.assertEqual(frame.attrs["kline_source"], "binance_history_um:1m+coinbase")
            self.assertNotIn("kline_missing_months", frame.attrs)

    def test_still_missing_day_recorded_after_both_sources_fail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resume_path = Path(tmpdir) / "progress.json"
            with (
                mock.patch.object(
                    market_history,
                    "fetch_klines_from_binance_history",
                    return_value=_binance_frame(missing_months=["2023-06-02"]),
                ),
                mock.patch.object(mhms, "fetch_gap_fill_day", return_value=(None, "coinbase: x; twelve_data: y")),
                mock.patch.object(mhms.time, "sleep"),
            ):
                frame = mhms.build_history_with_gap_fill(
                    "BTCUSDT", 0, 1000, request_gap_sec=0.01, resume_path=resume_path
                )
            self.assertEqual(len(frame), 1)  # only the original archive row, no gap frame merged
            self.assertEqual(frame.attrs.get("kline_missing_months"), ["2023-06-02"])
            progress = json.loads(resume_path.read_text(encoding="utf-8"))
            self.assertEqual(progress["day_sources"]["2023-06-02"], "unavailable")

    def test_previously_unavailable_day_is_retried_on_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resume_path = Path(tmpdir) / "progress.json"
            resume_path.write_text(
                json.dumps({"completed_days": ["2023-06-02"], "day_sources": {"2023-06-02": "unavailable"}}),
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    market_history,
                    "fetch_klines_from_binance_history",
                    return_value=_binance_frame(missing_months=["2023-06-02"]),
                ),
                mock.patch.object(mhms, "fetch_gap_fill_day", return_value=(_gap_day_frame(), "coinbase")) as gap_fill,
                mock.patch.object(mhms.time, "sleep"),
            ):
                frame = mhms.build_history_with_gap_fill(
                    "BTCUSDT", 0, 1000, request_gap_sec=0.01, resume_path=resume_path
                )
            gap_fill.assert_called_once()
            self.assertEqual(len(frame), 2)

    def test_paces_requests_between_gap_days(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resume_path = Path(tmpdir) / "progress.json"
            sleeps = []
            with (
                mock.patch.object(
                    market_history,
                    "fetch_klines_from_binance_history",
                    return_value=_binance_frame(missing_months=["2023-06-02", "2023-06-03"]),
                ),
                mock.patch.object(mhms, "fetch_gap_fill_day", return_value=(_gap_day_frame(), "coinbase")),
                mock.patch.object(mhms.time, "sleep", side_effect=sleeps.append),
            ):
                mhms.build_history_with_gap_fill(
                    "BTCUSDT", 0, 1000, request_gap_sec=2.5, resume_path=resume_path
                )
            self.assertEqual(sleeps, [2.5, 2.5])


if __name__ == "__main__":
    unittest.main()
