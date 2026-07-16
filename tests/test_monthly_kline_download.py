import datetime as dt
import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import backtest
import monthly_kline_download
import program


def _monthly_zip_bytes(year, month, interval_ms=300_000):
    start = dt.datetime(year, month, 1, tzinfo=dt.timezone.utc)
    end = backtest._next_month(start)
    start_ms = int(start.timestamp() * 1000)
    count = int((end - start).total_seconds() * 1000 // interval_ms)
    rows = ["open_time,open,high,low,close,volume,close_time"]
    for index in range(count):
        open_time = start_ms + index * interval_ms
        rows.append(f"{open_time},2000,2001,1999,2000.5,10,{open_time + interval_ms - 1}")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"ETHUSDT-5m-{year:04d}-{month:02d}.csv", "\n".join(rows))
    return buffer.getvalue()


class MonthlyKlineDownloadTests(unittest.TestCase):
    def test_existing_historical_backtest_schedule_still_initializes(self):
        settings = program._get_historical_backtest_settings()
        self.assertIsInstance(settings, dict)
        self.assertEqual(settings["daily_hour"], 3)
        self.assertIn("startup_delay_sec", settings)

    def test_previous_month_uses_complete_utc_calendar_month(self):
        now = dt.datetime(2026, 7, 1, 2, 0, tzinfo=dt.timezone.utc)
        self.assertEqual(monthly_kline_download._previous_utc_month(now), (2026, 6))

    def test_validates_complete_month_and_rejects_truncated_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            full = Path(tmp) / "full.zip"
            full.write_bytes(_monthly_zip_bytes(2026, 6))
            self.assertTrue(backtest._validate_binance_history_zip(full, "ETHUSDT", "5m", 2026, 6))

            truncated = Path(tmp) / "truncated.zip"
            payload = _monthly_zip_bytes(2026, 6)
            truncated.write_bytes(payload[: len(payload) // 2])
            self.assertFalse(backtest._validate_binance_history_zip(truncated, "ETHUSDT", "5m", 2026, 6))

    def test_completed_report_schedules_next_month_instead_of_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            report_path.write_text(json.dumps({"success": True, "target_month": "2026-06"}), encoding="utf-8")
            settings = {"daily_hour": 10, "daily_minute": 0, "startup_delay_sec": 150}
            now = dt.datetime(2026, 7, 17, 4, 0, tzinfo=dt.timezone.utc).timestamp()
            with mock.patch.object(program, "MONTHLY_KLINE_DOWNLOAD_REPORT_PATH", report_path):
                scheduled = program._compute_initial_monthly_kline_ts(settings, now_ts=now)
            self.assertGreater(scheduled, now)
            self.assertGreaterEqual(dt.datetime.fromtimestamp(scheduled).day, 1)


if __name__ == "__main__":
    unittest.main()
