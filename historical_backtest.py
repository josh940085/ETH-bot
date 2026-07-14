#!/usr/bin/env python3
"""Run the panel's long-range backtests sequentially and publish only complete results."""

import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from telegram import REPO_DIR, data_path


BACKTEST_FILE = REPO_DIR / "backtest.py"
REPORT_PATH = data_path("historical_backtest_latest_report.json")
BACKTEST_DIR = data_path("backtests")
PERIODS = (
    ("2022", "2022-01-01", "2023-01-01", "backtest_2022_market_profile_try3"),
    ("2023", "2023-01-01", "2024-01-01", "backtest_2023_market_profile_try3"),
    ("2024", "2024-01-01", "2025-01-01", "backtest_2024_market_profile_try3"),
    ("2025", "2025-01-01", "2026-01-01", "backtest_2025_market_profile_try5"),
    ("2026H1", "2026-01-01", "2026-07-01", "backtest_2026h1_market_profile_try5"),
)


def _write_report(payload):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = REPORT_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, REPORT_PATH)


def main():
    started_at = dt.datetime.now().astimezone().isoformat()
    results = []
    report = {"started_at": started_at, "finished_at": None, "success": False, "periods": results}
    _write_report(report)
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="historical-backtest-", dir=str(BACKTEST_DIR)) as temp_dir:
        temp_dir = Path(temp_dir)
        for label, start, end, basename in PERIODS:
            summary_temp = temp_dir / f"{basename}_summary.json"
            trades_temp = temp_dir / f"{basename}_trades.csv"
            cmd = [
                sys.executable,
                str(BACKTEST_FILE),
                "--start", start,
                "--end", end,
                "--warmup-bars", os.getenv("HISTORICAL_BACKTEST_WARMUP_BARS", "1500"),
                "--data-source", os.getenv("HISTORICAL_BACKTEST_DATA_SOURCE", "binance-history"),
                "--summary-out", str(summary_temp),
                "--trades-out", str(trades_temp),
            ]
            print(f"📚 歷史回測 {label}: {' '.join(cmd)}", flush=True)
            result = subprocess.run(cmd, cwd=str(REPO_DIR), env=os.environ.copy(), check=False)
            period_result = {"period": label, "start": start, "end": end, "exit_code": result.returncode}
            results.append(period_result)
            if result.returncode != 0 or not summary_temp.exists() or not trades_temp.exists():
                report["failed_period"] = label
                report["finished_at"] = dt.datetime.now().astimezone().isoformat()
                _write_report(report)
                return result.returncode or 1

            os.replace(summary_temp, BACKTEST_DIR / summary_temp.name)
            os.replace(trades_temp, BACKTEST_DIR / trades_temp.name)
            period_result["published"] = True
            _write_report(report)

    report["success"] = True
    report["finished_at"] = dt.datetime.now().astimezone().isoformat()
    _write_report(report)
    print("✅ 歷年回測全部完成並更新 App 資料", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
