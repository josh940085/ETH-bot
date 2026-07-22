#!/usr/bin/env python3
import datetime as dt
import json
import os
from pathlib import Path

from market_history import download_binance_history_zip, validate_binance_history_zip
from runtime_paths import data_path


REPORT_PATH = data_path("monthly_kline_download_latest.json")


def _previous_utc_month(now=None):
    current = now or dt.datetime.now(dt.timezone.utc)
    current = current.astimezone(dt.timezone.utc)
    first = dt.datetime(current.year, current.month, 1, tzinfo=dt.timezone.utc)
    previous_end = first - dt.timedelta(days=1)
    return previous_end.year, previous_end.month


def _csv_env(name, default):
    return [item.strip() for item in str(os.getenv(name, default) or "").split(",") if item.strip()]


def download_previous_month(now=None, report_path=None):
    started_at = dt.datetime.now(dt.timezone.utc)
    year, month = _previous_utc_month(now)
    month_key = f"{year:04d}-{month:02d}"
    symbols = [item.upper() for item in _csv_env("MONTHLY_KLINE_SYMBOLS", "ETHUSDT")]
    intervals = _csv_env("MONTHLY_KLINE_INTERVALS", "5m")
    files = []
    errors = []
    for symbol in symbols:
        for interval in intervals:
            try:
                path = download_binance_history_zip(symbol, interval, year, month)
                if path is None:
                    errors.append(f"{symbol}/{interval}: Binance 月檔尚未發布")
                    continue
                if not validate_binance_history_zip(path, symbol, interval, year, month):
                    errors.append(f"{symbol}/{interval}: 月檔完整性驗證失敗")
                    continue
                files.append({
                    "symbol": symbol,
                    "interval": interval,
                    "path": str(Path(path).resolve()),
                    "bytes": Path(path).stat().st_size,
                })
            except Exception as exc:
                errors.append(f"{symbol}/{interval}: {exc}")
    expected = len(symbols) * len(intervals)
    success = bool(expected > 0 and len(files) == expected and not errors)
    report = {
        "success": success,
        "target_month": month_key,
        "source": "binance_futures_um_monthly",
        "files": files,
        "errors": errors,
        "started_at": started_at.isoformat(),
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    output = Path(report_path or REPORT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(output)
    return report


def main():
    report = download_previous_month()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
