"""Binance historical K-line archive download, validation, and loading."""

import datetime as dt
import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

from runtime_paths import data_path


HTTP_SESSION = requests.Session()
HTTP_SESSION.headers.update({"User-Agent": "ETH-bot-market-history/1.0"})
BINANCE_FUTURES_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"


def _month_start(value):
    return dt.datetime(value.year, value.month, 1, tzinfo=dt.timezone.utc)

def next_month(value):
    if value.month == 12:
        return dt.datetime(value.year + 1, 1, 1, tzinfo=dt.timezone.utc)
    return dt.datetime(value.year, value.month + 1, 1, tzinfo=dt.timezone.utc)

def _iter_months(start_dt, end_dt):
    cursor = _month_start(start_dt)
    while cursor < end_dt:
        yield cursor.year, cursor.month
        cursor = next_month(cursor)

def _iter_days(start_dt, end_dt):
    cursor = dt.datetime(start_dt.year, start_dt.month, start_dt.day, tzinfo=dt.timezone.utc)
    while cursor < end_dt:
        yield cursor.year, cursor.month, cursor.day
        cursor += dt.timedelta(days=1)

def _binance_history_zip_url(symbol, interval, year, month, day=None):
    symbol = str(symbol or "BTCUSDT").upper().strip()
    interval = str(interval)
    if day is not None:
        return (
            "https://data.binance.vision/data/futures/um/daily/klines/"
            f"{symbol}/{interval}/{symbol}-{interval}-{year:04d}-{month:02d}-{day:02d}.zip"
        )
    return (
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        f"{symbol}/{interval}/{symbol}-{interval}-{year:04d}-{month:02d}.zip"
    )

def _binance_history_cache_path(symbol, interval, year, month, day=None):
    symbol = str(symbol or "BTCUSDT").upper().strip()
    interval = str(interval)
    folder = "daily" if day is not None else "monthly"
    suffix = f"{year:04d}-{month:02d}-{day:02d}" if day is not None else f"{year:04d}-{month:02d}"
    return Path(
        data_path(
            "historical_klines",
            "binance_futures_um",
            folder,
            symbol,
            interval,
            f"{symbol}-{interval}-{suffix}.zip",
        )
    )

def download_binance_history_zip(symbol, interval, year, month, day=None):
    cache_path = _binance_history_cache_path(symbol, interval, year, month, day=day)
    if cache_path.exists() and validate_binance_history_zip(cache_path, symbol, interval, year, month, day=day):
        return cache_path

    url = _binance_history_zip_url(symbol, interval, year, month, day=day)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    response = HTTP_SESSION.get(url, timeout=30)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    tmp_path.write_bytes(response.content)
    if not validate_binance_history_zip(tmp_path, symbol, interval, year, month, day=day):
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Binance historical kline archive validation failed: {url}")
    tmp_path.replace(cache_path)
    return cache_path

def _read_binance_history_zip(path):
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if not csv_names:
            return pd.DataFrame()
        with archive.open(csv_names[0]) as fh:
            payload = fh.read()
    frame = pd.read_csv(io.BytesIO(payload))
    required = {"open_time", "open", "high", "low", "close", "volume", "close_time"}
    if required.issubset(frame.columns):
        return frame

    # Binance Futures UM archives before 2022 may be headerless. Re-read them
    # with the official 12-column K-line schema so a 2022 replay can load its
    # preceding warmup bars without treating the first candle as a header.
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    headerless = pd.read_csv(io.BytesIO(payload), header=None)
    if headerless.shape[1] == len(columns):
        headerless.columns = columns
        return headerless
    return frame

def _history_interval_ms(interval):
    match = re.fullmatch(r"(\d+)([mhd])", str(interval or "").strip().lower())
    if not match:
        return 0
    value = int(match.group(1))
    unit_ms = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}[match.group(2)]
    return value * unit_ms


def fetch_klines_from_binance_api(symbol, interval, start_ms, end_ms, *, session=None):
    """Fetch an inclusive historical range from the official Futures K-line API."""
    interval_ms = _history_interval_ms(interval)
    if interval_ms <= 0:
        raise ValueError(f"unsupported Binance Futures interval: {interval}")
    client = session or HTTP_SESSION
    rows = []
    cursor = int(start_ms)
    while cursor <= int(end_ms):
        response = client.get(
            BINANCE_FUTURES_KLINES_URL,
            params={
                "symbol": str(symbol).upper(),
                "interval": str(interval),
                "startTime": cursor,
                "endTime": int(end_ms),
                "limit": 1500,
            },
            timeout=30,
        )
        response.raise_for_status()
        page = response.json()
        if not isinstance(page, list) or not page:
            break
        rows.extend(page)
        next_cursor = int(page[-1][6]) + 1
        if next_cursor <= cursor:
            raise RuntimeError("Binance Futures K-line pagination did not advance")
        cursor = next_cursor
        if len(page) < 1500:
            break
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    raw = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
        ],
    )
    raw = raw.drop_duplicates(subset=["open_time"]).sort_values("open_time")
    raw["close_time"] = pd.to_datetime(raw["close_time"], unit="ms", utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        raw[column] = pd.to_numeric(raw[column], errors="raise").astype("float64")
    frame = raw.set_index("close_time")[["open", "high", "low", "close", "volume"]]
    frame.attrs["kline_source"] = "binance_futures_api"
    return frame

def validate_binance_history_zip(path, symbol, interval, year, month, day=None):
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                return False
        frame = _read_binance_history_zip(path)
        required = {"open_time", "open", "high", "low", "close", "volume", "close_time"}
        if frame.empty or not required.issubset(frame.columns):
            return False
        open_times = pd.to_numeric(frame["open_time"], errors="coerce").dropna()
        if open_times.empty:
            return False
        # Binance archives may use microseconds; normalize only for validation.
        if float(open_times.iloc[0]) > 100_000_000_000_000:
            open_times = open_times / 1000.0
        start_dt = dt.datetime(year, month, day or 1, tzinfo=dt.timezone.utc)
        end_dt = start_dt + dt.timedelta(days=1) if day is not None else next_month(start_dt)
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        interval_ms = _history_interval_ms(interval)
        if abs(float(open_times.iloc[0]) - start_ms) > max(interval_ms, 60_000):
            return False
        if float(open_times.iloc[-1]) >= end_ms or float(open_times.iloc[-1]) < end_ms - max(interval_ms * 2, 120_000):
            return False
        if interval_ms > 0:
            expected = max(1, (end_ms - start_ms) // interval_ms)
            if len(open_times) < int(expected * 0.995):
                return False
        return True
    except Exception:
        return False

def fetch_klines_from_binance_history(symbol, interval, start_ms, end_ms):
    if interval != "5m":
        raise RuntimeError(f"Binance history source only supports base 5m backtests, got {interval}")
    start_dt = dt.datetime.fromtimestamp(int(start_ms) / 1000, tz=dt.timezone.utc)
    end_dt = dt.datetime.fromtimestamp(int(end_ms) / 1000, tz=dt.timezone.utc)
    frames = []
    missing = []
    downloaded = 0
    daily_loaded = 0
    for year, month in _iter_months(start_dt, end_dt):
        path = download_binance_history_zip(symbol, interval, year, month)
        if path is None:
            month_start = dt.datetime(year, month, 1, tzinfo=dt.timezone.utc)
            month_end = next_month(month_start)
            day_start = max(start_dt, month_start)
            day_end = min(end_dt, month_end)
            for day_year, day_month, day in _iter_days(day_start, day_end):
                day_path = download_binance_history_zip(symbol, interval, day_year, day_month, day=day)
                if day_path is None:
                    missing.append(f"{day_year:04d}-{day_month:02d}-{day:02d}")
                    continue
                daily_loaded += 1
                frame = _read_binance_history_zip(day_path)
                if not frame.empty:
                    frames.append(frame)
            continue
        downloaded += 1
        frame = _read_binance_history_zip(path)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise RuntimeError(f"No Binance historical monthly klines found; missing={','.join(missing[:6])}")

    raw = pd.concat(frames, ignore_index=True)
    required = ["open_time", "open", "high", "low", "close", "volume", "close_time"]
    if not set(required).issubset(set(raw.columns)):
        raise RuntimeError(f"Binance historical CSV columns invalid: {list(raw.columns)}")
    raw = raw[(raw["open_time"] >= int(start_ms)) & (raw["open_time"] <= int(end_ms))]
    if raw.empty:
        raise RuntimeError("Binance historical monthly files loaded but no rows matched requested range")
    raw = raw.drop_duplicates(subset=["open_time"]).sort_values("open_time")
    raw["close_time"] = pd.to_datetime(raw["close_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        raw[col] = raw[col].astype(float)
    frame = raw.set_index("close_time")[["open", "high", "low", "close", "volume"]]
    source_parts = []
    if downloaded:
        source_parts.append(f"{downloaded}m")
    if daily_loaded:
        source_parts.append(f"{daily_loaded}d")
    frame.attrs["kline_source"] = f"binance_history_um:{'+'.join(source_parts) or 'cached'}"
    if missing:
        frame.attrs["kline_missing_months"] = missing
    return frame
