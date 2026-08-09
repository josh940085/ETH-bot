"""Download and validate official Binance USD-M trade archives."""

import datetime as dt
import hashlib
import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

from runtime_paths import data_path


BASE_URL = "https://data.binance.vision/data/futures/um"
SUPPORTED_DATA_TYPES = {"trades", "aggTrades"}
HTTP_SESSION = requests.Session()
HTTP_SESSION.headers.update({"User-Agent": "ETH-bot-trade-history/1.0"})


def _archive_name(symbol, data_type, year, month, day=None):
    suffix = f"{year:04d}-{month:02d}-{day:02d}" if day is not None else f"{year:04d}-{month:02d}"
    return f"{symbol}-{data_type}-{suffix}.zip"


def trade_archive_url(symbol, data_type, year, month, day=None):
    symbol = str(symbol or "BTCUSDT").upper().strip()
    if data_type not in SUPPORTED_DATA_TYPES:
        raise ValueError(f"unsupported Binance trade data type: {data_type}")
    period = "daily" if day is not None else "monthly"
    name = _archive_name(symbol, data_type, year, month, day=day)
    return f"{BASE_URL}/{period}/{data_type}/{symbol}/{name}"


def trade_archive_cache_path(symbol, data_type, year, month, day=None):
    symbol = str(symbol or "BTCUSDT").upper().strip()
    period = "daily" if day is not None else "monthly"
    name = _archive_name(symbol, data_type, year, month, day=day)
    return Path(data_path("historical_trades", "binance_futures_um", period, symbol, data_type, name))


def _parse_checksum(payload):
    match = re.search(r"\b([0-9a-fA-F]{64})\b", str(payload or ""))
    return match.group(1).lower() if match else ""


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_bounds_ms(year, month, day=None):
    start = dt.datetime(year, month, day or 1, tzinfo=dt.timezone.utc)
    if day is not None:
        end = start + dt.timedelta(days=1)
    elif month == 12:
        end = dt.datetime(year + 1, 1, 1, tzinfo=dt.timezone.utc)
    else:
        end = dt.datetime(year, month + 1, 1, tzinfo=dt.timezone.utc)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def read_trade_archive(path, data_type):
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(csv_names) != 1:
            return pd.DataFrame()
        with archive.open(csv_names[0]) as handle:
            payload = handle.read()
    frame = pd.read_csv(io.BytesIO(payload))
    if data_type == "aggTrades":
        columns = [
            "agg_trade_id",
            "price",
            "quantity",
            "first_trade_id",
            "last_trade_id",
            "transact_time",
            "is_buyer_maker",
        ]
        time_column = "transact_time"
    elif data_type == "trades":
        columns = ["trade_id", "price", "quantity", "quote_quantity", "time", "is_buyer_maker"]
        time_column = "time"
    else:
        raise ValueError(f"unsupported Binance trade data type: {data_type}")
    if not {"price", time_column}.issubset(frame.columns):
        frame = pd.read_csv(io.BytesIO(payload), header=None)
        if frame.shape[1] != len(columns):
            return pd.DataFrame()
        frame.columns = columns
    return frame


def validate_trade_archive(path, data_type, year, month, day=None, checksum=""):
    source = Path(path)
    if not source.exists() or source.stat().st_size <= 0:
        return False
    if checksum and _sha256(source) != str(checksum).lower():
        return False
    try:
        with zipfile.ZipFile(source) as archive:
            if archive.testzip() is not None:
                return False
        frame = read_trade_archive(source, data_type)
        time_column = "transact_time" if data_type == "aggTrades" else "time"
        if frame.empty or not {"price", "quantity", time_column}.issubset(frame.columns):
            return False
        timestamps = pd.to_numeric(frame[time_column], errors="coerce").dropna()
        prices = pd.to_numeric(frame["price"], errors="coerce").dropna()
        quantities = pd.to_numeric(frame["quantity"], errors="coerce").dropna()
        if timestamps.empty or prices.empty or quantities.empty or (prices <= 0).any() or (quantities <= 0).any():
            return False
        if float(timestamps.iloc[0]) > 100_000_000_000_000:
            timestamps = timestamps / 1000.0
        start_ms, end_ms = _expected_bounds_ms(year, month, day=day)
        return timestamps.is_monotonic_increasing and start_ms <= float(timestamps.iloc[0]) < end_ms and start_ms < float(timestamps.iloc[-1]) < end_ms
    except Exception:
        return False


def download_trade_archive(symbol, data_type, year, month, day=None):
    cache_path = trade_archive_cache_path(symbol, data_type, year, month, day=day)
    checksum_path = cache_path.with_suffix(cache_path.suffix + ".CHECKSUM")
    cached_checksum = _parse_checksum(checksum_path.read_text(encoding="utf-8")) if checksum_path.exists() else ""
    if cached_checksum and validate_trade_archive(
        cache_path, data_type, year, month, day=day, checksum=cached_checksum
    ):
        return cache_path

    url = trade_archive_url(symbol, data_type, year, month, day=day)
    checksum_response = HTTP_SESSION.get(url + ".CHECKSUM", timeout=30)
    if checksum_response.status_code == 404:
        return None
    checksum_response.raise_for_status()
    expected_checksum = _parse_checksum(checksum_response.text)
    if not expected_checksum:
        raise RuntimeError(f"Binance checksum missing or invalid: {url}.CHECKSUM")

    archive_response = HTTP_SESSION.get(url, timeout=120)
    if archive_response.status_code == 404:
        return None
    archive_response.raise_for_status()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp_path.write_bytes(archive_response.content)
    if not validate_trade_archive(
        tmp_path, data_type, year, month, day=day, checksum=expected_checksum
    ):
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Binance trade archive validation failed: {url}")
    tmp_path.replace(cache_path)
    checksum_path.write_text(checksum_response.text.strip() + "\n", encoding="utf-8")
    return cache_path


def load_trade_day(symbol, day, data_type="aggTrades"):
    day = pd.Timestamp(day, tz="UTC") if not isinstance(day, pd.Timestamp) else day
    if day.tzinfo is None:
        day = day.tz_localize("UTC")
    else:
        day = day.tz_convert("UTC")
    path = download_trade_archive(symbol, data_type, day.year, day.month, day=day.day)
    if path is None:
        return pd.DataFrame()
    frame = read_trade_archive(path, data_type).copy()
    time_column = "transact_time" if data_type == "aggTrades" else "time"
    timestamps = pd.to_numeric(frame[time_column], errors="coerce")
    unit = "us" if timestamps.dropna().iloc[0] > 100_000_000_000_000 else "ms"
    frame["event_time"] = pd.to_datetime(timestamps, unit=unit, utc=True, errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce")
    return frame.dropna(subset=["event_time", "price", "quantity"]).sort_values("event_time")
