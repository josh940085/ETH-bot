#!/usr/bin/env python3
import argparse
import bisect
import datetime as dt
import gzip
import io
import json
import os
import re
import warnings
import zipfile
from contextlib import contextmanager
from pathlib import Path

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+.*")

try:
    from urllib3.exceptions import NotOpenSSLWarning
except Exception:  # pragma: no cover - urllib3 variant fallback
    NotOpenSSLWarning = None

if NotOpenSSLWarning is not None:
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

import pandas as pd

os.environ.setdefault("ETH_BOT_DISABLE_LIVE", "1")

import eth
from mlx_learning import build_trade_factor_tags


INTERVAL_MS = {
    "5m": 5 * 60 * 1000,
}


NEWS_CATEGORY_PATTERNS = {
    "地緣政治/戰爭": re.compile(
        r"iran|israel|war|strike|ceasefire|russia|ukraine|conflict|missile|tariff|sanction|section 232|geopolitical",
        re.I,
    ),
    "央行/利率/CPI": re.compile(
        r"fed|fomc|powell|rate|rates|inflation|cpi|ppi|jobs|payroll|unemployment|yield|treasury",
        re.I,
    ),
    "BTC/ETH價格快訊": re.compile(r"bitcoin|btc|ether|ethereum|eth|crypto|cryptocurrency|token|coin", re.I),
    "監管/法律": re.compile(r"sec|cftc|regulat|lawsuit|court|legal|senate|congress|etf|probe|investigation", re.I),
    "交易所/平台": re.compile(r"binance|coinbase|bybit|kraken|okx|exchange|wallet|stablecoin|tether|usdt|circle", re.I),
    "股市大盤": re.compile(r"u\\.s\\. stocks|us stocks|dow jones|s&p|nasdaq|russell|stocks higher|stocks lower|futures|wall street", re.I),
    "科技股/AI": re.compile(r"nvidia|apple|microsoft|tesla|meta|amazon|google|alphabet|ai |semiconductor|chip|spacex", re.I),
    "油價/能源": re.compile(r"oil|crude|brent|wti|energy|opec|gas", re.I),
    "券商調評/個股目標價": re.compile(r"raises .*price target|cuts .*price target|stock price target|upgrades|downgrades|rating", re.I),
    "財報/指引": re.compile(r"earnings|revenue|profit|guidance|forecast|q[1-4]|results", re.I),
}

HIGH_IMPACT_NEWS_CATEGORIES = {
    "地緣政治/戰爭",
    "央行/利率/CPI",
    "BTC/ETH價格快訊",
    "監管/法律",
    "交易所/平台",
    "股市大盤",
}


def _parse_args():
    parser = argparse.ArgumentParser(description="Replay ETH-bot strategy on historical market klines.")
    parser.add_argument("--symbol", default="ETHUSDT", help="Trading symbol, e.g. ETHUSDT")
    parser.add_argument("--days", type=int, default=30, help="Lookback days when start/end are not provided")
    parser.add_argument("--start", help="UTC start time, e.g. 2026-05-01 or 2026-05-01T00:00:00")
    parser.add_argument("--end", help="UTC end time, e.g. 2026-05-22 or 2026-05-22T00:00:00")
    parser.add_argument("--warmup-bars", type=int, default=1500, help="5m warmup bars before evaluating signals")
    parser.add_argument(
        "--data-source",
        choices=("auto", "binance-history", "tradingview"),
        default=os.getenv("BACKTEST_DATA_SOURCE", "auto"),
        help="Historical kline source. auto uses Binance official monthly archives first.",
    )
    parser.add_argument("--trades-out", help="Optional CSV path for trade log export")
    parser.add_argument("--summary-out", help="Optional JSON path for summary export")
    parser.add_argument("--learn-out", help="Optional CSV path for AI learning sample export")
    return parser.parse_args()


def _parse_utc(raw):
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.lower() in {"last-month", "previous-month", "上個月"}:
        now = dt.datetime.now(dt.timezone.utc)
        return dt.datetime(now.year, now.month, 1, tzinfo=dt.timezone.utc)
    if "T" in text:
        dt_obj = dt.datetime.fromisoformat(text)
    else:
        dt_obj = dt.datetime.fromisoformat(f"{text}T00:00:00")
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)
    else:
        dt_obj = dt_obj.astimezone(dt.timezone.utc)
    return dt_obj


def _resolve_timerange(args):
    start_dt = _parse_utc(args.start)
    end_dt = _parse_utc(args.end)
    if end_dt is None:
        now = dt.datetime.now(dt.timezone.utc)
        if start_dt is not None:
            end_dt = dt.datetime(now.year, now.month, 1, tzinfo=dt.timezone.utc)
        else:
            end_dt = now
    if start_dt is None:
        start_dt = end_dt - dt.timedelta(days=max(1, int(args.days)))
    if start_dt >= end_dt:
        raise SystemExit("start must be earlier than end")
    return start_dt, end_dt


def _month_start(value):
    return dt.datetime(value.year, value.month, 1, tzinfo=dt.timezone.utc)


def _next_month(value):
    if value.month == 12:
        return dt.datetime(value.year + 1, 1, 1, tzinfo=dt.timezone.utc)
    return dt.datetime(value.year, value.month + 1, 1, tzinfo=dt.timezone.utc)


def _iter_months(start_dt, end_dt):
    cursor = _month_start(start_dt)
    while cursor < end_dt:
        yield cursor.year, cursor.month
        cursor = _next_month(cursor)


def _iter_days(start_dt, end_dt):
    cursor = dt.datetime(start_dt.year, start_dt.month, start_dt.day, tzinfo=dt.timezone.utc)
    while cursor < end_dt:
        yield cursor.year, cursor.month, cursor.day
        cursor += dt.timedelta(days=1)


def _binance_history_zip_url(symbol, interval, year, month, day=None):
    symbol = str(symbol or "ETHUSDT").upper().strip()
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
    symbol = str(symbol or "ETHUSDT").upper().strip()
    interval = str(interval)
    folder = "daily" if day is not None else "monthly"
    suffix = f"{year:04d}-{month:02d}-{day:02d}" if day is not None else f"{year:04d}-{month:02d}"
    return Path(
        eth.data_path(
            "historical_klines",
            "binance_futures_um",
            folder,
            symbol,
            interval,
            f"{symbol}-{interval}-{suffix}.zip",
        )
    )


def _download_binance_history_zip(symbol, interval, year, month, day=None):
    cache_path = _binance_history_cache_path(symbol, interval, year, month, day=day)
    if cache_path.exists() and _validate_binance_history_zip(cache_path, symbol, interval, year, month, day=day):
        return cache_path

    url = _binance_history_zip_url(symbol, interval, year, month, day=day)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    response = eth.HTTP_SESSION.get(url, timeout=30)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    tmp_path.write_bytes(response.content)
    if not _validate_binance_history_zip(tmp_path, symbol, interval, year, month, day=day):
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
            return pd.read_csv(io.BytesIO(fh.read()))


def _history_interval_ms(interval):
    match = re.fullmatch(r"(\d+)([mhd])", str(interval or "").strip().lower())
    if not match:
        return 0
    value = int(match.group(1))
    unit_ms = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}[match.group(2)]
    return value * unit_ms


def _validate_binance_history_zip(path, symbol, interval, year, month, day=None):
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
        end_dt = start_dt + dt.timedelta(days=1) if day is not None else _next_month(start_dt)
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


def _fetch_klines_from_binance_history(symbol, interval, start_ms, end_ms):
    if interval != "5m":
        raise RuntimeError(f"Binance history source only supports base 5m backtests, got {interval}")
    start_dt = dt.datetime.fromtimestamp(int(start_ms) / 1000, tz=dt.timezone.utc)
    end_dt = dt.datetime.fromtimestamp(int(end_ms) / 1000, tz=dt.timezone.utc)
    frames = []
    missing = []
    downloaded = 0
    daily_loaded = 0
    for year, month in _iter_months(start_dt, end_dt):
        path = _download_binance_history_zip(symbol, interval, year, month)
        if path is None:
            month_start = dt.datetime(year, month, 1, tzinfo=dt.timezone.utc)
            month_end = _next_month(month_start)
            day_start = max(start_dt, month_start)
            day_end = min(end_dt, month_end)
            for day_year, day_month, day in _iter_days(day_start, day_end):
                day_path = _download_binance_history_zip(symbol, interval, day_year, day_month, day=day)
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


def _fetch_klines_from_market_source(symbol, interval, start_ms, end_ms, limit=1500):
    step_ms = INTERVAL_MS[interval]
    required_bars = int(max(1, (int(end_ms) - int(start_ms)) // step_ms + 8))
    rows, source_name = eth._fetch_market_kline_rows(
        symbol,
        interval,
        limit=max(required_bars, int(limit)),
        start_time_ms=start_ms,
        end_time_ms=end_ms,
        timeout=20,
        prefix="回測K線",
    )
    all_rows = rows if isinstance(rows, list) else []

    if not all_rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    frame = pd.DataFrame(
        all_rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    frame = frame.drop_duplicates(subset=["open_time"]).sort_values("open_time")
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = frame[col].astype(float)
    frame = frame.set_index("close_time")[["open", "high", "low", "close", "volume"]]
    frame.attrs["kline_source"] = source_name
    return frame


def fetch_futures_klines(symbol, interval, start_ms, end_ms, limit=1500, data_source="auto"):
    source = str(data_source or "auto").strip().lower()
    errors = []
    if source in {"auto", "binance-history"}:
        try:
            frame = _fetch_klines_from_binance_history(symbol, interval, start_ms, end_ms)
            if not frame.empty:
                if source == "auto":
                    last_close_ms = int(frame.index.max().timestamp() * 1000)
                    interval_ms = INTERVAL_MS.get(interval, 5 * 60 * 1000)
                    if end_ms - last_close_ms > interval_ms:
                        try:
                            tail = _fetch_klines_from_market_source(
                                symbol,
                                interval,
                                last_close_ms + 1,
                                end_ms,
                                limit=max(12, int((end_ms - last_close_ms) // interval_ms) + 8),
                            )
                            if not tail.empty:
                                frame = pd.concat([frame, tail]).sort_index()
                                frame = frame[~frame.index.duplicated(keep="last")]
                                frame.attrs["kline_source"] = f"{frame.attrs.get('kline_source', 'binance_history')}+{tail.attrs.get('kline_source', 'market_tail')}"
                        except Exception as tail_exc:
                            print(f"⚠️ Binance歷史資料尾段未補齊，市場尾段補齊失敗: {tail_exc}")
                print(f"📈 回測K線來源: {frame.attrs.get('kline_source', 'binance_history')}")
                return frame
        except Exception as exc:
            errors.append(f"binance-history: {exc}")
            if source == "binance-history":
                raise SystemExit(f"No klines returned from Binance historical data source: {exc}") from exc
    if source in {"auto", "tradingview"}:
        try:
            frame = _fetch_klines_from_market_source(symbol, interval, start_ms, end_ms, limit=limit)
        except Exception as exc:
            errors.append(f"tradingview: {exc}")
            raise SystemExit(f"No klines returned from market data sources: {'; '.join(errors)}") from exc
        if not frame.empty:
            print(f"📈 回測K線來源: {frame.attrs.get('kline_source', 'unknown')}")
            return frame
    raise SystemExit(f"No klines returned from market data sources: {'; '.join(errors)}")


def resample_ohlcv(frame, rule):
    agg = (
        frame.resample(rule, label="right", closed="right")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )
    return agg


def _safe_change(frame, ts, lookback):
    if frame is None or frame.empty:
        return 0.0
    try:
        current_slice = frame.loc[:ts]
        previous_slice = frame.loc[: ts - lookback]
    except Exception:
        return 0.0
    if current_slice.empty or previous_slice.empty:
        return 0.0
    current = float(current_slice["close"].iloc[-1])
    previous = float(previous_slice["close"].iloc[-1])
    if current <= 0 or previous <= 0:
        return 0.0
    return (current - previous) / previous


def _build_change_index(frame):
    if frame is None or frame.empty or "close" not in frame.columns:
        return None
    work = frame[["close"]].dropna()
    if work.empty:
        return None
    return work.index, work["close"].astype(float).to_numpy()


def _safe_change_from_index(change_index, ts, lookback):
    if not change_index:
        return 0.0
    index, closes = change_index
    try:
        current_pos = int(index.searchsorted(ts, side="right")) - 1
        previous_pos = int(index.searchsorted(ts - lookback, side="right")) - 1
    except Exception:
        return 0.0
    if current_pos < 0 or previous_pos < 0:
        return 0.0
    current = float(closes[current_pos])
    previous = float(closes[previous_pos])
    if current <= 0 or previous <= 0:
        return 0.0
    return (current - previous) / previous


def _slice_frame_until(frame, index, ts, limit):
    if frame is None or frame.empty:
        return frame
    try:
        end_pos = int(index.searchsorted(ts, side="right"))
    except Exception:
        return frame.loc[:ts].tail(limit)
    if end_pos <= 0:
        return frame.iloc[:0]
    start_pos = max(0, end_pos - int(limit))
    return frame.iloc[start_pos:end_pos]


def _fetch_yahoo_chart_frame(symbol, start_dt, end_dt, interval="5m"):
    period1 = int((start_dt - dt.timedelta(days=2)).timestamp())
    period2 = int(end_dt.timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": period1,
        "period2": period2,
        "interval": interval,
        "includePrePost": "true",
        "events": "history",
    }
    response = eth.HTTP_SESSION.get(url, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    result = ((payload.get("chart") or {}).get("result") or [None])[0]
    if not isinstance(result, dict):
        raise RuntimeError("Yahoo chart empty result")
    timestamps = result.get("timestamp") or []
    quote = (((result.get("indicators") or {}).get("quote") or [None])[0]) or {}
    if not timestamps or not quote:
        raise RuntimeError("Yahoo chart empty candles")
    frame = pd.DataFrame(
        {
            "open": quote.get("open") or [],
            "high": quote.get("high") or [],
            "low": quote.get("low") or [],
            "close": quote.get("close") or [],
            "volume": quote.get("volume") or [],
        },
        index=pd.to_datetime(timestamps, unit="s", utc=True),
    )
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame.attrs["kline_source"] = f"yahoo:{symbol}"
    return frame


def _fetch_optional_macro_frame(symbol, start_dt, end_dt, *, data_source="tradingview"):
    start_ms = int((start_dt - dt.timedelta(days=2)).timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    try:
        if str(symbol).upper().endswith("USDT"):
            return fetch_futures_klines(symbol, "5m", start_ms, end_ms, data_source="auto")
        yahoo_symbol = {
            "ES1!": "ES=F",
            "NQ1!": "NQ=F",
            "DXY": "DX-Y.NYB",
        }.get(str(symbol).upper())
        requested_days = max(0.0, (end_dt - start_dt).total_seconds() / 86400.0)
        max_external_days = max(
            7.0,
            eth._safe_float(os.getenv("BACKTEST_EXTERNAL_MACRO_MAX_5M_DAYS", 60), 60),
        )
        if yahoo_symbol and requested_days > max_external_days:
            print(
                f"⚠️ 長窗口歷史宏觀略過 {symbol}: "
                f"{requested_days:.0f}天超過外部5m資料上限{max_external_days:.0f}天"
            )
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        if yahoo_symbol and eth._is_truthy(os.getenv("BACKTEST_MACRO_YAHOO_FIRST", "1")):
            try:
                return _fetch_yahoo_chart_frame(yahoo_symbol, start_dt, end_dt)
            except Exception as yahoo_exc:
                print(f"⚠️ Yahoo歷史宏觀資料失敗 {yahoo_symbol}: {yahoo_exc}")
        return _fetch_klines_from_market_source(symbol, "5m", start_ms, end_ms, limit=12000)
    except Exception as exc:
        print(f"⚠️ 歷史宏觀資料略過 {symbol}: {exc}")
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def _categorize_news_text(text):
    text = str(text or "")
    categories = [name for name, pattern in NEWS_CATEGORY_PATTERNS.items() if pattern.search(text)]
    return categories or ["其他"]


def _iter_news_prediction_files():
    base = Path(eth.NEWS_PERFORMANCE_LOG)
    candidates = [base]
    candidates.extend(sorted(base.parent.glob(base.name + ".*.gz")))
    for path in candidates:
        if path.exists() and path.is_file():
            yield path


def _load_historical_news_events(start_dt, end_dt):
    if not eth._is_truthy(os.getenv("BACKTEST_HISTORICAL_NEWS_ENABLED", "1")):
        return []
    events = []
    hourly_buckets = {}
    aggregate_hourly = eth._is_truthy(os.getenv("BACKTEST_HISTORICAL_NEWS_AGGREGATE_HOURLY", "1"))
    scan_gzip = eth._is_truthy(os.getenv("BACKTEST_HISTORICAL_NEWS_SCAN_GZIP", "1"))
    start_scan = start_dt - dt.timedelta(hours=8)
    end_scan = end_dt + dt.timedelta(hours=1)
    for path in _iter_news_prediction_files():
        if path.suffix == ".gz" and not scan_gzip:
            continue
        opener = gzip.open if path.suffix == ".gz" else open
        try:
            with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    raw_ts = item.get("timestamp")
                    if not raw_ts:
                        continue
                    try:
                        news_ts = dt.datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
                    except Exception:
                        continue
                    if news_ts.tzinfo is None:
                        news_ts = news_ts.replace(tzinfo=dt.timezone.utc)
                    news_ts = news_ts.astimezone(dt.timezone.utc)
                    if news_ts < start_scan or news_ts > end_scan:
                        continue
                    text = str(item.get("news") or item.get("news_key") or "").strip()
                    if not text:
                        continue
                    predicted_bias = eth._safe_float(item.get("predicted_bias"), 0.0)
                    confidence = max(0.0, min(1.0, eth._safe_float(item.get("ai_confidence"), 0.35)))
                    categories = (
                        _categorize_news_text(text)
                        if abs(predicted_bias) >= 0.20 or confidence >= 0.45
                        else ["其他"]
                    )
                    if aggregate_hourly:
                        bucket_ts = news_ts.replace(minute=0, second=0, microsecond=0)
                        bucket = hourly_buckets.setdefault(
                            bucket_ts,
                            {
                                "weighted": 0.0,
                                "total_weight": 0.0,
                                "confidence": 0.0,
                                "categories": set(),
                                "count": 0,
                                "max_abs_bias": 0.0,
                            },
                        )
                        weight = max(0.1, confidence)
                        bucket["weighted"] += predicted_bias * weight
                        bucket["total_weight"] += weight
                        bucket["confidence"] = max(bucket["confidence"], confidence)
                        bucket["count"] += 1
                        bucket["max_abs_bias"] = max(bucket["max_abs_bias"], abs(predicted_bias))
                        for category in categories:
                            if category != "其他":
                                bucket["categories"].add(category)
                        continue
                    events.append(
                        {
                            "ts": news_ts,
                            "bias": predicted_bias,
                            "confidence": confidence,
                            "categories": categories,
                            "text": text,
                        }
                    )
        except Exception as exc:
            print(f"⚠️ 歷史新聞資料略過 {path.name}: {exc}")
    if aggregate_hourly:
        events = []
        for bucket_ts, bucket in hourly_buckets.items():
            total_weight = float(bucket.get("total_weight", 0.0))
            bias = (float(bucket.get("weighted", 0.0)) / total_weight) if total_weight > 0 else 0.0
            categories = sorted(bucket.get("categories") or []) or ["其他"]
            events.append(
                {
                    "ts": bucket_ts,
                    "bias": max(-2.0, min(2.0, bias)),
                    "confidence": max(0.1, min(1.0, float(bucket.get("confidence", 0.35)))),
                    "categories": categories,
                    "text": f"hourly_news_batch:{int(bucket.get('count', 0))}",
                }
            )
    events.sort(key=lambda item: item["ts"])
    return events


class HistoricalMacroContext:
    def __init__(self, start_dt, end_dt):
        self.enabled = eth._is_truthy(os.getenv("BACKTEST_HISTORICAL_MACRO_ENABLED", "1"))
        self.frames = {}
        self.change_indexes = {}
        self.news_events = _load_historical_news_events(start_dt, end_dt)
        self.news_event_ts = [event["ts"] for event in self.news_events]
        self.news_event_count = len(self.news_events)
        if not self.enabled:
            return
        symbols = {
            "btc": "BTCUSDT",
            "sp": os.getenv("BACKTEST_SP_SYMBOL", "ES1!"),
            "nq": os.getenv("BACKTEST_NQ_SYMBOL", "NQ1!"),
            "dxy": os.getenv("BACKTEST_DXY_SYMBOL", "DXY"),
        }
        for key, symbol in symbols.items():
            frame = _fetch_optional_macro_frame(symbol, start_dt, end_dt)
            if not frame.empty:
                self.frames[key] = frame
                change_index = _build_change_index(frame)
                if change_index is not None:
                    self.change_indexes[key] = change_index

    def snapshot(self, ts):
        lookback = dt.timedelta(hours=max(1.0, eth._safe_float(os.getenv("BACKTEST_MACRO_LOOKBACK_HOURS", 24), 24)))
        sp_change = _safe_change_from_index(self.change_indexes.get("sp"), ts, lookback)
        nq_change = _safe_change_from_index(self.change_indexes.get("nq"), ts, lookback)
        btc_change = _safe_change_from_index(self.change_indexes.get("btc"), ts, lookback)
        dxy_change = _safe_change_from_index(self.change_indexes.get("dxy"), ts, lookback)
        news_bias, event_risk, categories = self._news_snapshot(ts)
        return {
            "sp_change": sp_change,
            "nq_change": nq_change,
            "btc_change": btc_change,
            "dxy_change": dxy_change,
            "news_bias": news_bias,
            "event_risk": event_risk,
            "news_categories": categories,
        }

    def _news_snapshot(self, ts):
        if not self.news_events:
            return 0.0, 0, []
        window_hours = max(1.0, eth._safe_float(os.getenv("BACKTEST_NEWS_LOOKBACK_HOURS", 6), 6))
        start_ts = ts - dt.timedelta(hours=window_hours)
        start_idx = bisect.bisect_left(self.news_event_ts, start_ts)
        end_idx = bisect.bisect_right(self.news_event_ts, ts)
        weighted = 0.0
        total_weight = 0.0
        event_risk = 0
        categories = []
        for event in self.news_events[start_idx:end_idx]:
            weight = max(0.1, float(event.get("confidence", 0.35)))
            bias = max(-2.0, min(2.0, float(event.get("bias", 0.0))))
            weighted += bias * weight
            total_weight += weight
            for category in event.get("categories") or []:
                if category not in categories:
                    categories.append(category)
                if category in HIGH_IMPACT_NEWS_CATEGORIES:
                    event_risk = max(event_risk, 1)
            if abs(bias) >= 2.0:
                event_risk = max(event_risk, 2)
        if total_weight <= 0:
            return 0.0, 0, []
        news_bias = max(-2.0, min(2.0, weighted / total_weight))
        return news_bias, event_risk, categories[:6]

    def summary(self):
        return {
            "enabled": bool(self.enabled),
            "frames_loaded": sorted(self.frames.keys()),
            "news_events": int(self.news_event_count),
        }


def build_frame_map(base_5m):
    frame_map = {
        "5m": eth.calc_indicators(base_5m.copy()),
        "15m": eth.calc_indicators(resample_ohlcv(base_5m, "15min")),
        "30m": eth.calc_indicators(resample_ohlcv(base_5m, "30min")),
        "1h": eth.calc_indicators(resample_ohlcv(base_5m, "1h")),
        "4h": eth.calc_indicators(resample_ohlcv(base_5m, "4h")),
        "12h": eth.calc_indicators(resample_ohlcv(base_5m, "12h")),
        "1d": eth.calc_indicators(resample_ohlcv(base_5m, "1D")),
        "1w": eth.calc_indicators(resample_ohlcv(base_5m, "7D")),
        "1M": eth.calc_indicators(resample_ohlcv(base_5m, "30D")),
    }
    return frame_map


def compute_max_drawdown(equity_curve):
    peak = 1.0
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, 1.0 - (value / peak))
    return max_dd


def _summarize_grouped_trades(df, column):
    if column not in df.columns or df.empty:
        return {}

    grouped = {}
    for raw_key, group in df.groupby(column, dropna=False):
        key = str(raw_key if pd.notna(raw_key) else "unknown")
        returns = pd.to_numeric(group["trade_return"], errors="coerce").fillna(0.0)
        count = int(len(group))
        wins = int((returns > 0).sum())
        grouped[key] = {
            "trades": count,
            "win_rate": round((wins / count) * 100, 2) if count else 0.0,
            "total_return_pct": round(float(returns.sum()) * 100, 3),
            "avg_return_pct": round(float(returns.mean()) * 100, 3) if count else 0.0,
        }
    return grouped


def _summarize_mlx_factors(df):
    if "mlx_factor_tags" not in df.columns or df.empty:
        return {}
    expanded = []
    for _, row in df.iterrows():
        try:
            factors = json.loads(row.get("mlx_factor_tags") or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            factors = []
        if not isinstance(factors, list):
            continue
        for factor in factors:
            clean = str(factor).strip()
            if clean:
                expanded.append(
                    {
                        "factor": clean,
                        "trade_return": row.get("trade_return", 0.0),
                    }
                )
    if not expanded:
        return {}
    factor_df = pd.DataFrame(expanded)
    return _summarize_grouped_trades(factor_df, "factor")


def _iter_backtest_due_dates(start_dt, end_dt):
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=dt.timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=dt.timezone.utc)
    due_hour = min(23, max(0, eth._safe_int(os.getenv("DAILY_MIN_TRADE_HOUR", 22), 22)))
    due_minute = min(59, max(0, eth._safe_int(os.getenv("DAILY_MIN_TRADE_MINUTE", 30), 30)))
    cursor = start_dt.astimezone(TAIPEI_TZ).date()
    end_date = end_dt.astimezone(TAIPEI_TZ).date()
    while cursor <= end_date:
        due_local = dt.datetime(cursor.year, cursor.month, cursor.day, due_hour, due_minute, tzinfo=TAIPEI_TZ)
        due_utc = due_local.astimezone(dt.timezone.utc)
        if start_dt <= due_utc < end_dt:
            yield cursor.isoformat()
        cursor += dt.timedelta(days=1)


def _summarize_trade_day_coverage(df, start_dt, end_dt):
    all_dates = list(_iter_backtest_due_dates(start_dt, end_dt))
    due_date_set = set(all_dates)
    if df.empty or "opened_at" not in df.columns:
        trade_dates = set()
    else:
        opened = pd.to_datetime(df["opened_at"], errors="coerce", utc=True).dropna()
        trade_dates = {
            trade_date
            for trade_date in (ts.tz_convert(TAIPEI_TZ).date().isoformat() for ts in opened)
            if trade_date in due_date_set
        }
    missing = [value for value in all_dates if value not in trade_dates]
    daily_min_trades = 0
    if not df.empty and "host_logic_mode" in df.columns:
        daily_min_trades = int((df["host_logic_mode"].astype(str) == "daily_minimum").sum())
    return {
        "calendar_days": int(len(all_dates)),
        "trade_days": int(len(trade_dates)),
        "missing_trade_days": int(len(missing)),
        "missing_trade_day_examples": missing[:10],
        "daily_min_trades": daily_min_trades,
    }


def summarize_trades(trades, start_dt, end_dt, symbol, model_loaded, data_source="futures", coverage_start_dt=None):
    coverage_start_dt = coverage_start_dt or start_dt
    if not trades:
        day_coverage = _summarize_trade_day_coverage(pd.DataFrame(), coverage_start_dt, end_dt)
        return {
            "symbol": symbol,
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "data_source": data_source,
            "strategy_version": eth.STRATEGY_VERSION,
            "model_loaded": bool(model_loaded),
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_return_pct": 0.0,
            "avg_trade_return_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "final_equity": 1.0,
            "long_trades": 0,
            "short_trades": 0,
            "exit_reason_counts": {},
            "expectancy_pct": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "avg_mfe_pct": 0.0,
            "avg_mae_pct": 0.0,
            "avg_rr_at_entry": 0.0,
            "avg_net_edge_rate_pct": 0.0,
            "by_regime": {},
            "by_direction": {},
            "by_strategy_version": {},
            "by_mlx_factor": {},
            "trade_day_coverage": day_coverage,
        }

    df = pd.DataFrame(trades)
    returns = pd.to_numeric(df["trade_return"], errors="coerce").fillna(0.0)
    wins = int((returns > 0).sum())
    losses = int((returns <= 0).sum())
    gross_profit = float(returns[returns > 0].sum())
    gross_loss = float(-returns[returns <= 0].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    equity_curve = df["equity"].tolist()
    exit_reason_counts = {str(k): int(v) for k, v in df["exit_reason"].value_counts().to_dict().items()}
    win_returns = returns[returns > 0]
    loss_returns = returns[returns <= 0]
    mfe = pd.to_numeric(df.get("max_favorable_move_pct", pd.Series(dtype=float)), errors="coerce")
    mae = pd.to_numeric(df.get("max_adverse_move_pct", pd.Series(dtype=float)), errors="coerce")
    rr_at_entry = pd.to_numeric(df.get("rr_at_entry", pd.Series(dtype=float)), errors="coerce")
    net_edge = pd.to_numeric(df.get("net_edge_rate_est_pct", pd.Series(dtype=float)), errors="coerce")

    return {
        "symbol": symbol,
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "data_source": data_source,
        "strategy_version": eth.STRATEGY_VERSION,
        "model_loaded": bool(model_loaded),
        "trades": int(len(df)),
        "wins": wins,
        "losses": losses,
        "win_rate": round((wins / len(df)) * 100, 2),
        "total_return_pct": round((float(df["equity"].iloc[-1]) - 1.0) * 100, 2),
        "avg_trade_return_pct": round(float(returns.mean()) * 100, 3),
        "profit_factor": None if profit_factor == float("inf") else round(profit_factor, 3),
        "max_drawdown_pct": round(compute_max_drawdown(equity_curve) * 100, 2),
        "final_equity": round(float(df["equity"].iloc[-1]), 6),
        "long_trades": int((df["direction"] == "long").sum()),
        "short_trades": int((df["direction"] == "short").sum()),
        "exit_reason_counts": exit_reason_counts,
        "expectancy_pct": round(float(returns.mean()) * 100, 3),
        "avg_win_pct": round(float(win_returns.mean()) * 100, 3) if not win_returns.empty else 0.0,
        "avg_loss_pct": round(float(loss_returns.mean()) * 100, 3) if not loss_returns.empty else 0.0,
        "avg_mfe_pct": round(float(mfe.mean()), 3) if not mfe.dropna().empty else 0.0,
        "avg_mae_pct": round(float(mae.mean()), 3) if not mae.dropna().empty else 0.0,
        "avg_rr_at_entry": round(float(rr_at_entry.mean()), 3) if not rr_at_entry.dropna().empty else 0.0,
        "avg_net_edge_rate_pct": round(float(net_edge.mean()), 3) if not net_edge.dropna().empty else 0.0,
        "by_regime": _summarize_grouped_trades(df, "regime"),
        "by_direction": _summarize_grouped_trades(df, "direction"),
        "by_strategy_version": _summarize_grouped_trades(df, "strategy_version"),
        "by_mlx_factor": _summarize_mlx_factors(df),
        "trade_day_coverage": _summarize_trade_day_coverage(df, coverage_start_dt, end_dt),
    }


def _write_csv_atomic(frame, path_str):
    out_path = Path(path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f"{out_path.name}.tmp")
    frame.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)


TAIPEI_TZ = dt.timezone(dt.timedelta(hours=8))


def _taipei_trade_date(ts):
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(TAIPEI_TZ).date().isoformat()


def _daily_min_due_for_backtest(ts, traded_dates):
    if not eth._is_truthy(os.getenv("DAILY_MIN_TRADE_ENABLED", "1")):
        return False
    due_hour = min(23, max(0, eth._safe_int(os.getenv("DAILY_MIN_TRADE_HOUR", 22), 22)))
    due_minute = min(59, max(0, eth._safe_int(os.getenv("DAILY_MIN_TRADE_MINUTE", 30), 30)))
    local_ts = ts.astimezone(TAIPEI_TZ) if ts.tzinfo is not None else ts.replace(tzinfo=dt.timezone.utc).astimezone(TAIPEI_TZ)
    return _taipei_trade_date(ts) not in traded_dates and (local_ts.hour, local_ts.minute) >= (due_hour, due_minute)


def _apply_daily_min_backtest_plan(decision, frame_now, current_price):
    plan = eth._build_daily_min_trade_plan(
        current_price,
        decision.get("atr", 0.0),
        frame_now["15m"],
        frame_now["5m"],
        decision.get("htf", 0),
        decision.get("mid_trend", 0),
        macro_bias=decision.get("macro_bias", 0.0),
        news_bias=0.0,
        breakout=decision.get("breakout", 0),
        volume_spike=bool(decision.get("volume_spike", False)),
        regime=decision.get("regime", "range"),
        candlestick_turning=decision.get("candlestick_turning"),
        df_1d=frame_now.get("1d"),
    )
    final = str(plan.get("final") or "觀望")
    if final.startswith("觀望"):
        return final, float(decision.get("score", 0.5)), decision

    direction = str(plan.get("direction") or "")
    score = 0.62 if direction == "long" else 0.38
    daily_decision = dict(decision)
    daily_decision.update(
        {
            "final": final,
            "tp": float(plan.get("tp")),
            "sl": float(plan.get("sl")),
            "position_size": float(plan.get("position_size", 0.0)),
            "max_position_size": float(plan.get("max_position_size", plan.get("position_size", 0.0))),
            "max_hold_sec": float(plan.get("max_hold_sec", eth._trade_max_hold_sec("short"))),
            "score": score,
            "host_logic_applied": True,
            "primary_indicator": "mlx_daily_minimum",
            "daily_min_style_2024": dict(plan.get("style_2024_profile") or {}),
        }
    )
    host_logic = daily_decision.get("host_opening_logic")
    if not isinstance(host_logic, dict):
        host_logic = {}
    host_logic = dict(host_logic)
    host_logic.update(
        {
            "direction": direction,
            "mode": "daily_minimum",
            "confidence": max(float(host_logic.get("confidence", 0.0) or 0.0), 0.55),
            "reasons": list(host_logic.get("reasons") or []) + ["backtest_daily_min_trade"],
        }
    )
    daily_decision["host_opening_logic"] = host_logic
    return final, score, daily_decision


def _maybe_force_daily_min_for_backtest(ts, traded_dates, decision, frame_now, current_price):
    if not _daily_min_due_for_backtest(ts, traded_dates):
        return None
    final, score, daily_decision = _apply_daily_min_backtest_plan(decision, frame_now, current_price)
    if final.startswith("觀望"):
        return None
    return final, score, daily_decision


def _noop(*args, **kwargs):
    return None


@contextmanager
def _patched_eth_runtime():
    originals = {
        "send_telegram": eth.send_telegram,
        "send_private_telegram": eth.send_private_telegram,
        "sync_position_panel": eth.sync_position_panel,
        "_get_follow_mode_enabled": eth._get_follow_mode_enabled,
        "_is_real_copy_enabled": eth._is_real_copy_enabled,
    }
    active_trade_snapshot = dict(eth.active_trade)
    scaling_snapshot = dict(eth.SCALING_MARKET_STATE)
    panel_snapshot = dict(eth.POSITION_PANEL_STATE)

    eth.send_telegram = _noop
    eth.send_private_telegram = _noop
    eth.sync_position_panel = _noop
    eth._get_follow_mode_enabled = lambda: False
    eth._is_real_copy_enabled = lambda: False

    try:
        yield
    finally:
        eth.send_telegram = originals["send_telegram"]
        eth.send_private_telegram = originals["send_private_telegram"]
        eth.sync_position_panel = originals["sync_position_panel"]
        eth._get_follow_mode_enabled = originals["_get_follow_mode_enabled"]
        eth._is_real_copy_enabled = originals["_is_real_copy_enabled"]
        eth.active_trade.clear()
        eth.active_trade.update(active_trade_snapshot)
        eth.SCALING_MARKET_STATE.clear()
        eth.SCALING_MARKET_STATE.update(scaling_snapshot)
        eth.POSITION_PANEL_STATE.clear()
        eth.POSITION_PANEL_STATE.update(panel_snapshot)


def _build_open_trade(ts, direction, signal, entry, score, decision):
    size = float(decision["position_size"])
    if size <= 0:
        size = 0.2
    host_logic = decision.get("host_opening_logic") if isinstance(decision.get("host_opening_logic"), dict) else {}
    regime = str(decision.get("regime") or "range")
    counter_trend_probe = (
        (direction == "long" and regime in {"bear_trend", "bear_trend_strong"})
        or (direction == "short" and regime in {"bull_trend", "bull_trend_strong"})
    )
    turn = decision.get("candlestick_turning") if isinstance(decision.get("candlestick_turning"), dict) else {}
    opposing_turn_probe = (
        int(decision.get("candlestick_turn_count", 0)) >= 2
        and float(decision.get("candlestick_turn_confidence", 0.0)) >= 0.60
        and str(turn.get("direction") or "neutral") not in {"neutral", direction}
    )
    min_open_size = 0.001 if (
        str(host_logic.get("mode") or "") == "daily_minimum"
        or counter_trend_probe
        or opposing_turn_probe
    ) else 0.1
    size = float(min(1.0, max(size, min_open_size)))
    max_size, min_size = eth._derive_scaling_bounds(size)
    if "max_position_size" in decision:
        max_size = max(size, float(decision.get("max_position_size") or size))
    raw_features = decision.get("features") if isinstance(decision.get("features"), dict) else {}
    learn_features = eth._build_directional_learning_features(raw_features, direction)
    decision_for_mlx = dict(decision)
    decision_for_mlx["price"] = float(entry)
    mlx_market = eth._build_actual_trade_mlx_market(
        decision_for_mlx,
        direction,
        source="backtest",
    )
    mlx_factor_tags = build_trade_factor_tags(mlx_market, direction)

    return {
        "opened_at": ts.isoformat(),
        "open_ts": float(ts.timestamp()),
        "direction": direction,
        "signal": signal,
        "strategy_version": eth.STRATEGY_VERSION,
        "mlx_factor_tags": list(mlx_factor_tags),
        "entry": float(entry),
        "avg_entry": float(entry),
        "tp": float(decision["tp"]),
        "sl": float(decision["sl"]),
        "size": size,
        "score": float(score),
        "regime": str(decision.get("regime", "")),
        "ai_prob": float(decision.get("ai_prob", 0.5)),
        "ai_long_prob": float(decision.get("ai_long_prob", 0.5)),
        "ai_short_prob": float(decision.get("ai_short_prob", 0.5)),
        "macro_bias": float(decision.get("macro_bias", 0.0)),
        "sp_change": float(decision.get("sp_change", 0.0)),
        "nq_change": float(decision.get("nq_change", 0.0)),
        "btc_change": float(decision.get("btc_change", 0.0)),
        "dxy_change": float(decision.get("dxy_change", 0.0)),
        "news_bias": float(decision.get("news_bias", 0.0)),
        "event_risk": int(decision.get("event_risk", 0)),
        "news_categories": list(decision.get("news_categories") or []),
        "entry_threshold": float(decision.get("entry_threshold", 0.0)),
        "total_trade_cost_rate_est": float(decision.get("total_trade_cost_rate_est", 0.0)),
        "fee_round_trip_rate": float(decision.get("fee_round_trip_rate", 0.0)),
        "funding_cost_rate_est": float(decision.get("funding_cost_rate_est", 0.0)),
        "support_hits": int(decision.get("support_hits", 0)),
        "resistance_hits": int(decision.get("resistance_hits", 0)),
        "derivatives_pressure": float(decision.get("derivatives_pressure", 0.0)),
        "open_interest_change": float(decision.get("open_interest_change", 0.0)),
        "mark_premium_rate": float(decision.get("mark_premium_rate", 0.0)),
        "funding_rate_live": float(decision.get("funding_rate_live", 0.0)),
        "taker_buy_ratio": float(decision.get("taker_buy_ratio", 0.5)),
        "rr_at_entry": float(decision.get("rr_at_entry", 0.0)),
        "risk_rate": float(decision.get("risk_rate", 0.0)),
        "reward_rate": float(decision.get("reward_rate", 0.0)),
        "net_edge_rate_est": float(decision.get("net_edge_rate_est", 0.0)),
        "htf": int(decision.get("htf", 0)),
        "mid_trend": int(decision.get("mid_trend", 0)),
        "breakout": int(decision.get("breakout", 0)),
        "atr": float(decision.get("atr", 0.0)),
        "rsi_15m": float(decision.get("rsi_15m", 50.0)),
        "ema50_deviation_15m": float(decision.get("ema50_deviation_15m", 0.0)),
        "repeated_support_tests": int(decision.get("repeated_support_tests", 0)),
        "repeated_resistance_tests": int(decision.get("repeated_resistance_tests", 0)),
        "repeated_test_pressure": float(decision.get("repeated_test_pressure", 0.0)),
        "candlestick_turning": decision.get("candlestick_turning") if isinstance(decision.get("candlestick_turning"), dict) else {},
        "candlestick_turn_score": float(decision.get("candlestick_turn_score", 0.0)),
        "candlestick_turn_confidence": float(decision.get("candlestick_turn_confidence", 0.0)),
        "candlestick_turn_count": int(decision.get("candlestick_turn_count", 0)),
        "content_override": decision.get("content_override") if isinstance(decision.get("content_override"), dict) else {},
        "host_opening_logic": host_logic,
        "trade_source": str(host_logic.get("mode") or ""),
        "host_logic_applied": bool(decision.get("host_logic_applied", False)),
        "learned_entry_logic": decision.get("learned_entry_logic") if isinstance(decision.get("learned_entry_logic"), dict) else {},
        "primary_indicator": str(decision.get("primary_indicator") or ""),
        "market_profile_phase": str(decision.get("market_profile_phase") or ""),
        "market_profile_indicator_family": str(decision.get("market_profile_indicator_family") or ""),
        "max_favorable_move_pct": 0.0,
        "max_adverse_move_pct": 0.0,
        "max_size": max_size,
        "min_size": min_size,
        "max_hold_sec": float(decision.get("max_hold_sec") or 0.0),
        "add_count": 0,
        "reduce_count": 0,
        "last_adjust_ts": 0.0,
        "break_even_active": False,
        "break_even_target": 0.0,
        "break_even_ts": 0.0,
        "tp_sl_adjusted_4h": False,
        "realized_partial_return": 0.0,
        "break_even_activations": 0,
        "tp_shrink_count": 0,
        "raw_features": dict(raw_features),
        "learn_features": dict(learn_features),
        "events": [],
    }


def _push_trade_state_to_eth(open_trade):
    eth.active_trade["direction"] = open_trade["direction"]
    eth.active_trade["entry"] = float(open_trade["entry"])
    eth.active_trade["avg_entry"] = float(open_trade.get("avg_entry", open_trade["entry"]))
    eth.active_trade["tp"] = float(open_trade["tp"])
    eth.active_trade["sl"] = float(open_trade["sl"])
    eth.active_trade["open"] = True
    eth.active_trade["size"] = float(open_trade["size"])
    eth.active_trade["trade_source"] = str(open_trade.get("trade_source") or "")
    eth.active_trade["candlestick_turn_direction"] = str(
        (open_trade.get("candlestick_turning") or {}).get("direction") or "neutral"
    )
    eth.active_trade["candlestick_turn_count"] = int(open_trade.get("candlestick_turn_count", 0))
    eth.active_trade["candlestick_turn_confidence"] = float(open_trade.get("candlestick_turn_confidence", 0.0))
    eth.active_trade["max_size"] = float(open_trade.get("max_size", 1.0))
    eth.active_trade["min_size"] = float(open_trade.get("min_size", 0.1))
    eth.active_trade["add_count"] = int(open_trade.get("add_count", 0))
    eth.active_trade["reduce_count"] = int(open_trade.get("reduce_count", 0))
    eth.active_trade["last_adjust_ts"] = float(open_trade.get("last_adjust_ts", 0.0))
    eth.active_trade["open_time"] = float(open_trade["open_ts"])
    eth.active_trade["tp_sl_adjusted_4h"] = bool(open_trade.get("tp_sl_adjusted_4h", False))
    eth.active_trade["break_even_active"] = bool(open_trade.get("break_even_active", False))
    eth.active_trade["break_even_target"] = float(open_trade.get("break_even_target", 0.0))
    eth.active_trade["break_even_ts"] = float(open_trade.get("break_even_ts", 0.0))


def _pull_trade_state_from_eth(open_trade):
    open_trade["entry"] = float(eth.active_trade.get("entry") or open_trade["entry"])
    open_trade["avg_entry"] = float(eth.active_trade.get("avg_entry") or open_trade["avg_entry"])
    open_trade["tp"] = float(eth.active_trade.get("tp") or open_trade["tp"])
    open_trade["sl"] = float(eth.active_trade.get("sl") or open_trade["sl"])
    open_trade["size"] = float(eth.active_trade.get("size") or open_trade["size"])
    open_trade["max_size"] = float(eth.active_trade.get("max_size") or open_trade["max_size"])
    open_trade["min_size"] = float(eth.active_trade.get("min_size") or open_trade["min_size"])
    open_trade["add_count"] = int(eth.active_trade.get("add_count") or open_trade["add_count"])
    open_trade["reduce_count"] = int(eth.active_trade.get("reduce_count") or open_trade["reduce_count"])
    open_trade["last_adjust_ts"] = float(eth.active_trade.get("last_adjust_ts") or open_trade["last_adjust_ts"])
    open_trade["tp_sl_adjusted_4h"] = bool(eth.active_trade.get("tp_sl_adjusted_4h", open_trade["tp_sl_adjusted_4h"]))
    open_trade["break_even_active"] = bool(eth.active_trade.get("break_even_active", open_trade["break_even_active"]))
    open_trade["break_even_target"] = float(eth.active_trade.get("break_even_target") or open_trade["break_even_target"])
    open_trade["break_even_ts"] = float(eth.active_trade.get("break_even_ts") or open_trade["break_even_ts"])
    return open_trade


def _estimate_trade_leg(direction, entry, exit_price, size, hold_hours):
    if direction == "long":
        gross_move = (exit_price - entry) / max(entry, 1e-9)
    else:
        gross_move = (entry - exit_price) / max(entry, 1e-9)
    net_move = gross_move - float(eth._estimate_trade_cost_rate_est(hold_hours=hold_hours))
    return gross_move, net_move, net_move * size


def _append_trade_event(open_trade, ts, event_type, **payload):
    event = {"ts": ts.isoformat(), "type": event_type}
    event.update(payload)
    open_trade["events"].append(event)


def _update_trade_excursion(open_trade, bar_high, bar_low):
    entry = float(open_trade.get("avg_entry") or open_trade.get("entry") or 0.0)
    if entry <= 0:
        return open_trade

    high = float(bar_high)
    low = float(bar_low)
    if open_trade.get("direction") == "long":
        favorable = max(0.0, (high - entry) / entry)
        adverse = max(0.0, (entry - low) / entry)
    else:
        favorable = max(0.0, (entry - low) / entry)
        adverse = max(0.0, (high - entry) / entry)

    open_trade["max_favorable_move_pct"] = max(
        float(open_trade.get("max_favorable_move_pct", 0.0)),
        favorable * 100.0,
    )
    open_trade["max_adverse_move_pct"] = max(
        float(open_trade.get("max_adverse_move_pct", 0.0)),
        adverse * 100.0,
    )
    return open_trade


def _apply_trade_management(open_trade, current_price, atr, ts):
    ts_sec = float(ts.timestamp())
    hold_hours = max(0.0, (ts_sec - float(open_trade["open_ts"])) / 3600.0)
    _push_trade_state_to_eth(open_trade)

    be_active_before = bool(eth.active_trade.get("break_even_active"))
    sl_before = float(eth.active_trade.get("sl") or open_trade["sl"])
    be_triggered = eth.maybe_activate_auto_break_even(current_price, atr=atr, now_ts=ts_sec)
    be_active_after = bool(eth.active_trade.get("break_even_active"))
    sl_after = float(eth.active_trade.get("sl") or sl_before)
    if not be_active_before and be_active_after:
        open_trade["break_even_activations"] += 1
        _append_trade_event(
            open_trade,
            ts,
            "break_even",
            old_sl=round(sl_before, 4),
            new_sl=round(sl_after, 4),
        )

    if not be_triggered:
        size_before = float(eth.active_trade.get("size") or open_trade["size"])
        entry_before = float(eth.active_trade.get("avg_entry") or open_trade["avg_entry"])
        add_count_before = int(eth.active_trade.get("add_count") or open_trade["add_count"])
        reduce_count_before = int(eth.active_trade.get("reduce_count") or open_trade["reduce_count"])

        eth.manage_position_scaling(current_price, atr=atr, now_ts=ts_sec)

        size_after = float(eth.active_trade.get("size") or size_before)
        entry_after = float(eth.active_trade.get("avg_entry") or entry_before)
        add_count_after = int(eth.active_trade.get("add_count") or add_count_before)
        reduce_count_after = int(eth.active_trade.get("reduce_count") or reduce_count_before)

        if size_after > size_before + 1e-9:
            _append_trade_event(
                open_trade,
                ts,
                "scale_add",
                delta_size=round(size_after - size_before, 4),
                price=round(current_price, 4),
                entry_before=round(entry_before, 4),
                entry_after=round(entry_after, 4),
                add_count=add_count_after,
            )
        elif size_after < size_before - 1e-9:
            reduced_size = size_before - size_after
            gross_move, net_move, partial_return = _estimate_trade_leg(
                open_trade["direction"],
                entry_before,
                current_price,
                reduced_size,
                hold_hours,
            )
            open_trade["realized_partial_return"] += partial_return
            _append_trade_event(
                open_trade,
                ts,
                "scale_reduce",
                delta_size=round(reduced_size, 4),
                price=round(current_price, 4),
                gross_move_pct=round(gross_move * 100, 3),
                net_move_pct=round(net_move * 100, 3),
                realized_return=round(partial_return, 6),
                reduce_count=reduce_count_after,
            )

    tp_before = float(eth.active_trade.get("tp") or open_trade["tp"])
    adjusted_before = bool(eth.active_trade.get("tp_sl_adjusted_4h", open_trade["tp_sl_adjusted_4h"]))
    tp_changed = eth.maybe_shrink_tp_after_hold(current_price=current_price, now_ts=ts_sec)
    tp_after = float(eth.active_trade.get("tp") or tp_before)
    adjusted_after = bool(eth.active_trade.get("tp_sl_adjusted_4h", adjusted_before))
    if tp_changed and (not adjusted_before) and adjusted_after:
        open_trade["tp_shrink_count"] += 1
        _append_trade_event(
            open_trade,
            ts,
            "tp_shrink_4h",
            old_tp=round(tp_before, 4),
            new_tp=round(tp_after, 4),
        )

    return _pull_trade_state_from_eth(open_trade)


def _close_trade(open_trade, exit_price, exit_reason, ts, equity):
    hold_hours = max(0.0, (float(ts.timestamp()) - float(open_trade["open_ts"])) / 3600.0)
    remaining_size = float(open_trade["size"])
    gross_move, net_move, final_leg_return = _estimate_trade_leg(
        open_trade["direction"],
        float(open_trade["avg_entry"]),
        float(exit_price),
        remaining_size,
        hold_hours,
    )
    trade_return = float(open_trade["realized_partial_return"]) + final_leg_return
    equity *= (1.0 + trade_return)
    learning_sample = None
    if exit_reason in {"TP", "SL"} and isinstance(open_trade.get("learn_features"), dict):
        learning_sample = {
            **eth._normalize_feature_payload(open_trade["learn_features"]),
            "label": 1 if exit_reason == "TP" else 0,
        }
    sl_review = {}
    if exit_reason == "SL":
        host_opening_logic = open_trade.get("host_opening_logic") if isinstance(open_trade.get("host_opening_logic"), dict) else {}
        context = {
            "strategy_version": str(open_trade.get("strategy_version", eth.STRATEGY_VERSION)),
            "htf": open_trade.get("htf"),
            "mid_trend": open_trade.get("mid_trend"),
            "breakout": open_trade.get("breakout"),
            "score": open_trade.get("score"),
            "ai_prob": open_trade.get("ai_prob"),
            "ai_long_prob": open_trade.get("ai_long_prob"),
            "ai_short_prob": open_trade.get("ai_short_prob"),
            "macro_bias": open_trade.get("macro_bias"),
            "support_hits": open_trade.get("support_hits"),
            "resistance_hits": open_trade.get("resistance_hits"),
            "derivatives_pressure": open_trade.get("derivatives_pressure"),
            "taker_buy_ratio": open_trade.get("taker_buy_ratio"),
            "open_interest_change": open_trade.get("open_interest_change"),
            "net_edge_rate_est": open_trade.get("net_edge_rate_est"),
            "risk_rate": open_trade.get("risk_rate"),
            "reward_rate": open_trade.get("reward_rate"),
            "rsi_15m": open_trade.get("rsi_15m"),
            "ema50_deviation_15m": open_trade.get("ema50_deviation_15m"),
            "repeated_support_tests": open_trade.get("repeated_support_tests"),
            "repeated_resistance_tests": open_trade.get("repeated_resistance_tests"),
            "repeated_test_pressure": open_trade.get("repeated_test_pressure"),
            "content_override": open_trade.get("content_override"),
            "host_opening_logic": open_trade.get("host_opening_logic"),
            "host_logic_applied": open_trade.get("host_logic_applied"),
            "host_logic_direction": host_opening_logic.get("direction"),
            "host_logic_mode": host_opening_logic.get("mode"),
            "host_logic_confidence": host_opening_logic.get("confidence"),
            "market_profile_phase": open_trade.get("market_profile_phase"),
            "market_profile_indicator_family": open_trade.get("market_profile_indicator_family"),
            "learned_entry_logic": open_trade.get("learned_entry_logic"),
            "primary_indicator": open_trade.get("primary_indicator"),
        }
        sl_review = eth._review_stop_loss_event(
            open_trade["direction"],
            open_trade.get("avg_entry", open_trade.get("entry")),
            open_trade.get("tp"),
            open_trade.get("sl"),
            exit_price,
            exit_price,
            exit_price,
            open_trade.get("atr"),
            context,
        )

    return equity, {
        "opened_at": open_trade["opened_at"],
        "closed_at": ts.isoformat(),
        "direction": open_trade["direction"],
        "signal": open_trade["signal"],
        "strategy_version": str(open_trade.get("strategy_version", eth.STRATEGY_VERSION)),
        "mlx_factor_tags": json.dumps(open_trade.get("mlx_factor_tags") or [], ensure_ascii=False),
        "entry": round(float(open_trade["entry"]), 4),
        "avg_entry": round(float(open_trade["avg_entry"]), 4),
        "exit": round(float(exit_price), 4),
        "tp": round(float(open_trade["tp"]), 4),
        "sl": round(float(open_trade["sl"]), 4),
        "size": round(remaining_size, 4),
        "score": round(float(open_trade["score"]), 4),
        "regime": str(open_trade.get("regime", "")),
        "ai_prob": round(float(open_trade.get("ai_prob", 0.5)), 4),
        "ai_long_prob": round(float(open_trade.get("ai_long_prob", 0.5)), 4),
        "ai_short_prob": round(float(open_trade.get("ai_short_prob", 0.5)), 4),
        "macro_bias": round(float(open_trade.get("macro_bias", 0.0)), 4),
        "sp_change_pct": round(float(open_trade.get("sp_change", 0.0)) * 100, 3),
        "nq_change_pct": round(float(open_trade.get("nq_change", 0.0)) * 100, 3),
        "btc_change_pct": round(float(open_trade.get("btc_change", 0.0)) * 100, 3),
        "dxy_change_pct": round(float(open_trade.get("dxy_change", 0.0)) * 100, 3),
        "news_bias": round(float(open_trade.get("news_bias", 0.0)), 4),
        "event_risk": int(open_trade.get("event_risk", 0)),
        "news_categories": json.dumps(open_trade.get("news_categories") or [], ensure_ascii=False),
        "entry_threshold": round(float(open_trade.get("entry_threshold", 0.0)), 4),
        "rr_at_entry": round(float(open_trade.get("rr_at_entry", 0.0)), 3),
        "risk_rate_pct": round(float(open_trade.get("risk_rate", 0.0)) * 100, 3),
        "reward_rate_pct": round(float(open_trade.get("reward_rate", 0.0)) * 100, 3),
        "net_edge_rate_est_pct": round(float(open_trade.get("net_edge_rate_est", 0.0)) * 100, 3),
        "total_trade_cost_rate_est_pct": round(float(open_trade.get("total_trade_cost_rate_est", 0.0)) * 100, 3),
        "fee_round_trip_rate_pct": round(float(open_trade.get("fee_round_trip_rate", 0.0)) * 100, 3),
        "funding_cost_rate_est_pct": round(float(open_trade.get("funding_cost_rate_est", 0.0)) * 100, 3),
        "support_hits": int(open_trade.get("support_hits", 0)),
        "resistance_hits": int(open_trade.get("resistance_hits", 0)),
        "candlestick_turn_direction": str((open_trade.get("candlestick_turning") or {}).get("direction") or ""),
        "candlestick_turn_count": int(open_trade.get("candlestick_turn_count", 0)),
        "candlestick_turn_confidence": round(float(open_trade.get("candlestick_turn_confidence", 0.0)), 4),
        "candlestick_turn_score": round(float(open_trade.get("candlestick_turn_score", 0.0)), 4),
        "candlestick_turn_reasons": json.dumps((open_trade.get("candlestick_turning") or {}).get("reasons") or [], ensure_ascii=False),
        "host_logic_applied": bool(open_trade.get("host_logic_applied", False)),
        "host_logic_direction": str((open_trade.get("host_opening_logic") or {}).get("direction") or ""),
        "host_logic_mode": str((open_trade.get("host_opening_logic") or {}).get("mode") or ""),
        "host_logic_confidence": round(float((open_trade.get("host_opening_logic") or {}).get("confidence", 0.0)), 4),
        "host_logic_reasons": json.dumps((open_trade.get("host_opening_logic") or {}).get("reasons") or [], ensure_ascii=False),
        "market_profile_phase": str(open_trade.get("market_profile_phase") or ""),
        "market_profile_indicator_family": str(open_trade.get("market_profile_indicator_family") or ""),
        "derivatives_pressure": round(float(open_trade.get("derivatives_pressure", 0.0)), 4),
        "open_interest_change_pct": round(float(open_trade.get("open_interest_change", 0.0)) * 100, 3),
        "mark_premium_rate_pct": round(float(open_trade.get("mark_premium_rate", 0.0)) * 100, 4),
        "funding_rate_live_pct": round(float(open_trade.get("funding_rate_live", 0.0)) * 100, 4),
        "taker_buy_ratio": round(float(open_trade.get("taker_buy_ratio", 0.5)), 4),
        "max_favorable_move_pct": round(float(open_trade.get("max_favorable_move_pct", 0.0)), 3),
        "max_adverse_move_pct": round(float(open_trade.get("max_adverse_move_pct", 0.0)), 3),
        "gross_move_pct": round(gross_move * 100, 3),
        "net_move_pct": round(net_move * 100, 3),
        "partial_realized_return": round(float(open_trade["realized_partial_return"]), 6),
        "trade_return": round(trade_return, 6),
        "equity": round(equity, 6),
        "exit_reason": exit_reason,
        "break_even_activations": int(open_trade["break_even_activations"]),
        "tp_shrink_count": int(open_trade["tp_shrink_count"]),
        "scale_add_count": int(open_trade["add_count"]),
        "scale_reduce_count": int(open_trade["reduce_count"]),
        "sl_review_issue_codes": json.dumps(sl_review.get("issue_codes", []), ensure_ascii=False),
        "sl_review_actions": json.dumps(sl_review.get("optimization_actions", []), ensure_ascii=False),
        "sl_review_json": json.dumps(sl_review, ensure_ascii=False, default=str) if sl_review else "",
        "management_events": json.dumps(open_trade["events"], ensure_ascii=False),
    }, learning_sample


def run_backtest(symbol, start_dt, end_dt, warmup_bars, data_source="auto"):
    base_5m = fetch_futures_klines(
        symbol=symbol,
        interval="5m",
        start_ms=int(start_dt.timestamp() * 1000),
        end_ms=int(end_dt.timestamp() * 1000),
        data_source=data_source,
    )
    if base_5m.empty:
        raise SystemExit("No klines returned from TradingView market data")

    macro_context = HistoricalMacroContext(start_dt, end_dt)
    print(f"🧭 歷史宏觀資料: {json.dumps(macro_context.summary(), ensure_ascii=False)}")

    frame_map = build_frame_map(base_5m)
    frame_indexes = {name: frame.index for name, frame in frame_map.items()}
    frame_tail_limits = {
        "5m": 360,
        "15m": 320,
        "30m": 280,
        "1h": 260,
        "4h": 260,
        "12h": 220,
        "1d": 260,
        "1w": 260,
        "1M": 120,
    }
    fast_tail_frames = eth._is_truthy(os.getenv("BACKTEST_FAST_TAIL_FRAMES", "0"))
    decision_every_bars = max(1, eth._safe_int(os.getenv("BACKTEST_DECISION_EVERY_BARS", 3), 3))
    sr_cfg = [
        ("日線", "1d", 180, 1.1),
        ("12h", "12h", 160, 1.0),
        ("4h", "4h", 140, 0.9),
        ("1h", "1h", 120, 0.7),
        ("30m", "30m", 100, 0.5),
    ]

    with _patched_eth_runtime():
        eth.load_model()
        model_loaded = eth.model is not None or eth.online_initialized

        last_trade_time = 0.0
        last_trade_signal = None
        last_entry_price = None
        last_signal_value = None
        losing_streak = 0
        equity = 1.0
        trades = []
        learning_samples = []
        open_trade = None
        traded_dates = set()
        first_decision_ts = None

        trade_cooldown_sec = 300
        min_price_change = 0.002
        min_signal_diff = 0.05

        base_rows = frame_map["5m"]
        warmup_bars = max(200, int(warmup_bars))

        for idx in range(warmup_bars, len(base_rows)):
            ts = base_rows.index[idx]
            row = base_rows.iloc[idx]
            current_price = float(row["close"])
            bar_high = float(row["high"])
            bar_low = float(row["low"])
            ts_sec = float(ts.timestamp())

            if open_trade is not None:
                open_trade = _update_trade_excursion(open_trade, bar_high, bar_low)
                direction = open_trade["direction"]
                tp = float(open_trade["tp"])
                sl = float(open_trade["sl"])
                exit_reason = None
                exit_price = None

                if direction == "long":
                    sl_hit = bar_low <= sl
                    tp_hit = bar_high >= tp
                    if sl_hit:
                        exit_reason = "SL"
                        exit_price = sl
                    elif tp_hit:
                        exit_reason = "TP"
                        exit_price = tp
                else:
                    sl_hit = bar_high >= sl
                    tp_hit = bar_low <= tp
                    if sl_hit:
                        exit_reason = "SL"
                        exit_price = sl
                    elif tp_hit:
                        exit_reason = "TP"
                        exit_price = tp

                if exit_reason:
                    equity, trade_record, learning_sample = _close_trade(open_trade, exit_price, exit_reason, ts, equity)
                    trades.append(trade_record)
                    if learning_sample is not None:
                        learning_samples.append(learning_sample)
                    losing_streak = 0 if trade_record["trade_return"] > 0 else (losing_streak + 1)
                    open_trade = None

            if (
                open_trade is not None
                and eth._is_truthy(os.getenv("BACKTEST_DAILY_MIN_ROLLOVER_ENABLED", "1"))
                and _daily_min_due_for_backtest(ts, traded_dates)
            ):
                equity, trade_record, learning_sample = _close_trade(open_trade, current_price, "DAILY_MIN_ROLLOVER", ts, equity)
                trades.append(trade_record)
                if learning_sample is not None:
                    learning_samples.append(learning_sample)
                losing_streak = 0 if trade_record["trade_return"] > 0 else (losing_streak + 1)
                open_trade = None

            if open_trade is not None:
                atr_5m = float(max(bar_high - bar_low, 0.0))
                max_hold_sec = float(open_trade.get("max_hold_sec") or 0.0)
                held_sec = float(ts.timestamp()) - float(open_trade["open_ts"])
                if max_hold_sec > 0 and held_sec >= max_hold_sec:
                    equity, trade_record, learning_sample = _close_trade(open_trade, current_price, "MAX_HOLD", ts, equity)
                    trades.append(trade_record)
                    if learning_sample is not None:
                        learning_samples.append(learning_sample)
                    losing_streak = 0 if trade_record["trade_return"] > 0 else (losing_streak + 1)
                    open_trade = None
                    continue
                open_trade = _apply_trade_management(open_trade, current_price, atr_5m, ts)
                continue

            daily_min_due_now = _daily_min_due_for_backtest(ts, traded_dates)
            if idx % decision_every_bars != 0 and not daily_min_due_now:
                continue

            if fast_tail_frames:
                frame_now = {
                    name: _slice_frame_until(
                        frame,
                        frame_indexes[name],
                        ts,
                        frame_tail_limits.get(name, 260),
                    )
                    for name, frame in frame_map.items()
                }
            else:
                frame_now = {name: frame.loc[:ts] for name, frame in frame_map.items()}
            if any(len(frame_now[key]) < 30 for key in ("15m", "30m", "1h", "4h")):
                continue
            if first_decision_ts is None:
                first_decision_ts = ts

            sr_frames = {key: frame_now.get(key) for key in ("1d", "12h", "4h", "1h", "30m")}
            sr_analysis = eth.analyze_multi_tf_sr_frames(current_price, sr_frames, tf_cfg=sr_cfg)
            macro_snapshot = macro_context.snapshot(ts)

            decision = eth.build_trade_signal_snapshot(
                df_4h=frame_now["4h"],
                df_1h=frame_now["1h"],
                df_30m=frame_now["30m"],
                df_15m=frame_now["15m"],
                df_5m=frame_now["5m"],
                price=current_price,
                sr_analysis=sr_analysis,
                sp_change=macro_snapshot["sp_change"],
                nq_change=macro_snapshot["nq_change"],
                btc_change=macro_snapshot["btc_change"],
                dxy_change=macro_snapshot["dxy_change"],
                news_bias=macro_snapshot["news_bias"],
                event_risk=macro_snapshot["event_risk"],
                last_signal=last_signal_value,
                losing_streak=losing_streak,
                df_1d=frame_now["1d"],
                df_1w=frame_now["1w"],
                df_1mth=frame_now.get("1M"),
            )
            decision["sp_change"] = macro_snapshot["sp_change"]
            decision["nq_change"] = macro_snapshot["nq_change"]
            decision["btc_change"] = macro_snapshot["btc_change"]
            decision["dxy_change"] = macro_snapshot["dxy_change"]
            decision["news_bias"] = macro_snapshot["news_bias"]
            decision["event_risk"] = macro_snapshot["event_risk"]
            decision["news_categories"] = list(macro_snapshot.get("news_categories") or [])

            eth._update_scaling_market_state(
                price=current_price,
                atr=float(decision["atr"]),
                htf=int(decision["htf"]),
                mid_trend=int(decision["mid_trend"]),
                regime=str(decision["regime"]),
                breakout=int(decision["breakout"]),
                sr_analysis=sr_analysis,
                volume_ratio=decision.get("volume_ratio", 0.0),
                volume_spike=decision.get("volume_spike", False),
                buy_pressure=decision.get("buy_pressure", False),
                sell_pressure=decision.get("sell_pressure", False),
            )

            score = float(decision["score"])
            final = str(decision["final"])
            current_direction = eth.get_signal_direction(final)
            last_direction_simple = eth.get_signal_direction(last_trade_signal) if last_trade_signal else None
            daily_min_forced = False
            market_profile = decision.get("market_profile") if isinstance(decision.get("market_profile"), dict) else {}
            market_phase = str(market_profile.get("phase") or "range_base")
            daily_anchor_guard = bool(
                eth._is_truthy(os.getenv("BACKTEST_DAILY_MIN_ANCHOR_GUARD_ENABLED", "1"))
                and _taipei_trade_date(ts) not in traded_dates
                and not daily_min_due_now
                and eth._daily_anchor_guard_should_wait(final, score, decision)
            )

            if current_direction == last_direction_simple:
                if last_entry_price is not None:
                    price_change = abs(current_price - last_entry_price) / max(current_price, 1e-9)
                    if price_change < min_price_change:
                        final = "觀望（防洗單-價格過近）"
                if last_signal_value is not None and abs(score - last_signal_value) < min_signal_diff:
                    final = "觀望（防洗單-信號重複）"

            if not final.startswith("觀望"):
                if ts_sec - last_trade_time < trade_cooldown_sec:
                    final = "觀望（冷卻中）"
                elif last_entry_price is not None:
                    price_change = abs(current_price - last_entry_price) / max(current_price, 1e-9)
                    if price_change < min_price_change:
                        final = "觀望（價格未達門檻）"

            last_signal_value = score

            if (
                daily_min_due_now
                and not daily_min_forced
                and market_phase != "bull"
                and not (
                    eth._is_truthy(os.getenv("BACKTEST_DAILY_MIN_DUE_ALLOW_QUALITY_SIGNAL", "1"))
                    and not eth._daily_anchor_guard_should_wait(final, score, decision)
                )
            ):
                forced = _maybe_force_daily_min_for_backtest(ts, traded_dates, decision, frame_now, current_price)
                if forced is not None:
                    final, score, decision = forced
                    daily_min_forced = True

            if daily_anchor_guard and not final.startswith("觀望"):
                final = "觀望（每日單錨定-等待保底）"

            if (not daily_min_forced) and market_phase == "bear" and "做多" in final:
                final = "觀望（熊市禁止非每日多單）"

            if final.startswith("觀望"):
                forced = _maybe_force_daily_min_for_backtest(ts, traded_dates, decision, frame_now, current_price)
                if forced is not None:
                    final, score, decision = forced
                    daily_min_forced = True
                if final.startswith("觀望"):
                    continue
            if (not daily_min_forced) and (
                decision["fake_breakout"] and abs(score - 0.5) < 0.22
                or ("做多" in final and decision["resistance_hits"] >= 2 and score < 0.72)
                or ("做空" in final and decision["support_hits"] >= 2 and score > 0.28)
            ):
                forced = _maybe_force_daily_min_for_backtest(ts, traded_dates, decision, frame_now, current_price)
                if forced is None:
                    continue
                final, score, decision = forced
                daily_min_forced = True

            entry = current_price
            direction = "long" if "做多" in final else "short"
            open_trade = _build_open_trade(ts, direction, final, entry, score, decision)
            last_trade_time = ts_sec
            last_trade_signal = final
            last_entry_price = entry
            traded_dates.add(_taipei_trade_date(ts))

        if open_trade is not None:
            last_bar = base_rows.iloc[-1]
            exit_price = float(last_bar["close"])
            equity, trade_record, _ = _close_trade(open_trade, exit_price, "EOD", base_rows.index[-1], equity)
            trades.append(trade_record)

    data_source = str(base_5m.attrs.get("kline_source") or "futures")
    coverage_start_dt = first_decision_ts or (base_rows.index[warmup_bars] if len(base_rows) > warmup_bars else start_dt)
    summary = summarize_trades(
        trades,
        start_dt,
        end_dt,
        symbol,
        model_loaded,
        data_source=data_source,
        coverage_start_dt=coverage_start_dt,
    )
    summary["historical_macro_context"] = macro_context.summary()
    return base_5m, trades, summary, learning_samples


def main():
    args = _parse_args()
    start_dt, end_dt = _resolve_timerange(args)
    base_5m, trades, summary, learning_samples = run_backtest(
        args.symbol,
        start_dt,
        end_dt,
        args.warmup_bars,
        data_source=args.data_source,
    )

    print("Backtest Summary")
    print(f"Symbol: {summary['symbol']}")
    print(f"Window: {summary['start']} -> {summary['end']}")
    print(f"Data source: {summary.get('data_source', 'futures')}")
    print(f"Strategy version: {summary.get('strategy_version')}")
    print(f"5m bars: {len(base_5m)}")
    print(f"Model loaded: {summary['model_loaded']}")
    print(f"Trades: {summary['trades']}")
    print(f"Win rate: {summary['win_rate']}%")
    print(f"Total return: {summary['total_return_pct']}%")
    print(f"Avg trade return: {summary['avg_trade_return_pct']}%")
    print(f"Max drawdown: {summary['max_drawdown_pct']}%")
    print(f"Profit factor: {summary['profit_factor']}")
    print(f"Avg MFE/MAE: {summary['avg_mfe_pct']}%/{summary['avg_mae_pct']}%")
    print(f"Avg RR/AI edge: {summary['avg_rr_at_entry']}/{summary['avg_net_edge_rate_pct']}%")
    print(f"Long/Short: {summary['long_trades']}/{summary['short_trades']}")
    print(f"Exit reasons: {json.dumps(summary['exit_reason_counts'], ensure_ascii=False)}")
    coverage = summary.get("trade_day_coverage") or {}
    if coverage:
        print(
            "Trade-day coverage: "
            f"{coverage.get('trade_days', 0)}/{coverage.get('calendar_days', 0)} days, "
            f"missing={coverage.get('missing_trade_days', 0)}, "
            f"daily_min={coverage.get('daily_min_trades', 0)}"
        )
    top_factors = sorted(
        (summary.get("by_mlx_factor") or {}).items(),
        key=lambda item: (item[1].get("trades", 0), item[1].get("win_rate", 0.0)),
        reverse=True,
    )[:5]
    if top_factors:
        print(
            "Top MLX factors: "
            + " | ".join(
                f"{name} {payload.get('win_rate')}%/{payload.get('trades')}筆"
                for name, payload in top_factors
            )
        )

    if args.trades_out:
        out_path = Path(args.trades_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(trades).to_csv(out_path, index=False)
        print(f"Trade log written to {out_path}")

    if args.summary_out:
        out_path = Path(args.summary_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Summary written to {out_path}")

    if args.learn_out:
        learn_df = pd.DataFrame(learning_samples, columns=eth.MODEL_FEATURE_COLUMNS + ["label"])
        _write_csv_atomic(learn_df, args.learn_out)
        print(f"Learning samples written to {args.learn_out} ({len(learn_df)})")


if __name__ == "__main__":
    main()
