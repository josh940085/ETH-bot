"""Resumable, multi-exchange historical kline builder.

market_history.py's Binance-archive fetch is the primary source for backtest
history - it's a bulk zip download, not rate-limited per-request, and has the
deepest, most complete record. But some days have no archive (too recent, or
a gap in Binance's own publication), and those days were previously just
dropped (`frame.attrs["kline_missing_months"]`). This module fills exactly
those gaps from a second exchange via market_data_sources, so backtest data
stops silently differing from what the live multi-exchange waterfall sees.

Kraken is intentionally excluded from gap-filling even though it's in the
live waterfall: its public OHLC endpoint only ever returns the most recent
~720 candles regardless of the `since` parameter (verified empirically), so
it cannot page arbitrarily far into the past. Coinbase's candles endpoint
supports arbitrary start/end windows with real multi-year history, so it is
the gap-fill source; Twelve Data (paid quota, same throttle discipline as
the live path) is the second fallback if Coinbase also fails for a day.

Every successfully gap-filled day is cached to its own CSV file, and a
progress JSON records which days were attempted and with what outcome. On
resume, already-attempted days are loaded from that per-day cache instead of
being re-fetched - both so an interrupted run doesn't re-fetch everything,
and so a fully-resumed run can still reconstruct the complete combined frame
instead of returning archive-only data with stale gap bookkeeping.
"""

import datetime
import json
import os
import time

import pandas as pd

import market_data_sources
import market_history
from runtime_paths import data_path, ensure_parent_dir

GAP_FILL_INTERVAL = "5m"  # matches market_history.fetch_klines_from_binance_history's base interval


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def progress_path(symbol, interval):
    return data_path("market_history_progress", f"{str(symbol).upper()}_{interval}_gapfill_progress.json")


def _day_cache_dir(symbol, interval, resume_path):
    return resume_path.parent / f"{str(symbol).upper()}_{interval}_days"


def _day_cache_path(symbol, interval, day_str, resume_path):
    return _day_cache_dir(symbol, interval, resume_path) / f"{day_str}.csv"


def _load_progress(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _save_progress(path, payload):
    ensure_parent_dir(path)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _save_day_frame(frame, cache_path):
    ensure_parent_dir(cache_path)
    tmp_path = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp")
    frame.to_csv(tmp_path)
    tmp_path.replace(cache_path)


def _load_day_frame(cache_path):
    if not cache_path.exists():
        return None
    try:
        frame = pd.read_csv(cache_path, index_col=0, parse_dates=[0])
        frame.index = pd.to_datetime(frame.index, utc=True)
        frame.index.name = "close_time"
        return frame if not frame.empty else None
    except Exception:
        return None


def _rows_to_frame(rows):
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
        raw[column] = pd.to_numeric(raw[column], errors="coerce").astype("float64")
    return raw.set_index("close_time")[["open", "high", "low", "close", "volume"]]


def _day_bounds_ms(day_str):
    day_dt = datetime.datetime.strptime(day_str, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    start_ms = int(day_dt.timestamp() * 1000)
    end_ms = start_ms + 24 * 60 * 60 * 1000 - 1
    return start_ms, end_ms


def fetch_gap_fill_day(symbol, day_str, *, timeout=10):
    """Fetch one missing day of GAP_FILL_INTERVAL candles for `symbol`.

    Tries Coinbase first (best historical depth of the non-Binance sources),
    then Twelve Data. Returns (frame, source_name) or (None, error_detail) if
    both sources fail for this day - callers should treat that as "still
    missing" and retry on a later run rather than as fatal.
    """
    start_ms, end_ms = _day_bounds_ms(day_str)
    errors = []
    try:
        rows = market_data_sources.fetch_coinbase_klines(
            symbol, GAP_FILL_INTERVAL, start_time_ms=start_ms, end_time_ms=end_ms, limit=300, timeout=timeout
        )
        return _rows_to_frame(rows), "coinbase"
    except Exception as exc:
        errors.append(f"coinbase: {exc}")
    try:
        rows = market_data_sources.fetch_twelve_data_klines(
            symbol, GAP_FILL_INTERVAL, start_time_ms=start_ms, end_time_ms=end_ms, limit=300, timeout=timeout
        )
        return _rows_to_frame(rows), "twelve_data"
    except Exception as exc:
        errors.append(f"twelve_data: {exc}")
    return None, "; ".join(errors)


def build_history_with_gap_fill(symbol, start_ms, end_ms, *, request_gap_sec=None, resume_path=None):
    """Binance archive first; fill any missing days via Coinbase/Twelve Data.

    Resumable: each successfully gap-filled day is cached to its own CSV file
    under the progress directory. On resume, a day already recorded in the
    progress JSON is loaded from that cache (or, for a day previously marked
    "unavailable", retried) instead of always being re-fetched from scratch.
    """
    frame = market_history.fetch_klines_from_binance_history(symbol, GAP_FILL_INTERVAL, start_ms, end_ms)
    missing_days = list(frame.attrs.get("kline_missing_months") or [])
    if not missing_days:
        return frame

    resume_path = resume_path or progress_path(symbol, GAP_FILL_INTERVAL)
    progress = _load_progress(resume_path)
    progress.setdefault("day_sources", {})
    day_sources = progress["day_sources"]
    gap_frames = []
    filled_sources = set()
    still_missing = []
    request_gap_sec = max(0.5, _safe_float(request_gap_sec, 2.0))

    for day_str in missing_days:
        cache_path = _day_cache_path(symbol, GAP_FILL_INTERVAL, day_str, resume_path)
        cached_source = day_sources.get(day_str)

        if cached_source and cached_source != "unavailable":
            cached_frame = _load_day_frame(cache_path)
            if cached_frame is not None:
                gap_frames.append(cached_frame)
                filled_sources.add(cached_source)
                continue
            # Recorded as filled but the cache file is gone/corrupt - re-fetch.

        gap_frame, source = fetch_gap_fill_day(symbol, day_str)
        if gap_frame is not None and not gap_frame.empty:
            gap_frames.append(gap_frame)
            filled_sources.add(source)
            day_sources[day_str] = source
            _save_day_frame(gap_frame, cache_path)
        else:
            still_missing.append(day_str)
            day_sources[day_str] = "unavailable"
        progress["completed_days"] = sorted(day_sources)
        _save_progress(resume_path, progress)
        time.sleep(request_gap_sec)

    if not gap_frames:
        return frame

    combined = pd.concat([frame] + gap_frames)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    combined.attrs["kline_source"] = frame.attrs.get("kline_source", "binance_history") + "+" + "+".join(sorted(filled_sources))
    if still_missing:
        combined.attrs["kline_missing_months"] = still_missing
    return combined
