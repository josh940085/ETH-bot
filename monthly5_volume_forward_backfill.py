#!/usr/bin/env python3
"""Backfill the frozen monthly5 volume candidate with unseen Binance 5m data."""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

import market_history
import monthly5_volume_forward_shadow as shadow
import verify_monthly5_volume_forward as verifier


DEFAULT_STATE = Path(".runtime/data/btcusdt_monthly5_volume_forward_state.json")
DEFAULT_HISTORY = Path(".runtime/data/btcusdt_monthly5_volume_forward_history.jsonl")
BINANCE_FUTURES_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"


def _fetch_binance_tail(symbol, interval, start_ms, end_ms):
    rows = []
    cursor = int(start_ms)
    session = requests.Session()
    session.headers.update({"User-Agent": "ETH-bot-monthly5-forward/1.0"})
    while cursor < int(end_ms):
        response = session.get(
            BINANCE_FUTURES_KLINES_URL,
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": int(end_ms),
                "limit": 1500,
            },
            timeout=20,
        )
        response.raise_for_status()
        page = response.json()
        if not isinstance(page, list) or not page:
            break
        rows.extend(page)
        next_cursor = int(page[-1][6]) + 1
        if next_cursor <= cursor:
            raise RuntimeError("Binance kline pagination did not advance")
        cursor = next_cursor
        if len(page) < 1500:
            break
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
        ],
    )
    frame = frame.drop_duplicates(subset=["open_time"]).sort_values("open_time")
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
    frame = frame.set_index("close_time")[["open", "high", "low", "close", "volume"]]
    frame.attrs["kline_source"] = "binance_futures_api"
    return frame


def fetch_forward_frame(symbol, interval, start_ms, end_ms):
    archived = market_history.fetch_klines_from_binance_history(
        symbol, interval, start_ms, end_ms
    )
    last_close_ms = int(archived.index.max().timestamp() * 1000)
    tail = _fetch_binance_tail(symbol, interval, last_close_ms + 1, end_ms)
    if tail.empty:
        return archived
    frame = pd.concat([archived, tail]).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame.attrs["kline_source"] = (
        f"{archived.attrs.get('kline_source', 'binance_history')}+binance_futures_api"
    )
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=shadow.FORWARD_START)
    parser.add_argument("--end", default=None)
    parser.add_argument("--warmup-days", type=int, default=30)
    parser.add_argument("--state", default=str(DEFAULT_STATE))
    parser.add_argument("--history", default=str(DEFAULT_HISTORY))
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = pd.Timestamp.now(tz="UTC") if args.end is None else pd.Timestamp(args.end)
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    research_end = pd.Timestamp(shadow.RESEARCH_HISTORY_END)
    if start <= research_end:
        raise SystemExit("forward start must be after frozen research history")
    if end <= start:
        raise SystemExit("end must be after start")

    fetch_start = start - pd.Timedelta(days=max(7, args.warmup_days))
    frame = fetch_forward_frame(
        "BTCUSDT",
        "5m",
        int(fetch_start.timestamp() * 1000),
        int(end.timestamp() * 1000),
    )
    probes = shadow.build_backfill_probes(
        frame,
        start_ts=start,
        now_ts=end.timestamp(),
    )
    if not probes:
        raise SystemExit("no completed forward probes available")
    rows = shadow.merge_history(args.history, probes)
    latest = rows[-1]
    state = shadow.update_files(args.state, args.history, latest, now_ts=time.time())
    failures = verifier.verify(state, rows, 1_000_000_000.0)
    if failures:
        raise SystemExit("forward shadow verification failed: " + "; ".join(failures))
    regimes = {name: 0 for name in ("up", "down", "range", "unknown")}
    for row in rows:
        name = str(row.get("market_regime_4h") or "unknown")
        regimes[name] = regimes.get(name, 0) + 1
    print(
        "PASS monthly5_volume_forward_backfill"
        f" rows={len(rows)} span_hours={state['span_hours']}"
        f" candidate_return_pct={state['candidate_paper']['return_pct']}"
        f" baseline_return_pct={state['baseline_paper']['return_pct']}"
        f" regimes={regimes} execution_allowed=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
