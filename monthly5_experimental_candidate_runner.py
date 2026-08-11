#!/usr/bin/env python3
"""Fetch fresh Binance 4h klines and advance the experimental candidate shadow.

This script is NOT wired into eth.py and is not run automatically by
anything in this repo. It is meant to be invoked manually, or by a cron job
/ supervisor entry you add yourself, to periodically advance
monthly5_experimental_candidate_shadow.py's paper trial. It only reads
public Binance market data and writes to its own state/history files; it
never places or touches any real order.
"""

import argparse
import time
from pathlib import Path

import market_history
import monthly5_experimental_candidate_shadow as shadow


DEFAULT_STATE = Path(".runtime/data/btcusdt_monthly5_experimental_candidate_state.json")
DEFAULT_HISTORY = Path(".runtime/data/btcusdt_monthly5_experimental_candidate_history.jsonl")
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "4h"


def fetch_recent_4h_frame(symbol, *, warmup_months):
    end = int(time.time() * 1000)
    start = end - int(warmup_months * 31 * 24 * 3600 * 1000)
    frame = market_history.fetch_klines_from_binance_api(symbol, DEFAULT_INTERVAL, start, end)
    frame.attrs.setdefault("kline_source", "binance_futures_api")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument(
        "--warmup-months",
        type=int,
        default=shadow.WALKFORWARD_LOOKBACK_MONTHS + 2,
        help="how many months of 4h history to fetch for the walk-forward scorer",
    )
    parser.add_argument("--state", default=str(DEFAULT_STATE))
    parser.add_argument("--history", default=str(DEFAULT_HISTORY))
    args = parser.parse_args()

    frame = fetch_recent_4h_frame(args.symbol, warmup_months=args.warmup_months)
    if frame.empty:
        raise SystemExit("no 4h klines returned from Binance")

    state = shadow.run_once(args.state, args.history, frame, now_ts=time.time())
    latest_probe = state.get("latest_probe") or {}
    print(
        "monthly5_experimental_candidate_runner"
        f" usable={bool(latest_probe.get('usable', False))}"
        f" candidate_signal={latest_probe.get('candidate_signal')}"
        f" recovery_active={state.get('recovery_active')}"
        f" span_hours={state.get('span_hours')}"
        f" candidate_return_pct={(state.get('candidate_paper') or {}).get('return_pct')}"
        f" baseline_return_pct={(state.get('baseline_paper') or {}).get('return_pct')}"
        f" promotion_ready={state.get('promotion_ready')}"
        f" execution_allowed=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
