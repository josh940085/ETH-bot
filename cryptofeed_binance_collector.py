"""Collect Binance Futures trades and L2 updates for later execution replay."""

import argparse
import asyncio
import json
import time
from pathlib import Path


def _write_jsonl(handle, payload):
    handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTC-USDT-PERP")
    parser.add_argument("--duration-sec", type=int, default=300)
    parser.add_argument(
        "--output",
        default=".runtime/data/market_microstructure/binance_futures_btcusdt.jsonl",
    )
    args = parser.parse_args()

    try:
        from cryptofeed import FeedHandler
        from cryptofeed.defines import L2_BOOK, TRADES
        from cryptofeed.exchanges import BinanceFutures
    except ImportError as exc:
        raise SystemExit(
            "Run with .runtime/research_envs/cryptofeed/bin/python: " + str(exc)
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"trade": 0, "l2_book": 0}
    started_at = time.time()
    with output.open("a", encoding="utf-8") as handle:

        async def trade_callback(trade, receipt_timestamp):
            counts["trade"] += 1
            _write_jsonl(
                handle,
                {
                    "type": "trade",
                    "receipt_timestamp": receipt_timestamp,
                    "data": trade.to_dict(numeric_type=float),
                },
            )

        async def book_callback(book, receipt_timestamp):
            counts["l2_book"] += 1
            _write_jsonl(
                handle,
                {
                    "type": "l2_book",
                    "receipt_timestamp": receipt_timestamp,
                    "data": book.to_dict(delta=True, numeric_type=float),
                },
            )

        feed = FeedHandler(config={"log": {"disabled": True}})
        # FeedHandler may install uvloop's policy during construction.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        feed.add_feed(
            BinanceFutures(
                symbols=[args.symbol],
                channels=[TRADES, L2_BOOK],
                callbacks={TRADES: trade_callback, L2_BOOK: book_callback},
                max_depth=100,
            )
        )
        loop.call_later(max(1, args.duration_sec), loop.stop)
        feed.run(install_signal_handlers=False)

    report = {
        "success": counts["trade"] > 0 and counts["l2_book"] > 0,
        "symbol": args.symbol,
        "output": str(output),
        "duration_sec": round(time.time() - started_at, 3),
        "counts": counts,
    }
    report_path = output.with_suffix(output.suffix + ".status.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
