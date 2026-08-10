"""Replay strategy signal timestamps against official Binance trade ticks."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

import binance_trade_history


TIME_ALIASES = {
    "entry_time": ("entry_time", "opened_at", "open_time"),
    "exit_time": ("exit_time", "closed_at", "close_time"),
}


def _first_column(frame, names):
    return next((name for name in names if name in frame.columns), "")


def _first_trade_at_or_after(frame, timestamp):
    matches = frame.loc[frame["event_time"] >= timestamp]
    return matches.iloc[0] if not matches.empty else None


def _direction(value):
    text = str(value or "").strip().lower()
    if text in {"long", "buy"}:
        return "long"
    if text in {"short", "sell"}:
        return "short"
    return ""


def replay_signals(
    signal_path,
    *,
    symbol="BTCUSDT",
    data_type="aggTrades",
    taker_fee_rate=0.0005,
    limit=0,
    candidate="",
):
    signals = pd.read_csv(signal_path)
    entry_column = _first_column(signals, TIME_ALIASES["entry_time"])
    exit_column = _first_column(signals, TIME_ALIASES["exit_time"])
    side_column = _first_column(signals, ("side", "direction", "position_side"))
    size_column = _first_column(signals, ("quantity", "qty", "size"))
    entry_reference_column = _first_column(signals, ("entry_reference_price", "entry", "avg_entry"))
    exit_reference_column = _first_column(signals, ("exit_reference_price", "exit"))
    required = {
        "entry_time": entry_column,
        "exit_time": exit_column,
        "side": side_column,
        "quantity": size_column,
    }
    missing = [name for name, column in required.items() if not column]
    if missing:
        raise ValueError(f"signal evidence missing columns: {', '.join(missing)}")
    if limit > 0:
        signals = signals.head(limit)

    parsed = signals.copy()
    parsed["_entry_time"] = pd.to_datetime(parsed[entry_column], utc=True, errors="coerce")
    parsed["_exit_time"] = pd.to_datetime(parsed[exit_column], utc=True, errors="coerce")
    required_days = sorted(
        {
            timestamp.floor("D")
            for timestamp in pd.concat([parsed["_entry_time"], parsed["_exit_time"]]).dropna()
        }
    )
    days = {
        day.strftime("%Y-%m-%d"): binance_trade_history.load_trade_day(symbol, day, data_type)
        for day in required_days
    }
    evidence = []
    errors = []
    for index, row in parsed.iterrows():
        entry_time = row["_entry_time"]
        exit_time = row["_exit_time"]
        side = _direction(row[side_column])
        quantity = float(row[size_column])
        if pd.isna(entry_time) or pd.isna(exit_time) or exit_time <= entry_time:
            errors.append({"row": int(index), "error": "invalid_signal_time"})
            continue
        if not side or quantity <= 0:
            errors.append({"row": int(index), "error": "invalid_side_or_quantity"})
            continue
        entry_ticks = days.get(entry_time.strftime("%Y-%m-%d"), pd.DataFrame())
        exit_ticks = days.get(exit_time.strftime("%Y-%m-%d"), pd.DataFrame())
        entry_fill = _first_trade_at_or_after(entry_ticks, entry_time) if not entry_ticks.empty else None
        exit_fill = _first_trade_at_or_after(exit_ticks, exit_time) if not exit_ticks.empty else None
        if entry_fill is None or exit_fill is None:
            errors.append({"row": int(index), "error": "official_tick_missing"})
            continue
        entry_price = float(entry_fill["price"])
        exit_price = float(exit_fill["price"])
        direction_factor = 1.0 if side == "long" else -1.0
        gross_pnl = direction_factor * (exit_price - entry_price) * quantity
        fee = (entry_price + exit_price) * quantity * max(0.0, float(taker_fee_rate))
        entry_reference = float(row[entry_reference_column]) if entry_reference_column else entry_price
        exit_reference = float(row[exit_reference_column]) if exit_reference_column else exit_price
        entry_slippage = direction_factor * (entry_price - entry_reference) * quantity
        exit_slippage = direction_factor * (exit_reference - exit_price) * quantity
        slippage = entry_slippage + exit_slippage
        evidence.append(
            {
                "entry_time": entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "side": side,
                "quantity": quantity,
                "entry_reference_price": entry_reference,
                "exit_reference_price": exit_reference,
                "entry_fill_time": entry_fill["event_time"].isoformat(),
                "exit_fill_time": exit_fill["event_time"].isoformat(),
                "entry_fill_price": entry_price,
                "exit_fill_price": exit_price,
                "gross_pnl": gross_pnl,
                "fee": fee,
                "entry_slippage": entry_slippage,
                "exit_slippage": exit_slippage,
                "slippage": slippage,
                "pnl": gross_pnl - fee,
                "data_source": f"binance_public_data_usdm_{data_type}",
                "candidate": str(candidate or ""),
            }
        )
    signal_sha256 = hashlib.sha256(Path(signal_path).read_bytes()).hexdigest()
    return pd.DataFrame(evidence), {
        "symbol": symbol,
        "data_type": data_type,
        "signal_rows": int(len(parsed)),
        "evidence_rows": len(evidence),
        "errors": errors,
        "complete": len(evidence) == len(parsed) and not errors,
        "taker_fee_rate": float(taker_fee_rate),
        "candidate": str(candidate or ""),
        "signal_source": str(Path(signal_path)),
        "signal_sha256": signal_sha256,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--data-type", choices=sorted(binance_trade_history.SUPPORTED_DATA_TYPES), default="aggTrades")
    parser.add_argument("--taker-fee-rate", type=float, default=0.0005)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--candidate", default="")
    args = parser.parse_args()
    evidence, report = replay_signals(
        args.signals,
        symbol=args.symbol,
        data_type=args.data_type,
        taker_fee_rate=args.taker_fee_rate,
        limit=args.limit,
        candidate=args.candidate,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(output, index=False)
    report_path = output.with_suffix(output.suffix + ".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report, "output": str(output), "report": str(report_path)}, ensure_ascii=False, indent=2))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
