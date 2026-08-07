#!/usr/bin/env python3
"""Verify the live Binance account is safe for monthly5 promotion."""

import argparse
import contextlib
import io
import json
import os
from types import SimpleNamespace
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if numeric != numeric:
        return default
    return numeric


def _load_eth_module():
    # eth.py prints startup notices on import; keep verifier output machine-readable.
    with contextlib.redirect_stdout(io.StringIO()):
        import eth  # pylint: disable=import-outside-toplevel

    return eth


def _open_positions(rows: Any, symbol: str) -> list[dict]:
    if not isinstance(rows, list):
        rows = [rows]
    symbol = str(symbol or "").upper()
    positions: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if symbol and str(row.get("symbol") or "").upper() != symbol:
            continue
        amt = _safe_float(row.get("positionAmt"), 0.0)
        if abs(amt) <= 0.0:
            continue
        positions.append(
            {
                "symbol": str(row.get("symbol") or symbol),
                "positionAmt": str(row.get("positionAmt") or ""),
                "entryPrice": str(row.get("entryPrice") or ""),
                "markPrice": str(row.get("markPrice") or ""),
                "leverage": str(row.get("leverage") or ""),
            }
        )
    return positions


def _verify_account_flat(eth_module=None, *, symbol: str | None = None) -> tuple[bool, dict]:
    eth_module = eth_module or _load_eth_module()
    symbol = str(symbol or getattr(eth_module, "COPY_TRADE_SYMBOL", "") or os.getenv("TRADE_SYMBOL", "BTCUSDT")).upper()
    rows = eth_module._binance_futures_signed_get("/fapi/v2/positionRisk", {"symbol": symbol})
    positions = _open_positions(rows, symbol)
    return len(positions) == 0, {
        "symbol": symbol,
        "open_count": len(positions),
        "positions": positions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="")
    args = parser.parse_args(argv)

    try:
        ok, payload = _verify_account_flat(symbol=args.symbol or None)
    except Exception as exc:  # pragma: no cover - live exchange error detail is runtime-dependent.
        print(f"FAIL monthly5_account error={type(exc).__name__} message={exc}")
        return 1

    summary = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if ok:
        print(f"PASS monthly5_account {summary}")
        return 0
    print(f"FAIL monthly5_account {summary}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
