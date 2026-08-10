#!/usr/bin/env python3
"""Audit monthly5 live risk using local state and optional Binance fills."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any


DEFAULT_POSITION = Path(".runtime/data/docs/position.json")
DEFAULT_SHADOW = Path(".runtime/data/btcusdt_monthly5_shadow_state.json")


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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _local_time(ts: float) -> str:
    if ts <= 0:
        return ""
    tz = dt.timezone(dt.timedelta(hours=8))
    return dt.datetime.fromtimestamp(ts, tz).isoformat(timespec="seconds")


def pair_futures_round_trips(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair simple one-leg Binance futures fills into round trips."""
    open_trade: dict[str, Any] | None = None
    closed: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: _safe_int(item.get("time"), 0)):
        side = str(row.get("side") or "").upper()
        position_side = str(row.get("positionSide") or "").upper()
        price = _safe_float(row.get("price"), 0.0)
        qty = _safe_float(row.get("qty"), 0.0)
        realized_pnl = _safe_float(row.get("realizedPnl"), 0.0)
        commission = _safe_float(row.get("commission"), 0.0)
        ts = _safe_int(row.get("time"), 0) / 1000.0

        is_long_open = position_side == "LONG" and side == "BUY" and realized_pnl == 0.0
        is_short_open = position_side == "SHORT" and side == "SELL" and realized_pnl == 0.0
        if is_long_open or is_short_open:
            open_trade = {
                "direction": "long" if is_long_open else "short",
                "entry_ts": ts,
                "entry_price": price,
                "qty": qty,
                "entry_commission": commission,
            }
            continue

        if not open_trade:
            continue
        closes_long = open_trade["direction"] == "long" and position_side == "LONG" and side == "SELL"
        closes_short = open_trade["direction"] == "short" and position_side == "SHORT" and side == "BUY"
        if not (closes_long or closes_short):
            continue

        fees = _safe_float(open_trade.get("entry_commission"), 0.0) + commission
        net = realized_pnl - fees
        closed.append(
            {
                "direction": open_trade["direction"],
                "entry_ts": open_trade["entry_ts"],
                "exit_ts": ts,
                "entry_local": _local_time(_safe_float(open_trade.get("entry_ts"), 0.0)),
                "exit_local": _local_time(ts),
                "entry_price": round(_safe_float(open_trade.get("entry_price"), 0.0), 4),
                "exit_price": round(price, 4),
                "qty": round(min(qty, _safe_float(open_trade.get("qty"), qty)), 6),
                "gross_pnl": round(realized_pnl, 6),
                "fees": round(fees, 6),
                "net_pnl": round(net, 6),
                "hold_min": round(max(0.0, ts - _safe_float(open_trade.get("entry_ts"), ts)) / 60.0, 2),
                "outcome": "win" if realized_pnl > 0 else "loss" if realized_pnl < 0 else "flat",
            }
        )
        open_trade = None
    return closed


def summarize_round_trips(trades: list[dict[str, Any]], *, limit: int = 12) -> dict[str, Any]:
    recent = trades[-max(1, limit) :]
    wins = sum(1 for item in recent if item.get("outcome") == "win")
    losses = sum(1 for item in recent if item.get("outcome") == "loss")
    loss_streak = 0
    for item in reversed(recent):
        if item.get("outcome") != "loss":
            break
        loss_streak += 1
    total = len(recent)
    return {
        "closed_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round((wins / total) * 100.0, 4) if total else 0.0,
        "loss_streak": loss_streak,
        "gross_pnl_sum": round(sum(_safe_float(item.get("gross_pnl"), 0.0) for item in recent), 6),
        "fees_sum": round(sum(_safe_float(item.get("fees"), 0.0) for item in recent), 6),
        "net_pnl_sum": round(sum(_safe_float(item.get("net_pnl"), 0.0) for item in recent), 6),
        "recent": recent,
    }


def build_audit(
    *,
    position: dict[str, Any],
    shadow: dict[str, Any],
    round_trips: list[dict[str, Any]] | None = None,
    recent_limit: int = 12,
) -> dict[str, Any]:
    readiness = position.get("monthly5_readiness") if isinstance(position.get("monthly5_readiness"), dict) else {}
    selection = shadow.get("market_selection") if isinstance(shadow.get("market_selection"), dict) else {}
    blockers = [str(item) for item in (readiness.get("promotion_blockers") or shadow.get("promotion_blockers") or [])]
    trades = summarize_round_trips(round_trips or [], limit=recent_limit)

    findings: list[str] = []
    actions: list[str] = []
    if not bool(readiness.get("promotion_ready", shadow.get("promotion_ready", False))):
        findings.append("promotion_not_ready")
    for code in ("active_underperforming_plan", "recovery_probe_probe_failed"):
        if code in blockers:
            findings.append(code)
    if _safe_float(readiness.get("shadow_paper_return_pct"), _safe_float(shadow.get("shadow_paper_return_pct"), 0.0)) < 0:
        findings.append("shadow_return_negative")
    if trades["closed_trades"] >= 4 and trades["win_rate_pct"] < 35.0:
        findings.append("recent_live_win_rate_low")
    if trades["loss_streak"] >= 3:
        findings.append("recent_live_loss_streak")

    if any(code in findings for code in ("active_underperforming_plan", "recovery_probe_probe_failed")):
        actions.append("block_monthly5_entry_until_recovery_probe_success")
    if "shadow_return_negative" in findings:
        actions.append("keep_monthly5_shadow_only")
    if "recent_live_win_rate_low" in findings or "recent_live_loss_streak" in findings:
        actions.append("require_live_trade_revalidation_before_takeover")

    severity = "ok"
    if actions:
        severity = "block"
    elif findings:
        severity = "watch"

    return {
        "schema_version": 1,
        "ts": int(time.time()),
        "severity": severity,
        "position": {
            "open": bool(position.get("open", False)),
            "direction": str(position.get("direction") or ""),
            "trade_source": str(position.get("trade_source") or ""),
            "last_close_reason": str(position.get("last_close_reason") or ""),
            "last_close_ts": _safe_int(position.get("last_close_ts"), 0),
        },
        "monthly5": {
            "promotion_ready": bool(readiness.get("promotion_ready", shadow.get("promotion_ready", False))),
            "promotion_blockers": blockers,
            "shadow_return_pct": round(_safe_float(readiness.get("shadow_paper_return_pct"), 0.0), 4),
            "shadow_rolling_return_pct": round(_safe_float(readiness.get("shadow_rolling_paper_return_pct"), 0.0), 4),
            "selected_plan": str(selection.get("selected_plan") or ""),
            "shadow_action": str(selection.get("shadow_action") or ""),
            "exposure_cap": round(_safe_float(selection.get("exposure_cap"), 0.0), 4),
            "reason_codes": [str(item) for item in selection.get("reason_codes") or []],
        },
        "live_trades": trades,
        "findings": sorted(set(findings)),
        "recommended_actions": actions,
    }


def fetch_binance_user_trades(symbol: str, hours: float) -> list[dict[str, Any]]:
    import eth

    start_ms = int((time.time() - max(1.0, hours) * 3600.0) * 1000)
    rows = eth._binance_futures_signed_get(
        "/fapi/v1/userTrades",
        {"symbol": symbol.upper(), "startTime": start_ms, "limit": 1000},
    )
    return rows if isinstance(rows, list) else []


def _print_text(audit: dict[str, Any]) -> None:
    monthly5 = audit["monthly5"]
    trades = audit["live_trades"]
    print(f"severity={audit['severity']}")
    print(
        "monthly5="
        f"promotion_ready={monthly5['promotion_ready']} "
        f"plan={monthly5['selected_plan']} action={monthly5['shadow_action']} "
        f"cap={monthly5['exposure_cap']} blockers={','.join(monthly5['promotion_blockers']) or 'none'}"
    )
    print(
        "live_trades="
        f"closed={trades['closed_trades']} wins={trades['wins']} losses={trades['losses']} "
        f"win_rate={trades['win_rate_pct']} net={trades['net_pnl_sum']} loss_streak={trades['loss_streak']}"
    )
    if audit["findings"]:
        print("findings=" + ",".join(audit["findings"]))
    if audit["recommended_actions"]:
        print("actions=" + ",".join(audit["recommended_actions"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position", type=Path, default=DEFAULT_POSITION)
    parser.add_argument("--shadow", type=Path, default=DEFAULT_SHADOW)
    parser.add_argument("--binance", action="store_true", help="Fetch recent Binance futures user trades.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--hours", type=float, default=72.0)
    parser.add_argument("--recent-limit", type=int, default=12)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    position = _load_json(args.position)
    shadow = _load_json(args.shadow)
    fills = fetch_binance_user_trades(args.symbol, args.hours) if args.binance else []
    audit = build_audit(
        position=position,
        shadow=shadow,
        round_trips=pair_futures_round_trips(fills),
        recent_limit=args.recent_limit,
    )
    if args.json:
        print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_text(audit)
    return 1 if audit.get("severity") == "block" else 0


if __name__ == "__main__":
    raise SystemExit(main())
