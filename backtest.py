#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import warnings
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


def _parse_args():
    parser = argparse.ArgumentParser(description="Replay ETH-bot strategy on historical TradingView klines.")
    parser.add_argument("--symbol", default="ETHUSDT", help="Trading symbol, e.g. ETHUSDT")
    parser.add_argument("--days", type=int, default=30, help="Lookback days when start/end are not provided")
    parser.add_argument("--start", help="UTC start time, e.g. 2026-05-01 or 2026-05-01T00:00:00")
    parser.add_argument("--end", help="UTC end time, e.g. 2026-05-22 or 2026-05-22T00:00:00")
    parser.add_argument("--warmup-bars", type=int, default=1500, help="5m warmup bars before evaluating signals")
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
    end_dt = _parse_utc(args.end) or dt.datetime.now(dt.timezone.utc)
    start_dt = _parse_utc(args.start)
    if start_dt is None:
        start_dt = end_dt - dt.timedelta(days=max(1, int(args.days)))
    if start_dt >= end_dt:
        raise SystemExit("start must be earlier than end")
    return start_dt, end_dt


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


def fetch_futures_klines(symbol, interval, start_ms, end_ms, limit=1500):
    try:
        frame = _fetch_klines_from_market_source(symbol, interval, start_ms, end_ms, limit=limit)
    except Exception as exc:
        raise SystemExit(f"No klines returned from TradingView market data source: {exc}") from exc
    if not frame.empty:
        print(f"📈 回測K線來源: {frame.attrs.get('kline_source', 'unknown')}")
        return frame
    raise SystemExit("No klines returned from TradingView market data source")


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


def build_frame_map(base_5m):
    frame_map = {
        "5m": eth.calc_indicators(base_5m.copy()),
        "15m": eth.calc_indicators(resample_ohlcv(base_5m, "15min")),
        "30m": eth.calc_indicators(resample_ohlcv(base_5m, "30min")),
        "1h": eth.calc_indicators(resample_ohlcv(base_5m, "1h")),
        "4h": eth.calc_indicators(resample_ohlcv(base_5m, "4h")),
        "12h": eth.calc_indicators(resample_ohlcv(base_5m, "12h")),
        "1d": eth.calc_indicators(resample_ohlcv(base_5m, "1d")),
        "1w": eth.calc_indicators(resample_ohlcv(base_5m, "7D")),
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


def summarize_trades(trades, start_dt, end_dt, symbol, model_loaded, data_source="futures"):
    if not trades:
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
    }


def _write_csv_atomic(frame, path_str):
    out_path = Path(path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f"{out_path.name}.tmp")
    frame.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)


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
    size = float(min(1.0, max(size, 0.1)))
    max_size, min_size = eth._derive_scaling_bounds(size)
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
        "content_override": decision.get("content_override") if isinstance(decision.get("content_override"), dict) else {},
        "learned_entry_logic": decision.get("learned_entry_logic") if isinstance(decision.get("learned_entry_logic"), dict) else {},
        "primary_indicator": str(decision.get("primary_indicator") or ""),
        "max_favorable_move_pct": 0.0,
        "max_adverse_move_pct": 0.0,
        "max_size": max_size,
        "min_size": min_size,
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


def run_backtest(symbol, start_dt, end_dt, warmup_bars):
    base_5m = fetch_futures_klines(
        symbol=symbol,
        interval="5m",
        start_ms=int(start_dt.timestamp() * 1000),
        end_ms=int(end_dt.timestamp() * 1000),
    )
    if base_5m.empty:
        raise SystemExit("No klines returned from Binance futures API")

    frame_map = build_frame_map(base_5m)
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

            if open_trade is not None:
                atr_5m = float(max(bar_high - bar_low, 0.0))
                open_trade = _apply_trade_management(open_trade, current_price, atr_5m, ts)
                continue

            if idx % 3 != 0:
                continue

            frame_now = {name: frame.loc[:ts] for name, frame in frame_map.items()}
            if any(len(frame_now[key]) < 30 for key in ("15m", "30m", "1h", "4h")):
                continue

            sr_frames = {key: frame_now.get(key) for key in ("1d", "12h", "4h", "1h", "30m")}
            sr_analysis = eth.analyze_multi_tf_sr_frames(current_price, sr_frames, tf_cfg=sr_cfg)

            decision = eth.build_trade_signal_snapshot(
                df_4h=frame_now["4h"],
                df_1h=frame_now["1h"],
                df_30m=frame_now["30m"],
                df_15m=frame_now["15m"],
                df_5m=frame_now["5m"],
                price=current_price,
                sr_analysis=sr_analysis,
                sp_change=0.0,
                nq_change=0.0,
                btc_change=0.0,
                dxy_change=0.0,
                news_bias=0.0,
                event_risk=0,
                last_signal=last_signal_value,
                losing_streak=losing_streak,
                df_1d=frame_now["1d"],
                df_1w=frame_now["1w"],
            )

            eth._update_scaling_market_state(
                price=current_price,
                atr=float(decision["atr"]),
                htf=int(decision["htf"]),
                mid_trend=int(decision["mid_trend"]),
                regime=str(decision["regime"]),
                breakout=int(decision["breakout"]),
                sr_analysis=sr_analysis,
            )

            score = float(decision["score"])
            final = str(decision["final"])
            current_direction = eth.get_signal_direction(final)
            last_direction_simple = eth.get_signal_direction(last_trade_signal) if last_trade_signal else None

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

            if final.startswith("觀望"):
                continue
            if decision["fake_breakout"] and abs(score - 0.5) < 0.22:
                continue
            if "做多" in final and decision["resistance_hits"] >= 2 and score < 0.72:
                continue
            if "做空" in final and decision["support_hits"] >= 2 and score > 0.28:
                continue

            entry = current_price
            direction = "long" if "做多" in final else "short"
            open_trade = _build_open_trade(ts, direction, final, entry, score, decision)
            last_trade_time = ts_sec
            last_trade_signal = final
            last_entry_price = entry

        if open_trade is not None:
            last_bar = base_rows.iloc[-1]
            exit_price = float(last_bar["close"])
            equity, trade_record, _ = _close_trade(open_trade, exit_price, "EOD", base_rows.index[-1], equity)
            trades.append(trade_record)

    data_source = str(base_5m.attrs.get("kline_source") or "futures")
    return base_5m, trades, summarize_trades(trades, start_dt, end_dt, symbol, model_loaded, data_source=data_source), learning_samples


def main():
    args = _parse_args()
    start_dt, end_dt = _resolve_timerange(args)
    base_5m, trades, summary, learning_samples = run_backtest(args.symbol, start_dt, end_dt, args.warmup_bars)

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
