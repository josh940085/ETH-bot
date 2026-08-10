"""Build focused causal monthly5 candidates with executable 5m trade stops."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import monthly5_selector_cache as base


SCHEMA_VERSION = 1
RETURN_SCHEMA = "causal_5m_trade_stop_pessimistic_v1"
ROUND_TRIP_FEE = 0.0008
LEVERAGES = (1, 2, 3, 4, 5)
PREFIXES = tuple(
    f"{family}_{mode}"
    for family in (
        "ma72_240",
        "ma120_480",
        "ma240_720",
        "mom480",
        "mom720",
        "don480",
        "don720",
    )
    for mode in ("lf", "ls", "sf")
)
RISK_CONFIGS = (
    (0.010, 0.020, 12),
    (0.015, 0.030, 12),
    (0.020, 0.040, 12),
    (0.030, 0.060, 12),
    (0.020, None, 12),
    (0.030, None, 12),
)
DEFAULT_OUTPUT = Path(
    ".runtime/data/backtests/monthly5_search/risk_selector_cache_causal_v1_2020_20260803.npz"
)
DEFAULT_REPORT = Path(
    ".runtime/data/backtests/monthly5_search/risk_selector_cache_causal_v1_2020_20260803_report.json"
)


def simulate_trade_risk_path(frame, desired_position, *, stop_pct, target_pct, cooldown_bars):
    """Return unlevered close-to-close PnL, turnover, and actual position.

    A signal is already shifted by one bar. Stops use the bar high/low; when
    stop and target are both touched in one bar, the stop is filled first.
    """
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype="float64")
    open_price = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype="float64")
    high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype="float64")
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype="float64")
    desired = np.asarray(desired_position, dtype="float64")
    count = len(frame)
    pnl = np.zeros(count, dtype="float64")
    turnover = np.zeros(count, dtype="float64")
    actual = np.zeros(count, dtype="float64")
    active = 0.0
    entry = 0.0
    cooldown = 0

    for index in range(count):
        previous_close = close[index - 1] if index else close[index]
        wanted = desired[index] if np.isfinite(desired[index]) else 0.0
        if active and wanted != active:
            turnover[index] += abs(active)
            active = 0.0
            entry = 0.0
        if not active:
            if cooldown > 0:
                cooldown -= 1
            elif wanted:
                active = float(np.sign(wanted))
                entry = previous_close
                turnover[index] += abs(active)

        exit_price = None
        if active > 0.0:
            stop_price = entry * (1.0 - float(stop_pct))
            target_price = entry * (1.0 + float(target_pct)) if target_pct else None
            if low[index] <= stop_price:
                exit_price = min(stop_price, open_price[index])
            elif target_price is not None and high[index] >= target_price:
                exit_price = target_price
        elif active < 0.0:
            stop_price = entry * (1.0 + float(stop_pct))
            target_price = entry * (1.0 - float(target_pct)) if target_pct else None
            if high[index] >= stop_price:
                exit_price = max(stop_price, open_price[index])
            elif target_price is not None and low[index] <= target_price:
                exit_price = target_price

        if active:
            mark = float(exit_price) if exit_price is not None else close[index]
            pnl[index] = active * (mark / previous_close - 1.0)
            actual[index] = active
        if exit_price is not None:
            turnover[index] += abs(active)
            active = 0.0
            entry = 0.0
            cooldown = max(0, int(cooldown_bars))

    return pnl, turnover, actual


def aggregate_daily(frame, pnl, turnover, actual, leverage, *, round_trip_fee=ROUND_TRIP_FEE):
    one_way_fee = float(round_trip_fee) / 2.0
    factor = 1.0 + float(leverage) * np.asarray(pnl)
    factor -= float(leverage) * np.asarray(turnover) * one_way_fee
    factor = np.maximum(factor, 1e-9)
    day_index = frame.index.floor("D")
    codes, unique_days = pd.factorize(day_index, sort=True)
    growth = np.bincount(codes, weights=np.log(factor), minlength=len(unique_days))
    counts = np.bincount(codes, minlength=len(unique_days))
    flat = np.bincount(
        codes,
        weights=(np.asarray(actual) == 0.0),
        minlength=len(unique_days),
    ) / np.maximum(1, counts)
    return (
        pd.Series(np.exp(growth) - 1.0, index=unique_days),
        pd.Series(flat, index=unique_days),
    )


def candidate_key(prefix, leverage, stop_pct, target_pct, cooldown_bars):
    target = "None" if target_pct is None else f"{target_pct:.3f}"
    return (
        f"{prefix}|lev{leverage}|stop-{stop_pct:.3f}|target{target}"
        f"|cooldown{int(cooldown_bars)}|redlev1.0"
    )


def build_cache(frame_5m, *, start_day, end_day):
    frame = frame_5m.sort_index().copy()
    _, features = base.build_daily_features(frame)
    target_days = pd.date_range(start_day, end_day, freq="1D", tz="UTC")
    feature_rows = features.reindex(target_days)
    if feature_rows.isna().any().any():
        raise RuntimeError("risk cache daily feature rows missing")

    keys = []
    return_rows = []
    flat_rows = []
    for prefix in PREFIXES:
        desired = base.build_signal(frame, prefix)
        for stop_pct, target_pct, cooldown_bars in RISK_CONFIGS:
            pnl, turnover, actual = simulate_trade_risk_path(
                frame,
                desired,
                stop_pct=stop_pct,
                target_pct=target_pct,
                cooldown_bars=cooldown_bars,
            )
            for leverage in LEVERAGES:
                daily_return, daily_flat = aggregate_daily(
                    frame, pnl, turnover, actual, leverage
                )
                return_rows.append(
                    daily_return.reindex(target_days).fillna(0.0).to_numpy(dtype="float32")
                )
                flat_rows.append(
                    daily_flat.reindex(target_days).fillna(1.0).to_numpy(dtype="float32")
                )
                keys.append(
                    candidate_key(prefix, leverage, stop_pct, target_pct, cooldown_bars)
                )
    return {
        "R": np.asarray(return_rows, dtype="float32"),
        "F": np.asarray(flat_rows, dtype="float32"),
        "Xday": feature_rows.to_numpy(dtype="float32"),
        "keys": np.asarray(keys),
        "days": np.asarray(target_days.strftime("%Y-%m-%d")),
        "fee": np.asarray([ROUND_TRIP_FEE], dtype="float32"),
        "schema_version": np.asarray([SCHEMA_VERSION], dtype="int32"),
        "feature_schema": np.asarray([base.FEATURE_SCHEMA]),
        "return_schema": np.asarray([RETURN_SCHEMA]),
    }


def verify_prefix_stability(cache, frame, *, start_day, cutoffs):
    checks = []
    for cutoff in cutoffs:
        cutoff_ts = pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        truncated = build_cache(
            frame.loc[frame.index <= cutoff_ts], start_day=start_day, end_day=cutoff
        )
        count = int(np.sum(np.asarray(cache["days"]).astype(str) <= str(cutoff)))
        stable = (
            np.array_equal(cache["R"][:, :count], truncated["R"])
            and np.array_equal(cache["F"][:, :count], truncated["F"])
            and np.array_equal(cache["Xday"][:count], truncated["Xday"])
            and np.array_equal(cache["keys"], truncated["keys"])
            and np.array_equal(cache["days"][:count], truncated["days"])
        )
        checks.append({"cutoff": cutoff, "days": count, "stable": bool(stable)})
    return {
        "prefix_stable": all(row["stable"] for row in checks),
        "recursive_stable": all(row["stable"] for row in checks),
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-08-03")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()
    frame = base.load_history(args.start, args.end)
    cache = build_cache(frame, start_day=args.start, end_day=args.end)
    recursive = verify_prefix_stability(
        cache,
        frame,
        start_day=args.start,
        cutoffs=("2021-12-31", "2023-12-31", "2025-12-31"),
    )
    cache["prefix_stable"] = np.asarray([recursive["prefix_stable"]], dtype="bool")
    cache["recursive_stable"] = np.asarray([recursive["recursive_stable"]], dtype="bool")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **cache)
    report = {
        "schema_version": SCHEMA_VERSION,
        "feature_schema": base.FEATURE_SCHEMA,
        "return_schema": RETURN_SCHEMA,
        "source": frame.attrs.get("kline_source", "binance_history_um"),
        "cache": str(output),
        "candidate_count": len(cache["keys"]),
        "days": len(cache["days"]),
        "start": args.start,
        "end": args.end,
        "max_leverage": max(LEVERAGES),
        "round_trip_fee": ROUND_TRIP_FEE,
        "same_bar_policy": "stop_before_target",
        "signal_lag_bars": 1,
        **recursive,
        "deployment_ready": False,
        "deployment_blockers": [
            "walk_forward_holdout_not_evaluated",
            "candidate_matched_tick_execution_evidence_missing",
            "live_shadow_promotion_not_met",
        ],
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if recursive["recursive_stable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
