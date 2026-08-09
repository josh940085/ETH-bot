"""Build a transparent causal monthly5 selector cache from Binance 5m K-lines."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import market_history


SCHEMA_VERSION = 1
FEATURE_SCHEMA = "completed_daily_ohlc_19_v1"
RETURN_SCHEMA = "causal_5m_shift1_turnover_fee_v1"
ROUND_TRIP_FEE = 0.0008
FEATURE_WINDOWS = (1, 3, 7, 14, 30, 60)
MA_PAIRS = ((6, 24), (12, 48), (24, 96), (48, 192), (72, 240), (120, 480), (240, 720))
MOMENTUM_WINDOWS = (48, 72, 120, 240, 480, 720)
DONCHIAN_WINDOWS = (24, 48, 72, 120, 240, 480, 720)
LEVERAGES = (1, 2, 3, 4, 5)
DEFAULT_OUTPUT = Path(
    ".runtime/data/backtests/monthly5_search/daily_selector_cache_causal_v1_2020_20260803.npz"
)
DEFAULT_REPORT = Path(
    ".runtime/data/backtests/monthly5_search/daily_selector_cache_causal_v1_2020_20260803_report.json"
)


def strategy_prefixes():
    prefixes = ["buy_hold"]
    prefixes.extend(f"ma{fast}_{slow}_{mode}" for fast, slow in MA_PAIRS for mode in ("lf", "ls"))
    prefixes.extend(f"mom{window}_{mode}" for window in MOMENTUM_WINDOWS for mode in ("lf", "ls"))
    prefixes.extend(f"don{window}_{mode}" for window in DONCHIAN_WINDOWS for mode in ("lf", "ls"))
    return prefixes


def candidate_keys():
    return [
        f"{prefix}|lev{leverage}|stopNone|targetNone|redlev1.0"
        for prefix in strategy_prefixes()
        for leverage in LEVERAGES
    ]


def _parse_prefix(prefix):
    if prefix == "buy_hold":
        return {"family": "buy_hold", "mode": "lf", "fast": 0, "slow": 0}
    family, mode = prefix.rsplit("_", 1)
    if family.startswith("ma"):
        fast, slow = family[2:].split("_", 1)
        return {"family": "ma", "mode": mode, "fast": int(fast), "slow": int(slow)}
    if family.startswith("mom"):
        return {"family": "mom", "mode": mode, "fast": int(family[3:]), "slow": 0}
    if family.startswith("don"):
        return {"family": "don", "mode": mode, "fast": int(family[3:]), "slow": 0}
    raise ValueError(f"unsupported strategy prefix: {prefix}")


def build_signal(frame, prefix):
    spec = _parse_prefix(prefix)
    close = pd.to_numeric(frame["close"], errors="coerce").astype("float64")
    if spec["family"] == "buy_hold":
        state = pd.Series(1.0, index=frame.index)
    elif spec["family"] == "ma":
        fast = close.rolling(spec["fast"], min_periods=spec["fast"]).mean()
        slow = close.rolling(spec["slow"], min_periods=spec["slow"]).mean()
        bullish = fast > slow
        if spec["mode"] == "lf":
            state = bullish.astype("float64")
        elif spec["mode"] == "sf":
            state = (~bullish).astype("float64") * -1.0
        else:
            state = bullish.map({True: 1.0, False: -1.0})
        state.loc[slow.isna()] = 0.0
    elif spec["family"] == "mom":
        reference = close.shift(spec["fast"])
        bullish = close > reference
        if spec["mode"] == "lf":
            state = bullish.astype("float64")
        elif spec["mode"] == "sf":
            state = (~bullish).astype("float64") * -1.0
        else:
            state = bullish.map({True: 1.0, False: -1.0})
        state.loc[reference.isna()] = 0.0
    else:
        prior_high = pd.to_numeric(frame["high"], errors="coerce").shift(1).rolling(
            spec["fast"], min_periods=spec["fast"]
        ).max()
        prior_low = pd.to_numeric(frame["low"], errors="coerce").shift(1).rolling(
            spec["fast"], min_periods=spec["fast"]
        ).min()
        events = pd.Series(np.nan, index=frame.index, dtype="float64")
        events.loc[close > prior_high] = 0.0 if spec["mode"] == "sf" else 1.0
        events.loc[close < prior_low] = 0.0 if spec["mode"] == "lf" else -1.0
        state = events.ffill().fillna(0.0)
    # The position for bar t is fixed before bar t from signal state t-1.
    return state.shift(1).fillna(0.0).astype("float64")


def build_daily_features(frame_5m):
    daily = frame_5m.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "high", "low", "close"])
    previous_close = daily["close"].shift(1)
    true_range = pd.concat(
        [
            daily["high"] - daily["low"],
            (daily["high"] - previous_close).abs(),
            (daily["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    columns = []
    for window in FEATURE_WINDOWS:
        high_max = daily["high"].rolling(window, min_periods=1).max()
        low_min = daily["low"].rolling(window, min_periods=1).min()
        atr = true_range.rolling(window, min_periods=1).mean()
        columns.extend(
            [
                (daily["close"] / high_max - 1.0) * 100.0,
                (high_max / low_min - 1.0) * 100.0,
                atr / daily["close"] * 100.0,
            ]
        )
    columns.append(pd.Series(0.0, index=daily.index))
    features = pd.DataFrame(np.column_stack(columns), index=daily.index)
    return daily, features.astype("float32")


def simulate_daily_returns(frame_5m, position, leverage, *, round_trip_fee=ROUND_TRIP_FEE):
    close = pd.to_numeric(frame_5m["close"], errors="coerce").astype("float64")
    bar_return = close.pct_change().fillna(0.0).to_numpy()
    position_values = position.to_numpy(dtype="float64")
    previous_position = np.concatenate(([0.0], position_values[:-1]))
    turnover = np.abs(position_values - previous_position)
    one_way_fee = float(round_trip_fee) / 2.0
    factor = 1.0 + float(leverage) * position_values * bar_return
    factor -= turnover * float(leverage) * one_way_fee
    factor = np.maximum(factor, 1e-9)

    day_index = frame_5m.index.floor("D")
    codes, unique_days = pd.factorize(day_index, sort=True)
    log_growth = np.bincount(codes, weights=np.log(factor), minlength=len(unique_days))
    daily_return = np.exp(log_growth) - 1.0
    flat_bars = np.bincount(codes, weights=(position_values == 0.0), minlength=len(unique_days))
    bar_counts = np.bincount(codes, minlength=len(unique_days))
    flat_fraction = flat_bars / np.maximum(1, bar_counts)
    return pd.Series(daily_return, index=unique_days), pd.Series(flat_fraction, index=unique_days)


def build_cache(frame_5m, *, start_day, end_day):
    frame = frame_5m.sort_index().copy()
    _, features = build_daily_features(frame)
    target_days = pd.date_range(start_day, end_day, freq="1D", tz="UTC")
    feature_rows = features.reindex(target_days)
    if feature_rows.isna().any().any():
        missing = feature_rows.index[feature_rows.isna().any(axis=1)]
        raise RuntimeError(f"daily feature rows missing: {list(missing[:5])}")

    keys = candidate_keys()
    returns = np.zeros((len(keys), len(target_days)), dtype="float32")
    flats = np.zeros_like(returns)
    metadata = np.zeros((len(keys), 5), dtype="float32")
    row = 0
    for prefix in strategy_prefixes():
        position = build_signal(frame, prefix)
        spec = _parse_prefix(prefix)
        for leverage in LEVERAGES:
            daily_return, daily_flat = simulate_daily_returns(frame, position, leverage)
            returns[row] = daily_return.reindex(target_days).fillna(0.0).to_numpy(dtype="float32")
            flats[row] = daily_flat.reindex(target_days).fillna(1.0).to_numpy(dtype="float32")
            family_code = {"buy_hold": 0, "ma": 1, "mom": 2, "don": 3}[spec["family"]]
            metadata[row] = [leverage, family_code, spec["fast"], spec["slow"], spec["mode"] == "ls"]
            row += 1
    return {
        "R": returns,
        "F": flats,
        "Xday": feature_rows.to_numpy(dtype="float32"),
        "C": metadata,
        "keys": np.asarray(keys),
        "days": np.asarray(target_days.strftime("%Y-%m-%d")),
        "fee": np.asarray([ROUND_TRIP_FEE], dtype="float32"),
        "schema_version": np.asarray([SCHEMA_VERSION], dtype="int32"),
        "feature_schema": np.asarray([FEATURE_SCHEMA]),
        "return_schema": np.asarray([RETURN_SCHEMA]),
    }


def verify_recursive_stability(frame_5m, *, cutoffs):
    prefixes = strategy_prefixes()
    _, full_features = build_daily_features(frame_5m)
    full_signals = {prefix: build_signal(frame_5m, prefix) for prefix in prefixes}
    checks = []
    for cutoff in cutoffs:
        cutoff_ts = pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        truncated = frame_5m.loc[frame_5m.index <= cutoff_ts]
        _, prefix_features = build_daily_features(truncated)
        overlap = prefix_features.index.intersection(full_features.index)
        feature_stable = bool(
            np.allclose(
                prefix_features.loc[overlap].to_numpy(),
                full_features.loc[overlap].to_numpy(),
                rtol=1e-6,
                atol=1e-6,
                equal_nan=True,
            )
        )
        unstable_signals = []
        unstable_returns = []
        for prefix in prefixes:
            prefix_signal = build_signal(truncated, prefix)
            full_signal = full_signals[prefix].reindex(prefix_signal.index)
            if not np.array_equal(prefix_signal.to_numpy(), full_signal.to_numpy()):
                unstable_signals.append(prefix)
                continue
            prefix_return, _ = simulate_daily_returns(truncated, prefix_signal, leverage=5)
            full_return, _ = simulate_daily_returns(
                frame_5m.loc[frame_5m.index <= cutoff_ts],
                full_signals[prefix].loc[full_signals[prefix].index <= cutoff_ts],
                leverage=5,
            )
            if not np.allclose(prefix_return.to_numpy(), full_return.to_numpy(), rtol=1e-9, atol=1e-9):
                unstable_returns.append(prefix)
        checks.append(
            {
                "cutoff": str(cutoff),
                "bars": len(truncated),
                "feature_stable": feature_stable,
                "unstable_signals": unstable_signals,
                "unstable_returns": unstable_returns,
                "stable": feature_stable and not unstable_signals and not unstable_returns,
            }
        )
    stable = all(check["stable"] for check in checks)
    return {"prefix_stable": stable, "recursive_stable": stable, "checks": checks}


def embed_verification(cache, verification):
    payload = dict(cache)
    payload["prefix_stable"] = np.asarray(
        [bool(verification.get("prefix_stable"))], dtype="bool"
    )
    payload["recursive_stable"] = np.asarray(
        [bool(verification.get("recursive_stable"))], dtype="bool"
    )
    return payload


def load_history(start_day, end_day):
    start = pd.Timestamp(start_day, tz="UTC") - pd.Timedelta(days=4)
    end = pd.Timestamp(end_day, tz="UTC") + pd.Timedelta(days=1)
    return market_history.fetch_klines_from_binance_history(
        "BTCUSDT",
        "5m",
        int(start.timestamp() * 1000),
        int(end.timestamp() * 1000),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-08-03")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()
    frame = load_history(args.start, args.end)
    cache = build_cache(frame, start_day=args.start, end_day=args.end)
    recursive = verify_recursive_stability(
        frame,
        cutoffs=("2021-12-31", "2023-12-31", "2025-12-31"),
    )
    cache = embed_verification(cache, recursive)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **cache)
    report = {
        "schema_version": SCHEMA_VERSION,
        "feature_schema": FEATURE_SCHEMA,
        "return_schema": RETURN_SCHEMA,
        "source": frame.attrs.get("kline_source", "binance_history_um"),
        "cache": str(output),
        "candidate_count": len(cache["keys"]),
        "days": len(cache["days"]),
        "start": args.start,
        "end": args.end,
        "max_leverage": max(LEVERAGES),
        "round_trip_fee": ROUND_TRIP_FEE,
        **recursive,
        "deployment_ready": False,
        "deployment_blockers": [
            "selector_holdout_not_evaluated",
            "candidate_matched_tick_execution_evidence_missing",
            "live_shadow_promotion_not_met",
        ],
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if recursive["recursive_stable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
