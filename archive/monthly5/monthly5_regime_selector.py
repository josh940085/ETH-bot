"""Causal bull/range/bear strategy-selector research for monthly5."""

import argparse
import json
from collections import Counter
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import market_history
import monthly5_selector_cache


DEFAULT_CACHE_PATH = monthly5_selector_cache.DEFAULT_OUTPUT
DEFAULT_OUTPUT = Path(
    ".runtime/data/backtests/monthly5_search/causal_4h_regime_selector_2020_20260803.json"
)
REPORT_START = "2020-01-01"
DEVELOPMENT_END = "2023-12-31"
HOLDOUT_START = "2024-01-01"
TARGET_RETURN = 0.05
REGIMES = ("up", "range", "down")
SUPPORTED_RETURN_SCHEMAS = {
    monthly5_selector_cache.RETURN_SCHEMA,
    "causal_5m_trade_stop_pessimistic_v1",
}
CONFIGS = (
    {"lookback_days": 365, "nearest_days": 40, "min_regime_days": 30},
    {"lookback_days": 365, "nearest_days": 80, "min_regime_days": 40},
    {"lookback_days": 730, "nearest_days": 40, "min_regime_days": 30},
    {"lookback_days": 730, "nearest_days": 80, "min_regime_days": 40},
)


def load_cache(path=DEFAULT_CACHE_PATH):
    with np.load(Path(path), allow_pickle=True) as cache:
        required_schema = {
            "schema_version": monthly5_selector_cache.SCHEMA_VERSION,
            "feature_schema": monthly5_selector_cache.FEATURE_SCHEMA,
        }
        for field, expected in required_schema.items():
            if field not in cache.files:
                raise ValueError(f"selector cache missing {field}")
            actual = np.asarray(cache[field]).reshape(-1)[0]
            if str(actual) != str(expected):
                raise ValueError(
                    f"selector cache {field} mismatch: expected {expected}, got {actual}"
                )
        if "return_schema" not in cache.files:
            raise ValueError("selector cache missing return_schema")
        return_schema = str(np.asarray(cache["return_schema"]).reshape(-1)[0])
        if return_schema not in SUPPORTED_RETURN_SCHEMAS:
            raise ValueError(f"selector cache unsupported return_schema: {return_schema}")
        for field in ("prefix_stable", "recursive_stable"):
            if field not in cache.files or not bool(np.asarray(cache[field]).reshape(-1)[0]):
                raise ValueError(f"selector cache {field} verification missing or false")
        payload = {
            "R": np.asarray(cache["R"], dtype="float64"),
            "F": np.asarray(cache["F"], dtype="float64"),
            "Xday": np.asarray(cache["Xday"], dtype="float64"),
            "keys": np.asarray(cache["keys"]).astype(str),
            "days": np.asarray(cache["days"]).astype(str),
            "fee": float(np.asarray(cache["fee"]).reshape(-1)[0]),
            "schema_version": int(np.asarray(cache["schema_version"]).reshape(-1)[0]),
            "feature_schema": str(np.asarray(cache["feature_schema"]).reshape(-1)[0]),
            "return_schema": str(np.asarray(cache["return_schema"]).reshape(-1)[0]),
            "cache_prefix_stable": bool(np.asarray(cache["prefix_stable"]).reshape(-1)[0]),
            "cache_recursive_stable": bool(
                np.asarray(cache["recursive_stable"]).reshape(-1)[0]
            ),
        }
    expected = (len(payload["keys"]), len(payload["days"]))
    if payload["R"].shape != expected or payload["F"].shape != expected:
        raise ValueError("selector cache return/flat shape mismatch")
    if payload["Xday"].shape[0] != len(payload["days"]):
        raise ValueError("selector cache feature/day shape mismatch")
    return payload


def load_4h_ohlc(cache_days):
    """Load local Binance 5m archives and aggregate them into completed 4h bars."""
    first_report_day = max(pd.Timestamp(REPORT_START), pd.Timestamp(str(cache_days[0])))
    start = first_report_day.to_pydatetime().replace(tzinfo=timezone.utc)
    end = (
        pd.Timestamp(str(cache_days[-1])).to_pydatetime().replace(tzinfo=timezone.utc)
        + timedelta(days=1)
    )
    frame_5m = market_history.fetch_klines_from_binance_history(
        "BTCUSDT", "5m", int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    )
    frame = frame_5m.resample("4h", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame.attrs["kline_source"] = frame_5m.attrs.get("kline_source", "binance_history_um")
    return frame


def classify_4h_regimes(frame):
    """Classify each completed 4h bar without using a future bar."""
    close = pd.to_numeric(frame["close"], errors="coerce").astype("float64")
    ma25 = close.rolling(25, min_periods=25).mean()
    distance = close / ma25 - 1.0
    slope_5 = ma25 / ma25.shift(4) - 1.0
    up = (distance > 0.006) & (slope_5 > 0.001)
    down = (distance < -0.006) & (slope_5 < -0.001)
    labels = pd.Series("range", index=frame.index, dtype="object")
    labels.loc[up] = "up"
    labels.loc[down] = "down"
    labels.loc[ma25.isna() | slope_5.isna()] = "unknown"
    return labels


def target_day_regimes(cache_days, completed_4h_regimes):
    """Map day t to the 4h regime completed at or before its UTC boundary."""
    target_index = pd.DatetimeIndex(
        pd.to_datetime(np.asarray(cache_days).astype(str), utc=True)
    ).as_unit("ns")
    targets = pd.DataFrame({"target": target_index}).sort_values("target")
    states = completed_4h_regimes.rename("regime").reset_index()
    states.columns = ["completed_at", "regime"]
    states["completed_at"] = pd.DatetimeIndex(
        pd.to_datetime(states["completed_at"], utc=True)
    ).as_unit("ns")
    states = states.sort_values("completed_at")
    merged = pd.merge_asof(
        targets,
        states,
        left_on="target",
        right_on="completed_at",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged["regime"].fillna("unknown").astype(str).to_numpy()


def summarize_4h_intervals(labels):
    valid = labels[labels.isin(REGIMES)]
    if valid.empty:
        return {
            "counts": {},
            "percentages": {},
            "latest": None,
            "interval_count": 0,
            "duration_bars": {},
            "yearly": {},
            "all_intervals": [],
            "recent_intervals": [],
        }
    # Group on the original series so an unknown gap cannot join two intervals.
    group_ids = labels.ne(labels.shift()).cumsum()
    intervals = []
    for _, group in labels.groupby(group_ids):
        if str(group.iloc[0]) not in REGIMES:
            continue
        intervals.append(
            {
                "regime": str(group.iloc[0]),
                "start": group.index[0].isoformat(),
                "end": group.index[-1].isoformat(),
                "bars": int(len(group)),
            }
        )
    counts = valid.value_counts().to_dict()
    total = len(valid)
    duration_bars = {}
    for regime in REGIMES:
        lengths = [row["bars"] for row in intervals if row["regime"] == regime]
        duration_bars[regime] = {
            "intervals": len(lengths),
            "median": round(float(np.median(lengths)), 2) if lengths else 0.0,
            "p90": round(float(np.percentile(lengths, 90)), 2) if lengths else 0.0,
            "max": int(max(lengths)) if lengths else 0,
        }
    yearly = {}
    for year, rows in valid.groupby(valid.index.year):
        year_counts = rows.value_counts().to_dict()
        year_total = len(rows)
        yearly[str(year)] = {
            regime: {
                "bars": int(year_counts.get(regime, 0)),
                "pct": round(float(year_counts.get(regime, 0)) / year_total * 100.0, 2),
            }
            for regime in REGIMES
        }
    return {
        "counts": {regime: int(counts.get(regime, 0)) for regime in REGIMES},
        "percentages": {
            regime: round(float(counts.get(regime, 0)) / total * 100.0, 2)
            for regime in REGIMES
        },
        "latest": intervals[-1],
        "interval_count": len(intervals),
        "duration_bars": duration_bars,
        "yearly": yearly,
        "all_intervals": intervals,
        "recent_intervals": intervals[-20:],
    }


def _candidate_scores(candidate_returns, target_return=TARGET_RETURN):
    q25 = np.nanpercentile(candidate_returns, 25, axis=1)
    hit_rate = np.nanmean(candidate_returns >= target_return, axis=1)
    mean_return = np.nanmean(candidate_returns, axis=1)
    return q25 + hit_rate * target_return + mean_return * 0.10


def run_causal_selector(
    cache,
    regimes,
    *,
    use_regime,
    lookback_days,
    nearest_days,
    min_regime_days,
    warmup_days=180,
    directional_regime_filter=False,
):
    """Select day t using only feature/return pairs ending no later than t-1."""
    returns = cache["R"]
    flats = cache["F"]
    features = cache["Xday"]
    day_count = len(cache["days"])
    chosen = np.full(day_count, -1, dtype="int32")
    selected_returns = np.full(day_count, np.nan, dtype="float64")
    selected_flats = np.full(day_count, np.nan, dtype="float64")
    training_counts = np.zeros(day_count, dtype="int32")
    fallback = np.zeros(day_count, dtype="bool")

    for target_idx in range(max(2, int(warmup_days)), day_count):
        start = max(1, target_idx - int(lookback_days))
        target_history = np.arange(start, target_idx, dtype="int32")
        if use_regime and regimes[target_idx] in REGIMES:
            matching = target_history[regimes[target_history] == regimes[target_idx]]
            if len(matching) >= int(min_regime_days):
                target_history = matching
            else:
                fallback[target_idx] = True

        # Historical target j is represented by information known after j-1.
        train_x = features[target_history - 1]
        query_x = features[target_idx - 1]
        mean = np.nanmean(train_x, axis=0)
        std = np.nanstd(train_x, axis=0)
        std = np.where(std > 1e-9, std, 1.0)
        normalized = np.nan_to_num((train_x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
        query = np.nan_to_num((query_x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
        distances = np.linalg.norm(normalized - query, axis=1)
        nearest_count = min(max(1, int(nearest_days)), len(target_history))
        nearest_targets = target_history[np.argsort(distances)[:nearest_count]]
        scores = _candidate_scores(returns[:, nearest_targets])
        if directional_regime_filter and regimes[target_idx] in REGIMES:
            suffix = {"up": "_lf|", "range": "_ls|", "down": "_sf|"}[
                regimes[target_idx]
            ]
            allowed = np.char.find(cache["keys"].astype(str), suffix) >= 0
            scores = np.where(allowed, scores, -np.inf)
        selected = int(np.argmax(np.nan_to_num(scores, nan=-1e12)))
        chosen[target_idx] = selected
        selected_returns[target_idx] = returns[selected, target_idx]
        selected_flats[target_idx] = flats[selected, target_idx]
        training_counts[target_idx] = len(target_history)

    return {
        "selected_indices": chosen,
        "returns": selected_returns,
        "flats": selected_flats,
        "training_counts": training_counts,
        "fallback": fallback,
    }


def _max_drawdown(returns):
    equity = np.cumprod(1.0 + np.asarray(returns, dtype="float64"))
    peaks = np.maximum.accumulate(equity)
    drawdown = equity / peaks - 1.0
    return float(np.nanmin(drawdown)) if len(drawdown) else 0.0


def summarize_result(cache, regimes, result, *, start, end=None):
    days = pd.to_datetime(cache["days"])
    mask = days >= pd.Timestamp(start)
    if end:
        mask &= days <= pd.Timestamp(end)
    mask &= np.isfinite(result["returns"])
    selected_days = days[mask]
    selected_returns = result["returns"][mask]
    selected_flats = result["flats"][mask]
    selected_indices = result["selected_indices"][mask]
    selected_regimes = regimes[mask]
    frame = pd.DataFrame(
        {"return": selected_returns, "flat": selected_flats}, index=selected_days
    )
    monthly = frame.groupby(frame.index.to_period("M")).agg(
        return_fraction=("return", lambda values: float(np.prod(1.0 + values) - 1.0)),
        flat_fraction=("flat", "mean"),
        days=("return", "size"),
    )
    if end is None and len(monthly):
        last_month = selected_days.max().to_period("M")
        monthly = monthly.loc[monthly.index < last_month]
    monthly_rows = [
        {
            "month": str(month),
            "return_pct": round(float(row.return_fraction) * 100.0, 4),
            "flat_time_pct": round(float(row.flat_fraction) * 100.0, 4),
            "days": int(row.days),
        }
        for month, row in monthly.iterrows()
    ]
    month_returns = monthly["return_fraction"].to_numpy(dtype="float64") if len(monthly) else np.array([])
    regime_rows = {}
    for regime in REGIMES:
        regime_mask = selected_regimes == regime
        picks = Counter(cache["keys"][selected_indices[regime_mask]])
        regime_returns = selected_returns[regime_mask]
        regime_rows[regime] = {
            "days": int(np.sum(regime_mask)),
            "avg_daily_return_pct": round(float(np.mean(regime_returns)) * 100.0, 4)
            if len(regime_returns)
            else 0.0,
            "candidate_count": len(picks),
            "top_picks": [
                {"strategy": key, "days": count} for key, count in picks.most_common(5)
            ],
        }
    return {
        "start": str(selected_days.min().date()) if len(selected_days) else start,
        "end": str(selected_days.max().date()) if len(selected_days) else (end or ""),
        "days": int(len(selected_returns)),
        "months": int(len(monthly)),
        "months_ge_5": int(np.sum(month_returns >= 0.05)),
        "months_ge_0": int(np.sum(month_returns >= 0.0)),
        "min_month_pct": round(float(np.min(month_returns)) * 100.0, 4) if len(month_returns) else 0.0,
        "avg_month_pct": round(float(np.mean(month_returns)) * 100.0, 4) if len(month_returns) else 0.0,
        "median_month_pct": round(float(np.median(month_returns)) * 100.0, 4) if len(month_returns) else 0.0,
        "total_return_pct": round(float(np.prod(1.0 + selected_returns) - 1.0) * 100.0, 4)
        if len(selected_returns)
        else 0.0,
        "max_drawdown_pct": round(_max_drawdown(selected_returns) * 100.0, 4),
        "avg_flat_time_pct": round(float(np.mean(selected_flats)) * 100.0, 4)
        if len(selected_flats)
        else 0.0,
        "fallback_days": int(np.sum(result["fallback"][mask])),
        "by_regime": regime_rows,
        "monthly": monthly_rows,
    }


def _development_rank(summary):
    return (
        summary["months_ge_5"],
        summary["months_ge_0"],
        summary["min_month_pct"],
        summary["max_drawdown_pct"],
        -summary["avg_flat_time_pct"],
    )


def select_development_variant(variants):
    """Pick only from variants whose development tail risk stays within research limits."""
    growth_winner = max(variants, key=lambda row: _development_rank(row["regime_development"]))
    eligible = [
        row
        for row in variants
        if row["regime_development"]["min_month_pct"] >= -25.0
        and row["regime_development"]["max_drawdown_pct"] >= -70.0
    ]
    winner = max(eligible, key=lambda row: _development_rank(row["regime_development"])) if eligible else None
    return winner, growth_winner


def verify_prefix_stability(cache, regimes, config, full_result, use_regime):
    checks = []
    for cutoff in ("2021-12-31", "2023-12-31", "2025-12-31"):
        count = int(np.sum(np.asarray(cache["days"]) <= cutoff))
        sliced = {
            key: value[:, :count] if key in {"R", "F"} else value[:count]
            for key, value in cache.items()
            if key in {"R", "F", "Xday", "days"}
        }
        sliced["keys"] = cache["keys"]
        sliced["fee"] = cache["fee"]
        replay = run_causal_selector(
            sliced,
            regimes[:count],
            use_regime=use_regime,
            directional_regime_filter=bool(
                np.any(np.char.find(cache["keys"].astype(str), "_sf|") >= 0)
            ),
            **config,
        )
        stable = np.array_equal(
            replay["selected_indices"], full_result["selected_indices"][:count]
        )
        checks.append({"cutoff": cutoff, "days": count, "stable": bool(stable)})
    return {"prefix_stable": all(row["stable"] for row in checks), "checks": checks}


def build_report(cache_path=DEFAULT_CACHE_PATH):
    cache = load_cache(cache_path)
    frame_4h = load_4h_ohlc(cache["days"])
    completed_regimes = classify_4h_regimes(frame_4h)
    regimes = target_day_regimes(cache["days"], completed_regimes)
    variants = []
    computed = {}
    directional_filter = bool(
        np.any(np.char.find(cache["keys"].astype(str), "_sf|") >= 0)
    )
    for config in CONFIGS:
        key = f"lb{config['lookback_days']}_k{config['nearest_days']}_min{config['min_regime_days']}"
        regime_result = run_causal_selector(
            cache,
            regimes,
            use_regime=True,
            directional_regime_filter=directional_filter,
            **config,
        )
        baseline_result = run_causal_selector(cache, regimes, use_regime=False, **config)
        computed[key] = (config, regime_result, baseline_result)
        variants.append(
            {
                "name": key,
                "config": config,
                "regime_development": summarize_result(
                    cache, regimes, regime_result, start=REPORT_START, end=DEVELOPMENT_END
                ),
                "baseline_development": summarize_result(
                    cache, regimes, baseline_result, start=REPORT_START, end=DEVELOPMENT_END
                ),
                "regime_holdout": summarize_result(
                    cache, regimes, regime_result, start=HOLDOUT_START
                ),
                "baseline_holdout": summarize_result(
                    cache, regimes, baseline_result, start=HOLDOUT_START
                ),
            }
        )
    winner, growth_winner = select_development_variant(variants)
    analysis_variant = winner or growth_winner
    config, regime_result, baseline_result = computed[analysis_variant["name"]]
    regime_full = summarize_result(cache, regimes, regime_result, start=REPORT_START)
    baseline_full = summarize_result(cache, regimes, baseline_result, start=REPORT_START)
    regime_holdout = summarize_result(cache, regimes, regime_result, start=HOLDOUT_START)
    baseline_holdout = summarize_result(cache, regimes, baseline_result, start=HOLDOUT_START)
    prefix = verify_prefix_stability(cache, regimes, config, regime_result, True)
    regime_wins_holdout = (
        regime_holdout["months_ge_5"] >= baseline_holdout["months_ge_5"]
        and regime_holdout["months_ge_0"] >= baseline_holdout["months_ge_0"]
        and regime_holdout["max_drawdown_pct"] >= baseline_holdout["max_drawdown_pct"]
    )
    return {
        "schema_version": 1,
        "method": "causal_completed_4h_up_range_down_prefix_walk_forward",
        "inputs": {
            "cache_path": str(cache_path),
            "kline_source": frame_4h.attrs.get("kline_source", "binance_history_um"),
            "four_hour_bars": len(frame_4h),
            "candidate_count": len(cache["keys"]),
            "cache_days": len(cache["days"]),
            "cache_start": str(cache["days"][0]),
            "cache_end": str(cache["days"][-1]),
            "fee_fraction_in_cache": cache["fee"],
            "max_leverage": 5,
            "directional_regime_filter": directional_filter,
            "schema_version": cache["schema_version"],
            "feature_schema": cache["feature_schema"],
            "return_schema": cache["return_schema"],
            "cache_prefix_stable": cache["cache_prefix_stable"],
            "cache_recursive_stable": cache["cache_recursive_stable"],
        },
        "regime_definition": {
            "up": "completed 4h close > MA25 by 0.6% and MA25 five-bar slope > 0.1%",
            "down": "completed 4h close < MA25 by 0.6% and MA25 five-bar slope < -0.1%",
            "range": "all other warm 4h states, including price/slope conflict",
            "decision_boundary": "last completed 4h bar at or before target day UTC boundary",
            **summarize_4h_intervals(completed_regimes),
        },
        "selection": {
            "development_period": f"{REPORT_START}..{DEVELOPMENT_END}",
            "holdout_start": HOLDOUT_START,
            "winner": winner["name"] if winner else None,
            "analysis_variant": analysis_variant["name"],
            "winner_config": config,
            "risk_gate": {"min_month_pct_gte": -25.0, "max_drawdown_pct_gte": -70.0},
            "growth_winner": growth_winner["name"],
            "growth_winner_rejected_for_tail_risk": winner is None
            or growth_winner["name"] != winner["name"],
            "variants": variants,
        },
        "regime_selector": {"full": regime_full, "holdout": regime_holdout},
        "causal_baseline": {"full": baseline_full, "holdout": baseline_holdout},
        "comparison": {
            "regime_wins_holdout": regime_wins_holdout,
            "holdout_months_ge_5_delta": regime_holdout["months_ge_5"]
            - baseline_holdout["months_ge_5"],
            "holdout_months_ge_0_delta": regime_holdout["months_ge_0"]
            - baseline_holdout["months_ge_0"],
            "holdout_max_drawdown_delta_pct": round(
                regime_holdout["max_drawdown_pct"] - baseline_holdout["max_drawdown_pct"], 4
            ),
            "holdout_flat_time_delta_pct": round(
                regime_holdout["avg_flat_time_pct"] - baseline_holdout["avg_flat_time_pct"], 4
            ),
        },
        "bias_evidence": {
            **prefix,
            "recursive_stable": cache["cache_recursive_stable"],
            "recursive_source": "transparent cache generator recursive prefix rebuild",
        },
        "prefix_stable": prefix["prefix_stable"],
        "recursive_stable": cache["cache_recursive_stable"],
        "shadow_only": True,
        "deployment_ready": False,
        "deployment_blockers": [
            *([] if winner else ["development_tail_risk_gate_failed"]),
            "monthly_return_target_not_consistent",
            "candidate_matched_tick_execution_evidence_missing",
            "live_shadow_promotion_not_met",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    report = build_report(args.cache)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
