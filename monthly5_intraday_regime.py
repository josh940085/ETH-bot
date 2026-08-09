"""Causal 4h regime policy with 5m execution and monthly 5% floor protection."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator

import monthly5_risk_cache
import monthly5_selector_cache


REPORT_START = "2020-01-01"
DEVELOPMENT_END = "2023-12-31"
HOLDOUT_START = "2024-01-01"
REGIME_CONFIGS = (
    (0.003, 0.0005),
    (0.006, 0.0010),
    (0.010, 0.0020),
)
RANGE_CONFIGS = (
    ("flat", 0.0, 0.0),
)
RISK_CONFIGS = (
    (0.010, 0.020, 12),
    (0.015, 0.030, 12),
    (0.020, 0.040, 12),
    (0.030, 0.060, 12),
)
LEVERAGES = (1, 2, 3, 4, 5)
# Keep the research aligned with the executable live shadow policy.
POST_LOCK_SCALES = (0.15,)
DAILY_STOPS = (None, -0.02, -0.04)
MONTHLY_STOP = -0.08
LOCK_TRIGGER = 0.055
LOCK_FLOOR = 0.051
DEFAULT_OUTPUT = Path(
    ".runtime/data/backtests/monthly5_search/intraday_4h_regime_v1_2020_20260803.json"
)


def classify_completed_4h(frame_5m, *, distance_threshold, slope_threshold):
    frame_4h = frame_5m.resample("4h", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "high", "low", "close"])
    close = pd.to_numeric(frame_4h["close"], errors="coerce").astype("float64")
    ma25 = close.rolling(25, min_periods=25).mean()
    distance = close / ma25 - 1.0
    slope = ma25 / ma25.shift(4) - 1.0
    labels = pd.Series("range", index=frame_4h.index, dtype="object")
    labels.loc[(distance > distance_threshold) & (slope > slope_threshold)] = "up"
    labels.loc[(distance < -distance_threshold) & (slope < -slope_threshold)] = "down"
    labels.loc[ma25.isna() | slope.isna()] = "unknown"
    # A 4h close is actionable only from the next 5m bar.
    labels.index = labels.index + pd.Timedelta(minutes=5)
    return labels


def completed_15m_rsi(frame_5m):
    close_15m = frame_5m["close"].resample("15min", label="right", closed="right").last().dropna()
    rsi = RSIIndicator(close_15m.astype("float64"), window=14, fillna=False).rsi()
    rsi.index = rsi.index + pd.Timedelta(minutes=5)
    return rsi


def completed_daily_trend(frame_5m, *, window):
    close = frame_5m["close"].resample("1D", label="right", closed="right").last().dropna()
    average = close.rolling(int(window), min_periods=int(window)).mean()
    trend = pd.Series("unknown", index=close.index, dtype="object")
    trend.loc[close > average] = "up"
    trend.loc[close < average] = "down"
    trend.index = trend.index + pd.Timedelta(minutes=5)
    return trend


def align_completed_series(target_index, values, default):
    source = values.rename("value").reset_index()
    source.columns = ["available_at", "value"]
    source["available_at"] = pd.DatetimeIndex(
        pd.to_datetime(source["available_at"], utc=True)
    ).as_unit("ns")
    target = pd.DataFrame(
        {"target": pd.DatetimeIndex(pd.to_datetime(target_index, utc=True)).as_unit("ns")}
    )
    merged = pd.merge_asof(
        target.sort_values("target"),
        source.sort_values("available_at"),
        left_on="target",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged["value"].fillna(default).to_numpy()


def build_regime_position(
    frame_5m,
    labels,
    rsi,
    *,
    range_mode,
    rsi_low,
    rsi_high,
    daily_trend=None,
    daily_alignment_mode="none",
):
    regimes = align_completed_series(frame_5m.index, labels, "unknown").astype(str)
    rsi_values = align_completed_series(frame_5m.index, rsi, 50.0).astype("float64")
    daily_values = (
        align_completed_series(frame_5m.index, daily_trend, "unknown").astype(str)
        if daily_trend is not None
        else np.full(len(frame_5m), "unknown", dtype="object")
    )
    position = np.zeros(len(frame_5m), dtype="float64")
    range_state = 0.0
    for index, regime in enumerate(regimes):
        aligned = daily_values[index]
        if daily_alignment_mode == "strict" and regime in {"up", "down"} and aligned != regime:
            regime = "range"
        elif daily_alignment_mode == "no_conflict" and (
            (regime == "up" and aligned == "down")
            or (regime == "down" and aligned == "up")
        ):
            regime = "range"
        if regime == "up":
            position[index] = 1.0
        elif regime == "down":
            position[index] = -1.0
        elif regime == "range" and range_mode == "rsi":
            value = rsi_values[index]
            if value <= rsi_low:
                range_state = 1.0
            elif value >= rsi_high:
                range_state = -1.0
            elif (range_state > 0 and value >= 50.0) or (range_state < 0 and value <= 50.0):
                range_state = 0.0
            position[index] = range_state
        else:
            range_state = 0.0
    return pd.Series(position, index=frame_5m.index), regimes


def apply_monthly_lock(
    frame_5m,
    pnl,
    turnover,
    actual,
    *,
    leverage,
    lock_scale,
    lock_trigger=LOCK_TRIGGER,
    lock_floor=LOCK_FLOOR,
    daily_stop=None,
    monthly_stop=MONTHLY_STOP,
    monthly_recovery_scale=0.0,
    recovery_exit=-0.03,
    recovery_pnl=None,
    recovery_turnover=None,
    recovery_actual=None,
    round_trip_fee=monthly5_risk_cache.ROUND_TRIP_FEE,
):
    one_way_fee = float(round_trip_fee) / 2.0
    month_keys = frame_5m.index.tz_localize(None).to_period("M")
    factors = np.ones(len(frame_5m), dtype="float64")
    scales = np.ones(len(frame_5m), dtype="float64")
    month_equity = 1.0
    day_equity = 1.0
    locked = False
    guarded = False
    daily_guarded = False
    monthly_recovery = False
    previous_month = None
    previous_day = None
    previous_scale = 1.0
    previous_position = 0.0
    previous_using_recovery = False
    recovery_pnl = np.asarray(recovery_pnl) if recovery_pnl is not None else None
    recovery_turnover = (
        np.asarray(recovery_turnover) if recovery_turnover is not None else None
    )
    recovery_actual = np.asarray(recovery_actual) if recovery_actual is not None else None
    day_keys = frame_5m.index.floor("D")
    for index, month in enumerate(month_keys):
        if month != previous_month:
            month_equity = 1.0
            locked = False
            guarded = False
            monthly_recovery = False
            previous_month = month
        day = day_keys[index]
        if day != previous_day:
            day_equity = 1.0
            daily_guarded = False
            previous_day = day
        risk_off = guarded or daily_guarded
        using_recovery = bool(monthly_recovery and recovery_pnl is not None)
        if risk_off:
            scale = 0.0
        elif locked:
            scale = float(lock_scale)
        elif monthly_recovery:
            scale = max(0.0, min(1.0, float(monthly_recovery_scale)))
        else:
            scale = 1.0
        active_pnl = recovery_pnl if using_recovery else pnl
        active_turnover = recovery_turnover if using_recovery else turnover
        active_actual = recovery_actual if using_recovery else actual
        scale_turnover = abs(scale - previous_scale) * abs(previous_position)
        cost_turnover = float(active_turnover[index]) * scale + scale_turnover
        if using_recovery != previous_using_recovery:
            # Conservatively close the prior path and open the alternate path.
            cost_turnover += (
                abs(previous_position) * previous_scale
                + abs(float(active_actual[index])) * scale
            )
        factor = 1.0 + float(leverage) * float(active_pnl[index]) * scale
        factor -= float(leverage) * cost_turnover * one_way_fee
        factor = max(factor, 1e-9)
        factors[index] = factor
        scales[index] = scale
        month_equity *= factor
        day_equity *= factor
        if daily_stop is not None and day_equity <= 1.0 + float(daily_stop):
            daily_guarded = True
        if (
            not monthly_recovery
            and monthly_stop is not None
            and month_equity <= 1.0 + float(monthly_stop)
        ):
            monthly_recovery = True
        elif monthly_recovery and month_equity >= 1.0 + float(recovery_exit):
            monthly_recovery = False
        if not locked and month_equity >= 1.0 + float(lock_trigger):
            locked = True
        elif locked and month_equity < 1.0 + float(lock_floor):
            guarded = True
        previous_scale = scale
        previous_position = float(active_actual[index])
        previous_using_recovery = using_recovery
    return factors, scales


def summarize_factors(frame_5m, factors, scales, *, start, end=None):
    mask = frame_5m.index >= pd.Timestamp(start, tz="UTC")
    if end:
        mask &= frame_5m.index <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    selected_factors = np.asarray(factors)[mask]
    selected_scales = np.asarray(scales)[mask]
    selected_index = frame_5m.index[mask]
    daily = pd.Series(selected_factors, index=selected_index).groupby(selected_index.floor("D")).prod()
    daily_months = daily.index.tz_localize(None).to_period("M")
    monthly = daily.groupby(daily_months).prod() - 1.0
    equity = np.cumprod(selected_factors)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    complete = monthly
    if end is None and len(complete):
        last_month = selected_index.max().tz_localize(None).to_period("M")
        complete = complete.loc[complete.index < last_month]
    return {
        "start": str(selected_index.min()) if len(selected_index) else start,
        "end": str(selected_index.max()) if len(selected_index) else (end or ""),
        "months": len(complete),
        "months_ge_5": int((complete >= 0.05).sum()),
        "months_ge_0": int((complete >= 0.0).sum()),
        "min_month_pct": round(float(complete.min()) * 100.0, 4) if len(complete) else 0.0,
        "avg_month_pct": round(float(complete.mean()) * 100.0, 4) if len(complete) else 0.0,
        "median_month_pct": round(float(complete.median()) * 100.0, 4) if len(complete) else 0.0,
        "total_return_pct": round(float(np.prod(selected_factors) - 1.0) * 100.0, 4),
        "max_drawdown_pct": round(float(np.min(drawdown)) * 100.0, 4),
        "avg_exposure_scale_pct": round(float(np.mean(selected_scales)) * 100.0, 4),
        "monthly": [
            {"month": str(month), "return_pct": round(float(value) * 100.0, 4)}
            for month, value in complete.items()
        ],
    }


def development_rank(summary):
    eligible = summary["min_month_pct"] >= -15.0 and summary["max_drawdown_pct"] >= -35.0
    return (
        int(eligible),
        summary["months_ge_5"],
        summary["months_ge_0"],
        summary["min_month_pct"],
        summary["max_drawdown_pct"],
        summary["avg_month_pct"],
    )


def build_report(frame_5m):
    rsi = completed_15m_rsi(frame_5m)
    candidates = []
    for distance, slope in REGIME_CONFIGS:
        labels = classify_completed_4h(
            frame_5m,
            distance_threshold=distance,
            slope_threshold=slope,
        )
        for range_mode, rsi_low, rsi_high in RANGE_CONFIGS:
            desired, regimes = build_regime_position(
                frame_5m,
                labels,
                rsi,
                range_mode=range_mode,
                rsi_low=rsi_low,
                rsi_high=rsi_high,
            )
            for stop_pct, target_pct, cooldown in RISK_CONFIGS:
                pnl, turnover, actual = monthly5_risk_cache.simulate_trade_risk_path(
                    frame_5m,
                    desired,
                    stop_pct=stop_pct,
                    target_pct=target_pct,
                    cooldown_bars=cooldown,
                )
                for leverage in LEVERAGES:
                    for lock_scale in POST_LOCK_SCALES:
                        for daily_stop in DAILY_STOPS:
                            factors, scales = apply_monthly_lock(
                                frame_5m,
                                pnl,
                                turnover,
                                actual,
                                leverage=leverage,
                                lock_scale=lock_scale,
                                daily_stop=daily_stop,
                            )
                            development = summarize_factors(
                                frame_5m,
                                factors,
                                scales,
                                start=REPORT_START,
                                end=DEVELOPMENT_END,
                            )
                            candidates.append(
                                {
                                    "name": (
                                        f"d{distance}_s{slope}_{range_mode}{rsi_low:g}-{rsi_high:g}"
                                        f"_stop{stop_pct}_target{target_pct}_lev{leverage}"
                                        f"_daystop{daily_stop}_lock{lock_scale}"
                                    ),
                                    "config": {
                                        "distance_threshold": distance,
                                        "slope_threshold": slope,
                                        "range_mode": range_mode,
                                        "rsi_low": rsi_low,
                                        "rsi_high": rsi_high,
                                        "stop_pct": stop_pct,
                                        "target_pct": target_pct,
                                        "cooldown_bars": cooldown,
                                        "leverage": leverage,
                                        "daily_stop": daily_stop,
                                        "monthly_stop": MONTHLY_STOP,
                                        "lock_trigger": LOCK_TRIGGER,
                                        "lock_floor": LOCK_FLOOR,
                                        "post_lock_scale": lock_scale,
                                    },
                                    "development": development,
                                    "_factors": factors,
                                    "_scales": scales,
                                    "_regimes": regimes,
                                }
                            )
    winner = max(candidates, key=lambda row: development_rank(row["development"]))
    holdout = summarize_factors(
        frame_5m,
        winner["_factors"],
        winner["_scales"],
        start=HOLDOUT_START,
    )
    full = summarize_factors(
        frame_5m,
        winner["_factors"],
        winner["_scales"],
        start=REPORT_START,
    )
    top = sorted(candidates, key=lambda row: development_rank(row["development"]), reverse=True)[:20]
    winner_eligible = development_rank(winner["development"])[0] == 1
    holdout_pass = (
        holdout["months_ge_5"] == holdout["months"]
        and holdout["min_month_pct"] >= -15.0
        and holdout["max_drawdown_pct"] >= -35.0
    )
    return {
        "schema_version": 1,
        "method": "completed_4h_intraday_switch_5m_pessimistic_stops_monthly_floor",
        "source": frame_5m.attrs.get("kline_source", "binance_history_um"),
        "development_period": f"{REPORT_START}..{DEVELOPMENT_END}",
        "holdout_start": HOLDOUT_START,
        "candidate_count": len(candidates),
        "selection_uses_holdout": False,
        "winner": {"name": winner["name"], "config": winner["config"]},
        "development": winner["development"],
        "holdout": holdout,
        "full": full,
        "top_development": [
            {"name": row["name"], "config": row["config"], "development": row["development"]}
            for row in top
        ],
        "deployment_ready": bool(winner_eligible and holdout_pass),
        "deployment_blockers": [
            *([] if winner_eligible else ["development_risk_gate_failed"]),
            *([] if holdout_pass else ["holdout_monthly5_or_risk_gate_failed"]),
            "candidate_matched_tick_execution_evidence_missing",
            "live_shadow_promotion_not_met",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-08-03")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    frame = monthly5_selector_cache.load_history(args.start, args.end)
    report = build_report(frame)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
