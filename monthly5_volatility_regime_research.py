"""Research volatility-adaptive completed-4h regimes for monthly5."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ta.trend import ADXIndicator

import monthly5_intraday_regime as regime
import monthly5_recovery_research as recovery
import monthly5_regime_hysteresis_research as hysteresis
import monthly5_regime_specialist_research as specialist
import monthly5_risk_cache
import monthly5_selector_cache


LABEL_CONFIGS = (
    {"name": "fixed", "mode": "fixed", "distance": 0.006, "slope": 0.001},
    *(
        {
            "name": f"atr_d{distance}_s{slope}",
            "mode": "atr",
            "distance_atr": distance,
            "slope_atr": slope,
        }
        for distance in (0.3, 0.5, 0.7)
        for slope in (0.05, 0.10)
    ),
    *(
        {
            "name": f"adx{threshold}_s{slope}",
            "mode": "adx",
            "adx_threshold": threshold,
            "slope_atr": slope,
        }
        for threshold in (15.0, 20.0, 25.0)
        for slope in (0.0, 0.10)
    ),
)
HYSTERESIS_CONFIGS = (
    {"confirmation_bars": 4, "range_grace_bars": 3},
    {"confirmation_bars": 2, "range_grace_bars": 2},
    {"confirmation_bars": 1, "range_grace_bars": 6},
)
RISK_PROFILE = {
    "stop_pct": 0.03,
    "target_pct": 0.06,
    "cooldown_bars": 12,
    "leverage": 1,
}
DEFAULT_OUTPUT = Path(
    ".runtime/data/backtests/monthly5_search/volatility_regime_v1_2020_20260803.json"
)


def aggregate_completed_4h(frame_5m):
    return frame_5m.resample("4h", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "high", "low", "close"])


def classify_4h_frame(frame_4h, config):
    high = pd.to_numeric(frame_4h["high"], errors="coerce").astype("float64")
    low = pd.to_numeric(frame_4h["low"], errors="coerce").astype("float64")
    close = pd.to_numeric(frame_4h["close"], errors="coerce").astype("float64")
    ma25 = close.rolling(25, min_periods=25).mean()
    previous_close = close.shift(1)
    true_range = pd.concat(
        ((high - low), (high - previous_close).abs(), (low - previous_close).abs()),
        axis=1,
    ).max(axis=1)
    atr14 = true_range.rolling(14, min_periods=14).mean()
    distance_atr = (close - ma25) / atr14.replace(0.0, np.nan)
    slope_atr = (ma25 - ma25.shift(4)) / atr14.replace(0.0, np.nan)
    labels = pd.Series("range", index=frame_4h.index, dtype="object")
    mode = str(config["mode"])
    if mode == "fixed":
        distance = close / ma25 - 1.0
        slope = ma25 / ma25.shift(4) - 1.0
        up = (distance > float(config["distance"])) & (slope > float(config["slope"]))
        down = (distance < -float(config["distance"])) & (slope < -float(config["slope"]))
        warm = ma25.notna() & slope.notna()
    elif mode == "atr":
        up = (distance_atr > float(config["distance_atr"])) & (
            slope_atr > float(config["slope_atr"])
        )
        down = (distance_atr < -float(config["distance_atr"])) & (
            slope_atr < -float(config["slope_atr"])
        )
        warm = distance_atr.notna() & slope_atr.notna()
    elif mode == "adx":
        adx_indicator = ADXIndicator(high, low, close, window=14, fillna=False)
        adx = adx_indicator.adx()
        positive = adx_indicator.adx_pos()
        negative = adx_indicator.adx_neg()
        threshold = float(config["adx_threshold"])
        minimum_slope = float(config["slope_atr"])
        up = (
            (adx >= threshold)
            & (positive > negative)
            & (close > ma25)
            & (slope_atr > minimum_slope)
        )
        down = (
            (adx >= threshold)
            & (negative > positive)
            & (close < ma25)
            & (slope_atr < -minimum_slope)
        )
        warm = ma25.notna() & slope_atr.notna() & adx.notna()
    else:
        raise ValueError(f"unknown volatility regime mode: {mode}")
    labels.loc[up] = "up"
    labels.loc[down] = "down"
    labels.loc[~warm] = "unknown"
    return labels


def classify_completed_4h(frame_5m, config):
    labels = classify_4h_frame(aggregate_completed_4h(frame_5m), config)
    labels.index = labels.index + pd.Timedelta(minutes=5)
    return labels


def apply_components(frame_5m, components):
    return regime.apply_monthly_lock(
        frame_5m,
        *components,
        leverage=RISK_PROFILE["leverage"],
        lock_scale=recovery.BASE_CONFIG["post_lock_scale"],
        lock_trigger=recovery.BASE_CONFIG["lock_trigger"],
        lock_floor=recovery.BASE_CONFIG["lock_floor"],
        daily_stop=recovery.BASE_CONFIG["daily_stop"],
        monthly_stop=-0.08,
        monthly_recovery_scale=0.0,
    )


def evaluate_config(frame_5m, config):
    labels = classify_completed_4h(frame_5m, config["label"])
    desired, _ = hysteresis.build_hysteresis_position(
        frame_5m,
        labels,
        confirmation_bars=config["confirmation_bars"],
        range_grace_bars=config["range_grace_bars"],
    )
    components = monthly5_risk_cache.simulate_trade_risk_path(
        frame_5m,
        desired,
        stop_pct=RISK_PROFILE["stop_pct"],
        target_pct=RISK_PROFILE["target_pct"],
        cooldown_bars=RISK_PROFILE["cooldown_bars"],
    )
    return apply_components(frame_5m, components)


def _without_monthly(summary):
    return {key: value for key, value in summary.items() if key != "monthly"}


def verify_prefix_stability(frame_5m, config, full_factors):
    checks = []
    for cutoff in ("2021-12-31", "2023-12-31", "2025-12-31"):
        cutoff_ts = pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        truncated = frame_5m.loc[frame_5m.index <= cutoff_ts]
        factors, _ = evaluate_config(truncated, config)
        stable = np.array_equal(np.asarray(full_factors)[: len(truncated)], np.asarray(factors))
        checks.append({"cutoff": cutoff, "bars": len(truncated), "stable": bool(stable)})
    return {
        "prefix_stable": all(row["stable"] for row in checks),
        "recursive_stable": all(row["stable"] for row in checks),
        "checks": checks,
    }


def build_report(frame_5m):
    candidates = []
    label_cache = {
        config["name"]: classify_completed_4h(frame_5m, config) for config in LABEL_CONFIGS
    }
    for label_config in LABEL_CONFIGS:
        for hysteresis_config in HYSTERESIS_CONFIGS:
            desired, _ = hysteresis.build_hysteresis_position(
                frame_5m,
                label_cache[label_config["name"]],
                **hysteresis_config,
            )
            components = monthly5_risk_cache.simulate_trade_risk_path(
                frame_5m,
                desired,
                stop_pct=RISK_PROFILE["stop_pct"],
                target_pct=RISK_PROFILE["target_pct"],
                cooldown_bars=RISK_PROFILE["cooldown_bars"],
            )
            factors, scales = apply_components(frame_5m, components)
            config = {"label": label_config, **hysteresis_config}
            candidates.append(
                {
                    "name": (
                        f"{label_config['name']}|confirm{hysteresis_config['confirmation_bars']}"
                        f"|grace{hysteresis_config['range_grace_bars']}"
                    ),
                    "config": config,
                    "training": regime.summarize_factors(
                        frame_5m,
                        factors,
                        scales,
                        start=regime.REPORT_START,
                        end=specialist.TRAIN_END,
                    ),
                    "validation": regime.summarize_factors(
                        frame_5m,
                        factors,
                        scales,
                        start=specialist.VALIDATION_START,
                        end=specialist.VALIDATION_END,
                    ),
                    "development": regime.summarize_factors(
                        frame_5m,
                        factors,
                        scales,
                        start=regime.REPORT_START,
                        end=regime.DEVELOPMENT_END,
                    ),
                    "holdout_diagnostic": regime.summarize_factors(
                        frame_5m,
                        factors,
                        scales,
                        start=regime.HOLDOUT_START,
                    ),
                    "_factors": factors,
                    "_scales": scales,
                }
            )
    winner = max(candidates, key=specialist.selection_rank)
    holdout = winner["holdout_diagnostic"]
    full = regime.summarize_factors(
        frame_5m, winner["_factors"], winner["_scales"], start=regime.REPORT_START
    )
    prefix = verify_prefix_stability(frame_5m, winner["config"], winner["_factors"])
    winning_labels = label_cache[winner["config"]["label"]["name"]]
    label_counts = winning_labels.value_counts().to_dict()
    holdout_pass = (
        holdout["months_ge_5"] == holdout["months"]
        and holdout["min_month_pct"] >= -15.0
        and holdout["max_drawdown_pct"] >= -35.0
        and prefix["recursive_stable"]
    )
    return {
        "schema_version": 1,
        "method": "completed_4h_atr_adx_regime_nested_validation",
        "confirmation_timebase": hysteresis.CONFIRMATION_TIMEBASE,
        "source": frame_5m.attrs.get("kline_source", "binance_history_um"),
        "selection_uses_holdout": False,
        "risk_profile": RISK_PROFILE,
        "candidate_count": len(candidates),
        "winner": {"name": winner["name"], "config": winner["config"]},
        "winner_label_counts": {key: int(value) for key, value in label_counts.items()},
        "training": winner["training"],
        "validation": winner["validation"],
        "development": winner["development"],
        "holdout": holdout,
        "full": full,
        "top_selection": [
            {
                "name": row["name"],
                "config": row["config"],
                "training": _without_monthly(row["training"]),
                "validation": _without_monthly(row["validation"]),
                "holdout_diagnostic_only": _without_monthly(row["holdout_diagnostic"]),
            }
            for row in sorted(candidates, key=specialist.selection_rank, reverse=True)[:20]
        ],
        "bias_evidence": prefix,
        "deployment_ready": bool(holdout_pass),
        "deployment_blockers": [
            *([] if holdout_pass else ["holdout_all_months_ge_5_failed"]),
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
