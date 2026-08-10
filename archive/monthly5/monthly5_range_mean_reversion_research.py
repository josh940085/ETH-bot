"""Research a completed-4h mean-reversion specialist while monthly5 is flat."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import monthly5_intraday_regime as regime
import monthly5_intramonth_recovery_research as account
import monthly5_regime_specialist_research as specialist
import monthly5_selector_cache
import monthly5_volatility_regime_research as volatility
import monthly5_volatility_walkforward as walkforward


RANGE_STRATEGY_ID = 10
ENTRY_ATR_VALUES = (0.3, 0.5, 0.7, 1.0)
EXIT_ATR_VALUES = (0.0, 0.2, 0.4)
STOP_PCT_VALUES = (0.01, 0.015, 0.02)
BASELINE_CONFIG = {
    "name": "baseline_4h_trend_only",
    "enabled": False,
    "entry_atr": None,
    "exit_atr": None,
    "stop_pct": None,
    "target_pct": None,
    "cooldown_bars": None,
}
ACCOUNT_CONFIG = {
    "name": "baseline_no_recovery",
    "mode": "none",
    "trigger": None,
    "scale": 0.0,
    "exit": 0.0,
}
DEFAULT_OUTPUT = Path(
    ".runtime/data/backtests/monthly5_search/range_mean_reversion_v1_2020_20260803.json"
)


def range_configs():
    rows = [BASELINE_CONFIG]
    for entry_atr in ENTRY_ATR_VALUES:
        for exit_atr in EXIT_ATR_VALUES:
            if exit_atr >= entry_atr:
                continue
            for stop_pct in STOP_PCT_VALUES:
                rows.append(
                    {
                        "name": (
                            f"range_entry{entry_atr}_exit{exit_atr}"
                            f"_stop{stop_pct}"
                        ),
                        "enabled": True,
                        "entry_atr": entry_atr,
                        "exit_atr": exit_atr,
                        "stop_pct": stop_pct,
                        "target_pct": 0.03,
                        "cooldown_bars": 12,
                    }
                )
    return tuple(rows)


def completed_4h_distance(frame_5m):
    frame_4h = volatility.aggregate_completed_4h(frame_5m)
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
    distance = (close - ma25) / atr14.replace(0.0, np.nan)
    distance.index = distance.index + pd.Timedelta(minutes=5)
    return distance


def selected_range_context(frame_5m, selected_profile_ids):
    selected = np.asarray(selected_profile_ids, dtype="int32")
    label_paths = []
    for config in walkforward.CANDIDATE_CONFIGS:
        labels = volatility.classify_completed_4h(frame_5m, config["label"])
        aligned = regime.align_completed_series(frame_5m.index, labels, "unknown").astype(str)
        label_paths.append(np.asarray(aligned))
    matrix = np.vstack(label_paths)
    return matrix[selected, np.arange(len(frame_5m))] == "range"


def build_range_desired(frame_5m, range_context, config):
    desired = np.zeros(len(frame_5m), dtype="float64")
    if not config["enabled"]:
        return desired
    distance = np.asarray(
        regime.align_completed_series(
            frame_5m.index, completed_4h_distance(frame_5m), np.nan
        ),
        dtype="float64",
    )
    active = 0.0
    entry_atr = float(config["entry_atr"])
    exit_atr = float(config["exit_atr"])
    for index, is_range in enumerate(np.asarray(range_context, dtype="bool")):
        value = distance[index]
        if not is_range or not np.isfinite(value):
            active = 0.0
        elif active > 0.0 and value >= -exit_atr:
            active = 0.0
        elif active < 0.0 and value <= exit_atr:
            active = 0.0
        elif active == 0.0:
            if value <= -entry_atr:
                active = 1.0
            elif value >= entry_atr:
                active = -1.0
        desired[index] = active
    return desired


def combine_paths(primary, primary_ids, range_desired):
    primary = np.asarray(primary, dtype="float64")
    fallback = np.asarray(range_desired, dtype="float64")
    ids = np.asarray(primary_ids, dtype="int32")
    use_range = (primary == 0.0) & (fallback != 0.0)
    return np.where(use_range, fallback, primary), np.where(use_range, RANGE_STRATEGY_ID, ids)


def evaluate_config(frame_5m, config):
    primary, primary_ids, _ = walkforward.build_primary_path(
        frame_5m, **account.PRIMARY_CONFIG
    )
    context = selected_range_context(frame_5m, primary_ids)
    range_desired = build_range_desired(frame_5m, context, config)
    desired, strategy_ids = combine_paths(primary, primary_ids, range_desired)
    risk_profiles = None
    if config["enabled"]:
        risk_profiles = {RANGE_STRATEGY_ID: config}
    return account.simulate_account_path(
        frame_5m,
        desired,
        strategy_ids,
        desired,
        ACCOUNT_CONFIG,
        risk_profiles=risk_profiles,
    )


def _without_monthly(summary):
    return {key: value for key, value in summary.items() if key != "monthly"}


def verify_prefix_stability(frame_5m, config, full_factors):
    checks = []
    for cutoff in ("2021-12-31", "2023-12-31", "2025-12-31"):
        cutoff_ts = pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        truncated = frame_5m.loc[frame_5m.index <= cutoff_ts]
        factors, _, _, _, _ = evaluate_config(truncated, config)
        stable = np.array_equal(np.asarray(full_factors)[: len(truncated)], np.asarray(factors))
        checks.append({"cutoff": cutoff, "bars": len(truncated), "stable": bool(stable)})
    return {
        "prefix_stable": all(row["stable"] for row in checks),
        "recursive_stable": all(row["stable"] for row in checks),
        "checks": checks,
    }


def build_report(frame_5m):
    primary, primary_ids, selections = walkforward.build_primary_path(
        frame_5m, **account.PRIMARY_CONFIG
    )
    context = selected_range_context(frame_5m, primary_ids)
    candidates = []
    for config in range_configs():
        range_desired = build_range_desired(frame_5m, context, config)
        desired, strategy_ids = combine_paths(primary, primary_ids, range_desired)
        risk_profiles = {RANGE_STRATEGY_ID: config} if config["enabled"] else None
        factors, scales, _, _, positions = account.simulate_account_path(
            frame_5m,
            desired,
            strategy_ids,
            desired,
            ACCOUNT_CONFIG,
            risk_profiles=risk_profiles,
        )
        candidates.append(
            {
                "name": config["name"],
                "config": config,
                "training": regime.summarize_factors(
                    frame_5m, factors, scales, start=regime.REPORT_START, end=specialist.TRAIN_END
                ),
                "validation": regime.summarize_factors(
                    frame_5m,
                    factors,
                    scales,
                    start=specialist.VALIDATION_START,
                    end=specialist.VALIDATION_END,
                ),
                "development": regime.summarize_factors(
                    frame_5m, factors, scales, start=regime.REPORT_START, end=regime.DEVELOPMENT_END
                ),
                "evaluation": regime.summarize_factors(
                    frame_5m, factors, scales, start=regime.HOLDOUT_START
                ),
                "range_signal_pct": round(float(np.mean(range_desired != 0.0)) * 100.0, 4),
                "actual_exposure_pct": round(float(np.mean(positions != 0.0)) * 100.0, 4),
                "_factors": factors,
                "_scales": scales,
            }
        )
    winner = max(candidates, key=specialist.selection_rank)
    evaluation = winner["evaluation"]
    full = regime.summarize_factors(
        frame_5m, winner["_factors"], winner["_scales"], start=regime.REPORT_START
    )
    prefix = verify_prefix_stability(frame_5m, winner["config"], winner["_factors"])
    target_pass = evaluation["months_ge_5"] == evaluation["months"]
    return {
        "schema_version": 1,
        "method": "completed_4h_range_mean_reversion_specialist",
        "source": frame_5m.attrs.get("kline_source", "binance_history_um"),
        "primary_config": account.PRIMARY_CONFIG,
        "primary_monthly_selections": selections,
        "selection_uses_evaluation_period": False,
        "evaluation_period_reused_during_research": True,
        "candidate_count": len(candidates),
        "winner": {
            "name": winner["name"],
            "config": winner["config"],
            "range_signal_pct": winner["range_signal_pct"],
            "actual_exposure_pct": winner["actual_exposure_pct"],
            "flat_time_pct": round(100.0 - winner["actual_exposure_pct"], 4),
        },
        "training": winner["training"],
        "validation": winner["validation"],
        "development": winner["development"],
        "evaluation": evaluation,
        "full": full,
        "top_selection": [
            {
                "name": row["name"],
                "config": row["config"],
                "training": _without_monthly(row["training"]),
                "validation": _without_monthly(row["validation"]),
                "evaluation_diagnostic_only": _without_monthly(row["evaluation"]),
                "range_signal_pct": row["range_signal_pct"],
                "actual_exposure_pct": row["actual_exposure_pct"],
            }
            for row in sorted(candidates, key=specialist.selection_rank, reverse=True)[:20]
        ],
        "bias_evidence": prefix,
        "deployment_ready": False,
        "deployment_blockers": [
            *([] if target_pass else ["evaluation_all_months_ge_5_failed"]),
            "evaluation_period_reused_during_research",
            *([] if prefix["recursive_stable"] else ["recursive_prefix_rebuild_failed"]),
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
