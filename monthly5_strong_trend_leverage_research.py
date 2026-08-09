"""Research selective leverage during strong completed-4h monthly5 trends."""

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


STRONG_ID_OFFSET = 20
DISTANCE_ATR_VALUES = (0.7, 1.0, 1.5, 2.0)
SLOPE_ATR_VALUES = (0.1, 0.2, 0.3)
LEVERAGE_VALUES = (1.5, 2.0)
BASELINE_CONFIG = {
    "name": "baseline_1x",
    "enabled": False,
    "distance_atr": None,
    "slope_atr": None,
    "strong_leverage": 1.0,
}
ACCOUNT_CONFIG = {
    "name": "baseline_no_recovery",
    "mode": "none",
    "trigger": None,
    "scale": 0.0,
    "exit": 0.0,
}
DEFAULT_OUTPUT = Path(
    ".runtime/data/backtests/monthly5_search/strong_trend_leverage_v1_2020_20260803.json"
)


def leverage_configs():
    rows = [BASELINE_CONFIG]
    for distance_atr in DISTANCE_ATR_VALUES:
        for slope_atr in SLOPE_ATR_VALUES:
            for strong_leverage in LEVERAGE_VALUES:
                rows.append(
                    {
                        "name": (
                            f"strong_d{distance_atr}_s{slope_atr}"
                            f"_lev{strong_leverage}"
                        ),
                        "enabled": True,
                        "distance_atr": distance_atr,
                        "slope_atr": slope_atr,
                        "strong_leverage": strong_leverage,
                    }
                )
    return tuple(rows)


def completed_4h_strength(frame_5m):
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
    atr14 = true_range.rolling(14, min_periods=14).mean().replace(0.0, np.nan)
    strength = pd.DataFrame(
        {
            "distance_atr": (close - ma25) / atr14,
            "slope_atr": (ma25 - ma25.shift(4)) / atr14,
        },
        index=frame_4h.index,
    )
    strength.index = strength.index + pd.Timedelta(minutes=5)
    return strength


def build_strategy_ids(frame_5m, desired, primary_ids, config, strength=None):
    ids = np.asarray(primary_ids, dtype="int32").copy()
    if not config["enabled"]:
        return ids, np.zeros(len(frame_5m), dtype="bool")
    if strength is None:
        strength = completed_4h_strength(frame_5m)
    distance = np.asarray(
        regime.align_completed_series(frame_5m.index, strength["distance_atr"], np.nan),
        dtype="float64",
    )
    slope = np.asarray(
        regime.align_completed_series(frame_5m.index, strength["slope_atr"], np.nan),
        dtype="float64",
    )
    wanted = np.asarray(desired, dtype="float64")
    threshold_distance = float(config["distance_atr"])
    threshold_slope = float(config["slope_atr"])
    strong = (
        ((wanted > 0.0) & (distance >= threshold_distance) & (slope >= threshold_slope))
        | ((wanted < 0.0) & (distance <= -threshold_distance) & (slope <= -threshold_slope))
    )
    ids[strong] += STRONG_ID_OFFSET
    return ids, strong


def risk_profiles(config):
    if not config["enabled"]:
        return None
    profile = {
        "stop_pct": volatility.RISK_PROFILE["stop_pct"],
        "target_pct": volatility.RISK_PROFILE["target_pct"],
        "cooldown_bars": volatility.RISK_PROFILE["cooldown_bars"],
        "leverage": config["strong_leverage"],
    }
    return {
        STRONG_ID_OFFSET + index: profile
        for index in range(len(walkforward.CANDIDATE_CONFIGS))
    }


def evaluate_config(frame_5m, config):
    desired, primary_ids, _ = walkforward.build_primary_path(
        frame_5m, **account.PRIMARY_CONFIG
    )
    strategy_ids, _ = build_strategy_ids(frame_5m, desired, primary_ids, config)
    return account.simulate_account_path(
        frame_5m,
        desired,
        strategy_ids,
        desired,
        ACCOUNT_CONFIG,
        risk_profiles=risk_profiles(config),
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
    desired, primary_ids, selections = walkforward.build_primary_path(
        frame_5m, **account.PRIMARY_CONFIG
    )
    strength = completed_4h_strength(frame_5m)
    candidates = []
    for config in leverage_configs():
        strategy_ids, strong = build_strategy_ids(
            frame_5m, desired, primary_ids, config, strength
        )
        factors, scales, _, _, positions = account.simulate_account_path(
            frame_5m,
            desired,
            strategy_ids,
            desired,
            ACCOUNT_CONFIG,
            risk_profiles=risk_profiles(config),
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
                "strong_signal_pct": round(float(np.mean(strong)) * 100.0, 4),
                "actual_exposure_pct": round(float(np.mean(positions != 0.0)) * 100.0, 4),
                "average_absolute_leverage": round(float(np.mean(np.abs(positions))), 4),
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
        "method": "selective_strong_completed_4h_trend_leverage",
        "source": frame_5m.attrs.get("kline_source", "binance_history_um"),
        "primary_config": account.PRIMARY_CONFIG,
        "primary_monthly_selections": selections,
        "selection_uses_evaluation_period": False,
        "evaluation_period_reused_during_research": True,
        "candidate_count": len(candidates),
        "winner": {
            "name": winner["name"],
            "config": winner["config"],
            "strong_signal_pct": winner["strong_signal_pct"],
            "actual_exposure_pct": winner["actual_exposure_pct"],
            "flat_time_pct": round(100.0 - winner["actual_exposure_pct"], 4),
            "average_absolute_leverage": winner["average_absolute_leverage"],
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
                "strong_signal_pct": row["strong_signal_pct"],
                "actual_exposure_pct": row["actual_exposure_pct"],
                "average_absolute_leverage": row["average_absolute_leverage"],
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
