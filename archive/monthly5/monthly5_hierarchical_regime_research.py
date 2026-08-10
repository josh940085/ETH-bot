"""Research completed-1h trend fallback while the monthly5 4h policy is flat."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import monthly5_intraday_regime as regime
import monthly5_intramonth_recovery_research as account
import monthly5_regime_hysteresis_research as hysteresis
import monthly5_regime_specialist_research as specialist
import monthly5_selector_cache
import monthly5_volatility_regime_research as volatility
import monthly5_volatility_walkforward as volatility_walkforward


ONE_HOUR_LABEL_CONFIGS = tuple(
    {
        "name": f"atr_d{distance}_s{slope}",
        "mode": "atr",
        "distance_atr": distance,
        "slope_atr": slope,
    }
    for distance in (0.3, 0.5, 0.7)
    for slope in (0.05, 0.10)
)
CONFIRMATION_BARS = (1, 2)
RANGE_GRACE_BARS = (0, 1, 3)
BASELINE_CONFIG = {
    "name": "baseline_4h_only",
    "enabled": False,
    "label": None,
    "confirmation_bars": 0,
    "range_grace_bars": 0,
}
ACCOUNT_CONFIG = {
    "name": "baseline_no_recovery",
    "mode": "none",
    "trigger": None,
    "scale": 0.0,
    "exit": 0.0,
}
DEFAULT_OUTPUT = Path(
    ".runtime/data/backtests/monthly5_search/hierarchical_1h_fallback_v1_2020_20260803.json"
)


def fallback_configs():
    rows = [BASELINE_CONFIG]
    for label in ONE_HOUR_LABEL_CONFIGS:
        for confirmation_bars in CONFIRMATION_BARS:
            for range_grace_bars in RANGE_GRACE_BARS:
                rows.append(
                    {
                        "name": (
                            f"{label['name']}|confirm{confirmation_bars}"
                            f"|grace{range_grace_bars}"
                        ),
                        "enabled": True,
                        "label": label,
                        "confirmation_bars": confirmation_bars,
                        "range_grace_bars": range_grace_bars,
                    }
                )
    return tuple(rows)


def classify_completed_1h(frame_5m, config):
    frame_1h = frame_5m.resample("1h", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "high", "low", "close"])
    labels = volatility.classify_4h_frame(frame_1h, config)
    labels.index = labels.index + pd.Timedelta(minutes=5)
    return labels


def combine_paths(primary_desired, primary_ids, fallback_desired):
    primary = np.asarray(primary_desired, dtype="float64")
    fallback = np.asarray(fallback_desired, dtype="float64")
    primary_ids = np.asarray(primary_ids, dtype="int32")
    use_fallback = primary == 0.0
    desired = np.where(use_fallback, fallback, primary)
    strategy_ids = np.where(use_fallback & (fallback != 0.0), 10, primary_ids)
    return desired, strategy_ids


def build_fallback(frame_5m, config):
    if not config["enabled"]:
        return np.zeros(len(frame_5m), dtype="float64")
    labels = classify_completed_1h(frame_5m, config["label"])
    desired, _ = hysteresis.build_hysteresis_position(
        frame_5m,
        labels,
        confirmation_bars=config["confirmation_bars"],
        range_grace_bars=config["range_grace_bars"],
    )
    return np.asarray(desired, dtype="float64")


def evaluate_config(frame_5m, config):
    primary, primary_ids, _ = volatility_walkforward.build_primary_path(
        frame_5m,
        **account.PRIMARY_CONFIG,
    )
    desired, strategy_ids = combine_paths(
        primary,
        primary_ids,
        build_fallback(frame_5m, config),
    )
    return account.simulate_account_path(
        frame_5m,
        desired,
        strategy_ids,
        desired,
        ACCOUNT_CONFIG,
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
    primary, primary_ids, primary_selections = volatility_walkforward.build_primary_path(
        frame_5m,
        **account.PRIMARY_CONFIG,
    )
    candidates = []
    for config in fallback_configs():
        fallback = build_fallback(frame_5m, config)
        desired, strategy_ids = combine_paths(primary, primary_ids, fallback)
        factors, scales, _, _, positions = account.simulate_account_path(
            frame_5m,
            desired,
            strategy_ids,
            desired,
            ACCOUNT_CONFIG,
        )
        candidates.append(
            {
                "name": config["name"],
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
                "evaluation": regime.summarize_factors(
                    frame_5m,
                    factors,
                    scales,
                    start=regime.HOLDOUT_START,
                ),
                "fallback_signal_pct": round(float(np.mean(fallback != 0.0)) * 100.0, 4),
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
        "method": "completed_1h_trend_fallback_when_4h_flat",
        "source": frame_5m.attrs.get("kline_source", "binance_history_um"),
        "primary_config": account.PRIMARY_CONFIG,
        "primary_monthly_selections": primary_selections,
        "selection_uses_evaluation_period": False,
        "evaluation_period_reused_during_research": True,
        "candidate_count": len(candidates),
        "winner": {
            "name": winner["name"],
            "config": winner["config"],
            "fallback_signal_pct": winner["fallback_signal_pct"],
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
                "fallback_signal_pct": row["fallback_signal_pct"],
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
