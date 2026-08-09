"""Research completed-4h efficiency gates for monthly5 trend entries."""

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


WINDOW_VALUES = (6, 12, 18, 24)
THRESHOLD_VALUES = (0.2, 0.3, 0.4)
BASELINE_CONFIG = {"name": "baseline_no_efficiency_gate", "window": None, "threshold": None}
ACCOUNT_CONFIG = {
    "name": "baseline_no_recovery",
    "mode": "none",
    "trigger": None,
    "scale": 0.0,
    "exit": 0.0,
}
DEFAULT_OUTPUT = Path(
    ".runtime/data/backtests/monthly5_search/trend_efficiency_v1_2020_20260803.json"
)


def efficiency_configs():
    return (
        BASELINE_CONFIG,
        *(
            {
                "name": f"efficiency_window{window}_min{threshold}",
                "window": window,
                "threshold": threshold,
            }
            for window in WINDOW_VALUES
            for threshold in THRESHOLD_VALUES
        ),
    )


def efficiency_ratio(close, window):
    values = pd.to_numeric(close, errors="coerce").astype("float64")
    net_change = values.diff(window).abs()
    path_length = values.diff().abs().rolling(window, min_periods=window).sum()
    return net_change / path_length.replace(0.0, np.nan)


def completed_4h_efficiency(frame_5m, window):
    frame_4h = volatility.aggregate_completed_4h(frame_5m)
    result = efficiency_ratio(frame_4h["close"], window)
    result.index = result.index + pd.Timedelta(minutes=5)
    return result


def build_entry_allowed(frame_5m, config, efficiency=None):
    if config["window"] is None:
        return np.ones(len(frame_5m), dtype="bool")
    if efficiency is None:
        efficiency = completed_4h_efficiency(frame_5m, config["window"])
    aligned = np.asarray(
        regime.align_completed_series(frame_5m.index, efficiency, np.nan),
        dtype="float64",
    )
    return np.isfinite(aligned) & (aligned >= float(config["threshold"]))


def evaluate_config(frame_5m, config):
    desired, strategy_ids, _ = walkforward.build_primary_path(
        frame_5m, **account.PRIMARY_CONFIG
    )
    return account.simulate_account_path(
        frame_5m,
        desired,
        strategy_ids,
        desired,
        ACCOUNT_CONFIG,
        entry_allowed=build_entry_allowed(frame_5m, config),
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
    desired, strategy_ids, selections = walkforward.build_primary_path(
        frame_5m, **account.PRIMARY_CONFIG
    )
    efficiency_cache = {
        window: completed_4h_efficiency(frame_5m, window) for window in WINDOW_VALUES
    }
    candidates = []
    for config in efficiency_configs():
        efficiency = efficiency_cache.get(config["window"])
        allowed = build_entry_allowed(frame_5m, config, efficiency)
        factors, scales, _, _, positions = account.simulate_account_path(
            frame_5m,
            desired,
            strategy_ids,
            desired,
            ACCOUNT_CONFIG,
            entry_allowed=allowed,
        )
        desired_mask = np.asarray(desired) != 0.0
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
                "entry_allowed_pct": round(
                    float(np.mean(allowed[desired_mask])) * 100.0 if desired_mask.any() else 0.0,
                    4,
                ),
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
        "method": "completed_4h_efficiency_entry_gate",
        "source": frame_5m.attrs.get("kline_source", "binance_history_um"),
        "primary_config": account.PRIMARY_CONFIG,
        "primary_monthly_selections": selections,
        "selection_uses_evaluation_period": False,
        "evaluation_period_reused_during_research": True,
        "candidate_count": len(candidates),
        "winner": {
            "name": winner["name"],
            "config": winner["config"],
            "entry_allowed_pct": winner["entry_allowed_pct"],
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
                "entry_allowed_pct": row["entry_allowed_pct"],
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
