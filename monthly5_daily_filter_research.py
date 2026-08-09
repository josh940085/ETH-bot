"""Evaluate completed-daily confirmation for the causal monthly5 4h policy."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import monthly5_intraday_regime as regime
import monthly5_recovery_research as recovery
import monthly5_risk_cache
import monthly5_selector_cache


FILTER_CONFIGS = (
    {"name": "none", "window": 0, "mode": "none"},
    *(
        {"name": f"daily_ma{window}_{mode}", "window": window, "mode": mode}
        for window in (10, 20, 50)
        for mode in ("strict", "no_conflict")
    ),
)
DEFAULT_OUTPUT = Path(
    ".runtime/data/backtests/monthly5_search/daily_filter_4h_regime_v1_2020_20260803.json"
)


def build_components(frame_5m, filter_config):
    config = recovery.BASE_CONFIG
    labels = regime.classify_completed_4h(
        frame_5m,
        distance_threshold=config["distance_threshold"],
        slope_threshold=config["slope_threshold"],
    )
    daily = (
        regime.completed_daily_trend(frame_5m, window=filter_config["window"])
        if int(filter_config["window"]) > 0
        else None
    )
    desired, _ = regime.build_regime_position(
        frame_5m,
        labels,
        regime.completed_15m_rsi(frame_5m),
        range_mode=config["range_mode"],
        rsi_low=config["rsi_low"],
        rsi_high=config["rsi_high"],
        daily_trend=daily,
        daily_alignment_mode=filter_config["mode"],
    )
    return monthly5_risk_cache.simulate_trade_risk_path(
        frame_5m,
        desired,
        stop_pct=config["stop_pct"],
        target_pct=config["target_pct"],
        cooldown_bars=config["cooldown_bars"],
    )


def apply_components(frame_5m, components):
    config = recovery.BASE_CONFIG
    return regime.apply_monthly_lock(
        frame_5m,
        *components,
        leverage=config["leverage"],
        lock_scale=config["post_lock_scale"],
        lock_trigger=config["lock_trigger"],
        lock_floor=config["lock_floor"],
        daily_stop=config["daily_stop"],
        monthly_stop=-0.08,
        monthly_recovery_scale=0.0,
    )


def select_development_winner(candidates):
    return max(candidates, key=lambda row: regime.development_rank(row["development"]))


def verify_prefix_stability(frame_5m, filter_config, full_factors):
    checks = []
    for cutoff in ("2021-12-31", "2023-12-31", "2025-12-31"):
        cutoff_ts = pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        truncated = frame_5m.loc[frame_5m.index <= cutoff_ts]
        factors, _ = apply_components(truncated, build_components(truncated, filter_config))
        stable = np.array_equal(np.asarray(full_factors)[: len(truncated)], np.asarray(factors))
        checks.append({"cutoff": cutoff, "bars": len(truncated), "stable": bool(stable)})
    return {
        "prefix_stable": all(row["stable"] for row in checks),
        "recursive_stable": all(row["stable"] for row in checks),
        "checks": checks,
    }


def _without_monthly(summary):
    return {key: value for key, value in summary.items() if key != "monthly"}


def build_report(frame_5m):
    candidates = []
    for config in FILTER_CONFIGS:
        factors, scales = apply_components(frame_5m, build_components(frame_5m, config))
        candidates.append(
            {
                "name": config["name"],
                "config": config,
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
    winner = select_development_winner(candidates)
    holdout = winner["holdout_diagnostic"]
    full = regime.summarize_factors(
        frame_5m,
        winner["_factors"],
        winner["_scales"],
        start=regime.REPORT_START,
    )
    prefix = verify_prefix_stability(frame_5m, winner["config"], winner["_factors"])
    holdout_pass = (
        holdout["months_ge_5"] == holdout["months"]
        and holdout["min_month_pct"] >= -15.0
        and holdout["max_drawdown_pct"] >= -35.0
        and prefix["recursive_stable"]
    )
    return {
        "schema_version": 1,
        "method": "completed_daily_ma_confirmation_for_4h_regime",
        "base_config": recovery.BASE_CONFIG,
        "selection_uses_holdout": False,
        "candidate_count": len(candidates),
        "winner": {"name": winner["name"], "config": winner["config"]},
        "development": winner["development"],
        "holdout": holdout,
        "full": full,
        "all_candidates": [
            {
                "name": row["name"],
                "config": row["config"],
                "development": _without_monthly(row["development"]),
                "holdout_diagnostic_only": _without_monthly(row["holdout_diagnostic"]),
            }
            for row in candidates
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
