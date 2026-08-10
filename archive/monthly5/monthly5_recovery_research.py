"""Research causal low-exposure recovery after a monthly5 drawdown trigger."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import monthly5_intraday_regime as regime
import monthly5_risk_cache
import monthly5_selector_cache


BASE_CONFIG = {
    "distance_threshold": 0.006,
    "slope_threshold": 0.001,
    "range_mode": "flat",
    "rsi_low": 0.0,
    "rsi_high": 0.0,
    "stop_pct": 0.02,
    "target_pct": 0.04,
    "cooldown_bars": 12,
    "leverage": 1,
    "daily_stop": -0.02,
    "lock_trigger": 0.055,
    "lock_floor": 0.051,
    "post_lock_scale": 0.15,
}
RECOVERY_TRIGGERS = (-0.06, -0.08, -0.10)
RECOVERY_SCALES = (0.15, 0.25, 0.50)
RECOVERY_EXITS = (-0.03, 0.0)
ALTERNATE_MODES = ("inverse", "long_only", "short_only", "buy_hold")
DEFAULT_OUTPUT = Path(
    ".runtime/data/backtests/monthly5_search/recovery_4h_regime_v1_2020_20260803.json"
)


def recovery_configs():
    configs = [
        {
            "name": "hard_stop_m8",
            "monthly_stop": -0.08,
            "monthly_recovery_scale": 0.0,
            "recovery_exit": -0.03,
            "recovery_mode": "same",
        }
    ]
    for trigger in RECOVERY_TRIGGERS:
        for scale in RECOVERY_SCALES:
            for exit_level in RECOVERY_EXITS:
                configs.append(
                    {
                        "name": f"recover_trig{trigger}_scale{scale}_exit{exit_level}",
                        "monthly_stop": trigger,
                        "monthly_recovery_scale": scale,
                        "recovery_exit": exit_level,
                        "recovery_mode": "same",
                    }
                )
    for mode in ALTERNATE_MODES:
        for scale in (0.25, 0.50):
            for exit_level in RECOVERY_EXITS:
                configs.append(
                    {
                        "name": f"recover_{mode}_trig-0.08_scale{scale}_exit{exit_level}",
                        "monthly_stop": -0.08,
                        "monthly_recovery_scale": scale,
                        "recovery_exit": exit_level,
                        "recovery_mode": mode,
                    }
                )
    return configs


def build_desired(frame_5m, config=BASE_CONFIG):
    labels = regime.classify_completed_4h(
        frame_5m,
        distance_threshold=config["distance_threshold"],
        slope_threshold=config["slope_threshold"],
    )
    desired, _ = regime.build_regime_position(
        frame_5m,
        labels,
        regime.completed_15m_rsi(frame_5m),
        range_mode=config["range_mode"],
        rsi_low=config["rsi_low"],
        rsi_high=config["rsi_high"],
    )
    return desired


def simulate_desired(frame_5m, desired, config=BASE_CONFIG):
    return monthly5_risk_cache.simulate_trade_risk_path(
        frame_5m,
        desired,
        stop_pct=config["stop_pct"],
        target_pct=config["target_pct"],
        cooldown_bars=config["cooldown_bars"],
    )


def build_components(frame_5m, config=BASE_CONFIG):
    return simulate_desired(frame_5m, build_desired(frame_5m, config), config)


def build_component_map(frame_5m, config=BASE_CONFIG):
    desired = build_desired(frame_5m, config)
    alternates = {
        "same": simulate_desired(frame_5m, desired, config),
        "inverse": simulate_desired(frame_5m, desired * -1.0, config),
        "long_only": simulate_desired(frame_5m, desired.clip(lower=0.0), config),
        "short_only": simulate_desired(frame_5m, desired.clip(upper=0.0), config),
    }
    buy_hold = pd.Series(1.0, index=frame_5m.index)
    buy_hold.iloc[0] = 0.0
    alternates["buy_hold"] = simulate_desired(frame_5m, buy_hold, config)
    return alternates


def apply_config(frame_5m, component_map, recovery_config):
    pnl, turnover, actual = component_map["same"]
    recovery_mode = str(recovery_config.get("recovery_mode") or "same")
    alternate = component_map.get(recovery_mode)
    use_alternate = recovery_mode != "same" and alternate is not None
    return regime.apply_monthly_lock(
        frame_5m,
        pnl,
        turnover,
        actual,
        leverage=BASE_CONFIG["leverage"],
        lock_scale=BASE_CONFIG["post_lock_scale"],
        lock_trigger=BASE_CONFIG["lock_trigger"],
        lock_floor=BASE_CONFIG["lock_floor"],
        daily_stop=BASE_CONFIG["daily_stop"],
        monthly_stop=recovery_config["monthly_stop"],
        monthly_recovery_scale=recovery_config["monthly_recovery_scale"],
        recovery_exit=recovery_config["recovery_exit"],
        recovery_pnl=alternate[0] if use_alternate else None,
        recovery_turnover=alternate[1] if use_alternate else None,
        recovery_actual=alternate[2] if use_alternate else None,
    )


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


def select_development_winner(candidates):
    return max(candidates, key=lambda row: development_rank(row["development"]))


def verify_prefix_stability(frame_5m, recovery_config, full_factors):
    checks = []
    for cutoff in ("2021-12-31", "2023-12-31", "2025-12-31"):
        cutoff_ts = pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        truncated = frame_5m.loc[frame_5m.index <= cutoff_ts]
        factors, _ = apply_config(truncated, build_component_map(truncated), recovery_config)
        count = len(truncated)
        stable = np.array_equal(np.asarray(full_factors)[:count], np.asarray(factors))
        checks.append({"cutoff": cutoff, "bars": count, "stable": bool(stable)})
    return {
        "prefix_stable": all(row["stable"] for row in checks),
        "recursive_stable": all(row["stable"] for row in checks),
        "checks": checks,
    }


def _without_monthly(summary):
    return {key: value for key, value in summary.items() if key != "monthly"}


def _monthly_map(summary):
    return {str(row["month"]): float(row["return_pct"]) for row in summary["monthly"]}


def build_report(frame_5m):
    components = build_component_map(frame_5m)
    candidates = []
    for config in recovery_configs():
        factors, scales = apply_config(frame_5m, components, config)
        development = regime.summarize_factors(
            frame_5m,
            factors,
            scales,
            start=regime.REPORT_START,
            end=regime.DEVELOPMENT_END,
        )
        holdout_diagnostic = regime.summarize_factors(
            frame_5m,
            factors,
            scales,
            start=regime.HOLDOUT_START,
        )
        candidates.append(
            {
                "name": config["name"],
                "config": config,
                "development": development,
                "holdout_diagnostic": holdout_diagnostic,
                "_factors": factors,
                "_scales": scales,
            }
        )
    winner = select_development_winner(candidates)
    baseline = next(row for row in candidates if row["name"] == "hard_stop_m8")
    for row in (winner, baseline):
        row["holdout"] = regime.summarize_factors(
            frame_5m,
            row["_factors"],
            row["_scales"],
            start=regime.HOLDOUT_START,
        )
        row["full"] = regime.summarize_factors(
            frame_5m,
            row["_factors"],
            row["_scales"],
            start=regime.REPORT_START,
        )
    prefix = verify_prefix_stability(frame_5m, winner["config"], winner["_factors"])
    winner_holdout = _monthly_map(winner["holdout"])
    baseline_holdout = _monthly_map(baseline["holdout"])
    comparison = [
        {
            "month": month,
            "baseline_return_pct": baseline_holdout.get(month, 0.0),
            "recovery_return_pct": value,
            "delta_pct": round(value - baseline_holdout.get(month, 0.0), 4),
            "recovery_ge_5": value >= 5.0,
        }
        for month, value in winner_holdout.items()
        if baseline_holdout.get(month, 0.0) < 5.0 or value < 5.0
    ]
    holdout = winner["holdout"]
    holdout_pass = (
        holdout["months_ge_5"] == holdout["months"]
        and holdout["min_month_pct"] >= -15.0
        and holdout["max_drawdown_pct"] >= -35.0
        and prefix["recursive_stable"]
    )
    ranked = sorted(candidates, key=lambda row: development_rank(row["development"]), reverse=True)
    holdout_oracle = sorted(
        candidates,
        key=lambda row: (
            row["holdout_diagnostic"]["months_ge_5"],
            row["holdout_diagnostic"]["months_ge_0"],
            row["holdout_diagnostic"]["min_month_pct"],
            row["holdout_diagnostic"]["max_drawdown_pct"],
        ),
        reverse=True,
    )
    return {
        "schema_version": 1,
        "method": "development_selected_same_signal_monthly_recovery_5m_causal",
        "source": frame_5m.attrs.get("kline_source", "binance_history_um"),
        "base_config": BASE_CONFIG,
        "development_period": f"{regime.REPORT_START}..{regime.DEVELOPMENT_END}",
        "holdout_start": regime.HOLDOUT_START,
        "selection_uses_holdout": False,
        "candidate_count": len(candidates),
        "winner": {"name": winner["name"], "config": winner["config"]},
        "development": winner["development"],
        "holdout": winner["holdout"],
        "full": winner["full"],
        "baseline": {
            "name": baseline["name"],
            "development": baseline["development"],
            "holdout": baseline["holdout"],
            "full": baseline["full"],
        },
        "holdout_failed_month_comparison": comparison,
        "top_development": [
            {
                "name": row["name"],
                "config": row["config"],
                "development": _without_monthly(row["development"]),
            }
            for row in ranked[:10]
        ],
        "holdout_oracle_diagnostic_only": {
            "selectable": False,
            "reason": "holdout cannot select or tune a deployable recovery policy",
            "top": [
                {
                    "name": row["name"],
                    "config": row["config"],
                    "holdout": _without_monthly(row["holdout_diagnostic"]),
                }
                for row in holdout_oracle[:10]
            ],
        },
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
