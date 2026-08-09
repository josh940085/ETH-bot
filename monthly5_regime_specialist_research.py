"""Research specialist entry policies for completed 4h market regimes."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import monthly5_intraday_regime as regime
import monthly5_recovery_research as recovery
import monthly5_risk_cache
import monthly5_selector_cache


TRAIN_END = "2022-12-31"
VALIDATION_START = "2023-01-01"
VALIDATION_END = "2023-12-31"
TREND_MODES = ("always", "momentum", "pullback")
RANGE_MODES = ("flat", "rsi30", "rsi35")
RISK_CONFIGS = (
    (0.010, 0.020, 12),
    (0.015, 0.030, 12),
    (0.020, 0.040, 12),
    (0.030, 0.060, 12),
)
LEVERAGES = (1, 2, 3, 4, 5)
DEFAULT_OUTPUT = Path(
    ".runtime/data/backtests/monthly5_search/regime_specialists_v1_2020_20260803.json"
)


def policy_configs():
    return tuple(
        {
            "name": f"trend_{trend_mode}_range_{range_mode}",
            "trend_mode": trend_mode,
            "range_mode": range_mode,
        }
        for trend_mode in TREND_MODES
        for range_mode in RANGE_MODES
    )


def build_specialist_position(frame_5m, labels, rsi, *, trend_mode, range_mode):
    regimes = regime.align_completed_series(frame_5m.index, labels, "unknown").astype(str)
    rsi_values = regime.align_completed_series(frame_5m.index, rsi, 50.0).astype("float64")
    position = np.zeros(len(frame_5m), dtype="float64")
    state = 0.0
    previous_regime = "unknown"
    for index, market_regime in enumerate(regimes):
        value = rsi_values[index]
        if market_regime != previous_regime:
            state = 0.0
            previous_regime = market_regime

        if market_regime == "up":
            if trend_mode == "always":
                state = 1.0
            elif trend_mode == "momentum":
                if state <= 0.0 and value >= 55.0:
                    state = 1.0
                elif state > 0.0 and value < 45.0:
                    state = 0.0
            elif trend_mode == "pullback":
                if state <= 0.0 and value <= 40.0:
                    state = 1.0
                elif state > 0.0 and value >= 60.0:
                    state = 0.0
        elif market_regime == "down":
            if trend_mode == "always":
                state = -1.0
            elif trend_mode == "momentum":
                if state >= 0.0 and value <= 45.0:
                    state = -1.0
                elif state < 0.0 and value > 55.0:
                    state = 0.0
            elif trend_mode == "pullback":
                if state >= 0.0 and value >= 60.0:
                    state = -1.0
                elif state < 0.0 and value <= 40.0:
                    state = 0.0
        elif market_regime == "range" and range_mode != "flat":
            low, high = (30.0, 70.0) if range_mode == "rsi30" else (35.0, 65.0)
            if state == 0.0 and value <= low:
                state = 1.0
            elif state == 0.0 and value >= high:
                state = -1.0
            elif (state > 0.0 and value >= 50.0) or (state < 0.0 and value <= 50.0):
                state = 0.0
        else:
            state = 0.0
        position[index] = state
    return pd.Series(position, index=frame_5m.index), regimes


def apply_path(frame_5m, desired, *, stop_pct, target_pct, cooldown_bars, leverage):
    components = monthly5_risk_cache.simulate_trade_risk_path(
        frame_5m,
        desired,
        stop_pct=stop_pct,
        target_pct=target_pct,
        cooldown_bars=cooldown_bars,
    )
    return regime.apply_monthly_lock(
        frame_5m,
        *components,
        leverage=leverage,
        lock_scale=recovery.BASE_CONFIG["post_lock_scale"],
        lock_trigger=recovery.BASE_CONFIG["lock_trigger"],
        lock_floor=recovery.BASE_CONFIG["lock_floor"],
        daily_stop=recovery.BASE_CONFIG["daily_stop"],
        monthly_stop=-0.08,
        monthly_recovery_scale=0.0,
    )


def selection_rank(row):
    training = row["training"]
    validation = row["validation"]
    eligible = all(
        summary["min_month_pct"] >= -15.0 and summary["max_drawdown_pct"] >= -35.0
        for summary in (training, validation)
    )
    training_hit_rate = training["months_ge_5"] / max(1, training["months"])
    validation_hit_rate = validation["months_ge_5"] / max(1, validation["months"])
    training_nonnegative_rate = training["months_ge_0"] / max(1, training["months"])
    validation_nonnegative_rate = validation["months_ge_0"] / max(1, validation["months"])
    return (
        int(eligible),
        min(training_hit_rate, validation_hit_rate),
        training["months_ge_5"] + validation["months_ge_5"],
        min(training_nonnegative_rate, validation_nonnegative_rate),
        min(training["min_month_pct"], validation["min_month_pct"]),
        min(training["max_drawdown_pct"], validation["max_drawdown_pct"]),
        training["avg_month_pct"] + validation["avg_month_pct"],
    )


def select_validation_winner(candidates):
    return max(candidates, key=selection_rank)


def _without_monthly(summary):
    return {key: value for key, value in summary.items() if key != "monthly"}


def evaluate_config(frame_5m, config):
    labels = regime.classify_completed_4h(
        frame_5m,
        distance_threshold=recovery.BASE_CONFIG["distance_threshold"],
        slope_threshold=recovery.BASE_CONFIG["slope_threshold"],
    )
    desired, _ = build_specialist_position(
        frame_5m,
        labels,
        regime.completed_15m_rsi(frame_5m),
        trend_mode=config["trend_mode"],
        range_mode=config["range_mode"],
    )
    return apply_path(
        frame_5m,
        desired,
        stop_pct=config["stop_pct"],
        target_pct=config["target_pct"],
        cooldown_bars=config["cooldown_bars"],
        leverage=config["leverage"],
    )


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
    labels = regime.classify_completed_4h(
        frame_5m,
        distance_threshold=recovery.BASE_CONFIG["distance_threshold"],
        slope_threshold=recovery.BASE_CONFIG["slope_threshold"],
    )
    rsi = regime.completed_15m_rsi(frame_5m)
    candidates = []
    for policy in policy_configs():
        desired, _ = build_specialist_position(
            frame_5m,
            labels,
            rsi,
            trend_mode=policy["trend_mode"],
            range_mode=policy["range_mode"],
        )
        for stop_pct, target_pct, cooldown_bars in RISK_CONFIGS:
            components = monthly5_risk_cache.simulate_trade_risk_path(
                frame_5m,
                desired,
                stop_pct=stop_pct,
                target_pct=target_pct,
                cooldown_bars=cooldown_bars,
            )
            for leverage in LEVERAGES:
                factors, scales = regime.apply_monthly_lock(
                    frame_5m,
                    *components,
                    leverage=leverage,
                    lock_scale=recovery.BASE_CONFIG["post_lock_scale"],
                    lock_trigger=recovery.BASE_CONFIG["lock_trigger"],
                    lock_floor=recovery.BASE_CONFIG["lock_floor"],
                    daily_stop=recovery.BASE_CONFIG["daily_stop"],
                    monthly_stop=-0.08,
                    monthly_recovery_scale=0.0,
                )
                config = {
                    **policy,
                    "stop_pct": stop_pct,
                    "target_pct": target_pct,
                    "cooldown_bars": cooldown_bars,
                    "leverage": leverage,
                }
                candidates.append(
                    {
                        "name": (
                            f"{policy['name']}|stop{stop_pct}|target{target_pct}"
                            f"|cooldown{cooldown_bars}|lev{leverage}"
                        ),
                        "config": config,
                        "training": regime.summarize_factors(
                            frame_5m,
                            factors,
                            scales,
                            start=regime.REPORT_START,
                            end=TRAIN_END,
                        ),
                        "validation": regime.summarize_factors(
                            frame_5m,
                            factors,
                            scales,
                            start=VALIDATION_START,
                            end=VALIDATION_END,
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
    winner = select_validation_winner(candidates)
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
        "method": "completed_4h_regime_specialists_nested_validation",
        "source": frame_5m.attrs.get("kline_source", "binance_history_um"),
        "selection_uses_holdout": False,
        "training_period": f"{regime.REPORT_START}..{TRAIN_END}",
        "validation_period": f"{VALIDATION_START}..{VALIDATION_END}",
        "holdout_start": regime.HOLDOUT_START,
        "candidate_count": len(candidates),
        "winner": {"name": winner["name"], "config": winner["config"]},
        "training": winner["training"],
        "validation": winner["validation"],
        "development": winner["development"],
        "holdout": holdout,
        "full": full,
        "top_validation": [
            {
                "name": row["name"],
                "config": row["config"],
                "training": _without_monthly(row["training"]),
                "validation": _without_monthly(row["validation"]),
                "holdout_diagnostic_only": _without_monthly(row["holdout_diagnostic"]),
            }
            for row in sorted(
                candidates,
                key=selection_rank,
                reverse=True,
            )[:20]
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
