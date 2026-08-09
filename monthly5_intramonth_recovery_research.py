"""Research exact intramonth recovery execution for the causal ATR policy."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import monthly5_intraday_regime as regime
import monthly5_regime_hysteresis_research as hysteresis
import monthly5_regime_specialist_research as specialist
import monthly5_risk_cache
import monthly5_risk_profile_walkforward as risk_walkforward
import monthly5_selector_cache
import monthly5_volatility_regime_research as volatility
import monthly5_volatility_walkforward as volatility_walkforward


PRIMARY_CONFIG = {"lookback_months": 24, "score_mode": "tail"}
RECOVERY_MODES = ("alternate_atr", "adx", "inverse")
RECOVERY_TRIGGERS = (-0.02, -0.04, -0.06)
RECOVERY_SCALES = (0.25, 0.50)
RECOVERY_EXITS = (-0.01, 0.0)
MONTHLY_HARD_STOP = -0.08
DEFAULT_OUTPUT = Path(
    ".runtime/data/backtests/monthly5_search/intramonth_recovery_v1_2020_20260803.json"
)


def recovery_configs():
    rows = [
        {
            "name": "baseline_no_recovery",
            "mode": "none",
            "trigger": None,
            "scale": 0.0,
            "exit": 0.0,
        }
    ]
    for mode in RECOVERY_MODES:
        for trigger in RECOVERY_TRIGGERS:
            for scale in RECOVERY_SCALES:
                for exit_level in RECOVERY_EXITS:
                    rows.append(
                        {
                            "name": f"{mode}_trigger{trigger}_scale{scale}_exit{exit_level}",
                            "mode": mode,
                            "trigger": trigger,
                            "scale": scale,
                            "exit": exit_level,
                        }
                    )
    return rows


def build_recovery_paths(frame_5m, primary_desired, primary_ids):
    desired_paths = volatility_walkforward.build_desired_paths(frame_5m)
    desired_matrix = np.vstack([np.asarray(path, dtype="float64") for path in desired_paths])
    alternate = desired_matrix[1 - np.asarray(primary_ids, dtype="int32"), np.arange(len(frame_5m))]
    adx_labels = volatility.classify_completed_4h(
        frame_5m,
        {
            "name": "adx25.0_s0.0",
            "mode": "adx",
            "adx_threshold": 25.0,
            "slope_atr": 0.0,
        },
    )
    adx, _ = hysteresis.build_hysteresis_position(
        frame_5m,
        adx_labels,
        confirmation_bars=1,
        range_grace_bars=6,
    )
    return {
        "none": np.asarray(primary_desired, dtype="float64"),
        "alternate_atr": np.asarray(alternate, dtype="float64"),
        "adx": np.asarray(adx, dtype="float64"),
        "inverse": np.asarray(primary_desired, dtype="float64") * -1.0,
    }


def simulate_account_path(
    frame_5m,
    primary_desired,
    primary_ids,
    recovery_desired,
    config,
    *,
    round_trip_fee=monthly5_risk_cache.ROUND_TRIP_FEE,
):
    close = pd.to_numeric(frame_5m["close"], errors="coerce").to_numpy(dtype="float64")
    open_price = pd.to_numeric(frame_5m["open"], errors="coerce").to_numpy(dtype="float64")
    high = pd.to_numeric(frame_5m["high"], errors="coerce").to_numpy(dtype="float64")
    low = pd.to_numeric(frame_5m["low"], errors="coerce").to_numpy(dtype="float64")
    primary = np.asarray(primary_desired, dtype="float64")
    primary_ids = np.asarray(primary_ids, dtype="int32")
    recovery_path = np.asarray(recovery_desired, dtype="float64")
    count = len(frame_5m)
    factors = np.ones(count, dtype="float64")
    scales = np.ones(count, dtype="float64")
    recovery_flags = np.zeros(count, dtype="bool")
    one_way_fee = float(round_trip_fee) / 2.0
    month_keys = frame_5m.index.tz_localize(None).to_period("M")
    day_keys = frame_5m.index.floor("D")

    month_equity = 1.0
    day_equity = 1.0
    previous_month = None
    previous_day = None
    locked = False
    floor_guarded = False
    monthly_guarded = False
    daily_guarded = False
    recovery_active = False
    active = 0.0
    entry = 0.0
    cooldown = 0
    previous_scale = 1.0
    previous_position = 0.0
    previous_strategy_id = None
    trigger_count = 0

    for index, month in enumerate(month_keys):
        if month != previous_month:
            month_equity = 1.0
            locked = False
            floor_guarded = False
            monthly_guarded = False
            recovery_active = False
            previous_month = month
        day = day_keys[index]
        if day != previous_day:
            day_equity = 1.0
            daily_guarded = False
            previous_day = day

        risk_off = floor_guarded or monthly_guarded or daily_guarded
        if risk_off:
            scale = 0.0
        elif locked:
            scale = 0.15
        elif recovery_active:
            scale = max(0.0, min(1.0, float(config["scale"])))
        else:
            scale = 1.0
        scales[index] = scale
        recovery_flags[index] = recovery_active
        wanted = recovery_path[index] if recovery_active else primary[index]
        wanted = float(np.sign(wanted)) if np.isfinite(wanted) else 0.0
        if risk_off:
            wanted = 0.0
        strategy_id = 100 if recovery_active else int(primary_ids[index])
        strategy_changed = previous_strategy_id is not None and strategy_id != previous_strategy_id
        previous_close = close[index - 1] if index else close[index]
        turnover = 0.0
        if active and (wanted != active or strategy_changed):
            turnover += abs(active)
            active = 0.0
            entry = 0.0
        if strategy_changed:
            cooldown = 0
        if not active:
            if cooldown > 0:
                cooldown -= 1
            elif wanted:
                active = wanted
                entry = previous_close
                turnover += abs(active)

        exit_price = None
        if active > 0.0:
            stop_price = entry * (1.0 - volatility.RISK_PROFILE["stop_pct"])
            target_price = entry * (1.0 + volatility.RISK_PROFILE["target_pct"])
            if low[index] <= stop_price:
                exit_price = min(stop_price, open_price[index])
            elif high[index] >= target_price:
                exit_price = target_price
        elif active < 0.0:
            stop_price = entry * (1.0 + volatility.RISK_PROFILE["stop_pct"])
            target_price = entry * (1.0 - volatility.RISK_PROFILE["target_pct"])
            if high[index] >= stop_price:
                exit_price = max(stop_price, open_price[index])
            elif low[index] <= target_price:
                exit_price = target_price

        pnl = 0.0
        actual = active
        if active:
            mark = float(exit_price) if exit_price is not None else close[index]
            pnl = active * (mark / previous_close - 1.0)
        if exit_price is not None:
            turnover += abs(active)
            active = 0.0
            entry = 0.0
            cooldown = max(0, int(volatility.RISK_PROFILE["cooldown_bars"]))

        scale_turnover = abs(scale - previous_scale) * abs(previous_position)
        cost_turnover = turnover * scale + scale_turnover
        factor = 1.0 + pnl * scale - cost_turnover * one_way_fee
        factor = max(factor, 1e-9)
        factors[index] = factor
        month_equity *= factor
        day_equity *= factor

        if day_equity <= 1.0 - 0.02:
            daily_guarded = True
        if month_equity <= 1.0 + MONTHLY_HARD_STOP:
            monthly_guarded = True
        trigger = config.get("trigger")
        if (
            trigger is not None
            and not recovery_active
            and not monthly_guarded
            and month_equity <= 1.0 + float(trigger)
        ):
            recovery_active = True
            trigger_count += 1
        elif recovery_active and month_equity >= 1.0 + float(config["exit"]):
            recovery_active = False
        if not locked and month_equity >= 1.055:
            locked = True
        elif locked and month_equity < 1.051:
            floor_guarded = True

        previous_scale = scale
        previous_position = actual
        previous_strategy_id = strategy_id
    return factors, scales, recovery_flags, trigger_count


def _without_monthly(summary):
    return {key: value for key, value in summary.items() if key != "monthly"}


def evaluate_config(frame_5m, config):
    primary_desired, primary_ids, _ = volatility_walkforward.build_primary_path(
        frame_5m,
        **PRIMARY_CONFIG,
    )
    recovery_paths = build_recovery_paths(frame_5m, primary_desired, primary_ids)
    return simulate_account_path(
        frame_5m,
        primary_desired,
        primary_ids,
        recovery_paths[config["mode"]],
        config,
    )


def verify_prefix_stability(frame_5m, config, full_factors):
    checks = []
    for cutoff in ("2021-12-31", "2023-12-31", "2025-12-31"):
        cutoff_ts = pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        truncated = frame_5m.loc[frame_5m.index <= cutoff_ts]
        factors, _, _, _ = evaluate_config(truncated, config)
        stable = np.array_equal(np.asarray(full_factors)[: len(truncated)], np.asarray(factors))
        checks.append({"cutoff": cutoff, "bars": len(truncated), "stable": bool(stable)})
    return {
        "prefix_stable": all(row["stable"] for row in checks),
        "recursive_stable": all(row["stable"] for row in checks),
        "checks": checks,
    }


def build_report(frame_5m):
    primary_desired, primary_ids, primary_selections = volatility_walkforward.build_primary_path(
        frame_5m,
        **PRIMARY_CONFIG,
    )
    recovery_paths = build_recovery_paths(frame_5m, primary_desired, primary_ids)
    candidates = []
    for config in recovery_configs():
        factors, scales, recovery_flags, trigger_count = simulate_account_path(
            frame_5m,
            primary_desired,
            primary_ids,
            recovery_paths[config["mode"]],
            config,
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
                "holdout_diagnostic": regime.summarize_factors(
                    frame_5m,
                    factors,
                    scales,
                    start=regime.HOLDOUT_START,
                ),
                "trigger_count": trigger_count,
                "recovery_time_pct": round(float(np.mean(recovery_flags)) * 100.0, 4),
                "_factors": factors,
                "_scales": scales,
            }
        )
    winner = max(candidates, key=specialist.selection_rank)
    baseline = next(row for row in candidates if row["name"] == "baseline_no_recovery")
    holdout = winner["holdout_diagnostic"]
    full = regime.summarize_factors(
        frame_5m, winner["_factors"], winner["_scales"], start=regime.REPORT_START
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
        "method": "integrated_intramonth_account_recovery",
        "source": frame_5m.attrs.get("kline_source", "binance_history_um"),
        "execution_semantics": "risk_off_closes_position_and_reentry_uses_new_entry_price",
        "primary_config": PRIMARY_CONFIG,
        "primary_monthly_selections": primary_selections,
        "selection_uses_evaluation_period": False,
        "evaluation_period_reused_during_research": True,
        "candidate_count": len(candidates),
        "winner": {
            "name": winner["name"],
            "config": winner["config"],
            "trigger_count": winner["trigger_count"],
            "recovery_time_pct": winner["recovery_time_pct"],
        },
        "training": winner["training"],
        "validation": winner["validation"],
        "development": winner["development"],
        "holdout": holdout,
        "full": full,
        "baseline": {
            "training": baseline["training"],
            "validation": baseline["validation"],
            "holdout": baseline["holdout_diagnostic"],
        },
        "top_selection": [
            {
                "name": row["name"],
                "config": row["config"],
                "training": _without_monthly(row["training"]),
                "validation": _without_monthly(row["validation"]),
                "holdout_diagnostic_only": _without_monthly(row["holdout_diagnostic"]),
                "trigger_count": row["trigger_count"],
                "recovery_time_pct": row["recovery_time_pct"],
            }
            for row in sorted(candidates, key=specialist.selection_rank, reverse=True)[:20]
        ],
        "bias_evidence": prefix,
        "deployment_ready": False,
        "deployment_blockers": [
            *([] if holdout_pass else ["evaluation_all_months_ge_5_failed"]),
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
