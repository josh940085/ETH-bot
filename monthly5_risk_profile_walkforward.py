"""Causally switch robust 4h risk profiles at month boundaries."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import monthly5_intraday_regime as regime
import monthly5_recovery_research as recovery
import monthly5_regime_specialist_research as specialist
import monthly5_risk_cache
import monthly5_selector_cache


PROFILE_CONFIGS = (
    {
        "name": "trend_flat_stop2_target4",
        "trend_mode": "always",
        "range_mode": "flat",
        "stop_pct": 0.02,
        "target_pct": 0.04,
        "cooldown_bars": 12,
        "leverage": 1,
    },
    {
        "name": "trend_flat_stop3_target6",
        "trend_mode": "always",
        "range_mode": "flat",
        "stop_pct": 0.03,
        "target_pct": 0.06,
        "cooldown_bars": 12,
        "leverage": 1,
    },
)
LOOKBACKS = (6, 12, 24, 36)
SCORE_MODES = ("balanced", "tail")
DEFAULT_OUTPUT = Path(
    ".runtime/data/backtests/monthly5_search/risk_profile_walkforward_v1_2020_20260803.json"
)


def simulate_dynamic_risk_path(
    frame,
    desired_position,
    profile_indices,
    profiles=PROFILE_CONFIGS,
):
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype="float64")
    open_price = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype="float64")
    high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype="float64")
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype="float64")
    desired = np.asarray(desired_position, dtype="float64")
    selected = np.asarray(profile_indices, dtype="int32")
    count = len(frame)
    pnl = np.zeros(count, dtype="float64")
    turnover = np.zeros(count, dtype="float64")
    actual = np.zeros(count, dtype="float64")
    active = 0.0
    entry = 0.0
    entry_stop = 0.0
    entry_target = 0.0
    cooldown = 0

    for index in range(count):
        previous_close = close[index - 1] if index else close[index]
        profile_changed = index > 0 and selected[index] != selected[index - 1]
        wanted = desired[index] if np.isfinite(desired[index]) else 0.0
        if active and (wanted != active or profile_changed):
            turnover[index] += abs(active)
            active = 0.0
            entry = 0.0
        if profile_changed:
            cooldown = 0
        if not active:
            if cooldown > 0:
                cooldown -= 1
            elif wanted:
                profile = profiles[int(selected[index])]
                active = float(np.sign(wanted))
                entry = previous_close
                entry_stop = float(profile["stop_pct"])
                entry_target = float(profile["target_pct"])
                turnover[index] += abs(active)

        exit_price = None
        if active > 0.0:
            stop_price = entry * (1.0 - entry_stop)
            target_price = entry * (1.0 + entry_target)
            if low[index] <= stop_price:
                exit_price = min(stop_price, open_price[index])
            elif high[index] >= target_price:
                exit_price = target_price
        elif active < 0.0:
            stop_price = entry * (1.0 + entry_stop)
            target_price = entry * (1.0 - entry_target)
            if high[index] >= stop_price:
                exit_price = max(stop_price, open_price[index])
            elif low[index] <= target_price:
                exit_price = target_price

        if active:
            mark = float(exit_price) if exit_price is not None else close[index]
            pnl[index] = active * (mark / previous_close - 1.0)
            actual[index] = active
        if exit_price is not None:
            profile = profiles[int(selected[index])]
            turnover[index] += abs(active)
            active = 0.0
            entry = 0.0
            cooldown = max(0, int(profile["cooldown_bars"]))
    return pnl, turnover, actual


def candidate_score(history, mode):
    hit_rate = np.mean(history >= 0.05, axis=1)
    nonnegative = np.mean(history >= 0.0, axis=1)
    q25 = np.percentile(history, 25, axis=1)
    minimum = np.min(history, axis=1)
    mean = np.mean(history, axis=1)
    if mode == "tail":
        return q25 + 0.25 * minimum + 0.05 * hit_rate + 0.01 * nonnegative
    return mean + 0.50 * q25 + 0.05 * hit_rate + 0.01 * nonnegative


def select_monthly_profiles(frame_5m, standalone_factors, *, lookback_months, score_mode):
    month_keys = frame_5m.index.tz_localize(None).to_period("M")
    months = month_keys.unique()
    matrix = np.zeros((len(standalone_factors), len(months)), dtype="float64")
    for profile_index, factors in enumerate(standalone_factors):
        monthly = pd.Series(factors, index=month_keys).groupby(level=0).prod() - 1.0
        matrix[profile_index] = monthly.reindex(months).to_numpy(dtype="float64")
    selected_months = np.zeros(len(months), dtype="int32")
    for target in range(max(1, int(lookback_months)), len(months)):
        history = matrix[:, max(0, target - int(lookback_months)) : target]
        selected_months[target] = int(np.argmax(candidate_score(history, score_mode)))
    month_lookup = {month: index for index, month in enumerate(months)}
    selected_bars = np.asarray([selected_months[month_lookup[month]] for month in month_keys])
    selections = [
        {
            "month": str(month),
            "profile": PROFILE_CONFIGS[int(selected_months[index])]["name"],
            "switched": bool(index and selected_months[index] != selected_months[index - 1]),
            "history_start": str(months[max(0, index - int(lookback_months))]) if index else "",
            "history_end": str(months[index - 1]) if index else "",
        }
        for index, month in enumerate(months)
    ]
    return selected_bars, selections


def apply_lock(frame_5m, components):
    return regime.apply_monthly_lock(
        frame_5m,
        *components,
        leverage=1,
        lock_scale=recovery.BASE_CONFIG["post_lock_scale"],
        lock_trigger=recovery.BASE_CONFIG["lock_trigger"],
        lock_floor=recovery.BASE_CONFIG["lock_floor"],
        daily_stop=recovery.BASE_CONFIG["daily_stop"],
        monthly_stop=-0.08,
        monthly_recovery_scale=0.0,
    )


def build_desired_paths(frame_5m):
    labels = regime.classify_completed_4h(
        frame_5m,
        distance_threshold=recovery.BASE_CONFIG["distance_threshold"],
        slope_threshold=recovery.BASE_CONFIG["slope_threshold"],
    )
    rsi = regime.completed_15m_rsi(frame_5m)
    paths = []
    for profile in PROFILE_CONFIGS:
        desired, _ = specialist.build_specialist_position(
            frame_5m,
            labels,
            rsi,
            trend_mode=profile["trend_mode"],
            range_mode=profile["range_mode"],
        )
        paths.append(desired)
    return paths


def run_variant(frame_5m, *, lookback_months, score_mode):
    desired_paths = build_desired_paths(frame_5m)
    standalone_factors = []
    for profile, desired in zip(PROFILE_CONFIGS, desired_paths):
        components = monthly5_risk_cache.simulate_trade_risk_path(
            frame_5m,
            desired,
            stop_pct=profile["stop_pct"],
            target_pct=profile["target_pct"],
            cooldown_bars=profile["cooldown_bars"],
        )
        factors, _ = apply_lock(frame_5m, components)
        standalone_factors.append(factors)
    selected, selections = select_monthly_profiles(
        frame_5m,
        standalone_factors,
        lookback_months=lookback_months,
        score_mode=score_mode,
    )
    desired_matrix = np.vstack([np.asarray(path, dtype="float64") for path in desired_paths])
    desired = desired_matrix[selected, np.arange(len(frame_5m))]
    components = simulate_dynamic_risk_path(frame_5m, desired, selected)
    factors, scales = apply_lock(frame_5m, components)
    return factors, scales, selections


def selection_rank(row):
    return specialist.selection_rank(row)


def _without_monthly(summary):
    return {key: value for key, value in summary.items() if key != "monthly"}


def verify_prefix_stability(frame_5m, config, full_factors):
    checks = []
    for cutoff in ("2021-12-31", "2023-12-31", "2025-12-31"):
        cutoff_ts = pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        truncated = frame_5m.loc[frame_5m.index <= cutoff_ts]
        factors, _, _ = run_variant(truncated, **config)
        stable = np.array_equal(np.asarray(full_factors)[: len(truncated)], np.asarray(factors))
        checks.append({"cutoff": cutoff, "bars": len(truncated), "stable": bool(stable)})
    return {
        "prefix_stable": all(row["stable"] for row in checks),
        "recursive_stable": all(row["stable"] for row in checks),
        "checks": checks,
    }


def build_report(frame_5m):
    variants = []
    for lookback in LOOKBACKS:
        for score_mode in SCORE_MODES:
            config = {"lookback_months": lookback, "score_mode": score_mode}
            factors, scales, selections = run_variant(frame_5m, **config)
            variants.append(
                {
                    "name": f"lb{lookback}_{score_mode}",
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
                    "holdout_diagnostic": regime.summarize_factors(
                        frame_5m, factors, scales, start=regime.HOLDOUT_START
                    ),
                    "_factors": factors,
                    "_scales": scales,
                    "_selections": selections,
                }
            )
    winner = max(variants, key=selection_rank)
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
        "method": "month_boundary_dynamic_risk_profile_walkforward",
        "source": frame_5m.attrs.get("kline_source", "binance_history_um"),
        "selection_uses_holdout": False,
        "profiles": PROFILE_CONFIGS,
        "winner": {"name": winner["name"], "config": winner["config"]},
        "training": winner["training"],
        "validation": winner["validation"],
        "development": winner["development"],
        "holdout": holdout,
        "full": full,
        "monthly_selections": winner["_selections"],
        "switch_count": sum(bool(row["switched"]) for row in winner["_selections"]),
        "variants": [
            {
                "name": row["name"],
                "config": row["config"],
                "training": _without_monthly(row["training"]),
                "validation": _without_monthly(row["validation"]),
                "development": _without_monthly(row["development"]),
                "holdout_diagnostic_only": _without_monthly(row["holdout_diagnostic"]),
            }
            for row in variants
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
