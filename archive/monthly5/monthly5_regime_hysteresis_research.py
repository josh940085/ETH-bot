"""Research confirmation and range-grace hysteresis for completed 4h regimes."""

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


CONFIRMATION_BARS = (1, 2, 3, 4)
RANGE_GRACE_BARS = (0, 1, 2, 3, 6)
RISK_PROFILES = (
    {"stop_pct": 0.02, "target_pct": 0.04, "cooldown_bars": 12, "leverage": 1},
    {"stop_pct": 0.03, "target_pct": 0.06, "cooldown_bars": 12, "leverage": 1},
)
DEFAULT_OUTPUT = Path(
    ".runtime/data/backtests/monthly5_search/regime_hysteresis_v1_2020_20260803.json"
)
CONFIRMATION_TIMEBASE = "completed_4h_events"


def build_hysteresis_position(frame_5m, labels, *, confirmation_bars, range_grace_bars):
    regimes = regime.align_completed_series(frame_5m.index, labels, "unknown").astype(str)
    active = 0.0
    pending = ""
    pending_count = 0
    range_count = 0
    event_positions = []
    for market_regime in labels.astype(str):
        if market_regime in {"up", "down"}:
            range_count = 0
            if market_regime == pending:
                pending_count += 1
            else:
                pending = market_regime
                pending_count = 1
            if pending_count >= max(1, int(confirmation_bars)):
                active = 1.0 if market_regime == "up" else -1.0
        elif market_regime == "range":
            pending = ""
            pending_count = 0
            range_count += 1
            if range_count > max(0, int(range_grace_bars)):
                active = 0.0
        else:
            active = 0.0
            pending = ""
            pending_count = 0
            range_count = 0
        event_positions.append(active)
    position_events = pd.Series(
        event_positions,
        index=labels.index,
        dtype="float64",
    )
    position = regime.align_completed_series(frame_5m.index, position_events, 0.0).astype(
        "float64"
    )
    return pd.Series(position, index=frame_5m.index), regimes


def apply_components(frame_5m, components, profile):
    return regime.apply_monthly_lock(
        frame_5m,
        *components,
        leverage=profile["leverage"],
        lock_scale=recovery.BASE_CONFIG["post_lock_scale"],
        lock_trigger=recovery.BASE_CONFIG["lock_trigger"],
        lock_floor=recovery.BASE_CONFIG["lock_floor"],
        daily_stop=recovery.BASE_CONFIG["daily_stop"],
        monthly_stop=-0.08,
        monthly_recovery_scale=0.0,
    )


def evaluate_config(frame_5m, config):
    labels = regime.classify_completed_4h(
        frame_5m,
        distance_threshold=recovery.BASE_CONFIG["distance_threshold"],
        slope_threshold=recovery.BASE_CONFIG["slope_threshold"],
    )
    desired, _ = build_hysteresis_position(
        frame_5m,
        labels,
        confirmation_bars=config["confirmation_bars"],
        range_grace_bars=config["range_grace_bars"],
    )
    components = monthly5_risk_cache.simulate_trade_risk_path(
        frame_5m,
        desired,
        stop_pct=config["stop_pct"],
        target_pct=config["target_pct"],
        cooldown_bars=config["cooldown_bars"],
    )
    return apply_components(frame_5m, components, config)


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
    labels = regime.classify_completed_4h(
        frame_5m,
        distance_threshold=recovery.BASE_CONFIG["distance_threshold"],
        slope_threshold=recovery.BASE_CONFIG["slope_threshold"],
    )
    candidates = []
    for confirmation_bars in CONFIRMATION_BARS:
        for range_grace_bars in RANGE_GRACE_BARS:
            desired, _ = build_hysteresis_position(
                frame_5m,
                labels,
                confirmation_bars=confirmation_bars,
                range_grace_bars=range_grace_bars,
            )
            for profile in RISK_PROFILES:
                components = monthly5_risk_cache.simulate_trade_risk_path(
                    frame_5m,
                    desired,
                    stop_pct=profile["stop_pct"],
                    target_pct=profile["target_pct"],
                    cooldown_bars=profile["cooldown_bars"],
                )
                factors, scales = apply_components(frame_5m, components, profile)
                config = {
                    "confirmation_bars": confirmation_bars,
                    "range_grace_bars": range_grace_bars,
                    **profile,
                }
                candidates.append(
                    {
                        "name": (
                            f"confirm{confirmation_bars}_grace{range_grace_bars}"
                            f"|stop{profile['stop_pct']}|target{profile['target_pct']}"
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
    holdout_pass = (
        holdout["months_ge_5"] == holdout["months"]
        and holdout["min_month_pct"] >= -15.0
        and holdout["max_drawdown_pct"] >= -35.0
        and prefix["recursive_stable"]
    )
    return {
        "schema_version": 1,
        "method": "completed_4h_regime_confirmation_and_range_grace",
        "confirmation_timebase": CONFIRMATION_TIMEBASE,
        "source": frame_5m.attrs.get("kline_source", "binance_history_um"),
        "selection_uses_holdout": False,
        "candidate_count": len(candidates),
        "winner": {"name": winner["name"], "config": winner["config"]},
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
