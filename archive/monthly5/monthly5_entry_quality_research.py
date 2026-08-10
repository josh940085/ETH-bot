"""Research completed-4h quality gates for every actual monthly5 entry."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator

import monthly5_intraday_regime as regime
import monthly5_intramonth_recovery_research as account
import monthly5_regime_specialist_research as specialist
import monthly5_selector_cache
import monthly5_volatility_regime_research as volatility
import monthly5_volatility_walkforward as volatility_walkforward


MAX_RSI_VALUES = (None, 65.0, 70.0, 75.0, 80.0)
MAX_EXTENSION_ATR_VALUES = (None, 1.5, 2.0, 3.0, 4.0)
BASELINE_CONFIG = {
    "name": "baseline_no_recovery",
    "mode": "none",
    "trigger": None,
    "scale": 0.0,
    "exit": 0.0,
}
DEFAULT_OUTPUT = Path(
    ".runtime/data/backtests/monthly5_search/entry_quality_v1_2020_20260803.json"
)


def completed_4h_quality(frame_5m):
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
    quality = pd.DataFrame(
        {
            "rsi": RSIIndicator(close, window=14, fillna=False).rsi(),
            "distance_atr": (close - ma25) / atr14.replace(0.0, np.nan),
        },
        index=frame_4h.index,
    )
    quality.index = quality.index + pd.Timedelta(minutes=5)
    return quality


def build_entry_allowed(frame_5m, desired, quality, *, max_rsi, max_extension_atr):
    rsi = regime.align_completed_series(frame_5m.index, quality["rsi"], np.nan).astype(
        "float64"
    )
    extension = regime.align_completed_series(
        frame_5m.index, quality["distance_atr"], np.nan
    ).astype("float64")
    wanted = np.asarray(desired, dtype="float64")
    allowed = np.ones(len(frame_5m), dtype="bool")
    filtered = max_rsi is not None or max_extension_atr is not None
    if filtered:
        allowed &= np.isfinite(rsi) & np.isfinite(extension)
    if max_rsi is not None:
        allowed &= np.where(
            wanted > 0.0,
            rsi <= float(max_rsi),
            np.where(wanted < 0.0, rsi >= 100.0 - float(max_rsi), True),
        )
    if max_extension_atr is not None:
        allowed &= np.where(
            wanted > 0.0,
            extension <= float(max_extension_atr),
            np.where(wanted < 0.0, extension >= -float(max_extension_atr), True),
        )
    return allowed


def quality_configs():
    return tuple(
        {
            "name": f"rsi{max_rsi}_extension{max_extension}",
            "max_rsi": max_rsi,
            "max_extension_atr": max_extension,
        }
        for max_rsi in MAX_RSI_VALUES
        for max_extension in MAX_EXTENSION_ATR_VALUES
    )


def evaluate_config(frame_5m, config):
    desired, primary_ids, _ = volatility_walkforward.build_primary_path(
        frame_5m,
        **account.PRIMARY_CONFIG,
    )
    allowed = build_entry_allowed(
        frame_5m,
        desired,
        completed_4h_quality(frame_5m),
        max_rsi=config["max_rsi"],
        max_extension_atr=config["max_extension_atr"],
    )
    return account.simulate_account_path(
        frame_5m,
        desired,
        primary_ids,
        desired,
        BASELINE_CONFIG,
        entry_allowed=allowed,
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
    desired, primary_ids, primary_selections = volatility_walkforward.build_primary_path(
        frame_5m,
        **account.PRIMARY_CONFIG,
    )
    quality = completed_4h_quality(frame_5m)
    candidates = []
    for config in quality_configs():
        allowed = build_entry_allowed(
            frame_5m,
            desired,
            quality,
            max_rsi=config["max_rsi"],
            max_extension_atr=config["max_extension_atr"],
        )
        factors, scales, _, _, positions = account.simulate_account_path(
            frame_5m,
            desired,
            primary_ids,
            desired,
            BASELINE_CONFIG,
            entry_allowed=allowed,
        )
        desired_mask = np.asarray(desired) != 0.0
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
        "method": "completed_4h_entry_and_reentry_quality_gate",
        "source": frame_5m.attrs.get("kline_source", "binance_history_um"),
        "primary_config": account.PRIMARY_CONFIG,
        "primary_monthly_selections": primary_selections,
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
