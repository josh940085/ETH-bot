"""Causally switch robust ATR-normalized 4h regimes at month boundaries."""

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


CANDIDATE_CONFIGS = (
    {
        "name": "atr_d0.7_s0.05_confirm4_grace3",
        "label": {
            "name": "atr_d0.7_s0.05",
            "mode": "atr",
            "distance_atr": 0.7,
            "slope_atr": 0.05,
        },
        "confirmation_bars": 4,
        "range_grace_bars": 3,
        **volatility.RISK_PROFILE,
    },
    {
        "name": "atr_d0.7_s0.1_confirm1_grace6",
        "label": {
            "name": "atr_d0.7_s0.1",
            "mode": "atr",
            "distance_atr": 0.7,
            "slope_atr": 0.10,
        },
        "confirmation_bars": 1,
        "range_grace_bars": 6,
        **volatility.RISK_PROFILE,
    },
)
LOOKBACKS = (6, 12, 24, 36)
SCORE_MODES = ("balanced", "tail")
DEFAULT_OUTPUT = Path(
    ".runtime/data/backtests/monthly5_search/volatility_walkforward_v1_2020_20260803.json"
)


def build_desired_paths(frame_5m):
    paths = []
    for config in CANDIDATE_CONFIGS:
        labels = volatility.classify_completed_4h(frame_5m, config["label"])
        desired, _ = hysteresis.build_hysteresis_position(
            frame_5m,
            labels,
            confirmation_bars=config["confirmation_bars"],
            range_grace_bars=config["range_grace_bars"],
        )
        paths.append(desired)
    return paths


def run_variant(frame_5m, *, lookback_months, score_mode):
    desired_paths = build_desired_paths(frame_5m)
    standalone_factors = []
    for config, desired in zip(CANDIDATE_CONFIGS, desired_paths):
        components = monthly5_risk_cache.simulate_trade_risk_path(
            frame_5m,
            desired,
            stop_pct=config["stop_pct"],
            target_pct=config["target_pct"],
            cooldown_bars=config["cooldown_bars"],
        )
        factors, _ = volatility.apply_components(frame_5m, components)
        standalone_factors.append(factors)
    selected, selections = risk_walkforward.select_monthly_profiles(
        frame_5m,
        standalone_factors,
        lookback_months=lookback_months,
        score_mode=score_mode,
        profiles=CANDIDATE_CONFIGS,
    )
    desired_matrix = np.vstack([np.asarray(path, dtype="float64") for path in desired_paths])
    desired = desired_matrix[selected, np.arange(len(frame_5m))]
    components = risk_walkforward.simulate_dynamic_risk_path(
        frame_5m,
        desired,
        selected,
        profiles=CANDIDATE_CONFIGS,
    )
    factors, scales = volatility.apply_components(frame_5m, components)
    return factors, scales, selections


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
                    "_selections": selections,
                }
            )
    winner = max(variants, key=specialist.selection_rank)
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
        "method": "month_boundary_atr_regime_walkforward",
        "source": frame_5m.attrs.get("kline_source", "binance_history_um"),
        "selection_uses_holdout": False,
        "candidates": CANDIDATE_CONFIGS,
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
