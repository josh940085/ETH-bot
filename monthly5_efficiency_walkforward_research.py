"""Research causal monthly selection of completed-4h efficiency entry gates."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import monthly5_intraday_regime as regime
import monthly5_intramonth_recovery_research as account
import monthly5_regime_specialist_research as specialist
import monthly5_risk_profile_walkforward as selector
import monthly5_selector_cache
import monthly5_trend_efficiency_research as efficiency
import monthly5_volatility_walkforward as walkforward


LOOKBACK_VALUES = (6, 12, 24, 36)
SCORE_MODES = ("balanced", "tail")
DEFAULT_OUTPUT = Path(
    ".runtime/data/backtests/monthly5_search/efficiency_walkforward_v1_2020_20260803.json"
)


def build_candidate_paths(frame_5m, desired, strategy_ids):
    efficiency_cache = {
        window: efficiency.completed_4h_efficiency(frame_5m, window)
        for window in efficiency.WINDOW_VALUES
    }
    rows = []
    for config in efficiency.efficiency_configs():
        allowed = efficiency.build_entry_allowed(
            frame_5m, config, efficiency_cache.get(config["window"])
        )
        factors, scales, _, _, positions = account.simulate_account_path(
            frame_5m,
            desired,
            strategy_ids,
            desired,
            efficiency.ACCOUNT_CONFIG,
            entry_allowed=allowed,
        )
        rows.append(
            {
                "config": config,
                "allowed": allowed,
                "factors": factors,
                "scales": scales,
                "positions": positions,
            }
        )
    return rows


def prepare_paths(frame_5m):
    desired, strategy_ids, primary_selections = walkforward.build_primary_path(
        frame_5m, **account.PRIMARY_CONFIG
    )
    candidates = build_candidate_paths(frame_5m, desired, strategy_ids)
    return desired, strategy_ids, primary_selections, candidates


def run_variant(frame_5m, lookback_months, score_mode, prepared=None):
    desired, strategy_ids, primary_selections, candidates = (
        prepared if prepared is not None else prepare_paths(frame_5m)
    )
    profiles = tuple(row["config"] for row in candidates)
    selected, selections = selector.select_monthly_profiles(
        frame_5m,
        [row["factors"] for row in candidates],
        lookback_months=lookback_months,
        score_mode=score_mode,
        profiles=profiles,
    )
    allowed_matrix = np.vstack([row["allowed"] for row in candidates])
    allowed = allowed_matrix[selected, np.arange(len(frame_5m))]
    factors, scales, _, _, positions = account.simulate_account_path(
        frame_5m,
        desired,
        strategy_ids,
        desired,
        efficiency.ACCOUNT_CONFIG,
        entry_allowed=allowed,
    )
    return factors, scales, positions, selections, primary_selections


def _without_monthly(summary):
    return {key: value for key, value in summary.items() if key != "monthly"}


def verify_prefix_stability(frame_5m, config, full_factors):
    checks = []
    for cutoff in ("2021-12-31", "2023-12-31", "2025-12-31"):
        cutoff_ts = pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        truncated = frame_5m.loc[frame_5m.index <= cutoff_ts]
        if config["mode"] == "fixed_baseline":
            factors, _, _, _, _ = efficiency.evaluate_config(
                truncated, efficiency.BASELINE_CONFIG
            )
        else:
            factors, _, _, _, _ = run_variant(
                truncated, config["lookback_months"], config["score_mode"]
            )
        stable = np.array_equal(np.asarray(full_factors)[: len(truncated)], np.asarray(factors))
        checks.append({"cutoff": cutoff, "bars": len(truncated), "stable": bool(stable)})
    return {
        "prefix_stable": all(row["stable"] for row in checks),
        "recursive_stable": all(row["stable"] for row in checks),
        "checks": checks,
    }


def build_report(frame_5m):
    candidates = []
    prepared = prepare_paths(frame_5m)
    _, _, primary_selections, standalone = prepared
    baseline = standalone[0]
    baseline_config = {
        "name": "fixed_baseline_no_efficiency_gate",
        "mode": "fixed_baseline",
        "lookback_months": None,
        "score_mode": None,
    }
    candidates.append(
        {
            "name": baseline_config["name"],
            "config": baseline_config,
            "selections": [],
            "primary_selections": primary_selections,
            "training": regime.summarize_factors(
                frame_5m,
                baseline["factors"],
                baseline["scales"],
                start=regime.REPORT_START,
                end=specialist.TRAIN_END,
            ),
            "validation": regime.summarize_factors(
                frame_5m,
                baseline["factors"],
                baseline["scales"],
                start=specialist.VALIDATION_START,
                end=specialist.VALIDATION_END,
            ),
            "development": regime.summarize_factors(
                frame_5m,
                baseline["factors"],
                baseline["scales"],
                start=regime.REPORT_START,
                end=regime.DEVELOPMENT_END,
            ),
            "evaluation": regime.summarize_factors(
                frame_5m,
                baseline["factors"],
                baseline["scales"],
                start=regime.HOLDOUT_START,
            ),
            "actual_exposure_pct": round(
                float(np.mean(baseline["positions"] != 0.0)) * 100.0, 4
            ),
            "_factors": baseline["factors"],
            "_scales": baseline["scales"],
        }
    )
    for lookback_months in LOOKBACK_VALUES:
        for score_mode in SCORE_MODES:
            factors, scales, positions, selections, primary_selections = run_variant(
                frame_5m, lookback_months, score_mode, prepared
            )
            config = {
                "name": f"lookback{lookback_months}_{score_mode}",
                "mode": "walkforward",
                "lookback_months": lookback_months,
                "score_mode": score_mode,
            }
            candidates.append(
                {
                    "name": config["name"],
                    "config": config,
                    "selections": selections,
                    "primary_selections": primary_selections,
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
                        frame_5m, factors, scales, start=regime.HOLDOUT_START
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
        "method": "causal_monthly_walkforward_efficiency_gate",
        "source": frame_5m.attrs.get("kline_source", "binance_history_um"),
        "primary_config": account.PRIMARY_CONFIG,
        "primary_monthly_selections": winner["primary_selections"],
        "selection_uses_evaluation_period": False,
        "evaluation_period_reused_during_research": True,
        "candidate_count": len(candidates),
        "winner": {
            "name": winner["name"],
            "config": winner["config"],
            "actual_exposure_pct": winner["actual_exposure_pct"],
            "flat_time_pct": round(100.0 - winner["actual_exposure_pct"], 4),
        },
        "efficiency_monthly_selections": winner["selections"],
        "training": winner["training"],
        "validation": winner["validation"],
        "development": winner["development"],
        "evaluation": evaluation,
        "full": full,
        "ranking": [
            {
                "name": row["name"],
                "config": row["config"],
                "training": _without_monthly(row["training"]),
                "validation": _without_monthly(row["validation"]),
                "evaluation_diagnostic_only": _without_monthly(row["evaluation"]),
                "actual_exposure_pct": row["actual_exposure_pct"],
            }
            for row in sorted(candidates, key=specialist.selection_rank, reverse=True)
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
