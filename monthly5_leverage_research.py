"""Research risk-equivalent leverage for the causal completed-4h monthly5 policy."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import monthly5_intraday_regime as regime
import monthly5_intramonth_recovery_research as account
import monthly5_regime_specialist_research as specialist
import monthly5_selector_cache
import monthly5_volatility_walkforward as walkforward


LEVERAGE_VALUES = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
ACCOUNT_CONFIG = {
    "name": "baseline_no_recovery",
    "mode": "none",
    "trigger": None,
    "scale": 0.0,
    "exit": 0.0,
}
DEFAULT_OUTPUT = Path(
    ".runtime/data/backtests/monthly5_search/leverage_v1_2020_20260803.json"
)


def evaluate_leverage(frame_5m, leverage):
    desired, strategy_ids, _ = walkforward.build_primary_path(
        frame_5m, **account.PRIMARY_CONFIG
    )
    return account.simulate_account_path(
        frame_5m,
        desired,
        strategy_ids,
        desired,
        ACCOUNT_CONFIG,
        leverage=leverage,
    )


def _without_monthly(summary):
    return {key: value for key, value in summary.items() if key != "monthly"}


def verify_prefix_stability(frame_5m, leverage, full_factors):
    checks = []
    for cutoff in ("2021-12-31", "2023-12-31", "2025-12-31"):
        cutoff_ts = pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        truncated = frame_5m.loc[frame_5m.index <= cutoff_ts]
        factors, _, _, _, _ = evaluate_leverage(truncated, leverage)
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
    candidates = []
    for leverage in LEVERAGE_VALUES:
        factors, scales, _, _, positions = account.simulate_account_path(
            frame_5m,
            desired,
            strategy_ids,
            desired,
            ACCOUNT_CONFIG,
            leverage=leverage,
        )
        candidates.append(
            {
                "name": f"leverage_{leverage:g}x",
                "leverage": leverage,
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
                "actual_exposure_pct": round(float(np.mean(positions != 0.0)) * 100.0, 4),
                "average_absolute_leverage": round(float(np.mean(np.abs(positions))), 4),
                "_factors": factors,
                "_scales": scales,
            }
        )
    winner = max(candidates, key=specialist.selection_rank)
    evaluation = winner["evaluation"]
    full = regime.summarize_factors(
        frame_5m, winner["_factors"], winner["_scales"], start=regime.REPORT_START
    )
    prefix = verify_prefix_stability(frame_5m, winner["leverage"], winner["_factors"])
    target_pass = evaluation["months_ge_5"] == evaluation["months"]
    return {
        "schema_version": 1,
        "method": "risk_equivalent_uniform_leverage_completed_4h_policy",
        "source": frame_5m.attrs.get("kline_source", "binance_history_um"),
        "primary_config": account.PRIMARY_CONFIG,
        "primary_monthly_selections": selections,
        "selection_uses_evaluation_period": False,
        "evaluation_period_reused_during_research": True,
        "candidate_count": len(candidates),
        "winner": {
            "name": winner["name"],
            "leverage": winner["leverage"],
            "actual_exposure_pct": winner["actual_exposure_pct"],
            "flat_time_pct": round(100.0 - winner["actual_exposure_pct"], 4),
            "average_absolute_leverage": winner["average_absolute_leverage"],
        },
        "training": winner["training"],
        "validation": winner["validation"],
        "development": winner["development"],
        "evaluation": evaluation,
        "full": full,
        "ranking": [
            {
                "name": row["name"],
                "leverage": row["leverage"],
                "training": _without_monthly(row["training"]),
                "validation": _without_monthly(row["validation"]),
                "evaluation_diagnostic_only": _without_monthly(row["evaluation"]),
                "actual_exposure_pct": row["actual_exposure_pct"],
                "average_absolute_leverage": row["average_absolute_leverage"],
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
