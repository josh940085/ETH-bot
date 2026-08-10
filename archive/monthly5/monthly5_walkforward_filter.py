"""Causally select a daily-confirmation filter at each month boundary."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import monthly5_daily_filter_research as daily_filter
import monthly5_intraday_regime as regime
import monthly5_selector_cache


LOOKBACKS = (6, 12, 24, 36)
SCORE_MODES = ("balanced", "tail")
TRAIN_END = "2022-12-31"
VALIDATION_START = "2023-01-01"
VALIDATION_END = "2023-12-31"
DEFAULT_OUTPUT = Path(
    ".runtime/data/backtests/monthly5_search/walkforward_daily_filter_v1_2020_20260803.json"
)


def build_candidate_paths(frame_5m):
    rows = []
    for config in daily_filter.FILTER_CONFIGS:
        factors, scales = daily_filter.apply_components(
            frame_5m,
            daily_filter.build_components(frame_5m, config),
        )
        rows.append(
            {
                "name": config["name"],
                "config": config,
                "factors": np.asarray(factors),
                "scales": np.asarray(scales),
            }
        )
    return rows


def monthly_matrix(frame_5m, candidates):
    month_keys = frame_5m.index.tz_localize(None).to_period("M")
    months = np.asarray(month_keys.unique().astype(str))
    matrix = np.zeros((len(candidates), len(months)), dtype="float64")
    for row_index, row in enumerate(candidates):
        series = pd.Series(row["factors"], index=month_keys)
        monthly = series.groupby(level=0).prod() - 1.0
        matrix[row_index] = monthly.reindex(pd.PeriodIndex(months, freq="M")).to_numpy()
    return months, matrix


def candidate_scores(history, mode):
    hit_rate = np.mean(history >= 0.05, axis=1)
    nonnegative = np.mean(history >= 0.0, axis=1)
    mean = np.mean(history, axis=1)
    q25 = np.percentile(history, 25, axis=1)
    minimum = np.min(history, axis=1)
    if mode == "tail":
        return q25 + 0.25 * minimum + 0.05 * hit_rate + 0.01 * nonnegative
    return mean + 0.50 * q25 + 0.05 * hit_rate + 0.01 * nonnegative


def run_selector(
    frame_5m,
    candidates,
    *,
    lookback_months,
    score_mode,
    switch_round_trip_fee=0.0008,
):
    months, returns = monthly_matrix(frame_5m, candidates)
    selected = np.zeros(len(months), dtype="int32")
    warmup = max(1, int(lookback_months))
    for target in range(warmup, len(months)):
        start = max(0, target - int(lookback_months))
        selected[target] = int(np.argmax(candidate_scores(returns[:, start:target], score_mode)))

    month_keys = frame_5m.index.tz_localize(None).to_period("M").astype(str).to_numpy()
    month_lookup = {month: index for index, month in enumerate(months)}
    factors = np.ones(len(frame_5m), dtype="float64")
    scales = np.zeros(len(frame_5m), dtype="float64")
    selections = []
    previous = None
    for month in months:
        month_index = month_lookup[month]
        candidate_index = int(selected[month_index])
        mask = month_keys == month
        factors[mask] = candidates[candidate_index]["factors"][mask]
        scales[mask] = candidates[candidate_index]["scales"][mask]
        switched = previous is not None and candidate_index != previous
        first_bar = int(np.flatnonzero(mask)[0])
        if switched:
            factors[first_bar] *= max(1e-9, 1.0 - float(switch_round_trip_fee))
        selections.append(
            {
                "month": month,
                "candidate": candidates[candidate_index]["name"],
                "switched": switched,
                "history_start": months[max(0, month_index - int(lookback_months))]
                if month_index
                else "",
                "history_end": months[month_index - 1] if month_index else "",
            }
        )
        previous = candidate_index
    return factors, scales, selections


def select_validation_winner(candidates):
    return max(
        candidates,
        key=lambda row: (
            regime.development_rank(row["validation"]),
            regime.development_rank(row["training"]),
        ),
    )


def verify_prefix_stability(frame_5m, config, full_factors):
    checks = []
    for cutoff in ("2021-12-31", "2023-12-31", "2025-12-31"):
        cutoff_ts = pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        truncated = frame_5m.loc[frame_5m.index <= cutoff_ts]
        factors, _, _ = run_selector(
            truncated,
            build_candidate_paths(truncated),
            lookback_months=config["lookback_months"],
            score_mode=config["score_mode"],
        )
        stable = np.array_equal(np.asarray(full_factors)[: len(truncated)], factors)
        checks.append({"cutoff": cutoff, "bars": len(truncated), "stable": bool(stable)})
    return {
        "prefix_stable": all(row["stable"] for row in checks),
        "recursive_stable": all(row["stable"] for row in checks),
        "checks": checks,
    }


def _without_monthly(summary):
    return {key: value for key, value in summary.items() if key != "monthly"}


def build_report(frame_5m):
    paths = build_candidate_paths(frame_5m)
    variants = []
    for lookback in LOOKBACKS:
        for score_mode in SCORE_MODES:
            factors, scales, selections = run_selector(
                frame_5m,
                paths,
                lookback_months=lookback,
                score_mode=score_mode,
            )
            variants.append(
                {
                    "name": f"lb{lookback}_{score_mode}",
                    "config": {"lookback_months": lookback, "score_mode": score_mode},
                    "development": regime.summarize_factors(
                        frame_5m,
                        factors,
                        scales,
                        start=regime.REPORT_START,
                        end=regime.DEVELOPMENT_END,
                    ),
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
    winner = select_validation_winner(variants)
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
        "method": "month_boundary_walkforward_daily_filter_selection",
        "selection_uses_holdout": False,
        "hyperparameter_training_period": f"{regime.REPORT_START}..{TRAIN_END}",
        "hyperparameter_validation_period": f"{VALIDATION_START}..{VALIDATION_END}",
        "candidate_filters": [row["name"] for row in paths],
        "variant_count": len(variants),
        "winner": {"name": winner["name"], "config": winner["config"]},
        "development": winner["development"],
        "training": winner["training"],
        "validation": winner["validation"],
        "holdout": holdout,
        "full": full,
        "monthly_selections": winner["_selections"],
        "variants": [
            {
                "name": row["name"],
                "config": row["config"],
                "development": _without_monthly(row["development"]),
                "training": _without_monthly(row["training"]),
                "validation": _without_monthly(row["validation"]),
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
