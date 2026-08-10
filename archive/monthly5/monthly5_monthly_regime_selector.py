"""Causal month-held monthly5 selector using completed 4h market regimes."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MATRIX = Path(
    ".runtime/data/backtests/monthly5_search/"
    "focused_monthly5_grid_all_2020_20260804_monthly.json"
)
DEFAULT_RANGE_REPORT = Path(
    ".runtime/data/backtests/monthly5_search/"
    "range_mean_reversion_v1_2020_20260803.json"
)
DEFAULT_REGIME_REPORT = Path(
    ".runtime/data/backtests/monthly5_search/"
    "causal_4h_regime_selector_2020_20260803.json"
)
DEFAULT_OUTPUT = Path(
    ".runtime/data/backtests/monthly5_search/"
    "causal_month_held_4h_regime_selector_2020_202607.json"
)
DEVELOPMENT_END = "2023-12"
HOLDOUT_START = "2024-01"
MONTHLY_TARGET = 0.05
MONTHLY_SCREENING_FLOOR = -0.08
REGIMES = ("up", "range", "down")
CONFIGS = tuple(
    {
        "use_regime": use_regime,
        "min_regime_months": min_regime_months,
        "lookback_months": lookback_months,
        "q25_weight": q25_weight,
        "hit_weight": hit_weight,
        "volatility_weight": volatility_weight,
    }
    for use_regime in (False, True)
    for min_regime_months in (3, 6, 9, 12)
    for lookback_months in (12, 24, 36, 48, 72)
    for q25_weight in (0.0, 0.5, 1.0)
    for hit_weight in (0.0, 1.0, 2.0)
    for volatility_weight in (0.25, 0.5, 1.0, 1.5)
)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_leverage(key):
    if key.startswith("range_"):
        return 1
    marker = "|lev"
    if marker not in key:
        raise ValueError(f"candidate leverage missing: {key}")
    return int(key.split(marker, 1)[1].split("|", 1)[0])


def load_candidate_matrix(matrix_path=DEFAULT_MATRIX, range_path=DEFAULT_RANGE_REPORT):
    payload = _load_json(matrix_path)
    if not payload:
        raise ValueError("monthly candidate matrix is empty")
    keys = list(payload)
    month_sets = [
        {row["month"] for row in payload[key] if row["month"] < "2026-08"}
        for key in keys
    ]
    months = sorted(set.intersection(*month_sets))
    if not months:
        raise ValueError("monthly candidate matrix has no common complete months")
    returns = []
    flats = []
    for key in keys:
        rows = {row["month"]: row for row in payload[key]}
        returns.append([float(rows[month]["return_pct"]) / 100.0 for month in months])
        flats.append([float(rows[month]["flat_time_pct"]) / 100.0 for month in months])

    range_report = _load_json(range_path)
    range_rows = {row["month"]: row for row in range_report["full"]["monthly"]}
    if all(month in range_rows for month in months):
        keys.append(str(range_report["winner"]["name"]))
        returns.append(
            [float(range_rows[month]["return_pct"]) / 100.0 for month in months]
        )
        range_flat = 1.0 - float(range_report["winner"]["actual_exposure_pct"]) / 100.0
        flats.append([range_flat] * len(months))

    leverages = np.asarray([_parse_leverage(key) for key in keys], dtype="int32")
    if np.max(leverages) > 5:
        raise ValueError("candidate matrix exceeds 5x leverage cap")
    return {
        "keys": np.asarray(keys),
        "months": np.asarray(months),
        "returns": np.asarray(returns, dtype="float64"),
        "flats": np.asarray(flats, dtype="float64"),
        "leverages": leverages,
    }


def month_start_regimes(months, regime_report_path=DEFAULT_REGIME_REPORT):
    intervals = _load_json(regime_report_path)["regime_definition"]["all_intervals"]
    starts = np.asarray([pd.Timestamp(row["start"]).value for row in intervals])
    ends = np.asarray([pd.Timestamp(row["end"]).value for row in intervals])
    states = np.asarray([str(row["regime"]) for row in intervals])
    result = []
    for month in np.asarray(months).astype(str):
        boundary = pd.Timestamp(f"{month}-01", tz="UTC").value
        containing = np.flatnonzero((starts <= boundary) & (ends >= boundary))
        if len(containing):
            result.append(states[containing[-1]])
            continue
        prior = np.flatnonzero(ends <= boundary)
        result.append(states[prior[-1]] if len(prior) else "unknown")
    return np.asarray(result)


def _candidate_scores(candidate_returns, config):
    mean = np.mean(candidate_returns, axis=1)
    q25 = np.percentile(candidate_returns, 25, axis=1)
    hit_rate = np.mean(candidate_returns >= MONTHLY_TARGET, axis=1)
    volatility = np.std(candidate_returns, axis=1)
    loss_rate = np.mean(candidate_returns < 0.0, axis=1)
    return (
        mean
        + float(config["q25_weight"]) * q25
        + float(config["hit_weight"]) * hit_rate * MONTHLY_TARGET
        - float(config["volatility_weight"]) * volatility
        - loss_rate * 0.005
    )


def run_causal_selector(matrix, regimes, config, *, cold_start_index=0):
    raw_returns = np.asarray(matrix["returns"], dtype="float64")
    screened_returns = np.maximum(raw_returns, MONTHLY_SCREENING_FLOOR)
    month_count = len(matrix["months"])
    selected = np.full(month_count, int(cold_start_index), dtype="int32")
    fallback = np.zeros(month_count, dtype="bool")
    training_counts = np.zeros(month_count, dtype="int32")

    for target in range(month_count):
        start = max(0, target - int(config["lookback_months"]))
        history = np.arange(start, target, dtype="int32")
        if bool(config["use_regime"]) and regimes[target] in REGIMES:
            matching = history[regimes[history] == regimes[target]]
            if len(matching) >= int(config["min_regime_months"]):
                history = matching
            elif len(history):
                fallback[target] = True
        if len(history) < 3:
            fallback[target] = True
            continue
        scores = _candidate_scores(screened_returns[:, history], config)
        selected[target] = int(np.nanargmax(scores))
        training_counts[target] = len(history)

    columns = np.arange(month_count)
    return {
        "selected_indices": selected,
        "returns": screened_returns[selected, columns],
        "raw_returns": raw_returns[selected, columns],
        "flats": matrix["flats"][selected, columns],
        "training_counts": training_counts,
        "fallback": fallback,
    }


def summarize(matrix, regimes, result, *, start=None, end=None, include_monthly=False):
    months = np.asarray(matrix["months"]).astype(str)
    mask = np.ones(len(months), dtype="bool")
    if start:
        mask &= months >= str(start)
    if end:
        mask &= months <= str(end)
    values = result["returns"][mask]
    flats = result["flats"][mask]
    picks = result["selected_indices"][mask]
    equity = np.cumprod(1.0 + values)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    summary = {
        "start": str(months[mask][0]),
        "end": str(months[mask][-1]),
        "months": int(len(values)),
        "months_ge_5": int(np.sum(values >= MONTHLY_TARGET)),
        "months_ge_0": int(np.sum(values >= 0.0)),
        "min_month_pct": round(float(np.min(values)) * 100.0, 4),
        "avg_month_pct": round(float(np.mean(values)) * 100.0, 4),
        "total_return_pct": round(float(np.prod(1.0 + values) - 1.0) * 100.0, 4),
        "max_drawdown_pct": round(float(np.min(drawdown)) * 100.0, 4),
        "avg_flat_time_pct": round(float(np.mean(flats)) * 100.0, 4),
        "fallback_months": int(np.sum(result["fallback"][mask])),
        "max_selected_leverage": int(np.max(matrix["leverages"][picks])),
    }
    if include_monthly:
        summary["monthly"] = [
            {
                "month": str(month),
                "regime": str(regime),
                "strategy": str(matrix["keys"][pick]),
                "return_pct": round(float(value) * 100.0, 4),
                "raw_return_pct": round(float(raw) * 100.0, 4),
                "flat_time_pct": round(float(flat) * 100.0, 4),
                "screening_floor_applied": bool(raw < MONTHLY_SCREENING_FLOOR),
            }
            for month, regime, pick, value, raw, flat in zip(
                months[mask],
                regimes[mask],
                picks,
                values,
                result["raw_returns"][mask],
                flats,
            )
        ]
    return summary


def _rank(summary):
    return (
        summary["months_ge_5"],
        summary["months_ge_0"],
        summary["min_month_pct"],
        summary["max_drawdown_pct"],
        summary["avg_month_pct"],
        -summary["avg_flat_time_pct"],
    )


def verify_prefix_stability(matrix, regimes, config, full_result):
    checks = []
    months = np.asarray(matrix["months"]).astype(str)
    for cutoff in ("2021-12", "2023-12", "2025-12"):
        count = int(np.sum(months <= cutoff))
        sliced = {
            key: value[:, :count] if key in {"returns", "flats"} else value
            for key, value in matrix.items()
        }
        sliced["months"] = matrix["months"][:count]
        replay = run_causal_selector(sliced, regimes[:count], config)
        stable = np.array_equal(
            replay["selected_indices"], full_result["selected_indices"][:count]
        )
        checks.append({"cutoff": cutoff, "months": count, "stable": bool(stable)})
    return {
        "prefix_stable": all(row["stable"] for row in checks),
        "recursive_stable": all(row["stable"] for row in checks),
        "checks": checks,
    }


def build_report(
    matrix_path=DEFAULT_MATRIX,
    range_path=DEFAULT_RANGE_REPORT,
    regime_report_path=DEFAULT_REGIME_REPORT,
):
    matrix = load_candidate_matrix(matrix_path, range_path)
    regimes = month_start_regimes(matrix["months"], regime_report_path)
    variants = []
    results = []
    for config in CONFIGS:
        result = run_causal_selector(matrix, regimes, config)
        development = summarize(
            matrix, regimes, result, end=DEVELOPMENT_END, include_monthly=False
        )
        variants.append({"config": config, "development": development})
        results.append(result)
    winner_index = max(range(len(variants)), key=lambda index: _rank(variants[index]["development"]))
    winner = variants[winner_index]
    result = results[winner_index]
    holdout = summarize(matrix, regimes, result, start=HOLDOUT_START, include_monthly=True)
    full = summarize(matrix, regimes, result, include_monthly=True)
    prefix = verify_prefix_stability(matrix, regimes, winner["config"], result)
    regime_counts = {
        regime: int(np.sum(regimes == regime)) for regime in (*REGIMES, "unknown")
    }
    return {
        "schema_version": 1,
        "method": "causal_month_start_completed_4h_regime_month_held_candidate",
        "inputs": {
            "matrix_path": str(matrix_path),
            "range_path": str(range_path),
            "regime_report_path": str(regime_report_path),
            "candidate_count": len(matrix["keys"]),
            "month_count": len(matrix["months"]),
            "max_leverage": int(np.max(matrix["leverages"])),
            "regime_counts": regime_counts,
        },
        "selection": {
            "development_period": f"{matrix['months'][0]}..{DEVELOPMENT_END}",
            "holdout_start": HOLDOUT_START,
            "config_count": len(CONFIGS),
            "winner_config": winner["config"],
            "development": winner["development"],
        },
        "holdout": holdout,
        "full": full,
        "screening_assumptions": {
            "monthly_loss_floor": MONTHLY_SCREENING_FLOOR,
            "loss_floor_application": "clip completed monthly candidate return",
            "candidate_hold_period": "whole UTC calendar month",
            "decision_data_cutoff": "month start using prior months and last completed 4h regime",
        },
        "bias_evidence": prefix,
        **prefix,
        "shadow_only": True,
        "deployment_ready": False,
        "deployment_blockers": [
            "monthly_return_target_not_consistent",
            "monthly_loss_clip_not_matched_5m_execution",
            "candidate_matrix_evaluation_period_reused_during_research",
            "candidate_matched_tick_execution_evidence_missing",
            "live_shadow_promotion_not_met",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--range", dest="range_path", default=str(DEFAULT_RANGE_REPORT))
    parser.add_argument("--regimes", default=str(DEFAULT_REGIME_REPORT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    report = build_report(args.matrix, args.range_path, args.regimes)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
