#!/usr/bin/env python3
"""Research separate completed-4h confirmation speeds for up and down regimes."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import monthly5_intraday_regime as regime
import monthly5_intramonth_recovery_research as account
import monthly5_regime_hysteresis_research as hysteresis
import monthly5_regime_specialist_research as specialist
import monthly5_selector_cache
import monthly5_volume_confirmation_research as volume
import monthly5_volatility_regime_research as volatility
import monthly5_volatility_walkforward as walkforward


UP_CONFIRMATIONS = (1, 2, 3, 4)
DOWN_CONFIRMATIONS = (1, 2, 3, 4)
RANGE_GRACE_BARS = (0, 3, 6)
VOLUME_CONFIG = {"name": "volume_window24_min0.5", "window": 24, "min_ratio": 0.5}
DEFAULT_OUTPUT = Path(
    ".runtime/data/backtests/monthly5_search/asymmetric_regime_v1_2020_20260803.json"
)


def configs():
    return tuple(
        {
            "name": f"up{up}_down{down}_grace{grace}",
            "up_confirmation_bars": up,
            "down_confirmation_bars": down,
            "range_grace_bars": grace,
        }
        for up in UP_CONFIRMATIONS
        for down in DOWN_CONFIRMATIONS
        for grace in RANGE_GRACE_BARS
    )


def build_asymmetric_position(
    frame_5m,
    labels,
    *,
    up_confirmation_bars,
    down_confirmation_bars,
    range_grace_bars,
):
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
            required = (
                max(1, int(up_confirmation_bars))
                if market_regime == "up"
                else max(1, int(down_confirmation_bars))
            )
            if pending_count >= required:
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


def build_candidate_path(frame_5m, primary_ids, config):
    paths = []
    regime_paths = []
    for profile in walkforward.CANDIDATE_CONFIGS:
        labels = volatility.classify_completed_4h(frame_5m, profile["label"])
        desired, aligned = build_asymmetric_position(
            frame_5m,
            labels,
            up_confirmation_bars=config["up_confirmation_bars"],
            down_confirmation_bars=config["down_confirmation_bars"],
            range_grace_bars=config["range_grace_bars"],
        )
        paths.append(desired.to_numpy(dtype="float64"))
        regime_paths.append(aligned)
    selected = np.asarray(primary_ids, dtype="int32")
    indexes = np.arange(len(frame_5m))
    desired = np.vstack(paths)[selected, indexes]
    aligned_regimes = np.vstack(regime_paths)[selected, indexes]
    return desired, aligned_regimes


def evaluate_config(frame_5m, primary_ids, config, entry_allowed):
    desired, aligned_regimes = build_candidate_path(frame_5m, primary_ids, config)
    factors, scales, _, _, positions = account.simulate_account_path(
        frame_5m,
        desired,
        primary_ids,
        desired,
        volume.ACCOUNT_CONFIG,
        entry_allowed=entry_allowed,
    )
    return factors, scales, positions, aligned_regimes


def _summary_without_monthly(summary):
    return {key: value for key, value in summary.items() if key != "monthly"}


def regime_attribution(frame_5m, factors, positions, aligned_regimes, *, start, end=None):
    mask = frame_5m.index >= pd.Timestamp(start, tz="UTC")
    if end is not None:
        mask &= frame_5m.index < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    factor_values = np.asarray(factors, dtype="float64")
    position_values = np.asarray(positions, dtype="float64")
    rows = {}
    for market_regime in ("up", "down", "range", "unknown"):
        selected = mask & (aligned_regimes == market_regime)
        rows[market_regime] = {
            "bars": int(selected.sum()),
            "time_pct": round(float(selected.sum() / max(1, mask.sum())) * 100.0, 4),
            "exposure_pct": round(
                float(np.mean(position_values[selected] != 0.0)) * 100.0
                if selected.any()
                else 0.0,
                4,
            ),
            "return_pct": round(
                float(np.prod(factor_values[selected]) - 1.0) * 100.0
                if selected.any()
                else 0.0,
                4,
            ),
        }
    return rows


def verify_prefix_stability(frame_5m, config, expected_factors):
    checks = []
    for cutoff in ("2021-12-31", "2023-12-31", "2025-12-31"):
        cutoff_ts = pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1)
        truncated = frame_5m.loc[frame_5m.index < cutoff_ts]
        _, primary_ids, _ = walkforward.build_primary_path(
            truncated, **account.PRIMARY_CONFIG
        )
        allowed = volume.build_entry_allowed(truncated, VOLUME_CONFIG)
        factors, _, _, _ = evaluate_config(truncated, primary_ids, config, allowed)
        stable = np.array_equal(
            np.asarray(expected_factors)[: len(truncated)], np.asarray(factors)
        )
        checks.append({"cutoff": cutoff, "bars": len(truncated), "stable": bool(stable)})
    return {
        "prefix_stable": all(row["stable"] for row in checks),
        "recursive_stable": all(row["stable"] for row in checks),
        "checks": checks,
    }


def build_report(frame_5m):
    _, primary_ids, selections = walkforward.build_primary_path(
        frame_5m, **account.PRIMARY_CONFIG
    )
    entry_allowed = volume.build_entry_allowed(frame_5m, VOLUME_CONFIG)
    candidates = []
    for config in configs():
        factors, scales, positions, aligned_regimes = evaluate_config(
            frame_5m, primary_ids, config, entry_allowed
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
                "evaluation": regime.summarize_factors(
                    frame_5m, factors, scales, start=regime.HOLDOUT_START
                ),
                "_factors": factors,
                "_scales": scales,
                "_positions": positions,
                "_regimes": aligned_regimes,
            }
        )
    winner = max(candidates, key=specialist.selection_rank)
    baseline = next(row for row in candidates if row["name"] == "up1_down1_grace6")
    full = regime.summarize_factors(
        frame_5m, winner["_factors"], winner["_scales"], start=regime.REPORT_START
    )
    prefix = verify_prefix_stability(frame_5m, winner["config"], winner["_factors"])
    evaluation = winner["evaluation"]
    baseline_development = regime.summarize_factors(
        frame_5m,
        baseline["_factors"],
        baseline["_scales"],
        start=regime.REPORT_START,
        end=regime.DEVELOPMENT_END,
    )
    baseline_full = regime.summarize_factors(
        frame_5m,
        baseline["_factors"],
        baseline["_scales"],
        start=regime.REPORT_START,
    )
    evaluation_regressed = (
        evaluation["months_ge_5"] < baseline["evaluation"]["months_ge_5"]
        or evaluation["total_return_pct"] < baseline["evaluation"]["total_return_pct"]
    )
    return {
        "schema_version": 1,
        "method": "completed_4h_asymmetric_confirmation_with_frozen_volume_gate",
        "confirmation_timebase": hysteresis.CONFIRMATION_TIMEBASE,
        "source": frame_5m.attrs.get("kline_source", "binance_history_um"),
        "primary_config": account.PRIMARY_CONFIG,
        "primary_monthly_selections": selections,
        "volume_config": VOLUME_CONFIG,
        "selection_uses_evaluation_period": False,
        "evaluation_period_reused_during_research": True,
        "candidate_count": len(candidates),
        "winner": {"name": winner["name"], "config": winner["config"]},
        "training": winner["training"],
        "validation": winner["validation"],
        "development": regime.summarize_factors(
            frame_5m,
            winner["_factors"],
            winner["_scales"],
            start=regime.REPORT_START,
            end=regime.DEVELOPMENT_END,
        ),
        "evaluation": evaluation,
        "full": full,
        "baseline": {
            "name": baseline["name"],
            "config": baseline["config"],
            "training": baseline["training"],
            "validation": baseline["validation"],
            "development": baseline_development,
            "evaluation_diagnostic_only": baseline["evaluation"],
            "full": baseline_full,
        },
        "comparison_vs_baseline": {
            "development_months_ge_5_delta": (
                winner["training"]["months_ge_5"]
                + winner["validation"]["months_ge_5"]
                - baseline["training"]["months_ge_5"]
                - baseline["validation"]["months_ge_5"]
            ),
            "evaluation_months_ge_5_delta_diagnostic_only": (
                evaluation["months_ge_5"] - baseline["evaluation"]["months_ge_5"]
            ),
            "evaluation_total_return_pct_delta_diagnostic_only": round(
                evaluation["total_return_pct"]
                - baseline["evaluation"]["total_return_pct"],
                4,
            ),
            "evaluation_regressed": bool(evaluation_regressed),
        },
        "regime_attribution": {
            "training": regime_attribution(
                frame_5m,
                winner["_factors"],
                winner["_positions"],
                winner["_regimes"],
                start=regime.REPORT_START,
                end=specialist.TRAIN_END,
            ),
            "validation": regime_attribution(
                frame_5m,
                winner["_factors"],
                winner["_positions"],
                winner["_regimes"],
                start=specialist.VALIDATION_START,
                end=specialist.VALIDATION_END,
            ),
            "evaluation_diagnostic_only": regime_attribution(
                frame_5m,
                winner["_factors"],
                winner["_positions"],
                winner["_regimes"],
                start=regime.HOLDOUT_START,
            ),
        },
        "top_selection": [
            {
                "name": row["name"],
                "config": row["config"],
                "training": _summary_without_monthly(row["training"]),
                "validation": _summary_without_monthly(row["validation"]),
                "evaluation_diagnostic_only": _summary_without_monthly(row["evaluation"]),
            }
            for row in sorted(candidates, key=specialist.selection_rank, reverse=True)[:20]
        ],
        "bias_evidence": prefix,
        "deployment_ready": False,
        "deployment_blockers": [
            *(
                []
                if evaluation["months_ge_5"] == evaluation["months"]
                else ["evaluation_all_months_ge_5_failed"]
            ),
            "evaluation_period_reused_during_research",
            *([] if not evaluation_regressed else ["evaluation_regresses_baseline"]),
            *([] if prefix["recursive_stable"] else ["recursive_prefix_rebuild_failed"]),
            "candidate_matched_tick_execution_evidence_missing",
            "forward_shadow_not_justified",
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
