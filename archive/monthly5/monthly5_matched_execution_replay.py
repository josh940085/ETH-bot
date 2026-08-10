"""Replay the causal month-held selector with executable hourly and 5m paths."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import monthly5_intraday_regime as regime_summary
import monthly5_monthly_regime_selector as month_selector
import monthly5_regime_selector
import monthly5_selector_cache


DEFAULT_SELECTOR_REPORT = month_selector.DEFAULT_OUTPUT
DEFAULT_OUTPUT = Path(
    ".runtime/data/backtests/monthly5_search/"
    "causal_month_held_matched_execution_2020_202607.json"
)
ONE_WAY_FEE = 0.0004
ACCOUNT_OVERLAYS = (
    {
        "name": "candidate_native",
        "monthly_hard_stop": None,
        "target": None,
        "reduced_leverage": None,
        "floor_guard": None,
    },
    {
        "name": "monthly_stop8_native_target",
        "monthly_hard_stop": -0.08,
        "target": None,
        "reduced_leverage": None,
        "floor_guard": None,
    },
    {
        "name": "monthly_stop8_target5_red0.5",
        "monthly_hard_stop": -0.08,
        "target": 0.05,
        "reduced_leverage": 0.5,
        "floor_guard": None,
    },
    {
        "name": "monthly_stop8_lock5.5_red0.15_guard5.1",
        "monthly_hard_stop": -0.08,
        "target": 0.055,
        "reduced_leverage": 0.15,
        "floor_guard": 0.051,
    },
)
BASE_OVERLAYS = tuple(
    {
        **account,
        "name": f"{account['name']}__{direction_policy}",
        "direction_policy": direction_policy,
    }
    for account in ACCOUNT_OVERLAYS
    for direction_policy in (
        "selected_signal",
        "completed_4h_gate",
        "completed_4h_force",
        "completed_4h_specialist_ma24_96",
        "completed_4h_specialist_ma72_240",
        "completed_4h_specialist_mom480",
        "completed_4h_specialist_don480",
    )
)
RECOVERY_OVERLAYS = tuple(
    {
        **ACCOUNT_OVERLAYS[-1],
        "name": (
            "monthly_stop8_lock5.5_red0.15_guard5.1"
            f"__completed_4h_gate__recovery_{recovery_policy}"
            f"_trigger{trigger}_lev{recovery_leverage}"
        ),
        "direction_policy": "completed_4h_gate",
        "recovery_policy": recovery_policy,
        "recovery_trigger": trigger,
        "recovery_leverage": recovery_leverage,
        "recovery_exit": 0.0,
    }
    for recovery_policy in (
        "inverse_selected",
        "completed_4h_force",
        "specialist_ma24_96",
        "specialist_mom480",
    )
    for trigger in (-0.02, -0.04, -0.06)
    for recovery_leverage in (0.25, 0.5, 1.0)
)
OVERLAYS = BASE_OVERLAYS + RECOVERY_OVERLAYS


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_candidate(key):
    parts = str(key).split("|")
    if len(parts) < 2:
        raise ValueError(f"unsupported candidate key: {key}")
    result = {
        "key": str(key),
        "family": parts[0],
        "leverage": None,
        "stop": None,
        "target": None,
        "reduced_leverage": None,
    }
    for part in parts[1:]:
        if part.startswith("lev"):
            result["leverage"] = float(part[3:])
        elif part.startswith("stop"):
            raw = part[4:]
            result["stop"] = None if raw == "None" else float(raw)
        elif part.startswith("target"):
            raw = part[6:]
            result["target"] = None if raw == "None" else float(raw)
        elif part.startswith("redlev"):
            result["reduced_leverage"] = float(part[6:])
    if result["leverage"] is None or not 0.0 < result["leverage"] <= 5.0:
        raise ValueError(f"candidate leverage must be within 5x: {key}")
    if result["family"].startswith("ensemble_"):
        raise ValueError(f"ensemble candidate execution is not implemented: {key}")
    return result


def selected_strategy_by_month(selector_report):
    rows = selector_report.get("full", {}).get("monthly", [])
    selected = {
        str(row["month"]): str(row["strategy"])
        for row in rows
        if isinstance(row, dict) and row.get("month") and row.get("strategy")
    }
    if not selected:
        raise ValueError("selector report has no selected monthly strategies")
    return selected


def aggregate_hourly(frame_5m):
    return (
        frame_5m.resample("1h", label="right", closed="right")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
    )


def build_hourly_signals(hourly, candidate_keys):
    signals = {}
    for key in sorted(set(candidate_keys)):
        spec = parse_candidate(key)
        signals[key] = monthly5_selector_cache.build_signal(hourly, spec["family"])
    return signals


def map_signals_to_frame(frame, hourly_signals):
    result = {}
    union = frame.index
    for key, signal in hourly_signals.items():
        expanded = signal.reindex(signal.index.union(union)).sort_index().ffill()
        result[key] = expanded.reindex(union).fillna(0.0).to_numpy(dtype="float64")
    return result


def build_completed_4h_regimes(frame_5m):
    completed = (
        frame_5m.resample("4h", label="right", closed="right")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
    )
    return monthly5_regime_selector.classify_4h_regimes(completed)


def map_regimes_to_frame(frame, completed_regimes):
    expanded = (
        completed_regimes.reindex(completed_regimes.index.union(frame.index))
        .sort_index()
        .ffill()
    )
    return expanded.reindex(frame.index).fillna("unknown").astype(str).to_numpy()


def simulate(
    frame,
    selected_by_month,
    desired_by_key,
    overlay,
    *,
    market_regimes=None,
    specialist_desired=None,
):
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype="float64")
    returns = np.zeros(len(frame), dtype="float64")
    returns[1:] = close[1:] / close[:-1] - 1.0
    month_keys = frame.index.tz_localize(None).to_period("M").astype(str)
    factors = np.ones(len(frame), dtype="float64")
    exposures = np.zeros(len(frame), dtype="float64")
    month_equity = 1.0
    previous_month = None
    previous_exposure = 0.0
    effective_leverage = 0.0
    risk_off = False
    reduced = False
    recovery_active = False

    for index, month in enumerate(month_keys):
        key = selected_by_month.get(str(month))
        if key is None:
            previous_exposure = 0.0
            continue
        spec = parse_candidate(key)
        if month != previous_month:
            month_equity = 1.0
            effective_leverage = float(spec["leverage"])
            risk_off = False
            reduced = False
            recovery_active = False
            previous_month = month

        selected_wanted = float(np.sign(desired_by_key[key][index]))
        wanted = selected_wanted
        market_regime = (
            str(market_regimes[index]) if market_regimes is not None else "unknown"
        )
        direction_policy = str(overlay.get("direction_policy") or "selected_signal")
        if direction_policy == "completed_4h_gate":
            if (market_regime == "up" and wanted < 0.0) or (
                market_regime == "down" and wanted > 0.0
            ):
                wanted = 0.0
        elif direction_policy == "completed_4h_force":
            if market_regime == "up":
                wanted = 1.0
            elif market_regime == "down":
                wanted = -1.0
        elif direction_policy.startswith("completed_4h_specialist_"):
            specialist = direction_policy.removeprefix("completed_4h_specialist_")
            specialist_paths = (specialist_desired or {}).get(specialist)
            if specialist_paths is None:
                raise ValueError(f"specialist path missing: {specialist}")
            if market_regime == "up":
                wanted = max(0.0, float(specialist_paths["long"][index]))
            elif market_regime == "down":
                wanted = min(0.0, float(specialist_paths["short"][index]))
        elif direction_policy != "selected_signal":
            raise ValueError(f"unsupported direction policy: {direction_policy}")
        recovery_policy = str(overlay.get("recovery_policy") or "")
        if recovery_active:
            if recovery_policy == "inverse_selected":
                wanted = -selected_wanted
            elif recovery_policy == "completed_4h_force":
                wanted = {"up": 1.0, "down": -1.0}.get(market_regime, 0.0)
            elif recovery_policy.startswith("specialist_"):
                specialist = recovery_policy.removeprefix("specialist_")
                specialist_paths = (specialist_desired or {}).get(specialist)
                if specialist_paths is None:
                    raise ValueError(f"recovery specialist path missing: {specialist}")
                if market_regime == "up":
                    wanted = max(0.0, float(specialist_paths["long"][index]))
                elif market_regime == "down":
                    wanted = min(0.0, float(specialist_paths["short"][index]))
                else:
                    wanted = -selected_wanted
            else:
                raise ValueError(f"unsupported recovery policy: {recovery_policy}")
        exposure = 0.0 if risk_off else wanted * effective_leverage
        turnover = abs(exposure - previous_exposure)
        factor = max(
            1e-9,
            1.0 + exposure * returns[index] - turnover * ONE_WAY_FEE,
        )
        hard_stop = overlay.get("monthly_hard_stop")
        candidate_stop = spec.get("stop")
        active_stop = hard_stop if hard_stop is not None else candidate_stop
        projected_equity = month_equity * factor
        guard_triggered = False
        if (
            not risk_off
            and active_stop is not None
            and projected_equity <= 1.0 + float(active_stop)
        ):
            projected_equity = 1.0 + float(active_stop)
            factor = max(1e-9, projected_equity / month_equity)
            risk_off = True
            guard_triggered = True

        floor_guard = overlay.get("floor_guard")
        if (
            not risk_off
            and reduced
            and floor_guard is not None
            and projected_equity < 1.0 + float(floor_guard)
        ):
            projected_equity = 1.0 + float(floor_guard)
            factor = max(1e-9, projected_equity / month_equity)
            risk_off = True
            guard_triggered = True
        factors[index] = factor
        exposures[index] = exposure
        month_equity = projected_equity

        target = overlay.get("target")
        if target is None:
            target = spec.get("target")
        reduced_leverage = overlay.get("reduced_leverage")
        if reduced_leverage is None:
            reduced_leverage = spec.get("reduced_leverage")
        if (
            not reduced
            and target is not None
            and reduced_leverage is not None
            and month_equity >= 1.0 + float(target)
        ):
            effective_leverage = min(
                float(spec["leverage"]), float(reduced_leverage)
            )
            reduced = True
            recovery_active = False
        recovery_trigger = overlay.get("recovery_trigger")
        recovery_exit = overlay.get("recovery_exit")
        if (
            recovery_active
            and not reduced
            and recovery_exit is not None
            and month_equity >= 1.0 + float(recovery_exit)
        ):
            recovery_active = False
            effective_leverage = float(spec["leverage"])
        elif (
            not recovery_active
            and not reduced
            and not risk_off
            and recovery_trigger is not None
            and month_equity <= 1.0 + float(recovery_trigger)
        ):
            recovery_active = True
            effective_leverage = min(
                float(spec["leverage"]), float(overlay["recovery_leverage"])
            )
        previous_exposure = 0.0 if guard_triggered else exposure
    return factors, exposures


def summarize(frame, factors, exposures, *, start=None, end=None):
    scales = np.abs(exposures) / 5.0
    summary = regime_summary.summarize_factors(
        frame,
        factors,
        scales,
        start=start,
        end=end,
    )
    monthly = summary.get("monthly", [])
    mask = frame.index >= pd.Timestamp(start, tz="UTC") if start else np.ones(len(frame), dtype="bool")
    if end:
        mask &= frame.index < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    selected_exposures = np.asarray(exposures)[mask]
    return {
        **{key: value for key, value in summary.items() if key != "monthly"},
        "avg_flat_time_pct": round(
            float(np.mean(selected_exposures == 0.0)) * 100.0, 4
        ),
        "max_effective_leverage": round(
            float(np.max(np.abs(selected_exposures))), 4
        ),
        "monthly": monthly,
    }


def compare_monthly(expected_rows, actual_rows):
    expected = {
        str(row["month"]): float(row.get("raw_return_pct", row["return_pct"]))
        for row in expected_rows
    }
    actual = {str(row["month"]): float(row["return_pct"]) for row in actual_rows}
    common = sorted(set(expected) & set(actual))
    differences = np.asarray([actual[month] - expected[month] for month in common])
    return {
        "months": len(common),
        "mean_abs_error_pct": round(float(np.mean(np.abs(differences))), 4),
        "median_abs_error_pct": round(float(np.median(np.abs(differences))), 4),
        "max_abs_error_pct": round(float(np.max(np.abs(differences))), 4),
        "largest_errors": [
            {
                "month": month,
                "expected_pct": round(expected[month], 4),
                "actual_pct": round(actual[month], 4),
                "error_pct": round(actual[month] - expected[month], 4),
            }
            for month in sorted(
                common,
                key=lambda value: abs(actual[value] - expected[value]),
                reverse=True,
            )[:10]
        ],
    }


def _rank(summary):
    return (
        summary["months_ge_5"],
        summary["months_ge_0"],
        summary["min_month_pct"],
        summary["max_drawdown_pct"],
        -summary["avg_flat_time_pct"],
    )


def build_report(frame_5m, selector_report):
    selected = selected_strategy_by_month(selector_report)
    candidate_keys = sorted(set(selected.values()))
    hourly = aggregate_hourly(frame_5m)
    hourly_signals = build_hourly_signals(hourly, candidate_keys)
    hourly_desired = {key: value.to_numpy(dtype="float64") for key, value in hourly_signals.items()}
    five_minute_desired = map_signals_to_frame(frame_5m, hourly_signals)
    specialist_hourly = {
        family: {
            "long": monthly5_selector_cache.build_signal(hourly, f"{family}_lf"),
            "short": monthly5_selector_cache.build_signal(hourly, f"{family}_sf"),
        }
        for family in ("ma24_96", "ma72_240", "mom480", "don480")
    }
    specialist_hourly_arrays = {
        family: {
            direction: signal.to_numpy(dtype="float64")
            for direction, signal in paths.items()
        }
        for family, paths in specialist_hourly.items()
    }
    specialist_5m_arrays = {
        family: map_signals_to_frame(frame_5m, paths)
        for family, paths in specialist_hourly.items()
    }
    completed_regimes = build_completed_4h_regimes(frame_5m)
    hourly_regimes = map_regimes_to_frame(hourly, completed_regimes)
    five_minute_regimes = map_regimes_to_frame(frame_5m, completed_regimes)
    variants = []
    computed = []
    for overlay in OVERLAYS:
        hourly_factors, hourly_exposure = simulate(
            hourly,
            selected,
            hourly_desired,
            overlay,
            market_regimes=hourly_regimes,
            specialist_desired=specialist_hourly_arrays,
        )
        factors_5m, exposure_5m = simulate(
            frame_5m,
            selected,
            five_minute_desired,
            overlay,
            market_regimes=five_minute_regimes,
            specialist_desired=specialist_5m_arrays,
        )
        development = summarize(
            frame_5m,
            factors_5m,
            exposure_5m,
            start="2020-01-01",
            end="2023-12-31",
        )
        variants.append(
            {
                "overlay": overlay,
                "development_5m": {
                    key: value for key, value in development.items() if key != "monthly"
                },
            }
        )
        computed.append(
            (hourly_factors, hourly_exposure, factors_5m, exposure_5m)
        )
    winner_index = max(
        range(len(variants)), key=lambda index: _rank(variants[index]["development_5m"])
    )
    winner = variants[winner_index]
    _, _, factors_5m, exposure_5m = computed[winner_index]
    compatibility_factors, compatibility_exposure, _, _ = computed[0]
    hourly_full = summarize(
        hourly,
        compatibility_factors,
        compatibility_exposure,
        start="2020-01-01",
    )
    full_5m = summarize(frame_5m, factors_5m, exposure_5m, start="2020-01-01")
    holdout_5m = summarize(
        frame_5m, factors_5m, exposure_5m, start="2024-01-01"
    )
    comparison = compare_monthly(
        selector_report["full"]["monthly"], hourly_full["monthly"]
    )
    return {
        "schema_version": 1,
        "method": "month_held_candidate_completed_1h_signal_hourly_and_5m_execution",
        "inputs": {
            "selector_report": str(DEFAULT_SELECTOR_REPORT),
            "kline_source": frame_5m.attrs.get("kline_source", "binance_history_um"),
            "five_minute_bars": len(frame_5m),
            "hourly_bars": len(hourly),
            "candidate_count": len(candidate_keys),
            "candidate_keys": candidate_keys,
            "max_leverage": 5,
            "one_way_fee": ONE_WAY_FEE,
        },
        "selection": {
            "development_period": "2020-01..2023-12",
            "holdout_start": "2024-01",
            "winner_overlay": winner["overlay"],
            "variants": variants,
        },
        "hourly_compatibility": {
            "full": hourly_full,
            "matrix_comparison": comparison,
        },
        "five_minute_execution": {
            "full": full_5m,
            "holdout": holdout_5m,
        },
        "shadow_only": True,
        "deployment_ready": False,
        "deployment_blockers": [
            "monthly_return_target_not_consistent",
            "flat_time_target_not_met",
            "recovery_holdout_consistency_not_met",
            "hourly_matrix_replay_error_not_zero",
            "candidate_matrix_evaluation_period_reused_during_research",
            "live_shadow_promotion_not_met",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", default=str(DEFAULT_SELECTOR_REPORT))
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-08-03")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    selector_report = _load_json(args.selector)
    frame = monthly5_selector_cache.load_history(args.start, args.end)
    report = build_report(frame, selector_report)
    report["inputs"]["selector_report"] = str(args.selector)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
