#!/usr/bin/env python3
"""Verify monthly 5% strategy runtime state against the research spec."""

import argparse
import json
import time
from pathlib import Path

import monthly5_shadow
from verify_monthly5_candidate import _failures as candidate_failures


DEFAULT_SPEC = Path("docs/strategy_specs/monthly5_postlock_hourly.json")
DEFAULT_POSITION = Path(".runtime/data/docs/position.json")
DEFAULT_SHADOW = Path(".runtime/data/btcusdt_monthly5_shadow_state.json")


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"failed to read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"expected object JSON: {path}")
    return payload


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if numeric != numeric:
        return default
    return numeric


def _shadow_from_position(position: dict) -> dict:
    shadow = position.get("monthly5_shadow")
    return shadow if isinstance(shadow, dict) else {}


def _require(condition: bool, failures: list[str], message: str):
    if not condition:
        failures.append(message)


def _verify_spec_and_summary(spec: dict) -> list[str]:
    evidence = spec.get("backtest_evidence") if isinstance(spec.get("backtest_evidence"), dict) else {}
    summary_path = Path(str(evidence.get("source_summary") or ""))
    summary = _load_json(summary_path)
    failures = candidate_failures(summary, spec)
    policy = spec.get("policy") if isinstance(spec.get("policy"), dict) else {}
    risk = policy.get("risk") if isinstance(policy.get("risk"), dict) else {}
    objective = spec.get("objective") if isinstance(spec.get("objective"), dict) else {}

    _require(spec.get("strategy_id") == monthly5_shadow.STRATEGY_ID, failures, "strategy_id does not match code")
    _require(
        str(evidence.get("candidate_name") or "") == monthly5_shadow.SELECTED_CANDIDATE,
        failures,
        "selected candidate does not match code",
    )
    _require(_safe_float(objective.get("monthly_return_floor_pct")) == 5.0, failures, "monthly objective is not 5%")
    _require(int(objective.get("max_leverage", 0)) <= 5, failures, "spec leverage exceeds 5x")
    _require(
        _safe_float(risk.get("monthly_lock_pct")) == monthly5_shadow.MONTHLY_LOCK_PCT,
        failures,
        "monthly lock pct drift",
    )
    _require(
        _safe_float(risk.get("monthly_recovery_trigger_pct")) == monthly5_shadow.MONTHLY_RECOVERY_TRIGGER_PCT,
        failures,
        "monthly recovery trigger drift",
    )
    _require(
        _safe_float(risk.get("intraday_stop_pct")) == monthly5_shadow.INTRADAY_STOP_PCT,
        failures,
        "intraday stop pct drift",
    )
    _require(
        _safe_float(risk.get("post_lock_exposure_scale")) == monthly5_shadow.POST_LOCK_EXPOSURE_SCALE,
        failures,
        "post-lock exposure scale drift",
    )
    return failures


def _verify_shadow_state(name: str, shadow: dict, spec: dict, max_age_sec: float | None) -> list[str]:
    failures: list[str] = []
    evidence = spec.get("backtest_evidence") if isinstance(spec.get("backtest_evidence"), dict) else {}
    _require(bool(shadow), failures, f"{name} missing monthly5_shadow")
    if not shadow:
        return failures
    _require(shadow.get("strategy_id") == spec.get("strategy_id"), failures, f"{name} strategy_id mismatch")
    _require(shadow.get("selected_candidate") == evidence.get("candidate_name"), failures, f"{name} candidate mismatch")
    _require(shadow.get("shadow_only") is True, failures, f"{name} must remain shadow_only")
    _require(_safe_float(shadow.get("max_leverage")) <= 5.0, failures, f"{name} leverage exceeds 5x")
    _require(_safe_float(shadow.get("monthly_lock_pct")) == monthly5_shadow.MONTHLY_LOCK_PCT, failures, f"{name} lock pct drift")
    _require(
        _safe_float(shadow.get("monthly_recovery_trigger_pct")) == monthly5_shadow.MONTHLY_RECOVERY_TRIGGER_PCT,
        failures,
        f"{name} recovery trigger drift",
    )
    _require(_safe_float(shadow.get("intraday_stop_pct")) == monthly5_shadow.INTRADAY_STOP_PCT, failures, f"{name} intraday stop drift")
    _require(
        _safe_float(shadow.get("post_lock_exposure_scale")) == monthly5_shadow.POST_LOCK_EXPOSURE_SCALE,
        failures,
        f"{name} post-lock exposure drift",
    )
    _require(
        _safe_float(shadow.get("recovery_exposure_scale")) == monthly5_shadow.RECOVERY_EXPOSURE_SCALE,
        failures,
        f"{name} recovery exposure drift",
    )
    selection = shadow.get("market_selection") if isinstance(shadow.get("market_selection"), dict) else {}
    _require(bool(selection), failures, f"{name} missing market_selection")
    if selection:
        _require(_safe_float(selection.get("max_leverage")) <= 5.0, failures, f"{name} selection leverage exceeds 5x")
        _require(0.0 <= _safe_float(selection.get("exposure_cap"), -1.0) <= 1.0, failures, f"{name} exposure cap out of range")
        _require(str(selection.get("selected_plan") or ""), failures, f"{name} selected_plan missing")
        _require(str(selection.get("shadow_action") or ""), failures, f"{name} shadow_action missing")
    if max_age_sec is not None:
        age = time.time() - _safe_float(shadow.get("updated_ts"), 0.0)
        _require(age <= max_age_sec, failures, f"{name} shadow state stale: age={age:.1f}s > {max_age_sec:.1f}s")
    return failures


def _verify_guard_scenarios() -> list[str]:
    failures: list[str] = []
    risk_off = {
        "mode": "post_lock_floor_guard",
        "max_leverage": 5,
        "market_selection": {
            "selected_plan": "risk_off",
            "shadow_action": "risk_off",
            "exposure_cap": 0.0,
        },
    }
    execution = monthly5_shadow.build_execution_guard(risk_off, direction="long", requested_size=0.5)
    _require(not execution.get("allowed"), failures, "execution guard must block risk_off")
    _require(_safe_float(execution.get("adjusted_size")) == 0.0, failures, "risk_off execution size must be 0")

    post_lock = {
        "mode": "post_lock",
        "max_leverage": 5,
        "market_selection": {
            "selected_plan": "post_lock_low_exposure",
            "shadow_action": "reduced_exposure",
            "exposure_cap": monthly5_shadow.POST_LOCK_EXPOSURE_SCALE,
        },
    }
    execution = monthly5_shadow.build_execution_guard(post_lock, direction="long", requested_size=0.5)
    _require(bool(execution.get("allowed")), failures, "post-lock execution should remain allowed")
    _require(
        _safe_float(execution.get("adjusted_size")) == monthly5_shadow.POST_LOCK_EXPOSURE_SCALE,
        failures,
        "post-lock execution guard must cap to low exposure",
    )
    position = monthly5_shadow.build_position_guard(post_lock, current_size=0.5)
    _require(position.get("action") == "reduce_to_cap", failures, "post-lock position guard must reduce to cap")
    _require(
        _safe_float(position.get("target_size")) == monthly5_shadow.POST_LOCK_EXPOSURE_SCALE,
        failures,
        "post-lock position target must equal exposure scale",
    )
    position = monthly5_shadow.build_position_guard(risk_off, current_size=0.5)
    _require(position.get("action") == "close_all", failures, "risk_off position guard must close all")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--position", default=str(DEFAULT_POSITION))
    parser.add_argument("--shadow", default=str(DEFAULT_SHADOW))
    parser.add_argument("--max-age-sec", type=float, default=None)
    args = parser.parse_args()

    spec = _load_json(Path(args.spec))
    position = _load_json(Path(args.position))
    shadow = _load_json(Path(args.shadow))

    failures = []
    failures.extend(_verify_spec_and_summary(spec))
    failures.extend(_verify_shadow_state("position", _shadow_from_position(position), spec, args.max_age_sec))
    failures.extend(_verify_shadow_state("shadow_file", shadow, spec, args.max_age_sec))
    failures.extend(_verify_guard_scenarios())

    if failures:
        for item in failures:
            print(f"FAIL {item}")
        return 1

    position_shadow = _shadow_from_position(position)
    selection = position_shadow.get("market_selection") if isinstance(position_shadow.get("market_selection"), dict) else {}
    print(
        "PASS monthly5_runtime "
        f"strategy_id={position_shadow.get('strategy_id')} "
        f"mode={position_shadow.get('mode')} "
        f"selected_plan={selection.get('selected_plan')} "
        f"exposure_cap={selection.get('exposure_cap')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
