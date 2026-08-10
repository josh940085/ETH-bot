#!/usr/bin/env python3
"""Verify monthly5 is actionable for live takeover after promotion."""

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_POSITION = Path(".runtime/data/docs/position.json")

_BLOCKING_PLANS = {
    "normal_wait",
    "research_selector_wait",
    "recovery_wait_for_long_flat",
    "macro_block_wait",
    "underperforming_wait",
    "profile_quality_wait",
    "risk_off",
}
_PROBE_REASON_CODES = {
    "underperforming_micro_probe",
    "profile_quality_recovery_probe",
    "mixed_bias_shadow_probe",
    "context_grace_shadow_probe",
    "profile_quality_shadow_probe",
}


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"failed to read JSON {path}: {exc}") from exc
    return payload if isinstance(payload, dict) else {}


def _selection(position: dict) -> dict:
    shadow = position.get("monthly5_shadow") if isinstance(position.get("monthly5_shadow"), dict) else {}
    selection = shadow.get("market_selection") if isinstance(shadow.get("market_selection"), dict) else {}
    return selection if isinstance(selection, dict) else {}


def _promotion_ready(position: dict) -> bool:
    readiness = position.get("monthly5_readiness") if isinstance(position.get("monthly5_readiness"), dict) else {}
    shadow = position.get("monthly5_shadow") if isinstance(position.get("monthly5_shadow"), dict) else {}
    source = readiness or shadow
    return bool(source.get("promotion_ready", False))


def _activation_failures(position: dict) -> list[str]:
    if not _promotion_ready(position):
        return []
    selection = _selection(position)
    selected_plan = str(selection.get("selected_plan") or "")
    shadow_action = str(selection.get("shadow_action") or "")
    reason_codes = {str(code) for code in selection.get("reason_codes") or [] if str(code)}
    failures: list[str] = []

    if not selected_plan:
        failures.append("selected_plan missing")
    if selected_plan in _BLOCKING_PLANS or selected_plan.endswith("_probe"):
        failures.append(f"selected_plan not actionable: {selected_plan}")
    blocking_reasons = sorted(reason_codes & _PROBE_REASON_CODES)
    if blocking_reasons:
        failures.append(f"probe reason_codes active: {','.join(blocking_reasons)}")
    if shadow_action not in {"evaluate_long", "evaluate_short", "reduced_exposure"}:
        failures.append(f"shadow_action not actionable: {shadow_action or 'missing'}")

    strategy_signal = str(position.get("strategy_signal") or "").lower()
    strategy_status = str(position.get("strategy_execution_status") or "").lower()
    override = position.get("monthly5_signal_override") if isinstance(position.get("monthly5_signal_override"), dict) else {}
    if strategy_signal == "wait" and strategy_status == "waiting" and shadow_action in {"evaluate_long", "evaluate_short"}:
        if override.get("applied") is not True:
            failures.append(f"waiting state missing applied monthly5 override: {override.get('reason') or 'missing'}")
    return failures


def _summary(position: dict) -> dict[str, Any]:
    selection = _selection(position)
    override = position.get("monthly5_signal_override") if isinstance(position.get("monthly5_signal_override"), dict) else {}
    return {
        "promotion_ready": _promotion_ready(position),
        "selected_plan": selection.get("selected_plan"),
        "shadow_action": selection.get("shadow_action"),
        "reason_codes": list(selection.get("reason_codes") or []),
        "strategy_signal": position.get("strategy_signal"),
        "strategy_execution_status": position.get("strategy_execution_status"),
        "override_applied": override.get("applied"),
        "override_reason": override.get("reason"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position", default=str(DEFAULT_POSITION))
    args = parser.parse_args(argv)

    position = _load_json(Path(args.position))
    summary = _summary(position)
    summary_text = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    if not summary["promotion_ready"]:
        print(f"PASS monthly5_activation status=pending_promotion {summary_text}")
        return 0
    failures = _activation_failures(position)
    if failures:
        print(f"FAIL monthly5_activation status=blocked {summary_text}")
        for item in failures:
            print(f"BLOCKER {item}")
        return 1
    print(f"PASS monthly5_activation status=actionable {summary_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
