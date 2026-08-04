#!/usr/bin/env python3
"""Verify the monthly 5% research-candidate summary against its strategy spec."""

import argparse
import json
from pathlib import Path


DEFAULT_SPEC = Path("docs/strategy_specs/monthly5_postlock_hourly.json")


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"failed to read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"expected object JSON: {path}")
    return payload


def _failures(summary: dict, spec: dict) -> list[str]:
    evidence = spec.get("backtest_evidence") or {}
    top = summary.get("top") or []
    if not top or not isinstance(top[0], dict):
        return ["summary.top[0] is missing"]
    best = top[0]
    failures: list[str] = []

    expected_name = str(evidence.get("candidate_name") or "")
    if best.get("name") != expected_name:
        failures.append(f"best candidate mismatch: got {best.get('name')!r}, want {expected_name!r}")

    checks = {
        "months_ge_5": int(evidence.get("months_ge_5", 0)),
        "months_ge_0": int(evidence.get("months_ge_0", 0)),
    }
    for key, expected in checks.items():
        actual = int(best.get(key, -1))
        if actual < expected:
            failures.append(f"{key} too low: got {actual}, want >= {expected}")

    complete_months_ge_5 = int(evidence.get("complete_months_ge_5", 0))
    complete_months = int(evidence.get("complete_months", 0))
    failed = best.get("failed_months") or []
    incomplete_month = str(evidence.get("incomplete_month") or "")
    complete_failures = [row for row in failed if str(row.get("month") or "") != incomplete_month]
    if complete_failures:
        failures.append(f"complete month failures present: {complete_failures}")
    if complete_months_ge_5 != complete_months:
        failures.append(f"spec complete-month target is inconsistent: {complete_months_ge_5}/{complete_months}")

    worst_limit = float(evidence.get("worst_intramonth_pnl_pct", -100.0))
    actual_worst = float(best.get("worst_intramonth_pnl_pct", -1000.0))
    if actual_worst < worst_limit:
        failures.append(f"worst intramonth PnL too low: got {actual_worst}, want >= {worst_limit}")

    flat_limit = float(evidence.get("avg_flat_time_pct", 100.0))
    actual_flat = float(best.get("avg_flat_time_pct", 1000.0))
    if actual_flat > flat_limit:
        failures.append(f"average flat time too high: got {actual_flat}, want <= {flat_limit}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    args = parser.parse_args()

    spec_path = Path(args.spec)
    spec = _load_json(spec_path)
    evidence = spec.get("backtest_evidence") or {}
    summary_path = Path(str(evidence.get("source_summary") or ""))
    summary = _load_json(summary_path)
    failures = _failures(summary, spec)
    if failures:
        for item in failures:
            print(f"FAIL {item}")
        return 1

    best = summary["top"][0]
    print(
        "PASS "
        f"{best.get('name')} months_ge_5={best.get('months_ge_5')} "
        f"months_ge_0={best.get('months_ge_0')} "
        f"worst_intramonth_pnl_pct={best.get('worst_intramonth_pnl_pct')} "
        f"avg_flat_time_pct={best.get('avg_flat_time_pct')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
