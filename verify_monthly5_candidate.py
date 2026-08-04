#!/usr/bin/env python3
"""Verify the monthly 5% research-candidate summary against its strategy spec."""

import argparse
import json
import re
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


def _month_range(start: str, end: str) -> list[str]:
    try:
        year, month = [int(part) for part in str(start).split("-", 1)]
        end_year, end_month = [int(part) for part in str(end).split("-", 1)]
    except Exception:
        return []
    months = []
    while (year, month) <= (end_year, end_month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            year += 1
            month = 1
    return months


def _pick_leverage(top_pick: str) -> int | None:
    match = re.search(r"(?:^|\|)lev(\d+)(?:\||$)", str(top_pick or ""))
    if not match:
        return None
    return int(match.group(1))


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

    monthly_path = Path(str(evidence.get("source_monthly") or ""))
    if not monthly_path.exists():
        failures.append(f"monthly evidence file missing: {monthly_path}")
        return failures
    monthly = _load_json(monthly_path)
    rows = monthly.get(expected_name)
    if not isinstance(rows, list) or not rows:
        failures.append(f"monthly rows missing for candidate: {expected_name}")
        return failures

    period_start = str(evidence.get("period_start") or "")
    period_end = str(evidence.get("period_end") or "")
    expected_months = _month_range(period_start, period_end)
    actual_months = [str(row.get("month") or "") for row in rows if isinstance(row, dict)]
    if expected_months and actual_months != expected_months:
        failures.append(
            f"monthly period mismatch: got {actual_months[:1]}..{actual_months[-1:]}, "
            f"want {expected_months[:1]}..{expected_months[-1:]}"
        )

    expected_total_months = int(evidence.get("months", 0))
    if len(rows) != expected_total_months:
        failures.append(f"monthly row count mismatch: got {len(rows)}, want {expected_total_months}")

    floor = float((spec.get("objective") or {}).get("monthly_return_floor_pct", 5.0))
    monthly_failed = [
        row
        for row in rows
        if isinstance(row, dict) and float(row.get("return_pct", -1000.0)) < floor
    ]
    monthly_failed_months = [str(row.get("month") or "") for row in monthly_failed]
    summary_failed_months = [str(row.get("month") or "") for row in failed if isinstance(row, dict)]
    if monthly_failed_months != summary_failed_months:
        failures.append(f"summary/monthly failed months mismatch: got {monthly_failed_months}, summary {summary_failed_months}")

    complete_months_end = str(evidence.get("complete_months_end") or "")
    complete_rows = [
        row
        for row in rows
        if isinstance(row, dict) and str(row.get("month") or "") <= complete_months_end
    ]
    if len(complete_rows) != complete_months:
        failures.append(f"complete monthly row count mismatch: got {len(complete_rows)}, want {complete_months}")
    complete_failed = [
        row
        for row in complete_rows
        if float(row.get("return_pct", -1000.0)) < floor
    ]
    if complete_failed:
        failures.append(f"complete monthly rows below {floor}%: {complete_failed}")

    monthly_worst = min(float(row.get("min_intramonth_pnl_pct", 0.0)) for row in rows if isinstance(row, dict))
    if abs(monthly_worst - actual_worst) > 0.01:
        failures.append(f"summary/monthly worst intramonth mismatch: got {monthly_worst}, summary {actual_worst}")

    monthly_avg_flat = sum(float(row.get("flat_time_pct", 0.0)) for row in rows if isinstance(row, dict)) / len(rows)
    if abs(monthly_avg_flat - actual_flat) > 0.01:
        failures.append(f"summary/monthly avg flat mismatch: got {monthly_avg_flat:.3f}, summary {actual_flat:.3f}")

    max_leverage = int((spec.get("objective") or {}).get("max_leverage", 5))
    over_leverage = [
        {"month": row.get("month"), "top_pick": row.get("top_pick")}
        for row in rows
        if isinstance(row, dict)
        for leverage in [_pick_leverage(str(row.get("top_pick") or ""))]
        if leverage is None or leverage > max_leverage
    ]
    if over_leverage:
        failures.append(f"monthly rows exceed leverage cap {max_leverage}x or miss leverage: {over_leverage}")

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
