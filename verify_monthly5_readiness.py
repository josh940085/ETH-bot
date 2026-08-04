#!/usr/bin/env python3
"""Assess monthly 5% shadow-history readiness for promotion review."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from verify_monthly5_runtime import DEFAULT_HISTORY, DEFAULT_SPEC, _load_json, _load_jsonl, _safe_float


DEFAULT_MIN_RECORDS = 48
DEFAULT_MIN_SPAN_HOURS = 24.0
DEFAULT_MAX_AGE_SEC = 900.0


def _safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _history_readiness(rows: list[dict], spec: dict, *, min_records: int, min_span_hours: float, max_age_sec: float | None):
    failures: list[str] = []
    warnings: list[str] = []
    evidence = spec.get("backtest_evidence") if isinstance(spec.get("backtest_evidence"), dict) else {}
    candidate = str(evidence.get("candidate_name") or "")
    strategy_id = str(spec.get("strategy_id") or "")
    now = time.time()

    valid_rows = [row for row in rows if isinstance(row, dict)]
    if len(valid_rows) != len(rows):
        failures.append("history contains non-object rows")
    if not valid_rows:
        failures.append("history has no rows")
        return {"status": "invalid", "ready": False, "failures": failures, "warnings": warnings}

    strategy_drift = [
        row for row in valid_rows
        if row.get("strategy_id") != strategy_id or row.get("selected_candidate") != candidate
    ]
    if strategy_drift:
        failures.append(f"strategy/candidate drift rows={len(strategy_drift)}")

    unsafe_rows = [
        row for row in valid_rows
        if row.get("shadow_only") is not True
        or _safe_float(row.get("max_leverage"), 99.0) > 5.0
        or not (0.0 <= _safe_float(row.get("exposure_cap"), -1.0) <= 1.0)
    ]
    if unsafe_rows:
        failures.append(f"unsafe shadow rows={len(unsafe_rows)}")

    timestamps = sorted(_safe_int(row.get("updated_ts"), 0) for row in valid_rows if _safe_int(row.get("updated_ts"), 0) > 0)
    if not timestamps:
        failures.append("history timestamps missing")
        span_hours = 0.0
        age_sec = 10**9
    else:
        span_hours = (timestamps[-1] - timestamps[0]) / 3600.0
        age_sec = now - timestamps[-1]
    if max_age_sec is not None and age_sec > max_age_sec:
        failures.append(f"history stale: age={age_sec:.1f}s > {max_age_sec:.1f}s")

    selected_plan_counts = Counter(str(row.get("selected_plan") or "") for row in valid_rows)
    action_counts = Counter(str(row.get("shadow_action") or "") for row in valid_rows)
    mode_counts = Counter(str(row.get("mode") or "") for row in valid_rows)
    market_bias_counts = Counter(str(row.get("market_bias") or "") for row in valid_rows)
    risk_rows = [
        row for row in valid_rows
        if row.get("mode") in {"intraday_stop", "post_lock_floor_guard", "post_lock", "recovery"}
        or row.get("guard_allowed") is False
    ]
    evaluate_rows = [
        row for row in valid_rows
        if str(row.get("shadow_action") or "") in {"evaluate_long", "evaluate_short", "reduced_exposure"}
    ]

    if len(valid_rows) < min_records:
        warnings.append(f"sample count collecting: rows={len(valid_rows)} < {min_records}")
    if span_hours < min_span_hours:
        warnings.append(f"sample span collecting: hours={span_hours:.2f} < {min_span_hours:.2f}")
    if not evaluate_rows:
        warnings.append("no evaluate_long/evaluate_short/reduced_exposure samples yet")
    if not risk_rows:
        warnings.append("no live risk-mode samples yet")

    ready = not failures and len(valid_rows) >= min_records and span_hours >= min_span_hours and bool(evaluate_rows)
    status = "ready" if ready else "collecting"
    if failures:
        status = "invalid"
    return {
        "status": status,
        "ready": ready,
        "failures": failures,
        "warnings": warnings,
        "rows": len(valid_rows),
        "span_hours": round(max(0.0, span_hours), 4),
        "latest_age_sec": round(max(0.0, age_sec), 1),
        "selected_plan_counts": dict(sorted(selected_plan_counts.items())),
        "shadow_action_counts": dict(sorted(action_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "market_bias_counts": dict(sorted(market_bias_counts.items())),
        "risk_rows": len(risk_rows),
        "evaluate_rows": len(evaluate_rows),
        "min_records": min_records,
        "min_span_hours": min_span_hours,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--history", default=str(DEFAULT_HISTORY))
    parser.add_argument("--min-records", type=int, default=DEFAULT_MIN_RECORDS)
    parser.add_argument("--min-span-hours", type=float, default=DEFAULT_MIN_SPAN_HOURS)
    parser.add_argument("--max-age-sec", type=float, default=DEFAULT_MAX_AGE_SEC)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    spec = _load_json(Path(args.spec))
    history_path = Path(args.history)
    rows = _load_jsonl(history_path) if history_path.exists() else []
    report = _history_readiness(
        rows,
        spec,
        min_records=max(1, args.min_records),
        min_span_hours=max(0.0, args.min_span_hours),
        max_age_sec=args.max_age_sec,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        print(
            "PASS monthly5_readiness "
            f"status={report.get('status')} "
            f"ready={str(bool(report.get('ready'))).lower()} "
            f"rows={report.get('rows', 0)} "
            f"span_hours={report.get('span_hours', 0.0)} "
            f"evaluate_rows={report.get('evaluate_rows', 0)} "
            f"risk_rows={report.get('risk_rows', 0)}"
        )
        for item in report.get("failures") or []:
            print(f"FAIL {item}")
        for item in report.get("warnings") or []:
            print(f"WARN {item}")
    if report.get("failures") or (args.require_ready and not report.get("ready")):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
