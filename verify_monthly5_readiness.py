#!/usr/bin/env python3
"""Assess monthly 5% shadow-history readiness for promotion review."""

import argparse
import json
from pathlib import Path

import monthly5_shadow
from verify_monthly5_runtime import DEFAULT_HISTORY, DEFAULT_SPEC, _load_json, _load_jsonl


DEFAULT_MIN_RECORDS = 48
DEFAULT_MIN_SPAN_HOURS = 24.0
DEFAULT_MAX_AGE_SEC = 900.0


def _history_readiness(rows: list[dict], spec: dict, *, min_records: int, min_span_hours: float, max_age_sec: float | None):
    evidence = spec.get("backtest_evidence") if isinstance(spec.get("backtest_evidence"), dict) else {}
    max_flat_time_pct = evidence.get("avg_flat_time_pct")
    return monthly5_shadow.build_readiness_report(
        rows,
        strategy_id=str(spec.get("strategy_id") or ""),
        selected_candidate=str(evidence.get("candidate_name") or ""),
        min_records=min_records,
        min_span_hours=min_span_hours,
        max_age_sec=max_age_sec,
        max_flat_time_pct=max_flat_time_pct,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--history", default=str(DEFAULT_HISTORY))
    parser.add_argument("--min-records", type=int, default=DEFAULT_MIN_RECORDS)
    parser.add_argument("--min-span-hours", type=float, default=DEFAULT_MIN_SPAN_HOURS)
    parser.add_argument("--max-age-sec", type=float, default=DEFAULT_MAX_AGE_SEC)
    parser.add_argument("--max-flat-time-pct", type=float, default=None)
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
    if args.max_flat_time_pct is not None:
        report = monthly5_shadow.build_readiness_report(
            rows,
            strategy_id=str(spec.get("strategy_id") or ""),
            selected_candidate=str((spec.get("backtest_evidence") or {}).get("candidate_name") or ""),
            min_records=max(1, args.min_records),
            min_span_hours=max(0.0, args.min_span_hours),
            max_age_sec=args.max_age_sec,
            max_flat_time_pct=args.max_flat_time_pct,
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
            f"risk_rows={report.get('risk_rows', 0)} "
            f"shadow_flat_time_pct={report.get('shadow_flat_time_pct', 0.0)} "
            f"actual_flat_time_pct={report.get('actual_flat_time_pct', 0.0)} "
            f"shadow_paper_return_pct={report.get('shadow_paper_return_pct', 0.0)} "
            f"shadow_paper_intervals={report.get('shadow_paper_intervals', 0)} "
            f"shadow_projected_monthly_return_pct={report.get('shadow_projected_monthly_return_pct', 0.0)} "
            f"shadow_monthly_projection_valid={str(bool(report.get('shadow_monthly_projection_valid'))).lower()} "
            f"shadow_monthly_target_met={str(bool(report.get('shadow_monthly_target_met'))).lower()} "
            f"promotion_ready={str(bool(report.get('promotion_ready'))).lower()}"
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
