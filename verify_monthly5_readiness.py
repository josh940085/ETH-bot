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
    policy = spec.get("policy") if isinstance(spec.get("policy"), dict) else {}
    selector = policy.get("selector") if isinstance(policy.get("selector"), dict) else {}
    normal_selector = selector.get("normal") if isinstance(selector.get("normal"), dict) else {}
    max_flat_time_pct = evidence.get("avg_flat_time_pct")
    return monthly5_shadow.build_readiness_report(
        rows,
        strategy_id=str(spec.get("strategy_id") or ""),
        selected_candidate=str(evidence.get("candidate_name") or ""),
        min_records=min_records,
        min_span_hours=min_span_hours,
        max_age_sec=max_age_sec,
        max_flat_time_pct=max_flat_time_pct,
        required_selector_source=str(normal_selector.get("type") or ""),
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
    parser.add_argument("--require-promotion-ready", action="store_true")
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
            required_selector_source=str(
                (((spec.get("policy") or {}).get("selector") or {}).get("normal") or {}).get("type") or ""
            ),
        )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        print(
            "PASS monthly5_readiness "
            f"status={report.get('status')} "
            f"ready={str(bool(report.get('ready'))).lower()} "
            f"rows={report.get('rows', 0)} "
            f"min_selector_policy_version={report.get('min_selector_policy_version', 0)} "
            f"ignored_legacy_selector_policy_rows={report.get('ignored_legacy_selector_policy_rows', 0)} "
            f"required_selector_source={report.get('required_selector_source', '')} "
            f"selector_policy_aligned={str(bool(report.get('selector_policy_aligned', True))).lower()} "
            f"selector_policy_proxy_rows={report.get('selector_policy_proxy_rows', 0)} "
            f"span_hours={report.get('span_hours', 0.0)} "
            f"sample_count_remaining={report.get('sample_count_remaining', 0)} "
            f"sample_count_ready_ts={report.get('sample_count_ready_ts', 0)} "
            f"sample_rows_per_hour_est={report.get('sample_rows_per_hour_est', 0.0)} "
            f"sample_median_interval_sec={report.get('sample_median_interval_sec', 0.0)} "
            f"sample_span_remaining_hours={report.get('sample_span_remaining_hours', 0.0)} "
            f"sample_span_progress_pct={report.get('sample_span_progress_pct', 0.0)} "
            f"evaluate_rows={report.get('evaluate_rows', 0)} "
            f"risk_rows={report.get('risk_rows', 0)} "
            f"shadow_flat_time_pct={report.get('shadow_flat_time_pct', 0.0)} "
            f"shadow_flat_time_gap_pct={report.get('shadow_flat_time_gap_pct', 0.0)} "
            f"actual_flat_time_pct={report.get('actual_flat_time_pct', 0.0)} "
            f"shadow_paper_return_pct={report.get('shadow_paper_return_pct', 0.0)} "
            f"shadow_observed_target_gap_pct={report.get('shadow_observed_target_gap_pct', 0.0)} "
            f"shadow_paper_intervals={report.get('shadow_paper_intervals', 0)} "
            f"shadow_projected_monthly_return_pct={report.get('shadow_projected_monthly_return_pct', 0.0)} "
            f"shadow_monthly_projection_valid={str(bool(report.get('shadow_monthly_projection_valid'))).lower()} "
            f"shadow_monthly_target_met={str(bool(report.get('shadow_monthly_target_met'))).lower()} "
            f"shadow_rolling_paper_return_pct={report.get('shadow_rolling_paper_return_pct', 0.0)} "
            f"shadow_rolling_observed_target_gap_pct={report.get('shadow_rolling_observed_target_gap_pct', 0.0)} "
            f"shadow_rolling_projected_monthly_return_pct={report.get('shadow_rolling_projected_monthly_return_pct', 0.0)} "
            f"shadow_rolling_monthly_target_met={str(bool(report.get('shadow_rolling_monthly_target_met'))).lower()} "
            f"shadow_activation_rows={report.get('shadow_activation_rows', 0)} "
            f"shadow_activation_span_hours={report.get('shadow_activation_span_hours', 0.0)} "
            f"shadow_activation_paper_return_pct={report.get('shadow_activation_paper_return_pct', 0.0)} "
            f"shadow_activation_projected_monthly_return_pct={report.get('shadow_activation_projected_monthly_return_pct', 0.0)} "
            f"shadow_activation_monthly_target_met={str(bool(report.get('shadow_activation_monthly_target_met'))).lower()} "
            f"shadow_underperforming_plan_count={report.get('shadow_underperforming_plan_count', 0)} "
            f"shadow_active_underperforming_plan_count={report.get('shadow_active_underperforming_plan_count', 0)} "
            f"shadow_suppressed_recovering_plan_count={report.get('shadow_suppressed_recovering_plan_count', 0)} "
            f"shadow_suppressed_observed_intervals={report.get('shadow_suppressed_observed_intervals', 0)} "
            f"shadow_suppressed_recovery_remaining_intervals={report.get('shadow_suppressed_recovery_remaining_intervals', 0)} "
            f"shadow_recovery_probe_state={report.get('shadow_recovery_probe_state', '')} "
            f"shadow_recovery_probe_success_count={report.get('shadow_recovery_probe_success_count', 0)} "
            f"shadow_recovery_probe_candidate_count={report.get('shadow_recovery_probe_candidate_count', 0)} "
            f"shadow_recovery_probe_failed_count={report.get('shadow_recovery_probe_failed_count', 0)} "
            f"shadow_recovery_probe_observed_intervals={report.get('shadow_recovery_probe_observed_intervals', 0)} "
            f"shadow_recovery_probe_remaining_intervals={report.get('shadow_recovery_probe_remaining_intervals', 0)} "
            f"shadow_recovery_probe_progress_pct={report.get('shadow_recovery_probe_progress_pct', 0.0)} "
            f"promotion_blocker_count={report.get('promotion_blocker_count', 0)} "
            f"promotion_blockers={','.join(str(item) for item in (report.get('promotion_blockers') or []))} "
            f"promotion_ready={str(bool(report.get('promotion_ready'))).lower()}"
        )
        for item in report.get("failures") or []:
            print(f"FAIL {item}")
        for item in report.get("warnings") or []:
            print(f"WARN {item}")
        for item in report.get("promotion_blocker_details") or []:
            if not isinstance(item, dict):
                continue
            parts = [
                f"code={item.get('code', '')}",
                f"label={item.get('label', '')}",
                f"current={item.get('current', '')}",
                f"target={item.get('target', '')}",
            ]
            if "remaining_hours" in item:
                parts.append(f"remaining_hours={item.get('remaining_hours')}")
            if "ready_ts" in item:
                parts.append(f"ready_ts={item.get('ready_ts')}")
            if "remaining" in item:
                parts.append(f"remaining={item.get('remaining')}")
            if "remaining_intervals" in item:
                parts.append(f"remaining_intervals={item.get('remaining_intervals')}")
            if "observed_target_gap_pct" in item:
                parts.append(f"observed_target_gap_pct={item.get('observed_target_gap_pct')}")
            if "over_target_pct" in item:
                parts.append(f"over_target_pct={item.get('over_target_pct')}")
            print("BLOCKER " + " ".join(str(part) for part in parts))
    if (
        report.get("failures")
        or (args.require_ready and not report.get("ready"))
        or (args.require_promotion_ready and not report.get("promotion_ready"))
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
