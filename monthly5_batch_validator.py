"""Batch-validate monthly5 research artifacts before live promotion."""

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from monthly5_research_selector import parse_top_pick


DEFAULT_MONTHLY_PATH = Path(
    ".runtime/data/backtests/monthly5_search/postlock_low_exposure_hourly_monthly.json"
)
DEFAULT_TARGET_PCT = 5.0
DEFAULT_MAX_FLAT_TIME_PCT = 50.0
DEFAULT_MAX_LEVERAGE = 5
DEFAULT_HOLDOUT_START = "2024-01"
DEFAULT_COMPLETE_THROUGH = "2026-07"


def _safe_float(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _month_key(value):
    text = str(value or "").strip()
    try:
        return pd.Period(text, freq="M").strftime("%Y-%m")
    except (TypeError, ValueError):
        return ""


def _expected_months(start, complete_through):
    try:
        return [period.strftime("%Y-%m") for period in pd.period_range(start, complete_through, freq="M")]
    except (TypeError, ValueError):
        return []


def _compound_return_pct(returns_pct):
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in returns_pct:
        equity *= 1.0 + (value / 100.0)
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = min(max_drawdown, ((equity / peak) - 1.0) * 100.0)
    return (equity - 1.0) * 100.0, max_drawdown


def _requested_leverage(top_pick):
    match = re.search(r"(?:^|\|)lev(\d+)(?:\||$)", str(top_pick or ""))
    return int(match.group(1)) if match else 0


def _summarize_rows(rows, target_pct):
    returns = [_safe_float(row.get("return_pct")) for row in rows]
    flat_times = [_safe_float(row.get("flat_time_pct")) for row in rows]
    compound, drawdown = _compound_return_pct(returns)
    return {
        "months": len(rows),
        "months_ge_target": sum(value >= target_pct for value in returns),
        "target_hit_rate": round(
            (sum(value >= target_pct for value in returns) / len(rows)) if rows else 0.0,
            6,
        ),
        "avg_return_pct": round(sum(returns) / len(returns), 6) if returns else 0.0,
        "min_return_pct": round(min(returns), 6) if returns else 0.0,
        "compound_return_pct": round(compound, 6),
        "max_monthly_equity_drawdown_pct": round(drawdown, 6),
        "avg_flat_time_pct": round(sum(flat_times) / len(flat_times), 6) if flat_times else 100.0,
        "max_flat_time_pct": round(max(flat_times), 6) if flat_times else 100.0,
    }


def inspect_trade_evidence(path, *, expected_candidate="", expected_months=None):
    required = [
        "entry_time",
        "exit_time",
        "entry_fill_time",
        "exit_fill_time",
        "side",
        "quantity",
        "pnl",
        "fee",
        "slippage",
        "data_source",
        "candidate",
    ]
    if not path:
        return {
            "available": False,
            "usable": False,
            "path": "",
            "rows": 0,
            "missing_columns": required,
        }
    source = Path(path)
    if not source.exists():
        return {
            "available": False,
            "usable": False,
            "path": str(source),
            "rows": 0,
            "missing_columns": required,
        }
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = {str(name or "").strip().lower() for name in (reader.fieldnames or [])}
            evidence_rows = list(reader)
            rows = len(evidence_rows)
    except Exception as exc:
        return {
            "available": True,
            "usable": False,
            "path": str(source),
            "rows": 0,
            "error": str(exc),
            "missing_columns": required,
        }
    aliases = {
        "entry_time": {"entry_time", "open_time", "entry_ts"},
        "exit_time": {"exit_time", "close_time", "exit_ts"},
        "entry_fill_time": {"entry_fill_time"},
        "exit_fill_time": {"exit_fill_time"},
        "side": {"side", "direction", "position_side"},
        "quantity": {"quantity", "qty", "size"},
        "pnl": {"pnl", "net_pnl", "realized_pnl", "return_pct"},
        "fee": {"fee", "fees", "commission", "commission_usdt"},
        "slippage": {"slippage", "slippage_pct", "slippage_usdt"},
        "data_source": {"data_source"},
        "candidate": {"candidate"},
    }
    missing = [name for name, names in aliases.items() if not (fieldnames & names)]
    invalid_rows = 0
    for row in evidence_rows:
        try:
            if float(row.get("quantity", 0)) <= 0 or float(row.get("fee", -1)) < 0:
                invalid_rows += 1
            if not str(row.get("data_source") or "").startswith("binance_public_data_"):
                invalid_rows += 1
        except (TypeError, ValueError):
            invalid_rows += 1
    candidate_values = {str(row.get("candidate") or "") for row in evidence_rows}
    candidate_matches = bool(expected_candidate) and candidate_values == {str(expected_candidate)}
    evidence_months = sorted(
        {
            str(row.get("entry_time") or "")[:7]
            for row in evidence_rows
            if len(str(row.get("entry_time") or "")) >= 7
        }
    )
    required_months = sorted(set(expected_months or []))
    missing_evidence_months = [month for month in required_months if month not in evidence_months]
    return {
        "available": True,
        "usable": (
            rows > 0
            and not missing
            and invalid_rows == 0
            and candidate_matches
            and not missing_evidence_months
        ),
        "path": str(source),
        "rows": rows,
        "missing_columns": missing,
        "invalid_rows": invalid_rows,
        "candidate_values": sorted(candidate_values),
        "candidate_matches": candidate_matches,
        "evidence_months": evidence_months,
        "missing_evidence_months": missing_evidence_months,
    }


def validate_candidate(
    name,
    rows,
    *,
    start="2020-01",
    complete_through=DEFAULT_COMPLETE_THROUGH,
    holdout_start=DEFAULT_HOLDOUT_START,
    target_pct=DEFAULT_TARGET_PCT,
    max_flat_time_pct=DEFAULT_MAX_FLAT_TIME_PCT,
    max_leverage=DEFAULT_MAX_LEVERAGE,
    trade_evidence=None,
):
    normalized = []
    invalid_month_rows = 0
    for raw in rows if isinstance(rows, list) else []:
        if not isinstance(raw, dict):
            continue
        month = _month_key(raw.get("month"))
        if not month:
            invalid_month_rows += 1
            continue
        if month < start or month > complete_through:
            continue
        row = dict(raw)
        row["month"] = month
        normalized.append(row)
    normalized.sort(key=lambda row: row["month"])

    expected = _expected_months(start, complete_through)
    present = {row["month"] for row in normalized}
    missing_months = [month for month in expected if month not in present]
    duplicate_months = sorted(
        month for month in present if sum(row["month"] == month for row in normalized) > 1
    )
    leverage_violations = []
    invalid_picks = []
    for row in normalized:
        parsed = parse_top_pick(row.get("top_pick"))
        requested_leverage = _requested_leverage(row.get("top_pick"))
        if not parsed.get("valid"):
            invalid_picks.append(row["month"])
        if requested_leverage > max_leverage:
            leverage_violations.append(
                {"month": row["month"], "leverage": requested_leverage}
            )

    train_rows = [row for row in normalized if row["month"] < holdout_start]
    holdout_rows = [row for row in normalized if row["month"] >= holdout_start]
    all_stats = _summarize_rows(normalized, target_pct)
    train_stats = _summarize_rows(train_rows, target_pct)
    holdout_stats = _summarize_rows(holdout_rows, target_pct)
    exact_floor_months = [
        row["month"]
        for row in normalized
        if abs(_safe_float(row.get("return_pct")) - target_pct) <= 1e-9
    ]
    exact_floor_ratio = len(exact_floor_months) / len(normalized) if normalized else 0.0
    evidence = inspect_trade_evidence(
        trade_evidence,
        expected_candidate=name,
        expected_months=[row["month"] for row in normalized],
    )

    metric_blockers = []
    if missing_months:
        metric_blockers.append("month_coverage_incomplete")
    if duplicate_months:
        metric_blockers.append("duplicate_months")
    if invalid_month_rows:
        metric_blockers.append("invalid_month_rows")
    if not normalized:
        metric_blockers.append("no_complete_months")
    elif all_stats["months_ge_target"] != all_stats["months"]:
        metric_blockers.append("monthly_target_missed")
    if all_stats["avg_flat_time_pct"] > max_flat_time_pct:
        metric_blockers.append("average_flat_time_too_high")
    if leverage_violations:
        metric_blockers.append("max_leverage_exceeded")
    if invalid_picks:
        metric_blockers.append("candidate_pick_unparseable")
    if not holdout_rows:
        metric_blockers.append("holdout_missing")
    elif holdout_stats["months_ge_target"] != holdout_stats["months"]:
        metric_blockers.append("holdout_target_missed")

    evidence_blockers = []
    if exact_floor_ratio >= 0.20:
        evidence_blockers.append("exact_target_floor_saturation")
    if not evidence["available"]:
        evidence_blockers.append("trade_evidence_missing")
    elif not evidence["usable"]:
        evidence_blockers.append("trade_cost_evidence_incomplete")

    metric_qualified = not metric_blockers
    deployment_ready = metric_qualified and not evidence_blockers
    if deployment_ready:
        verdict = "deployment_candidate"
    elif metric_qualified:
        verdict = "research_only"
    else:
        verdict = "rejected"
    return {
        "candidate": str(name),
        "verdict": verdict,
        "metric_qualified": metric_qualified,
        "deployment_ready": deployment_ready,
        "metric_blockers": metric_blockers,
        "evidence_blockers": evidence_blockers,
        "period": {
            "start": start,
            "complete_through": complete_through,
            "holdout_start": holdout_start,
            "expected_months": len(expected),
            "missing_months": missing_months,
            "duplicate_months": duplicate_months,
        },
        "limits": {
            "monthly_target_pct": target_pct,
            "max_average_flat_time_pct": max_flat_time_pct,
            "max_leverage": max_leverage,
        },
        "all": all_stats,
        "train": train_stats,
        "holdout": holdout_stats,
        "integrity": {
            "invalid_month_rows": invalid_month_rows,
            "invalid_pick_months": invalid_picks,
            "leverage_violations": leverage_violations,
            "exact_target_floor_months": exact_floor_months,
            "exact_target_floor_ratio": round(exact_floor_ratio, 6),
            "trade_evidence": evidence,
        },
    }


def build_batch_report(
    monthly_path=DEFAULT_MONTHLY_PATH,
    *,
    candidate="",
    start="2020-01",
    complete_through=DEFAULT_COMPLETE_THROUGH,
    holdout_start=DEFAULT_HOLDOUT_START,
    target_pct=DEFAULT_TARGET_PCT,
    max_flat_time_pct=DEFAULT_MAX_FLAT_TIME_PCT,
    max_leverage=DEFAULT_MAX_LEVERAGE,
    trade_evidence=None,
):
    source = Path(monthly_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("monthly artifact must be a candidate-to-rows object")
    names = [candidate] if candidate else sorted(payload)
    missing_candidates = [name for name in names if name not in payload]
    if missing_candidates:
        raise ValueError(f"candidate not found: {', '.join(missing_candidates)}")
    results = [
        validate_candidate(
            name,
            payload[name],
            start=start,
            complete_through=complete_through,
            holdout_start=holdout_start,
            target_pct=target_pct,
            max_flat_time_pct=max_flat_time_pct,
            max_leverage=max_leverage,
            trade_evidence=trade_evidence,
        )
        for name in names
    ]
    verdict_order = {"deployment_candidate": 0, "research_only": 1, "rejected": 2}
    results.sort(
        key=lambda row: (
            verdict_order[row["verdict"]],
            -row["holdout"]["target_hit_rate"],
            row["all"]["avg_flat_time_pct"],
            -row["holdout"]["avg_return_pct"],
        )
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": str(source),
        "candidate_count": len(results),
        "deployment_candidate_count": sum(row["deployment_ready"] for row in results),
        "metric_qualified_count": sum(row["metric_qualified"] for row in results),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monthly", default=str(DEFAULT_MONTHLY_PATH))
    parser.add_argument("--candidate", default="")
    parser.add_argument("--start", default="2020-01")
    parser.add_argument("--complete-through", default=DEFAULT_COMPLETE_THROUGH)
    parser.add_argument("--holdout-start", default=DEFAULT_HOLDOUT_START)
    parser.add_argument("--target-pct", type=float, default=DEFAULT_TARGET_PCT)
    parser.add_argument("--max-flat-time-pct", type=float, default=DEFAULT_MAX_FLAT_TIME_PCT)
    parser.add_argument("--max-leverage", type=int, default=DEFAULT_MAX_LEVERAGE)
    parser.add_argument("--trade-evidence")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = build_batch_report(
        args.monthly,
        candidate=args.candidate,
        start=args.start,
        complete_through=args.complete_through,
        holdout_start=args.holdout_start,
        target_pct=args.target_pct,
        max_flat_time_pct=args.max_flat_time_pct,
        max_leverage=args.max_leverage,
        trade_evidence=args.trade_evidence,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["deployment_candidate_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
