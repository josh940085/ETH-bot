"""Detect market-regime drift from monthly5 shadow history using River ADWIN."""

import argparse
import json
import math
import os
import signal
import threading
import time
from pathlib import Path

from runtime_python import require_supported_python


require_supported_python("regime-drift")


DEFAULT_INPUT = Path(".runtime/data/btcusdt_monthly5_shadow_history.jsonl")
DEFAULT_OUTPUT = Path(".runtime/data/btcusdt_market_regime_drift_latest.json")
FEATURE_NAMES = (
    "price_return_bps",
    "score_margin",
    "score_intensity",
    "selector_hit_rate",
    "selector_q25_return_pct",
    "monthly_pnl_pct",
)


def _finite_float(value, default=None):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def load_shadow_rows(path, *, min_policy_version=8):
    source = Path(path)
    if not source.exists():
        return []
    rows = []
    seen_timestamps = set()
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(raw_line)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(row, dict) or not bool(row.get("shadow_only", False)):
            continue
        try:
            policy_version = int(row.get("selector_policy_version", 0) or 0)
            updated_ts = int(row.get("updated_ts", 0) or 0)
        except (TypeError, ValueError):
            continue
        mark_price = _finite_float(row.get("mark_price"), 0.0)
        if policy_version < min_policy_version or updated_ts <= 0 or mark_price <= 0:
            continue
        if updated_ts in seen_timestamps:
            continue
        seen_timestamps.add(updated_ts)
        rows.append(row)
    return sorted(rows, key=lambda row: int(row["updated_ts"]))


def build_feature_rows(rows):
    feature_rows = []
    previous_price = None
    for row in rows:
        mark_price = _finite_float(row.get("mark_price"), 0.0)
        if mark_price <= 0:
            continue
        price_return_bps = 0.0
        if previous_price and previous_price > 0:
            price_return_bps = ((mark_price / previous_price) - 1.0) * 10_000.0
        previous_price = mark_price
        bull_score = _finite_float(row.get("bull_score"), 0.0)
        bear_score = _finite_float(row.get("bear_score"), 0.0)
        features = {
            "price_return_bps": max(-5_000.0, min(5_000.0, price_return_bps)),
            "score_margin": bull_score - bear_score,
            "score_intensity": bull_score + bear_score,
            "selector_hit_rate": _finite_float(row.get("selector_hit_rate"), 0.0),
            "selector_q25_return_pct": _finite_float(
                row.get("selector_q25_return_pct"), 0.0
            ),
            "monthly_pnl_pct": _finite_float(row.get("monthly_pnl_pct"), 0.0),
        }
        feature_rows.append(
            {
                "updated_ts": int(row["updated_ts"]),
                "market_bias": str(row.get("market_bias") or ""),
                "market_state": str(row.get("market_state") or ""),
                "features": features,
            }
        )
    return feature_rows


def _river_detector_factory(delta):
    from river import drift

    return lambda _feature: drift.ADWIN(delta=delta)


def analyze_feature_rows(
    feature_rows,
    *,
    detector_factory,
    min_rows=48,
    recent_window_rows=48,
):
    detectors = {name: detector_factory(name) for name in FEATURE_NAMES}
    events = []
    for index, item in enumerate(feature_rows):
        for feature, value in item["features"].items():
            detector = detectors[feature]
            detector.update(value)
            if bool(getattr(detector, "drift_detected", False)):
                events.append(
                    {
                        "row_index": index,
                        "updated_ts": item["updated_ts"],
                        "feature": feature,
                        "value": round(value, 8),
                    }
                )

    row_count = len(feature_rows)
    recent_start = max(0, row_count - max(1, int(recent_window_rows)))
    recent_events = [event for event in events if event["row_index"] >= recent_start]
    latest = feature_rows[-1] if feature_rows else {}
    ready = row_count >= max(1, int(min_rows))
    return {
        "schema_version": 1,
        "generated_ts": int(time.time()),
        "method": "river_adwin_multifeature_shadow",
        "shadow_only": True,
        "live_control_enabled": False,
        "promotion_eligible": False,
        "ready": ready,
        "rows_analyzed": row_count,
        "minimum_rows": max(1, int(min_rows)),
        "features": list(FEATURE_NAMES),
        "drift_detected": bool(ready and recent_events),
        "drift_event_count": len(events),
        "recent_window_rows": max(1, int(recent_window_rows)),
        "recent_events": recent_events[-50:],
        "latest_observation": {
            "updated_ts": latest.get("updated_ts", 0),
            "market_bias": latest.get("market_bias", ""),
            "market_state": latest.get("market_state", ""),
            "features": latest.get("features", {}),
        },
        "observation_status": (
            "insufficient_history"
            if not ready
            else "recent_drift_observed"
            if recent_events
            else "no_recent_drift"
        ),
    }


def write_report(path, report):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def _run_once(args):
    rows = load_shadow_rows(
        args.input,
        min_policy_version=max(0, args.min_policy_version),
    )
    feature_rows = build_feature_rows(rows)
    report = analyze_feature_rows(
        feature_rows,
        detector_factory=_river_detector_factory(max(1e-12, min(1.0, args.delta))),
        min_rows=args.min_rows,
        recent_window_rows=args.recent_window_rows,
    )
    report["input"] = str(Path(args.input))
    report["output"] = str(Path(args.output))
    report["min_policy_version"] = max(0, args.min_policy_version)
    report["delta"] = max(1e-12, min(1.0, args.delta))
    write_report(args.output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--min-policy-version", type=int, default=8)
    parser.add_argument("--min-rows", type=int, default=48)
    parser.add_argument("--recent-window-rows", type=int, default=48)
    parser.add_argument("--delta", type=float, default=0.002)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-sec", type=float, default=900.0)
    args = parser.parse_args()

    if not args.watch:
        report = _run_once(args)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ready"] else 1

    stop_event = threading.Event()

    def request_stop(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    interval_sec = max(60.0, float(args.interval_sec))
    while not stop_event.is_set():
        try:
            report = _run_once(args)
            print(
                "regime-drift "
                f"rows={report['rows_analyzed']} ready={str(report['ready']).lower()} "
                f"recent={str(report['drift_detected']).lower()} "
                f"status={report['observation_status']}",
                flush=True,
            )
        except Exception as exc:
            print(f"regime-drift error={type(exc).__name__}: {exc}", flush=True)
        stop_event.wait(interval_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
