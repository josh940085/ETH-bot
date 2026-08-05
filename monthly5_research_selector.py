"""Read monthly5 research selector artifacts for live shadow diagnostics."""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import monthly5_shadow


DEFAULT_SPEC_PATH = Path("docs/strategy_specs/monthly5_postlock_hourly.json")


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


def _safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _load_json(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _current_month(now_ts=None, tz_name="Asia/Taipei"):
    ts = time.time() if now_ts is None else _safe_float(now_ts, time.time())
    return datetime.fromtimestamp(ts, ZoneInfo(tz_name)).strftime("%Y-%m")


def parse_top_pick(top_pick):
    key = str(top_pick or "").strip()
    parts = [part for part in key.split("|") if part]
    if not parts:
        return {
            "top_pick": "",
            "valid": False,
            "direction_mode": "",
            "primary_direction": "wait",
            "max_leverage": 0,
        }
    family = parts[0]
    direction_mode = ""
    if "_" in family:
        direction_mode = family.rsplit("_", 1)[-1]
    if direction_mode == "lf":
        primary_direction = "long"
        direction_label = "long_flat"
    elif direction_mode == "ls":
        primary_direction = "long_or_short"
        direction_label = "long_short"
    elif family == "buy_hold":
        primary_direction = "long"
        direction_label = "long_only"
    else:
        primary_direction = "wait"
        direction_label = "unknown"

    leverage = 0
    target_pct = 0.0
    recovery_scale = 0.0
    stop_pct = None
    for part in parts[1:]:
        if part.startswith("lev"):
            leverage = _safe_int(part[3:], 0)
        elif part.startswith("target"):
            target_pct = _safe_float(part[6:], 0.0) * 100.0
        elif part.startswith("redlev"):
            recovery_scale = _safe_float(part[6:], 0.0)
        elif part.startswith("stop"):
            raw = part[4:]
            stop_pct = None if raw == "None" else _safe_float(raw, 0.0) * 100.0

    return {
        "top_pick": key,
        "valid": bool(key),
        "family": family,
        "direction_mode": direction_mode,
        "direction_label": direction_label,
        "primary_direction": primary_direction,
        "max_leverage": min(5, max(0, leverage)),
        "target_pct": round(target_pct, 4),
        "stop_pct": stop_pct if stop_pct is None else round(stop_pct, 4),
        "recovery_exposure_scale": round(recovery_scale, 4),
    }


def build_research_selector_probe(spec_path=DEFAULT_SPEC_PATH, *, now_ts=None):
    spec = _load_json(spec_path)
    evidence = spec.get("backtest_evidence") if isinstance(spec.get("backtest_evidence"), dict) else {}
    candidate = str(evidence.get("candidate_name") or monthly5_shadow.SELECTED_CANDIDATE)
    source_monthly = Path(str(evidence.get("source_monthly") or ""))
    monthly = _load_json(source_monthly)
    rows = monthly.get(candidate)
    rows = rows if isinstance(rows, list) else []
    month = _current_month(now_ts)
    selected = next((row for row in rows if isinstance(row, dict) and str(row.get("month") or "") == month), None)
    stale = False
    if selected is None and rows:
        selected = next((row for row in reversed(rows) if isinstance(row, dict)), None)
        stale = True
    selected = selected if isinstance(selected, dict) else {}
    parsed = parse_top_pick(selected.get("top_pick"))
    return {
        "schema_version": 1,
        "selector_source": monthly5_shadow.RESEARCH_SELECTOR_SOURCE,
        "selected_candidate": candidate,
        "source_monthly": str(source_monthly),
        "month": str(selected.get("month") or month),
        "current_month": month,
        "stale": stale or str(selected.get("month") or "") != month,
        "artifact_available": bool(selected),
        "return_pct": round(_safe_float(selected.get("return_pct"), 0.0), 4),
        "flat_time_pct": round(_safe_float(selected.get("flat_time_pct"), 0.0), 4),
        "lock_hit_day": selected.get("lock_hit_day"),
        "recovery_used": bool(selected.get("recovery_used", False)),
        **parsed,
    }
