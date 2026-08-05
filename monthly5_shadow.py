import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


STRATEGY_ID = "monthly5_postlock_hourly_v0"
SELECTED_CANDIDATE = "postlock_scale0.15_floor_pdaystopNone"
SELECTOR_POLICY_VERSION = 3
MONTHLY_LOCK_PCT = 5.0
MONTHLY_TARGET_PCT = 5.0
MONTHLY_PROJECTION_HOURS = 24.0 * 30.0
ROLLING_WINDOW_HOURS = 24.0
RECOVERY_PROBE_EXPOSURE_CAP = 0.15
UNDERPERFORMING_MICRO_PROBE_EXPOSURE_CAP = 0.05
MIXED_BIAS_PROBE_EXPOSURE_CAP = 0.10
MIXED_BIAS_PROBE_MIN_SCORE = 1.8
MIXED_BIAS_PROBE_MAX_GAP = 0.6
SUPPRESSED_RECOVERY_MIN_INTERVALS = 12
RECOVERY_PROBE_MIN_INTERVALS = 12
MONTHLY_RECOVERY_TRIGGER_PCT = -8.0
INTRADAY_STOP_PCT = -8.0
POST_LOCK_EXPOSURE_SCALE = 0.15
RECOVERY_EXPOSURE_SCALE = 0.5
NORMAL_EXPOSURE_SCALE = 1.0


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


def _safe_pct(current, start):
    current = _safe_float(current, 0.0)
    start = _safe_float(start, 0.0)
    if start <= 0:
        return 0.0
    return ((current / start) - 1.0) * 100.0


def _datetime_from_ts(now_ts=None, tz_name="Asia/Taipei"):
    ts = time.time() if now_ts is None else _safe_float(now_ts, time.time())
    return datetime.fromtimestamp(ts, ZoneInfo(tz_name))


def month_key_from_ts(now_ts=None, tz_name="Asia/Taipei"):
    return _datetime_from_ts(now_ts, tz_name).strftime("%Y-%m")


def day_key_from_ts(now_ts=None, tz_name="Asia/Taipei"):
    return _datetime_from_ts(now_ts, tz_name).strftime("%Y-%m-%d")


def load_state(path):
    state_path = Path(path)
    if not state_path.exists():
        return {}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def save_state(path, payload):
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_name(f".{state_path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp_path, state_path)


def _last_jsonl_record(path):
    history_path = Path(path)
    if not history_path.exists():
        return {}
    try:
        lines = history_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return {}
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def load_history(path):
    history_path = Path(path)
    if not history_path.exists():
        return []
    rows = []
    try:
        lines = history_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def build_history_record(snapshot, guard=None):
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    selection = (
        snapshot.get("market_selection")
        if isinstance(snapshot.get("market_selection"), dict)
        else {}
    )
    guard = guard if isinstance(guard, dict) else {}
    return {
        "schema_version": 1,
        "strategy_id": str(snapshot.get("strategy_id") or STRATEGY_ID),
        "selected_candidate": str(snapshot.get("selected_candidate") or SELECTED_CANDIDATE),
        "shadow_only": bool(snapshot.get("shadow_only", True)),
        "updated_ts": _safe_int(snapshot.get("updated_ts"), int(time.time())),
        "month_key": str(snapshot.get("month_key") or ""),
        "day_key": str(snapshot.get("day_key") or ""),
        "mode": str(snapshot.get("mode") or ""),
        "monthly_pnl_pct": round(_safe_float(snapshot.get("monthly_pnl_pct"), 0.0), 4),
        "intraday_pnl_pct": round(_safe_float(snapshot.get("intraday_pnl_pct"), 0.0), 4),
        "equity_valid": bool(snapshot.get("equity_valid", True)),
        "lock_reached": bool(snapshot.get("lock_reached", False)),
        "floor_guard_required": bool(snapshot.get("floor_guard_required", False)),
        "intraday_stop_active": bool(snapshot.get("intraday_stop_active", False)),
        "recovery_active": bool(snapshot.get("recovery_active", False)),
        "suggested_exposure_scale": round(_safe_float(snapshot.get("suggested_exposure_scale"), 0.0), 4),
        "max_leverage": min(5, max(0, _safe_int(snapshot.get("max_leverage"), 5))),
        "position_open": bool(snapshot.get("position_open", False)),
        "position_side": str(snapshot.get("position_side") or ""),
        "position_notional": round(_safe_float(snapshot.get("position_notional"), 0.0), 4),
        "mark_price": round(_safe_float(snapshot.get("mark_price"), 0.0), 4),
        "market_bias": str(selection.get("market_bias") or ""),
        "market_action": str(selection.get("market_action") or ""),
        "bull_score": round(_safe_float(selection.get("bull_score"), 0.0), 4),
        "bear_score": round(_safe_float(selection.get("bear_score"), 0.0), 4),
        "market_state": str(selection.get("market_state") or ""),
        "selector_policy_version": _safe_int(
            selection.get("selector_policy_version"),
            _safe_int(snapshot.get("selector_policy_version"), 0),
        ),
        "selected_plan": str(selection.get("selected_plan") or ""),
        "shadow_action": str(selection.get("shadow_action") or ""),
        "exposure_cap": round(_safe_float(selection.get("exposure_cap"), 0.0), 4),
        "suppressed_plan": str(selection.get("suppressed_plan") or ""),
        "suppressed_action": str(selection.get("suppressed_action") or ""),
        "suppressed_key": str(selection.get("suppressed_key") or ""),
        "suppressed_exposure_cap": round(_safe_float(selection.get("suppressed_exposure_cap"), 0.0), 4),
        "recovery_probe": bool(selection.get("recovery_probe", False)),
        "recovery_probe_key": str(selection.get("recovery_probe_key") or ""),
        "strategy_signal": str(selection.get("strategy_signal") or ""),
        "reason_codes": sorted(str(code) for code in selection.get("reason_codes") or [] if str(code)),
        "guard_allowed": bool(guard.get("allowed", True)),
        "guard_reason_code": str(guard.get("reason_code") or ""),
        "guard_adjusted_size": round(_safe_float(guard.get("adjusted_size"), 0.0), 4),
    }


def _active_shadow_action(row):
    return str((row or {}).get("shadow_action") or "") in {
        "evaluate_long",
        "evaluate_short",
        "reduced_exposure",
    }


def _open_position_wait_row(row):
    if not isinstance(row, dict):
        return False
    side = str(row.get("position_side") or "").lower()
    if side not in {"", "long", "short"}:
        return False
    if not (bool(row.get("position_open", False)) or _safe_float(row.get("position_notional"), 0.0) > 0.0):
        return False
    action = str(row.get("shadow_action") or "")
    plan = str(row.get("selected_plan") or "")
    return action == "wait" and plan in {"", "normal_wait"}


def _selection_from_active_row(row):
    if not isinstance(row, dict) or not _active_shadow_action(row):
        return {}
    return {
        "market_bias": str(row.get("market_bias") or ""),
        "market_action": str(row.get("market_action") or ""),
        "bull_score": round(_safe_float(row.get("bull_score"), 0.0), 4),
        "bear_score": round(_safe_float(row.get("bear_score"), 0.0), 4),
        "market_state": str(row.get("market_state") or ""),
        "selector_policy_version": _safe_int(row.get("selector_policy_version"), 0),
        "selected_plan": str(row.get("selected_plan") or ""),
        "shadow_action": str(row.get("shadow_action") or ""),
        "exposure_cap": round(_safe_float(row.get("exposure_cap"), 0.0), 4),
        "reason_codes": list(row.get("reason_codes") or []),
        "strategy_signal": str(row.get("strategy_signal") or row.get("position_side") or "").lower(),
    }


def _apply_active_selection(row, selection):
    if not isinstance(row, dict) or not isinstance(selection, dict) or not selection:
        return row
    patched = dict(row)
    for key in (
        "market_bias",
        "market_action",
        "bull_score",
        "bear_score",
        "market_state",
        "selector_policy_version",
        "selected_plan",
        "shadow_action",
        "exposure_cap",
        "reason_codes",
    ):
        patched[key] = selection.get(key, patched.get(key))
    side = str(patched.get("position_side") or "").lower()
    selection_signal = str(selection.get("strategy_signal") or "").lower()
    if side not in {"long", "short"} and selection_signal in {"long", "short"}:
        patched["position_side"] = selection_signal
        side = selection_signal
    patched["strategy_signal"] = side if side in {"long", "short"} else str(selection_signal or patched.get("strategy_signal") or "")
    return patched


def _carry_forward_active_position_rows(rows):
    timed_rows = sorted(
        (
            (idx, row)
            for idx, row in enumerate(rows or [])
            if isinstance(row, dict)
        ),
        key=lambda item: (_safe_int(item[1].get("updated_ts"), 0), item[0]),
    )
    last_selection_by_side = {}
    last_selection_any = {}
    patched_by_idx = {}
    for idx, row in timed_rows:
        side = str(row.get("position_side") or "").lower()
        if _open_position_wait_row(row):
            selection = last_selection_by_side.get(side) if side in {"long", "short"} else {}
            row = _apply_active_selection(row, selection or last_selection_any)
            side = str(row.get("position_side") or "").lower()
        if _active_shadow_action(row) and side in {"long", "short"}:
            selection = _selection_from_active_row(row)
            last_selection_by_side[side] = selection
            last_selection_any = selection
        patched_by_idx[idx] = row
    return [patched_by_idx.get(idx, row) for idx, row in enumerate(rows or [])]


def _latest_active_selection(rows, side):
    side = str(side or "").lower()
    timed_rows = sorted(
        (row for row in rows or [] if isinstance(row, dict)),
        key=lambda row: _safe_int(row.get("updated_ts"), 0),
        reverse=True,
    )
    for row in timed_rows:
        row_side = str(row.get("position_side") or "").lower()
        if _active_shadow_action(row) and (side not in {"long", "short"} or row_side == side):
            return _selection_from_active_row(row)
    return {}


def append_history(path, snapshot, guard=None, *, min_interval_sec=300):
    record = build_history_record(snapshot, guard)
    history_path = Path(path)
    last = _last_jsonl_record(history_path)
    if _open_position_wait_row(record):
        side = str(record.get("position_side") or "").lower()
        selection = _selection_from_active_row(last)
        if str(last.get("position_side") or "").lower() != side or not selection:
            selection = _latest_active_selection(load_history(history_path), side)
        record = _apply_active_selection(record, selection)
    min_interval = max(0, _safe_int(min_interval_sec, 300))
    if last:
        elapsed = _safe_int(record.get("updated_ts"), 0) - _safe_int(last.get("updated_ts"), 0)
        same_state = all(
            record.get(key) == last.get(key)
            for key in (
                "mode",
                "selector_policy_version",
                "selected_plan",
                "shadow_action",
                "exposure_cap",
                "market_bias",
                "market_state",
                "guard_allowed",
                "guard_reason_code",
            )
        )
        if same_state and elapsed < min_interval:
            return False
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return True


def _shadow_paper_direction(row):
    action = str(row.get("shadow_action") or "").lower()
    if action == "evaluate_long":
        return 1
    if action == "evaluate_short":
        return -1
    if action != "reduced_exposure":
        return 0
    hints = [
        str(row.get("position_side") or "").lower(),
        str(row.get("strategy_signal") or "").lower(),
        str(row.get("selected_plan") or "").lower(),
    ]
    if any("short" in hint for hint in hints):
        return -1
    if any("long" in hint for hint in hints):
        return 1
    return 0


def _shadow_paper_intervals(rows, *, suppressed=False, recovery_probe=False):
    timed_rows = sorted(
        (row for row in rows if isinstance(row, dict) and _safe_int(row.get("updated_ts"), 0) > 0),
        key=lambda row: _safe_int(row.get("updated_ts"), 0),
    )
    intervals = []
    for idx in range(len(timed_rows) - 1):
        row = timed_rows[idx]
        next_row = timed_rows[idx + 1]
        if row.get("guard_allowed") is False:
            continue
        if recovery_probe and not bool(row.get("recovery_probe", False)):
            continue
        direction = _shadow_paper_direction(row)
        selected_plan = str(row.get("selected_plan") or "")
        shadow_action = str(row.get("shadow_action") or "")
        exposure_cap = max(0.0, min(1.0, _safe_float(row.get("exposure_cap"), 0.0)))
        if suppressed:
            shadow_action = str(row.get("suppressed_action") or "")
            selected_plan = str(row.get("suppressed_plan") or "")
            exposure_cap = max(0.0, min(1.0, _safe_float(row.get("suppressed_exposure_cap"), 0.0)))
            direction = 1 if shadow_action == "evaluate_long" else -1 if shadow_action == "evaluate_short" else 0
        leverage = max(0.0, min(5.0, _safe_float(row.get("max_leverage"), 5.0)))
        start_price = _safe_float(row.get("mark_price"), 0.0)
        end_price = _safe_float(next_row.get("mark_price"), 0.0)
        if direction == 0 or exposure_cap <= 0.0 or leverage <= 0.0 or start_price <= 0.0 or end_price <= 0.0:
            continue
        interval_pct = direction * ((end_price / start_price) - 1.0) * 100.0 * exposure_cap * leverage
        intervals.append(
            {
                "return_pct": interval_pct,
                "direction": direction,
                "selected_plan": selected_plan,
                "shadow_action": shadow_action,
                "market_bias": str(row.get("market_bias") or ""),
                "market_state": str(row.get("market_state") or ""),
            }
        )
    return intervals


def _build_shadow_paper_return(rows, *, suppressed=False, recovery_probe=False):
    intervals = _shadow_paper_intervals(rows, suppressed=suppressed, recovery_probe=recovery_probe)
    total_return_pct = 0.0
    cumulative_pct = 0.0
    peak_pct = 0.0
    max_drawdown_pct = 0.0
    interval_count = 0
    win_count = 0
    loss_count = 0
    long_count = 0
    short_count = 0
    last_interval_pct = 0.0
    for interval in intervals:
        interval_pct = _safe_float(interval.get("return_pct"), 0.0)
        total_return_pct += interval_pct
        cumulative_pct += interval_pct
        peak_pct = max(peak_pct, cumulative_pct)
        max_drawdown_pct = min(max_drawdown_pct, cumulative_pct - peak_pct)
        interval_count += 1
        if interval_pct > 0:
            win_count += 1
        elif interval_pct < 0:
            loss_count += 1
        if _safe_int(interval.get("direction"), 0) > 0:
            long_count += 1
        else:
            short_count += 1
        last_interval_pct = interval_pct
    win_rate_pct = (win_count / interval_count) * 100.0 if interval_count else 0.0
    return {
        "shadow_paper_return_pct": round(total_return_pct, 4),
        "shadow_paper_max_drawdown_pct": round(max_drawdown_pct, 4),
        "shadow_paper_intervals": interval_count,
        "shadow_paper_win_intervals": win_count,
        "shadow_paper_loss_intervals": loss_count,
        "shadow_paper_win_rate_pct": round(win_rate_pct, 4),
        "shadow_paper_long_intervals": long_count,
        "shadow_paper_short_intervals": short_count,
        "shadow_paper_last_interval_pct": round(last_interval_pct, 4),
    }


def _build_grouped_paper_return(rows, *, min_intervals=12, max_groups=5, suppressed=False, recovery_probe=False):
    grouped = defaultdict(lambda: {"return_pct": 0.0, "intervals": 0, "wins": 0, "losses": 0})
    for interval in _shadow_paper_intervals(rows, suppressed=suppressed, recovery_probe=recovery_probe):
        key = (
            str(interval.get("selected_plan") or ""),
            str(interval.get("shadow_action") or ""),
            str(interval.get("market_bias") or ""),
            str(interval.get("market_state") or ""),
        )
        item = grouped[key]
        interval_pct = _safe_float(interval.get("return_pct"), 0.0)
        item["return_pct"] += interval_pct
        item["intervals"] += 1
        if interval_pct > 0:
            item["wins"] += 1
        elif interval_pct < 0:
            item["losses"] += 1
    groups = []
    for key, item in grouped.items():
        intervals = max(0, _safe_int(item.get("intervals"), 0))
        win_rate_pct = (_safe_int(item.get("wins"), 0) / intervals) * 100.0 if intervals else 0.0
        groups.append(
            {
                "key": "|".join(key),
                "selected_plan": key[0],
                "shadow_action": key[1],
                "market_bias": key[2],
                "market_state": key[3],
                "return_pct": round(_safe_float(item.get("return_pct"), 0.0), 4),
                "intervals": intervals,
                "win_rate_pct": round(win_rate_pct, 4),
                "wins": _safe_int(item.get("wins"), 0),
                "losses": _safe_int(item.get("losses"), 0),
            }
        )
    groups.sort(key=lambda item: (item["return_pct"], -item["intervals"]))
    weak = [
        item
        for item in groups
        if item["intervals"] >= max(1, _safe_int(min_intervals, 12))
        and item["return_pct"] < 0.0
    ]
    recovering = [
        item
        for item in groups
        if item["intervals"] >= max(1, _safe_int(min_intervals, 12))
        and item["return_pct"] > 0.0
    ]
    return {
        "shadow_grouped_paper_returns": groups[: max(1, _safe_int(max_groups, 5))],
        "shadow_underperforming_plan_keys": [item["key"] for item in weak[: max(1, _safe_int(max_groups, 5))]],
        "shadow_underperforming_plan_count": len(weak),
        "shadow_recovering_plan_keys": [item["key"] for item in recovering[: max(1, _safe_int(max_groups, 5))]],
        "shadow_recovering_plan_count": len(recovering),
        "shadow_group_min_intervals": max(1, _safe_int(min_intervals, 12)),
    }


def _candidate_probe_keys(grouped_paper, *, min_intervals=None, min_win_rate_pct=50.0):
    min_intervals = max(
        1,
        _safe_int(
            min_intervals,
            max(1, RECOVERY_PROBE_MIN_INTERVALS // 2),
        ),
    )
    min_win_rate_pct = max(0.0, min(100.0, _safe_float(min_win_rate_pct, 50.0)))
    candidates = []
    for item in grouped_paper.get("shadow_grouped_paper_returns") or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "")
        if not key:
            continue
        if _safe_int(item.get("intervals"), 0) < min_intervals:
            continue
        if _safe_float(item.get("return_pct"), 0.0) <= 0.0:
            continue
        if _safe_float(item.get("win_rate_pct"), 0.0) < min_win_rate_pct:
            continue
        candidates.append(key)
    return candidates


def _weighted_shadow_time_groups(timed_rows, *, now_ts=None, max_groups=8):
    groups = defaultdict(lambda: {"duration_sec": 0.0, "active": False, "rows": 0})
    total_sec = 0.0
    rows = list(timed_rows or [])
    if len(rows) < 2:
        return {"active": [], "flat": []}
    end_ts = max(_safe_float(now_ts, time.time()), _safe_int(rows[-1].get("updated_ts"), 0))
    for idx, row in enumerate(rows):
        start_ts = _safe_int(row.get("updated_ts"), 0)
        next_ts = _safe_int(rows[idx + 1].get("updated_ts"), 0) if idx + 1 < len(rows) else end_ts
        duration = max(0.0, next_ts - start_ts)
        if duration <= 0:
            continue
        action = str(row.get("shadow_action") or "")
        exposure_cap = _safe_float(row.get("exposure_cap"), 0.0)
        active = action in {"evaluate_long", "evaluate_short", "reduced_exposure"} and exposure_cap > 0.0 and row.get("guard_allowed") is not False
        key = (
            str(row.get("selected_plan") or ""),
            action,
            str(row.get("market_bias") or ""),
            str(row.get("market_state") or ""),
        )
        groups[key]["duration_sec"] += duration
        groups[key]["active"] = active
        groups[key]["rows"] += 1
        total_sec += duration
    active_groups = []
    flat_groups = []
    for key, item in groups.items():
        duration = _safe_float(item.get("duration_sec"), 0.0)
        payload = {
            "key": "|".join(key),
            "selected_plan": key[0],
            "shadow_action": key[1],
            "market_bias": key[2],
            "market_state": key[3],
            "duration_sec": round(duration, 4),
            "time_pct": round((duration / total_sec) * 100.0, 4) if total_sec > 0 else 0.0,
            "rows": _safe_int(item.get("rows"), 0),
        }
        if item.get("active"):
            active_groups.append(payload)
        else:
            flat_groups.append(payload)
    active_groups.sort(key=lambda item: item["duration_sec"], reverse=True)
    flat_groups.sort(key=lambda item: item["duration_sec"], reverse=True)
    max_groups = max(1, _safe_int(max_groups, 8))
    return {
        "active": active_groups[:max_groups],
        "flat": flat_groups[:max_groups],
    }


def _build_shadow_monthly_projection(shadow_paper, span_hours, *, min_projection_span_hours=24.0):
    span_hours = max(0.0, _safe_float(span_hours, 0.0))
    min_projection_span_hours = max(0.0, _safe_float(min_projection_span_hours, 24.0))
    paper_return_pct = _safe_float(shadow_paper.get("shadow_paper_return_pct"), 0.0)
    paper_intervals = _safe_int(shadow_paper.get("shadow_paper_intervals"), 0)
    projection_valid = span_hours >= min_projection_span_hours and paper_intervals > 0
    if span_hours <= 0.0 or paper_intervals <= 0:
        projected_monthly_pct = 0.0
    else:
        projected_monthly_pct = paper_return_pct * (MONTHLY_PROJECTION_HOURS / span_hours)
    target_progress_pct = (
        (projected_monthly_pct / MONTHLY_TARGET_PCT) * 100.0
        if MONTHLY_TARGET_PCT > 0
        else 0.0
    )
    observed_target_return_pct = (
        MONTHLY_TARGET_PCT * (span_hours / MONTHLY_PROJECTION_HOURS)
        if MONTHLY_PROJECTION_HOURS > 0
        else 0.0
    )
    observed_target_gap_pct = observed_target_return_pct - paper_return_pct
    return {
        "shadow_projected_monthly_return_pct": round(projected_monthly_pct, 4),
        "shadow_monthly_target_pct": round(MONTHLY_TARGET_PCT, 4),
        "shadow_monthly_projection_valid": projection_valid,
        "shadow_monthly_target_met": projection_valid and projected_monthly_pct >= MONTHLY_TARGET_PCT,
        "shadow_monthly_target_progress_pct": round(target_progress_pct, 4),
        "shadow_observed_target_return_pct": round(observed_target_return_pct, 4),
        "shadow_observed_target_gap_pct": round(observed_target_gap_pct, 4),
        "shadow_monthly_projection_hours": round(MONTHLY_PROJECTION_HOURS, 4),
        "shadow_monthly_min_projection_span_hours": round(min_projection_span_hours, 4),
    }


def _prefix_metrics(prefix, payload):
    prefixed = {}
    for key, value in payload.items():
        name = str(key)
        if name.startswith("shadow_"):
            name = name[len("shadow_") :]
        prefixed[f"{prefix}_{name}"] = value
    return prefixed


def _rolling_window_rows(rows, *, latest_ts, window_hours=ROLLING_WINDOW_HOURS):
    latest_ts = _safe_int(latest_ts, 0)
    if latest_ts <= 0:
        return []
    cutoff_ts = latest_ts - int(max(0.0, _safe_float(window_hours, ROLLING_WINDOW_HOURS)) * 3600)
    return [
        row
        for row in rows
        if isinstance(row, dict) and _safe_int(row.get("updated_ts"), 0) >= cutoff_ts
    ]


def _activation_rows(rows):
    timed_rows = sorted(
        (row for row in rows if isinstance(row, dict) and _safe_int(row.get("updated_ts"), 0) > 0),
        key=lambda row: _safe_int(row.get("updated_ts"), 0),
    )
    start_idx = None
    for idx, row in enumerate(timed_rows):
        action = str(row.get("shadow_action") or "")
        signal = str(row.get("strategy_signal") or "").lower()
        if (
            bool(row.get("position_open", False))
            and _safe_float(row.get("position_notional"), 0.0) > 0.0
            and signal in {"long", "short"}
            and action in {"evaluate_long", "evaluate_short", "reduced_exposure"}
        ):
            start_idx = idx
            break
    return timed_rows[start_idx:] if start_idx is not None else []


def build_readiness_report(
    rows,
    *,
    strategy_id=STRATEGY_ID,
    selected_candidate=SELECTED_CANDIDATE,
    min_records=48,
    min_span_hours=24.0,
    max_age_sec=900.0,
    max_flat_time_pct=None,
    min_selector_policy_version=SELECTOR_POLICY_VERSION,
    now_ts=None,
):
    failures = []
    warnings = []
    valid_rows = [row for row in rows if isinstance(row, dict)]
    if len(valid_rows) != len(rows or []):
        failures.append("history contains non-object rows")
    equity_shock_rows = [
        row
        for row in valid_rows
        if "equity_valid" not in row
        and (
            _safe_float(row.get("monthly_pnl_pct"), 0.0) <= -80.0
            or _safe_float(row.get("intraday_pnl_pct"), 0.0) <= -80.0
        )
    ]
    if equity_shock_rows:
        valid_rows = [row for row in valid_rows if row not in equity_shock_rows]
        warnings.append(f"ignored invalid legacy equity shock rows={len(equity_shock_rows)}")
    min_selector_policy_version = max(0, _safe_int(min_selector_policy_version, SELECTOR_POLICY_VERSION))
    legacy_selector_rows = [
        row
        for row in valid_rows
        if _safe_int(row.get("selector_policy_version"), 0) < min_selector_policy_version
    ]
    if legacy_selector_rows:
        valid_rows = [row for row in valid_rows if row not in legacy_selector_rows]
        warnings.append(
            "ignored legacy selector policy rows="
            f"{len(legacy_selector_rows)} < v{min_selector_policy_version}"
        )
    valid_rows = _carry_forward_active_position_rows(valid_rows)
    if not valid_rows:
        return {
            "status": "collecting",
            "ready": False,
            "promotion_ready": False,
            "promotion_blockers": [
                "sample_count",
                "sample_span",
                "no_evaluate_samples",
                "shadow_projection_not_valid",
                "shadow_rolling_projection_not_valid",
            ],
            "promotion_blocker_details": [
                {
                    "code": "sample_count",
                    "label": "樣本數",
                    "current": "0",
                    "target": str(max(1, _safe_int(min_records, 48))),
                    "remaining": max(1, _safe_int(min_records, 48)),
                    "detail": "等待目前 selector policy 的 shadow history rows",
                },
                {
                    "code": "sample_span",
                    "label": "樣本時間",
                    "current": 0.0,
                    "target": round(max(0.0, _safe_float(min_span_hours, 24.0)), 4),
                    "remaining_hours": round(max(0.0, _safe_float(min_span_hours, 24.0)), 4),
                    "ready_ts": 0,
                    "detail": "至少滿 24 小時才允許月化投影作為上線證據",
                },
                {
                    "code": "no_evaluate_samples",
                    "label": "評估樣本",
                    "current": 0,
                    "target": "evaluate_long/evaluate_short/reduced_exposure",
                    "detail": "沒有實際策略評估樣本時不可 promotion",
                },
                {
                    "code": "shadow_projection_not_valid",
                    "label": "月化投影有效性",
                    "current": 0.0,
                    "target": round(max(0.0, _safe_float(min_span_hours, 24.0)), 4),
                    "observed_target_gap_pct": 0.0,
                    "detail": "樣本時間不足時，shadow projected monthly return 只作參考不作 promotion 證據",
                },
                {
                    "code": "shadow_rolling_projection_not_valid",
                    "label": "滾動投影有效性",
                    "current": 0.0,
                    "target": round(max(0.0, _safe_float(min_span_hours, 24.0)), 4),
                    "observed_target_gap_pct": 0.0,
                    "detail": "滾動 24 小時樣本不足時不可 promotion",
                },
            ],
            "promotion_blocker_count": 5,
            "failures": failures,
            "warnings": warnings,
            "rows": 0,
            "span_hours": 0.0,
            "sample_count_remaining": max(1, _safe_int(min_records, 48)),
            "sample_count_progress_pct": 0.0,
            "sample_span_remaining_hours": round(max(0.0, _safe_float(min_span_hours, 24.0)), 4),
            "sample_span_progress_pct": 0.0,
            "sample_span_ready_ts": 0,
            "promotion_earliest_review_ts": 0,
            "latest_age_sec": 0.0,
            "ignored_legacy_selector_policy_rows": len(legacy_selector_rows),
            "min_selector_policy_version": min_selector_policy_version,
        }

    strategy_drift = [
        row
        for row in valid_rows
        if row.get("strategy_id") != strategy_id
        or row.get("selected_candidate") != selected_candidate
    ]
    if strategy_drift:
        failures.append(f"strategy/candidate drift rows={len(strategy_drift)}")

    unsafe_rows = [
        row
        for row in valid_rows
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
        age_sec = _safe_float(now_ts, time.time()) - timestamps[-1]
    if max_age_sec is not None and age_sec > _safe_float(max_age_sec, 0.0):
        failures.append(f"history stale: age={age_sec:.1f}s > {_safe_float(max_age_sec, 0.0):.1f}s")

    selected_plan_counts = Counter(str(row.get("selected_plan") or "") for row in valid_rows)
    action_counts = Counter(str(row.get("shadow_action") or "") for row in valid_rows)
    mode_counts = Counter(str(row.get("mode") or "") for row in valid_rows)
    market_bias_counts = Counter(str(row.get("market_bias") or "") for row in valid_rows)
    risk_rows = [
        row
        for row in valid_rows
        if row.get("mode") in {"intraday_stop", "post_lock_floor_guard", "post_lock", "recovery"}
        or row.get("guard_allowed") is False
    ]
    evaluate_rows = [
        row
        for row in valid_rows
        if str(row.get("shadow_action") or "") in {"evaluate_long", "evaluate_short", "reduced_exposure"}
    ]
    shadow_active_rows = [
        row
        for row in valid_rows
        if str(row.get("shadow_action") or "") in {"evaluate_long", "evaluate_short", "reduced_exposure"}
        and _safe_float(row.get("exposure_cap"), 0.0) > 0.0
        and row.get("guard_allowed") is not False
    ]
    open_rows = [
        row
        for row in valid_rows
        if bool(row.get("position_open", False)) or _safe_float(row.get("position_notional"), 0.0) > 0
    ]
    open_sample_pct = (len(open_rows) / len(valid_rows)) * 100.0
    flat_sample_pct = 100.0 - open_sample_pct
    shadow_active_sample_pct = (len(shadow_active_rows) / len(valid_rows)) * 100.0
    shadow_flat_sample_pct = 100.0 - shadow_active_sample_pct
    timed_rows = sorted(
        (row for row in valid_rows if _safe_int(row.get("updated_ts"), 0) > 0),
        key=lambda row: _safe_int(row.get("updated_ts"), 0),
    )
    weighted_total_sec = 0.0
    weighted_open_sec = 0.0
    weighted_flat_sec = 0.0
    weighted_shadow_active_sec = 0.0
    weighted_shadow_flat_sec = 0.0
    if len(timed_rows) >= 2:
        end_ts = max(_safe_float(now_ts, time.time()), _safe_int(timed_rows[-1].get("updated_ts"), 0))
        for idx, row in enumerate(timed_rows):
            start_ts = _safe_int(row.get("updated_ts"), 0)
            next_ts = _safe_int(timed_rows[idx + 1].get("updated_ts"), 0) if idx + 1 < len(timed_rows) else end_ts
            duration = max(0.0, next_ts - start_ts)
            if duration <= 0:
                continue
            weighted_total_sec += duration
            if bool(row.get("position_open", False)) or _safe_float(row.get("position_notional"), 0.0) > 0:
                weighted_open_sec += duration
            else:
                weighted_flat_sec += duration
            shadow_active = (
                str(row.get("shadow_action") or "") in {"evaluate_long", "evaluate_short", "reduced_exposure"}
                and _safe_float(row.get("exposure_cap"), 0.0) > 0.0
                and row.get("guard_allowed") is not False
            )
            if shadow_active:
                weighted_shadow_active_sec += duration
            else:
                weighted_shadow_flat_sec += duration
    if weighted_total_sec > 0:
        open_time_pct = (weighted_open_sec / weighted_total_sec) * 100.0
        flat_time_pct = (weighted_flat_sec / weighted_total_sec) * 100.0
        shadow_active_time_pct = (weighted_shadow_active_sec / weighted_total_sec) * 100.0
        shadow_flat_time_pct = (weighted_shadow_flat_sec / weighted_total_sec) * 100.0
    else:
        open_time_pct = open_sample_pct
        flat_time_pct = flat_sample_pct
        shadow_active_time_pct = shadow_active_sample_pct
        shadow_flat_time_pct = shadow_flat_sample_pct
    min_records = max(1, _safe_int(min_records, 48))
    min_span_hours = max(0.0, _safe_float(min_span_hours, 24.0))
    shadow_paper = _build_shadow_paper_return(valid_rows)
    shadow_projection = _build_shadow_monthly_projection(
        shadow_paper,
        span_hours,
        min_projection_span_hours=min_span_hours,
    )
    latest_ts = timestamps[-1] if timestamps else 0
    rolling_rows = _rolling_window_rows(valid_rows, latest_ts=latest_ts)
    rolling_timestamps = sorted(
        _safe_int(row.get("updated_ts"), 0)
        for row in rolling_rows
        if _safe_int(row.get("updated_ts"), 0) > 0
    )
    rolling_span_hours = (
        (rolling_timestamps[-1] - rolling_timestamps[0]) / 3600.0
        if len(rolling_timestamps) >= 2
        else 0.0
    )
    rolling_paper = _build_shadow_paper_return(rolling_rows)
    rolling_projection = _build_shadow_monthly_projection(
        rolling_paper,
        rolling_span_hours,
        min_projection_span_hours=min_span_hours,
    )
    shadow_time_groups = _weighted_shadow_time_groups(timed_rows, now_ts=now_ts)
    activation_rows = _activation_rows(valid_rows)
    activation_timestamps = sorted(
        _safe_int(row.get("updated_ts"), 0)
        for row in activation_rows
        if _safe_int(row.get("updated_ts"), 0) > 0
    )
    activation_span_hours = (
        (activation_timestamps[-1] - activation_timestamps[0]) / 3600.0
        if len(activation_timestamps) >= 2
        else 0.0
    )
    activation_paper = _build_shadow_paper_return(activation_rows)
    activation_projection = _build_shadow_monthly_projection(
        activation_paper,
        activation_span_hours,
        min_projection_span_hours=min_span_hours,
    )
    grouped_paper = _build_grouped_paper_return(valid_rows)
    rolling_grouped_paper = _build_grouped_paper_return(rolling_rows)
    rolling_suppressed_grouped_paper = _build_grouped_paper_return(
        rolling_rows,
        min_intervals=SUPPRESSED_RECOVERY_MIN_INTERVALS,
        suppressed=True,
    )
    rolling_probe_grouped_paper = _build_grouped_paper_return(
        rolling_rows,
        min_intervals=RECOVERY_PROBE_MIN_INTERVALS,
        recovery_probe=True,
    )
    suppressed_observed_intervals = sum(
        _safe_int(item.get("intervals"), 0)
        for item in rolling_suppressed_grouped_paper["shadow_grouped_paper_returns"]
    )
    suppressed_recovery_remaining_intervals = max(
        0,
        SUPPRESSED_RECOVERY_MIN_INTERVALS - suppressed_observed_intervals,
    )
    suppressed_recovery_progress_pct = (
        min(100.0, (suppressed_observed_intervals / SUPPRESSED_RECOVERY_MIN_INTERVALS) * 100.0)
        if SUPPRESSED_RECOVERY_MIN_INTERVALS > 0
        else 100.0
    )
    recovery_probe_observed_intervals = sum(
        _safe_int(item.get("intervals"), 0)
        for item in rolling_probe_grouped_paper["shadow_grouped_paper_returns"]
    )
    recovery_probe_candidate_min_intervals = max(1, RECOVERY_PROBE_MIN_INTERVALS // 2)
    probe_candidate_keys = _candidate_probe_keys(
        rolling_probe_grouped_paper,
        min_intervals=recovery_probe_candidate_min_intervals,
    )
    recovery_probe_remaining_intervals = max(
        0,
        RECOVERY_PROBE_MIN_INTERVALS - recovery_probe_observed_intervals,
    )
    recovery_probe_progress_pct = (
        min(100.0, (recovery_probe_observed_intervals / RECOVERY_PROBE_MIN_INTERVALS) * 100.0)
        if RECOVERY_PROBE_MIN_INTERVALS > 0
        else 100.0
    )
    recovering_keys = set(rolling_suppressed_grouped_paper["shadow_recovering_plan_keys"])
    probe_failed_keys = set(rolling_probe_grouped_paper["shadow_underperforming_plan_keys"])
    probe_success_keys = set(rolling_probe_grouped_paper["shadow_recovering_plan_keys"])
    if probe_failed_keys:
        recovery_probe_state = "probe_failed"
    elif probe_success_keys:
        recovery_probe_state = "probe_success"
    elif rolling_probe_grouped_paper["shadow_grouped_paper_returns"]:
        recovery_probe_state = "probing"
    elif recovering_keys:
        recovery_probe_state = "probe_ready"
    elif suppressed_observed_intervals > 0:
        recovery_probe_state = "collecting"
    else:
        recovery_probe_state = "idle"
    active_underperforming_keys = sorted(
        (
            set(rolling_grouped_paper["shadow_underperforming_plan_keys"])
            | probe_failed_keys
        )
        - recovering_keys
        - probe_success_keys
    )
    flat_cap = None if max_flat_time_pct is None else max(0.0, min(100.0, _safe_float(max_flat_time_pct, 100.0)))
    if len(valid_rows) < min_records:
        warnings.append(f"sample count collecting: rows={len(valid_rows)} < {min_records}")
    if span_hours < min_span_hours:
        warnings.append(f"sample span collecting: hours={span_hours:.2f} < {min_span_hours:.2f}")
    if not evaluate_rows:
        warnings.append("no evaluate_long/evaluate_short/reduced_exposure samples yet")
    if flat_cap is not None and shadow_flat_time_pct > flat_cap:
        warnings.append(f"shadow flat time pct high: {shadow_flat_time_pct:.2f}% > {flat_cap:.2f}%")
    if span_hours >= min_span_hours and shadow_paper["shadow_paper_intervals"] > 0 and shadow_paper["shadow_paper_return_pct"] < 0:
        warnings.append(f"shadow paper return negative: {shadow_paper['shadow_paper_return_pct']:.4f}%")
    if (
        shadow_projection["shadow_monthly_projection_valid"]
        and not shadow_projection["shadow_monthly_target_met"]
    ):
        warnings.append(
            "shadow monthly projection below target: "
            f"{shadow_projection['shadow_projected_monthly_return_pct']:.4f}% < "
            f"{shadow_projection['shadow_monthly_target_pct']:.4f}%"
        )
    if (
        rolling_projection["shadow_monthly_projection_valid"]
        and not rolling_projection["shadow_monthly_target_met"]
    ):
        warnings.append(
            "shadow rolling 24h projection below target: "
            f"{rolling_projection['shadow_projected_monthly_return_pct']:.4f}% < "
            f"{rolling_projection['shadow_monthly_target_pct']:.4f}%"
        )
    if active_underperforming_keys:
        warnings.append(
            "shadow active underperforming plan groups: "
            + ", ".join(active_underperforming_keys[:2])
        )

    flat_ok = flat_cap is None or shadow_flat_time_pct <= flat_cap
    ready = not failures and len(valid_rows) >= min_records and span_hours >= min_span_hours and bool(evaluate_rows) and flat_ok
    sample_count_remaining = max(0, min_records - len(valid_rows))
    sample_count_progress_pct = min(100.0, (len(valid_rows) / min_records) * 100.0) if min_records > 0 else 100.0
    sample_span_remaining_hours = max(0.0, min_span_hours - span_hours)
    sample_span_progress_pct = min(100.0, (span_hours / min_span_hours) * 100.0) if min_span_hours > 0 else 100.0
    sample_span_ready_ts = (
        latest_ts + int(round(sample_span_remaining_hours * 3600.0))
        if latest_ts > 0 and sample_span_remaining_hours > 0
        else 0
    )
    promotion_earliest_review_ts = sample_span_ready_ts if sample_span_ready_ts > 0 else 0
    promotion_blockers = []
    if failures:
        promotion_blockers.append("invalid_history")
    if len(valid_rows) < min_records:
        promotion_blockers.append("sample_count")
    if span_hours < min_span_hours:
        promotion_blockers.append("sample_span")
    if not evaluate_rows:
        promotion_blockers.append("no_evaluate_samples")
    if not flat_ok:
        promotion_blockers.append("shadow_flat_time_high")
    if not shadow_projection["shadow_monthly_projection_valid"]:
        promotion_blockers.append("shadow_projection_not_valid")
    elif not shadow_projection["shadow_monthly_target_met"]:
        promotion_blockers.append("shadow_monthly_target")
    if not rolling_projection["shadow_monthly_projection_valid"]:
        promotion_blockers.append("shadow_rolling_projection_not_valid")
    elif not rolling_projection["shadow_monthly_target_met"]:
        promotion_blockers.append("shadow_rolling_monthly_target")
    if active_underperforming_keys:
        promotion_blockers.append("active_underperforming_plan")
    if recovery_probe_state in {"collecting", "probe_ready", "probing", "probe_failed"}:
        promotion_blockers.append(f"recovery_probe_{recovery_probe_state}")

    def _blocker_detail(code):
        code = str(code or "")
        if code == "invalid_history":
            return {
                "code": code,
                "label": "歷史資料無效",
                "current": "; ".join(failures[:2]) if failures else "invalid",
                "target": "history valid",
                "detail": "修正月報酬5% shadow history 後才可 promotion",
            }
        if code == "sample_count":
            return {
                "code": code,
                "label": "樣本數",
                "current": str(len(valid_rows)),
                "target": str(min_records),
                "remaining": sample_count_remaining,
                "detail": "等待更多 shadow history rows",
            }
        if code == "sample_span":
            return {
                "code": code,
                "label": "樣本時間",
                "current": round(max(0.0, span_hours), 4),
                "target": round(max(0.0, min_span_hours), 4),
                "remaining_hours": round(max(0.0, sample_span_remaining_hours), 4),
                "ready_ts": sample_span_ready_ts,
                "detail": "至少滿 24 小時才允許月化投影作為上線證據",
            }
        if code == "no_evaluate_samples":
            return {
                "code": code,
                "label": "評估樣本",
                "current": len(evaluate_rows),
                "target": "evaluate_long/evaluate_short/reduced_exposure",
                "detail": "沒有實際策略評估樣本時不可 promotion",
            }
        if code == "shadow_flat_time_high":
            return {
                "code": code,
                "label": "shadow 空倉時間",
                "current": round(max(0.0, shadow_flat_time_pct), 4),
                "target": round(flat_cap, 4) if flat_cap is not None else None,
                "detail": "空倉時間需低於候選策略歷史平均空倉上限",
            }
        if code == "shadow_projection_not_valid":
            return {
                "code": code,
                "label": "月化投影有效性",
                "current": round(max(0.0, span_hours), 4),
                "target": round(max(0.0, min_span_hours), 4),
                "observed_target_gap_pct": shadow_projection["shadow_observed_target_gap_pct"],
                "detail": "樣本時間不足時，shadow projected monthly return 只作參考不作 promotion 證據",
            }
        if code == "shadow_monthly_target":
            return {
                "code": code,
                "label": "月化目標",
                "current": shadow_projection["shadow_projected_monthly_return_pct"],
                "target": shadow_projection["shadow_monthly_target_pct"],
                "observed_target_gap_pct": shadow_projection["shadow_observed_target_gap_pct"],
                "detail": "全量 shadow 月化投影需達 +5%",
            }
        if code == "shadow_rolling_projection_not_valid":
            return {
                "code": code,
                "label": "滾動投影有效性",
                "current": round(max(0.0, rolling_span_hours), 4),
                "target": round(max(0.0, min_span_hours), 4),
                "observed_target_gap_pct": rolling_projection["shadow_observed_target_gap_pct"],
                "detail": "滾動 24 小時樣本不足時不可 promotion",
            }
        if code == "shadow_rolling_monthly_target":
            return {
                "code": code,
                "label": "滾動月化目標",
                "current": rolling_projection["shadow_projected_monthly_return_pct"],
                "target": rolling_projection["shadow_monthly_target_pct"],
                "observed_target_gap_pct": rolling_projection["shadow_observed_target_gap_pct"],
                "detail": "最近 24 小時 shadow 月化投影需達 +5%",
            }
        if code == "active_underperforming_plan":
            return {
                "code": code,
                "label": "低效策略組",
                "current": ", ".join(active_underperforming_keys[:2]),
                "target": "no active underperforming plan",
                "detail": "近期仍虧損的市場組不可 promotion",
            }
        if code.startswith("recovery_probe_"):
            return {
                "code": code,
                "label": "恢復探測",
                "current": recovery_probe_state,
                "target": "probe_success or idle",
                "remaining_intervals": recovery_probe_remaining_intervals,
                "detail": "低效策略組需完成恢復探測後才可 promotion",
            }
        return {
            "code": code,
            "label": code,
            "current": "",
            "target": "",
            "detail": "",
        }

    promotion_blocker_details = [_blocker_detail(code) for code in promotion_blockers]
    promotion_ready = (
        ready
        and shadow_projection["shadow_monthly_target_met"]
        and rolling_projection["shadow_monthly_target_met"]
        and not promotion_blockers
    )
    status = "ready" if ready else "collecting"
    if failures:
        status = "invalid"
    return {
        "schema_version": 1,
        "status": status,
        "ready": ready,
        "promotion_ready": promotion_ready,
        "promotion_blockers": promotion_blockers,
        "promotion_blocker_details": promotion_blocker_details,
        "promotion_blocker_count": len(promotion_blockers),
        "failures": failures,
        "warnings": warnings,
        "rows": len(valid_rows),
        "ignored_legacy_selector_policy_rows": len(legacy_selector_rows),
        "min_selector_policy_version": min_selector_policy_version,
        "span_hours": round(max(0.0, span_hours), 4),
        "sample_count_remaining": sample_count_remaining,
        "sample_count_progress_pct": round(max(0.0, sample_count_progress_pct), 4),
        "sample_span_remaining_hours": round(max(0.0, sample_span_remaining_hours), 4),
        "sample_span_progress_pct": round(max(0.0, sample_span_progress_pct), 4),
        "sample_span_ready_ts": sample_span_ready_ts,
        "promotion_earliest_review_ts": promotion_earliest_review_ts,
        "latest_age_sec": round(max(0.0, age_sec), 1),
        "selected_plan_counts": dict(sorted(selected_plan_counts.items())),
        "shadow_action_counts": dict(sorted(action_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "market_bias_counts": dict(sorted(market_bias_counts.items())),
        "risk_rows": len(risk_rows),
        "evaluate_rows": len(evaluate_rows),
        "shadow_active_rows": len(shadow_active_rows),
        "shadow_flat_rows": len(valid_rows) - len(shadow_active_rows),
        "open_rows": len(open_rows),
        "flat_rows": len(valid_rows) - len(open_rows),
        "open_sample_pct": round(max(0.0, open_sample_pct), 4),
        "flat_sample_pct": round(max(0.0, flat_sample_pct), 4),
        "shadow_active_sample_pct": round(max(0.0, shadow_active_sample_pct), 4),
        "shadow_flat_sample_pct": round(max(0.0, shadow_flat_sample_pct), 4),
        "open_time_pct": round(max(0.0, open_time_pct), 4),
        "flat_time_pct": round(max(0.0, flat_time_pct), 4),
        "actual_open_time_pct": round(max(0.0, open_time_pct), 4),
        "actual_flat_time_pct": round(max(0.0, flat_time_pct), 4),
        "shadow_active_time_pct": round(max(0.0, shadow_active_time_pct), 4),
        "shadow_flat_time_pct": round(max(0.0, shadow_flat_time_pct), 4),
        "weighted_total_sec": round(max(0.0, weighted_total_sec), 4),
        "weighted_open_sec": round(max(0.0, weighted_open_sec), 4),
        "weighted_flat_sec": round(max(0.0, weighted_flat_sec), 4),
        "weighted_shadow_active_sec": round(max(0.0, weighted_shadow_active_sec), 4),
        "weighted_shadow_flat_sec": round(max(0.0, weighted_shadow_flat_sec), 4),
        "shadow_active_time_groups": list(shadow_time_groups["active"]),
        "shadow_flat_time_groups": list(shadow_time_groups["flat"]),
        **shadow_paper,
        **shadow_projection,
        "shadow_rolling_window_hours": round(ROLLING_WINDOW_HOURS, 4),
        "shadow_rolling_rows": len(rolling_rows),
        "shadow_rolling_span_hours": round(max(0.0, rolling_span_hours), 4),
        **_prefix_metrics("shadow_rolling", rolling_paper),
        **_prefix_metrics("shadow_rolling", rolling_projection),
        "shadow_activation_rows": len(activation_rows),
        "shadow_activation_span_hours": round(max(0.0, activation_span_hours), 4),
        **_prefix_metrics("shadow_activation", activation_paper),
        **_prefix_metrics("shadow_activation", activation_projection),
        **grouped_paper,
        "shadow_active_underperforming_plan_keys": list(active_underperforming_keys),
        "shadow_active_underperforming_plan_count": len(active_underperforming_keys),
        "shadow_active_grouped_paper_returns": list(rolling_grouped_paper["shadow_grouped_paper_returns"]),
        "shadow_suppressed_recovering_plan_keys": list(rolling_suppressed_grouped_paper["shadow_recovering_plan_keys"]),
        "shadow_suppressed_recovering_plan_count": rolling_suppressed_grouped_paper["shadow_recovering_plan_count"],
        "shadow_suppressed_grouped_paper_returns": list(rolling_suppressed_grouped_paper["shadow_grouped_paper_returns"]),
        "shadow_suppressed_recovery_min_intervals": SUPPRESSED_RECOVERY_MIN_INTERVALS,
        "shadow_suppressed_observed_intervals": suppressed_observed_intervals,
        "shadow_suppressed_recovery_remaining_intervals": suppressed_recovery_remaining_intervals,
        "shadow_suppressed_recovery_progress_pct": round(max(0.0, suppressed_recovery_progress_pct), 4),
        "shadow_recovery_probe_success_keys": list(probe_success_keys),
        "shadow_recovery_probe_success_count": len(probe_success_keys),
        "shadow_recovery_probe_candidate_keys": list(probe_candidate_keys),
        "shadow_recovery_probe_candidate_count": len(probe_candidate_keys),
        "shadow_recovery_probe_candidate_min_intervals": recovery_probe_candidate_min_intervals,
        "shadow_recovery_probe_failed_keys": list(probe_failed_keys),
        "shadow_recovery_probe_failed_count": len(probe_failed_keys),
        "shadow_recovery_probe_grouped_paper_returns": list(rolling_probe_grouped_paper["shadow_grouped_paper_returns"]),
        "shadow_recovery_probe_min_intervals": RECOVERY_PROBE_MIN_INTERVALS,
        "shadow_recovery_probe_observed_intervals": recovery_probe_observed_intervals,
        "shadow_recovery_probe_remaining_intervals": recovery_probe_remaining_intervals,
        "shadow_recovery_probe_progress_pct": round(max(0.0, recovery_probe_progress_pct), 4),
        "shadow_recovery_probe_state": recovery_probe_state,
        "min_records": min_records,
        "min_span_hours": min_span_hours,
        "max_flat_time_pct": flat_cap,
        "max_age_sec": max_age_sec,
    }


def _account_equity(*, wallet_balance, margin_balance, unrealized_pnl):
    margin = _safe_float(margin_balance, 0.0)
    if margin > 0:
        return margin, True
    wallet = _safe_float(wallet_balance, 0.0)
    if wallet <= 0:
        return 0.0, False
    unrealized = _safe_float(unrealized_pnl, 0.0)
    return max(0.0, wallet + unrealized), True


def update_shadow_state(
    previous,
    *,
    now_ts=None,
    mark_price=0.0,
    wallet_balance=0.0,
    margin_balance=0.0,
    unrealized_pnl=0.0,
    position_open=False,
    position_side="",
    position_notional=0.0,
    selected_candidate=None,
):
    now = int(time.time() if now_ts is None else _safe_float(now_ts, time.time()))
    month_key = month_key_from_ts(now)
    day_key = day_key_from_ts(now)
    previous = previous if isinstance(previous, dict) else {}
    raw_equity, equity_valid = _account_equity(
        wallet_balance=wallet_balance,
        margin_balance=margin_balance,
        unrealized_pnl=unrealized_pnl,
    )
    previous_equity = _safe_float(previous.get("current_equity"), 0.0)
    equity = round(raw_equity if equity_valid or previous_equity <= 0 else previous_equity, 8)

    month_changed = previous.get("month_key") != month_key
    day_changed = previous.get("day_key") != day_key
    previous_month_start = _safe_float(previous.get("month_start_equity"), 0.0)
    previous_day_start = _safe_float(previous.get("day_start_equity"), 0.0)
    month_start_equity = equity if month_changed or previous_month_start <= 0 else previous_month_start
    day_start_equity = equity if day_changed or previous_day_start <= 0 else previous_day_start

    monthly_pnl_pct = round(_safe_pct(equity, month_start_equity), 4)
    intraday_pnl_pct = round(_safe_pct(equity, day_start_equity), 4)
    previous_month_high = 0.0 if month_changed else _safe_float(previous.get("month_high_pnl_pct"), 0.0)
    month_high_pnl_pct = round(max(previous_month_high, monthly_pnl_pct), 4)
    previous_lock_reached = False if month_changed else bool(previous.get("lock_reached"))
    lock_reached = previous_lock_reached or monthly_pnl_pct >= MONTHLY_LOCK_PCT
    intraday_stop_active = intraday_pnl_pct <= INTRADAY_STOP_PCT
    recovery_active = monthly_pnl_pct <= MONTHLY_RECOVERY_TRIGGER_PCT
    floor_guard_required = lock_reached and monthly_pnl_pct < MONTHLY_LOCK_PCT

    reasons = []
    if not equity_valid:
        reasons.append("equity_unavailable")
    if intraday_stop_active:
        reasons.append("intraday_stop")
    if recovery_active:
        reasons.append("monthly_recovery")
    if lock_reached:
        reasons.append("monthly_lock_reached")
    if floor_guard_required:
        reasons.append("post_lock_floor_guard")

    if floor_guard_required:
        mode = "post_lock_floor_guard"
        suggested_exposure_scale = 0.0
    elif intraday_stop_active:
        mode = "intraday_stop"
        suggested_exposure_scale = 0.0
    elif monthly_pnl_pct >= MONTHLY_LOCK_PCT or lock_reached:
        mode = "post_lock"
        suggested_exposure_scale = POST_LOCK_EXPOSURE_SCALE
    elif recovery_active:
        mode = "recovery"
        suggested_exposure_scale = RECOVERY_EXPOSURE_SCALE
    else:
        mode = "normal"
        suggested_exposure_scale = NORMAL_EXPOSURE_SCALE

    return {
        "schema_version": 1,
        "enabled": True,
        "shadow_only": True,
        "strategy_id": STRATEGY_ID,
        "selected_candidate": str(selected_candidate or previous.get("selected_candidate") or SELECTED_CANDIDATE),
        "mode": mode,
        "reason_codes": reasons,
        "month_key": month_key,
        "day_key": day_key,
        "month_start_equity": round(month_start_equity, 8),
        "day_start_equity": round(day_start_equity, 8),
        "current_equity": equity,
        "equity_valid": bool(equity_valid),
        "monthly_pnl_pct": monthly_pnl_pct,
        "intraday_pnl_pct": intraday_pnl_pct,
        "month_high_pnl_pct": month_high_pnl_pct,
        "lock_reached": lock_reached,
        "floor_guard_required": floor_guard_required,
        "intraday_stop_active": intraday_stop_active,
        "recovery_active": recovery_active,
        "suggested_exposure_scale": round(suggested_exposure_scale, 4),
        "max_leverage": 5,
        "monthly_lock_pct": MONTHLY_LOCK_PCT,
        "monthly_recovery_trigger_pct": MONTHLY_RECOVERY_TRIGGER_PCT,
        "intraday_stop_pct": INTRADAY_STOP_PCT,
        "post_lock_exposure_scale": POST_LOCK_EXPOSURE_SCALE,
        "recovery_exposure_scale": RECOVERY_EXPOSURE_SCALE,
        "position_open": bool(position_open),
        "position_side": str(position_side or ""),
        "position_notional": round(max(0.0, _safe_float(position_notional, 0.0)), 4),
        "mark_price": round(max(0.0, _safe_float(mark_price, 0.0)), 4),
        "updated_ts": now,
    }


def _safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _direction_score(value):
    value = _safe_float(value, 0.0)
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def build_market_selection(
    shadow_state,
    *,
    strategy_signal="wait",
    strategy_execution_reason="",
    strategy_context=None,
    host_logic=None,
    macro_alignment=None,
    donchian_state=None,
    underperforming_plan_keys=None,
    recovering_plan_keys=None,
    probe_success_plan_keys=None,
    probe_candidate_plan_keys=None,
):
    shadow_state = shadow_state if isinstance(shadow_state, dict) else {}
    strategy_context = strategy_context if isinstance(strategy_context, dict) else {}
    host_logic = host_logic if isinstance(host_logic, dict) else {}
    macro_alignment = macro_alignment if isinstance(macro_alignment, dict) else {}
    donchian_state = donchian_state if isinstance(donchian_state, dict) else {}

    votes = []
    htf = _direction_score(strategy_context.get("htf"))
    mid = _direction_score(strategy_context.get("mid_trend"))
    macro_bias = _safe_float(strategy_context.get("macro_bias"), 0.0)
    host_direction = str(host_logic.get("direction") or "").lower()
    host_confidence = _safe_float(host_logic.get("confidence"), 0.0)
    macro_score = _safe_float(macro_alignment.get("score"), 0.0)
    hard_block = bool(macro_alignment.get("hard_block", False))
    market_state = str(donchian_state.get("state") or "").lower()
    market_action = str(donchian_state.get("action") or "").lower()

    if htf:
        votes.append(("htf", htf, 1.2))
    if mid:
        votes.append(("mid_trend", mid, 1.0))
    if macro_bias >= 0.5:
        votes.append(("macro_bias", 1, 0.8))
    elif macro_bias <= -0.5:
        votes.append(("macro_bias", -1, 0.8))
    if host_direction in {"long", "short"} and host_confidence >= 0.65:
        votes.append(("host_logic", 1 if host_direction == "long" else -1, 1.2))
    if macro_score >= 1.15 and not hard_block:
        host_vote = 1 if host_direction == "long" else -1 if host_direction == "short" else 0
        if host_vote:
            votes.append(("macro_alignment", host_vote, 0.8))

    bull_score = round(sum(weight for _, vote, weight in votes if vote > 0), 4)
    bear_score = round(sum(weight for _, vote, weight in votes if vote < 0), 4)
    if hard_block:
        market_bias = "blocked"
    elif bull_score >= bear_score + 1.0 and bull_score >= 1.8:
        market_bias = "bullish"
    elif bear_score >= bull_score + 1.0 and bear_score >= 1.8:
        market_bias = "bearish"
    elif bull_score > 0 or bear_score > 0:
        market_bias = "mixed"
    else:
        market_bias = "neutral"

    mode = str(shadow_state.get("mode") or "normal")
    exposure_cap = max(0.0, min(1.0, _safe_float(shadow_state.get("suggested_exposure_scale"), 0.0)))
    reason_codes = list(shadow_state.get("reason_codes") or [])
    signal = str(strategy_signal or "wait").lower()
    profile_quality_wait = (
        signal == "wait"
        and "MLX回測輪廓不佳" in str(strategy_execution_reason or "")
    )
    underperforming_keys = {
        str(key)
        for key in (underperforming_plan_keys or [])
        if str(key)
    }
    recovering_keys = {
        str(key)
        for key in (recovering_plan_keys or [])
        if str(key)
    }
    probe_success_keys = {
        str(key)
        for key in (probe_success_plan_keys or [])
        if str(key)
    }
    probe_candidate_keys = {
        str(key)
        for key in (probe_candidate_plan_keys or [])
        if str(key)
    }

    selected_plan = "normal_wait"
    shadow_action = "wait"
    rationale = "等待月報酬5%策略與市場方向同時確認"

    if mode in {"intraday_stop", "post_lock_floor_guard"}:
        selected_plan = "risk_off"
        shadow_action = "risk_off"
        rationale = "日內停損或月度鎖利地板被觸發，優先空倉保護"
    elif mode == "post_lock":
        selected_plan = "post_lock_low_exposure"
        shadow_action = "reduced_exposure"
        rationale = "月報酬已達5%鎖利區，僅允許低曝險續跑"
    elif mode == "recovery":
        if market_bias == "bullish" and signal in {"long", "wait"}:
            selected_plan = "recovery_long_flat_selector"
            shadow_action = "evaluate_long"
            rationale = "月度回撤觸發恢復模式，僅評估回測指定的低曝險偏多恢復路徑"
        else:
            selected_plan = "recovery_wait_for_long_flat"
            shadow_action = "wait"
            rationale = "恢復模式未取得偏多共振，等待更適合的低曝險恢復行情"
    elif market_bias == "bullish" and signal in {"long", "wait"}:
        selected_plan = "normal_long_selector"
        shadow_action = "evaluate_long"
        rationale = "短中期與宏觀偏多，使用正常相似日策略評估多方機會"
    elif market_bias == "bearish" and signal in {"short", "wait"}:
        selected_plan = "normal_short_selector"
        shadow_action = "evaluate_short"
        rationale = "市場偏空，使用正常相似日策略評估空方機會"
    elif market_bias == "mixed" and signal == "wait" and max(bull_score, bear_score) >= MIXED_BIAS_PROBE_MIN_SCORE:
        score_gap = abs(bull_score - bear_score)
        if score_gap <= MIXED_BIAS_PROBE_MAX_GAP:
            if bull_score >= bear_score:
                selected_plan = "mixed_bias_long_probe"
                shadow_action = "evaluate_long"
                rationale = "多空分數接近但多方略強，僅用低曝險 shadow probe 降低空倉時間"
            else:
                selected_plan = "mixed_bias_short_probe"
                shadow_action = "evaluate_short"
                rationale = "多空分數接近但空方略強，僅用低曝險 shadow probe 降低空倉時間"
            exposure_cap = round(min(exposure_cap, MIXED_BIAS_PROBE_EXPOSURE_CAP), 4)
            reason_codes.append("mixed_bias_shadow_probe")
    elif market_bias == "blocked":
        selected_plan = "macro_block_wait"
        shadow_action = "wait"
        rationale = "宏觀或事件風險硬阻擋，等待解除"

    if market_state == "chop" and shadow_action in {"evaluate_long", "evaluate_short"}:
        exposure_cap = round(min(exposure_cap, 0.35), 4)
        reason_codes.append("chop_market_reduce")
    profile_suppressed_plan = ""
    profile_suppressed_action = ""
    profile_suppressed_key = ""
    profile_suppressed_exposure_cap = 0.0
    if profile_quality_wait and shadow_action in {"evaluate_long", "evaluate_short"}:
        profile_suppressed_plan = selected_plan
        profile_suppressed_action = shadow_action
        profile_suppressed_key = "|".join((selected_plan, shadow_action, market_bias, market_state))
        profile_suppressed_exposure_cap = exposure_cap
        if profile_suppressed_key in recovering_keys or profile_suppressed_key in probe_candidate_keys:
            selected_plan = "profile_quality_recovery_probe"
            exposure_cap = round(min(exposure_cap, RECOVERY_PROBE_EXPOSURE_CAP), 4)
            reason_codes.append("profile_quality_recovery_probe")
            rationale = "profile 品質擋單候選近期 counterfactual 轉正，僅做 shadow 低曝險恢復探測"
        else:
            selected_plan = "profile_quality_wait"
            shadow_action = "wait"
            exposure_cap = 0.0
            reason_codes.append("profile_quality_wait")
            rationale = "主策略 MLX 回測輪廓不佳，月報酬5% shadow 等待更高品質接管條件"
    selection_key = "|".join((selected_plan, shadow_action, market_bias, market_state))
    suppressed_plan = profile_suppressed_plan
    suppressed_action = profile_suppressed_action
    suppressed_key = profile_suppressed_key
    suppressed_exposure_cap = profile_suppressed_exposure_cap
    recovery_probe = False
    recovery_probe_key = ""
    probe_success = False
    if selected_plan == "profile_quality_recovery_probe":
        recovery_probe = True
        recovery_probe_key = profile_suppressed_key
    if shadow_action in {"evaluate_long", "evaluate_short", "reduced_exposure"} and selection_key in probe_success_keys:
        reason_codes.append("underperforming_probe_success")
        rationale = "低曝險探測表現轉正，恢復正常月報酬5%策略評估"
        probe_success = True
    elif shadow_action in {"evaluate_long", "evaluate_short", "reduced_exposure"} and selection_key in recovering_keys:
        exposure_cap = round(min(exposure_cap, RECOVERY_PROBE_EXPOSURE_CAP), 4)
        reason_codes.append("underperforming_recovery_probe")
        recovery_probe = True
        recovery_probe_key = selection_key
        rationale = "近期假想恢復表現轉正，使用低曝險探測是否可恢復月報酬5%策略"
    if (
        not probe_success
        and shadow_action in {"evaluate_long", "evaluate_short", "reduced_exposure"}
        and selection_key in underperforming_keys
        and selection_key in probe_candidate_keys
    ):
        exposure_cap = round(min(exposure_cap, UNDERPERFORMING_MICRO_PROBE_EXPOSURE_CAP), 4)
        reason_codes.append("underperforming_micro_probe")
        recovery_probe = True
        recovery_probe_key = selection_key
        rationale = "同類市場仍偏弱但低曝險探測半程轉正，僅允許5%倉位驗證以降低空倉"
    elif (
        not probe_success
        and shadow_action in {"evaluate_long", "evaluate_short", "reduced_exposure"}
        and selection_key in underperforming_keys
    ):
        suppressed_plan = selected_plan
        suppressed_action = shadow_action
        suppressed_key = selection_key
        suppressed_exposure_cap = exposure_cap
        reason_codes.append("underperforming_plan_wait")
        selected_plan = "underperforming_wait"
        shadow_action = "wait"
        exposure_cap = 0.0
        rationale = "近期同類市場選擇已累積負報酬，暫停評估以保護月報酬5%目標"

    return {
        "schema_version": 1,
        "selector_policy_version": SELECTOR_POLICY_VERSION,
        "market_bias": market_bias,
        "bull_score": bull_score,
        "bear_score": bear_score,
        "market_state": market_state,
        "market_action": market_action,
        "selected_plan": selected_plan,
        "shadow_action": shadow_action,
        "exposure_cap": round(exposure_cap, 4),
        "suppressed_plan": suppressed_plan,
        "suppressed_action": suppressed_action,
        "suppressed_key": suppressed_key,
        "suppressed_exposure_cap": round(suppressed_exposure_cap, 4),
        "recovery_probe": recovery_probe,
        "recovery_probe_key": recovery_probe_key,
        "max_leverage": min(5, max(0, _safe_int(shadow_state.get("max_leverage"), 5))),
        "strategy_signal": str(strategy_signal or "wait"),
        "strategy_execution_reason": str(strategy_execution_reason or ""),
        "reason_codes": sorted(set(str(code) for code in reason_codes if code)),
        "rationale": rationale,
    }


def build_execution_guard(
    shadow_state,
    *,
    direction,
    requested_size,
):
    shadow_state = shadow_state if isinstance(shadow_state, dict) else {}
    selection = (
        shadow_state.get("market_selection")
        if isinstance(shadow_state.get("market_selection"), dict)
        else {}
    )
    direction = str(direction or "").lower()
    requested_size = max(0.0, min(1.0, _safe_float(requested_size, 0.0)))
    shadow_action = str(selection.get("shadow_action") or "wait")
    selected_plan = str(selection.get("selected_plan") or "normal_wait")
    exposure_cap = max(0.0, min(1.0, _safe_float(selection.get("exposure_cap"), requested_size)))
    mode = str(shadow_state.get("mode") or "normal")

    allowed = True
    reason_code = "allowed"
    reason = "月報酬5%風控允許"

    if mode in {"intraday_stop", "post_lock_floor_guard"} or shadow_action == "risk_off":
        allowed = False
        reason_code = "monthly5_risk_off"
        reason = "日內停損或月度鎖利地板啟動，禁止新倉"
    elif direction == "long" and shadow_action == "evaluate_short":
        allowed = False
        reason_code = "monthly5_direction_mismatch"
        reason = "月報酬5%策略目前只評估空方，禁止多單"
    elif direction == "short" and shadow_action == "evaluate_long":
        allowed = False
        reason_code = "monthly5_direction_mismatch"
        reason = "月報酬5%策略目前只評估多方，禁止空單"
    elif selected_plan == "macro_block_wait":
        allowed = False
        reason_code = "monthly5_macro_block"
        reason = "宏觀或事件風險阻擋，禁止新倉"

    adjusted_size = min(requested_size, exposure_cap) if allowed else 0.0
    capped = bool(allowed and adjusted_size < requested_size - 1e-9)

    return {
        "schema_version": 1,
        "enabled": True,
        "allowed": bool(allowed),
        "reason_code": reason_code,
        "reason": reason,
        "direction": direction,
        "requested_size": round(requested_size, 4),
        "adjusted_size": round(adjusted_size, 4),
        "exposure_cap": round(exposure_cap, 4),
        "capped": capped,
        "selected_plan": selected_plan,
        "shadow_action": shadow_action,
        "max_leverage": min(5, max(0, _safe_int(shadow_state.get("max_leverage"), 5))),
    }


def build_position_guard(
    shadow_state,
    *,
    current_size,
):
    shadow_state = shadow_state if isinstance(shadow_state, dict) else {}
    selection = (
        shadow_state.get("market_selection")
        if isinstance(shadow_state.get("market_selection"), dict)
        else {}
    )
    current_size = max(0.0, min(1.0, _safe_float(current_size, 0.0)))
    exposure_cap = max(0.0, min(1.0, _safe_float(selection.get("exposure_cap"), current_size)))
    mode = str(shadow_state.get("mode") or "normal")
    shadow_action = str(selection.get("shadow_action") or "wait")
    selected_plan = str(selection.get("selected_plan") or "normal_wait")

    action = "hold"
    target_size = current_size
    reduce_delta = 0.0
    reason_code = "within_cap"
    reason = "月報酬5%持倉風控未要求調整"

    if current_size <= 0:
        reason_code = "flat"
        reason = "目前空倉"
    elif mode in {"intraday_stop", "post_lock_floor_guard"} or shadow_action == "risk_off":
        action = "close_all"
        target_size = 0.0
        reduce_delta = current_size
        reason_code = "monthly5_close_all"
        reason = "日內停損或月度鎖利地板啟動，持倉需平倉"
    elif current_size > exposure_cap + 1e-9:
        action = "reduce_to_cap"
        target_size = exposure_cap
        reduce_delta = current_size - exposure_cap
        reason_code = "monthly5_reduce_to_cap"
        reason = "持倉超過月報酬5%策略曝險上限，需降倉"

    return {
        "schema_version": 1,
        "enabled": True,
        "action": action,
        "reason_code": reason_code,
        "reason": reason,
        "current_size": round(current_size, 4),
        "target_size": round(max(0.0, target_size), 4),
        "reduce_delta": round(max(0.0, reduce_delta), 4),
        "exposure_cap": round(exposure_cap, 4),
        "selected_plan": selected_plan,
        "shadow_action": shadow_action,
        "mode": mode,
    }
