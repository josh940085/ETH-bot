import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


STRATEGY_ID = "monthly5_postlock_hourly_v0"
SELECTED_CANDIDATE = "postlock_scale0.15_floor_pdaystopNone"
MONTHLY_LOCK_PCT = 5.0
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


def _account_equity(*, wallet_balance, margin_balance, unrealized_pnl):
    margin = _safe_float(margin_balance, 0.0)
    if margin > 0:
        return margin
    wallet = _safe_float(wallet_balance, 0.0)
    unrealized = _safe_float(unrealized_pnl, 0.0)
    return max(0.0, wallet + unrealized)


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
    equity = round(_account_equity(
        wallet_balance=wallet_balance,
        margin_balance=margin_balance,
        unrealized_pnl=unrealized_pnl,
    ), 8)

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
