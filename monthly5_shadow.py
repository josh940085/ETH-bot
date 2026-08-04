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
    elif market_bias == "blocked":
        selected_plan = "macro_block_wait"
        shadow_action = "wait"
        rationale = "宏觀或事件風險硬阻擋，等待解除"

    if market_state == "chop" and shadow_action in {"evaluate_long", "evaluate_short"}:
        exposure_cap = round(min(exposure_cap, 0.35), 4)
        reason_codes.append("chop_market_reduce")

    return {
        "schema_version": 1,
        "market_bias": market_bias,
        "bull_score": bull_score,
        "bear_score": bear_score,
        "market_state": market_state,
        "market_action": market_action,
        "selected_plan": selected_plan,
        "shadow_action": shadow_action,
        "exposure_cap": round(exposure_cap, 4),
        "max_leverage": min(5, max(0, _safe_int(shadow_state.get("max_leverage"), 5))),
        "strategy_signal": str(strategy_signal or "wait"),
        "strategy_execution_reason": str(strategy_execution_reason or ""),
        "reason_codes": sorted(set(str(code) for code in reason_codes if code)),
        "rationale": rationale,
    }
