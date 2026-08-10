"""Standalone, fully observational experimental monthly5 shadow candidate.

This module makes no live trades, is never imported by eth.py, and has no
effect on the real order-execution path. It exists purely to forward-test a
combination of three previously-unintegrated monthly5 research findings as
one paper strategy, so there is real (not backtested) evidence before anyone
considers folding any of this into the actual shadow candidate in
monthly5_shadow.py (STRATEGY_ID monthly5_postlock_hourly_v0), let alone
live trading.

Run it out-of-band (manually, or wire it into a cron/supervisor job yourself)
via monthly5_experimental_candidate_runner.py, which fetches fresh Binance
4h klines and calls run_once() below.

Combines:
  1. Monthly walk-forward switching between the two ATR-normalized 4h regime
     configs in monthly5_volatility_walkforward.CANDIDATE_CONFIGS, using the
     "lb24_tail" selection rule (lookback_months=24, score_mode="tail") that
     scored best in that module's own holdout backtest.
  2. The 24-bar relative-volume confirmation gate from
     monthly5_volume_forward_shadow.py (VOLUME_WINDOW=24,
     MIN_VOLUME_RATIO=0.5).
  3. A graduated ADX-based recovery override
     ("adx_trigger-0.06_scale0.25_exit-0.01") from the grid search in
     monthly5_intramonth_recovery_research.py.

METHODOLOGY CAVEATS (read before trusting promotion_ready=True from here):
  - monthly5_intramonth_recovery_research.py's own grid-search selection is
    flagged evaluation_period_reused_during_research=True, and its declared
    winner is actually "baseline_no_recovery" (no signal swap at all). The
    ADX variant used here only edges out that baseline by roughly
    +0.12pp/month and -0.28pp max drawdown on its holdout window — a narrow
    margin from a search whose own methodology is caveated.
  - monthly5_volume_forward_shadow.RESEARCH_VALID is False: its original
    historical backtest was invalidated for a look-ahead bug, and only data
    it collects after its own FORWARD_START is trustworthy. That
    forward-only re-validation was still short of its own promotion bar as
    of this module's introduction (see monthly5_volume_forward_shadow's
    state file for current status).
  - Each of the three pieces above was validated (or is still being
    validated) independently. Running all three together as one strategy is
    itself an untested hypothesis: this combination has zero backtest or
    forward evidence prior to this module existing. MIN_FORWARD_DAYS below
    is set higher than monthly5_volume_forward_shadow's 30 days for that
    reason, and promotion_ready here should be read as "cleared this
    module's own bar", never as investment-grade evidence on its own.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import monthly5_intraday_regime as regime
import monthly5_intramonth_recovery_research as account
import monthly5_risk_profile_walkforward as risk_walkforward
import monthly5_shadow
import monthly5_volatility_regime_research as volatility
import monthly5_volatility_walkforward as volatility_walkforward
import monthly5_volume_forward_shadow


SCHEMA_VERSION = 1
CANDIDATE_ID = "monthly5_atr_walkforward_volume_adx_recovery_experimental_v1"

WALKFORWARD_LOOKBACK_MONTHS = 24
WALKFORWARD_SCORE_MODE = "tail"
PRIMARY_CONFIGS = volatility_walkforward.CANDIDATE_CONFIGS

RECOVERY_LABEL_CONFIG = {
    "name": "adx25.0_s0.0",
    "mode": "adx",
    "adx_threshold": 25.0,
    "slope_atr": 0.0,
}
RECOVERY_CONFIRMATION_BARS = 1
RECOVERY_RANGE_GRACE_BARS = 6
RECOVERY_TRIGGER_PCT = -6.0
RECOVERY_EXIT_PCT = -1.0
RECOVERY_EXPOSURE_SCALE = 0.25

VOLUME_WINDOW = monthly5_volume_forward_shadow.VOLUME_WINDOW
MIN_VOLUME_RATIO = monthly5_volume_forward_shadow.MIN_VOLUME_RATIO

REQUIRED_4H_WARMUP_BARS = 30
MIN_FORWARD_DAYS = 45.0
MIN_FORWARD_COVERAGE = 0.80
MAX_FLAT_TIME_PCT = 80.0

METHODOLOGY_CAVEATS = [
    "recovery_grid_own_winner_is_baseline_no_recovery",
    "recovery_grid_evaluation_period_reused_during_research",
    "volume_component_research_invalidated_forward_pending",
    "three_way_combination_never_independently_backtested",
]


def _safe_float(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _completed_frame(frame):
    current = frame.copy() if frame is not None else pd.DataFrame()
    return current.iloc[:-1].copy() if len(current) > 1 else current.iloc[0:0].copy()


def _build_position_series(labels, confirmation_bars, range_grace_bars):
    active = 0.0
    pending = ""
    pending_count = 0
    range_count = 0
    positions = []
    for label in labels.astype(str):
        if label in {"up", "down"}:
            range_count = 0
            if label == pending:
                pending_count += 1
            else:
                pending = label
                pending_count = 1
            if pending_count >= max(1, int(confirmation_bars)):
                active = 1.0 if label == "up" else -1.0
        elif label == "range":
            pending = ""
            pending_count = 0
            range_count += 1
            if range_count > max(0, int(range_grace_bars)):
                active = 0.0
        else:
            active = 0.0
            pending = ""
            pending_count = 0
            range_count = 0
        positions.append(active)
    return pd.Series(positions, index=labels.index, dtype="float64")


def select_walkforward_config(
    frame_4h,
    *,
    lookback_months=WALKFORWARD_LOOKBACK_MONTHS,
    score_mode=WALKFORWARD_SCORE_MODE,
    configs=PRIMARY_CONFIGS,
):
    """Pick whichever of `configs` scored better over the trailing
    `lookback_months` *completed* months (the current/ongoing month is never
    part of the scoring window, to avoid look-ahead), using the same
    candidate_score() rule monthly5_risk_profile_walkforward.py uses.
    """
    close = pd.to_numeric(frame_4h["close"], errors="coerce").astype("float64")
    bar_return = close.pct_change().fillna(0.0)
    index = pd.DatetimeIndex(frame_4h.index)
    month_keys = (index.tz_localize(None) if index.tz is not None else index).to_period("M")
    months = month_keys.unique()

    config_names = []
    monthly_matrix = []
    for config in configs:
        labels = volatility.classify_4h_frame(frame_4h, config["label"])
        position = _build_position_series(
            labels, config["confirmation_bars"], config["range_grace_bars"]
        )
        strat_return = position.shift(1).fillna(0.0).to_numpy() * bar_return.to_numpy()
        monthly = (
            pd.Series(1.0 + strat_return, index=month_keys).groupby(level=0).prod() - 1.0
        )
        monthly_matrix.append(monthly.reindex(months).fillna(0.0).to_numpy(dtype="float64"))
        config_names.append(config["name"])
    matrix = np.asarray(monthly_matrix, dtype="float64")

    completed_month_count = max(0, len(months) - 1)
    trailing_used = min(max(1, int(lookback_months)), completed_month_count)
    if trailing_used <= 0:
        selected_index = 0
        scores = [0.0] * len(configs)
        low_confidence = True
    else:
        history = matrix[:, -(trailing_used + 1) : -1]
        scores = risk_walkforward.candidate_score(history, score_mode).tolist()
        selected_index = int(np.argmax(scores))
        low_confidence = trailing_used < lookback_months

    return {
        "selected_config_name": config_names[selected_index],
        "selected_config": configs[selected_index],
        "trailing_months_used": trailing_used,
        "low_confidence": low_confidence,
        "scores": dict(zip(config_names, (round(float(s), 6) for s in scores))),
    }


def build_live_probe(frame_4h, *, now_ts=None, recovery_active=False):
    completed_4h = _completed_frame(frame_4h)
    if len(completed_4h) < max(REQUIRED_4H_WARMUP_BARS, WALKFORWARD_LOOKBACK_MONTHS + 2):
        return {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": CANDIDATE_ID,
            "shadow_only": True,
            "execution_allowed": False,
            "usable": False,
            "blocking_reasons": ["completed_4h_history_insufficient"],
        }
    required_columns = {"open", "high", "low", "close", "volume"}
    if not required_columns.issubset(completed_4h.columns):
        return {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": CANDIDATE_ID,
            "shadow_only": True,
            "execution_allowed": False,
            "usable": False,
            "blocking_reasons": ["required_ohlcv_missing"],
        }

    walkforward = select_walkforward_config(completed_4h)
    primary_labels = volatility.classify_4h_frame(
        completed_4h, walkforward["selected_config"]["label"]
    )
    primary_position = _build_position_series(
        primary_labels,
        walkforward["selected_config"]["confirmation_bars"],
        walkforward["selected_config"]["range_grace_bars"],
    )
    walkforward_primary_signal = float(primary_position.iloc[-1]) if len(primary_position) else 0.0

    recovery_labels = volatility.classify_4h_frame(completed_4h, RECOVERY_LABEL_CONFIG)
    recovery_position = _build_position_series(
        recovery_labels, RECOVERY_CONFIRMATION_BARS, RECOVERY_RANGE_GRACE_BARS
    )
    recovery_signal = float(recovery_position.iloc[-1]) if len(recovery_position) else 0.0

    pre_volume_signal = recovery_signal if recovery_active else walkforward_primary_signal

    volumes = pd.to_numeric(completed_4h["volume"], errors="coerce").astype("float64")
    prior_median = (
        volumes.iloc[-(VOLUME_WINDOW + 1) : -1].median() if len(volumes) > VOLUME_WINDOW else float("nan")
    )
    current_volume = _safe_float(volumes.iloc[-1], 0.0)
    volume_ratio = current_volume / prior_median if prior_median and prior_median > 0 else 0.0
    volume_allowed = bool(np.isfinite(volume_ratio) and volume_ratio >= MIN_VOLUME_RATIO)
    candidate_signal = pre_volume_signal if volume_allowed else 0.0

    bar = completed_4h.iloc[-1]
    bar_close_ts_ms = int(pd.Timestamp(completed_4h.index[-1]).timestamp() * 1000)
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "shadow_only": True,
        "execution_allowed": False,
        "usable": True,
        "blocking_reasons": [],
        "observed_ts": int(time.time() if now_ts is None else now_ts),
        "bar_close_ts_ms": bar_close_ts_ms,
        "open": _safe_float(bar.get("open"), 0.0),
        "high": _safe_float(bar.get("high"), 0.0),
        "low": _safe_float(bar.get("low"), 0.0),
        "close": _safe_float(bar.get("close"), 0.0),
        "volume": _safe_float(bar.get("volume"), 0.0),
        "walkforward_selected_config": walkforward["selected_config_name"],
        "walkforward_trailing_months_used": walkforward["trailing_months_used"],
        "walkforward_low_confidence": walkforward["low_confidence"],
        "walkforward_scores": walkforward["scores"],
        "walkforward_primary_signal": walkforward_primary_signal,
        "recovery_signal": recovery_signal,
        "recovery_active_applied": bool(recovery_active),
        "baseline_signal": walkforward_primary_signal,
        "relative_volume_4h": round(float(volume_ratio), 6),
        "volume_allowed": volume_allowed,
        "candidate_signal": float(candidate_signal),
        "completed_4h_bars": int(len(completed_4h)),
        "recovery_label_config": RECOVERY_LABEL_CONFIG,
        "volume_window": VOLUME_WINDOW,
        "min_volume_ratio": MIN_VOLUME_RATIO,
    }


def load_history(path):
    history_path = Path(path)
    if not history_path.exists():
        return []
    rows = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def append_probe(path, probe):
    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_history(history_path)
    bar_close_ts_ms = int(_safe_float(probe.get("bar_close_ts_ms"), 0.0))
    if rows and int(_safe_float(rows[-1].get("bar_close_ts_ms"), 0.0)) >= bar_close_ts_ms:
        return rows
    payload = dict(probe)
    payload["execution_allowed"] = False
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    rows.append(payload)
    return rows


def _interval_returns(rows, signal_key):
    valid = [row for row in rows if row.get("usable") and _safe_float(row.get("close"), 0.0) > 0]
    valid = sorted(valid, key=lambda row: _safe_float(row.get("bar_close_ts_ms"), 0.0))
    intervals = []
    for idx in range(len(valid) - 1):
        row, nxt = valid[idx], valid[idx + 1]
        ts_ms = _safe_float(nxt.get("bar_close_ts_ms"), 0.0)
        direction = _safe_float(row.get(signal_key), 0.0)
        start_price = _safe_float(row.get("close"), 0.0)
        end_price = _safe_float(nxt.get("close"), 0.0)
        if direction == 0.0 or start_price <= 0 or end_price <= 0:
            intervals.append({"ts_ms": ts_ms, "return_pct": 0.0, "flat": direction == 0.0})
            continue
        pct = direction * ((end_price / start_price) - 1.0) * 100.0
        intervals.append({"ts_ms": ts_ms, "return_pct": pct, "flat": False})
    return intervals


def _paper_summary(rows, signal_key):
    intervals = _interval_returns(rows, signal_key)
    if not intervals:
        return {"rows": 0, "return_pct": 0.0, "flat_time_pct": 100.0, "max_drawdown_pct": 0.0}
    total_return_pct = 0.0
    cumulative_pct = 0.0
    peak_pct = 0.0
    max_drawdown_pct = 0.0
    flat_count = 0
    for interval in intervals:
        total_return_pct += interval["return_pct"]
        cumulative_pct += interval["return_pct"]
        peak_pct = max(peak_pct, cumulative_pct)
        max_drawdown_pct = min(max_drawdown_pct, cumulative_pct - peak_pct)
        if interval["flat"]:
            flat_count += 1
    return {
        "rows": len(intervals),
        "return_pct": round(total_return_pct, 4),
        "flat_time_pct": round((flat_count / len(intervals)) * 100.0, 4),
        "max_drawdown_pct": round(max_drawdown_pct, 4),
    }


def _month_to_date_return_pct(rows, signal_key, month_key):
    total = 0.0
    for interval in _interval_returns(rows, signal_key):
        ts_month = monthly5_shadow.month_key_from_ts(interval["ts_ms"] / 1000.0)
        if ts_month == month_key:
            total += interval["return_pct"]
    return total


def _next_recovery_active(previous_active, month_to_date_pct):
    if previous_active:
        return month_to_date_pct < RECOVERY_EXIT_PCT
    return month_to_date_pct <= RECOVERY_TRIGGER_PCT


def update_state(previous_state, rows, latest_probe, *, history_path=None, now_ts=None):
    now = int(time.time() if now_ts is None else now_ts)
    previous_state = previous_state if isinstance(previous_state, dict) else {}
    month_key = monthly5_shadow.month_key_from_ts(
        _safe_float(latest_probe.get("bar_close_ts_ms"), now * 1000.0) / 1000.0
    )
    mtd_candidate_pct = _month_to_date_return_pct(rows, "candidate_signal", month_key)
    previous_active = (
        bool(previous_state.get("recovery_active", False))
        if previous_state.get("month_key") == month_key
        else False
    )
    recovery_active = _next_recovery_active(previous_active, mtd_candidate_pct)

    candidate = _paper_summary(rows, "candidate_signal")
    baseline = _paper_summary(rows, "baseline_signal")

    first_ts = int(_safe_float(rows[0].get("bar_close_ts_ms"), 0.0)) // 1000 if rows else now
    span_hours = max(0.0, (now - first_ts) / 3600.0)
    minimum_rows = int(MIN_FORWARD_DAYS * 6.0 * MIN_FORWARD_COVERAGE)

    blockers = []
    if span_hours < MIN_FORWARD_DAYS * 24.0:
        blockers.append("forward_span_lt_min_days")
    if candidate["rows"] < minimum_rows:
        blockers.append("forward_rows_insufficient")
    if span_hours < MIN_FORWARD_DAYS * 24.0 or candidate["return_pct"] < 5.0:
        blockers.append("forward_month_target_unproven")
    if candidate["return_pct"] < baseline["return_pct"]:
        blockers.append("forward_underperforms_walkforward_only_baseline")
    if not monthly5_volume_forward_shadow.RESEARCH_VALID:
        blockers.append("volume_component_research_invalidated_forward_pending")
    if candidate["flat_time_pct"] > MAX_FLAT_TIME_PCT:
        blockers.append("forward_flat_time_high")

    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "shadow_only": True,
        "execution_allowed": False,
        "updated_ts": now,
        "trial_started_ts": first_ts,
        "month_key": month_key,
        "month_to_date_candidate_return_pct": round(mtd_candidate_pct, 4),
        "recovery_active": recovery_active,
        "recovery_trigger_pct": RECOVERY_TRIGGER_PCT,
        "recovery_exit_pct": RECOVERY_EXIT_PCT,
        "recovery_exposure_scale": RECOVERY_EXPOSURE_SCALE,
        "span_hours": round(span_hours, 4),
        "minimum_rows": minimum_rows,
        "min_forward_days": MIN_FORWARD_DAYS,
        "promotion_ready": not blockers,
        "promotion_blockers": blockers,
        "methodology_caveats": list(METHODOLOGY_CAVEATS),
        "volume_component_research_valid": monthly5_volume_forward_shadow.RESEARCH_VALID,
        "latest_probe": dict(latest_probe),
        "candidate_paper": candidate,
        "baseline_paper": baseline,
        "history_path": str(history_path) if history_path is not None else previous_state.get("history_path"),
    }


def run_once(state_path, history_path, frame_4h, *, now_ts=None):
    previous_state = monthly5_shadow.load_state(state_path)
    recovery_active = bool(previous_state.get("recovery_active", False))
    probe = build_live_probe(frame_4h, now_ts=now_ts, recovery_active=recovery_active)
    if not probe.get("usable"):
        return previous_state
    rows = append_probe(history_path, probe)
    new_state = update_state(previous_state, rows, probe, history_path=history_path, now_ts=now_ts)
    monthly5_shadow.save_state(state_path, new_state)
    return new_state
