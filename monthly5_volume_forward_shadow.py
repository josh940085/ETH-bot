"""Independent forward-only paper shadow for the monthly5 volume candidate."""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import monthly5_intraday_regime as regime
import monthly5_intramonth_recovery_research as account
import monthly5_shadow
import monthly5_volatility_regime_research as volatility


SCHEMA_VERSION = 1
CANDIDATE_ID = "monthly5_4h_atr_volume24_min0.5_v1"
LABEL_CONFIG = {
    "name": "atr_d0.7_s0.1",
    "mode": "atr",
    "distance_atr": 0.7,
    "slope_atr": 0.1,
}
CONFIRMATION_BARS = 1
RANGE_GRACE_BARS = 6
VOLUME_WINDOW = 24
MIN_VOLUME_RATIO = 0.5
MIN_FORWARD_DAYS = 30.0
ACCOUNT_CONFIG = {
    "name": "baseline_no_recovery",
    "mode": "none",
    "trigger": None,
    "scale": 0.0,
    "exit": 0.0,
}


def _safe_float(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _completed_frame(frame):
    current = frame.copy() if frame is not None else pd.DataFrame()
    return current.iloc[:-1].copy() if len(current) > 1 else current.iloc[0:0].copy()


def _hysteresis_signal(labels):
    active = 0.0
    pending = ""
    pending_count = 0
    range_count = 0
    for label in labels.astype(str):
        if label in {"up", "down"}:
            range_count = 0
            if label == pending:
                pending_count += 1
            else:
                pending = label
                pending_count = 1
            if pending_count >= CONFIRMATION_BARS:
                active = 1.0 if label == "up" else -1.0
        elif label == "range":
            pending = ""
            pending_count = 0
            range_count += 1
            if range_count > RANGE_GRACE_BARS:
                active = 0.0
        else:
            active = 0.0
            pending = ""
            pending_count = 0
            range_count = 0
    return active


def _row_open_ts_ms(frame, row):
    if "time" in frame:
        raw = int(_safe_float(row.get("time"), 0.0))
        return raw * 1000 if 0 < raw < 100_000_000_000 else raw
    try:
        return int(pd.Timestamp(row.name).timestamp() * 1000)
    except Exception:
        return 0


def build_live_probe(df_4h, df_5m, *, now_ts=None):
    completed_4h = _completed_frame(df_4h)
    completed_5m = _completed_frame(df_5m)
    required = max(30, VOLUME_WINDOW + 2)
    if len(completed_4h) < required or completed_5m.empty:
        return {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": CANDIDATE_ID,
            "shadow_only": True,
            "usable": False,
            "blocking_reasons": ["completed_kline_history_insufficient"],
        }
    required_columns = {"open", "high", "low", "close", "volume"}
    if not required_columns.issubset(completed_4h.columns) or not required_columns.issubset(
        completed_5m.columns
    ):
        return {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": CANDIDATE_ID,
            "shadow_only": True,
            "usable": False,
            "blocking_reasons": ["required_ohlcv_missing"],
        }
    labels = volatility.classify_4h_frame(completed_4h, LABEL_CONFIG)
    baseline_signal = _hysteresis_signal(labels)
    volumes = pd.to_numeric(completed_4h["volume"], errors="coerce").astype("float64")
    prior_median = volumes.iloc[-(VOLUME_WINDOW + 1) : -1].median()
    current_volume = _safe_float(volumes.iloc[-1], 0.0)
    volume_ratio = current_volume / prior_median if prior_median > 0 else 0.0
    volume_allowed = bool(np.isfinite(volume_ratio) and volume_ratio >= MIN_VOLUME_RATIO)
    candidate_signal = baseline_signal if volume_allowed else 0.0
    bar = completed_5m.iloc[-1]
    bar_open_ts_ms = _row_open_ts_ms(completed_5m, bar)
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "shadow_only": True,
        "usable": bool(bar_open_ts_ms > 0),
        "blocking_reasons": [] if bar_open_ts_ms > 0 else ["completed_5m_timestamp_missing"],
        "observed_ts": int(time.time() if now_ts is None else now_ts),
        "bar_open_ts_ms": bar_open_ts_ms,
        "bar_close_ts_ms": bar_open_ts_ms + 5 * 60 * 1000,
        "open": _safe_float(bar.get("open"), 0.0),
        "high": _safe_float(bar.get("high"), 0.0),
        "low": _safe_float(bar.get("low"), 0.0),
        "close": _safe_float(bar.get("close"), 0.0),
        "volume": _safe_float(bar.get("volume"), 0.0),
        "baseline_signal": float(baseline_signal),
        "candidate_signal": float(candidate_signal),
        "market_regime_4h": str(labels.iloc[-1]),
        "relative_volume_4h": round(float(volume_ratio), 6),
        "volume_allowed": volume_allowed,
        "completed_4h_bars": int(len(completed_4h)),
        "label_config": LABEL_CONFIG,
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


def _paper_summary(rows, signal_key):
    valid = [row for row in rows if row.get("usable") and _safe_float(row.get("close"), 0.0) > 0]
    if not valid:
        return {"rows": 0, "return_pct": 0.0, "flat_time_pct": 100.0, "position": 0.0}
    index = pd.to_datetime(
        [int(row["bar_close_ts_ms"]) for row in valid], unit="ms", utc=True
    )
    frame = pd.DataFrame(
        {
            key: [_safe_float(row.get(key), 0.0) for row in valid]
            for key in ("open", "high", "low", "close", "volume")
        },
        index=index,
    )
    desired = np.asarray([_safe_float(row.get(signal_key), 0.0) for row in valid])
    factors, scales, _, _, positions = account.simulate_account_path(
        frame,
        desired,
        np.zeros(len(frame), dtype="int32"),
        desired,
        ACCOUNT_CONFIG,
    )
    summary = regime.summarize_factors(frame, factors, scales, start=str(index[0]))
    return {
        "rows": len(valid),
        "return_pct": round(float(np.prod(factors) - 1.0) * 100.0, 6),
        "flat_time_pct": round(float(np.mean(positions == 0.0)) * 100.0, 4),
        "position": round(float(positions[-1]), 4),
        "max_drawdown_pct": summary["max_drawdown_pct"],
        "months_ge_5": summary["months_ge_5"],
        "months": summary["months"],
    }


def update_files(state_path, history_path, probe, *, now_ts=None):
    now = int(time.time() if now_ts is None else now_ts)
    rows = append_probe(history_path, probe) if probe.get("usable") else load_history(history_path)
    candidate = _paper_summary(rows, "candidate_signal")
    baseline = _paper_summary(rows, "baseline_signal")
    first_ts = int(_safe_float(rows[0].get("bar_close_ts_ms"), 0.0)) // 1000 if rows else now
    span_hours = max(0.0, (now - first_ts) / 3600.0)
    blockers = []
    if span_hours < MIN_FORWARD_DAYS * 24.0:
        blockers.append("forward_span_lt_30d")
    if candidate.get("months_ge_5", 0) < candidate.get("months", 0):
        blockers.append("forward_month_target_unproven")
    if candidate["return_pct"] < baseline["return_pct"]:
        blockers.append("forward_underperforms_baseline")
    state = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "shadow_only": True,
        "execution_allowed": False,
        "promotion_ready": not blockers,
        "promotion_blockers": blockers,
        "updated_ts": now,
        "trial_started_ts": first_ts,
        "span_hours": round(span_hours, 4),
        "latest_probe": dict(probe),
        "candidate_paper": candidate,
        "baseline_paper": baseline,
        "history_path": str(history_path),
    }
    monthly5_shadow.save_state(state_path, state)
    return state
