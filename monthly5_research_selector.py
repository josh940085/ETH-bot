"""Read monthly5 research selector artifacts for live shadow diagnostics."""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import monthly5_shadow
import monthly5_regime_selector


DEFAULT_SPEC_PATH = Path("docs/strategy_specs/monthly5_postlock_hourly.json")
DEFAULT_SELECTOR_CACHE_PATH = Path(".runtime/data/backtests/monthly5_search/daily_selector_cache_220_2020_20260804.npz")
REQUIRED_LIVE_DAILY_ROWS = 365
EXPECTED_SHORT_MARKET_STATE_FEATURES = 19
DEFAULT_NEAREST_DAYS = 80
DEFAULT_CANDIDATE_POOL = 40
FEATURE_WINDOWS = (1, 3, 7, 14, 30, 60)


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
    elif direction_mode == "sf":
        primary_direction = "short"
        direction_label = "short_flat"
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


def _latest_daily_key(df_1d):
    if df_1d is None or len(df_1d) == 0:
        return ""
    try:
        if isinstance(df_1d.index, pd.DatetimeIndex):
            ts = pd.to_datetime(df_1d.index[-1], utc=True, errors="coerce")
        elif "time" in df_1d:
            ts = pd.to_datetime(df_1d["time"].iloc[-1], unit="ms", utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(df_1d.index[-1], utc=True, errors="coerce")
        if pd.isna(ts):
            return ""
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return ""


def _completed_daily_frame(df_1d, *, now_ts=None):
    frame = df_1d.copy() if df_1d is not None else pd.DataFrame()
    if frame.empty:
        return frame
    timestamps = None
    if "time" in frame:
        timestamps = pd.to_datetime(frame["time"], unit="ms", utc=True, errors="coerce")
    elif isinstance(frame.index, pd.DatetimeIndex):
        timestamps = pd.Series(
            pd.to_datetime(frame.index, utc=True, errors="coerce"),
            index=frame.index,
        )
    if timestamps is None:
        return frame
    now = (
        pd.Timestamp.now(tz="UTC")
        if now_ts is None
        else pd.Timestamp(float(now_ts), unit="s", tz="UTC")
    )
    complete = timestamps < now.normalize()
    if bool(complete.any()):
        return frame.loc[np.asarray(complete, dtype=bool)].copy()
    return frame.iloc[0:0].copy()


def _live_4h_regime_probe(df_4h):
    frame = df_4h.copy() if df_4h is not None else pd.DataFrame()
    # Runtime frames include the currently forming candle. This label is
    # diagnostic only and must use a completed 4h candle.
    completed = frame.iloc[:-1] if len(frame) > 1 else frame.iloc[0:0]
    if len(completed) < 30 or "close" not in completed:
        return {"usable": False, "regime": "unknown", "completed_bars": len(completed)}
    labels = monthly5_regime_selector.classify_4h_regimes(completed)
    regime = str(labels.iloc[-1]) if len(labels) else "unknown"
    try:
        if "time" in completed:
            completed_at = pd.to_datetime(
                completed["time"].iloc[-1],
                unit="ms",
                utc=True,
            ) + pd.Timedelta(hours=4)
            completed_at = completed_at.isoformat()
        elif isinstance(completed.index, pd.DatetimeIndex):
            completed_at = pd.to_datetime(completed.index[-1], utc=True).isoformat()
        else:
            completed_at = ""
    except Exception:
        completed_at = ""
    return {
        "usable": regime in monthly5_regime_selector.REGIMES,
        "regime": regime,
        "completed_bars": len(completed),
        "completed_at": completed_at,
        "decision_role": "shadow_diagnostic_only",
    }


def _numeric_series(df, column):
    if df is None or column not in df:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").astype("float64")


def _live_short_market_state_vector(df_1d):
    df = df_1d.copy() if df_1d is not None else pd.DataFrame()
    high = _numeric_series(df, "high")
    low = _numeric_series(df, "low")
    close = _numeric_series(df, "close")
    frame = pd.DataFrame({"high": high, "low": low, "close": close}).dropna()
    if len(frame) <= 0:
        return np.array([], dtype="float32")

    prev_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    latest_close = max(1e-12, float(frame["close"].iloc[-1]))
    features = []
    for window in FEATURE_WINDOWS:
        segment = frame.tail(window)
        segment_tr = true_range.tail(window)
        high_max = max(1e-12, float(segment["high"].max()))
        low_min = max(1e-12, float(segment["low"].min()))
        close_vs_high_pct = ((latest_close / high_max) - 1.0) * 100.0
        range_pct = ((high_max / low_min) - 1.0) * 100.0
        atr_pct = (float(segment_tr.mean()) / latest_close) * 100.0
        features.extend([close_vs_high_pct, range_pct, atr_pct])
    features.append(0.0)
    return np.asarray(features, dtype="float32")


def _load_selector_cache(cache_path):
    cache = np.load(Path(cache_path), allow_pickle=True)
    return {
        "R": np.asarray(cache["R"], dtype="float32"),
        "F": np.asarray(cache["F"], dtype="float32"),
        "Xday": np.asarray(cache["Xday"], dtype="float32"),
        "keys": np.asarray(cache["keys"]).astype(str),
        "days": np.asarray(cache["days"]).astype(str),
    }


def build_live_selector_input_probe(
    df_1d=None,
    *,
    cache_path=DEFAULT_SELECTOR_CACHE_PATH,
    required_rows=REQUIRED_LIVE_DAILY_ROWS,
    now_ts=None,
):
    """Report whether live data can feed the research similar-day selector."""
    required_rows = max(1, _safe_int(required_rows, REQUIRED_LIVE_DAILY_ROWS))
    daily_rows = max(0, len(df_1d) if df_1d is not None else 0)
    completed_frame = _completed_daily_frame(df_1d, now_ts=now_ts)
    completed_daily_rows = len(completed_frame)
    required_columns = {"high", "low", "close"}
    available_columns = set(list(getattr(df_1d, "columns", [])))
    missing_columns = sorted(required_columns - available_columns)
    latest_day = _latest_daily_key(df_1d)
    latest_completed_day = _latest_daily_key(completed_frame)
    cache_path = Path(cache_path)
    blocking_reasons = []
    cache_feature_count = 0
    cache_candidate_count = 0
    cache_day_count = 0
    cache_latest_day = ""
    cache_available = cache_path.exists()

    if not cache_available:
        blocking_reasons.append("selector_cache_missing")
    else:
        try:
            cache = np.load(cache_path, allow_pickle=True)
            xday = cache["Xday"]
            keys = cache["keys"]
            days = cache["days"]
            cache_feature_count = int(xday.shape[1]) if getattr(xday, "ndim", 0) == 2 else 0
            cache_candidate_count = int(len(keys))
            cache_day_count = int(len(days))
            cache_latest_day = str(days[-1]) if len(days) else ""
            if cache_feature_count != EXPECTED_SHORT_MARKET_STATE_FEATURES:
                blocking_reasons.append("selector_feature_count_mismatch")
            if cache_candidate_count <= 0:
                blocking_reasons.append("selector_candidate_cache_empty")
            if cache_day_count <= 1:
                blocking_reasons.append("selector_day_cache_empty")
        except Exception as exc:
            blocking_reasons.append(f"selector_cache_unreadable:{exc}")

    if completed_daily_rows < required_rows:
        blocking_reasons.append("daily_warmup_insufficient")
    if missing_columns:
        blocking_reasons.append("daily_columns_missing")
    if not latest_completed_day:
        blocking_reasons.append("daily_latest_day_missing")

    usable = not blocking_reasons
    return {
        "schema_version": 1,
        "selector_source": monthly5_shadow.RESEARCH_SELECTOR_SOURCE,
        "input_source": "live_daily_kline",
        "feature_set": "short_market_state",
        "usable": usable,
        "blocking_reasons": blocking_reasons,
        "daily_rows": daily_rows,
        "completed_daily_rows": completed_daily_rows,
        "required_daily_rows": required_rows,
        "latest_daily_key": latest_day,
        "latest_completed_daily_key": latest_completed_day,
        "missing_columns": missing_columns,
        "cache_path": str(cache_path),
        "cache_available": cache_available,
        "cache_feature_count": cache_feature_count,
        "expected_feature_count": EXPECTED_SHORT_MARKET_STATE_FEATURES,
        "cache_candidate_count": cache_candidate_count,
        "cache_day_count": cache_day_count,
        "cache_latest_day": cache_latest_day,
    }


def build_live_selector_decision(
    df_1d=None,
    *,
    cache_path=DEFAULT_SELECTOR_CACHE_PATH,
    nearest_days=DEFAULT_NEAREST_DAYS,
    candidate_pool=DEFAULT_CANDIDATE_POOL,
    target_return=0.05,
    df_4h=None,
    now_ts=None,
):
    input_probe = build_live_selector_input_probe(
        df_1d,
        cache_path=cache_path,
        now_ts=now_ts,
    )
    if not input_probe.get("usable"):
        return {
            "schema_version": 1,
            "selector_source": monthly5_shadow.RESEARCH_SELECTOR_SOURCE,
            "usable": False,
            "blocking_reasons": list(input_probe.get("blocking_reasons") or []),
            "input_probe": input_probe,
        }

    completed_daily = _completed_daily_frame(df_1d, now_ts=now_ts)
    vector = _live_short_market_state_vector(completed_daily)
    if vector.shape[0] != EXPECTED_SHORT_MARKET_STATE_FEATURES:
        return {
            "schema_version": 1,
            "selector_source": monthly5_shadow.RESEARCH_SELECTOR_SOURCE,
            "usable": False,
            "blocking_reasons": ["live_feature_count_mismatch"],
            "live_feature_count": int(vector.shape[0]),
            "expected_feature_count": EXPECTED_SHORT_MARKET_STATE_FEATURES,
            "input_probe": input_probe,
        }

    try:
        cache = _load_selector_cache(cache_path)
    except Exception as exc:
        return {
            "schema_version": 1,
            "selector_source": monthly5_shadow.RESEARCH_SELECTOR_SOURCE,
            "usable": False,
            "blocking_reasons": [f"selector_cache_unreadable:{exc}"],
            "input_probe": input_probe,
        }

    xday = cache["Xday"]
    returns = cache["R"]
    flats = cache["F"]
    keys = cache["keys"]
    days = cache["days"]
    expected_shape = (len(keys), len(days))
    if (
        xday.ndim != 2
        or xday.shape != (len(days), vector.shape[0])
        or returns.shape != expected_shape
        or flats.shape != expected_shape
        or len(keys) <= 0
        or len(days) <= 1
    ):
        return {
            "schema_version": 1,
            "selector_source": monthly5_shadow.RESEARCH_SELECTOR_SOURCE,
            "usable": False,
            "blocking_reasons": ["selector_cache_shape_mismatch"],
            "input_probe": input_probe,
        }

    # Completed state day j-1 predicts the candidate return on target day j.
    # X[j] -> R[j] would leak target-day information into candidate ranking.
    historical_features = xday[:-1]
    target_returns = returns[:, 1:]
    target_flats = flats[:, 1:]
    state_days = days[:-1]
    target_days = days[1:]
    mean = np.nanmean(historical_features, axis=0)
    std = np.nanstd(historical_features, axis=0)
    std = np.where(std > 1e-9, std, 1.0)
    normalized_cache = np.nan_to_num(
        (historical_features - mean) / std,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    normalized_vector = np.nan_to_num((vector - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    distances = np.linalg.norm(normalized_cache - normalized_vector, axis=1)
    nearest_count = max(1, min(_safe_int(nearest_days, DEFAULT_NEAREST_DAYS), len(distances)))
    nearest_idx = np.argsort(distances)[:nearest_count]

    candidate_returns = target_returns[:, nearest_idx]
    candidate_flats = target_flats[:, nearest_idx]
    q25 = np.nanpercentile(candidate_returns, 25, axis=1)
    hit_rate = np.nanmean(candidate_returns >= max(0.0, _safe_float(target_return, 0.05)), axis=1)
    mean_return = np.nanmean(candidate_returns, axis=1)
    flat_time = np.nanmean(candidate_flats, axis=1)
    score = q25 + (hit_rate * max(0.0, _safe_float(target_return, 0.05))) + (mean_return * 0.10)
    pool_size = max(1, min(_safe_int(candidate_pool, DEFAULT_CANDIDATE_POOL), len(keys)))
    ranked = np.argsort(-np.nan_to_num(score, nan=-1e9))[:pool_size]
    best_idx = int(ranked[0])
    parsed = parse_top_pick(keys[best_idx])
    sample_idx = nearest_idx[: min(5, len(nearest_idx))]
    nearest_days_sample = [str(target_days[int(idx)]) for idx in sample_idx]
    nearest_state_days_sample = [str(state_days[int(idx)]) for idx in sample_idx]
    regime_4h = _live_4h_regime_probe(df_4h)

    return {
        "schema_version": 1,
        "selector_source": monthly5_shadow.RESEARCH_SELECTOR_SOURCE,
        "usable": True,
        "blocking_reasons": [],
        "input_probe": input_probe,
        "feature_set": "short_market_state",
        "causal_lag_days": 1,
        "live_state_day": input_probe.get("latest_completed_daily_key", ""),
        "market_regime_4h": regime_4h,
        "live_feature_count": int(vector.shape[0]),
        "nearest_days": nearest_count,
        "candidate_pool": pool_size,
        "selected_index": best_idx,
        "selected_key": str(keys[best_idx]),
        "selected_score": round(float(score[best_idx]), 6),
        "selected_q25_return_pct": round(float(q25[best_idx]) * 100.0, 4),
        "selected_mean_return_pct": round(float(mean_return[best_idx]) * 100.0, 4),
        "selected_hit_rate": round(float(hit_rate[best_idx]), 4),
        "selected_flat_time_pct": round(float(flat_time[best_idx]) * 100.0, 4),
        "nearest_sample_days": nearest_days_sample,
        "nearest_state_sample_days": nearest_state_days_sample,
        **parsed,
    }
