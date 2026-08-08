#!/usr/bin/env python3
"""Verify monthly 5% strategy runtime state against the research spec."""

import argparse
import json
import os
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import monthly5_shadow
import monthly5_research_selector
from verify_monthly5_candidate import _failures as candidate_failures


DEFAULT_SPEC = Path("docs/strategy_specs/monthly5_postlock_hourly.json")
DEFAULT_POSITION = Path(".runtime/data/docs/position.json")
DEFAULT_SHADOW = Path(".runtime/data/btcusdt_monthly5_shadow_state.json")
DEFAULT_HISTORY = Path(".runtime/data/btcusdt_monthly5_shadow_history.jsonl")
_ACTIVE_SYMBOL = str(os.getenv("TRADE_SYMBOL", "BTCUSDT") or "BTCUSDT").strip().upper()
_ACTIVE_SYMBOL_SLUG = re.sub(r"[^a-z0-9]+", "_", _ACTIVE_SYMBOL.lower()).strip("_") or "btcusdt"
DEFAULT_MLX_DB = Path(".runtime/ai") / f"{_ACTIVE_SYMBOL_SLUG}_mlx_agent_learning.sqlite3"
LOCAL_TZ = ZoneInfo("Asia/Taipei")


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"failed to read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"expected object JSON: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        raise SystemExit(f"failed to read JSONL {path}: {exc}") from exc
    rows = []
    for idx, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception as exc:
            raise SystemExit(f"failed to parse JSONL {path}:{idx}: {exc}") from exc
        if not isinstance(payload, dict):
            raise SystemExit(f"expected object JSONL row: {path}:{idx}")
        rows.append(payload)
    return rows


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


def _shadow_from_position(position: dict) -> dict:
    shadow = position.get("monthly5_shadow")
    return shadow if isinstance(shadow, dict) else {}


def _monthly5_readiness_summary_tokens(position: dict) -> list[str]:
    shadow = _shadow_from_position(position)
    readiness = position.get("monthly5_readiness")
    readiness = readiness if isinstance(readiness, dict) else shadow
    blockers = [str(item) for item in readiness.get("promotion_blockers") or []]
    if blockers:
        blocker_text = ",".join(blockers[:5])
    else:
        blocker_text = "none"
    tokens = [
        f"promotion_ready={str(bool(readiness.get('promotion_ready', False))).lower()}",
        f"promotion_blockers={blocker_text}",
    ]
    if "rows" in readiness:
        tokens.append(f"readiness_rows={int(_safe_float(readiness.get('rows'), 0.0))}")
    if "span_hours" in readiness:
        tokens.append(f"readiness_span_hours={_safe_float(readiness.get('span_hours'), 0.0):.4f}")
    review_ts = _safe_float(readiness.get("promotion_earliest_review_ts"), 0.0)
    if review_ts > 0.0:
        tokens.append(f"promotion_eta_ts={int(review_ts)}")
        tokens.append(f"promotion_eta_remaining_hours={max(0.0, (review_ts - time.time()) / 3600.0):.4f}")
        tokens.append(f"promotion_eta_local={datetime.fromtimestamp(review_ts, LOCAL_TZ).isoformat(timespec='seconds')}")
    return tokens


def _require(condition: bool, failures: list[str], message: str):
    if not condition:
        failures.append(message)


def _has_monthly5_promotion_wait_condition(position: dict) -> bool:
    wait_conditions = [
        item
        for item in (position.get("strategy_wait_conditions") or [])
        if isinstance(item, dict)
    ]
    return any(
        str(item.get("key") or "") == "monthly5_override_gate"
        and str(item.get("target") or "") == "promotion_ready=true"
        for item in wait_conditions
    )


def _selector_allows_action(primary_direction: str, shadow_action: str) -> bool:
    primary_direction = str(primary_direction or "").lower()
    shadow_action = str(shadow_action or "").lower()
    if shadow_action not in {"evaluate_long", "evaluate_short"}:
        return True
    if primary_direction == "long_or_short":
        return True
    if primary_direction == "long":
        return shadow_action == "evaluate_long"
    return False


def _load_recent_actual_trade_markets(db_path: Path, limit: int = 500) -> list[dict]:
    if not db_path.exists():
        return []
    try:
        with sqlite3.connect(str(db_path), timeout=5) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT id, question, market_json
                  FROM analysis_episode
                 WHERE question LIKE 'actual-trade:%'
                    OR market_json LIKE '%"actual_trade"%'
                 ORDER BY created_at DESC, id DESC
                 LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
    except sqlite3.Error:
        return []

    markets = []
    for row in rows:
        try:
            market = json.loads(row["market_json"] or "{}")
        except Exception:
            market = {}
        if not isinstance(market, dict):
            market = {}
        market["_episode_id"] = int(row["id"])
        market["_question"] = str(row["question"] or "")
        markets.append(market)
    return markets


def _is_monthly5_actual_trade_market(market: dict) -> bool:
    if not bool(market.get("actual_trade")):
        return False
    monthly5 = market.get("monthly5") if isinstance(market.get("monthly5"), dict) else {}
    markers = {
        str(market.get("actual_trade_source") or ""),
        str(market.get("source") or ""),
        str(market.get("monthly5_trade_source") or ""),
        str(monthly5.get("trade_source") or ""),
    }
    if "monthly5_market_selection" in markers:
        return True
    return "actual-trade:monthly5_market_selection:" in str(market.get("_question") or "")


def _verify_actual_trade_monthly5_metadata(db_path: Path, limit: int = 500) -> list[str]:
    failures: list[str] = []
    for market in _load_recent_actual_trade_markets(db_path, limit=limit):
        if not _is_monthly5_actual_trade_market(market):
            continue
        prefix = f"monthly5 actual trade episode={market.get('_episode_id')}"
        monthly5 = market.get("monthly5") if isinstance(market.get("monthly5"), dict) else {}
        _require(bool(monthly5), failures, f"{prefix} missing monthly5 metadata")
        _require(
            str(monthly5.get("trade_source") or "") == "monthly5_market_selection",
            failures,
            f"{prefix} monthly5 trade_source mismatch",
        )
        _require(
            str(market.get("monthly5_trade_source") or "") == "monthly5_market_selection",
            failures,
            f"{prefix} missing flat monthly5_trade_source",
        )
        _require(bool(monthly5.get("selected_plan")), failures, f"{prefix} missing selected_plan")
        _require(bool(monthly5.get("shadow_action")), failures, f"{prefix} missing shadow_action")
        _require(bool(monthly5.get("selector_key")), failures, f"{prefix} missing selector_key")
        _require(bool(market.get("monthly5_selector_key")), failures, f"{prefix} missing flat selector_key")
        _require(
            str(monthly5.get("selector_source") or "") == monthly5_shadow.RESEARCH_SELECTOR_SOURCE,
            failures,
            f"{prefix} selector source mismatch",
        )
        leverage = _safe_float(monthly5.get("max_leverage"), 0.0)
        _require(1.0 <= leverage <= 5.0, failures, f"{prefix} leverage missing or exceeds 5x")
        _require(
            _safe_float(market.get("monthly5_max_leverage"), 0.0) == leverage,
            failures,
            f"{prefix} flat leverage mismatch",
        )
    return failures


def _verify_spec_and_summary(spec: dict) -> list[str]:
    evidence = spec.get("backtest_evidence") if isinstance(spec.get("backtest_evidence"), dict) else {}
    summary_path = Path(str(evidence.get("source_summary") or ""))
    summary = _load_json(summary_path)
    failures = candidate_failures(summary, spec)
    policy = spec.get("policy") if isinstance(spec.get("policy"), dict) else {}
    risk = policy.get("risk") if isinstance(policy.get("risk"), dict) else {}
    objective = spec.get("objective") if isinstance(spec.get("objective"), dict) else {}

    _require(spec.get("strategy_id") == monthly5_shadow.STRATEGY_ID, failures, "strategy_id does not match code")
    _require(
        str(evidence.get("candidate_name") or "") == monthly5_shadow.SELECTED_CANDIDATE,
        failures,
        "selected candidate does not match code",
    )
    _require(_safe_float(objective.get("monthly_return_floor_pct")) == 5.0, failures, "monthly objective is not 5%")
    _require(int(objective.get("max_leverage", 0)) <= 5, failures, "spec leverage exceeds 5x")
    _require(
        _safe_float(risk.get("monthly_lock_pct")) == monthly5_shadow.MONTHLY_LOCK_PCT,
        failures,
        "monthly lock pct drift",
    )
    _require(
        _safe_float(risk.get("monthly_recovery_trigger_pct")) == monthly5_shadow.MONTHLY_RECOVERY_TRIGGER_PCT,
        failures,
        "monthly recovery trigger drift",
    )
    _require(
        _safe_float(risk.get("intraday_stop_pct")) == monthly5_shadow.INTRADAY_STOP_PCT,
        failures,
        "intraday stop pct drift",
    )
    _require(
        _safe_float(risk.get("post_lock_exposure_scale")) == monthly5_shadow.POST_LOCK_EXPOSURE_SCALE,
        failures,
        "post-lock exposure scale drift",
    )
    return failures


def _verify_shadow_state(name: str, shadow: dict, spec: dict, max_age_sec: float | None) -> list[str]:
    failures: list[str] = []
    evidence = spec.get("backtest_evidence") if isinstance(spec.get("backtest_evidence"), dict) else {}
    _require(bool(shadow), failures, f"{name} missing monthly5_shadow")
    if not shadow:
        return failures
    _require(shadow.get("strategy_id") == spec.get("strategy_id"), failures, f"{name} strategy_id mismatch")
    _require(shadow.get("selected_candidate") == evidence.get("candidate_name"), failures, f"{name} candidate mismatch")
    _require(shadow.get("shadow_only") is True, failures, f"{name} must remain shadow_only")
    _require(_safe_float(shadow.get("max_leverage")) <= 5.0, failures, f"{name} leverage exceeds 5x")
    _require(_safe_float(shadow.get("monthly_lock_pct")) == monthly5_shadow.MONTHLY_LOCK_PCT, failures, f"{name} lock pct drift")
    _require(
        _safe_float(shadow.get("monthly_recovery_trigger_pct")) == monthly5_shadow.MONTHLY_RECOVERY_TRIGGER_PCT,
        failures,
        f"{name} recovery trigger drift",
    )
    _require(_safe_float(shadow.get("intraday_stop_pct")) == monthly5_shadow.INTRADAY_STOP_PCT, failures, f"{name} intraday stop drift")
    _require(
        _safe_float(shadow.get("post_lock_exposure_scale")) == monthly5_shadow.POST_LOCK_EXPOSURE_SCALE,
        failures,
        f"{name} post-lock exposure drift",
    )
    _require(
        _safe_float(shadow.get("recovery_exposure_scale")) == monthly5_shadow.RECOVERY_EXPOSURE_SCALE,
        failures,
        f"{name} recovery exposure drift",
    )
    selection = shadow.get("market_selection") if isinstance(shadow.get("market_selection"), dict) else {}
    _require(bool(selection), failures, f"{name} missing market_selection")
    if selection:
        _require(
            int(selection.get("selector_policy_version", 0)) >= monthly5_shadow.SELECTOR_POLICY_VERSION,
            failures,
            f"{name} selector policy version stale",
        )
        _require(
            str(selection.get("selector_source") or ""),
            failures,
            f"{name} selector_source missing",
        )
        _require(
            str(selection.get("selector_policy_source") or ""),
            failures,
            f"{name} selector_policy_source missing",
        )
        _require(_safe_float(selection.get("max_leverage")) <= 5.0, failures, f"{name} selection leverage exceeds 5x")
        _require(0.0 <= _safe_float(selection.get("exposure_cap"), -1.0) <= 1.0, failures, f"{name} exposure cap out of range")
        _require(str(selection.get("selected_plan") or ""), failures, f"{name} selected_plan missing")
        _require(str(selection.get("shadow_action") or ""), failures, f"{name} shadow_action missing")
    if max_age_sec is not None:
        age = time.time() - _safe_float(shadow.get("updated_ts"), 0.0)
        _require(age <= max_age_sec, failures, f"{name} shadow state stale: age={age:.1f}s > {max_age_sec:.1f}s")
    return failures


def _verify_shadow_history(name: str, rows: list[dict], latest_shadow: dict, spec: dict, max_age_sec: float | None) -> list[str]:
    failures: list[str] = []
    evidence = spec.get("backtest_evidence") if isinstance(spec.get("backtest_evidence"), dict) else {}
    _require(bool(rows), failures, f"{name} history missing rows")
    if not rows:
        return failures
    latest = rows[-1]
    _require(latest.get("schema_version") == 1, failures, f"{name} history schema mismatch")
    _require(latest.get("strategy_id") == spec.get("strategy_id"), failures, f"{name} history strategy_id mismatch")
    _require(latest.get("selected_candidate") == evidence.get("candidate_name"), failures, f"{name} history candidate mismatch")
    _require(latest.get("shadow_only") is True, failures, f"{name} history must remain shadow_only")
    _require(_safe_float(latest.get("max_leverage")) <= 5.0, failures, f"{name} history leverage exceeds 5x")
    _require(
        int(latest.get("selector_policy_version", 0)) >= monthly5_shadow.SELECTOR_POLICY_VERSION,
        failures,
        f"{name} history selector policy version stale",
    )
    _require(str(latest.get("selector_source") or ""), failures, f"{name} history selector_source missing")
    _require(str(latest.get("selector_policy_source") or ""), failures, f"{name} history selector_policy_source missing")
    _require(0.0 <= _safe_float(latest.get("exposure_cap"), -1.0) <= 1.0, failures, f"{name} history exposure cap out of range")
    _require(str(latest.get("selected_plan") or ""), failures, f"{name} history selected_plan missing")
    _require(str(latest.get("shadow_action") or ""), failures, f"{name} history shadow_action missing")
    if latest_shadow:
        _require(latest.get("mode") == latest_shadow.get("mode"), failures, f"{name} history mode does not match shadow")
    if max_age_sec is not None:
        age = time.time() - _safe_float(latest.get("updated_ts"), 0.0)
        _require(age <= max_age_sec, failures, f"{name} history stale: age={age:.1f}s > {max_age_sec:.1f}s")
    return failures


def _verify_guard_scenarios() -> list[str]:
    failures: list[str] = []
    risk_off = {
        "mode": "post_lock_floor_guard",
        "max_leverage": 5,
        "market_selection": {
            "selected_plan": "risk_off",
            "shadow_action": "risk_off",
            "exposure_cap": 0.0,
        },
    }
    execution = monthly5_shadow.build_execution_guard(risk_off, direction="long", requested_size=0.5)
    _require(not execution.get("allowed"), failures, "execution guard must block risk_off")
    _require(_safe_float(execution.get("adjusted_size")) == 0.0, failures, "risk_off execution size must be 0")

    post_lock = {
        "mode": "post_lock",
        "max_leverage": 5,
        "market_selection": {
            "selected_plan": "post_lock_low_exposure",
            "shadow_action": "reduced_exposure",
            "exposure_cap": monthly5_shadow.POST_LOCK_EXPOSURE_SCALE,
        },
    }
    execution = monthly5_shadow.build_execution_guard(post_lock, direction="long", requested_size=0.5)
    _require(bool(execution.get("allowed")), failures, "post-lock execution should remain allowed")
    _require(
        _safe_float(execution.get("adjusted_size")) == monthly5_shadow.POST_LOCK_EXPOSURE_SCALE,
        failures,
        "post-lock execution guard must cap to low exposure",
    )
    position = monthly5_shadow.build_position_guard(post_lock, current_size=0.5)
    _require(position.get("action") == "reduce_to_cap", failures, "post-lock position guard must reduce to cap")
    _require(
        _safe_float(position.get("target_size")) == monthly5_shadow.POST_LOCK_EXPOSURE_SCALE,
        failures,
        "post-lock position target must equal exposure scale",
    )
    position = monthly5_shadow.build_position_guard(risk_off, current_size=0.5)
    _require(position.get("action") == "close_all", failures, "risk_off position guard must close all")
    return failures


def _verify_promotion_gate(position: dict, max_age_sec: float | None = None) -> list[str]:
    failures: list[str] = []
    gate_enabled = str(
        os.getenv("MONTHLY5_SIGNAL_OVERRIDE_REQUIRE_PROMOTION_READY", "1") or "1"
    ).strip().lower() not in {"0", "false", "no", "off"}
    _require(gate_enabled, failures, "monthly5 signal override promotion gate disabled")

    shadow = _shadow_from_position(position)
    if shadow:
        _require(
            isinstance(shadow.get("promotion_ready"), bool),
            failures,
            "position monthly5_shadow missing promotion_ready boolean",
        )
    _require(
        isinstance(shadow.get("promotion_blockers"), list),
        failures,
        "position monthly5_shadow missing promotion_blockers list",
    )
    readiness = position.get("monthly5_readiness") if isinstance(position.get("monthly5_readiness"), dict) else {}
    if readiness:
        if max_age_sec is not None:
            _require(
                _safe_float(readiness.get("latest_age_sec"), max_age_sec + 1.0) <= max_age_sec,
                failures,
                "monthly5 readiness state stale",
            )
        _require(
            bool(shadow.get("promotion_ready", False)) == bool(readiness.get("promotion_ready", False)),
            failures,
            "position monthly5_shadow promotion_ready does not match readiness",
        )
        _require(
            list(shadow.get("promotion_blockers") or []) == list(readiness.get("promotion_blockers") or []),
            failures,
            "position monthly5_shadow promotion_blockers do not match readiness",
        )
        blockers = [str(item) for item in readiness.get("promotion_blockers") or []]
        details = {
            str(item.get("code") or ""): item
            for item in (readiness.get("promotion_blocker_details") or [])
            if isinstance(item, dict)
        }
        if (
            not bool(readiness.get("promotion_ready", False))
            and ("sample_count" in blockers or "sample_span" in blockers)
        ):
            _require(
                _safe_float(readiness.get("promotion_earliest_review_ts"), 0.0) > 0.0,
                failures,
                "monthly5 readiness missing earliest promotion review ts",
            )
        if "sample_count" in blockers:
            sample_count_ready_ts = _safe_float(readiness.get("sample_count_ready_ts"), 0.0)
            _require(sample_count_ready_ts > 0.0, failures, "monthly5 readiness missing sample_count_ready_ts")
            _require(
                _safe_float(details.get("sample_count", {}).get("ready_ts"), 0.0) == sample_count_ready_ts,
                failures,
                "monthly5 readiness sample_count blocker ready_ts mismatch",
            )
        if "sample_span" in blockers:
            sample_span_ready_ts = _safe_float(readiness.get("sample_span_ready_ts"), 0.0)
            _require(sample_span_ready_ts > 0.0, failures, "monthly5 readiness missing sample_span_ready_ts")
            _require(
                _safe_float(details.get("sample_span", {}).get("ready_ts"), 0.0) == sample_span_ready_ts,
                failures,
                "monthly5 readiness sample_span blocker ready_ts mismatch",
            )
            _require(
                _safe_float(readiness.get("promotion_earliest_review_ts"), 0.0) >= sample_span_ready_ts,
                failures,
                "monthly5 earliest promotion review before sample span ready",
            )
    override = position.get("monthly5_signal_override")
    override = override if isinstance(override, dict) else {}
    selection = shadow.get("market_selection") if isinstance(shadow.get("market_selection"), dict) else {}
    shadow_action = str(selection.get("shadow_action") or "")
    strategy_signal = str(position.get("strategy_signal") or "").lower()
    strategy_status = str(position.get("strategy_execution_status") or "").lower()
    if (
        shadow
        and shadow_action in {"evaluate_long", "evaluate_short"}
        and not bool(shadow.get("promotion_ready", False))
        and strategy_signal == "wait"
        and strategy_status == "waiting"
    ):
        if str(override.get("reason") or "") == "monthly5_promotion_not_ready":
            _require(
                list(override.get("promotion_blockers") or []) == list(shadow.get("promotion_blockers") or [])[:5],
                failures,
                "monthly5 waiting entry override blockers do not match readiness",
            )
        else:
            _require(
                _has_monthly5_promotion_wait_condition(position),
                failures,
                "monthly5 waiting entry promotion gate missing from wait conditions",
            )
    if override.get("applied") is True:
        _require(
            bool(shadow.get("promotion_ready", False)),
            failures,
            "monthly5 signal override applied before promotion_ready",
        )
    monthly5_entry_selection = (
        position.get("monthly5_entry_selection")
        if isinstance(position.get("monthly5_entry_selection"), dict)
        else {}
    )
    if bool(position.get("open", False)) and monthly5_entry_selection:
        _require(
            str(position.get("trade_source") or "") == "monthly5_market_selection",
            failures,
            "monthly5 open position trade_source mismatch",
        )
    if str(position.get("trade_source") or "") == "monthly5_market_selection":
        _require(
            bool(monthly5_entry_selection),
            failures,
            "monthly5 trade_source missing entry selection",
        )
    return failures


def _verify_monthly5_price_state(position: dict, max_age_sec: float | None) -> list[str]:
    failures: list[str] = []
    shadow = _shadow_from_position(position)
    selection = shadow.get("market_selection") if isinstance(shadow.get("market_selection"), dict) else {}
    if str(selection.get("shadow_action") or "") not in {"evaluate_long", "evaluate_short"}:
        return failures
    if str(position.get("strategy_signal") or "").lower() != "wait":
        return failures
    if str(position.get("strategy_execution_status") or "").lower() != "waiting":
        return failures

    strategy_price = _safe_float(position.get("strategy_price"), 0.0)
    mark_price = _safe_float(position.get("binance_mark_price"), 0.0)
    strategy_ts = _safe_float(position.get("strategy_price_ts"), 0.0)
    mark_ts = _safe_float(position.get("binance_mark_price_ts"), 0.0)
    _require(strategy_price > 0.0, failures, "monthly5 waiting entry missing strategy price")
    _require(mark_price > 0.0, failures, "monthly5 waiting entry missing Binance Mark Price")
    _require(
        str(position.get("strategy_price_source") or "") == "binance_mark_price",
        failures,
        "monthly5 waiting entry strategy price source is not Binance Mark Price",
    )
    _require(strategy_ts > 0.0, failures, "monthly5 waiting entry missing strategy price timestamp")
    _require(mark_ts > 0.0, failures, "monthly5 waiting entry missing Binance Mark Price timestamp")
    if strategy_price > 0.0 and mark_price > 0.0:
        price_gap = abs(strategy_price - mark_price) / max(mark_price, 1e-9)
        _require(price_gap <= 0.0005, failures, "monthly5 waiting entry strategy price drifted from Binance Mark Price")
    if strategy_ts > 0.0 and mark_ts > 0.0:
        _require(abs(strategy_ts - mark_ts) <= 2.0, failures, "monthly5 waiting entry price timestamps diverged")
    if max_age_sec is not None:
        now_ts = time.time()
        _require(
            now_ts - mark_ts <= max_age_sec,
            failures,
            "monthly5 waiting entry Binance Mark Price stale",
        )
        _require(
            now_ts - strategy_ts <= max_age_sec,
            failures,
            "monthly5 waiting entry strategy price stale",
        )
    return failures


def _verify_monthly5_real_execution_state(position: dict) -> list[str]:
    failures: list[str] = []
    shadow = _shadow_from_position(position)
    selection = shadow.get("market_selection") if isinstance(shadow.get("market_selection"), dict) else {}
    if str(selection.get("shadow_action") or "") not in {"evaluate_long", "evaluate_short"}:
        return failures
    if str(position.get("strategy_signal") or "").lower() != "wait":
        return failures

    _require(
        str(position.get("execution_priority") or "") == "real_order",
        failures,
        "monthly5 execution priority is not real_order",
    )
    _require(
        str(position.get("execution_mode") or "") == "real",
        failures,
        "monthly5 execution mode is not real",
    )
    if bool(position.get("strategy_actual_open", False)):
        _require(
            str(position.get("position_source") or "") == "binance",
            failures,
            "monthly5 actual open position source is not Binance",
        )
    return failures


def _verify_monthly5_guard_alignment(position: dict) -> list[str]:
    failures: list[str] = []
    shadow = _shadow_from_position(position)
    selection = shadow.get("market_selection") if isinstance(shadow.get("market_selection"), dict) else {}
    if not selection:
        return failures

    execution_guard = (
        position.get("monthly5_execution_guard")
        if isinstance(position.get("monthly5_execution_guard"), dict)
        else {}
    )
    position_guard = (
        position.get("monthly5_position_guard")
        if isinstance(position.get("monthly5_position_guard"), dict)
        else {}
    )
    _require(bool(execution_guard), failures, "monthly5 execution guard missing")
    if execution_guard:
        _require(
            str(execution_guard.get("selected_plan") or "") == str(selection.get("selected_plan") or ""),
            failures,
            "monthly5 execution guard selected_plan mismatch",
        )
        _require(
            str(execution_guard.get("shadow_action") or "") == str(selection.get("shadow_action") or ""),
            failures,
            "monthly5 execution guard shadow_action mismatch",
        )
        _require(
            abs(_safe_float(execution_guard.get("exposure_cap"), -1.0) - _safe_float(selection.get("exposure_cap"), -2.0))
            <= 1e-9,
            failures,
            "monthly5 execution guard exposure_cap mismatch",
        )
        _require(
            int(_safe_float(execution_guard.get("max_leverage"), 0.0))
            == int(_safe_float(selection.get("max_leverage"), 0.0)),
            failures,
            "monthly5 execution guard max_leverage mismatch",
        )
        if str(selection.get("selected_plan") or "") == "risk_off":
            _require(execution_guard.get("allowed") is False, failures, "monthly5 risk_off execution guard must block")
            _require(_safe_float(execution_guard.get("adjusted_size"), 0.0) == 0.0, failures, "monthly5 risk_off size must be zero")
        else:
            _require(
                0.0 <= _safe_float(execution_guard.get("adjusted_size"), 0.0) <= _safe_float(selection.get("exposure_cap"), 0.0),
                failures,
                "monthly5 execution guard adjusted size exceeds exposure cap",
            )
    if position_guard:
        _require(
            str(position_guard.get("selected_plan") or "") == str(selection.get("selected_plan") or ""),
            failures,
            "monthly5 position guard selected_plan mismatch",
        )
        _require(
            str(position_guard.get("shadow_action") or "") == str(selection.get("shadow_action") or ""),
            failures,
            "monthly5 position guard shadow_action mismatch",
        )
        _require(
            abs(_safe_float(position_guard.get("exposure_cap"), -1.0) - _safe_float(selection.get("exposure_cap"), -2.0))
            <= 1e-9,
            failures,
            "monthly5 position guard exposure_cap mismatch",
        )
    return failures


def _verify_monthly5_wait_conditions(position: dict) -> list[str]:
    failures: list[str] = []
    shadow = _shadow_from_position(position)
    selection = shadow.get("market_selection") if isinstance(shadow.get("market_selection"), dict) else {}
    if str(selection.get("shadow_action") or "") not in {"evaluate_long", "evaluate_short"}:
        return failures
    if str(position.get("strategy_signal") or "").lower() != "wait":
        return failures
    if str(position.get("strategy_execution_status") or "").lower() != "waiting":
        return failures

    if bool(shadow.get("promotion_ready", False)):
        return failures
    wait_conditions = [
        item
        for item in (position.get("strategy_wait_conditions") or [])
        if isinstance(item, dict)
    ]
    keys = [str(item.get("key") or "") for item in wait_conditions]
    _require(
        "monthly5_override_gate" in keys,
        failures,
        "monthly5 promotion blocker missing from strategy wait conditions",
    )
    if "monthly5_override_gate" in keys:
        item = wait_conditions[keys.index("monthly5_override_gate")]
        _require(
            str(item.get("target") or "") == "promotion_ready=true",
            failures,
            "monthly5 wait condition target must be promotion_ready=true",
        )
    return failures


def _verify_monthly5_open_position_safety(position: dict) -> list[str]:
    failures: list[str] = []
    monthly5_entry_selection = (
        position.get("monthly5_entry_selection")
        if isinstance(position.get("monthly5_entry_selection"), dict)
        else {}
    )
    is_monthly5_position = (
        bool(position.get("open", False))
        and (
            str(position.get("trade_source") or "") == "monthly5_market_selection"
            or bool(monthly5_entry_selection)
        )
    )
    if not is_monthly5_position:
        return failures

    direction = str(position.get("direction") or "").lower()
    entry = _safe_float(position.get("entry"), 0.0)
    tp = _safe_float(position.get("tp"), 0.0)
    sl = _safe_float(position.get("sl"), 0.0)
    binance_qty = _safe_float(position.get("binance_qty"), 0.0)
    max_leverage = int(_safe_float(monthly5_entry_selection.get("max_leverage"), 5.0) or 5)
    leverage = int(_safe_float(position.get("lev"), 0.0) or 0)

    _require(direction in {"long", "short"}, failures, "monthly5 open position direction invalid")
    _require(entry > 0.0, failures, "monthly5 open position missing entry")
    _require(tp > 0.0, failures, "monthly5 open position missing TP")
    _require(sl > 0.0, failures, "monthly5 open position missing SL")
    _require(binance_qty > 0.0, failures, "monthly5 open position missing Binance quantity")
    _require(
        str(position.get("position_source") or "") == "binance",
        failures,
        "monthly5 open position source is not Binance",
    )
    _require(1 <= leverage <= max(1, min(5, max_leverage)), failures, "monthly5 open position leverage exceeds selector cap")
    if direction == "long" and entry > 0.0 and tp > 0.0 and sl > 0.0:
        _require(tp > entry, failures, "monthly5 long TP must be above entry")
        _require(sl < entry, failures, "monthly5 long SL must be below entry")
    if direction == "short" and entry > 0.0 and tp > 0.0 and sl > 0.0:
        _require(tp < entry, failures, "monthly5 short TP must be below entry")
        _require(sl > entry, failures, "monthly5 short SL must be above entry")
    return failures


def _verify_research_selector_artifact(position: dict, spec: dict) -> list[str]:
    failures: list[str] = []
    probe = monthly5_research_selector.build_research_selector_probe()
    evidence = spec.get("backtest_evidence") if isinstance(spec.get("backtest_evidence"), dict) else {}
    _require(bool(probe.get("artifact_available")), failures, "monthly5 research selector artifact missing")
    _require(
        probe.get("selected_candidate") == evidence.get("candidate_name"),
        failures,
        "monthly5 research selector candidate mismatch",
    )
    _require(
        probe.get("selector_source") == monthly5_shadow.RESEARCH_SELECTOR_SOURCE,
        failures,
        "monthly5 research selector source mismatch",
    )
    _require(
        0 < _safe_float(probe.get("max_leverage")) <= 5.0,
        failures,
        "monthly5 research selector leverage missing or exceeds 5x",
    )
    _require(
        str(probe.get("primary_direction") or "") in {"long", "long_or_short"},
        failures,
        "monthly5 research selector direction is not actionable",
    )
    panel_probe = position.get("monthly5_research_selector")
    panel_probe = panel_probe if isinstance(panel_probe, dict) else {}
    if panel_probe:
        _require(
            panel_probe.get("top_pick") == probe.get("top_pick"),
            failures,
            "position monthly5_research_selector top_pick mismatch",
        )
    live_input = position.get("monthly5_live_selector_input")
    live_input = live_input if isinstance(live_input, dict) else {}
    if live_input:
        _require(
            live_input.get("selector_source") == monthly5_shadow.RESEARCH_SELECTOR_SOURCE,
            failures,
            "monthly5 live selector input source mismatch",
        )
        _require(bool(live_input.get("usable")), failures, "monthly5 live selector input not usable")
        _require(
            int(live_input.get("cache_feature_count") or 0)
            == monthly5_research_selector.EXPECTED_SHORT_MARKET_STATE_FEATURES,
            failures,
            "monthly5 live selector input feature count mismatch",
        )
        _require(
            int(live_input.get("daily_rows") or 0)
            >= monthly5_research_selector.REQUIRED_LIVE_DAILY_ROWS,
            failures,
            "monthly5 live selector input daily warmup insufficient",
        )
    live_decision = position.get("monthly5_live_selector_decision")
    live_decision = live_decision if isinstance(live_decision, dict) else {}
    if live_decision:
        _require(
            live_decision.get("selector_source") == monthly5_shadow.RESEARCH_SELECTOR_SOURCE,
            failures,
            "monthly5 live selector decision source mismatch",
        )
        _require(bool(live_decision.get("usable")), failures, "monthly5 live selector decision not usable")
        _require(
            str(live_decision.get("selected_key") or ""),
            failures,
            "monthly5 live selector decision missing selected key",
        )
        _require(
            str(live_decision.get("primary_direction") or "") in {"long", "long_or_short"},
            failures,
            "monthly5 live selector decision direction unsupported",
        )
        shadow = _shadow_from_position(position)
        selection = shadow.get("market_selection") if isinstance(shadow.get("market_selection"), dict) else {}
        if selection and selection.get("selector_source") == monthly5_shadow.RESEARCH_SELECTOR_SOURCE:
            _require(
                selection.get("selector_alignment") == "live_similar_day",
                failures,
                "monthly5 market selection not aligned to live similar-day selector",
            )
            _require(
                str(selection.get("selector_key") or "") == str(live_decision.get("selected_key") or ""),
                failures,
                "monthly5 market selection selector key mismatch",
            )
            _require(
                str(selection.get("selector_primary_direction") or "")
                == str(live_decision.get("primary_direction") or ""),
                failures,
                "monthly5 market selection selector direction mismatch",
            )
            _require(
                _selector_allows_action(
                    str(live_decision.get("primary_direction") or ""),
                    str(selection.get("shadow_action") or ""),
                ),
                failures,
                "monthly5 market selection action not allowed by live selector direction",
            )
            _require(
                int(selection.get("max_leverage") or 0)
                <= int(live_decision.get("max_leverage") or 5),
                failures,
                "monthly5 market selection leverage exceeds live selector key",
            )
    return failures


def _taipei_ts(value: str) -> float:
    return datetime.fromisoformat(value).replace(tzinfo=ZoneInfo("Asia/Taipei")).timestamp()


def _bullish_selection(shadow: dict, *, signal: str = "wait", market_state: str = "trend") -> dict:
    return monthly5_shadow.build_market_selection(
        shadow,
        strategy_signal=signal,
        strategy_context={"htf": 1, "mid_trend": 1, "macro_bias": 1.0},
        host_logic={"direction": "long", "confidence": 0.8},
        macro_alignment={"score": 1.8, "hard_block": False},
        donchian_state={"state": market_state, "action": "open"},
    )


def _bullish_profile_wait_selection(shadow: dict) -> dict:
    return monthly5_shadow.build_market_selection(
        shadow,
        strategy_signal="wait",
        strategy_execution_reason="觀望（MLX回測輪廓不佳）",
        strategy_context={"htf": 1, "mid_trend": 1, "macro_bias": 1.0},
        host_logic={"direction": "long", "confidence": 0.8},
        macro_alignment={"score": 1.8, "hard_block": False},
        donchian_state={"state": "chop", "action": "reduce"},
    )


def _bearish_selection(shadow: dict, *, signal: str = "wait") -> dict:
    return monthly5_shadow.build_market_selection(
        shadow,
        strategy_signal=signal,
        strategy_context={"htf": -1, "mid_trend": -1, "macro_bias": -1.0},
        host_logic={"direction": "short", "confidence": 0.8},
        macro_alignment={"score": 1.8, "hard_block": False},
        donchian_state={"state": "trend", "action": "open"},
    )


def _with_selection(shadow: dict, selection: dict) -> dict:
    payload = dict(shadow)
    payload["market_selection"] = selection
    return payload


def _verify_end_to_end_scenarios() -> list[str]:
    failures: list[str] = []

    locked_jan = {
        "month_key": "2026-01",
        "day_key": "2026-01-31",
        "month_start_equity": 1000.0,
        "day_start_equity": 1040.0,
        "lock_reached": True,
        "month_high_pnl_pct": 6.0,
    }
    reset = monthly5_shadow.update_shadow_state(
        locked_jan,
        now_ts=_taipei_ts("2026-02-01T00:05:00"),
        margin_balance=1000.0,
        mark_price=64000.0,
    )
    _require(reset.get("month_key") == "2026-02", failures, "e2e month rollover must reset month key")
    _require(reset.get("month_start_equity") == 1000.0, failures, "e2e month rollover must reset start equity")
    _require(reset.get("lock_reached") is False, failures, "e2e month rollover must clear monthly lock")
    _require(reset.get("mode") == "normal", failures, "e2e month rollover must return to normal mode")

    normal_short_selection = _bearish_selection(reset)
    _require(
        normal_short_selection.get("selected_plan") == "normal_short_selector",
        failures,
        "e2e bearish market must select normal short strategy",
    )
    normal_short = _with_selection(reset, normal_short_selection)
    short_guard = monthly5_shadow.build_execution_guard(normal_short, direction="short", requested_size=0.8)
    long_guard = monthly5_shadow.build_execution_guard(normal_short, direction="long", requested_size=0.8)
    _require(short_guard.get("allowed") is True, failures, "e2e bearish short execution must be allowed")
    _require(long_guard.get("reason_code") == "monthly5_direction_mismatch", failures, "e2e bearish market must block long entries")

    profile_wait_selection = _bullish_profile_wait_selection(reset)
    _require(
        profile_wait_selection.get("selected_plan") == "profile_quality_shadow_probe",
        failures,
        "e2e strong profile-bad host wait must use low exposure shadow probe",
    )
    _require(
        profile_wait_selection.get("shadow_action") == "evaluate_long",
        failures,
        "e2e strong profile-bad host wait must collect long shadow evidence",
    )
    _require(
        _safe_float(profile_wait_selection.get("exposure_cap")) == monthly5_shadow.PROFILE_QUALITY_PROBE_EXPOSURE_CAP,
        failures,
        "e2e profile-quality shadow probe exposure cap must match spec",
    )

    locked = monthly5_shadow.update_shadow_state(
        {
            "month_key": "2026-02",
            "day_key": "2026-02-10",
            "month_start_equity": 1000.0,
            "day_start_equity": 1000.0,
        },
        now_ts=_taipei_ts("2026-02-10T12:00:00"),
        margin_balance=1060.0,
        mark_price=65000.0,
    )
    locked_selection = _bullish_selection(locked)
    locked_with_selection = _with_selection(locked, locked_selection)
    _require(locked.get("mode") == "post_lock", failures, "e2e +5% month must enter post_lock")
    _require(locked_selection.get("selected_plan") == "post_lock_low_exposure", failures, "e2e post-lock must select low exposure plan")
    _require(
        _safe_float(locked_selection.get("exposure_cap")) == monthly5_shadow.POST_LOCK_EXPOSURE_SCALE,
        failures,
        "e2e post-lock exposure cap must match spec",
    )
    locked_execution = monthly5_shadow.build_execution_guard(locked_with_selection, direction="long", requested_size=0.5)
    locked_position = monthly5_shadow.build_position_guard(locked_with_selection, current_size=0.5)
    _require(_safe_float(locked_execution.get("adjusted_size")) == 0.15, failures, "e2e post-lock execution must cap new size")
    _require(locked_position.get("action") == "reduce_to_cap", failures, "e2e post-lock holding must reduce to cap")

    rolled_back = monthly5_shadow.update_shadow_state(
        locked,
        now_ts=_taipei_ts("2026-02-10T13:00:00"),
        margin_balance=1040.0,
        mark_price=64500.0,
    )
    rollback_selection = _bullish_selection(rolled_back)
    rollback_with_selection = _with_selection(rolled_back, rollback_selection)
    rollback_position = monthly5_shadow.build_position_guard(rollback_with_selection, current_size=0.15)
    _require(rolled_back.get("mode") == "post_lock_floor_guard", failures, "e2e lock rollback must trigger floor guard")
    _require(rollback_selection.get("selected_plan") == "risk_off", failures, "e2e floor guard must select risk_off")
    _require(rollback_position.get("action") == "close_all", failures, "e2e floor guard must close current position")

    intraday_stop = monthly5_shadow.update_shadow_state(
        {
            "month_key": "2026-02",
            "day_key": "2026-02-11",
            "month_start_equity": 1000.0,
            "day_start_equity": 1000.0,
        },
        now_ts=_taipei_ts("2026-02-11T14:00:00"),
        margin_balance=920.0,
        mark_price=63000.0,
    )
    intraday_selection = _bullish_selection(intraday_stop)
    intraday_execution = monthly5_shadow.build_execution_guard(
        _with_selection(intraday_stop, intraday_selection),
        direction="long",
        requested_size=0.3,
    )
    _require(intraday_stop.get("mode") == "intraday_stop", failures, "e2e -8% intraday must enter stop mode")
    _require(intraday_selection.get("selected_plan") == "risk_off", failures, "e2e intraday stop must select risk_off")
    _require(intraday_execution.get("allowed") is False, failures, "e2e intraday stop must block new entries")

    recovery = monthly5_shadow.update_shadow_state(
        {
            "month_key": "2026-02",
            "day_key": "2026-02-12",
            "month_start_equity": 1000.0,
            "day_start_equity": 950.0,
        },
        now_ts=_taipei_ts("2026-02-12T15:00:00"),
        margin_balance=910.0,
        mark_price=62000.0,
    )
    recovery_selection = _bullish_selection(recovery)
    recovery_execution = monthly5_shadow.build_execution_guard(
        _with_selection(recovery, recovery_selection),
        direction="long",
        requested_size=0.8,
    )
    _require(recovery.get("mode") == "recovery", failures, "e2e monthly drawdown must enter recovery without intraday stop")
    _require(
        recovery_selection.get("selected_plan") == "recovery_long_flat_selector",
        failures,
        "e2e bullish recovery must select recovery long-flat strategy",
    )
    _require(_safe_float(recovery_execution.get("adjusted_size")) == 0.5, failures, "e2e recovery execution must cap size to 0.5")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--position", default=str(DEFAULT_POSITION))
    parser.add_argument("--shadow", default=str(DEFAULT_SHADOW))
    parser.add_argument("--history", default=str(DEFAULT_HISTORY))
    parser.add_argument("--mlx-db", default=str(DEFAULT_MLX_DB))
    parser.add_argument("--require-history", action="store_true")
    parser.add_argument("--max-age-sec", type=float, default=None)
    args = parser.parse_args()

    spec = _load_json(Path(args.spec))
    position = _load_json(Path(args.position))
    shadow = _load_json(Path(args.shadow))

    failures = []
    failures.extend(_verify_spec_and_summary(spec))
    failures.extend(_verify_shadow_state("position", _shadow_from_position(position), spec, args.max_age_sec))
    failures.extend(_verify_shadow_state("shadow_file", shadow, spec, args.max_age_sec))
    failures.extend(_verify_promotion_gate(position, args.max_age_sec))
    failures.extend(_verify_monthly5_price_state(position, args.max_age_sec))
    failures.extend(_verify_monthly5_real_execution_state(position))
    failures.extend(_verify_monthly5_guard_alignment(position))
    failures.extend(_verify_monthly5_wait_conditions(position))
    failures.extend(_verify_monthly5_open_position_safety(position))
    failures.extend(_verify_research_selector_artifact(position, spec))
    history_path = Path(args.history)
    if history_path.exists():
        failures.extend(_verify_shadow_history("history_file", _load_jsonl(history_path), shadow, spec, args.max_age_sec))
    elif args.require_history:
        failures.append(f"history file missing: {history_path}")
    failures.extend(_verify_actual_trade_monthly5_metadata(Path(args.mlx_db)))
    failures.extend(_verify_guard_scenarios())
    failures.extend(_verify_end_to_end_scenarios())

    if failures:
        for item in failures:
            print(f"FAIL {item}")
        return 1

    position_shadow = _shadow_from_position(position)
    selection = position_shadow.get("market_selection") if isinstance(position_shadow.get("market_selection"), dict) else {}
    research_probe = monthly5_research_selector.build_research_selector_probe()
    live_input = position.get("monthly5_live_selector_input")
    live_input = live_input if isinstance(live_input, dict) else {}
    live_decision = position.get("monthly5_live_selector_decision")
    live_decision = live_decision if isinstance(live_decision, dict) else {}
    history_status = "history=ok" if Path(args.history).exists() else "history=missing"
    readiness_status = " ".join(_monthly5_readiness_summary_tokens(position))
    print(
        "PASS monthly5_runtime "
        f"strategy_id={position_shadow.get('strategy_id')} "
        f"mode={position_shadow.get('mode')} "
        f"selected_plan={selection.get('selected_plan')} "
        f"exposure_cap={selection.get('exposure_cap')} "
        f"research_top_pick={research_probe.get('top_pick')} "
        f"research_direction={research_probe.get('primary_direction')} "
        f"research_leverage={research_probe.get('max_leverage')} "
        f"live_selector_input={'ok' if live_input.get('usable') else 'pending'} "
        f"live_selector_decision={'ok' if live_decision.get('usable') else 'pending'} "
        f"{readiness_status} "
        f"{history_status} "
        "e2e=ok"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
