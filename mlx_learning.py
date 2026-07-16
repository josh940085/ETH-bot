import csv
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path

from runtime_paths import ai_data_path, ensure_parent_dir


DB_PATH = ai_data_path("mlx_agent_learning.sqlite3")
EVALUATION_HOURS = max(0.25, float(os.getenv("MLX_LEARNING_EVALUATION_HOURS", "1")))
MIN_MOVE_PCT = max(0.05, float(os.getenv("MLX_LEARNING_MIN_MOVE_PCT", "0.25")))
EVALUATION_INTERVAL_SEC = max(
    5.0, float(os.getenv("MLX_LEARNING_EVALUATION_INTERVAL_SEC", "15"))
)
LEARNING_CONTEXT_LIMIT = max(5, int(os.getenv("MLX_LEARNING_CONTEXT_LIMIT", "12")))
HISTORY_IMPORT_INTERVAL_SEC = max(
    60.0, float(os.getenv("MLX_HISTORY_IMPORT_INTERVAL_SEC", "300"))
)
_LOCK = threading.Lock()
_LAST_EVALUATION_TS = 0.0
_LAST_HISTORY_IMPORT_TS = 0.0
_LAST_TURNING_POINTS_MTIME = 0.0
_LAST_CONTRAST_EXAMPLES_MTIME = 0.0
_LAST_METADATA_BACKFILL_TS = 0.0


def _connect():
    ensure_parent_dir(DB_PATH)
    connection = sqlite3.connect(str(DB_PATH), timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_episode (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL NOT NULL,
            evaluated_at REAL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            direction TEXT NOT NULL,
            question TEXT NOT NULL,
            response TEXT NOT NULL,
            market_json TEXT NOT NULL,
            return_pct REAL,
            success INTEGER,
            tp_price REAL,
            sl_price REAL,
            activated_at REAL
        )
        """
    )
    existing_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(analysis_episode)")
    }
    text_columns = {
        "structured_json": "TEXT",
        "primary_reason": "TEXT",
        "market_regime": "TEXT",
        "shadow_grade": "TEXT",
        "factor_json": "TEXT",
        "strategy_version": "TEXT",
    }
    real_columns = {"confidence": "REAL"}
    for column in ("tp_price", "sl_price", "activated_at"):
        if column not in existing_columns:
            connection.execute(
                f"ALTER TABLE analysis_episode ADD COLUMN {column} REAL"
            )
    for column, column_type in text_columns.items():
        if column not in existing_columns:
            connection.execute(
                f"ALTER TABLE analysis_episode ADD COLUMN {column} {column_type}"
            )
    for column, column_type in real_columns.items():
        if column not in existing_columns:
            connection.execute(
                f"ALTER TABLE analysis_episode ADD COLUMN {column} {column_type}"
            )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_outcome (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            closed_at REAL NOT NULL,
            result INTEGER NOT NULL,
            close_reason TEXT NOT NULL,
            close_price REAL NOT NULL
        )
        """
    )
    strategy_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(strategy_outcome)")
    }
    if "strategy_version" not in strategy_columns:
        connection.execute(
            "ALTER TABLE strategy_outcome ADD COLUMN strategy_version TEXT"
        )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS historical_example (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            source_key TEXT NOT NULL UNIQUE,
            imported_at REAL NOT NULL,
            direction TEXT NOT NULL,
            response TEXT NOT NULL,
            market_json TEXT NOT NULL,
            success INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS higher_timeframe_observation (
            candle_key TEXT PRIMARY KEY,
            observed_at REAL NOT NULL,
            context_json TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS auto_analysis_log (
            period_key TEXT PRIMARY KEY,
            claimed_at REAL NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_report_log (
            report_date TEXT PRIMARY KEY,
            sent_at REAL NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS sl_review_event (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL NOT NULL,
            direction TEXT NOT NULL,
            verdict TEXT NOT NULL,
            issue_json TEXT NOT NULL,
            review_json TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS gpt_teacher_review (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_key TEXT NOT NULL UNIQUE,
            created_at REAL NOT NULL,
            completed_at REAL,
            teacher_model TEXT NOT NULL,
            episode_ids_json TEXT NOT NULL,
            lesson TEXT,
            status TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection


def claim_auto_analysis(period_key):
    period_key = str(period_key or "").strip()
    if not period_key:
        return False
    with _LOCK, _connect() as connection:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO auto_analysis_log (period_key, claimed_at)
            VALUES (?, ?)
            """,
            (period_key, time.time()),
        )
        connection.commit()
        return cursor.rowcount > 0


def release_auto_analysis(period_key):
    with _LOCK, _connect() as connection:
        connection.execute(
            "DELETE FROM auto_analysis_log WHERE period_key = ?",
            (str(period_key or ""),),
        )
        connection.commit()


def gpt_teacher_review_batch(limit=8):
    """Return evaluated MLX episodes that have not been reviewed by GPT yet."""
    limit = max(2, min(20, int(limit)))
    with _LOCK, _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, created_at, evaluated_at, direction, question, response,
                   market_json, return_pct, success, tp_price, sl_price,
                   primary_reason, market_regime, confidence
            FROM analysis_episode
            WHERE evaluated_at IS NOT NULL
              AND question NOT LIKE 'actual-trade:%'
              AND id NOT IN (
                  SELECT CAST(value AS INTEGER)
                  FROM gpt_teacher_review, json_each(episode_ids_json)
                  WHERE status = 'completed'
              )
            ORDER BY evaluated_at DESC
            LIMIT ?
            """,
            (max(40, limit * 10),),
        ).fetchall()
    rows = [dict(row) for row in rows]
    failed_limit = (limit + 1) // 2
    successful_limit = limit // 2
    selected = (
        [row for row in rows if not int(row.get("success") or 0)][:failed_limit]
        + [row for row in rows if int(row.get("success") or 0) > 0][:successful_limit]
    )
    selected_ids = {int(row["id"]) for row in selected}
    selected.extend(row for row in rows if int(row["id"]) not in selected_ids)
    return selected[:limit]


def claim_gpt_teacher_review(review_key, episode_ids, teacher_model):
    review_key = str(review_key or "").strip()
    clean_ids = [int(item) for item in episode_ids if int(item) > 0]
    if not review_key or not clean_ids:
        return False
    with _LOCK, _connect() as connection:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO gpt_teacher_review
                (review_key, created_at, teacher_model, episode_ids_json, status)
            VALUES (?, ?, ?, ?, 'pending')
            """,
            (
                review_key,
                time.time(),
                str(teacher_model or "unknown")[:120],
                json.dumps(clean_ids),
            ),
        )
        connection.commit()
        return cursor.rowcount > 0


def complete_gpt_teacher_review(review_key, lesson):
    clean_lesson = re.sub(r"\s+", " ", str(lesson or "")).strip()[:8000]
    if not clean_lesson:
        return False
    with _LOCK, _connect() as connection:
        cursor = connection.execute(
            """
            UPDATE gpt_teacher_review
            SET completed_at = ?, lesson = ?, status = 'completed'
            WHERE review_key = ? AND status = 'pending'
            """,
            (time.time(), clean_lesson, str(review_key or "")),
        )
        connection.commit()
        return cursor.rowcount > 0


def release_gpt_teacher_review(review_key):
    with _LOCK, _connect() as connection:
        connection.execute(
            "DELETE FROM gpt_teacher_review WHERE review_key = ? AND status = 'pending'",
            (str(review_key or ""),),
        )
        connection.commit()


def recent_gpt_teacher_lessons(limit=3):
    limit = max(0, min(10, int(limit)))
    if not limit:
        return []
    with _LOCK, _connect() as connection:
        rows = connection.execute(
            """
            SELECT teacher_model, lesson, completed_at
            FROM gpt_teacher_review
            WHERE status = 'completed' AND lesson IS NOT NULL AND lesson <> ''
            ORDER BY completed_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def gpt_teacher_last_completed_at():
    with _LOCK, _connect() as connection:
        row = connection.execute(
            """
            SELECT MAX(completed_at) AS completed_at
            FROM gpt_teacher_review
            WHERE status = 'completed'
            """
        ).fetchone()
    return float(row["completed_at"] or 0.0)


def record_higher_timeframe_context(context):
    if not isinstance(context, dict) or not context:
        return False
    candle_key = str(context.get("candle_key") or "").strip()
    if not candle_key:
        return False
    with _LOCK, _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO higher_timeframe_observation
                (candle_key, observed_at, context_json)
            VALUES (?, ?, ?)
            ON CONFLICT(candle_key) DO UPDATE SET
                observed_at = excluded.observed_at,
                context_json = excluded.context_json
            WHERE higher_timeframe_observation.context_json <> excluded.context_json
            """,
            (
                candle_key,
                time.time(),
                json.dumps(context, ensure_ascii=False, default=str),
            ),
        )
        connection.commit()
        return cursor.rowcount > 0


def _historical_source_paths():
    ai_dir = Path(ai_data_path("ai_data.csv")).parent
    data_dir = DB_PATH.parent.parent / "data"
    paths = []
    seen = set()
    for path in [
        ai_dir / "ai_data.csv",
        ai_dir / "backtest_ai_data.csv",
        *ai_dir.glob("*learn*.csv"),
        *ai_dir.glob("*ai_data*.csv"),
        *data_dir.glob("*trades.csv"),
    ]:
        if not path.exists() or path in seen:
            continue
        seen.add(path)
        stem = re.sub(r"[^a-zA-Z0-9_]+", "_", path.stem).strip("_").lower()
        if path.parent == ai_dir:
            source = "model_" + stem
        else:
            source = "trade_" + stem
        paths.append((source[:80], path))
    return tuple(paths)


def import_turning_point_history(force=False):
    global _LAST_TURNING_POINTS_MTIME
    path = DB_PATH.parent.parent / "data" / "ethusdt_turning_points" / "all_turning_points.csv"
    if not path.exists():
        return 0
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return 0
    if not force and mtime <= _LAST_TURNING_POINTS_MTIME:
        return 0

    prepared = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = csv.DictReader(handle)
            for row in rows:
                timeframe = str(row.get("timeframe") or "").strip()
                timestamp = str(row.get("time_utc") or "").strip()
                pivot_type = str(row.get("type") or "").strip().lower()
                if not timeframe or not timestamp or pivot_type not in {"high", "low"}:
                    continue
                rsi = _number(row.get("rsi"), 50.0)
                volume_z = _number(row.get("volume_z"), 0.0)
                direction = "short" if pivot_type == "high" else "long"
                market = {
                    "analysis_timeframe": timeframe,
                    "timeframe": timeframe,
                    "timestamp": timestamp,
                    "price": _number(row.get("price")),
                    "rsi": rsi,
                    "rsi_bucket": "overbought" if rsi >= 70 else "oversold" if rsi <= 30 else "neutral",
                    "volume_spike": volume_z >= 2.0,
                    "volume_z": volume_z,
                    "pivot_type": pivot_type,
                    "swing_to_next_pct": _number(row.get("swing_to_next_pct")),
                    "technical_reasons": str(row.get("technical_reasons") or ""),
                    "event_reason": str(row.get("event_reason") or ""),
                }
                canonical = json.dumps(market, ensure_ascii=False, sort_keys=True)
                source_key = hashlib.sha256(
                    f"turning_point:{timeframe}:{timestamp}:{pivot_type}".encode("utf-8")
                ).hexdigest()
                response = (
                    f"{timeframe} 已驗證{('高點反轉' if pivot_type == 'high' else '低點反轉')}；"
                    f"後續方向應為{('做空' if direction == 'short' else '做多')}；"
                    f"因素：{market['technical_reasons']}；"
                    f"至下一轉折幅度 {market['swing_to_next_pct']:+.2f}%。"
                )
                prepared.append(
                    (
                        "turning_point",
                        source_key,
                        time.time(),
                        direction,
                        response,
                        canonical,
                        1,
                    )
                )
    except (OSError, csv.Error):
        return 0

    imported = 0
    if prepared:
        with _LOCK, _connect() as connection:
            before = connection.total_changes
            connection.executemany(
                """
                INSERT OR IGNORE INTO historical_example
                    (source, source_key, imported_at, direction, response, market_json, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                prepared,
            )
            imported = connection.total_changes - before
            connection.commit()
    _LAST_TURNING_POINTS_MTIME = mtime
    return imported


def import_non_turning_contrast_history(force=False):
    global _LAST_CONTRAST_EXAMPLES_MTIME
    path = (
        DB_PATH.parent.parent
        / "data"
        / "ethusdt_turning_points"
        / "non_turning_contrast_examples.csv"
    )
    if not path.exists():
        return 0
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return 0
    if not force and mtime <= _LAST_CONTRAST_EXAMPLES_MTIME:
        return 0

    prepared = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                timeframe = str(row.get("timeframe") or "").strip()
                timestamp = str(row.get("time_utc") or "").strip()
                direction = str(row.get("continuation_direction") or "").strip().lower()
                if not timeframe or not timestamp or direction not in {"long", "short"}:
                    continue
                market = {
                    "analysis_timeframe": timeframe,
                    "timeframe": timeframe,
                    "timestamp": timestamp,
                    "price": _number(row.get("price")),
                    "rsi": _number(row.get("rsi"), 50.0),
                    "rsi_bucket": str(row.get("rsi_bucket") or "neutral"),
                    "volume_spike": str(row.get("volume_spike") or "").lower()
                    in {"1", "true", "yes"},
                    "volume_z": _number(row.get("volume_z")),
                    "ema50_deviation_pct": _number(row.get("ema50_deviation_pct")),
                    "past_return_pct": _number(row.get("past_return_pct")),
                    "future_return_pct": _number(row.get("future_return_pct")),
                    "outcome": "non_reversal_continuation",
                }
                source_key = hashlib.sha256(
                    f"non_turning:{timeframe}:{timestamp}".encode("utf-8")
                ).hexdigest()
                response = (
                    f"{timeframe} 已驗證非變盤對照；RSI={market['rsi']:.1f}"
                    f"（{market['rsi_bucket']}），EMA50乖離"
                    f"{market['ema50_deviation_pct']:+.2f}%，但沒有反轉；"
                    f"前段{market['past_return_pct']:+.2f}%，後續"
                    f"{market['future_return_pct']:+.2f}%，方向延續為"
                    f"{('做多' if direction == 'long' else '做空')}。"
                    "不可只因過熱或超賣就逆勢。"
                )
                prepared.append(
                    (
                        "non_turning_point",
                        source_key,
                        time.time(),
                        direction,
                        response,
                        json.dumps(market, ensure_ascii=False, sort_keys=True),
                        1,
                    )
                )
    except (OSError, csv.Error):
        return 0

    imported = 0
    if prepared:
        with _LOCK, _connect() as connection:
            before = connection.total_changes
            connection.execute(
                "DELETE FROM historical_example WHERE source = 'non_turning_point'"
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO historical_example
                    (source, source_key, imported_at, direction, response, market_json, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                prepared,
            )
            imported = connection.total_changes - before
            connection.commit()
    _LAST_CONTRAST_EXAMPLES_MTIME = mtime
    return imported


def _load_backtest_directions():
    path = DB_PATH.parent.parent / "data" / "backtest_latest_trades.csv"
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return [
            str(row.get("direction") or "setup").strip().lower()
            for row in rows
            if str(row.get("exit_reason") or "").strip().upper() in {"TP", "SL"}
        ]
    except (OSError, csv.Error):
        return []


def _row_direction(row, fallback="setup"):
    direction = str(
        row.get("direction")
        or row.get("side")
        or row.get("signal")
        or row.get("trade_direction")
        or fallback
    ).strip().lower()
    if direction in {"long", "buy", "多", "做多"}:
        return "long"
    if direction in {"short", "sell", "空", "做空"}:
        return "short"
    return fallback


def _row_success(row):
    if "label" in row and str(row.get("label") or "").strip() != "":
        try:
            return 1 if int(float(row.get("label", 0))) > 0 else 0
        except (TypeError, ValueError):
            return None
    exit_reason = str(row.get("exit_reason") or row.get("close_reason") or "").strip().upper()
    if exit_reason == "TP":
        return 1
    if exit_reason == "SL":
        return 0
    for key in ("trade_return_pct", "return_pct", "pnl_pct", "profit_pct", "realized_pnl"):
        if key in row and str(row.get(key) or "").strip() != "":
            try:
                return 1 if float(row.get(key) or 0) > 0 else 0
            except (TypeError, ValueError):
                continue
    result = str(row.get("result") or row.get("success") or "").strip().lower()
    if result in {"1", "true", "win", "wins", "success", "tp", "成功"}:
        return 1
    if result in {"0", "false", "loss", "losses", "fail", "failed", "sl", "失敗"}:
        return 0
    return None


def _market_from_history_row(row):
    market = {}
    for key, value in row.items():
        if value in (None, ""):
            continue
        if key == "label":
            continue
        numeric = _number(value, None)
        market[key] = numeric if numeric is not None else str(value)
    return market


def _history_response(source, direction, success, market):
    result = "成功" if success else "失敗"
    trade_return = market.get("trade_return_pct", market.get("return_pct", market.get("pnl_pct")))
    extra = ""
    if trade_return not in (None, ""):
        extra = f"，報酬 {float(_number(trade_return, 0.0)):+.2f}%"
    direction_text = {"long": "做多", "short": "做空"}.get(direction, "型態")
    return (
        f"既有模型/回測資料 {source}；{direction_text}案例結果為{result}{extra}。"
        "已匯入 MLX 學習庫作為相似型態勝率校準，不代表跳過現行風控。"
    )


def import_existing_model_history(force=False):
    """Import existing classifier/backtest/trade samples as retrieval examples for MLX."""
    global _LAST_HISTORY_IMPORT_TS
    now_ts = time.time()
    if not force and now_ts - _LAST_HISTORY_IMPORT_TS < HISTORY_IMPORT_INTERVAL_SEC:
        return 0

    directions = _load_backtest_directions()
    prepared = []
    for source, path in _historical_source_paths():
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, csv.Error):
            continue

        for index, row in enumerate(rows):
            success = _row_success(row)
            if success is None:
                continue
            market = _market_from_history_row(row)
            if not market:
                continue
            canonical = json.dumps(market, ensure_ascii=False, sort_keys=True)
            source_key = hashlib.sha256(
                f"{source}:{path.name}:{canonical}:{success}".encode("utf-8")
            ).hexdigest()
            direction = (
                directions[index]
                if source in {"model_backtest_ai_data", "backtest_model"} and index < len(directions)
                else _row_direction(row, "setup")
            )
            response = _history_response(source, direction, success, market)
            prepared.append(
                (
                    source,
                    source_key,
                    now_ts,
                    direction,
                    response,
                    canonical,
                    success,
                )
            )

    imported = 0
    if prepared:
        with _LOCK, _connect() as connection:
            before = connection.total_changes
            connection.executemany(
                """
                INSERT OR IGNORE INTO historical_example
                    (source, source_key, imported_at, direction, response, market_json, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                prepared,
            )
            imported = connection.total_changes - before
            connection.commit()
    _LAST_HISTORY_IMPORT_TS = now_ts
    return imported


def _similarity_weight(current, past):
    if not isinstance(current, dict) or not isinstance(past, dict):
        return 0.0
    numeric_weight = 0.0
    numeric_score = 0.0
    for key, current_value in current.items():
        if key not in past:
            continue
        current_number = _number(current_value, None)
        past_number = _number(past.get(key), None)
        if current_number is None or past_number is None:
            continue
        scale = max(abs(current_number), abs(past_number), 1.0)
        distance = min(3.0, abs(current_number - past_number) / scale)
        weight = 1.0 / (1.0 + distance)
        numeric_score += weight
        numeric_weight += 1.0

    categorical_hits = 0.0
    categorical_total = 0.0
    for key in (
        "direction",
        "htf",
        "mid_trend",
        "macd",
        "breakout",
        "regime",
        "triangle",
        "event_risk",
        "volume_spike",
        "analysis_timeframe",
        "timeframe",
        "rsi_bucket",
    ):
        if key not in current or key not in past:
            continue
        categorical_total += 1.0
        if str(current.get(key)).strip().lower() == str(past.get(key)).strip().lower():
            categorical_hits += 1.0

    if numeric_weight <= 0 and categorical_total <= 0:
        return 0.0
    numeric_part = numeric_score / numeric_weight if numeric_weight else 0.0
    categorical_part = categorical_hits / categorical_total if categorical_total else 0.0
    if numeric_weight and categorical_total:
        return 0.78 * numeric_part + 0.22 * categorical_part
    return numeric_part or categorical_part


def _candidate_rows_for_replacement(limit):
    with _LOCK, _connect() as connection:
        history_rows = connection.execute(
            """
            SELECT source, direction, market_json, success, NULL AS return_pct, imported_at AS ts
            FROM historical_example
            WHERE source NOT IN ('turning_point', 'non_turning_point')
            ORDER BY id DESC
            LIMIT ?
            """,
            (max(100, int(limit)),),
        ).fetchall()
        analysis_rows = connection.execute(
            """
            SELECT 'analysis_episode' AS source, direction, market_json, success, return_pct, evaluated_at AS ts
            FROM analysis_episode
            WHERE evaluated_at IS NOT NULL
            ORDER BY evaluated_at DESC
            LIMIT ?
            """,
            (max(50, int(limit // 4)),),
        ).fetchall()
    return [*history_rows, *analysis_rows]


def predict_replacement_probability(features, direction="setup", limit=900):
    """Low-latency MLX learning-store replacement for the old sklearn probability."""
    import_existing_model_history()
    current = dict(features or {})
    clean_direction = _row_direction({"direction": direction}, "setup")
    current["direction"] = clean_direction

    scored = []
    source_summary = {}
    for row in _candidate_rows_for_replacement(limit):
        if clean_direction in {"long", "short"} and str(row["direction"] or "").lower() not in {
            clean_direction,
            "setup",
        }:
            continue
        past_market = _json_loads_safe(row["market_json"], {})
        if not isinstance(past_market, dict):
            continue
        past_market.setdefault("direction", str(row["direction"] or "setup").lower())
        weight = _similarity_weight(current, past_market)
        if weight <= 0:
            continue
        source = str(row["source"] or "unknown")
        if source.startswith("trade_"):
            weight *= 1.25
        elif source == "analysis_episode":
            weight *= 1.15
        scored.append((weight, int(row["success"] or 0), source))

    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[: max(25, min(250, int(limit // 3)))]
    prior_weight = 12.0
    weighted_success = prior_weight * 0.5
    total_weight = prior_weight
    for weight, success, source in top:
        weighted_success += weight * success
        total_weight += weight
        item = source_summary.setdefault(source, {"samples": 0, "weight": 0.0, "wins": 0.0})
        item["samples"] += 1
        item["weight"] += weight
        item["wins"] += weight * success

    probability = weighted_success / total_weight if total_weight > 0 else 0.5
    confidence = min(1.0, max(0.0, (total_weight - prior_weight) / 80.0))
    if confidence < 0.25:
        probability = 0.5 + (probability - 0.5) * (confidence / 0.25)
    sources = []
    for source, item in source_summary.items():
        source_weight = item["weight"]
        sources.append(
            {
                "source": source,
                "samples": int(item["samples"]),
                "weight": round(source_weight, 4),
                "winrate": (item["wins"] / source_weight * 100) if source_weight else 0.0,
            }
        )
    sources.sort(key=lambda item: item["weight"], reverse=True)
    return {
        "probability": max(0.05, min(0.95, float(probability))),
        "sample_count": len(top),
        "effective_weight": max(0.0, total_weight - prior_weight),
        "confidence": confidence,
        "sources": sources[:8],
    }


def _number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None if default is None else float(default)


def _extract_direction(response):
    text = str(response or "")
    recommendation = re.search(
        r"(?:建議|結論|方向)[：:\s]*(?:偏向|考慮)?\s*(做多|做空|觀望)",
        text,
        flags=re.IGNORECASE,
    )
    if recommendation:
        return {"做多": "long", "做空": "short", "觀望": "neutral"}[recommendation.group(1)]
    if "觀望" in text:
        return "neutral"
    has_long = "做多" in text or "偏多" in text
    has_short = "做空" in text or "偏空" in text
    if has_long != has_short:
        return "long" if has_long else "short"
    return "neutral"


def _json_loads_safe(value, default=None):
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {} if default is None else default


def _extract_structured_payload(response):
    text = str(response or "")
    payload = {}
    block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = []
    if block:
        candidates.append(block.group(1))
    first = text.find("{")
    last = text.rfind("}")
    if 0 <= first < last:
        candidates.append(text[first : last + 1])
    for candidate in candidates:
        try:
            loaded = json.loads(candidate)
            if isinstance(loaded, dict):
                payload = loaded
                break
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

    if not payload:
        reason_match = re.search(r"(?:主因|primary_reason)[：:\s]+([^\n；;]+)", text)
        confidence_match = re.search(r"(?:信心|confidence)[：:\s]+([0-9]+(?:\.[0-9]+)?)", text)
        if reason_match:
            payload["primary_reason"] = reason_match.group(1).strip()
        if confidence_match:
            confidence = _number(confidence_match.group(1), 0.0)
            payload["confidence"] = confidence / 100 if confidence > 1 else confidence

    reason = str(payload.get("primary_reason") or payload.get("主因") or "").strip()
    confidence = _number(payload.get("confidence", payload.get("信心", 0.0)), 0.0)
    if confidence > 1:
        confidence /= 100
    confidence = max(0.0, min(1.0, confidence))
    if not reason:
        text_lower = text.lower()
        for candidate in ("支撐壓力", "震盪", "突破", "趨勢", "新聞", "量能"):
            if candidate in text_lower or candidate in text:
                reason = candidate
                break
    return payload, reason[:64], confidence


def _classify_market_regime(market):
    regime = str(market.get("regime") or "").strip()
    breakout = int(_number(market.get("breakout"), 0))
    macro = _number(market.get("macro"), _number(market.get("macro_bias"), 0.0))
    volume_spike = bool(market.get("volume_spike"))
    htf = int(_number(market.get("htf"), _number(market.get("four_hour_trend"), 0)))
    one_hour = int(_number(market.get("one_hour_trend"), 0))
    daily = int(_number(market.get("daily_trend"), 0))
    weekly = int(_number(market.get("weekly_trend"), 0))
    if abs(macro) >= 0.55:
        return "news_driven"
    if breakout and volume_spike:
        return "breakout"
    if breakout and not volume_spike:
        return "fake_breakout_risk"
    if htf and daily and htf != daily:
        return "high_tf_conflict"
    if one_hour and htf and daily and len({one_hour, htf, daily}) == 1:
        return "trend"
    if regime:
        return regime
    if weekly and daily and weekly != daily:
        return "higher_tf_transition"
    return "range"


def _build_factor_tags(market, direction, structured_payload=None):
    direction_sign = 1 if direction == "long" else -1 if direction == "short" else 0
    tags = []

    def add_directional(name, value, threshold=0.0):
        signed = _number(value, 0.0) * direction_sign
        if direction_sign == 0:
            return
        if signed > threshold:
            tags.append(f"{name}:同向")
        elif signed < -threshold:
            tags.append(f"{name}:逆向")
        else:
            tags.append(f"{name}:中性")

    add_directional("30m_MACD", market.get("mid_trend"), 0.0)
    add_directional("4H趨勢", market.get("htf", market.get("four_hour_trend")), 0.0)
    add_directional("1H趨勢", market.get("one_hour_trend"), 0.0)
    add_directional("日線趨勢", market.get("daily_trend"), 0.0)
    add_directional("週線趨勢", market.get("weekly_trend"), 0.0)
    add_directional("支撐壓力", market.get("sr_bias"), 0.10)
    add_directional("新聞宏觀", market.get("macro", market.get("macro_bias")), 0.20)
    add_directional("衍生品", market.get("derivatives_pressure"), 0.10)

    rsi_bucket = str(market.get("rsi_bucket") or "").strip()
    if rsi_bucket:
        tags.append(f"RSI:{rsi_bucket}")
    if bool(market.get("volume_spike")):
        tags.append("量能:放大")
    if _number(market.get("breakout"), 0.0) != 0:
        tags.append("突破:存在")
    for key, label in (
        ("fifteen_min_range_pos", "15m區間"),
        ("one_hour_range_pos", "1H區間"),
        ("four_hour_range_pos", "4H區間"),
        ("daily_range_pos", "日線區間"),
        ("weekly_range_pos", "週線區間"),
    ):
        pos = _number(market.get(key), -1.0)
        if 0 <= pos <= 0.25:
            tags.append(f"{label}:近下緣")
        elif pos >= 0.75:
            tags.append(f"{label}:近上緣")

    if isinstance(structured_payload, dict):
        raw_factors = structured_payload.get("factors") or structured_payload.get("因素")
        if isinstance(raw_factors, list):
            for factor in raw_factors[:8]:
                clean = str(factor).strip()
                if clean:
                    tags.append(f"MLX:{clean[:32]}")
    return list(dict.fromkeys(tags))


def _grade_shadow_order(market, order, factors):
    direction = order.get("direction")
    entry = _number(order.get("entry_price"))
    tp = _number(order.get("tp_price"))
    sl = _number(order.get("sl_price"))
    if entry <= 0 or tp <= 0 or sl <= 0:
        return "C"
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    rr = reward / max(risk, 1e-9)
    aligned = sum(1 for factor in factors if factor.endswith(":同向"))
    reversed_count = sum(1 for factor in factors if factor.endswith(":逆向"))
    range_edge_ok = (
        any("近下緣" in factor for factor in factors)
        if direction == "long"
        else any("近上緣" in factor for factor in factors)
    )
    if rr >= 1.6 and aligned >= 3 and reversed_count <= 1 and range_edge_ok:
        return "A"
    if rr >= 1.2 and aligned >= 2 and reversed_count <= 2:
        return "B"
    return "C"


def build_trade_factor_tags(market, direction, structured_payload=None):
    """Public wrapper used by backtests and live records for MLX-style factor tags."""
    market = market if isinstance(market, dict) else {}
    payload = structured_payload if isinstance(structured_payload, dict) else {}
    return _build_factor_tags(market, direction, payload)


def classify_trade_market_regime(market):
    market = market if isinstance(market, dict) else {}
    return _classify_market_regime(market)


def _extract_shadow_orders(response):
    orders = []
    text = str(response or "")
    pattern = re.compile(
        r"影子(?:開單|訂單)\s*\d*\s*[：:]\s*(做多|做空)"
        r"[^\n]*?進場\s*[=：:]\s*([0-9]+(?:\.[0-9]+)?)"
        r"[^\n]*?TP\s*[=：:]\s*([0-9]+(?:\.[0-9]+)?)"
        r"[^\n]*?SL\s*[=：:]\s*([0-9]+(?:\.[0-9]+)?)",
        flags=re.IGNORECASE,
    )
    parsed = pattern.findall(text)
    if not parsed:
        range_plans = {
            {"做多": "long", "做空": "short"}[direction_text]: (
                _number(entry_text),
                _number(tp_text),
                _number(sl_text),
            )
            for direction_text, entry_text, tp_text, sl_text in re.findall(
                r"震盪(做多|做空)[^\n]*?進場\s*[=：:]?\s*([0-9]+(?:\.[0-9]+)?)"
                r"[^\n]*?TP\s*[=：:]?\s*([0-9]+(?:\.[0-9]+)?)"
                r"[^\n]*?SL\s*[=：:]?\s*([0-9]+(?:\.[0-9]+)?)",
                text,
                flags=re.IGNORECASE,
            )
        }
        parsed = []
        for direction_text in re.findall(
            r"影子(?:開單|訂單)\s*\d*\s*[：:]\s*(做多|做空)",
            text,
            flags=re.IGNORECASE,
        ):
            direction = {"做多": "long", "做空": "short"}[direction_text]
            if direction in range_plans:
                parsed.append((direction_text, *range_plans[direction]))
    for direction_text, entry_text, tp_text, sl_text in parsed:
        direction = {"做多": "long", "做空": "short"}[direction_text]
        entry_price = _number(entry_text)
        tp_price = _number(tp_text)
        sl_price = _number(sl_text)
        valid = (
            sl_price < entry_price < tp_price
            if direction == "long"
            else tp_price < entry_price < sl_price
        )
        if valid:
            orders.append(
                {
                    "direction": direction,
                    "entry_price": entry_price,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                }
            )
    return orders


def record_analysis(question, response, market):
    market = market if isinstance(market, dict) else {}
    price = _number(market.get("price"))
    if price <= 0 or not response:
        return None
    clean_question = str(question or "")[:2000]
    clean_response = str(response)[:8000]
    shadow_orders = (
        _extract_shadow_orders(clean_response)
        if clean_question.startswith("auto-shadow:")
        else []
    )
    if clean_question.startswith("auto-shadow:") and not shadow_orders:
        return 0
    structured_payload, primary_reason, confidence = _extract_structured_payload(
        clean_response
    )
    market_regime = _classify_market_regime(market)
    orders = shadow_orders or [
        {
            "direction": _extract_direction(clean_response),
            "entry_price": price,
            "tp_price": None,
            "sl_price": None,
        }
    ]
    with _LOCK, _connect() as connection:
        created_at = time.time()
        market_json = json.dumps(market, ensure_ascii=False, default=str)
        cursor = connection.executemany(
            """
            INSERT INTO analysis_episode
                (
                    created_at, entry_price, direction, question, response, market_json,
                    tp_price, sl_price, activated_at, structured_json, primary_reason,
                    confidence, market_regime, shadow_grade, factor_json, strategy_version
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                _analysis_insert_row(
                    created_at,
                    clean_question,
                    clean_response,
                    market_json,
                    market,
                    structured_payload,
                    primary_reason,
                    confidence,
                    market_regime,
                    order,
                    index,
                    len(orders),
                    price,
                )
                for index, order in enumerate(orders, start=1)
            ],
        )
        connection.commit()
        return cursor.rowcount


def record_actual_trade_open(
    direction,
    entry_price,
    tp_price,
    sl_price,
    market,
    reason_text="",
    opened_at=None,
    source="live_trade",
):
    market = market if isinstance(market, dict) else {}
    direction = str(direction or "").strip().lower()
    if direction not in {"long", "short"}:
        return None
    entry = _number(entry_price)
    tp = _number(tp_price)
    sl = _number(sl_price)
    if entry <= 0:
        return None
    opened_at = float(opened_at or time.time())
    market = dict(market)
    market["price"] = entry
    market["actual_trade"] = True
    market["actual_trade_source"] = str(source or "live_trade")
    direction_text = "做多" if direction == "long" else "做空"
    payload = {
        "direction": direction_text,
        "primary_reason": str(market.get("primary_reason") or "實單策略觸發"),
        "confidence": _number(market.get("ai_prob"), _number(market.get("confidence"), 0.0)),
        "market_regime": _classify_market_regime(market),
        "entry_zone": [entry, entry],
        "tp": tp,
        "sl": sl,
        "factors": _build_factor_tags(market, direction, {}),
        "source": source,
    }
    response = (
        "```json\n"
        + json.dumps(payload, ensure_ascii=False, default=str)
        + "\n```\n"
        + f"實際開單：{direction_text}；進場={entry:.4f}；TP={tp:.4f}；SL={sl:.4f}；"
        + f"依據={str(reason_text or '策略觸發')[:1000]}"
    )
    structured_payload, primary_reason, confidence = _extract_structured_payload(response)
    market_regime = _classify_market_regime(market)
    order = {
        "direction": direction,
        "entry_price": entry,
        "tp_price": tp if tp > 0 else None,
        "sl_price": sl if sl > 0 else None,
    }
    with _LOCK, _connect() as connection:
        market_json = json.dumps(market, ensure_ascii=False, default=str)
        row = _analysis_insert_row(
            opened_at,
            f"actual-trade:{source}:{int(opened_at)}",
            response,
            market_json,
            market,
            structured_payload,
            primary_reason,
            confidence,
            market_regime,
            order,
            1,
            1,
            entry,
        )
        cursor = connection.execute(
            """
            INSERT INTO analysis_episode
                (
                    created_at, entry_price, direction, question, response, market_json,
                    tp_price, sl_price, activated_at, structured_json, primary_reason,
                    confidence, market_regime, shadow_grade, factor_json, strategy_version
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )
        connection.commit()
        return int(cursor.lastrowid)


def update_actual_trade_outcome(episode_id, close_price, success, closed_at=None):
    episode_id = int(_number(episode_id, 0))
    close_price = _number(close_price)
    if episode_id <= 0 or close_price <= 0:
        return False
    closed_at = float(closed_at or time.time())
    with _LOCK, _connect() as connection:
        row = connection.execute(
            "SELECT entry_price FROM analysis_episode WHERE id = ?",
            (episode_id,),
        ).fetchone()
        if not row:
            return False
        return_pct = (close_price - _number(row["entry_price"])) / max(
            _number(row["entry_price"]), 1e-9
        ) * 100
        connection.execute(
            """
            UPDATE analysis_episode
            SET evaluated_at = ?, exit_price = ?, return_pct = ?, success = ?
            WHERE id = ?
            """,
            (closed_at, close_price, return_pct, 1 if int(success) > 0 else 0, episode_id),
        )
        connection.commit()
        return True


def backfill_analysis_metadata(limit=2000, force=False):
    global _LAST_METADATA_BACKFILL_TS
    now_ts = time.time()
    if not force and now_ts - _LAST_METADATA_BACKFILL_TS < 300:
        return 0
    _LAST_METADATA_BACKFILL_TS = now_ts
    with _LOCK, _connect() as connection:
        connection.execute(
            """
            UPDATE analysis_episode
            SET strategy_version = 'pre-version-tracking'
            WHERE strategy_version IS NULL
               OR strategy_version = ''
               OR strategy_version = 'unknown'
            """
        )
        connection.execute(
            """
            UPDATE strategy_outcome
            SET strategy_version = 'pre-version-tracking'
            WHERE strategy_version IS NULL
               OR strategy_version = ''
               OR strategy_version = 'unknown'
            """
        )
        rows = connection.execute(
            """
            SELECT id, direction, question, response, market_json, entry_price, tp_price, sl_price
            FROM analysis_episode
            WHERE factor_json IS NULL
               OR factor_json = ''
               OR market_regime IS NULL
               OR market_regime = ''
               OR primary_reason IS NULL
               OR primary_reason = ''
               OR strategy_version IS NULL
               OR strategy_version = ''
               OR (question LIKE 'auto-shadow:%' AND (shadow_grade IS NULL OR shadow_grade = ''))
            ORDER BY id DESC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        ).fetchall()
        updated = 0
        for row in rows:
            market = _json_loads_safe(row["market_json"], {})
            if not isinstance(market, dict):
                market = {}
            payload, reason, confidence = _extract_structured_payload(row["response"])
            regime = _classify_market_regime(market)
            factors = _build_factor_tags(market, row["direction"], payload)
            shadow_grade = ""
            if str(row["question"] or "").startswith("auto-shadow:"):
                shadow_grade = _grade_shadow_order(
                    market,
                    {
                        "direction": row["direction"],
                        "entry_price": row["entry_price"],
                        "tp_price": row["tp_price"],
                        "sl_price": row["sl_price"],
                    },
                    factors,
                )
            connection.execute(
                """
                UPDATE analysis_episode
                SET structured_json = COALESCE(NULLIF(structured_json, ''), ?),
                    primary_reason = COALESCE(NULLIF(primary_reason, ''), ?),
                    confidence = COALESCE(confidence, ?),
                    market_regime = COALESCE(NULLIF(market_regime, ''), ?),
                    shadow_grade = COALESCE(NULLIF(shadow_grade, ''), ?),
                    factor_json = COALESCE(NULLIF(factor_json, ''), ?),
                    strategy_version = COALESCE(NULLIF(strategy_version, ''), ?)
                WHERE id = ?
                """,
                (
                    json.dumps(payload, ensure_ascii=False, default=str),
                    reason,
                    confidence,
                    regime,
                    shadow_grade,
                    json.dumps(factors, ensure_ascii=False),
                    str(market.get("strategy_version") or "pre-version-tracking"),
                    row["id"],
                ),
            )
            updated += 1
        connection.commit()
        return updated


def _analysis_insert_row(
    created_at,
    clean_question,
    clean_response,
    market_json,
    market,
    structured_payload,
    primary_reason,
    confidence,
    market_regime,
    order,
    index,
    order_count,
    price,
):
    factors = _build_factor_tags(market, order["direction"], structured_payload)
    grade = (
        _grade_shadow_order(market, order, factors)
        if clean_question.startswith("auto-shadow:")
        else ""
    )
    return (
        created_at,
        order["entry_price"],
        order["direction"],
        f"{clean_question}:order:{index}" if order_count > 1 else clean_question,
        clean_response,
        market_json,
        order["tp_price"],
        order["sl_price"],
        created_at if abs(order["entry_price"] - price) / price <= 0.0005 else None,
        json.dumps(structured_payload, ensure_ascii=False, default=str),
        primary_reason,
        confidence,
        market_regime,
        grade,
        json.dumps(factors, ensure_ascii=False),
        str(market.get("strategy_version") or ""),
    )


def evaluate_pending(current_price, now_ts=None):
    global _LAST_EVALUATION_TS
    price = _number(current_price)
    now_ts = float(now_ts or time.time())
    if price <= 0 or now_ts - _LAST_EVALUATION_TS < EVALUATION_INTERVAL_SEC:
        return 0
    _LAST_EVALUATION_TS = now_ts
    cutoff = now_ts - EVALUATION_HOURS * 3600

    with _LOCK, _connect() as connection:
        rows = connection.execute(
            """
            SELECT
                id, created_at, entry_price, direction, tp_price, sl_price,
                activated_at
            FROM analysis_episode
            WHERE evaluated_at IS NULL
            """
        ).fetchall()
        evaluated = 0
        for row in rows:
            move_pct = (price - row["entry_price"]) / row["entry_price"] * 100
            if row["tp_price"] is not None and row["sl_price"] is not None:
                if row["activated_at"] is None:
                    entry_reached = (
                        price <= row["entry_price"]
                        if row["direction"] == "long"
                        else price >= row["entry_price"]
                    )
                    if not entry_reached:
                        continue
                    connection.execute(
                        "UPDATE analysis_episode SET activated_at = ? WHERE id = ?",
                        (now_ts, row["id"]),
                    )
                if row["direction"] == "long" and price >= row["tp_price"]:
                    success = True
                elif row["direction"] == "long" and price <= row["sl_price"]:
                    success = False
                elif row["direction"] == "short" and price <= row["tp_price"]:
                    success = True
                elif row["direction"] == "short" and price >= row["sl_price"]:
                    success = False
                else:
                    continue
            else:
                if row["created_at"] > cutoff:
                    continue
                if row["direction"] == "long":
                    success = move_pct >= MIN_MOVE_PCT
                elif row["direction"] == "short":
                    success = move_pct <= -MIN_MOVE_PCT
                else:
                    success = abs(move_pct) < MIN_MOVE_PCT
            connection.execute(
                """
                UPDATE analysis_episode
                SET evaluated_at = ?, exit_price = ?, return_pct = ?, success = ?
                WHERE id = ?
                """,
                (now_ts, price, move_pct, int(success), row["id"]),
            )
            evaluated += 1
        connection.commit()
        return evaluated


def build_learning_context(market, limit=None):
    market = market if isinstance(market, dict) else {}
    import_existing_model_history()
    import_turning_point_history()
    import_non_turning_contrast_history()
    limit = LEARNING_CONTEXT_LIMIT if limit is None else max(0, int(limit))
    with _LOCK, _connect() as connection:
        analysis_rows = connection.execute(
            """
            SELECT direction, response, market_json, return_pct, success, 'analysis' AS source
            FROM analysis_episode
            WHERE evaluated_at IS NOT NULL
            ORDER BY success DESC, evaluated_at DESC
            LIMIT 100
            """
        ).fetchall()
        historical_rows = list(connection.execute(
            """
            SELECT direction, response, market_json, NULL AS return_pct, success, source
            FROM historical_example
            WHERE source NOT IN ('turning_point', 'non_turning_point')
            ORDER BY id DESC
            LIMIT 500
            """
        ).fetchall())
        for timeframe, row_limit in (
            ("15m", 150),
            ("1h", 150),
            ("4h", 150),
            ("1d", 100),
            ("1w", 50),
            ("1M", 20),
        ):
            historical_rows.extend(
                connection.execute(
                    """
                    SELECT direction, response, market_json, NULL AS return_pct, success, source
                    FROM historical_example
                    WHERE source = 'turning_point'
                      AND json_extract(market_json, '$.timeframe') = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (timeframe, row_limit),
                ).fetchall()
            )
            historical_rows.extend(
                connection.execute(
                    """
                    SELECT direction, response, market_json, NULL AS return_pct, success, source
                    FROM historical_example
                    WHERE source = 'non_turning_point'
                      AND json_extract(market_json, '$.timeframe') = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (timeframe, row_limit),
                ).fetchall()
            )

    scored = []
    for row in [*analysis_rows, *historical_rows]:
        try:
            past_market = json.loads(row["market_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            past_market = {}
        similarity = 0
        for key in (
            "htf",
            "daily_trend",
            "weekly_trend",
            "regime",
            "breakout",
            "triangle",
            "macro",
            "volume_spike",
            "analysis_timeframe",
            "rsi_bucket",
        ):
            current_value = market.get(key)
            if current_value is not None and str(current_value) == str(past_market.get(key)):
                similarity += 1
        scored.append((similarity, int(row["success"]), row))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    turning_scored = [item for item in scored if item[2]["source"] == "turning_point"]
    contrast_scored = [item for item in scored if item[2]["source"] == "non_turning_point"]
    other_scored = [
        item
        for item in scored
        if item[2]["source"] not in {"turning_point", "non_turning_point"}
    ]
    turning_limit = min(len(turning_scored), max(1, limit // 3)) if limit else 0
    contrast_limit = min(len(contrast_scored), max(1, limit // 3)) if limit else 0
    selected = (
        turning_scored[:turning_limit]
        + contrast_scored[:contrast_limit]
        + other_scored[: max(0, limit - turning_limit - contrast_limit)]
    )
    if len(selected) < limit:
        selected_ids = {id(item[2]) for item in selected}
        selected.extend(
            item for item in scored if id(item[2]) not in selected_ids
        )
        selected = selected[:limit]
    examples = []
    for _, _, row in selected:
        result = "成功" if row["success"] else "失敗"
        response = re.sub(r"\s+", " ", str(row["response"]))[:500]
        if row["return_pct"] is None:
            examples.append(
                f"- {row['direction']}｜{result}｜來源 {row['source']}｜案例：{response}"
            )
        else:
            examples.append(
                f"- {row['direction']}｜{result}｜後續漲跌 {row['return_pct']:+.2f}%"
                f"｜當時分析：{response}"
            )
    sections = []
    teacher_lessons = recent_gpt_teacher_lessons(
        max(0, int(os.getenv("MLX_GPT_TEACHER_CONTEXT_LIMIT", "3") or "3"))
    )
    if teacher_lessons:
        sections.append(
            "【GPT教師校正】\n"
            "以下是從已有真實結果的 MLX 案例歸納出的分析規則；只用來改善推理，"
            "不得取代目前行情、風控或直接觸發實單：\n"
            + "\n".join(
                f"- {' '.join(str(row['lesson']).split())[:1800]}"
                for row in teacher_lessons
            )
        )
    if examples:
        sections.append(
            "【過去已驗證經驗】\n"
            "以下案例只作校準，不可取代目前市場資料；應避免重複失敗案例的推理：\n"
            + "\n".join(examples)
        )
    return "\n\n".join(sections)


def learning_stats():
    import_existing_model_history()
    import_turning_point_history()
    import_non_turning_contrast_history()
    backfill_analysis_metadata()
    with _LOCK, _connect() as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN evaluated_at IS NOT NULL THEN 1 ELSE 0 END) AS evaluated,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS successful
            FROM analysis_episode
            """
        ).fetchone()
        historical = connection.execute(
            """
            SELECT COUNT(*) AS total, SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS successful
            FROM historical_example
            """
        ).fetchone()
        turning_points = connection.execute(
            "SELECT COUNT(*) AS total FROM historical_example WHERE source = 'turning_point'"
        ).fetchone()
        contrast_examples = connection.execute(
            "SELECT COUNT(*) AS total FROM historical_example WHERE source = 'non_turning_point'"
        ).fetchone()
        higher_tf_observations = connection.execute(
            "SELECT COUNT(*) AS total FROM higher_timeframe_observation"
        ).fetchone()
        auto_analyses = connection.execute(
            "SELECT COUNT(*) AS total FROM auto_analysis_log"
        ).fetchone()
        structured = connection.execute(
            """
            SELECT
                SUM(CASE WHEN structured_json IS NOT NULL AND structured_json <> '{}' THEN 1 ELSE 0 END) AS total
            FROM analysis_episode
            """
        ).fetchone()
        sl_reviews = connection.execute(
            "SELECT COUNT(*) AS total FROM sl_review_event"
        ).fetchone()
        gpt_teacher = connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                MAX(completed_at) AS last_completed_at
            FROM gpt_teacher_review
            WHERE status = 'completed'
            """
        ).fetchone()
        actual_trades = connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN evaluated_at IS NOT NULL THEN 1 ELSE 0 END) AS evaluated,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS successful
            FROM analysis_episode
            WHERE question LIKE 'actual-trade:%'
            """
        ).fetchone()
        version_rows = connection.execute(
            """
            SELECT
                COALESCE(NULLIF(strategy_version, ''), 'pre-version-tracking') AS name,
                COUNT(*) AS total,
                SUM(CASE WHEN evaluated_at IS NOT NULL THEN 1 ELSE 0 END) AS evaluated,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS wins
            FROM analysis_episode
            GROUP BY COALESCE(NULLIF(strategy_version, ''), 'pre-version-tracking')
            ORDER BY total DESC
            LIMIT 8
            """
        ).fetchall()
    imported = int(historical["total"] or 0)
    imported_successful = int(historical["successful"] or 0)
    total = int(row["total"] or 0)
    evaluated = int(row["evaluated"] or 0)
    successful = int(row["successful"] or 0)
    accuracy = successful / evaluated * 100 if evaluated else 0.0
    return {
        "total": total,
        "evaluated": evaluated,
        "successful": successful,
        "accuracy": accuracy,
        "evaluation_hours": EVALUATION_HOURS,
        "imported": imported,
        "imported_successful": imported_successful,
        "context_total": total + imported,
        "turning_points": int(turning_points["total"] or 0),
        "contrast_examples": int(contrast_examples["total"] or 0),
        "higher_tf_observations": int(higher_tf_observations["total"] or 0),
        "auto_analyses": int(auto_analyses["total"] or 0),
        "structured_analyses": int(structured["total"] or 0),
        "sl_reviews": int(sl_reviews["total"] or 0),
        "gpt_teacher_lessons": int(gpt_teacher["total"] or 0),
        "gpt_teacher_last_completed_at": float(gpt_teacher["last_completed_at"] or 0.0),
        "actual_trades": int(actual_trades["total"] or 0),
        "actual_trades_evaluated": int(actual_trades["evaluated"] or 0),
        "actual_trades_successful": int(actual_trades["successful"] or 0),
        "strategy_versions": _version_rows_to_summary(version_rows),
        "top_factors": factor_performance(limit=5),
        "primary_reasons": primary_reason_stats(limit=5),
        "shadow_grades": shadow_grade_stats(),
        "market_regimes": market_regime_stats(limit=5),
    }


def _version_rows_to_summary(rows):
    result = []
    for row in rows:
        total = int(row["total"] or 0)
        evaluated = int(row["evaluated"] or 0)
        wins = int(row["wins"] or 0)
        result.append(
            {
                "name": str(row["name"] or "pre-version-tracking"),
                "total": total,
                "evaluated": evaluated,
                "wins": wins,
                "losses": max(0, evaluated - wins),
                "winrate": wins / evaluated * 100 if evaluated else 0.0,
            }
        )
    return result


def _rate_summary(rows, key_name="name", limit=8):
    summary = {}
    for row in rows:
        key = str(row[key_name] or "").strip()
        if not key:
            continue
        item = summary.setdefault(key, {"name": key, "total": 0, "wins": 0})
        item["total"] += 1
        item["wins"] += 1 if int(row["success"] or 0) > 0 else 0
    result = []
    for item in summary.values():
        total = item["total"]
        wins = item["wins"]
        item["losses"] = total - wins
        item["winrate"] = wins / total * 100 if total else 0.0
        result.append(item)
    result.sort(key=lambda item: (item["total"], item["winrate"]), reverse=True)
    return result[:limit]


def factor_performance(limit=8, min_samples=1):
    with _LOCK, _connect() as connection:
        rows = connection.execute(
            """
            SELECT factor_json, success
            FROM analysis_episode
            WHERE evaluated_at IS NOT NULL
              AND factor_json IS NOT NULL
              AND factor_json <> ''
            ORDER BY evaluated_at DESC
            LIMIT 2000
            """
        ).fetchall()
    expanded = []
    for row in rows:
        factors = _json_loads_safe(row["factor_json"], [])
        if not isinstance(factors, list):
            continue
        for factor in factors:
            expanded.append({"name": str(factor), "success": row["success"]})
    stats = _rate_summary(expanded, limit=max(limit * 3, limit))
    return [item for item in stats if item["total"] >= min_samples][:limit]


def primary_reason_stats(limit=6):
    with _LOCK, _connect() as connection:
        rows = connection.execute(
            """
            SELECT primary_reason AS name, success
            FROM analysis_episode
            WHERE evaluated_at IS NOT NULL
              AND primary_reason IS NOT NULL
              AND primary_reason <> ''
            ORDER BY evaluated_at DESC
            LIMIT 1000
            """
        ).fetchall()
    return _rate_summary(rows, limit=limit)


def market_regime_stats(limit=6):
    with _LOCK, _connect() as connection:
        rows = connection.execute(
            """
            SELECT market_regime AS name, success
            FROM analysis_episode
            WHERE evaluated_at IS NOT NULL
              AND market_regime IS NOT NULL
              AND market_regime <> ''
            ORDER BY evaluated_at DESC
            LIMIT 1000
            """
        ).fetchall()
    return _rate_summary(rows, limit=limit)


def shadow_grade_stats():
    with _LOCK, _connect() as connection:
        rows = connection.execute(
            """
            SELECT shadow_grade AS name, success
            FROM analysis_episode
            WHERE evaluated_at IS NOT NULL
              AND shadow_grade IS NOT NULL
              AND shadow_grade <> ''
            ORDER BY evaluated_at DESC
            LIMIT 1000
            """
        ).fetchall()
    return _rate_summary(rows, limit=3)


def record_sl_review(review):
    if not isinstance(review, dict) or not review:
        return False
    direction = str(review.get("direction") or "unknown")[:16]
    verdict = str(review.get("verdict") or "")[:128]
    issues = review.get("issues")
    if not isinstance(issues, list):
        issues = []
    with _LOCK, _connect() as connection:
        connection.execute(
            """
            INSERT INTO sl_review_event
                (created_at, direction, verdict, issue_json, review_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                float(review.get("ts") or time.time()),
                direction,
                verdict,
                json.dumps(issues, ensure_ascii=False),
                json.dumps(review, ensure_ascii=False, default=str),
            ),
        )
        connection.commit()
    return True


def sl_review_summary(limit=5):
    with _LOCK, _connect() as connection:
        total = connection.execute(
            "SELECT COUNT(*) AS total FROM sl_review_event"
        ).fetchone()
        rows = connection.execute(
            """
            SELECT verdict, issue_json
            FROM sl_review_event
            ORDER BY created_at DESC
            LIMIT 500
            """
        ).fetchall()
    issue_counts = {}
    revalidation = 0
    for row in rows:
        if "需重新確認" in str(row["verdict"] or ""):
            revalidation += 1
        issues = _json_loads_safe(row["issue_json"], [])
        if not isinstance(issues, list):
            continue
        for issue in issues:
            clean = str(issue).strip()
            if clean:
                issue_counts[clean] = issue_counts.get(clean, 0) + 1
    top_issues = sorted(issue_counts.items(), key=lambda item: item[1], reverse=True)[:limit]
    return {
        "total": int(total["total"] or 0),
        "recent_checked": len(rows),
        "revalidation": revalidation,
        "top_issues": top_issues,
    }


def record_strategy_outcome(
    result,
    close_reason="",
    close_price=0.0,
    closed_at=None,
    strategy_version="",
):
    clean_result = 1 if int(result) > 0 else 0
    with _LOCK, _connect() as connection:
        connection.execute(
            """
            INSERT INTO strategy_outcome
                (closed_at, result, close_reason, close_price, strategy_version)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                float(closed_at or time.time()),
                clean_result,
                str(close_reason or "UNKNOWN").upper()[:32],
                _number(close_price),
                str(strategy_version or ""),
            ),
        )
        connection.commit()


def strategy_stats(days=1, now_ts=None):
    now_ts = float(now_ts or time.time())
    cutoff = now_ts - max(1.0, float(days)) * 86400
    with _LOCK, _connect() as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN result = 1 THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN result = 0 THEN 1 ELSE 0 END) AS losses
            FROM strategy_outcome
            WHERE closed_at >= ?
            """,
            (cutoff,),
        ).fetchone()
    total = int(row["total"] or 0)
    wins = int(row["wins"] or 0)
    losses = int(row["losses"] or 0)
    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "winrate": wins / total * 100 if total else 0.0,
    }


def daily_report_was_sent(report_date):
    with _LOCK, _connect() as connection:
        row = connection.execute(
            "SELECT 1 FROM daily_report_log WHERE report_date = ?",
            (str(report_date),),
        ).fetchone()
    return row is not None


def mark_daily_report_sent(report_date, sent_at=None):
    with _LOCK, _connect() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO daily_report_log (report_date, sent_at)
            VALUES (?, ?)
            """,
            (str(report_date), float(sent_at or time.time())),
        )
        connection.commit()


def build_daily_strategy_report():
    daily = strategy_stats(days=1)
    weekly = strategy_stats(days=7)
    mlx = learning_stats()
    top_factors = factor_performance(limit=3)
    reasons = primary_reason_stats(limit=3)
    regimes = market_regime_stats(limit=3)
    grades = shadow_grade_stats()
    sl_summary = sl_review_summary(limit=3)

    if daily["total"]:
        daily_line = (
            f"近24小時: {daily['winrate']:.1f}% "
            f"（{daily['wins']}勝/{daily['losses']}敗，共{daily['total']}筆）"
        )
    else:
        daily_line = "近24小時: 無已平倉樣本，暫無法判定勝率"

    if weekly["total"]:
        weekly_line = (
            f"近7日: {weekly['winrate']:.1f}% "
            f"（{weekly['wins']}勝/{weekly['losses']}敗，共{weekly['total']}筆）"
        )
    else:
        weekly_line = "近7日: 無已平倉樣本"

    if mlx["evaluated"]:
        mlx_line = (
            f"MLX分析: {mlx['accuracy']:.1f}% "
            f"（成功 {mlx['successful']}/{mlx['evaluated']}，待驗證 {mlx['total'] - mlx['evaluated']}）"
        )
    else:
        mlx_line = f"MLX分析: 尚無已驗證案例（待驗證 {mlx['total']}）"

    warning = ""
    if 0 < daily["total"] < 5:
        warning = "\n⚠️ 24小時樣本少於5筆，勝率僅供參考"

    factor_line = "因子勝率: 尚無已驗證因子"
    if top_factors:
        factor_line = "因子勝率: " + "；".join(
            f"{item['name']} {item['winrate']:.1f}%({item['wins']}/{item['total']})"
            for item in top_factors
        )
    reason_line = "主因勝率: 尚無已驗證主因"
    if reasons:
        reason_line = "主因勝率: " + "；".join(
            f"{item['name']} {item['winrate']:.1f}%({item['wins']}/{item['total']})"
            for item in reasons
        )
    regime_line = "盤型勝率: 尚無已驗證盤型"
    if regimes:
        regime_line = "盤型勝率: " + "；".join(
            f"{item['name']} {item['winrate']:.1f}%({item['wins']}/{item['total']})"
            for item in regimes
        )
    grade_line = "影子單分級: 尚無已驗證分級"
    if grades:
        grade_line = "影子單分級: " + "；".join(
            f"{item['name']}級 {item['winrate']:.1f}%({item['wins']}/{item['total']})"
            for item in grades
        )
    actual_total = int(mlx.get("actual_trades", 0))
    actual_eval = int(mlx.get("actual_trades_evaluated", 0))
    actual_success = int(mlx.get("actual_trades_successful", 0))
    actual_line = (
        f"實單分析: {actual_total}筆；已驗證 {actual_eval}；"
        f"成功 {actual_success}"
    )
    version_items = mlx.get("strategy_versions") or []
    version_line = "策略版本: 尚無版本統計"
    if version_items:
        version_line = "策略版本: " + "；".join(
            f"{item['name']} {item['winrate']:.1f}%({item['wins']}/{item['evaluated']})"
            for item in version_items[:3]
        )
    sl_line = (
        f"SL檢討: 累積 {sl_summary['total']} 筆；"
        f"近{sl_summary['recent_checked']}筆需重審 {sl_summary['revalidation']} 筆"
    )
    if sl_summary["top_issues"]:
        sl_line += "；常見問題 " + "、".join(
            f"{issue}({count})" for issue, count in sl_summary["top_issues"]
        )

    return (
        "📊 每日策略勝率巡檢\n"
        f"{daily_line}\n{weekly_line}\n{mlx_line}\n"
        f"{factor_line}\n{reason_line}\n{regime_line}\n{grade_line}\n"
        f"{actual_line}\n{version_line}\n{sl_line}{warning}"
    )
