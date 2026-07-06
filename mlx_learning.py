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
            sl_price REAL
        )
        """
    )
    existing_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(analysis_episode)")
    }
    for column in ("tp_price", "sl_price"):
        if column not in existing_columns:
            connection.execute(
                f"ALTER TABLE analysis_episode ADD COLUMN {column} REAL"
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
    return (
        ("live_model", Path(ai_data_path("ai_data.csv"))),
        ("backtest_model", Path(ai_data_path("backtest_ai_data.csv"))),
    )


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


def import_existing_model_history(force=False):
    """Import existing classifier samples as retrieval examples for MLX."""
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
            try:
                success = 1 if int(float(row.get("label", 0))) > 0 else 0
            except (TypeError, ValueError):
                continue
            market = {
                key: value
                for key, value in row.items()
                if key != "label" and value not in (None, "")
            }
            if not market:
                continue
            canonical = json.dumps(market, ensure_ascii=False, sort_keys=True)
            source_key = hashlib.sha256(
                f"{source}:{canonical}:{success}".encode("utf-8")
            ).hexdigest()
            direction = (
                directions[index]
                if source == "backtest_model" and index < len(directions)
                else "setup"
            )
            result = "成功" if success else "失敗"
            response = (
                f"既有{source}訓練案例；此市場結構的交易結果為{result}。"
                "用於校準相似型態，不代表目前應直接進場。"
            )
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


def _number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


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


def _extract_shadow_orders(response, entry_price):
    orders = []
    pattern = re.compile(
        r"影子(?:開單|訂單)\s*\d*\s*[：:]\s*(做多|做空)"
        r"[^\n]*?TP\s*[=：:]\s*([0-9]+(?:\.[0-9]+)?)"
        r"[^\n]*?SL\s*[=：:]\s*([0-9]+(?:\.[0-9]+)?)",
        flags=re.IGNORECASE,
    )
    for direction_text, tp_text, sl_text in pattern.findall(str(response or "")):
        direction = {"做多": "long", "做空": "short"}[direction_text]
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
        _extract_shadow_orders(clean_response, price)
        if clean_question.startswith("auto-shadow:")
        else []
    )
    if clean_question.startswith("auto-shadow:") and not shadow_orders:
        return 0
    orders = shadow_orders or [
        {
            "direction": _extract_direction(clean_response),
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
                    tp_price, sl_price
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    created_at,
                    price,
                    order["direction"],
                    (
                        f"{clean_question}:order:{index}"
                        if len(orders) > 1
                        else clean_question
                    ),
                    clean_response,
                    market_json,
                    order["tp_price"],
                    order["sl_price"],
                )
                for index, order in enumerate(orders, start=1)
            ],
        )
        connection.commit()
        return cursor.rowcount


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
            SELECT id, created_at, entry_price, direction, tp_price, sl_price
            FROM analysis_episode
            WHERE evaluated_at IS NULL
            """
        ).fetchall()
        evaluated = 0
        for row in rows:
            move_pct = (price - row["entry_price"]) / row["entry_price"] * 100
            if row["tp_price"] is not None and row["sl_price"] is not None:
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
    if not examples:
        return ""
    return (
        "【過去已驗證經驗】\n"
        "以下案例只作校準，不可取代目前市場資料；應避免重複失敗案例的推理：\n"
        + "\n".join(examples)
    )


def learning_stats():
    import_existing_model_history()
    import_turning_point_history()
    import_non_turning_contrast_history()
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
    }


def record_strategy_outcome(result, close_reason="", close_price=0.0, closed_at=None):
    clean_result = 1 if int(result) > 0 else 0
    with _LOCK, _connect() as connection:
        connection.execute(
            """
            INSERT INTO strategy_outcome (closed_at, result, close_reason, close_price)
            VALUES (?, ?, ?, ?)
            """,
            (
                float(closed_at or time.time()),
                clean_result,
                str(close_reason or "UNKNOWN").upper()[:32],
                _number(close_price),
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

    return f"📊 每日策略勝率巡檢\n{daily_line}\n{weekly_line}\n{mlx_line}{warning}"
