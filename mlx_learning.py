import json
import os
import re
import sqlite3
import threading
import time

from runtime_paths import ai_data_path, ensure_parent_dir


DB_PATH = ai_data_path("mlx_agent_learning.sqlite3")
EVALUATION_HOURS = max(1.0, float(os.getenv("MLX_LEARNING_EVALUATION_HOURS", "4")))
MIN_MOVE_PCT = max(0.05, float(os.getenv("MLX_LEARNING_MIN_MOVE_PCT", "0.25")))
_LOCK = threading.Lock()
_LAST_EVALUATION_TS = 0.0


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
            success INTEGER
        )
        """
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
        CREATE TABLE IF NOT EXISTS daily_report_log (
            report_date TEXT PRIMARY KEY,
            sent_at REAL NOT NULL
        )
        """
    )
    connection.commit()
    return connection


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


def record_analysis(question, response, market):
    market = market if isinstance(market, dict) else {}
    price = _number(market.get("price"))
    if price <= 0 or not response:
        return None
    direction = _extract_direction(response)
    with _LOCK, _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO analysis_episode
                (created_at, entry_price, direction, question, response, market_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                price,
                direction,
                str(question or "")[:2000],
                str(response)[:8000],
                json.dumps(market, ensure_ascii=False, default=str),
            ),
        )
        connection.commit()
        return cursor.lastrowid


def evaluate_pending(current_price, now_ts=None):
    global _LAST_EVALUATION_TS
    price = _number(current_price)
    now_ts = float(now_ts or time.time())
    if price <= 0 or now_ts - _LAST_EVALUATION_TS < 60:
        return 0
    _LAST_EVALUATION_TS = now_ts
    cutoff = now_ts - EVALUATION_HOURS * 3600

    with _LOCK, _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, entry_price, direction
            FROM analysis_episode
            WHERE evaluated_at IS NULL AND created_at <= ?
            """,
            (cutoff,),
        ).fetchall()
        for row in rows:
            move_pct = (price - row["entry_price"]) / row["entry_price"] * 100
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
        connection.commit()
        return len(rows)


def build_learning_context(market, limit=5):
    market = market if isinstance(market, dict) else {}
    with _LOCK, _connect() as connection:
        rows = connection.execute(
            """
            SELECT direction, response, market_json, return_pct, success
            FROM analysis_episode
            WHERE evaluated_at IS NOT NULL
            ORDER BY success DESC, evaluated_at DESC
            LIMIT 30
            """
        ).fetchall()

    scored = []
    for row in rows:
        try:
            past_market = json.loads(row["market_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            past_market = {}
        similarity = 0
        for key in ("htf", "regime", "breakout", "triangle", "macro", "volume_spike"):
            current_value = market.get(key)
            if current_value is not None and str(current_value) == str(past_market.get(key)):
                similarity += 1
        scored.append((similarity, int(row["success"]), row))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    examples = []
    for _, _, row in scored[: max(0, int(limit))]:
        result = "成功" if row["success"] else "失敗"
        response = re.sub(r"\s+", " ", str(row["response"]))[:500]
        examples.append(
            f"- {row['direction']}｜{result}｜後續漲跌 {row['return_pct']:+.2f}%｜當時分析：{response}"
        )
    if not examples:
        return ""
    return (
        "【過去已驗證經驗】\n"
        "以下案例只作校準，不可取代目前市場資料；應避免重複失敗案例的推理：\n"
        + "\n".join(examples)
    )


def learning_stats():
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
