#!/usr/bin/env python3
import asyncio
import base64
import csv
import hashlib
import hmac
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qsl

import requests

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from runtime_config import (
    env_bool as _safe_bool_env,
    env_float as _safe_float_env,
    env_int as _safe_int_env,
    load_local_env,
    read_local_env_values as _read_local_env_values,
)
from runtime_paths import data_path


load_local_env()

TWELVE_API_USAGE_CACHE = {"ts": 0.0, "data": None, "error": ""}


def _load_origins():
    raw = str(os.getenv("POSITION_PANEL_ALLOWED_ORIGINS", "*") or "").strip()
    if not raw or raw == "*":
        return ["*"]
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    # A directly opened docs/index.html has the opaque Origin "null".  Permit
    # that local preview to read public market data from the loopback server;
    # protected position/state endpoints still enforce viewer authorization.
    if "null" not in origins:
        origins.append("null")
    return origins


def _runtime_env_value(name: str, default=""):
    local_values = _read_local_env_values()
    if name in local_values:
        return local_values[name]
    return os.getenv(name, default)


def _build_api_token_usage() -> dict:
    usage_path = data_path("api_token_usage.json")
    persisted = {}
    try:
        persisted = json.loads(usage_path.read_text(encoding="utf-8"))
    except Exception:
        persisted = {}
    today = time.strftime("%Y-%m-%d", time.gmtime())
    twelve = persisted.get("twelve_data") if isinstance(persisted, dict) else {}
    twelve = twelve if isinstance(twelve, dict) else {}
    twelve_used = max(0, int(twelve.get("count", 0))) if str(twelve.get("day") or "") == today else 0
    twelve_limit = max(1, _safe_int_env("TWELVE_DATA_DAILY_REQUEST_LIMIT", 800))
    twelve_key = str(_runtime_env_value("TWELVE_DATA_API_KEY", "") or "").strip()
    official = TWELVE_API_USAGE_CACHE.get("data") if isinstance(TWELVE_API_USAGE_CACHE.get("data"), dict) else {}
    refresh_sec = max(60, _safe_int_env("TWELVE_DATA_USAGE_REFRESH_SEC", 300))
    if twelve_key and time.time() - float(TWELVE_API_USAGE_CACHE.get("ts", 0) or 0) >= refresh_sec:
        try:
            response = requests.get(
                "https://api.twelvedata.com/api_usage",
                headers={"Authorization": f"apikey {twelve_key}", "User-Agent": "ETH-bot/1.0"},
                timeout=8,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or payload.get("status") == "error":
                raise RuntimeError(str((payload or {}).get("message") or "invalid response"))
            official = payload
            TWELVE_API_USAGE_CACHE.update({"ts": time.time(), "data": payload, "error": ""})
        except Exception as exc:
            TWELVE_API_USAGE_CACHE.update({"ts": time.time(), "error": str(exc)})

    minute_used = max(0, int(official.get("current_usage", 0) or 0))
    minute_limit = max(1, int(official.get("plan_limit", 8) or 8))
    daily_used = max(0, int(official.get("daily_usage", twelve_used) or 0))
    daily_limit = max(1, int(official.get("plan_daily_limit", twelve_limit) or twelve_limit))
    official_ts = float(TWELVE_API_USAGE_CACHE.get("ts", 0) or 0)

    items = [
        {
            "id": "twelve_data", "name": "Twelve Data", "configured": bool(twelve_key),
            "used": minute_used, "limit": minute_limit, "remaining": max(0, minute_limit - minute_used),
            "daily_used": daily_used, "daily_limit": daily_limit, "daily_remaining": max(0, daily_limit - daily_used),
            "plan": str(official.get("plan_category") or ""), "official": bool(official),
            "unit": "credits/min", "reset": "每分鐘重置；每日額度於 00:00 UTC 重置", "measurable": True,
            "official_updated_ts": official_ts,
            "error": str(TWELVE_API_USAGE_CACHE.get("error") or ""),
            "last_request_ts": float(twelve.get("last_request_ts", 0) or 0),
        },
        {
            "id": "binance", "name": "Binance", "configured": bool(str(_runtime_env_value("BINANCE_API_KEY", "") or "").strip()),
            "measurable": False, "note": "交易所採動態 request weight，程式已啟用429保護",
        },
        {
            "id": "telegram", "name": "Telegram", "configured": bool(str(_runtime_env_value("TELEGRAM_TOKEN", "") or "").strip()),
            "measurable": False, "note": "Bot API 不提供每日用量配額",
        },
    ]
    return {"ok": True, "day_utc": today, "items": items, "ts": int(time.time())}


def _load_allowed_user_ids(raw: str):
    values = set()
    for item in str(raw or "").strip().split(","):
        text = str(item or "").strip()
        if not text:
            continue
        try:
            values.add(int(text))
        except Exception:
            continue
    return values

def _resolve_panel_port(default: int = 8787) -> int:
    raw = (
        str(_runtime_env_value("POSITION_PANEL_REALTIME_PORT", "") or "").strip()
        or str(_runtime_env_value("PORT", "") or "").strip()
    )
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


WS_PING_INTERVAL_SEC = max(10.0, _safe_float_env("POSITION_PANEL_WS_PING_INTERVAL_SEC", 20.0))

app = FastAPI(title="ETH Bot Panel Realtime", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_load_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LATEST_STATE = {"open": False, "ts": 0}
STATE_LOCK = asyncio.Lock()
CLIENTS = set()
CLIENTS_LOCK = asyncio.Lock()
MARKET_DATA_CACHE = {}
MARKET_DATA_LOCK = asyncio.Lock()
MARKET_PRICE_REFRESH_LOCK = asyncio.Lock()
MARKET_PRICE_FAILURE_UNTIL = {}
MARKET_DATA_TTL_SEC = max(3.0, _safe_float_env("POSITION_PANEL_MARKET_DATA_TTL_SEC", 8.0))
MARKET_LIVE_DATA_TTL_SEC = 3.0
MARKET_PRICE_TTL_SEC = max(0.5, _safe_float_env("POSITION_PANEL_MARKET_PRICE_TTL_SEC", 1.0))
MARKET_PRICE_ERROR_COOLDOWN_SEC = max(
    0.5,
    _safe_float_env("POSITION_PANEL_MARKET_PRICE_ERROR_COOLDOWN_SEC", 2.0),
)
MARKET_PRICE_MAX_AGE_SEC = max(2.0, _safe_float_env("POSITION_PANEL_MARKET_PRICE_MAX_AGE_SEC", 10.0))
MARKET_PRICE_MAX_DEVIATION_RATE = max(
    0.001,
    min(0.05, _safe_float_env("POSITION_PANEL_MARKET_PRICE_MAX_DEVIATION_RATE", 0.01)),
)
PANEL_ERROR_LOG_ENABLED = _safe_bool_env("POSITION_PANEL_ERROR_LOG_ENABLED", True)
APP_VERSION_RE = re.compile(r'const\s+APP_VERSION\s*=\s*["\']([^"\']*)["\'];')


class PanelHttpErrorLogMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        start_ts = time.time()
        status_code = {"value": 0}

        async def send_wrapper(message):
            if message.get("type") == "http.response.start":
                status_code["value"] = int(message.get("status") or 0)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            if PANEL_ERROR_LOG_ENABLED:
                print(f"⚠️ panel request error | {scope.get('method')} {scope.get('path')} | {exc!r}")
            raise

        if PANEL_ERROR_LOG_ENABLED and status_code["value"] >= 400:
            elapsed_ms = (time.time() - start_ts) * 1000.0
            print(
                f"⚠️ panel HTTP {status_code['value']} | "
                f"{scope.get('method')} {scope.get('path')} | {elapsed_ms:.1f}ms"
            )


app.add_middleware(PanelHttpErrorLogMiddleware)


def _viewer_auth_settings():
    telegram_token = str(_runtime_env_value("TELEGRAM_TOKEN", "") or "").strip()
    panel_token = str(_runtime_env_value("POSITION_PANEL_REALTIME_TOKEN", "") or "").strip()
    max_age_sec = max(
        60,
        _safe_int_env("POSITION_PANEL_TELEGRAM_AUTH_MAX_AGE_SEC", 86400),
    )
    session_ttl_sec = max(
        300,
        _safe_int_env("POSITION_PANEL_SESSION_TTL_SEC", 2592000),
    )
    allowed_ids = _load_allowed_user_ids(
        _runtime_env_value(
            "POSITION_PANEL_ALLOWED_TELEGRAM_USER_IDS",
            _runtime_env_value("TELEGRAM_CHAT_ID", ""),
        )
    )
    return {
        "telegram_token": telegram_token,
        "panel_token": panel_token,
        "max_age_sec": max_age_sec,
        "session_ttl_sec": session_ttl_sec,
        "allowed_ids": allowed_ids,
    }


def _panel_app_version(path: Path) -> str:
    raw = str(os.getenv("POSITION_PANEL_APP_VERSION", "") or "").strip()
    if raw:
        return raw
    try:
        return time.strftime("%Y%m%d-%H%M%S", time.localtime(path.stat().st_mtime))
    except Exception:
        return str(int(time.time()))


def _render_panel_html(path: Path) -> str:
    html = path.read_text(encoding="utf-8")
    version = _panel_app_version(path)
    replacement = f'const APP_VERSION = "{version}";'
    if APP_VERSION_RE.search(html):
        return APP_VERSION_RE.sub(replacement, html, count=1)
    return html


def _token_valid(token: str) -> bool:
    panel_token = _viewer_auth_settings().get("panel_token", "")
    if not panel_token:
        return True
    return str(token or "").strip() == panel_token


def _extract_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return str(request.headers.get("x-panel-token", "") or request.query_params.get("token", "")).strip()


def _extract_init_data_http(request: Request) -> str:
    return str(
        request.headers.get("x-telegram-init-data", "")
        or request.query_params.get("tg_init_data", "")
        or request.query_params.get("initData", "")
        or ""
    ).strip()


def _extract_init_data_ws(websocket: WebSocket) -> str:
    return str(
        websocket.query_params.get("tg_init_data", "")
        or websocket.query_params.get("initData", "")
        or ""
    ).strip()


def _extract_panel_session_http(request: Request) -> str:
    return str(
        request.headers.get("x-panel-session", "")
        or request.query_params.get("panel_session", "")
        or ""
    ).strip()


def _extract_panel_session_ws(websocket: WebSocket) -> str:
    return str(websocket.query_params.get("panel_session", "") or "").strip()


def _normalize_market_symbol(symbol: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]", "", str(symbol or "ETHUSDT")).upper()
    return clean if clean else "ETHUSDT"


def _fetch_binance_mark_price_sync(symbol: str) -> dict:
    expected_symbol = _normalize_market_symbol(symbol)
    mark_response = requests.get(
        "https://fapi.binance.com/fapi/v1/premiumIndex",
        params={"symbol": expected_symbol},
        timeout=6,
    )
    mark_response.raise_for_status()
    mark_payload = mark_response.json()
    ticker_response = requests.get(
        "https://fapi.binance.com/fapi/v1/ticker/price",
        params={"symbol": expected_symbol},
        timeout=6,
    )
    ticker_response.raise_for_status()
    ticker_payload = ticker_response.json()

    if str(mark_payload.get("symbol") or "").upper() != expected_symbol:
        raise RuntimeError("Binance mark price symbol mismatch")
    if str(ticker_payload.get("symbol") or "").upper() != expected_symbol:
        raise RuntimeError("Binance ticker symbol mismatch")

    mark_price = float(mark_payload.get("markPrice", 0) or 0)
    index_price = float(mark_payload.get("indexPrice", 0) or 0)
    last_price = float(ticker_payload.get("price", 0) or 0)
    exchange_ts_ms = int(float(mark_payload.get("time", 0) or 0))
    ticker_ts_ms = int(float(ticker_payload.get("time", 0) or 0))
    if min(mark_price, index_price, last_price, exchange_ts_ms, ticker_ts_ms) <= 0:
        raise RuntimeError("Binance price payload is incomplete")

    now_ms = int(time.time() * 1000)
    mark_age_sec = max(0.0, (now_ms - exchange_ts_ms) / 1000.0)
    ticker_age_sec = max(0.0, (now_ms - ticker_ts_ms) / 1000.0)
    if mark_age_sec > MARKET_PRICE_MAX_AGE_SEC or ticker_age_sec > MARKET_PRICE_MAX_AGE_SEC:
        raise RuntimeError("Binance price payload is stale")

    index_deviation = abs(mark_price - index_price) / index_price
    ticker_deviation = abs(mark_price - last_price) / last_price
    max_deviation = max(index_deviation, ticker_deviation)
    if max_deviation > MARKET_PRICE_MAX_DEVIATION_RATE:
        raise RuntimeError("Binance mark price cross-check failed")

    return {
        "symbol": expected_symbol,
        "price": mark_price,
        "index_price": index_price,
        "last_price": last_price,
        "exchange_ts": exchange_ts_ms / 1000.0,
        "max_deviation_rate": max_deviation,
    }


def _usable_cached_market_price(cached, now_ts: float):
    if not isinstance(cached, dict):
        return None
    payload = cached.get("payload")
    if not isinstance(payload, dict):
        return None
    try:
        exchange_ts = float(payload.get("exchange_ts", 0.0) or 0.0)
        price = float(payload.get("price", 0.0) or 0.0)
    except Exception:
        return None
    if payload.get("validated") is not True or price <= 0 or exchange_ts <= 0:
        return None
    if max(0.0, float(now_ts) - exchange_ts) > MARKET_PRICE_MAX_AGE_SEC:
        return None
    return dict(payload)


def _fetch_market_klines_sync(symbol: str, interval: str, limit: int):
    os.environ.setdefault("ETH_BOT_DISABLE_LIVE", "1")
    import eth  # noqa: WPS433 - lazy import keeps panel startup light and disables live side effects.

    rows, source = eth._fetch_market_kline_rows(
        symbol,
        interval,
        limit=limit,
        timeout=10,
        prefix="面板非Binance K線",
        source_preference="kraken_first",
        allow_binance_fallback=False,
    )
    if str(source or "").lower().startswith("binance"):
        raise RuntimeError("panel kline source must not use Binance")
    interval_ms = getattr(eth, "KLINE_INTERVAL_MS", {}).get(interval, 60 * 1000)
    parsed = []
    for row in rows if isinstance(rows, list) else []:
        try:
            open_ms = int(float(row[0]))
            parsed.append(
                {
                    "ts": open_ms,
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]) if len(row) > 5 else 0.0,
                    "close_ts": open_ms + int(interval_ms) - 1,
                }
            )
        except Exception:
            continue
    if interval == "1m" and str(source or "").lower().startswith("coinbase") and parsed:
        try:
            ticker_response = requests.get(
                "https://api.exchange.coinbase.com/products/ETH-USD/ticker",
                headers={"User-Agent": "ETH-bot/1.0"},
                timeout=5,
            )
            ticker_response.raise_for_status()
            ticker_payload = ticker_response.json()
            ticker_price = float((ticker_payload or {}).get("price", 0) or 0)
            previous_close = float(parsed[-1].get("close", 0) or 0)
            deviation = abs(ticker_price - previous_close) / max(ticker_price, previous_close, 1e-9)
            if ticker_price <= 0 or previous_close <= 0 or deviation > 0.01:
                raise RuntimeError("Coinbase ticker cross-check failed")
            bucket_ms = (int(time.time() * 1000) // int(interval_ms)) * int(interval_ms)
            last = parsed[-1]
            if int(last.get("ts", 0)) == bucket_ms:
                last["high"] = max(float(last.get("high", ticker_price)), ticker_price)
                last["low"] = min(float(last.get("low", ticker_price)), ticker_price)
                last["close"] = ticker_price
            elif bucket_ms > int(last.get("ts", 0)):
                parsed.append(
                    {
                        "ts": bucket_ms,
                        "open": previous_close,
                        "high": max(previous_close, ticker_price),
                        "low": min(previous_close, ticker_price),
                        "close": ticker_price,
                        "volume": 0.0,
                        "close_ts": bucket_ms + int(interval_ms) - 1,
                    }
                )
            source = "coinbase_live"
        except Exception:
            # Keep the last valid Coinbase candle when the ticker is temporarily unavailable.
            pass
    return parsed, str(source or "tradingview")


BACKTEST_PANEL_ARTIFACTS = [
    {
        "period": "2022",
        "label": "2022",
        "market_label": "熊市",
        "summary": "backtest_2022_market_profile_try3_summary.json",
        "trades": "backtest_2022_market_profile_try3_trades.csv",
    },
    {
        "period": "2023",
        "label": "2023",
        "market_label": "震盪/築底",
        "summary": "backtest_2023_market_profile_try3_summary.json",
        "trades": "backtest_2023_market_profile_try3_trades.csv",
    },
    {
        "period": "2024",
        "label": "2024",
        "market_label": "牛市",
        "summary": "backtest_2024_market_profile_try3_summary.json",
        "trades": "backtest_2024_market_profile_try3_trades.csv",
    },
    {
        "period": "2025",
        "label": "2025",
        "market_label": "牛市高波動",
        "summary": "backtest_2025_market_profile_try5_summary.json",
        "trades": "backtest_2025_market_profile_try5_trades.csv",
    },
    {
        "period": "2026H1",
        "label": "2026H1",
        "market_label": "2026 上半年",
        "summary": "backtest_2026h1_market_profile_try5_summary.json",
        "trades": "backtest_2026h1_market_profile_try5_trades.csv",
    },
]


MARKET_PROFILE_LABELS = {
    "bear": "熊市",
    "range_base": "震盪/築底",
    "bull": "牛市",
    "bull_high_vol": "牛市高波動",
}


def _panel_backtest_data_dir() -> Path:
    repo_dir = Path(__file__).resolve().parent
    data_dir = Path(os.getenv("BOT_DATA_DIR", repo_dir / ".runtime" / "data")).expanduser()
    return data_dir / "backtests"


def _safe_json_file(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _safe_float_value(value, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if number == number else float(default)
    except Exception:
        return float(default)


def _safe_int_value(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _dominant_counter_value(counter: Counter, fallback: str = "") -> str:
    if not counter:
        return fallback
    value, _count = counter.most_common(1)[0]
    return str(value or fallback)


def _is_daily_min_trade_row(row: dict) -> bool:
    signal = str(row.get("signal") or "")
    host_mode = str(row.get("host_logic_mode") or "")
    exit_reason = str(row.get("exit_reason") or "")
    return (
        "每日保底" in signal
        or host_mode == "daily_minimum"
        or exit_reason == "DAILY_MIN_ROLLOVER"
    )


def _empty_return_bucket() -> dict:
    return {
        "trades": 0,
        "wins": 0,
        "return_pct": 0.0,
    }


def _add_return_bucket(bucket: dict, trade_return: float) -> None:
    bucket["trades"] += 1
    bucket["return_pct"] += trade_return * 100.0
    if trade_return > 0:
        bucket["wins"] += 1


def _finish_return_bucket(bucket: dict) -> dict:
    trades = int(bucket.get("trades") or 0)
    wins = int(bucket.get("wins") or 0)
    return {
        "trades": trades,
        "wins": wins,
        "win_rate": round((wins / trades) * 100.0, 2) if trades else 0.0,
        "return_pct": round(float(bucket.get("return_pct") or 0.0), 3),
    }


def _aggregate_trade_source_returns(trades_path: Path) -> dict:
    daily = _empty_return_bucket()
    other = _empty_return_bucket()
    if not trades_path.exists():
        return {
            "daily_min": _finish_return_bucket(daily),
            "other": _finish_return_bucket(other),
        }

    try:
        with trades_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                trade_return = _safe_float_value(row.get("trade_return"), 0.0)
                _add_return_bucket(daily if _is_daily_min_trade_row(row) else other, trade_return)
    except Exception:
        pass

    return {
        "daily_min": _finish_return_bucket(daily),
        "other": _finish_return_bucket(other),
    }


def _aggregate_monthly_backtests(trades_path: Path, period: str) -> list[dict]:
    if not trades_path.exists():
        return []

    buckets = defaultdict(
        lambda: {
            "period": period,
            "month": "",
            "trades": 0,
            "wins": 0,
            "return_pct": 0.0,
            "daily_min": _empty_return_bucket(),
            "other": _empty_return_bucket(),
            "long_trades": 0,
            "short_trades": 0,
            "phase_counts": Counter(),
            "indicator_counts": Counter(),
        }
    )

    try:
        with trades_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                opened_at = str(row.get("opened_at") or "").strip()
                month = opened_at[:7]
                if not re.match(r"^\d{4}-\d{2}$", month):
                    continue
                bucket = buckets[month]
                bucket["month"] = month
                bucket["trades"] += 1
                trade_return = _safe_float_value(row.get("trade_return"), 0.0)
                bucket["return_pct"] += trade_return * 100.0
                if trade_return > 0:
                    bucket["wins"] += 1
                _add_return_bucket(bucket["daily_min"] if _is_daily_min_trade_row(row) else bucket["other"], trade_return)
                direction = str(row.get("direction") or "").strip().lower()
                if direction == "long":
                    bucket["long_trades"] += 1
                elif direction == "short":
                    bucket["short_trades"] += 1
                phase = str(row.get("market_profile_phase") or "").strip()
                indicator = str(row.get("market_profile_indicator_family") or "").strip()
                if phase:
                    bucket["phase_counts"][phase] += 1
                if indicator:
                    bucket["indicator_counts"][indicator] += 1
    except Exception:
        return []

    rows = []
    for month in sorted(buckets):
        bucket = buckets[month]
        phase = _dominant_counter_value(bucket["phase_counts"], "unknown")
        indicator = _dominant_counter_value(bucket["indicator_counts"], "")
        trades = int(bucket["trades"])
        rows.append(
            {
                "period": period,
                "month": month,
                "market_phase": phase,
                "market_label": MARKET_PROFILE_LABELS.get(phase, phase),
                "indicator_family": indicator,
                "trades": trades,
                "wins": int(bucket["wins"]),
                "win_rate": round((bucket["wins"] / trades) * 100.0, 2) if trades else 0.0,
                "return_pct": round(float(bucket["return_pct"]), 3),
                "daily_min": _finish_return_bucket(bucket["daily_min"]),
                "other": _finish_return_bucket(bucket["other"]),
                "long_trades": int(bucket["long_trades"]),
                "short_trades": int(bucket["short_trades"]),
            }
        )
    return rows


def _build_backtest_panel_summary() -> dict:
    backtest_dir = _panel_backtest_data_dir()
    yearly_rows = []
    monthly_rows = []
    compound_equity = 1.0
    newest_mtime = 0.0

    for artifact in BACKTEST_PANEL_ARTIFACTS:
        summary_path = backtest_dir / artifact["summary"]
        trades_path = backtest_dir / artifact["trades"]
        summary = _safe_json_file(summary_path)
        if not summary:
            continue

        total_return_pct = _safe_float_value(summary.get("total_return_pct"), 0.0)
        compound_equity *= 1.0 + (total_return_pct / 100.0)
        coverage = summary.get("trade_day_coverage") if isinstance(summary.get("trade_day_coverage"), dict) else {}
        source_returns = _aggregate_trade_source_returns(trades_path)
        for path in (summary_path, trades_path):
            try:
                newest_mtime = max(newest_mtime, path.stat().st_mtime)
            except Exception:
                pass

        yearly_rows.append(
            {
                "period": artifact["period"],
                "label": artifact["label"],
                "market_label": artifact["market_label"],
                "summary_file": artifact["summary"],
                "trades_file": artifact["trades"],
                "trades": _safe_int_value(summary.get("trades"), 0),
                "wins": _safe_int_value(summary.get("wins"), 0),
                "losses": _safe_int_value(summary.get("losses"), 0),
                "win_rate": _safe_float_value(summary.get("win_rate"), 0.0),
                "total_return_pct": total_return_pct,
                "profit_factor": _safe_float_value(summary.get("profit_factor"), 0.0),
                "max_drawdown_pct": _safe_float_value(summary.get("max_drawdown_pct"), 0.0),
                "long_trades": _safe_int_value(summary.get("long_trades"), 0),
                "short_trades": _safe_int_value(summary.get("short_trades"), 0),
                "covered_days": _safe_int_value(coverage.get("covered_days", coverage.get("trade_days")), 0),
                "expected_days": _safe_int_value(coverage.get("expected_days", coverage.get("calendar_days")), 0),
                "missing_days": _safe_int_value(coverage.get("missing_days", coverage.get("missing_trade_days")), 0),
                "daily_min_trades": _safe_int_value(coverage.get("daily_min_trades"), 0),
                "daily_min": source_returns["daily_min"],
                "other": source_returns["other"],
            }
        )
        monthly_rows.extend(_aggregate_monthly_backtests(trades_path, artifact["period"]))

    return {
        "ok": True,
        "ts": int(time.time()),
        "updated_at": int(newest_mtime) if newest_mtime > 0 else 0,
        "strategy": "market_profile_monthly",
        "compound_return_pct": round((compound_equity - 1.0) * 100.0, 4),
        "yearly": yearly_rows,
        "monthly": monthly_rows,
    }


def _parse_telegram_init_data(init_data: str) -> dict:
    values = {}
    for key, value in parse_qsl(str(init_data or ""), keep_blank_values=True):
        if key:
            values[key] = value
    return values


def _urlsafe_b64decode(text: str) -> bytes:
    raw = str(text or "").strip()
    if not raw:
        return b""
    padding = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode((raw + padding).encode("ascii"))


def _panel_session_secret(settings=None) -> str:
    settings = settings or _viewer_auth_settings()
    return str(settings.get("panel_token") or settings.get("telegram_token") or "").strip()


def _validate_panel_session(panel_session: str):
    settings = _viewer_auth_settings()
    secret = _panel_session_secret(settings)
    if not secret:
        return None

    token = str(panel_session or "").strip()
    if "." not in token:
        return None

    body, their_sig = token.split(".", 1)
    if not body or not their_sig:
        return None

    expected_sig = base64.urlsafe_b64encode(
        hmac.new(secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).digest()
    ).decode("ascii").rstrip("=")
    if not hmac.compare_digest(expected_sig, their_sig):
        return None

    try:
        payload = json.loads(_urlsafe_b64decode(body).decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    try:
        user_id = int(payload.get("uid"))
        issued_at = int(payload.get("iat", 0))
        expires_at = int(payload.get("exp", 0))
    except Exception:
        return None

    now_ts = int(time.time())
    max_ttl = max(300, int(settings.get("session_ttl_sec", 43200) or 43200))
    if issued_at <= 0 or expires_at <= issued_at:
        return None
    if (expires_at - issued_at) > max_ttl:
        return None
    if now_ts >= expires_at:
        return None

    allowed_ids = settings.get("allowed_ids") or set()
    if allowed_ids and user_id not in allowed_ids:
        return None

    payload["user_id"] = user_id
    payload["iat"] = issued_at
    payload["exp"] = expires_at
    return payload


def _validate_telegram_init_data(init_data: str):
    settings = _viewer_auth_settings()
    telegram_token = str(settings.get("telegram_token", "") or "").strip()
    max_age_sec = max(60, int(settings.get("max_age_sec", 86400) or 86400))
    allowed_ids = settings.get("allowed_ids") or set()

    if not telegram_token:
        return None

    payload = _parse_telegram_init_data(init_data)
    their_hash = str(payload.pop("hash", "") or "").strip()
    if not payload or not their_hash:
        return None

    try:
        auth_date = int(str(payload.get("auth_date", "") or "").strip())
    except Exception:
        return None

    now_ts = int(time.time())
    if auth_date <= 0 or (now_ts - auth_date) > max_age_sec:
        return None

    data_check_string = "\n".join(f"{key}={payload[key]}" for key in sorted(payload.keys()))
    secret_key = hmac.new(b"WebAppData", telegram_token.encode("utf-8"), hashlib.sha256).digest()
    expected_hash = hmac.new(secret_key, data_check_string.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_hash, their_hash):
        return None

    try:
        user = json.loads(payload.get("user", "{}"))
    except Exception:
        user = {}

    user_id = None
    try:
        user_id = int(user.get("id"))
    except Exception:
        user_id = None

    if allowed_ids and user_id not in allowed_ids:
        return None

    payload["user"] = user
    payload["user_id"] = user_id
    payload["auth_date"] = auth_date
    return payload


def _viewer_authorized_http(request: Request) -> bool:
    settings = _viewer_auth_settings()
    panel_token = str(settings.get("panel_token", "") or "").strip()
    telegram_token = str(settings.get("telegram_token", "") or "").strip()

    if panel_token and _token_valid(_extract_token(request)):
        return True
    if _validate_panel_session(_extract_panel_session_http(request)) is not None:
        return True
    if not telegram_token:
        return not panel_token
    return _validate_telegram_init_data(_extract_init_data_http(request)) is not None


def _is_loopback_request(request: Request) -> bool:
    host = str(getattr(getattr(request, "client", None), "host", "") or "").strip().lower()
    return host in {"127.0.0.1", "::1", "localhost", "testclient"}


def _market_data_authorized_http(request: Request) -> bool:
    # Mark price and K-lines are public exchange data.  Local previews may read
    # them without exposing private position/state data to unauthenticated users.
    return _is_loopback_request(request) or _viewer_authorized_http(request)


def _viewer_authorized_ws(websocket: WebSocket) -> bool:
    settings = _viewer_auth_settings()
    panel_token = str(settings.get("panel_token", "") or "").strip()
    telegram_token = str(settings.get("telegram_token", "") or "").strip()
    token = str(websocket.query_params.get("token", "") or "").strip()
    if panel_token and _token_valid(token):
        return True
    if _validate_panel_session(_extract_panel_session_ws(websocket)) is not None:
        return True
    if not telegram_token:
        return not panel_token
    return _validate_telegram_init_data(_extract_init_data_ws(websocket)) is not None


async def _broadcast_state(payload: dict):
    message = json.dumps(
        {
            "type": "position",
            "data": payload,
            "server_ts": int(time.time()),
        },
        ensure_ascii=False,
    )

    async with CLIENTS_LOCK:
        sockets = list(CLIENTS)

    stale = []
    for websocket in sockets:
        try:
            await websocket.send_text(message)
        except Exception:
            stale.append(websocket)

    if stale:
        async with CLIENTS_LOCK:
            for websocket in stale:
                CLIENTS.discard(websocket)


@app.get("/")
async def root():
    panel_path = Path(__file__).resolve().parent / "docs" / "index.html"
    if panel_path.exists():
        return HTMLResponse(_render_panel_html(panel_path))
    return {"ok": True, "service": "panel-realtime"}


@app.head("/")
async def root_head():
    return Response(status_code=200)


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.get("/position.json")
@app.get("/ETH-bot/docs/position.json")
@app.get("/ETH-bot/position.json")
async def local_position_snapshot():
    position_path = Path(__file__).resolve().parent / "docs" / "position.json"
    if not position_path.exists():
        raise HTTPException(status_code=404, detail="position snapshot not found")
    # The bot replaces this snapshot frequently. Buffer one complete version so
    # a concurrent replacement cannot make FileResponse's Content-Length differ
    # from the bytes that uvicorn eventually sends.
    try:
        payload = position_path.read_bytes()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="position snapshot not found") from exc
    return Response(content=payload, media_type="application/json", headers={"Cache-Control": "no-store"})


@app.get("/world-map.svg")
@app.get("/ETH-bot/docs/world-map.svg")
async def local_world_map():
    map_path = Path(__file__).resolve().parent / "docs" / "world-map.svg"
    if not map_path.exists():
        raise HTTPException(status_code=404, detail="world map not found")
    return FileResponse(map_path, media_type="image/svg+xml")


@app.get("/backtest_latest_summary.json")
async def local_backtest_summary():
    repo_dir = Path(__file__).resolve().parent
    data_dir = Path(os.getenv("BOT_DATA_DIR", repo_dir / ".runtime" / "data")).expanduser()
    summary_path = data_dir / "backtest_latest_summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="backtest summary not found")
    return FileResponse(summary_path, media_type="application/json")


@app.get("/api/backtests/summary")
async def get_backtests_summary(request: Request):
    # Native Mac app reads the historical report directly from localhost.
    # Remote/tunnel clients still require the regular Telegram/panel auth.
    if not _is_loopback_request(request) and not _viewer_authorized_http(request):
        raise HTTPException(status_code=401, detail="unauthorized")
    return JSONResponse(_build_backtest_panel_summary())


@app.get("/healthz")
async def healthz():
    return {"ok": True, "ts": int(time.time())}


@app.get("/api/market/klines")
async def get_market_klines(request: Request):
    if not _market_data_authorized_http(request):
        raise HTTPException(status_code=401, detail="unauthorized")

    symbol = _normalize_market_symbol(request.query_params.get("symbol", "ETHUSDT"))
    interval = str(request.query_params.get("interval", "1m") or "1m").strip()
    if interval not in {"1m", "4h"}:
        raise HTTPException(status_code=400, detail="unsupported interval")
    try:
        limit = max(1, min(500, int(str(request.query_params.get("limit", "32")).strip())))
    except Exception:
        limit = 32

    cache_key = f"{symbol}:{interval}:{limit}"
    now_ts = time.time()
    cache_ttl_sec = MARKET_LIVE_DATA_TTL_SEC if interval == "1m" else MARKET_DATA_TTL_SEC
    async with MARKET_DATA_LOCK:
        cached = MARKET_DATA_CACHE.get(cache_key)
        if cached and now_ts - float(cached.get("ts", 0.0)) < cache_ttl_sec:
            return JSONResponse(dict(cached.get("payload") or {}))

    try:
        rows, source = await asyncio.to_thread(_fetch_market_klines_sync, symbol, interval, limit)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"market data unavailable: {exc}") from exc
    if not rows:
        raise HTTPException(status_code=502, detail="market data empty")

    payload = {
        "ok": True,
        "symbol": symbol,
        "interval": interval,
        "source": source,
        "rows": rows,
        "ts": int(now_ts),
    }
    async with MARKET_DATA_LOCK:
        MARKET_DATA_CACHE[cache_key] = {"ts": now_ts, "payload": payload}
    return JSONResponse(payload)


@app.get("/api/market/price")
async def get_market_price(request: Request):
    if not _market_data_authorized_http(request):
        raise HTTPException(status_code=401, detail="unauthorized")

    symbol = _normalize_market_symbol(request.query_params.get("symbol", "ETHUSDT"))
    cache_key = f"mark_price:{symbol}"
    now_ts = time.time()
    async with MARKET_DATA_LOCK:
        cached = MARKET_DATA_CACHE.get(cache_key)
        if cached and now_ts - float(cached.get("ts", 0.0)) < MARKET_PRICE_TTL_SEC:
            return JSONResponse(dict(cached.get("payload") or {}))

    # Collapse simultaneous browser/bot refreshes into one Binance request.
    # Without this lock, a brief upstream failure causes every waiting client
    # to retry independently and amplifies the outage into a local 502 storm.
    async with MARKET_PRICE_REFRESH_LOCK:
        now_ts = time.time()
        async with MARKET_DATA_LOCK:
            cached = MARKET_DATA_CACHE.get(cache_key)
            if cached and now_ts - float(cached.get("ts", 0.0)) < MARKET_PRICE_TTL_SEC:
                return JSONResponse(dict(cached.get("payload") or {}))

        failure_until = float(MARKET_PRICE_FAILURE_UNTIL.get(cache_key, 0.0) or 0.0)
        if now_ts < failure_until:
            fallback = _usable_cached_market_price(cached, now_ts)
            if fallback is not None:
                fallback["cached"] = True
                return JSONResponse(fallback)
            raise HTTPException(status_code=503, detail="mark price refresh cooling down")

        try:
            snapshot = await asyncio.to_thread(_fetch_binance_mark_price_sync, symbol)
        except Exception as exc:
            MARKET_PRICE_FAILURE_UNTIL[cache_key] = now_ts + MARKET_PRICE_ERROR_COOLDOWN_SEC
            fallback = _usable_cached_market_price(cached, now_ts)
            if fallback is not None:
                fallback["cached"] = True
                return JSONResponse(fallback)
            raise HTTPException(status_code=502, detail=f"mark price unavailable: {exc}") from exc

        MARKET_PRICE_FAILURE_UNTIL.pop(cache_key, None)

        payload = {
            "ok": True,
            "symbol": symbol,
            "source": "binance_mark_price",
            "validated": True,
            "price": snapshot["price"],
            "index_price": snapshot["index_price"],
            "last_price": snapshot["last_price"],
            "exchange_ts": snapshot["exchange_ts"],
            "max_deviation_rate": snapshot["max_deviation_rate"],
            "ts": int(now_ts),
        }
        async with MARKET_DATA_LOCK:
            MARKET_DATA_CACHE[cache_key] = {"ts": now_ts, "payload": payload}
        return JSONResponse(payload)


@app.get("/api/panel/state")
async def get_panel_state(request: Request):
    if not _viewer_authorized_http(request):
        raise HTTPException(status_code=401, detail="unauthorized")

    async with STATE_LOCK:
        payload = dict(LATEST_STATE)
    return JSONResponse(payload)


@app.get("/api/panel/token-usage")
async def get_api_token_usage(request: Request):
    # The native Mac app embeds the localhost panel without a Telegram
    # session. This endpoint contains counters/status only (never secrets), so
    # permit loopback reads while keeping tunnel clients behind normal auth.
    if not _is_loopback_request(request) and not _viewer_authorized_http(request):
        raise HTTPException(status_code=401, detail="unauthorized")
    return JSONResponse(_build_api_token_usage())


@app.post("/api/panel/publish")
async def publish_panel_state(request: Request):
    if not _token_valid(_extract_token(request)):
        raise HTTPException(status_code=401, detail="unauthorized")

    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid json: {exc}") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be a json object")

    async with STATE_LOCK:
        LATEST_STATE.clear()
        LATEST_STATE.update(payload)
        if not isinstance(LATEST_STATE.get("ts"), int):
            try:
                LATEST_STATE["ts"] = int(LATEST_STATE.get("ts", 0))
            except Exception:
                LATEST_STATE["ts"] = int(time.time())
        snapshot = dict(LATEST_STATE)

    await _broadcast_state(snapshot)
    return {"ok": True, "ts": snapshot.get("ts", 0)}


@app.websocket("/ws/panel")
async def panel_ws(websocket: WebSocket):
    if not _viewer_authorized_ws(websocket):
        await websocket.close(code=4401)
        return

    await websocket.accept()
    async with CLIENTS_LOCK:
        CLIENTS.add(websocket)

    try:
        async with STATE_LOCK:
            snapshot = dict(LATEST_STATE)
        if snapshot:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "position",
                        "data": snapshot,
                        "server_ts": int(time.time()),
                    },
                    ensure_ascii=False,
                )
            )

        while True:
            await asyncio.sleep(WS_PING_INTERVAL_SEC)
            await websocket.send_text(json.dumps({"type": "ping", "server_ts": int(time.time())}))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        async with CLIENTS_LOCK:
            CLIENTS.discard(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "panel_realtime_server:app",
        host=str(os.getenv("POSITION_PANEL_REALTIME_HOST", "0.0.0.0") or "0.0.0.0"),
        port=_resolve_panel_port(8787),
        log_level=str(os.getenv("POSITION_PANEL_REALTIME_LOG_LEVEL", "info") or "info"),
        access_log=_safe_bool_env("POSITION_PANEL_ACCESS_LOG_ENABLED", False),
    )
