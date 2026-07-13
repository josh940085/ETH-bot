# ===== 移除 gevent（穩定版）=====
import warnings

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+.*")

try:
    from urllib3.exceptions import NotOpenSSLWarning
except Exception:  # pragma: no cover - urllib3 variant fallback
    NotOpenSSLWarning = None

if NotOpenSSLWarning is not None:
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

import requests
import datetime
import time
import base64
import hashlib
import hmac
import math
import subprocess
import sys
import pandas as pd
import numpy as np
import threading
import websocket
import json
import pickle
import os
import re
import html
import sqlite3
try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None
from collections import deque
from pathlib import Path
from zoneinfo import ZoneInfo
from urllib.parse import urlencode, urlparse, urlunparse
import xml.etree.ElementTree as ET

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from mlx_learning import (
    DB_PATH as MLX_LEARNING_DB_PATH,
    EVALUATION_HOURS,
    build_daily_strategy_report,
    build_learning_context as build_mlx_learning_context,
    claim_auto_analysis,
    daily_report_was_sent,
    evaluate_pending as evaluate_mlx_learning,
    learning_stats as get_mlx_learning_stats,
    mark_daily_report_sent,
    predict_replacement_probability as predict_mlx_replacement_probability,
    record_analysis as record_mlx_analysis,
    record_actual_trade_open,
    record_higher_timeframe_context,
    record_sl_review,
    record_strategy_outcome,
    release_auto_analysis,
    sl_review_summary,
    update_actual_trade_outcome,
)
from telegram import (
    REPO_DIR,
    ai_data_path,
    data_path,
    ensure_parent_dir,
    fetch_telegram_commands as tg_fetch_telegram_commands,
    get_follow_mode_enabled as tg_get_follow_mode_enabled,
    get_notification_chat_ids as tg_get_notification_chat_ids,
    inspect_telegram_delivery as tg_inspect_telegram_delivery,
    is_private_chat_id as tg_is_private_chat_id,
    note_telegram_delivery_event as tg_note_telegram_delivery_event,
    normalize_chat_id as tg_normalize_chat_id,
    read_telegram_state_locked as tg_read_telegram_state_locked,
    remove_notification_chat as tg_remove_notification_chat,
    remember_notification_chat as tg_remember_notification_chat,
    resolve_private_chat_id_for_controls as tg_resolve_private_chat_id_for_controls,
    set_follow_mode_enabled as tg_set_follow_mode_enabled,
    toggle_follow_mode_enabled as tg_toggle_follow_mode_enabled,
    update_telegram_state as tg_update_telegram_state,
)

LIVE_RUNTIME_ENABLED = str(os.getenv("ETH_BOT_DISABLE_LIVE", "0") or "0").strip().lower() not in {
    "1",
    "true",
    "yes",
    "on",
}

# ===== Macro / News Engine =====

MACRO_CACHE = {"sp": 0, "nq": 0, "btc": 0, "dxy": 0, "news": 0, "event": 0, "news_list": [], "ts": 0}
NEWS_CACHE = {"news": 0, "event": 0, "news_list": [], "ts": 0}

# ===== AI 新聞分類 =====
NEWS_MODEL_PATH = data_path("news_model.pkl")
NEWS_VECTORIZER_PATH = data_path("news_vectorizer.pkl")
NEWS_MODEL_META_PATH = data_path("news_model_meta.json")
NEWS_PERFORMANCE_LOG = data_path("news_predictions.jsonl")   # 記錄所有預測結果用於評估
NEWS_LEARNING_BUFFER = data_path("learning_buffer.pkl")       # 增量學習緩衝區
NEWS_EVAL_PENDING_PATH = data_path("news_eval_pending.pkl")   # 待市場驗證的新聞預測
NEWS_STATS_CACHE_PATH = data_path("news_stats_cache.json")
BINANCE_HOST_LEARNING_STATE_PATH = data_path("binance_host_learning_state.json")
BINANCE_HOST_LIVE_LEARNING_STATE_PATH = data_path("binance_host_live_learning_state.json")
NEWS_MODEL_VERSION = 2
news_model = None
news_vectorizer = None
PREDICTION_ACCURACY_CACHE = {"cache_key": None, "stats": None}
NEWS_EVAL_PENDING = None

# 增量學習配置
INCREMENTAL_LEARNING_ENABLED = True
MIN_PREDICTIONS_FOR_RETRAIN = 50  # 每50個預測後考慮重新訓練
NEWS_EVAL_HORIZON_SEC = 1800
NEWS_EVAL_MAX_OVERDUE_SEC = 3600
NEWS_EVAL_MIN_MOVE_RATE = 0.0012
NEWS_EVAL_STRONG_MOVE_RATE = 0.0035
NEWS_EVAL_QUEUE_MAX = 400
NEWS_EVAL_PROCESS_INTERVAL_SEC = 15.0
NEWS_RETRAIN_MIN_INTERVAL_SEC = 900.0

HTTP_SESSION = requests.Session()
HTTP_SESSION.headers.update({"User-Agent": "Mozilla/5.0"})
TRADINGVIEW_WS_URL = "wss://data.tradingview.com/socket.io/websocket"
TRADINGVIEW_SYMBOL_MAP = {
    "ETHUSDT": "BINANCE:ETHUSDT.P",
    "BTCUSDT": "BINANCE:BTCUSDT.P",
    "ES1!": "CME_MINI:ES1!",
    "NQ1!": "CME_MINI:NQ1!",
    "DXY": "TVC:DXY",
}
TRADINGVIEW_INTERVAL_MAP = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "12h": "720",
    "1d": "D",
    "1w": "W",
    "1M": "M",
}
KLINE_INTERVAL_MS = {
    "1m": 60 * 1000,
    "3m": 3 * 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "2h": 2 * 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "12h": 12 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
    "1w": 7 * 24 * 60 * 60 * 1000,
    "1M": 31 * 24 * 60 * 60 * 1000,
}
BINANCE_KLINE_SOURCES = (
    ("futures", "https://fapi.binance.com/fapi/v1/klines"),
    ("spot", "https://api.binance.com/api/v3/klines"),
    ("vision_spot", "https://data-api.binance.vision/api/v3/klines"),
)

TRANSLATION_CACHE = {}

# ===== Environment variables / secrets =====
def _load_local_env():
    """簡易讀取 .env（不依賴 python-dotenv）。"""
    try:
        env_path = REPO_DIR / ".env"
        if not env_path.exists():
            return

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key:
                os.environ[key] = value
    except Exception as e:
        print(f"⚠️ .env 載入失敗: {e}")


def _get_required_env(name, default=None, mask=False, warn_if_missing=True):
    value = os.getenv(name, default)
    if value is None or str(value).strip() == "":
        if warn_if_missing:
            print(f"⚠️ 缺少環境變數: {name}")
        return default
    if mask:
        print(f"✅ 已載入 {name}")
    return value


_load_local_env()

ALLOW_BINANCE_MARKET_DATA_FALLBACK = str(os.getenv("ALLOW_BINANCE_MARKET_DATA_FALLBACK", "0") or "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ALLOW_BINANCE_DERIVATIVES_MARKET_DATA = str(os.getenv("ALLOW_BINANCE_DERIVATIVES_MARKET_DATA", "0") or "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
TELEGRAM_TOKEN = _get_required_env("TELEGRAM_TOKEN", "", mask=True)
TELEGRAM_CHAT_ID = _get_required_env("TELEGRAM_CHAT_ID", "", warn_if_missing=False)
DEFAULT_OPENAI_CHAT_MODEL = "gpt-5-mini"
OPENAI_CHAT_MODEL = (os.getenv("OPENAI_CHAT_MODEL", DEFAULT_OPENAI_CHAT_MODEL) or DEFAULT_OPENAI_CHAT_MODEL).strip()
OPENAI_TRANSLATION_MODEL = (os.getenv("OPENAI_TRANSLATION_MODEL", OPENAI_CHAT_MODEL) or OPENAI_CHAT_MODEL).strip()
OPENAI_REASONING_EFFORT = (os.getenv("OPENAI_REASONING_EFFORT", "low") or "low").strip().lower()
OPENAI_PAID_API_ENABLED = str(os.getenv("OPENAI_PAID_API_ENABLED", "0") or "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MLX_AGENT_ENABLED = str(os.getenv("MLX_AGENT_ENABLED", "1") or "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MLX_AGENT_BASE_URL = (
    os.getenv("MLX_AGENT_BASE_URL", "http://127.0.0.1:8080/v1") or "http://127.0.0.1:8080/v1"
).strip().rstrip("/")
MLX_MODEL = (os.getenv("MLX_MODEL", "Qwen/Qwen3-4B-MLX-4bit") or "Qwen/Qwen3-4B-MLX-4bit").strip()
BOT_SUPERVISOR = os.getenv("BOT_SUPERVISOR") == "1"
TELEGRAM_STATE_FILE = data_path(".telegram_state.json")
# ===== Telegram =====
LAST_TELEGRAM_TS = 0
TELEGRAM_POLL_BACKOFF_SEC = 1.0
TELEGRAM_POLL_LAST_ERROR_KEY = ""
TELEGRAM_POLL_LAST_LOG_TS = 0.0
TELEGRAM_POLL_TIMEOUT_BACKOFF_MAX = 12.0
TELEGRAM_GET_UPDATES_TIMEOUT_SEC = 12
TELEGRAM_HTTP_CONNECT_TIMEOUT_SEC = 4
TELEGRAM_HTTP_READ_TIMEOUT_SEC = TELEGRAM_GET_UPDATES_TIMEOUT_SEC + 4

# 若交易所拒絕設定槓桿（如子帳戶限制），快取已知最大槓桿，避免每次開倉都重試
_LEVERAGE_CAP: int = 0  # 0 = 尚未快取
TELEGRAM_POLL_BACKOFF_MAX = 60.0

# ===== Discord（同步通知） =====
DISCORD_WEBHOOK = _get_required_env("DISCORD_WEBHOOK", "", mask=True)
DISCORD_NEWS = _get_required_env("DISCORD_NEWS", "", mask=True)
POSITION_PANEL_FILE = data_path("docs", "position.json")
PENDING_TRAINING_SAMPLE_PATH = data_path("pending_training_sample.json")
MAINTENANCE_REPORT_FILE = data_path("maintenance_latest_report.json")
PROGRAM_LOG_FILE = data_path("..", "logs", "program.log").resolve()
DEFAULT_PAIR = "ETHUSDT"
DEFAULT_LEV = 10
COPY_TRADE_MAX_LEVERAGE = 5
PANEL_DEFAULT_MAX_MARGIN_USDT = 100.0
DEFAULT_MINI_APP_URL = "https://josh940085.github.io/ETH-bot/"
COPY_TRADE_SYMBOL = "ETHUSDT"
COPY_TRADE_MIN_QTY = 0.012
BINANCE_POSITION_SYNC_GRACE_SEC = 90
BINANCE_PROTECTION_ORDER_TYPES = {"STOP", "STOP_MARKET", "TAKE_PROFIT", "TAKE_PROFIT_MARKET"}
BINANCE_PROTECTION_CLIENT_PREFIX = "ethbot_"


def _normalize_mini_app_url(raw_url) -> str:
    url = str(raw_url or "").strip()
    if not url:
        return DEFAULT_MINI_APP_URL

    if "github.io" in url.lower():
        base = url.split("?", 1)[0].split("#", 1)[0].rstrip("/")

        if base.endswith("/docs/index.html"):
            return f"{base[: -len('/docs/index.html')]}/"
        if base.endswith("/docs"):
            return f"{base[: -len('/docs')]}/"
        if base.endswith("/index.html"):
            return f"{base[: -len('/index.html')]}/"

    return url


def _safe_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return float(default)


def _safe_int_env(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, default)).strip())
    except Exception:
        return int(default)


SHORT_TRADE_MAX_HOLD_SEC = max(3600.0, _safe_float_env("SHORT_TRADE_MAX_HOLD_SEC", 24 * 3600))
MLX_AGENT_TIMEOUT_SEC = _safe_int_env("MLX_AGENT_TIMEOUT_SEC", 120)


def _normalize_trade_time_horizon(value) -> str:
    # Live execution is short-term only. Longer-term words may still appear in
    # trend context, but they must not extend position lifetime.
    return "short"


def _infer_trade_time_horizon(final="", regime="", htf=0, mid_trend=0, daily_min_trade=False) -> str:
    """All executable positions are short-term; HTF data is trend context only."""
    return "short"


def _trade_max_hold_sec(horizon=None) -> float:
    return SHORT_TRADE_MAX_HOLD_SEC


def _trade_time_horizon_label(horizon=None) -> str:
    return "短線"


def _safe_float_env_names(names, default: float) -> float:
    if isinstance(names, str):
        names = (names,)
    for name in names:
        raw = os.getenv(name)
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        try:
            return float(text)
        except Exception:
            continue
    return float(default)


def _uses_reasoning_chat_model(model_name: str) -> bool:
    model = str(model_name or "").strip().lower()
    return model.startswith("gpt-5") or model.startswith(("o1", "o3", "o4"))


def _openai_instruction_role(model_name: str) -> str:
    return "developer" if _uses_reasoning_chat_model(model_name) else "system"


def _build_openai_chat_payload(model_name: str, messages, temperature=None):
    model = str(model_name or DEFAULT_OPENAI_CHAT_MODEL).strip() or DEFAULT_OPENAI_CHAT_MODEL
    payload = {
        "model": model,
        "messages": messages,
    }

    if _uses_reasoning_chat_model(model):
        if OPENAI_REASONING_EFFORT:
            payload["reasoning_effort"] = OPENAI_REASONING_EFFORT
    elif temperature is not None:
        payload["temperature"] = temperature

    return payload


def _extract_openai_chat_text(response_json) -> str:
    if not isinstance(response_json, dict):
        return ""
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    if not isinstance(message, dict):
        return ""
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict)
        ).strip()
    return ""


MINI_APP_URL = _normalize_mini_app_url(os.getenv("TELEGRAM_MINI_APP_URL", DEFAULT_MINI_APP_URL))
DISCORD_AUTO_DELETE_HOURS = max(0.0, _safe_float_env("DISCORD_AUTO_DELETE_HOURS", 24.0))
DISCORD_AUTO_DELETE_SEC = int(DISCORD_AUTO_DELETE_HOURS * 3600)
BINANCE_PANEL_ASSET_CACHE_TTL_SEC = max(5.0, _safe_float_env("BINANCE_PANEL_ASSET_CACHE_TTL_SEC", 15.0))
BINANCE_SPOT_MARKET_CACHE_TTL_SEC = max(60.0, _safe_float_env("BINANCE_SPOT_MARKET_CACHE_TTL_SEC", 300.0))
BINANCE_PANEL_ASSET_CACHE = {"ts": 0.0, "snapshot": {}}
BINANCE_PANEL_ASSET_CACHE_LOCK = threading.Lock()
BINANCE_SPOT_MARKET_CACHE = {"ts": 0.0, "graph": {}}
BINANCE_SPOT_MARKET_CACHE_LOCK = threading.Lock()
DERIVATIVES_FLOW_CACHE_TTL_SEC = max(15.0, _safe_float_env("DERIVATIVES_FLOW_CACHE_TTL_SEC", 60.0))
DERIVATIVES_FLOW_CACHE = {"ts": 0.0, "snapshot": {}}
DERIVATIVES_FLOW_CACHE_LOCK = threading.Lock()
FOLLOW_BUTTON_TEXT_DISABLED = "📈 開啟跟單"
FOLLOW_BUTTON_TEXT_ENABLED = "✅ 跟單中（點擊關閉）"
WEBAPP_COMMAND_PREFIX = "__webapp__:"
POSITION_PANEL_REALTIME_BASE_URL = str(os.getenv("POSITION_PANEL_REALTIME_BASE_URL", "") or "").strip().rstrip("/")
POSITION_PANEL_REALTIME_INTERNAL_BASE_URL = str(os.getenv("POSITION_PANEL_REALTIME_INTERNAL_BASE_URL", "") or "").strip().rstrip("/")
POSITION_PANEL_REALTIME_TOKEN = str(os.getenv("POSITION_PANEL_REALTIME_TOKEN", "") or "").strip()
POSITION_PANEL_REALTIME_HEARTBEAT_SEC = max(3.0, _safe_float_env("POSITION_PANEL_REALTIME_HEARTBEAT_SEC", 5.0))
POSITION_PANEL_REALTIME_TIMEOUT_SEC = max(0.5, _safe_float_env("POSITION_PANEL_REALTIME_TIMEOUT_SEC", 2.5))
POSITION_PANEL_SESSION_TTL_SEC = max(300, _safe_int_env("POSITION_PANEL_SESSION_TTL_SEC", 2592000))
POSITION_PANEL_REALTIME_PORT = max(1, _safe_int_env("POSITION_PANEL_REALTIME_PORT", 8787))
PANEL_REALTIME_PUBLIC_URL_FILE = data_path("panel_realtime_public_url.txt")


def _build_panel_realtime_urls(base_url: str, token: str = ""):
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return "", "", ""

    state_url = f"{base}/api/panel/state"
    publish_url = f"{base}/api/panel/publish"

    parsed = urlparse(base)
    if parsed.scheme == "https":
        ws_scheme = "wss"
    elif parsed.scheme == "http":
        ws_scheme = "ws"
    else:
        ws_scheme = parsed.scheme or "wss"
    ws_url = urlunparse((ws_scheme, parsed.netloc, "/ws/panel", "", "", ""))

    return state_url, ws_url, publish_url


def _read_panel_realtime_public_base_url() -> str:
    try:
        raw = PANEL_REALTIME_PUBLIC_URL_FILE.read_text(encoding="utf-8").strip()
        if raw:
            return raw.rstrip("/")
    except Exception:
        pass
    return POSITION_PANEL_REALTIME_BASE_URL


def _panel_realtime_internal_base_url() -> str:
    if POSITION_PANEL_REALTIME_INTERNAL_BASE_URL:
        return POSITION_PANEL_REALTIME_INTERNAL_BASE_URL
    return f"http://127.0.0.1:{POSITION_PANEL_REALTIME_PORT}"


def _current_panel_realtime_urls():
    state_url, ws_url, _ = _build_panel_realtime_urls(_read_panel_realtime_public_base_url())
    _, _, publish_url = _build_panel_realtime_urls(_panel_realtime_internal_base_url())
    return state_url, ws_url, publish_url


def _panel_session_secret() -> str:
    secret = str(POSITION_PANEL_REALTIME_TOKEN or TELEGRAM_TOKEN or "").strip()
    return secret


def _urlsafe_b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _create_panel_session(chat_id) -> str:
    if not _is_private_chat_id(chat_id):
        return ""

    secret = _panel_session_secret()
    if not secret:
        return ""

    try:
        user_id = int(str(chat_id).strip())
    except Exception:
        return ""

    now_ts = int(time.time())
    payload = {
        "v": 1,
        "uid": user_id,
        "iat": now_ts,
        "exp": now_ts + POSITION_PANEL_SESSION_TTL_SEC,
    }
    body = _urlsafe_b64encode(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signature = _urlsafe_b64encode(
        hmac.new(secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).digest()
    )
    return f"{body}.{signature}"


PANEL_REALTIME_PUBLISH_QUEUE = deque(maxlen=1)
PANEL_REALTIME_PUBLISH_EVENT = threading.Event()
PANEL_REALTIME_QUEUE_LOCK = threading.Lock()
PANEL_REALTIME_LAST_SIGNATURE = ""
PANEL_REALTIME_LAST_ENQUEUE_TS = 0.0
PANEL_REALTIME_LAST_ERROR_TS = 0.0
PANEL_REALTIME_WORKER_STARTED = False


def _parse_telegram_state(raw: str) -> dict:
    try:
        payload = json.loads(raw) if str(raw).strip() else {}
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _read_telegram_state_locked() -> dict:
    return tg_read_telegram_state_locked()


def _update_telegram_state(mutator):
    return tg_update_telegram_state(mutator)


def fetch_telegram_commands(last_update_id=None):
    return tg_fetch_telegram_commands(
        last_update_id=last_update_id,
        bot_supervisor=BOT_SUPERVISOR,
        telegram_token=TELEGRAM_TOKEN,
        webapp_command_prefix=WEBAPP_COMMAND_PREFIX,
    )


def _normalize_chat_id(value):
    return tg_normalize_chat_id(value)


def _is_private_chat_id(chat_id) -> bool:
    return tg_is_private_chat_id(chat_id)


def _remember_notification_chat(chat_id):
    tg_remember_notification_chat(chat_id)


def _remove_notification_chat(chat_id):
    tg_remove_notification_chat(chat_id)


def _inspect_telegram_delivery(status_code=None, body="", error=None):
    return tg_inspect_telegram_delivery(status_code=status_code, body=body, error=error)


def _note_telegram_delivery_event(chat_id=None, ok=False, status_code=None, body="", error=None, context=""):
    return tg_note_telegram_delivery_event(
        chat_id=chat_id,
        ok=ok,
        status_code=status_code,
        body=body,
        error=error,
        context=context,
    )


def _is_telegram_chat_not_found(status, body) -> bool:
    if int(_safe_int(status, 0)) != 400:
        return False

    text = str(body or "")
    low = text.lower()
    return "chat not found" in low or '"description":"bad request: chat not found"' in low


def _get_notification_chat_ids():
    return tg_get_notification_chat_ids()


def _get_follow_mode_enabled() -> bool:
    return tg_get_follow_mode_enabled()


def _set_follow_mode_enabled(value: bool):
    tg_set_follow_mode_enabled(value)


def _toggle_follow_mode_enabled() -> bool:
    return tg_toggle_follow_mode_enabled()


def _resolve_private_chat_id_for_controls(chat_id=None):
    return tg_resolve_private_chat_id_for_controls(chat_id)


def _get_follow_button_text() -> str:
    return FOLLOW_BUTTON_TEXT_ENABLED if _get_follow_mode_enabled() else FOLLOW_BUTTON_TEXT_DISABLED


def _build_control_panel_keyboard(chat_id=None):
    follow_text = _get_follow_button_text()
    panel_url = MINI_APP_URL
    state_url, ws_url, _ = _current_panel_realtime_urls()
    panel_session = _create_panel_session(chat_id) if (state_url or ws_url) else ""

    snapshot = {
        "t": int(time.time()),
        "snapshot_ts": int(_safe_int(POSITION_PANEL_STATE.get("ts"), 0)),
        "pair": str(POSITION_PANEL_STATE.get("pair") or DEFAULT_PAIR),
        "lev": int(_safe_int(POSITION_PANEL_STATE.get("lev"), DEFAULT_LEV) or DEFAULT_LEV),
    }
    if state_url:
        snapshot["state_url"] = state_url
    if ws_url:
        snapshot["ws_url"] = ws_url
    if panel_session:
        snapshot["panel_session"] = panel_session

    if state_url or ws_url:
        if not active_trade.get("open"):
            snapshot["restart"] = 1
    elif active_trade.get("open"):
        entry_price = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), 0.0)
        fee_round_trip_rate = _safe_float(POSITION_PANEL_STATE.get("fee_round_trip_rate"), 0.001)
        snapshot.update(
            {
                "total_assets_usdt": round(_safe_float(POSITION_PANEL_STATE.get("binance_total_assets_usdt"), 0.0), 4),
                "spot_assets_usdt": round(_safe_float(POSITION_PANEL_STATE.get("binance_spot_total_assets_usdt"), 0.0), 4),
                "futures_assets_usdt": round(_safe_float(POSITION_PANEL_STATE.get("binance_futures_total_assets_usdt"), 0.0), 4),
                "wallet_balance_usdt": round(_safe_float(POSITION_PANEL_STATE.get("account_wallet_balance_usdt"), 0.0), 4),
                "available_balance_usdt": round(_safe_float(POSITION_PANEL_STATE.get("account_available_balance_usdt"), 0.0), 4),
                "margin_balance_usdt": round(_safe_float(POSITION_PANEL_STATE.get("account_margin_balance_usdt"), 0.0), 4),
                "dir": str(active_trade.get("direction") or "long"),
                "entry": round(entry_price, 4),
                "tp": round(_safe_float(active_trade.get("tp"), 0.0), 4),
                "sl": round(_safe_float(active_trade.get("sl"), 0.0), 4),
                "size": round(max(0.0, _safe_float(active_trade.get("size"), 0.0)) * 100.0, 2),
                "capital_usage_ratio": round(max(0.0, _safe_float(POSITION_PANEL_STATE.get("capital_usage_ratio"), 0.0)), 4),
                "binance_qty": round(_safe_float(POSITION_PANEL_STATE.get("binance_qty"), _get_active_trade_position_qty()), 6),
                "position_notional_usdt": round(_safe_float(POSITION_PANEL_STATE.get("position_notional_usdt"), 0.0), 4),
                "position_margin_usdt": round(_safe_float(POSITION_PANEL_STATE.get("position_margin_usdt"), 0.0), 4),
                "binance_entry_price": round(entry_price, 4),
                "binance_mark_price": round(_safe_float(POSITION_PANEL_STATE.get("binance_mark_price"), 0.0), 4),
                "binance_mark_price_ts": int(_safe_int(POSITION_PANEL_STATE.get("binance_mark_price_ts"), snapshot["snapshot_ts"])),
                "binance_break_even_price": round(
                    entry_price * (1 + fee_round_trip_rate / 2),
                    4,
                )
                if str(active_trade.get("direction") or "long") == "long"
                else round(entry_price * (1 - fee_round_trip_rate / 2), 4),
                "binance_unrealized_pnl_usdt": round(_safe_float(POSITION_PANEL_STATE.get("binance_unrealized_pnl_usdt"), 0.0), 4),
                "binance_unrealized_pnl_ts": int(_safe_int(POSITION_PANEL_STATE.get("binance_unrealized_pnl_ts"), snapshot["snapshot_ts"])),
                "estimated_unrealized_pnl_usdt": round(_safe_float(POSITION_PANEL_STATE.get("estimated_unrealized_pnl_usdt"), 0.0), 4),
            }
        )
    else:
        snapshot["restart"] = 1

    if panel_url:
        sep = "&" if "?" in panel_url else "?"
        panel_url = f"{panel_url}{sep}{urlencode(snapshot)}"

    return {
        "keyboard": [
            [
                {"text": "📱 倉位面板", "web_app": {"url": panel_url}},
            ],
            [
                {"text": follow_text},
                {"text": "⛔ 一鍵平倉"},
            ],
        ],
        "resize_keyboard": True,
        "is_persistent": True,
        "one_time_keyboard": False,
    }


def _format_control_panel_usdt(value):
    return f"{_safe_float(value, 0.0):,.2f}"


def _build_control_panel_text(force_refresh=False):
    try:
        _refresh_position_panel_account_state(force=force_refresh, log_on_error=False)
    except Exception:
        pass

    total_assets = max(0.0, _safe_float(POSITION_PANEL_STATE.get("binance_total_assets_usdt"), 0.0))
    spot_assets = max(0.0, _safe_float(POSITION_PANEL_STATE.get("binance_spot_total_assets_usdt"), 0.0))
    futures_assets = max(
        0.0,
        _safe_float(
            POSITION_PANEL_STATE.get("binance_futures_total_assets_usdt"),
            POSITION_PANEL_STATE.get("account_margin_balance_usdt"),
        ),
    )
    available_balance = max(0.0, _safe_float(POSITION_PANEL_STATE.get("account_available_balance_usdt"), 0.0))

    lines = [
        "⚙️ 交易控制面板",
        f"💼 Binance 總資產: {_format_control_panel_usdt(total_assets)} USDT",
        (
            "現貨 "
            f"{_format_control_panel_usdt(spot_assets)} ｜ 合約 {_format_control_panel_usdt(futures_assets)}"
            f" ｜ 可用 {_format_control_panel_usdt(available_balance)}"
        ),
    ]

    if active_trade.get("open"):
        direction = str(active_trade.get("direction") or "").lower()
        direction_text = "多單" if direction == "long" else ("空單" if direction == "short" else "持倉中")
        entry_price = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), 0.0)
        size_pct = max(0.0, _safe_float(active_trade.get("size"), 0.0)) * 100.0
        lines.append(
            f"📡 目前持倉: {direction_text} ｜ 倉位 {size_pct:.0f}%"
            + (f" ｜ 進場 {entry_price:.2f}" if entry_price > 0 else "")
        )
    else:
        lines.append("📡 目前持倉: 無")

    return "\n".join(lines)


def send_control_panel(chat_id=None):
    target = _resolve_private_chat_id_for_controls(chat_id)
    if not TELEGRAM_TOKEN or not target:
        return

    try:
        _send_telegram_message(
            target,
            _build_control_panel_text(force_refresh=True),
            include_control_panel=True,
        )
    except Exception as e:
        print(f"⚠️ 發送控制面板失敗: {e}")


def _answer_callback_query(callback_id, text):
    if not TELEGRAM_TOKEN or not callback_id:
        return
    try:
        HTTP_SESSION.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery",
            data={"callback_query_id": callback_id, "text": text, "show_alert": True},
            timeout=5,
        )
    except Exception:
        pass


def _edit_control_panel_markup(chat_id, message_id):
    if not TELEGRAM_TOKEN or chat_id is None or message_id in (None, ""):
        return
    try:
        HTTP_SESSION.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageReplyMarkup",
            json={
                "chat_id": chat_id,
                "message_id": int(message_id),
                "reply_markup": _build_control_panel_keyboard(chat_id),
            },
            timeout=5,
        )
    except Exception:
        pass


def _perform_manual_close():
    current = _safe_float(WS_PRICE, _safe_float(active_trade.get("entry"), 0.0))

    # ===== 若實單啟用，向 Binance 送出市價平倉 =====
    binance_close_msg = ""
    if _get_follow_mode_enabled() and _is_real_copy_enabled():
        try:
            direction = str(active_trade.get("direction", "long")).lower()
            qty = _get_active_trade_position_qty()
            dual_side = _is_binance_dual_side_mode()
            position_side = "LONG" if direction == "long" else "SHORT"
            close_side = "SELL" if direction == "long" else "BUY"

            # 1. 取消所有 TP/SL 保護單
            _cancel_existing_binance_protection_orders(close_side, position_side, dual_side)

            # 2. 送出市價平倉單
            if qty > 0:
                order_params = {
                    "symbol": COPY_TRADE_SYMBOL,
                    "side": close_side,
                    "type": "MARKET",
                    "quantity": round(qty, 3),
                }
                if dual_side:
                    order_params["positionSide"] = position_side
                else:
                    order_params["reduceOnly"] = "true"

                _binance_futures_signed_request("POST", "/fapi/v1/order", order_params)
                binance_close_msg = f"✅ Binance 市價平倉已送出 qty={qty:.3f}"
            else:
                binance_close_msg = "⚠️ 無法取得持倉數量，僅取消保護單"
        except Exception as e:
            binance_close_msg = f"⚠️ Binance 平倉失敗: {e}"
            print(f"❌ Binance 一鍵平倉錯誤: {e}")

    record_position_close("MANUAL", current, current, current)
    active_trade["open"] = False
    active_trade["size"] = 0.0
    active_trade["position_qty"] = 0.0
    active_trade["add_count"] = 0
    active_trade["reduce_count"] = 0
    active_trade["quick_reduce_count"] = 0
    active_trade["quick_reduce_ts"] = 0.0
    active_trade["daily_min_size_enforce_ts"] = 0.0
    active_trade["open_time"] = None
    active_trade["tp_sl_adjusted_4h"] = False
    active_trade["time_horizon"] = "short"
    _set_break_even_state(False)
    _clear_pending_training_sample_state()
    sync_position_panel(current)
    return current, binance_close_msg


def _close_trade_at_max_hold(current_price, candle_high=0.0, candle_low=0.0, max_hold_label="24h"):
    """Close an expired position at market; retain local state if Binance close fails."""
    current_price = _safe_float(current_price, _safe_float(active_trade.get("entry"), 0.0))
    direction = str(active_trade.get("direction") or "long").lower()
    label = str(max_hold_label or "時間上限")
    close_msg = ""
    if _get_follow_mode_enabled() and _is_real_copy_enabled():
        try:
            qty = _get_active_trade_position_qty()
            if qty <= 0:
                return False, f"⚠️ {label} 到期但無法取得 Binance 持倉數量，保留本地持倉並重試"
            dual_side = _is_binance_dual_side_mode()
            position_side = "LONG" if direction == "long" else "SHORT"
            close_side = "SELL" if direction == "long" else "BUY"
            _cancel_existing_binance_protection_orders(close_side, position_side, dual_side)
            params = {"symbol": COPY_TRADE_SYMBOL, "side": close_side, "type": "MARKET", "quantity": round(qty, 3)}
            if dual_side:
                params["positionSide"] = position_side
            else:
                params["reduceOnly"] = "true"
            _binance_futures_signed_request("POST", "/fapi/v1/order", params)
            close_msg = f"✅ Binance {label} 到期市價平倉已送出 qty={qty:.3f}"
        except Exception as e:
            return False, f"⚠️ Binance {label} 到期平倉失敗: {e}"

    record_position_close("MAX_HOLD", current_price, candle_high, candle_low)
    _reset_active_trade_state()
    sync_position_panel(current_price)
    return True, close_msg or (f"✅ 模擬持倉已於 {label} 上限結束" if not _is_real_copy_enabled() else f"✅ {label} 到期持倉已結束")


def _handle_control_callback(raw_text, fallback_chat_id=None):
    text = str(raw_text or "")

    if text.startswith(WEBAPP_COMMAND_PREFIX):
        action = text[len(WEBAPP_COMMAND_PREFIX):].strip()
        chat_id = _resolve_private_chat_id_for_controls(fallback_chat_id)

        if action == "manual_close":
            if not active_trade.get("open"):
                if chat_id:
                    _send_telegram_message(chat_id, "目前無持倉", include_control_panel=True)
                return True

            _, binance_msg = _perform_manual_close()
            notice = "⛔ 已執行一鍵平倉（Mini App）"
            if binance_msg:
                notice += f"\n{binance_msg}"
            send_private_telegram(notice, priority=True)
            return True

        return True

    if not text.startswith("__callback__:"):
        return False

    try:
        _, data, callback_id, message_id = text.split(":", 3)
    except ValueError:
        return True

    chat_id = _resolve_private_chat_id_for_controls(fallback_chat_id)

    if data == "toggle_follow":
        enabled = _toggle_follow_mode_enabled()
        _answer_callback_query(callback_id, "✅ 跟單已開啟" if enabled else "⏹️ 跟單已關閉")
        _edit_control_panel_markup(chat_id, message_id)
        return True

    if data == "manual_close":
        if not active_trade.get("open"):
            _answer_callback_query(callback_id, "目前無持倉")
            return True

        _, binance_msg = _perform_manual_close()
        _answer_callback_query(callback_id, "✅ 已送出一鍵平倉")
        _edit_control_panel_markup(chat_id, message_id)
        notice = "⛔ 已執行一鍵平倉（Mini App / 控制面板）"
        if binance_msg:
            notice += f"\n{binance_msg}"
        send_private_telegram(notice, priority=True)
        return True

    return True


def _write_json_atomic(path: Path, payload: dict):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"⚠️ 寫入 {path.name} 失敗: {e}")


def _write_pickle_atomic(path, payload):
    target = Path(path)
    try:
        ensure_parent_dir(target)
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        with tmp_path.open("wb") as f:
            pickle.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
    except Exception as e:
        print(f"⚠️ 寫入 {target.name} 失敗: {e}")


def _get_active_trade_position_qty():
    qty = max(0.0, _safe_float(active_trade.get("position_qty"), 0.0))
    if qty > 0:
        return qty

    qty = max(0.0, _safe_float(POSITION_PANEL_STATE.get("binance_qty"), 0.0))
    if qty > 0:
        return qty

    if _get_follow_mode_enabled() and _is_real_copy_enabled():
        try:
            rows = _binance_futures_signed_get("/fapi/v2/positionRisk", {"symbol": COPY_TRADE_SYMBOL})
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict):
                    continue
                if str(row.get("symbol") or "").upper() != COPY_TRADE_SYMBOL:
                    continue
                position_amt = abs(_safe_float(row.get("positionAmt"), 0.0))
                if position_amt > 0:
                    return position_amt
        except Exception:
            pass

    return 0.0


def _read_pending_training_sample_state_raw():
    if not PENDING_TRAINING_SAMPLE_PATH.exists():
        return {}

    try:
        raw = json.loads(PENDING_TRAINING_SAMPLE_PATH.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _clear_pending_training_sample_state():
    try:
        if PENDING_TRAINING_SAMPLE_PATH.exists():
            PENDING_TRAINING_SAMPLE_PATH.unlink()
    except Exception as e:
        print(f"⚠️ 清除待學習樣本失敗: {e}")


def _load_position_panel_state():
    if not POSITION_PANEL_FILE.exists():
        return {
            "last_close_reason": "",
            "last_close_price": 0.0,
            "last_close_ts": 0,
            "last_close_candle_high": 0.0,
            "last_close_candle_low": 0.0,
            "close_hits": [],
            "latest_news": [],
            "size": 0.0,
            "size_ratio": 0.0,
            "capital_usage_ratio": 0.0,
            "position_notional_usdt": 0.0,
            "position_margin_usdt": 0.0,
            "binance_qty": 0.0,
            "binance_mark_price": 0.0,
            "binance_mark_price_ts": 0,
            "binance_unrealized_pnl_usdt": 0.0,
            "binance_unrealized_pnl_ts": 0,
            "estimated_unrealized_pnl_usdt": 0.0,
            "account_available_balance_usdt": 0.0,
            "account_wallet_balance_usdt": 0.0,
            "account_margin_balance_usdt": 0.0,
            "binance_spot_total_assets_usdt": 0.0,
            "binance_futures_total_assets_usdt": 0.0,
            "binance_total_assets_usdt": 0.0,
            "funding_rate": 0.0,
            "funding_next_ts": 0,
            "fee_round_trip_rate": 0.001,
            "break_even_active": False,
            "break_even_target": 0.0,
            "break_even_ts": 0,
            "time_horizon": "short",
            "max_hold_sec": SHORT_TRADE_MAX_HOLD_SEC,
            "daily_trade_date": "",
            "daily_trade_opened": False,
            "daily_trade_source": "",
            "daily_trade_opened_ts": 0,
        }

    try:
        raw = json.loads(POSITION_PANEL_FILE.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("position.json is not an object")
    except Exception:
        return {
            "last_close_reason": "",
            "last_close_price": 0.0,
            "last_close_ts": 0,
            "last_close_candle_high": 0.0,
            "last_close_candle_low": 0.0,
            "close_hits": [],
            "latest_news": [],
            "size": 0.0,
            "size_ratio": 0.0,
            "capital_usage_ratio": 0.0,
            "position_notional_usdt": 0.0,
            "position_margin_usdt": 0.0,
            "binance_qty": 0.0,
            "binance_mark_price": 0.0,
            "binance_mark_price_ts": 0,
            "binance_unrealized_pnl_usdt": 0.0,
            "binance_unrealized_pnl_ts": 0,
            "estimated_unrealized_pnl_usdt": 0.0,
            "account_available_balance_usdt": 0.0,
            "account_wallet_balance_usdt": 0.0,
            "account_margin_balance_usdt": 0.0,
            "binance_spot_total_assets_usdt": 0.0,
            "binance_futures_total_assets_usdt": 0.0,
            "binance_total_assets_usdt": 0.0,
            "funding_rate": 0.0,
            "funding_next_ts": 0,
            "fee_round_trip_rate": 0.001,
            "break_even_active": False,
            "break_even_target": 0.0,
            "break_even_ts": 0,
            "time_horizon": "short",
            "max_hold_sec": SHORT_TRADE_MAX_HOLD_SEC,
            "daily_trade_date": "",
            "daily_trade_opened": False,
            "daily_trade_source": "",
            "daily_trade_opened_ts": 0,
        }

    close_hits = raw.get("close_hits")
    if not isinstance(close_hits, list):
        close_hits = []
    latest_news = raw.get("latest_news")
    if not isinstance(latest_news, list):
        latest_news = []

    return {
        "last_close_reason": str(raw.get("last_close_reason", "") or "").upper(),
        "last_close_price": float(raw.get("last_close_price", 0.0) or 0.0),
        "last_close_ts": int(raw.get("last_close_ts", 0) or 0),
        "last_close_candle_high": float(raw.get("last_close_candle_high", 0.0) or 0.0),
        "last_close_candle_low": float(raw.get("last_close_candle_low", 0.0) or 0.0),
        "close_hits": close_hits[:10],
        "latest_news": latest_news[:8],
        "size": float(raw.get("size", raw.get("capital_usage_ratio", 0.0)) or 0.0),
        "size_ratio": float(raw.get("size_ratio", raw.get("size", 0.0)) or 0.0),
        "capital_usage_ratio": float(raw.get("capital_usage_ratio", raw.get("size", 0.0)) or 0.0),
        "position_notional_usdt": float(raw.get("position_notional_usdt", 0.0) or 0.0),
        "position_margin_usdt": float(raw.get("position_margin_usdt", 0.0) or 0.0),
        "binance_qty": float(raw.get("binance_qty", 0.0) or 0.0),
        "binance_mark_price": float(raw.get("binance_mark_price", 0.0) or 0.0),
        "binance_mark_price_ts": int(raw.get("binance_mark_price_ts", raw.get("ts", 0)) or 0),
        "binance_unrealized_pnl_usdt": float(raw.get("binance_unrealized_pnl_usdt", 0.0) or 0.0),
        "binance_unrealized_pnl_ts": int(raw.get("binance_unrealized_pnl_ts", raw.get("ts", 0)) or 0),
        "estimated_unrealized_pnl_usdt": float(raw.get("estimated_unrealized_pnl_usdt", 0.0) or 0.0),
        "account_available_balance_usdt": float(raw.get("account_available_balance_usdt", 0.0) or 0.0),
        "account_wallet_balance_usdt": float(raw.get("account_wallet_balance_usdt", 0.0) or 0.0),
        "account_margin_balance_usdt": float(raw.get("account_margin_balance_usdt", 0.0) or 0.0),
        "binance_spot_total_assets_usdt": float(raw.get("binance_spot_total_assets_usdt", 0.0) or 0.0),
        "binance_futures_total_assets_usdt": float(raw.get("binance_futures_total_assets_usdt", raw.get("account_margin_balance_usdt", 0.0)) or 0.0),
        "binance_total_assets_usdt": float(raw.get("binance_total_assets_usdt", raw.get("account_margin_balance_usdt", 0.0)) or 0.0),
        "funding_rate": float(raw.get("funding_rate", 0.0) or 0.0),
        "funding_next_ts": int(raw.get("funding_next_ts", 0) or 0),
        "fee_round_trip_rate": float(raw.get("fee_round_trip_rate", 0.001) or 0.001),
        "break_even_active": bool(raw.get("break_even_active", False)),
        "break_even_target": float(raw.get("break_even_target", 0.0) or 0.0),
        "break_even_ts": int(raw.get("break_even_ts", 0) or 0),
        "time_horizon": _normalize_trade_time_horizon(raw.get("time_horizon")),
        "max_hold_sec": float(raw.get("max_hold_sec", _trade_max_hold_sec(raw.get("time_horizon"))) or 0.0),
        "daily_trade_date": str(raw.get("daily_trade_date") or ""),
        "daily_trade_opened": bool(raw.get("daily_trade_opened", False)),
        "daily_trade_source": str(raw.get("daily_trade_source") or ""),
        "daily_trade_opened_ts": int(raw.get("daily_trade_opened_ts", 0) or 0),
    }


POSITION_PANEL_STATE = _load_position_panel_state()


def _normalize_finance_terms_zh(text):
    """將翻譯結果統一為較常見的股市/總經術語。"""
    s = str(text or "").strip()
    if not s:
        return s

    # 英文財經詞彙（先做，避免混雜）
    en_map = [
        (r"\bWall\s*St\.?\b", "華爾街"),
        (r"\bconsumer prices?\b", "消費者物價"),
        (r"\binflation\b", "通膨"),
        (r"\bcore inflation\b", "核心通膨"),
        (r"\bTreasury yields?\b", "美債殖利率"),
        (r"\bfutures?\b", "期貨"),
        (r"\bstock rating\b", "投資評級"),
        (r"\brating\b", "評級"),
        (r"\bprice target\b", "目標價"),
        (r"\breiterates?\b", "重申"),
        (r"\bcuts?\b", "下調"),
        (r"\braises?\b", "上調"),
        (r"\bmaintains?\b", "維持"),
        (r"\binitiates?\b", "首次覆蓋"),
        (r"\bheadwinds?\b", "逆風"),
        (r"\bupgrades?\b", "上調評級"),
        (r"\bdowngrades?\b", "下調評級"),
        (r"\bearnings\b", "財報"),
        (r"\bguidance\b", "財測"),
        (r"\boutlook\b", "展望"),
        (r"\bestimate\b", "預估"),
        (r"\bestimates\b", "預估"),
        (r"\bforecast\b", "預測"),
        (r"\bforecasts\b", "預測"),
        (r"\brevenue\b", "營收"),
        (r"\bprofit\b", "獲利"),
        (r"\bmargin\b", "利潤率"),
        (r"\bEPS\b", "每股盈餘"),
        (r"\bvaluation\b", "估值"),
        (r"\bshares?\b", "股價"),
    ]

    for pat, rep in en_map:
        s = re.sub(pat, rep, s, flags=re.I)

    # 中文術語正規化（將口語或陸式寫法統一）
    zh_map = [
        ("股价", "股價"),
        ("评级", "評級"),
        ("投資評等", "投資評級"),
        ("評級重申", "重申評級"),
        ("重申中立評級", "重申中立評級"),
        ("重申減持評級", "重申減碼評級"),
        ("重申增持評級", "重申加碼評級"),
        ("重申跑贏大盤評級", "重申優於大盤評級"),
        ("重申買入評級", "重申買進評級"),
        ("重申买入评级", "重申買進評級"),
        ("買入評級", "買進評級"),
        ("买入评级", "買進評級"),
        ("持有評級", "中立評級"),
        ("買入", "買進"),
        ("卖出", "賣出"),
        ("增持", "加碼"),
        ("减持", "減碼"),
        ("下调", "下調"),
        ("上调", "上調"),
        ("目标价", "目標價"),
        ("目标价格", "目標價"),
        ("目標價格", "目標價"),
        ("調升目標價", "上調目標價"),
        ("調降目標價", "下調目標價"),
        ("上修", "上調"),
        ("下修", "下調"),
        ("通货膨胀", "通膨"),
        ("华尔街", "華爾街"),
        ("美国", "美國"),
        ("美联储", "聯準會"),
        ("利率决议", "利率決議"),
        ("非农", "非農"),
        ("收益率", "殖利率"),
        ("業績指引", "財測"),
        ("指引", "財測"),
        ("營業收入", "營收"),
        ("每股收益", "每股盈餘"),
        ("每股盈利", "每股盈餘"),
    ]

    for old, new in zh_map:
        s = s.replace(old, new)

    s = re.sub(r"\s+", " ", s).strip()
    return s


def _google_translate_to_zh(text):
    """使用公開 Google 翻譯端點作為第二層備援。"""
    try:
        src = str(text or "").strip()
        if not src:
            return ""

        res = HTTP_SESSION.get(
            "https://translate.googleapis.com/translate_a/single",
            params={
                "client": "gtx",
                "sl": "auto",
                "tl": "zh-TW",
                "dt": "t",
                "q": src[:350],
            },
            timeout=6,
        )
        data = res.json()
        if not isinstance(data, list) or not data:
            return ""

        parts = data[0] if isinstance(data[0], list) else []
        text_parts = []
        for item in parts:
            if isinstance(item, list) and item:
                seg = str(item[0] or "").strip()
                if seg:
                    text_parts.append(seg)

        out = "".join(text_parts).strip()
        return _normalize_finance_terms_zh(out)
    except Exception:
        return ""


def _local_translate_news_fallback(text):
    """當 API 翻譯失敗時，使用本地詞彙表做可讀的中文轉換。"""
    s = str(text or "").strip()
    if not s:
        return s

    if re.search(r"[\u4e00-\u9fff]", s):
        return s

    table = [
        ("Federal Reserve", "聯準會"),
        ("interest rate", "利率"),
        ("rate cut", "降息"),
        ("rate hike", "升息"),
        ("inflation", "通膨"),
        ("nonfarm payrolls", "非農就業"),
        ("unemployment", "失業率"),
        ("Treasury", "美債"),
        ("yield", "殖利率"),
        ("US Dollar", "美元"),
        ("dollar index", "美元指數"),
        ("Bitcoin", "比特幣"),
        ("Ethereum", "以太幣"),
        ("crypto", "加密貨幣"),
        ("ETF", "ETF"),
        ("approved", "獲批准"),
        ("approval", "批准"),
        ("inflow", "資金流入"),
        ("outflow", "資金流出"),
        ("rally", "上漲"),
        ("surge", "飆升"),
        ("plunge", "大跌"),
        ("drop", "下跌"),
        ("sell-off", "拋售"),
        ("lawsuit", "訴訟"),
        ("hack", "駭客攻擊"),
        ("exchange", "交易所"),
        ("listing", "上架"),
        ("delist", "下架"),
        ("partnership", "合作"),
        ("tariff", "關稅"),
        ("ceasefire", "停火"),
        ("sanction", "制裁"),
    ]

    out = s
    for en, zh in table:
        out = re.sub(re.escape(en), zh, out, flags=re.I)

    if not re.search(r"[\u4e00-\u9fff]", out):
        return f"（原文）{s}"
    return _normalize_finance_terms_zh(out)



def normalize_news_text(text):
    try:
        text = html.unescape(str(text))
        text = text.replace("\\u002F", "/").replace("\\/", "/")
        text = text.replace("\\n", " ").replace("\\t", " ")
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\\u[0-9a-fA-F]{4}", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except:
        return str(text)


def _prepare_news_text_for_model(text):
    """將新聞文字轉成更適合模型學習的穩定格式。"""
    prepared = normalize_news_text(text)
    prepared = re.sub(r"https?://\S+", " ", prepared)
    prepared = prepared.replace("&amp;", " and ")
    prepared = re.sub(r"[$#]([A-Za-z]{2,12})", r" \1 ", prepared)
    prepared = re.sub(r"[^\w\u4e00-\u9fff%./:+-]+", " ", prepared)
    prepared = prepared.lower()
    prepared = re.sub(r"\s+", " ", prepared).strip()
    return prepared


def _is_crypto_relevant_news(text: str) -> bool:
    """判斷新聞是否與 BTC/ETH、加密貨幣市場或可能影響幣價的宏觀事件有關。
    只有相關新聞才會影響 news_bias 和進入 AI 訓練資料。"""
    low = str(text or "").lower()

    # 白名單關鍵字：含任一詞即視為相關
    CRYPTO_KEYWORDS = [
        # 幣種
        "bitcoin", "btc", "ethereum", "eth", "ether",
        "crypto", "cryptocurrency", "cryptocurrencies",
        # 機構與監管
        "sec", "cftc", "finra", "binance", "coinbase", "kraken",
        "ftx", "grayscale", "blackrock", "fidelity", "microstrategy",
        "bitfinex", "okx", "bybit", "deribit",
        # 產品與概念
        "defi", "nft", "staking", "stablecoin", "usdt", "usdc",
        "altcoin", "altcoins", "blockchain", "web3", "layer 2", "layer2",
        "lightning network", "proof of work", "proof of stake",
        "smart contract", "dex", "cefi", "dao",
        "spot etf", "crypto etf", "bitcoin etf", "ethereum etf",
        # 行情詞
        "crypto market", "digital asset", "digital assets",
        "on-chain", "hash rate", "mempool", "halving",
        "whale", "cold wallet", "hot wallet", "exchange hack",
        # 宏觀-加密連動
        "fed rate crypto", "btc halving", "bitcoin halving",
        # 宏觀經濟（影響風險資產）
        "fed", "fomc", "powell", "federal reserve",
        "cpi", "inflation", "interest rate", "rate hike", "rate cut",
        "gdp", "unemployment", "jobs report", "nonfarm", "non-farm",
        "tariff", "trade war", "trade deal", "sanctions",
        "recession", "stagflation", "liquidity",
        # 地緣政治與市場衝擊
        "war", "ceasefire", "conflict", "missile", "nuclear",
        "oil price", "crude oil", "energy crisis",
        "stock market", "nasdaq", "s&p 500", "dow jones", "risk-off", "risk off",
        "bank collapse", "bank run", "banking crisis", "default",
        "dollar", "dxy", "yen", "yuan", "usd",
    ]

    for kw in CRYPTO_KEYWORDS:
        if kw in low:
            return True
    return False


def _sanitize_news_label(label):
    try:
        return max(-2, min(2, int(label)))
    except Exception:
        return None


def _read_json_file(path, default):
    try:
        if not Path(path).exists():
            return default
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, type(default)) else default
    except Exception:
        return default


def _write_json_file(path, payload):
    try:
        _write_json_atomic(Path(path), payload)
    except Exception:
        pass


def _binance_host_learning_enabled():
    return _is_truthy(os.getenv("BINANCE_HOST_LEARNING_ENABLED", "1"))


def _binance_host_live_learning_enabled():
    return _is_truthy(os.getenv("BINANCE_HOST_LIVE_LEARNING_ENABLED", "1"))


def _binance_host_learning_sources():
    raw_urls = str(
        os.getenv(
            "BINANCE_HOST_LEARNING_URLS",
            "https://www.binance.com/zh-TC/square/profile/square-creator-885626428",
        )
        or ""
    )
    urls = [url.strip() for url in re.split(r"[,;\n]+", raw_urls) if url.strip()]
    texts = []
    inline_text = str(os.getenv("BINANCE_HOST_LEARNING_TEXT", "") or "").strip()
    if inline_text:
        texts.append(("env:BINANCE_HOST_LEARNING_TEXT", inline_text))

    text_path = str(os.getenv("BINANCE_HOST_LEARNING_TEXT_PATH", "") or "").strip()
    if text_path:
        try:
            path = Path(text_path).expanduser()
            if path.exists() and path.is_file():
                texts.append((str(path), path.read_text(encoding="utf-8", errors="ignore")))
        except Exception as exc:
            print(f"⚠️ 藍歌學習文字檔讀取失敗: {exc}")

    return urls, texts


def _binance_host_live_learning_sources():
    raw_urls = str(os.getenv("BINANCE_HOST_LIVE_TRANSCRIPT_URLS", "") or "")
    urls = [url.strip() for url in re.split(r"[,;\n]+", raw_urls) if url.strip()]
    texts = []
    inline_text = str(os.getenv("BINANCE_HOST_LIVE_TEXT", "") or "").strip()
    if inline_text:
        texts.append(("env:BINANCE_HOST_LIVE_TEXT", inline_text))

    text_path = str(os.getenv("BINANCE_HOST_LIVE_TEXT_PATH", "") or "").strip()
    if text_path:
        try:
            path = Path(text_path).expanduser()
            if path.exists() and path.is_file():
                texts.append((str(path), path.read_text(encoding="utf-8", errors="ignore")))
        except Exception as exc:
            print(f"⚠️ 藍歌直播學習文字檔讀取失敗: {exc}")

    archive_dir = str(
        os.getenv(
            "BINANCE_HOST_LIVE_ARCHIVE_DIR",
            str(data_path("binance_host_live_archives")),
        )
        or ""
    ).strip()
    if archive_dir:
        try:
            root = Path(archive_dir).expanduser()
            if root.exists() and root.is_dir():
                allowed_ext = {
                    ".txt",
                    ".md",
                    ".json",
                    ".jsonl",
                    ".srt",
                    ".vtt",
                    ".csv",
                    ".tsv",
                }
                max_files = max(1, _safe_int(os.getenv("BINANCE_HOST_LIVE_ARCHIVE_MAX_FILES", 500), 500))
                max_file_bytes = max(
                    10_000,
                    _safe_int(os.getenv("BINANCE_HOST_LIVE_ARCHIVE_MAX_FILE_BYTES", 2_000_000), 2_000_000),
                )
                files = [
                    path
                    for path in root.rglob("*")
                    if path.is_file() and path.suffix.lower() in allowed_ext
                ]
                for path in sorted(files, key=lambda item: str(item))[:max_files]:
                    try:
                        if path.stat().st_size > max_file_bytes:
                            print(f"⚠️ 略過過大的藍歌直播檔: {path}")
                            continue
                        texts.append((str(path), path.read_text(encoding="utf-8", errors="ignore")))
                    except Exception as exc:
                        print(f"⚠️ 藍歌直播檔讀取失敗 {path}: {exc}")
        except Exception as exc:
            print(f"⚠️ 藍歌直播檔案庫讀取失敗: {exc}")

    return urls, texts


def _fetch_binance_host_source(url):
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
                ),
                "Accept-Language": "zh-TW,zh-Hant;q=0.9,zh-CN;q=0.8,en;q=0.6",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
            timeout=max(3, _safe_int(os.getenv("BINANCE_HOST_LEARNING_TIMEOUT_SEC", 8), 8)),
        )
        if response.status_code >= 400:
            return ""
        return response.text or ""
    except Exception as exc:
        print(f"⚠️ 藍歌 Binance Square 來源讀取失敗: {exc}")
        return ""


def _fetch_binance_host_live_source(url):
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
                ),
                "Accept-Language": "zh-TW,zh-Hant;q=0.9,zh-CN;q=0.8,en;q=0.6",
                "Accept": "text/plain,text/html,application/json,*/*;q=0.8",
            },
            timeout=max(3, _safe_int(os.getenv("BINANCE_HOST_LIVE_TIMEOUT_SEC", 8), 8)),
        )
        if response.status_code >= 400:
            return ""
        return response.text or ""
    except Exception as exc:
        print(f"⚠️ 藍歌直播逐字稿來源讀取失敗: {exc}")
        return ""


def _html_to_learning_text(raw_text):
    text = html.unescape(str(raw_text or ""))
    text = re.sub(r"(?is)<script[^>]*>.*?</script>", "\n", text)
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", "\n", text)
    text = re.sub(r"<[^>]+>", "\n", text)
    text = text.replace("\\n", "\n").replace("\\t", " ")
    text = re.sub(r"\r+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_binance_host_posts(raw_text, host_name="蓝歌", limit=8):
    text = _html_to_learning_text(raw_text)
    if not text:
        return []

    separator = os.getenv("BINANCE_HOST_LEARNING_POST_SEPARATOR", "\n---\n")
    chunks = []
    if separator and separator in text:
        chunks = [part.strip() for part in text.split(separator)]
    else:
        pattern = rf"(?:^|\n)\s*{re.escape(host_name)}\s*\n\s*[·•]?\s*\n?\s*[-—]*\s*\n"
        parts = re.split(pattern, text)
        chunks = [part.strip() for part in parts[1:]]
        if not chunks:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            current = []
            for line in lines:
                if line == host_name and current:
                    chunks.append("\n".join(current))
                    current = []
                elif _is_crypto_relevant_news(line) or re.search(
                    r"[$#](ETH|BTC|SOL)|以太|比特|做多|做空|空单|多单",
                    line,
                    re.I,
                ):
                    current.append(line)
            if current:
                chunks.append("\n".join(current))

    posts = []
    seen = set()
    for chunk in chunks:
        clean = normalize_news_text(chunk)
        clean = re.sub(r"\b(Image|Log in|Sign up|登入|註冊|查看原文)\b", " ", clean, flags=re.I)
        clean = re.sub(r"\s+", " ", clean).strip()
        if len(clean) < 24:
            continue
        if not re.search(r"[$#](ETH|BTC|SOL)|ETH|BTC|以太|比特|做多|做空|空单|多单|牛市|熊市|跌|漲|涨", clean, re.I):
            continue
        clean = clean[:1600]
        digest = hashlib.sha256(clean.encode("utf-8", errors="ignore")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        posts.append({"hash": digest, "text": clean})
        if len(posts) >= max(1, _safe_int(limit, 8)):
            break
    return posts


def _extract_binance_host_live_segments(raw_text, host_name="蓝歌", limit=8):
    text = _html_to_learning_text(raw_text)
    if not text:
        return []

    separator = os.getenv("BINANCE_HOST_LIVE_SEGMENT_SEPARATOR", "\n---\n")
    if separator and separator in text:
        chunks = [part.strip() for part in text.split(separator)]
    else:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        chunks = []
        current = []
        for line in lines:
            clean_line = re.sub(r"^\s*(?:\[[^\]]+\]|\d{1,2}:\d{2}(?::\d{2})?)\s*", "", line).strip()
            if not clean_line:
                continue
            current.append(clean_line)
            joined = " ".join(current)
            if len(joined) >= 260 or re.search(r"[。！？!?]\s*$", clean_line):
                chunks.append(joined)
                current = []
        if current:
            chunks.append(" ".join(current))

    segments = []
    seen = set()
    for chunk in chunks:
        clean = normalize_news_text(chunk)
        clean = re.sub(r"^\s*(?:\[[^\]]+\]|\d{1,2}:\d{2}(?::\d{2})?)\s*", "", clean).strip()
        clean = re.sub(r"\b\d+\s+\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\b", " ", clean)
        clean = re.sub(r"\b\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\b", " ", clean)
        clean = re.sub(r"\S*\s*-->\s*\S*", " ", clean)
        clean = re.sub(r"^\s*\d+\s+", "", clean).strip()
        clean = re.sub(
            r"\b(Live|Replay|Subscribe|Follow|Share|Like|登入|註冊|訂閱|分享|直播回放)\b",
            " ",
            clean,
            flags=re.I,
        )
        clean = re.sub(r"\s+", " ", clean).strip()
        if len(clean) < 24:
            continue
        if not re.search(
            r"[$#](ETH|BTC|SOL)|ETH|BTC|以太|比特|大餅|大饼|做多|做空|空单|多单|牛市|熊市|支撐|支撑|壓力|压力|止盈|止損|止损|停損|停损|倉位|仓位|槓桿|杠杆|進場|进场|出場|出场|突破|跌破|洗盤|洗盘|回踩|回測|回测|跌|漲|涨",
            clean,
            re.I,
        ):
            continue
        max_chars = max(
            1200,
            _safe_int(os.getenv("BINANCE_HOST_LIVE_SEGMENT_MAX_CHARS", 4000), 4000),
        )
        clean = clean[:max_chars]
        digest = hashlib.sha256(clean.encode("utf-8", errors="ignore")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        segments.append({"hash": digest, "text": clean})
        max_items = _safe_int(limit, 8)
        if max_items > 0 and len(segments) >= max_items:
            break
    return segments


def _extract_binance_host_live_document(raw_text, source="", limit_chars=None):
    text = _html_to_learning_text(raw_text)
    if not text:
        return None
    lines = []
    for raw_line in text.splitlines():
        line = normalize_news_text(raw_line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if re.fullmatch(r"-{3,}", line):
            lines.append("---")
            continue
        if re.search(
            r"(方向|品種|日期|回放|主題|長度|聆聽|思路|交易邏輯|風控|支撐|壓力|止盈|止損|停損|進場|出場|突破|跌破|回踩|補倉|減倉|ETH|BTC|以太|比特|多軍|空軍|做多|做空|等待)",
            line,
            re.I,
        ):
            lines.append(line)
    if not lines:
        return None

    max_chars = max(
        2000,
        _safe_int(
            limit_chars
            if limit_chars is not None
            else os.getenv("BINANCE_HOST_LIVE_DOCUMENT_MAX_CHARS", 12000),
            12000,
        ),
    )
    document_text = "\n".join(lines)
    document_text = re.sub(r"\n{3,}", "\n\n", document_text).strip()[:max_chars]
    if len(document_text) < 80:
        return None
    source_key = str(source or "")[:240]
    digest = hashlib.sha256(
        f"live-document:{source_key}:{document_text}".encode("utf-8", errors="ignore")
    ).hexdigest()
    return {
        "hash": digest,
        "text": document_text,
        "kind": "full_live_archive_document",
        "line_count": len(lines),
        "char_count": len(document_text),
    }


def _infer_binance_host_direction(text):
    text = str(text or "")
    primary_parts = []
    title_parts = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^(?:主題|標題|标题)\s*[:：]", line):
            title_parts.append(line)
            continue
        if re.match(r"^(?:直播內容|直播内容|逐字稿|Transcript|思路|交易邏輯|交易逻辑|風控|风控)\s*[:：]", line, flags=re.I):
            primary_parts.append(line)
            continue
        if not re.match(r"^(?:方向|品種|品种|日期|回放|長度|长度|聆聽|收聽|来源|來源)\s*[:：]", line):
            primary_parts.append(line)
    primary_text = "\n".join(primary_parts).strip()
    scoring_text = primary_text or text
    declared = re.search(r"方向\s*[:：]\s*(做多|多方|看多|做空|空方|看空|等待|觀望|观望|中性)", text)
    if declared and not primary_text:
        value = declared.group(1)
        if value in {"做多", "多方", "看多"}:
            return "long", 1
        if value in {"做空", "空方", "看空"}:
            return "short", 1
        return "neutral", 0
    bullish = (
        "做多", "多单", "多單", "看多", "牛市", "上漲", "上涨", "拉升", "反弹",
        "反彈", "抄底", "帶上", "带上", "突破", "上去", "漲到", "涨到",
    )
    bearish = (
        "做空", "空单", "空單", "空軍", "空军", "看空", "熊市", "下跌", "跌破",
        "加速下跌", "大跌", "回落", "洗盤", "洗盘", "抗單", "抗单", "跌到",
    )
    bull_score = sum(scoring_text.count(word) for word in bullish)
    bear_score = sum(scoring_text.count(word) for word in bearish)
    if primary_text and title_parts:
        title_text = "\n".join(title_parts)
        bull_score += sum(title_text.count(word) for word in bullish) * 0.25
        bear_score += sum(title_text.count(word) for word in bearish) * 0.25
    bear_score -= len(re.findall(r"(?:沒有|未|沒|尚未)\s*(?:跌破|下跌|大跌|回落)", scoring_text))
    bull_score -= len(re.findall(r"(?:沒有|未|沒|尚未)\s*(?:突破|上漲|上涨|拉升|反彈|反弹)", scoring_text))
    bull_score = max(0, bull_score)
    bear_score = max(0, bear_score)
    if re.search(r"跌\s*\d+|跌\d+|下跌\s*\d+", scoring_text):
        bear_score += 2
    if re.search(r"上\s*\d+|漲\s*\d+|涨\s*\d+", scoring_text):
        bull_score += 1
    if declared and primary_text:
        value = declared.group(1)
        if value in {"做多", "多方", "看多"}:
            bull_score += 0.5
        elif value in {"做空", "空方", "看空"}:
            bear_score += 0.5

    if bull_score > bear_score:
        return "long", int(max(1, round(bull_score - bear_score)))
    if bear_score > bull_score:
        return "short", int(max(1, round(bear_score - bull_score)))
    return "neutral", 0


def _binance_host_content_ts(text, now_ts=None):
    text = str(text or "")
    now_dt = datetime.datetime.fromtimestamp(
        _safe_float(now_ts, time.time()), ZoneInfo("Asia/Taipei")
    )
    hours_match = re.search(r"日期\s*[:：]\s*(\d+)\s*(?:小時|小时)", text)
    if hours_match:
        return (now_dt - datetime.timedelta(hours=_safe_int(hours_match.group(1), 0))).timestamp()
    minutes_match = re.search(r"日期\s*[:：]\s*(\d+)\s*(?:分鐘|分钟|分)", text)
    if minutes_match:
        return (now_dt - datetime.timedelta(minutes=_safe_int(minutes_match.group(1), 0))).timestamp()

    explicit = re.search(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日", text)
    if explicit:
        try:
            return datetime.datetime(
                _safe_int(explicit.group(1), now_dt.year),
                _safe_int(explicit.group(2), 1),
                _safe_int(explicit.group(3), 1),
                tzinfo=ZoneInfo("Asia/Taipei"),
            ).timestamp()
        except Exception:
            pass

    month_day = re.search(r"(?<!年)(\d{1,2})月\s*(\d{1,2})日", text)
    if month_day:
        month = _safe_int(month_day.group(1), now_dt.month)
        day = _safe_int(month_day.group(2), now_dt.day)
        year = now_dt.year
        try:
            candidate = datetime.datetime(year, month, day, tzinfo=ZoneInfo("Asia/Taipei"))
            if candidate.timestamp() > now_dt.timestamp() + 86400:
                candidate = datetime.datetime(year - 1, month, day, tzinfo=ZoneInfo("Asia/Taipei"))
            return candidate.timestamp()
        except Exception:
            pass
    return 0.0


def _should_update_binance_host_latest_signal(state, direction, strength, text=None, now_ts=None):
    if direction not in {"long", "short"} or strength <= 0:
        return False
    latest = state.get("latest_signal") if isinstance(state, dict) else None
    if not isinstance(latest, dict):
        return True
    candidate_content_ts = _binance_host_content_ts(text, now_ts=now_ts)
    latest_content_ts = _safe_float(latest.get("content_ts"), 0.0)
    if latest_content_ts <= 0:
        latest_content_ts = _binance_host_content_ts(latest.get("text_preview"), now_ts=now_ts)
    if candidate_content_ts > 0 and latest_content_ts <= 0:
        return True
    if candidate_content_ts <= 0 and latest_content_ts > 0:
        return False
    if candidate_content_ts > 0 and latest_content_ts > 0:
        recency_grace_sec = max(
            0,
            _safe_int(os.getenv("BINANCE_HOST_LIVE_LATEST_RECENCY_GRACE_SEC", 6 * 3600), 6 * 3600),
        )
        if candidate_content_ts + recency_grace_sec < latest_content_ts:
            return False
    latest_direction = str(latest.get("direction") or "").lower()
    latest_strength = _safe_int(latest.get("strength"), 0)
    if latest_direction not in {"long", "short"}:
        return True
    return strength >= latest_strength


def _binance_host_validation_symbol(text):
    text = str(text or "").upper()
    if "ETH" in text or "以太" in text:
        return "ETHUSDT"
    if "BTC" in text or "大餅" in text or "大饼" in text or "比特" in text:
        return "BTCUSDT"
    return ""


def _fetch_binance_host_validation_klines(symbol, start_ts, hours=24):
    symbol = str(symbol or "").upper().strip()
    start_ts = _safe_float(start_ts, 0.0)
    if not symbol or start_ts <= 0:
        return []
    try:
        rows, _ = _fetch_market_kline_rows(
            symbol,
            "1h",
            limit=max(6, min(48, _safe_int(hours, 24) + 3)),
            start_time_ms=int(start_ts * 1000),
            timeout=max(3, _safe_int(os.getenv("BINANCE_HOST_VALIDATION_TIMEOUT_SEC", 8), 8)),
            prefix="主播直播走勢驗證K線",
            source_preference="kraken_first",
        )
    except Exception as exc:
        now_ts = time.time()
        state = getattr(_fetch_binance_host_validation_klines, "_fail_log_state", {})
        key = f"{symbol}:{type(exc).__name__}:{str(exc)[:80]}"
        last_ts = _safe_float(state.get(key), 0.0) if isinstance(state, dict) else 0.0
        if now_ts - last_ts >= max(300.0, _safe_float(os.getenv("BINANCE_HOST_VALIDATION_FAIL_LOG_COOLDOWN_SEC", 900), 900)):
            print(f"⚠️ 主播直播走勢驗證K線抓取失敗 {symbol}: {exc}")
            if not isinstance(state, dict):
                state = {}
            state[key] = now_ts
            setattr(_fetch_binance_host_validation_klines, "_fail_log_state", state)
        return []
    if not isinstance(rows, list):
        return []
    parsed = []
    for row in rows:
        try:
            parsed.append(
                {
                    "open_time": _safe_float(row[0], 0.0) / 1000.0,
                    "open": _safe_float(row[1], 0.0),
                    "high": _safe_float(row[2], 0.0),
                    "low": _safe_float(row[3], 0.0),
                    "close": _safe_float(row[4], 0.0),
                }
            )
        except Exception:
            continue
    return parsed


def _validate_binance_host_live_segment(segment, now_ts=None):
    text = str((segment or {}).get("text") or "")
    direction, strength = _infer_binance_host_direction(text)
    if direction not in {"long", "short"}:
        return None
    content_ts = _binance_host_content_ts(text, now_ts=now_ts)
    if content_ts <= 0:
        return None
    symbol = _binance_host_validation_symbol(text)
    if not symbol:
        return None

    now_ts = _safe_float(now_ts, time.time())
    min_age_hours = max(1, _safe_int(os.getenv("BINANCE_HOST_VALIDATION_MIN_HOURS", 4), 4))
    age_hours = int((now_ts - content_ts) // 3600)
    if age_hours < min_age_hours:
        return {
            "status": "pending",
            "reason": "not_enough_future_klines",
            "symbol": symbol,
            "direction": direction,
            "strength": strength,
            "content_ts": content_ts,
            "age_hours": max(0, age_hours),
        }

    target_hours = [24, 12, 4]
    available_hours = max(min_age_hours, min(24, age_hours))
    klines = _fetch_binance_host_validation_klines(symbol, content_ts, hours=available_hours)
    if len(klines) < 2 or _safe_float(klines[0].get("open"), 0.0) <= 0:
        return {
            "status": "pending",
            "reason": "missing_klines",
            "symbol": symbol,
            "direction": direction,
            "strength": strength,
            "content_ts": content_ts,
            "age_hours": max(0, age_hours),
        }

    entry = _safe_float(klines[0].get("open"), 0.0)
    horizon_returns = {}
    for horizon in target_hours:
        if age_hours < horizon or len(klines) <= horizon:
            continue
        close = _safe_float(klines[horizon].get("close"), 0.0)
        if close > 0:
            horizon_returns[f"{horizon}h"] = round((close - entry) / entry * 100.0, 4)
    if not horizon_returns:
        close = _safe_float(klines[-1].get("close"), 0.0)
        if close > 0:
            used_hours = max(1, int((klines[-1].get("open_time", content_ts) - content_ts) // 3600))
            horizon_returns[f"{used_hours}h"] = round((close - entry) / entry * 100.0, 4)

    preferred_key = "24h" if "24h" in horizon_returns else ("12h" if "12h" in horizon_returns else "4h")
    move_pct = _safe_float(horizon_returns.get(preferred_key), 0.0)
    signed_move_pct = move_pct if direction == "long" else -move_pct
    min_move_pct = max(0.05, _safe_float(os.getenv("BINANCE_HOST_VALIDATION_MIN_MOVE_PCT", "0.25"), 0.25))
    success = signed_move_pct >= min_move_pct
    return {
        "status": "evaluated",
        "symbol": symbol,
        "direction": direction,
        "strength": strength,
        "content_ts": content_ts,
        "age_hours": max(0, age_hours),
        "entry_price": round(entry, 4),
        "horizon_returns_pct": horizon_returns,
        "used_horizon": preferred_key,
        "signed_move_pct": round(signed_move_pct, 4),
        "success": bool(success),
        "evaluated_at": now_ts,
    }


def _update_binance_host_live_validation_state(state, raw_sources, now_ts=None):
    if not isinstance(state, dict):
        return state
    now_ts = _safe_float(now_ts, time.time())
    validations = state.get("validations")
    if not isinstance(validations, dict):
        validations = {}
    max_segments = _safe_int(os.getenv("BINANCE_HOST_LIVE_MAX_SEGMENTS", 200), 200)
    host_name = str(os.getenv("BINANCE_HOST_LEARNING_NAME", "蓝歌") or "蓝歌").strip()
    for source, raw_text in raw_sources:
        for segment in _extract_binance_host_live_segments(raw_text, host_name=host_name, limit=max_segments):
            digest = segment.get("hash")
            if not digest:
                continue
            result = _validate_binance_host_live_segment(segment, now_ts=now_ts)
            if not result:
                continue
            result["source_url"] = source
            result["text_preview"] = str(segment.get("text") or "")[:160]
            previous = validations.get(digest)
            if isinstance(previous, dict) and previous.get("status") == "evaluated" and result.get("status") != "evaluated":
                continue
            validations[digest] = result
    evaluated = [item for item in validations.values() if isinstance(item, dict) and item.get("status") == "evaluated"]
    successful = sum(1 for item in evaluated if item.get("success"))
    state["validations"] = validations
    state["validation_summary"] = {
        "evaluated": len(evaluated),
        "successful": successful,
        "accuracy": round(successful / len(evaluated) * 100.0, 2) if evaluated else 0.0,
        "pending": sum(1 for item in validations.values() if isinstance(item, dict) and item.get("status") == "pending"),
        "updated_at": now_ts,
    }
    return state


def _record_binance_host_learning_item(
    *,
    price,
    market_context,
    host_name,
    source,
    item,
    seen,
    prompt_prefix,
    market_source,
    primary_reason,
    risk_note,
    failure_label,
    success_label,
):
    digest = item.get("hash")
    if not digest or digest in seen:
        return 0
    question_key = f"{prompt_prefix}:{host_name}:{digest[:16]}"
    if _mlx_learning_question_exists(question_key):
        seen.add(digest)
        return 0

    direction, strength = _infer_binance_host_direction(item.get("text"))
    confidence = min(0.72, 0.38 + max(0, strength) * 0.06)
    direction_text = {"long": "做多", "short": "做空", "neutral": "觀望"}.get(direction, "觀望")
    response = json.dumps(
        {
            "direction": direction_text,
            "primary_reason": primary_reason,
            "confidence": round(confidence, 2),
            "market_regime": "news_driven",
            "support_zone": [],
            "resistance_zone": [],
            "risk_note": risk_note,
            "source": source,
            "text": item.get("text"),
        },
        ensure_ascii=False,
    )
    market = dict(market_context if isinstance(market_context, dict) else {})
    market.update(
        {
            "price": _safe_float(price, 0.0),
            "source": market_source,
            "host_name": host_name,
            "source_url": source,
            "primary_reason": primary_reason,
            "direction_hint": direction,
        }
    )
    try:
        inserted = record_mlx_analysis(question_key, response, market)
    except Exception as exc:
        print(f"⚠️ {failure_label}: {exc}")
        inserted = 0
    if inserted:
        seen.add(digest)
        print(f"🧠 已寫入幣安主播{host_name}{success_label}: {direction_text} | {digest[:8]}")
        return int(inserted)
    return 0


def _mlx_learning_question_exists(question_key):
    question_key = str(question_key or "").strip()
    if not question_key:
        return False
    try:
        with sqlite3.connect(str(MLX_LEARNING_DB_PATH), timeout=5) as connection:
            row = connection.execute(
                "SELECT 1 FROM analysis_episode WHERE question = ? LIMIT 1",
                (question_key,),
            ).fetchone()
            return row is not None
    except Exception:
        return False


def _count_binance_host_live_learning_episodes():
    try:
        with sqlite3.connect(str(MLX_LEARNING_DB_PATH), timeout=5) as connection:
            row = connection.execute(
                """
                SELECT COUNT(*)
                FROM analysis_episode
                WHERE question LIKE 'binance-host-live:%'
                   OR question LIKE 'binance-host-live-full:%'
                """
            ).fetchone()
            return _safe_int(row[0] if row else 0, 0)
    except Exception:
        return 0


def _process_binance_host_learning(price, market_context=None):
    if not _binance_host_learning_enabled():
        return 0

    now_ts = time.time()
    state = _read_json_file(BINANCE_HOST_LEARNING_STATE_PATH, {})
    if not isinstance(state, dict):
        state = {}
    interval = max(60, _safe_int(os.getenv("BINANCE_HOST_LEARNING_INTERVAL_SEC", 900), 900))
    if now_ts - _safe_float(state.get("last_run_ts"), 0.0) < interval:
        return 0

    state["last_run_ts"] = now_ts
    seen_hashes = state.get("seen_hashes")
    if not isinstance(seen_hashes, list):
        seen_hashes = []
    seen = set(str(item) for item in seen_hashes)

    host_name = str(os.getenv("BINANCE_HOST_LEARNING_NAME", "蓝歌") or "蓝歌").strip()
    max_posts = max(1, _safe_int(os.getenv("BINANCE_HOST_LEARNING_MAX_POSTS", 3), 3))
    urls, text_sources = _binance_host_learning_sources()
    raw_sources = list(text_sources)
    for url in urls:
        content = _fetch_binance_host_source(url)
        if content:
            raw_sources.append((url, content))

    learned = 0
    market_context = market_context if isinstance(market_context, dict) else {}
    for source, raw_text in raw_sources:
        for post in _extract_binance_host_posts(raw_text, host_name=host_name, limit=max_posts):
            inserted = _record_binance_host_learning_item(
                price=price,
                market_context=market_context,
                host_name=host_name,
                source=source,
                item=post,
                seen=seen,
                prompt_prefix="binance-host",
                market_source="binance_host_lange",
                primary_reason=f"幣安主播{host_name}公開觀點",
                risk_note="主播觀點只作 MLX 輔助學習，不直接下單。",
                failure_label="藍歌觀點寫入 MLX 學習庫失敗",
                success_label="觀點學習",
            )
            if inserted:
                learned += inserted
                direction, strength = _infer_binance_host_direction(post.get("text"))
                if _should_update_binance_host_latest_signal(state, direction, strength, post.get("text"), now_ts):
                    state["latest_signal"] = {
                        "direction": direction,
                        "strength": strength,
                        "confidence": min(0.72, 0.38 + max(0, strength) * 0.06),
                        "source_url": source,
                        "text_preview": str(post.get("text") or "")[:160],
                        "content_ts": _binance_host_content_ts(post.get("text"), now_ts=now_ts),
                        "learned_at": now_ts,
                    }
            if learned >= max_posts:
                break
        if learned >= max_posts:
            break

    state["seen_hashes"] = list(seen)[-300:]
    state["last_learned_at"] = now_ts if learned else state.get("last_learned_at", 0)
    db_live_total = _count_binance_host_live_learning_episodes()
    state["learned_total"] = max(
        _safe_int(state.get("learned_total"), 0) + learned,
        db_live_total,
    )
    _write_json_file(BINANCE_HOST_LEARNING_STATE_PATH, state)
    return learned


def _process_binance_host_live_learning(price, market_context=None):
    if not _binance_host_live_learning_enabled():
        return 0

    now_ts = time.time()
    state = _read_json_file(BINANCE_HOST_LIVE_LEARNING_STATE_PATH, {})
    if not isinstance(state, dict):
        state = {}
    interval = max(30, _safe_int(os.getenv("BINANCE_HOST_LIVE_INTERVAL_SEC", 120), 120))
    if now_ts - _safe_float(state.get("last_run_ts"), 0.0) < interval:
        return 0

    state["last_run_ts"] = now_ts
    seen_hashes = state.get("seen_hashes")
    if not isinstance(seen_hashes, list):
        seen_hashes = []
    seen = set(str(item) for item in seen_hashes)

    host_name = str(os.getenv("BINANCE_HOST_LEARNING_NAME", "蓝歌") or "蓝歌").strip()
    max_segments = _safe_int(os.getenv("BINANCE_HOST_LIVE_MAX_SEGMENTS", 0), 0)
    urls, text_sources = _binance_host_live_learning_sources()
    raw_sources = list(text_sources)
    for url in urls:
        content = _fetch_binance_host_live_source(url)
        if content:
            raw_sources.append((url, content))

    learned = 0
    market_context = market_context if isinstance(market_context, dict) else {}
    source_stats = []
    for source, raw_text in raw_sources:
        source_learned = 0
        source_segments = _extract_binance_host_live_segments(
            raw_text,
            host_name=host_name,
            limit=max_segments,
        )
        full_document = _extract_binance_host_live_document(raw_text, source=source)
        if full_document:
            inserted = _record_binance_host_learning_item(
                price=price,
                market_context=market_context,
                host_name=host_name,
                source=source,
                item=full_document,
                seen=seen,
                prompt_prefix="binance-host-live-full",
                market_source="binance_host_lange_live_full_archive",
                primary_reason=f"幣安主播{host_name}直播檔完整資訊",
                risk_note="直播檔完整資訊用於 MLX 策略學習；實際開單仍需價格結構、支撐壓力與風控確認。",
                failure_label="藍歌直播檔完整資訊寫入 MLX 學習庫失敗",
                success_label="直播檔完整資訊學習",
            )
            learned += inserted
            source_learned += inserted
        for segment in source_segments:
            inserted = _record_binance_host_learning_item(
                price=price,
                market_context=market_context,
                host_name=host_name,
                source=source,
                item=segment,
                seen=seen,
                prompt_prefix="binance-host-live",
                market_source="binance_host_lange_live",
                primary_reason=f"幣安主播{host_name}直播內容",
                risk_note="直播內容只作 MLX 輔助學習，不直接下單。",
                failure_label="藍歌直播內容寫入 MLX 學習庫失敗",
                success_label="直播內容學習",
            )
            if inserted:
                learned += inserted
                source_learned += inserted
                direction, strength = _infer_binance_host_direction(segment.get("text"))
                if _should_update_binance_host_latest_signal(state, direction, strength, segment.get("text"), now_ts):
                    state["latest_signal"] = {
                        "direction": direction,
                        "strength": strength,
                        "confidence": min(0.72, 0.38 + max(0, strength) * 0.06),
                        "source_url": source,
                        "text_preview": str(segment.get("text") or "")[:160],
                        "content_ts": _binance_host_content_ts(segment.get("text"), now_ts=now_ts),
                        "learned_at": now_ts,
                    }
            if max_segments > 0 and learned >= max_segments:
                break
        source_stats.append(
            {
                "source": source,
                "segments": len(source_segments),
                "full_document": bool(full_document),
                "learned": source_learned,
            }
        )
        if max_segments > 0 and learned >= max_segments:
            break

    state["seen_hashes"] = list(seen)[-2000:]
    state["last_learned_at"] = now_ts if learned else state.get("last_learned_at", 0)
    state["learned_total"] = _safe_int(state.get("learned_total"), 0) + learned
    state["source_stats"] = source_stats
    state["full_archive_learning_enabled"] = True
    state["max_segments"] = max_segments
    state = _update_binance_host_live_validation_state(state, raw_sources, now_ts=now_ts)
    _write_json_file(BINANCE_HOST_LIVE_LEARNING_STATE_PATH, state)
    return learned


def _load_binance_host_content_override_signal():
    if not _is_truthy(os.getenv("TRADE_ALLOW_HOST_CONTENT_OVERRIDE", "1")):
        return {"enabled": False, "usable": False}

    max_age_sec = max(60, _safe_int(os.getenv("TRADE_HOST_CONTENT_OVERRIDE_MAX_AGE_SEC", 6 * 3600), 6 * 3600))
    min_strength = max(1, _safe_int(os.getenv("TRADE_HOST_CONTENT_OVERRIDE_MIN_STRENGTH", 1), 1))
    now_ts = time.time()
    candidates = []

    for path, source_type, weight in (
        (BINANCE_HOST_LIVE_LEARNING_STATE_PATH, "live", 1.15),
        (BINANCE_HOST_LEARNING_STATE_PATH, "post", 1.0),
    ):
        state = _read_json_file(path, {})
        if not isinstance(state, dict):
            continue
        learned_at = _safe_float(state.get("last_learned_at"), 0.0)
        if learned_at <= 0 or now_ts - learned_at > max_age_sec:
            continue
        latest = state.get("latest_signal")
        if not isinstance(latest, dict):
            continue
        validation_summary = state.get("validation_summary") if isinstance(state.get("validation_summary"), dict) else {}
        direction = str(latest.get("direction") or "").lower()
        strength = _safe_int(latest.get("strength"), 0)
        if direction not in {"long", "short"} or strength < min_strength:
            continue
        age_sec = max(0.0, now_ts - learned_at)
        freshness = max(0.0, 1.0 - age_sec / max(max_age_sec, 1.0))
        confidence = max(0.0, min(0.90, _safe_float(latest.get("confidence"), 0.0)))
        if confidence <= 0:
            confidence = min(0.72, 0.38 + strength * 0.06)
        candidates.append(
            {
                "direction": direction,
                "strength": strength,
                "confidence": confidence,
                "freshness": freshness,
                "source_type": source_type,
                "source_url": str(latest.get("source_url") or ""),
                "text_preview": str(latest.get("text_preview") or "")[:160],
                "learned_at": learned_at,
                "validation_evaluated": _safe_int(validation_summary.get("evaluated"), 0),
                "validation_successful": _safe_int(validation_summary.get("successful"), 0),
                "validation_accuracy": _safe_float(validation_summary.get("accuracy"), 0.0),
                "weighted_score": (strength * 0.12 + confidence + freshness * 0.35) * weight,
            }
        )

    if not candidates:
        return {"enabled": True, "usable": False}

    best = max(candidates, key=lambda item: item.get("weighted_score", 0.0))
    best["usable"] = True
    best["enabled"] = True
    return best


def _assess_host_content_override(signal, *, htf, mid_trend, macro_bias, sr_bias, support_hits, resistance_hits, derivatives_pressure):
    signal = signal if isinstance(signal, dict) else {}
    if not signal.get("usable"):
        return {"enabled": bool(signal.get("enabled")), "usable": False, "applied": False}

    direction = str(signal.get("direction") or "").lower()
    if direction not in {"long", "short"}:
        return {"enabled": True, "usable": False, "applied": False}

    sign = 1 if direction == "long" else -1
    conflicts = 0
    supports = 0
    for value, threshold, weight in (
        (_safe_int(htf, 0), 1, 1),
        (_safe_int(mid_trend, 0), 1, 1),
        (_safe_float(macro_bias, 0.0), 0.35, 1),
        (_safe_float(sr_bias, 0.0), 0.10, 1),
        (_safe_float(derivatives_pressure, 0.0), 0.12, 1),
    ):
        aligned = value * sign
        if aligned >= threshold:
            supports += weight
        elif aligned <= -threshold:
            conflicts += weight

    if direction == "long":
        if _safe_int(resistance_hits, 0) >= 2:
            conflicts += 1
        if _safe_int(support_hits, 0) >= 1:
            supports += 1
    else:
        if _safe_int(support_hits, 0) >= 2:
            conflicts += 1
        if _safe_int(resistance_hits, 0) >= 1:
            supports += 1

    primary_mode = _is_truthy(os.getenv("TRADE_HOST_CONTENT_PRIMARY_MODE", "1"))
    default_max_conflicts = 2 if primary_mode else 1
    max_conflicts = max(
        0,
        _safe_int(
            os.getenv("TRADE_HOST_CONTENT_OVERRIDE_MAX_CONFLICTS", default_max_conflicts),
            default_max_conflicts,
        ),
    )
    strength = _safe_int(signal.get("strength"), 0)
    confidence = max(0.0, min(0.90, _safe_float(signal.get("confidence"), 0.0)))
    freshness = max(0.0, min(1.0, _safe_float(signal.get("freshness"), 0.0)))
    validation_evaluated = _safe_int(signal.get("validation_evaluated"), 0)
    validation_accuracy = _safe_float(signal.get("validation_accuracy"), 0.0)
    validation_edge = 0.0
    min_validation_samples = max(
        3,
        _safe_int(os.getenv("TRADE_HOST_CONTENT_MIN_VALIDATION_SAMPLES", 5), 5),
    )
    validation_ready = validation_evaluated >= min_validation_samples
    if validation_ready:
        validation_edge = max(-0.30, min(0.20, (validation_accuracy - 52.0) / 100.0))
    quality = (
        confidence
        + min(0.30, strength * 0.04)
        + freshness * 0.12
        + validation_edge * 0.35
        + supports * 0.03
        - conflicts * 0.08
    )
    default_min_quality = 0.45 if primary_mode else 0.58
    min_quality = max(
        0.40,
        min(
            0.85,
            _safe_float(
                os.getenv("TRADE_HOST_CONTENT_OVERRIDE_MIN_QUALITY", default_min_quality),
                default_min_quality,
            ),
        ),
    )
    min_primary_validation_accuracy = max(
        50.0,
        _safe_float(os.getenv("TRADE_HOST_CONTENT_PRIMARY_MIN_ACCURACY", 54.0), 54.0),
    )
    validation_blocks_primary = (
        primary_mode
        and validation_ready
        and validation_accuracy < min_primary_validation_accuracy
    )
    applied = (
        conflicts <= max_conflicts
        and quality >= min_quality
        and not validation_blocks_primary
    )

    payload = dict(signal)
    payload.update(
        {
            "usable": True,
            "applied": applied,
            "mode": "primary" if primary_mode else "override",
            "conflicts": conflicts,
            "supports": supports,
            "quality": round(quality, 3),
            "min_quality": min_quality,
            "max_conflicts": max_conflicts,
            "validation_evaluated": validation_evaluated,
            "validation_accuracy": round(validation_accuracy, 2),
            "validation_edge": round(validation_edge, 3),
            "validation_ready": validation_ready,
            "validation_blocks_primary": validation_blocks_primary,
        }
    )
    return payload


def _score_mlx_learned_entry_logic(
    technical_score,
    *,
    range_pos,
    htf,
    mid_trend,
    breakout,
    regime,
    volume_spike,
    buy_pressure,
    sell_pressure,
    sweep_high,
    sweep_low,
    support_hits,
    resistance_hits,
    repeated_support_tests,
    repeated_resistance_tests,
    repeated_test_pressure,
    macro_bias,
    derivatives_pressure,
):
    """Use the learned live-session playbook as the trade-entry decision layer."""
    long_setup = 0.0
    short_setup = 0.0
    reasons = []
    range_pos = max(0.0, min(1.0, _safe_float(range_pos, 0.5)))
    support_break_confirmed = (
        repeated_support_tests >= 2
        and breakout == -1
        and (volume_spike or sell_pressure)
        and not sweep_low
    )
    support_hold_unconfirmed = repeated_support_tests >= 2 and not support_break_confirmed
    resistance_break_confirmed = (
        repeated_resistance_tests >= 2
        and breakout == 1
        and (volume_spike or buy_pressure)
        and not sweep_high
    )
    resistance_hold_unconfirmed = repeated_resistance_tests >= 2 and not resistance_break_confirmed

    # Learned rule: do not chase; wait for support/resistance behavior first.
    if range_pos <= 0.28:
        long_setup += 1.0
        reasons.append("低位靠近支撐，優先等多方確認")
    elif range_pos >= 0.72:
        short_setup += 1.0
        reasons.append("高位靠近壓力，優先等空方確認")
    if range_pos >= 0.86 and not (breakout == 1 and volume_spike and buy_pressure):
        long_setup -= 0.8
        short_setup += 0.5
        reasons.append("高位未確認突破，不追多")
    if range_pos <= 0.14 and not (breakout == -1 and volume_spike and sell_pressure):
        short_setup -= 0.8
        long_setup += 0.5
        reasons.append("低位未確認跌破，不追空")

    # Learned rule: repeated tests are useful only with confirmation.
    if repeated_resistance_tests >= 2:
        if resistance_break_confirmed:
            long_setup += min(2.0, 0.7 + repeated_resistance_tests * 0.28)
            reasons.append("壓力連續測試後放量突破，偏多")
        else:
            short_setup += min(1.4, 0.5 + repeated_resistance_tests * 0.18)
            long_setup -= min(0.8, 0.25 + repeated_resistance_tests * 0.10)
            reasons.append("壓力連續測試但未確認突破，偏空等反彈失敗")
    if repeated_support_tests >= 2:
        if support_break_confirmed:
            short_setup += min(2.0, 0.7 + repeated_support_tests * 0.28)
            reasons.append("支撐連續測試後跌破，偏空")
        else:
            long_setup += min(1.4, 0.5 + repeated_support_tests * 0.18)
            short_setup -= min(0.8, 0.25 + repeated_support_tests * 0.10)
            reasons.append("支撐連續測試但未確認跌破，偏多等承接")

    if breakout == 1:
        if volume_spike and buy_pressure and not sweep_high:
            long_setup += 1.2
            reasons.append("突破有量且買盤確認")
        else:
            short_setup += 0.6
            reasons.append("突破量能不足，防假突破")
    elif breakout == -1:
        if volume_spike and sell_pressure and not sweep_low:
            short_setup += 1.2
            reasons.append("跌破有量且賣壓確認")
        else:
            long_setup += 0.6
            reasons.append("跌破量能不足，防假跌破")

    if sweep_high:
        short_setup += 1.1
        reasons.append("掃高回落，反彈失敗偏空")
    if sweep_low:
        long_setup += 1.1
        reasons.append("掃低收回，跌破失敗偏多")

    if _safe_int(resistance_hits, 0) >= 2:
        short_setup += 0.5
    elif _safe_int(resistance_hits, 0) == 1:
        short_setup += 0.25
    if _safe_int(support_hits, 0) >= 2:
        long_setup += 0.5
    elif _safe_int(support_hits, 0) == 1:
        long_setup += 0.25

    if str(regime).startswith("bull"):
        long_setup += 0.35
    elif str(regime).startswith("bear"):
        short_setup += 0.35
    if _safe_int(mid_trend, 0) > 0:
        long_setup += 0.35
    elif _safe_int(mid_trend, 0) < 0:
        short_setup += 0.35
    if _safe_int(htf, 0) > 0:
        long_setup += 0.25
    elif _safe_int(htf, 0) < 0:
        short_setup += 0.25

    macro_bias = _safe_float(macro_bias, 0.0)
    derivatives_pressure = _safe_float(derivatives_pressure, 0.0)
    if macro_bias > 0.45:
        long_setup += 0.25
    elif macro_bias < -0.45:
        short_setup += 0.25
    if derivatives_pressure > 0.12:
        long_setup += 0.20
    elif derivatives_pressure < -0.12:
        short_setup += 0.20

    if repeated_test_pressure > 0:
        long_setup += min(0.35, abs(repeated_test_pressure) * 0.4)
    elif repeated_test_pressure < 0:
        short_setup += min(0.35, abs(repeated_test_pressure) * 0.4)

    if support_hold_unconfirmed and not support_break_confirmed:
        short_setup -= 0.35
        reasons.append("支撐未破前降低追空權重")
    if resistance_hold_unconfirmed and not resistance_break_confirmed:
        long_setup -= 0.35
        reasons.append("壓力未破前降低追多權重")

    setup_delta = long_setup - short_setup
    learned_edge = max(-0.34, min(0.34, setup_delta * 0.11))
    technical_aux = max(-0.06, min(0.06, _safe_float(technical_score, 0.5) - 0.5))
    learned_score = max(0.05, min(0.95, 0.5 + learned_edge + technical_aux))
    direction = "long" if learned_score > 0.5 else "short" if learned_score < 0.5 else "neutral"
    confidence = min(0.72, 0.42 + abs(setup_delta) * 0.06)
    return {
        "score": learned_score,
        "direction": direction,
        "confidence": round(confidence, 3),
        "long_setup": round(long_setup, 3),
        "short_setup": round(short_setup, 3),
        "technical_aux": round(technical_aux, 3),
        "support_break_confirmed": support_break_confirmed,
        "support_hold_unconfirmed": support_hold_unconfirmed,
        "resistance_break_confirmed": resistance_break_confirmed,
        "resistance_hold_unconfirmed": resistance_hold_unconfirmed,
        "reasons": reasons[:8],
    }


def _classify_wait_state(final, *, repeated_support_tests, repeated_resistance_tests, learned_entry_logic, breakout, volume_spike):
    final_text = str(final or "")
    if not final_text.startswith("觀望"):
        return final_text
    learned = learned_entry_logic if isinstance(learned_entry_logic, dict) else {}
    if learned.get("support_hold_unconfirmed"):
        return "觀望（等支撐跌破或承接確認）"
    if learned.get("resistance_hold_unconfirmed"):
        return "觀望（等壓力突破或反彈失敗）"
    if _safe_int(repeated_support_tests, 0) >= 2 and breakout != -1:
        return "觀望（支撐連測未跌破）"
    if _safe_int(repeated_resistance_tests, 0) >= 2 and breakout != 1:
        return "觀望（壓力連測未突破）"
    if "逆向共振不足" in final_text:
        return final_text.replace("逆向共振不足", "等待共振確認")
    if "低信心" in final_text and not volume_spike:
        return "觀望（低信心且量能不足）"
    return final_text


def _load_learning_buffer_samples(max_per_label=40):
    try:
        with open(NEWS_LEARNING_BUFFER, "rb") as f:
            raw_buffer = pickle.load(f)
    except Exception:
        return []

    if not isinstance(raw_buffer, list):
        return []

    unique_samples = []
    seen_texts = set()

    for item in reversed(raw_buffer):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue

        text, label = item
        prepared = _prepare_news_text_for_model(text)
        clean_label = _sanitize_news_label(label)
        if not prepared or clean_label is None or prepared in seen_texts:
            continue

        seen_texts.add(prepared)
        unique_samples.append((prepared, clean_label))

    unique_samples.reverse()

    grouped = {}
    for text, label in unique_samples:
        grouped.setdefault(label, []).append((text, label))

    selected = []
    for label in sorted(grouped):
        selected.extend(grouped[label][-max_per_label:])

    return selected


def _build_news_vectorizer():
    return FeatureUnion(
        transformer_list=[
            (
                "word",
                TfidfVectorizer(
                    max_features=3000,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                    stop_words="english",
                    lowercase=False,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    max_features=2000,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    sublinear_tf=True,
                    lowercase=False,
                ),
            ),
        ]
    )


def _build_news_model():
    return VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=400, class_weight="balanced", random_state=42)),
            ("sgd", SGDClassifier(loss="log_loss", alpha=1e-4, class_weight="balanced", random_state=42)),
            ("nb", ComplementNB(alpha=0.35)),
        ],
        voting="soft",
    )

def translate_news_to_zh(text):
    """將新聞標題翻譯成繁中，失敗時回傳原文。"""
    try:
        src = str(text or "").strip()
        if not src:
            return src

        # 已含中文就不重翻
        if re.search(r"[\u4e00-\u9fff]", src):
            return src

        if src in TRANSLATION_CACHE:
            return TRANSLATION_CACHE[src]

        # 保守截斷，避免超長文本增加延遲與成本
        short_src = src[:220]

        zh = ""
        if OPENAI_PAID_API_ENABLED and OPENAI_API_KEY:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = _build_openai_chat_payload(
                OPENAI_TRANSLATION_MODEL,
                [
                    {
                        "role": _openai_instruction_role(OPENAI_TRANSLATION_MODEL),
                        "content": "你是專業翻譯員。請把輸入內容翻成繁體中文，只輸出翻譯結果，不要補充說明。"
                    },
                    {
                        "role": "user",
                        "content": short_src
                    }
                ],
                temperature=0,
            )

            res = requests.post(url, headers=headers, json=payload, timeout=6)
            data = res.json() if res is not None else {}
            zh = _extract_openai_chat_text(data)

        if not zh:
            zh = _google_translate_to_zh(src)

        if not re.search(r"[\u4e00-\u9fff]", zh):
            zh = _google_translate_to_zh(src)

        if not re.search(r"[\u4e00-\u9fff]", zh):
            zh = _local_translate_news_fallback(src)

        zh = _normalize_finance_terms_zh(zh)

        if len(TRANSLATION_CACHE) > 1000:
            TRANSLATION_CACHE.clear()
        TRANSLATION_CACHE[src] = zh
        return zh
    except Exception:
        zh = _google_translate_to_zh(text)
        if re.search(r"[\u4e00-\u9fff]", zh):
            return zh
        return _local_translate_news_fallback(text)


# ===== AI 新聞分類訓練數據 =====
NEWS_TRAINING_DATA = [
    # 利多 - 強信號
    ("Bitcoin ETF approved by SEC", 2),
    ("Ethereum staking approval approved", 2),
    ("MicroStrategy buys more Bitcoin holdings", 2),
    ("BlackRock files for spot Bitcoin ETF", 2),
    ("Crypto adoption in El Salvador", 2),
    ("Major bank launches crypto trading", 2),
    ("Hash rate reaches all-time high", 2),
    ("Network upgrade goes live successfully", 2),
    ("Institutional investors buy crypto", 2),
    ("New partnership for Bitcoin announced", 2),
    ("Crypto ETF inflow record high", 2),
    ("Spot Bitcoin ETF approval", 2),
    # 利多 - 弱信號
    ("Crypto market rallies higher", 1),
    ("Bitcoin surge", 1),
    ("Ethereum listing on exchange", 1),
    ("Adoption increases", 1),
    ("Record accumulation by whales", 1),
    ("Positive sentiment in market", 1),
    ("Upgrade launches", 1),
    ("Support for crypto regulation", 1),
    # 利空 - 強信號
    ("Crypto exchange hacked stolen funds", -2),
    ("FTX collapse bankruptcy", -2),
    ("SEC charges crypto company fraud", -2),
    ("Major exchange delisted", -2),
    ("Bitcoin hack $100 million stolen", -2),
    ("Lawsuit against crypto firm", -2),
    ("Investigation fraud charges", -2),
    ("Exchange suspends withdrawals", -2),
    ("Regulatory ban announced", -2),
    # 利空 - 弱信號
    ("Crypto market drops lower", -1),
    ("Sell-off in Bitcoin", -1),
    ("Ethereum down", -1),
    ("Outflow from crypto funds", -1),
    ("Bearish sentiment market", -1),
    ("Price decline", -1),
    ("Whale dump", -1),
    # 宏觀 / 事件類
    ("Fed raises interest rates", 0),
    ("FOMC meeting decision", 0),
    ("Inflation data released", 0),
    ("War tensions Middle East", 0),
    ("Tariff announcement", 0),
    ("CPI report misses", 0),
    ("Labor data released", 0),
    ("Economic event scheduled", 0),
    ("Geopolitical news", 0),
    ("Policy announcement", 0),
]

def train_news_model():
    global news_model, news_vectorizer
    if news_model is not None and news_vectorizer is not None:
        return

    sample_map = {}
    for text, label in NEWS_TRAINING_DATA:
        prepared = _prepare_news_text_for_model(text)
        clean_label = _sanitize_news_label(label)
        if prepared and clean_label is not None:
            sample_map[prepared] = clean_label

    for text, label in _load_learning_buffer_samples(max_per_label=40):
        sample_map[text] = label

    if not sample_map:
        sample_map = {
            _prepare_news_text_for_model(text): label
            for text, label in NEWS_TRAINING_DATA
            if _prepare_news_text_for_model(text)
        }

    texts = list(sample_map.keys())
    labels = [sample_map[text] for text in texts]

    news_vectorizer = _build_news_vectorizer()
    X = news_vectorizer.fit_transform(texts)
    y = np.array(labels)

    news_model = _build_news_model()
    news_model.fit(X, y)

    try:
        ensure_parent_dir(NEWS_MODEL_PATH)
        ensure_parent_dir(NEWS_VECTORIZER_PATH)
        _write_pickle_atomic(NEWS_MODEL_PATH, news_model)
        _write_pickle_atomic(NEWS_VECTORIZER_PATH, news_vectorizer)
        _write_json_file(
            NEWS_MODEL_META_PATH,
            {
                "version": NEWS_MODEL_VERSION,
                "sample_count": len(texts),
                "trained_at": datetime.datetime.now().isoformat(),
            },
        )
    except Exception:
        pass


def load_news_model(force_retrain=False):
    global news_model, news_vectorizer
    if news_model is not None and news_vectorizer is not None and not force_retrain:
        return

    meta = _read_json_file(NEWS_MODEL_META_PATH, {})
    needs_retrain = force_retrain or meta.get("version") != NEWS_MODEL_VERSION

    if not needs_retrain:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                with open(NEWS_MODEL_PATH, "rb") as f:
                    news_model = pickle.load(f)
                with open(NEWS_VECTORIZER_PATH, "rb") as f:
                    news_vectorizer = pickle.load(f)

            if not hasattr(news_model, "predict_proba") or not hasattr(news_vectorizer, "transform"):
                raise ValueError("news model/vectorizer is incompatible")
            return
        except Exception as e:
            print(f"♻️ 新聞模型已改用當前環境重建: {e}")

    news_model = None
    news_vectorizer = None
    train_news_model()

def predict_news_sentiment(text):
    """預測新聞情緒（舊函數，保持兼容性）"""
    global news_model, news_vectorizer
    if news_model is None:
        load_news_model()
    if news_model is None:
        return 0  # 預設中性

    prepared = _prepare_news_text_for_model(text)
    if not prepared:
        return 0

    X = news_vectorizer.transform([prepared])
    prediction = news_model.predict(X)[0]
    return int(prediction)


def predict_news_sentiment_with_confidence(text):
    """預測新聞情緒 + 置信度分數（新函數，更智能）"""
    global news_model, news_vectorizer
    if news_model is None:
        load_news_model()
    if news_model is None:
        return 0, 0.33  # 預設中性，低置信度

    try:
        prepared = _prepare_news_text_for_model(text)
        if not prepared:
            return 0, 0.33

        X = news_vectorizer.transform([prepared])
        prediction = news_model.predict(X)[0]
        
        # 獲取概率分布
        probabilities = news_model.predict_proba(X)[0]
        confidence = max(probabilities)  # 取最高概率
        
        return int(prediction), float(confidence)
    except:
        return 0, 0.33


def _keyword_bias_score(text):
    """中性修正用關鍵字分數：>0 偏多，<0 偏空。"""
    low = str(text or "").lower()

    bull_words = [
        "approval", "approved", "etf", "inflow", "surge", "rally", "breakout",
        "partnership", "adoption", "upgrade", "listing", "launch", "buyback",
        "accumulation", "institutional", "rate cut", "stimulus"
    ]
    bear_words = [
        "hack", "exploit", "lawsuit", "ban", "fraud", "bankruptcy", "delist",
        "outflow", "dump", "sell-off", "crash", "plunge", "investigation",
        "sanction", "rate hike", "liquidation", "withdrawal halt"
    ]

    score = 0
    for w in bull_words:
        if w in low:
            score += 1
    for w in bear_words:
        if w in low:
            score -= 1

    return score


def _refine_neutral_bias(text, ai_bias, ai_confidence):
    """修正中性判斷：AI 低信心或中性時，使用關鍵字進行二次判斷。"""
    final_bias = int(ai_bias)
    k_score = _keyword_bias_score(text)

    # AI 明確高信心時，不強行覆蓋
    if ai_confidence >= 0.68 and abs(final_bias) >= 1:
        return final_bias

    # AI 判中性時，優先用關鍵字修正
    if final_bias == 0:
        if k_score >= 2:
            return 1
        if k_score <= -2:
            return -1
        return 0

    # AI 低信心弱訊號時，關鍵字可升降一級
    if abs(final_bias) == 1 and ai_confidence < 0.52:
        if k_score >= 3:
            return 2
        if k_score <= -3:
            return -2

    return final_bias


# ===== 增量學習系統：記錄預測並持續改進 =====
def log_prediction_result(
    news_text,
    predicted_bias,
    actual_market_move=None,
    correct=None,
    actual_bias=None,
    ai_confidence=None,
    source="News",
    schedule_eval=True,
):
    """記錄預測結果用於增量學習和精準度評估"""
    try:
        # 只記錄與 BTC/ETH 相關的新聞，避免無關資訊污染訓練資料
        if not _is_crypto_relevant_news(news_text):
            return

        prepared = _prepare_news_text_for_model(news_text)
        raw_text = normalize_news_text(news_text)
        if not prepared or not raw_text or "\n" in raw_text:
            return

        recent = getattr(log_prediction_result, "_recent", {})
        now_ts = time.time()
        dedupe_window_sec = 900 if actual_market_move is None and correct is None else 0
        if dedupe_window_sec > 0 and now_ts - recent.get(prepared, 0) < dedupe_window_sec:
            return

        recent[prepared] = now_ts
        if len(recent) > 2000:
            recent = {k: v for k, v in recent.items() if now_ts - v < 3600}
        log_prediction_result._recent = recent

        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "news": raw_text[:150],
            "news_key": prepared[:120],
            "predicted_bias": predicted_bias,
            "actual_move": actual_market_move,
            "actual_bias": actual_bias,
            "is_correct": correct,
            "ai_confidence": ai_confidence,
            "source": str(source or "News")[:60],
        }

        ensure_parent_dir(NEWS_PERFORMANCE_LOG)
        with open(NEWS_PERFORMANCE_LOG, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        PREDICTION_ACCURACY_CACHE["cache_key"] = None
        PREDICTION_ACCURACY_CACHE["stats"] = None

        if schedule_eval and actual_market_move is None and correct is None:
            _queue_news_prediction_for_evaluation(
                news_text=raw_text,
                predicted_bias=predicted_bias,
                ai_confidence=ai_confidence,
                source=source,
            )
    except Exception:
        pass


def get_prediction_accuracy():
    """計算模型預測準確度"""
    default_stats = {"accuracy": 0, "total": 0, "correct": 0}
    try:
        log_path = Path(NEWS_PERFORMANCE_LOG)
        if not log_path.exists():
            return default_stats

        stat = log_path.stat()
        cache_key = f"{int(stat.st_mtime)}:{stat.st_size}"

        if PREDICTION_ACCURACY_CACHE.get("cache_key") == cache_key and PREDICTION_ACCURACY_CACHE.get("stats"):
            return dict(PREDICTION_ACCURACY_CACHE["stats"])

        cached = _read_json_file(NEWS_STATS_CACHE_PATH, {})
        if cached.get("cache_key") == cache_key:
            stats = {
                "accuracy": round(float(cached.get("accuracy", 0)), 2),
                "total": int(cached.get("total", 0)),
                "correct": int(cached.get("correct", 0)),
            }
            PREDICTION_ACCURACY_CACHE["cache_key"] = cache_key
            PREDICTION_ACCURACY_CACHE["stats"] = dict(stats)
            return stats

        total = 0
        correct = 0
        
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("is_correct") is not None:
                        total += 1
                        if record["is_correct"]:
                            correct += 1
                except:
                    pass
        accuracy = (correct / total * 100) if total > 0 else 0
        stats = {"accuracy": round(accuracy, 2), "total": total, "correct": correct}
        PREDICTION_ACCURACY_CACHE["cache_key"] = cache_key
        PREDICTION_ACCURACY_CACHE["stats"] = dict(stats)
        _write_json_file(NEWS_STATS_CACHE_PATH, {"cache_key": cache_key, **stats})
        return stats
    except Exception:
        return default_stats


def update_learning_buffer(news_text, true_label):
    """將新樣本添加到增量學習緩衝區"""
    try:
        # 只學習與 BTC/ETH 相關的新聞
        if not _is_crypto_relevant_news(news_text):
            return

        prepared = _prepare_news_text_for_model(news_text)
        clean_label = _sanitize_news_label(true_label)
        if not prepared or clean_label is None:
            return

        buffer = []
        try:
            with open(NEWS_LEARNING_BUFFER, "rb") as f:
                buffer = pickle.load(f)
        except Exception:
            buffer = []

        if not isinstance(buffer, list):
            buffer = []

        buffer.append((prepared, clean_label))

        deduped = []
        seen = set()
        for text, label in reversed(buffer):
            key = (str(text), _sanitize_news_label(label))
            if not key[0] or key[1] is None or key in seen:
                continue
            seen.add(key)
            deduped.append((key[0], key[1]))

        deduped.reverse()

        # 緩衝區最多保留 200 個樣本
        if len(deduped) > 200:
            deduped = deduped[-200:]

        _write_pickle_atomic(NEWS_LEARNING_BUFFER, deduped)
    except Exception:
        pass


def incremental_train_news_model():
    """增量學習：結合原始訓練數據 + 學習緩衝區新樣本進行重新訓練"""
    global news_model, news_vectorizer
    news_model = None
    news_vectorizer = None

    try:
        train_news_model()
        meta = _read_json_file(NEWS_MODEL_META_PATH, {})
        print(f"✓ 增量學習完成：使用 {meta.get('sample_count', len(NEWS_TRAINING_DATA))} 個樣本重新訓練模型")
    except Exception as e:
        print(f"✗ 增量學習失敗: {e}")


# 新聞情緒/事件分析（更穩定的分類）
def analyze_news_text(raw_text, log_result=True):
    """更穩定的新聞分類：拆分多空 / 事件 / 影響，避免單一關鍵字誤判。"""
    raw_text = str(raw_text or "").strip()
    text = _prepare_news_text_for_model(raw_text)
    if not text:
        return {
            "sentiment": "中性",
            "impact": "影響有限",
            "bias": 0,
            "event_risk": 0,
            "score": 0,
            "ai_bias": 0,
            "ai_confidence": 0.33,
            "tags": ["empty_text"],
            "is_event": False,
            "fusion_method": "empty_text",
        }

    # ===== 直接使用 AI 模型判斷，不再依賴關鍵字規則 =====
    ai_bias, ai_confidence = predict_news_sentiment_with_confidence(text)
    tags = [f"ai_conf:{ai_confidence:.2f}"]
    fusion_note = "ai_only"
    final_bias = _refine_neutral_bias(text, ai_bias, ai_confidence)
    event_risk = 0

    if ai_confidence < 0.4:
        tags.append("low_confidence")
    else:
        tags.append("high_confidence")

    if final_bias >= 2:
        sentiment = "偏多 (強)"
        impact = "利多（價格可能上漲）"
    elif final_bias == 1:
        sentiment = "偏多"
        impact = "輕微利多（偏正面）"
    elif final_bias == 0:
        sentiment = "中性"
        impact = "影響有限"
    elif final_bias == -1:
        sentiment = "偏空"
        impact = "輕微利空（偏負面）"
    else:
        sentiment = "偏空 (強)"
        impact = "利空（價格可能下跌）"

    if log_result:
        log_prediction_result(raw_text, final_bias, ai_confidence=ai_confidence)

    return {
        "sentiment": sentiment,
        "impact": impact,
        "bias": final_bias,
        "event_risk": event_risk,
        "score": final_bias,
        "ai_bias": ai_bias,
        "ai_confidence": ai_confidence,
        "tags": tags,
        "is_event": False,
        "fusion_method": fusion_note,
    }



# 新增: 標準化新聞訊息格式 + 顯示 AI 學習進度
def build_news_message(news_text, now_time=None):
    if now_time is None:
        now_time = datetime.datetime.now().strftime("%H:%M:%S")

    source = "News"
    if str(news_text).startswith("[CoinDesk]"):
        source = "CoinDesk"
    elif str(news_text).startswith("[Cointelegraph]"):
        source = "Cointelegraph"

    raw_text = re.sub(r"^\[[^\]]+\]\s*", "", str(news_text)).strip()
    zh_text = translate_news_to_zh(raw_text)

    # ===== AI 交易解讀 =====
    analysis = analyze_news_text(raw_text)
    sentiment = analysis["sentiment"]
    impact = analysis["impact"]
    confidence = analysis["ai_confidence"]
    
    # ===== 顯示 AI 學習狀態 =====
    accuracy_info = get_prediction_accuracy()
    accuracy_str = f"準率: {accuracy_info['accuracy']}% ({accuracy_info['correct']}/{accuracy_info['total']})" if accuracy_info['total'] > 0 else "準率: 初始化中"

    return (
        f"🟡 市場快訊（即時）\n"
        f"⏰ {now_time}\n"
        f"━━━━━━━━━━━━━━\n"
        f"來源: {source}\n"
        f"📊 解讀: {sentiment}\n"
        f"🎯 置信度: {confidence:.1%}\n"
        f"🧠 {accuracy_str}\n"
        f"🔥 影響: {impact}\n"
        f"🌐 新聞(中文): {zh_text}\n"
        f"📝 原文: {raw_text}\n"
        f"━━━━━━━━━━━━━━"
    )


NEWS_LOCATION_HINTS = [
    (("canada", "air canada", "加拿大"), "加拿大", 56.1304, -106.3468),
    (("united states", "u.s.", " us ", "美國", "華爾街", "wall street", "new york", "紐約"), "美國", 39.8283, -98.5795),
    (("washington", "white house", "federal reserve", "fed", "白宮", "聯準會"), "華盛頓", 38.9072, -77.0369),
    (("china", "beijing", "中國", "北京"), "中國", 35.8617, 104.1954),
    (("taiwan", "台灣", "臺灣"), "台灣", 23.6978, 120.9605),
    (("japan", "tokyo", "日本", "東京"), "日本", 36.2048, 138.2529),
    (("south korea", "korea", "seoul", "韓國", "首爾"), "韓國", 35.9078, 127.7669),
    (("india", "new delhi", "印度"), "印度", 20.5937, 78.9629),
    (("pakistan", "巴基斯坦"), "巴基斯坦", 30.3753, 69.3451),
    (("iran", "tehran", "伊朗"), "伊朗", 32.4279, 53.6880),
    (("israel", "gaza", "jerusalem", "以色列", "加薩", "耶路撒冷"), "以色列/加薩", 31.0461, 34.8516),
    (("syria", "damascus", "敘利亞", "大馬士革"), "敘利亞", 34.8021, 38.9968),
    (("russia", "moscow", "俄羅斯", "莫斯科"), "俄羅斯", 61.5240, 105.3188),
    (("ukraine", "kyiv", "烏克蘭", "基輔"), "烏克蘭", 48.3794, 31.1656),
    (("united kingdom", "britain", "london", "英國", "倫敦"), "英國", 55.3781, -3.4360),
    (("france", "paris", "法國", "巴黎"), "法國", 46.2276, 2.2137),
    (("germany", "berlin", "德國", "柏林"), "德國", 51.1657, 10.4515),
    (("europe", "eurozone", "eu ", "歐洲", "歐盟", "歐元區"), "歐洲", 54.5260, 15.2551),
    (("zambia", "尚比亞", "贊比亞"), "尚比亞", -13.1339, 27.8493),
    (("south africa", "南非"), "南非", -30.5595, 22.9375),
    (("brazil", "巴西"), "巴西", -14.2350, -51.9253),
    (("mexico", "墨西哥"), "墨西哥", 23.6345, -102.5528),
    (("australia", "澳洲", "澳大利亞"), "澳洲", -25.2744, 133.7751),
    (("saudi", "riyadh", "沙烏地", "沙特"), "沙烏地阿拉伯", 23.8859, 45.0792),
]


def infer_news_location(title, title_zh="", source=""):
    haystack = f" {title or ''} {title_zh or ''} {source or ''} ".lower()
    for keys, name, lat, lon in NEWS_LOCATION_HINTS:
        if any(str(key).lower() in haystack for key in keys):
            return {"location": name, "lat": lat, "lon": lon}
    return {}


def build_panel_news_items(news_list, limit=5):
    items = []
    seen = set()

    for raw_item in list(news_list or [])[: max(limit * 2, limit)]:
        raw_text = str(raw_item or "").strip()
        if not raw_text:
            continue

        match = re.match(r"^\[([^\]]+)\]\s*(.*)$", raw_text)
        if match:
            source = match.group(1).strip() or "News"
            title = match.group(2).strip()
        else:
            source = "News"
            title = raw_text

        if not title:
            continue

        key = normalize_news_text(title).lower()
        if key in seen:
            continue
        seen.add(key)

        analysis = analyze_news_text(title, log_result=False)
        bias = _safe_int(analysis.get("bias"), 0) if isinstance(analysis, dict) else 0
        confidence = _safe_float(analysis.get("confidence"), 0.0) if isinstance(analysis, dict) else 0.0
        title_zh = translate_news_to_zh(title)
        item = {
            "source": source[:40],
            "title": title[:220],
            "title_zh": str(title_zh or title)[:220],
            "bias": bias,
            "confidence": round(confidence, 4),
            "ts": int(time.time()),
        }
        item.update(infer_news_location(title, title_zh, source))
        items.append(item)

        if len(items) >= limit:
            break

    return items


def _walk_strings(obj, limit=3000):
    """遞迴抽取 JSON / list / dict 內所有字串，避免 Binance 改版就整個抓不到。"""
    out = []
    q = deque([obj])

    while q and len(out) < limit:
        cur = q.popleft()

        if isinstance(cur, dict):
            for v in cur.values():
                q.append(v)
        elif isinstance(cur, list):
            for v in cur:
                q.append(v)
        elif isinstance(cur, str):
            s = normalize_news_text(cur)
            if s:
                out.append(s)

    return out


def _looks_like_news_title(text):
    low = text.lower().strip()

    bad_keywords = [
        "login", "sign up", "cookie", "terms of use", "notification center",
        "history", "creator center", "download app", "markets overview",
        "english", "discover", "bookmarks", "platform t&cs", "sitemap",
        "risk warning", "privacy", "copyright", "all rights reserved",
        "binance app", "binance feed", "latest binance news"
    ]

    if len(text) < 18 or len(text) > 220:
        return False
    if any(k in low for k in bad_keywords):
        return False
    if low.count("binance") >= 4:
        return False
    if low.startswith("/") or low.startswith("{") or low.startswith("["):
        return False
    if text.count("|") >= 4:
        return False

    # 至少要像標題，不要只是零碎 UI 字串
    has_signal_word = any(k in low for k in [
        "will", "launch", "list", "listing", "delist", "support", "upgrade",
        "airdrop", "futures", "alpha", "sec", "etf", "bitcoin", "ether",
        "crypto", "market", "fed", "cpi", "inflation", "tariff", "trump",
        "token", "partnership", "hack", "exploit", "lawsuit", "outflow",
        "inflow", "surge", "drop", "plunge", "rally", "approval",
        "fomc", "powell", "pce", "jobs", "nonfarm", "rate cut", "rate hike"
    ])

    word_count = len(text.split())
    return has_signal_word or word_count >= 5


def _extract_binance_titles_from_html(body):
    """先抓 script JSON，再退回 regex，盡量避免因頁面改版而抓不到。"""
    candidates = []

    # 1) 優先解析 script 內嵌 JSON
    script_patterns = [
        r'<script[^>]*id="__APP_DATA"[^>]*>(.*?)</script>',
        r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>',
        r'window\.__APP_DATA__\s*=\s*(\{.*?\})\s*;</script>',
        r'window\.__INITIAL_STATE__\s*=\s*(\{.*?\})\s*;</script>',
    ]

    for pattern in script_patterns:
        for raw in re.findall(pattern, body, flags=re.S):
            raw = raw.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
                for s in _walk_strings(parsed):
                    if _looks_like_news_title(s):
                        candidates.append(s)
            except Exception:
                continue

    # 2) regex 後備方案
    title_patterns = [
        r'"title":"([^"\\]{12,220}(?:\\.[^"\\]*)*)"',
        r'"headline":"([^"\\]{12,220}(?:\\.[^"\\]*)*)"',
        r'"subTitle":"([^"\\]{12,220}(?:\\.[^"\\]*)*)"',
        r'>([^<>]{18,220})</a>',
        r'>([^<>]{18,220})</h3>',
        r'>([^<>]{18,220})</h2>',
    ]

    for pattern in title_patterns:
        for raw in re.findall(pattern, body, flags=re.S):
            text = normalize_news_text(raw)
            if _looks_like_news_title(text):
                candidates.append(text)

    return candidates


# === RSS/ATOM 快訊聚合 ===

def _looks_like_macro_news_title(text):
    low = text.lower().strip()

    bad_keywords = [
        "podcast", "newsletter", "advertisement", "sponsored", "opinion",
        "privacy policy", "terms of use", "sign up", "subscribe", "contact us"
    ]

    if len(text) < 18 or len(text) > 240:
        return False
    if any(k in low for k in bad_keywords):
        return False

    return any(k in low for k in [
        "bitcoin", "btc", "ether", "eth", "ethereum", "crypto", "binance",
        "sec", "etf", "fed", "fomc", "powell", "inflation", "cpi", "tariff",
        "market", "token", "listing", "delist", "hack", "exploit", "lawsuit",
        "approval", "approved", "debut", "launch", "listing", "surge", "drop",
        "plunge", "rally", "outflow", "inflow", "ceasefire", "war", "sanction",
        "bank", "banks", "digital franc", "institutional"
    ]) or len(text.split()) >= 6


def fetch_rss_news(feed_url, source_name):
    """抓 RSS / Atom，回傳 [{source, text}]，比抓 HTML 穩定很多。"""
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9, */*;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache"
    }

    res = HTTP_SESSION.get(feed_url, headers=headers, timeout=8)
    res.raise_for_status()

    body = res.text.strip()
    if not body:
        return []

    results = []
    seen = set()

    try:
        root = ET.fromstring(body)
    except Exception:
        return []

    channel_items = root.findall(".//item")
    atom_entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")

    for item in channel_items:
        title_el = item.find("title")
        title = normalize_news_text(title_el.text if title_el is not None else "")
        if not title or not _looks_like_macro_news_title(title):
            continue
        low = title.lower()
        if low in seen:
            continue
        seen.add(low)
        results.append({"source": source_name, "text": title})

    for entry in atom_entries:
        title_el = entry.find("{http://www.w3.org/2005/Atom}title")
        title = normalize_news_text(title_el.text if title_el is not None else "")
        if not title or not _looks_like_macro_news_title(title):
            continue
        low = title.lower()
        if low in seen:
            continue
        seen.add(low)
        results.append({"source": source_name, "text": title})

    return results[:30]


def fetch_macro_rss_news():
    """聚合較穩定的 RSS / Atom 快訊來源。"""
    feeds = [
        # 1. Investing（新聞）- 替換失效鏈接
        ("https://www.investing.com/rss/news.rss", "Investing"),
        ("https://www.investing.com/rss/news_25.rss", "Investing Crypto"),
        ("https://www.investing.com/rss/news_301.rss", "Investing Commodities"),
        ("https://www.investing.com/rss/news_1.rss", "Investing Forex"),
        # 移除失效的 news_6.rss，改用更可靠的來源

        # 2. 替代新聞來源
        ("https://feeds.bloomberg.com/markets/news.rss", "Bloomberg"),
        ("https://www.cnbc.com/id/100003114/device/rss/rss.html", "CNBC"),
        ("https://finance.yahoo.com/rss/", "Yahoo Finance"),

        # 3. 加密貨幣新聞
        ("https://www.coindesk.com/arc/outboundfeeds/rss/", "CoinDesk"),
        ("https://cointelegraph.com/rss", "Cointelegraph"),

        # 4. 外匯分析
        ("https://www.investing.com/rss/forex.rss", "Technical Analysis"),
    ]
    if _is_truthy(os.getenv("RSS_ENABLE_FOREXLIVE", "0")):
        feeds.append(("https://www.forexlive.com/feed/", "ForexLive"))

    aggregated = []
    for feed_url, source_name in feeds:
        now_feed = time.time()
        cooldown_key = f"rss_cooldown_until_{source_name.lower()}"
        cooldown_until = _safe_float(getattr(fetch_macro_rss_news, cooldown_key, 0.0), 0.0)
        if cooldown_until > now_feed:
            continue
        try:
            aggregated.extend(fetch_rss_news(feed_url, source_name))
            setattr(fetch_macro_rss_news, f"rss_fail_count_{source_name.lower()}", 0)
        except Exception as e:
            now_err = now_feed
            key = f"rss_err_{source_name.lower()}"
            fail_key = f"rss_fail_count_{source_name.lower()}"
            fail_count = _safe_int(getattr(fetch_macro_rss_news, fail_key, 0), 0) + 1
            setattr(fetch_macro_rss_news, fail_key, fail_count)
            if fail_count >= max(2, _safe_int(os.getenv("RSS_SOURCE_COOLDOWN_FAILS", 3), 3)):
                cooldown_sec = max(300, _safe_int(os.getenv("RSS_SOURCE_FAIL_COOLDOWN_SEC", 3600), 3600))
                setattr(fetch_macro_rss_news, cooldown_key, now_err + cooldown_sec)
            last_err = getattr(fetch_macro_rss_news, key, 0)
            if now_err - last_err > max(300, _safe_int(os.getenv("RSS_ERROR_LOG_INTERVAL_SEC", 1800), 1800)):
                print(f"⚠️ {source_name} RSS 暫時略過，已進入冷卻:", repr(e))
                setattr(fetch_macro_rss_news, key, now_err)

    dedup = []
    seen = set()
    for item in aggregated:
        if not isinstance(item, dict):
            continue
        src = str(item.get("source", "RSS")).strip() or "RSS"
        text = normalize_news_text(item.get("text", ""))
        if not text:
            continue
        key = f"{src}|{text.lower()}"
        if key in seen:
            continue
        seen.add(key)
        dedup.append({"source": src, "text": text})

    return dedup[:50]


# 新增: 獨立刷新 RSS 新聞快取
def refresh_rss_news_cache(force=False):
    """獨立刷新 RSS 新聞快取，避免整個 macro 每 3 秒都重抓新聞。"""
    global NEWS_CACHE

    now = time.time()
    if not force and now - NEWS_CACHE.get("ts", 0) < 20:
        return NEWS_CACHE.get("news", 0), NEWS_CACHE.get("event", 0), NEWS_CACHE.get("news_list", [])

    news_bias = 0
    event_risk = 0
    news_list = NEWS_CACHE.get("news_list", [])

    try:
        aggregated_items = fetch_macro_rss_news()
        if not aggregated_items:
            NEWS_CACHE = {
                "news": NEWS_CACHE.get("news", 0),
                "event": NEWS_CACHE.get("event", 0),
                "news_list": NEWS_CACHE.get("news_list", []),
                "ts": now
            }
            return NEWS_CACHE["news"], NEWS_CACHE["event"], NEWS_CACHE["news_list"]

        if not hasattr(refresh_rss_news_cache, "seen_news"):
            refresh_rss_news_cache.seen_news = set()
        if not hasattr(refresh_rss_news_cache, "bootstrapped_news"):
            refresh_rss_news_cache.bootstrapped_news = False

        normalized_items = []
        dedup_now = set()
        for item in aggregated_items:
            if not isinstance(item, dict):
                continue
            src = str(item.get("source", "News")).strip() or "News"
            text = normalize_news_text(item.get("text", ""))
            if not text:
                continue
            key = f"{src}|{text.lower()}"
            if key in dedup_now:
                continue
            dedup_now.add(key)
            normalized_items.append({"source": src, "text": text})

        news_list = []
        news_bias = 0
        event_risk = 0

        # 即使沒有新快訊，也保留近期標題供監控顯示
        latest_news = [f"[{item['source']}] {item['text'][:200]}" for item in normalized_items[:12]]

        if not refresh_rss_news_cache.bootstrapped_news:
            startup_items = normalized_items[:8]
            for item in startup_items:
                src = item["source"]
                text = item["text"]
                refresh_rss_news_cache.seen_news.add(f"{src}|{text}")
                if not _is_crypto_relevant_news(text):
                    continue
                analysis = analyze_news_text(text)
                news_bias += int(analysis.get("bias", 0))
                event_risk += int(analysis.get("event_risk", 0))
                news_list.append(f"[{src}] {text[:200]}")

            for item in normalized_items[8:]:
                refresh_rss_news_cache.seen_news.add(f"{item['source']}|{item['text']}")

            refresh_rss_news_cache.bootstrapped_news = True
        else:
            for item in normalized_items:
                src = item["source"]
                text = item["text"]
                seen_key = f"{src}|{text}"
                if seen_key in refresh_rss_news_cache.seen_news:
                    continue

                refresh_rss_news_cache.seen_news.add(seen_key)
                if not _is_crypto_relevant_news(text):
                    continue
                analysis = analyze_news_text(text)
                news_bias += int(analysis.get("bias", 0))
                event_risk += int(analysis.get("event_risk", 0))
                news_list.append(f"[{src}] {text[:200]}")

        # 若本輪沒有新快訊，回退為近期標題（僅保留加密相關），避免監控面板長期顯示「暫無資料」
        if not news_list and latest_news:
            news_list = [n for n in latest_news if _is_crypto_relevant_news(n)]

        if len(refresh_rss_news_cache.seen_news) > 4000:
            refresh_rss_news_cache.seen_news = set(list(refresh_rss_news_cache.seen_news)[-2000:])

        news_bias = max(-3, min(news_bias, 3))
        event_risk = min(event_risk, 3)

        NEWS_CACHE = {
            "news": news_bias,
            "event": event_risk,
            "news_list": news_list,
            "ts": now
        }
        return news_bias, event_risk, news_list

    except Exception as e:
        last_err = getattr(refresh_rss_news_cache, "last_err_ts", 0)
        if now - last_err > 60:
            print("⚠️ RSS refresh error:", repr(e), "| use cached news")
            refresh_rss_news_cache.last_err_ts = now
        NEWS_CACHE = {
            "news": NEWS_CACHE.get("news", 0),
            "event": NEWS_CACHE.get("event", 0),
            "news_list": NEWS_CACHE.get("news_list", []),
            "ts": now
        }
        return NEWS_CACHE["news"], NEWS_CACHE["event"], NEWS_CACHE["news_list"]


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def _default_derivatives_flow_snapshot():
    return {
        "ts": 0,
        "symbol": COPY_TRADE_SYMBOL,
        "open_interest": 0.0,
        "open_interest_change": 0.0,
        "mark_price": 0.0,
        "index_price": 0.0,
        "mark_premium_rate": 0.0,
        "funding_rate_live": 0.0,
        "funding_next_ts": 0,
        "taker_buy_ratio": 0.5,
        "long_short_ratio": 1.0,
        "derivatives_pressure": 0.0,
        "stale": False,
    }


def _normalize_derivatives_flow_snapshot(snapshot):
    payload = _default_derivatives_flow_snapshot()
    if isinstance(snapshot, dict):
        payload.update(snapshot)

    payload["symbol"] = str(payload.get("symbol") or COPY_TRADE_SYMBOL).upper()
    payload["open_interest"] = max(0.0, _safe_float(payload.get("open_interest"), 0.0))
    payload["open_interest_change"] = max(-1.0, min(1.0, _safe_float(payload.get("open_interest_change"), 0.0)))
    payload["mark_price"] = max(0.0, _safe_float(payload.get("mark_price"), 0.0))
    payload["index_price"] = max(0.0, _safe_float(payload.get("index_price"), 0.0))
    payload["mark_premium_rate"] = max(-0.05, min(0.05, _safe_float(payload.get("mark_premium_rate"), 0.0)))
    payload["funding_rate_live"] = max(-0.05, min(0.05, _safe_float(payload.get("funding_rate_live"), 0.0)))
    payload["funding_next_ts"] = _safe_int(payload.get("funding_next_ts"), 0)
    payload["taker_buy_ratio"] = max(0.0, min(1.0, _safe_float(payload.get("taker_buy_ratio"), 0.5)))
    payload["long_short_ratio"] = max(0.0, _safe_float(payload.get("long_short_ratio"), 1.0))
    payload["derivatives_pressure"] = max(-1.0, min(1.0, _safe_float(payload.get("derivatives_pressure"), 0.0)))
    payload["ts"] = _safe_int(payload.get("ts"), 0)
    payload["stale"] = bool(payload.get("stale", False))
    return payload


def get_derivatives_flow_snapshot(symbol=COPY_TRADE_SYMBOL, force=False):
    """Public Binance Futures flow snapshot; fail-soft so signal generation never blocks on it."""
    now_ts = time.time()
    symbol = str(symbol or COPY_TRADE_SYMBOL).upper()

    with DERIVATIVES_FLOW_CACHE_LOCK:
        cached_ts = _safe_float(DERIVATIVES_FLOW_CACHE.get("ts"), 0.0)
        cached_snapshot = dict(DERIVATIVES_FLOW_CACHE.get("snapshot") or {})
        if (
            (not force)
            and cached_snapshot
            and str(cached_snapshot.get("symbol") or "").upper() == symbol
            and (now_ts - cached_ts) < DERIVATIVES_FLOW_CACHE_TTL_SEC
        ):
            return _normalize_derivatives_flow_snapshot(cached_snapshot)

    previous = _normalize_derivatives_flow_snapshot(cached_snapshot) if cached_snapshot else _default_derivatives_flow_snapshot()
    snapshot = _default_derivatives_flow_snapshot()
    snapshot["symbol"] = symbol
    snapshot["ts"] = int(now_ts)

    if not ALLOW_BINANCE_DERIVATIVES_MARKET_DATA:
        snapshot.update(previous)
        snapshot["symbol"] = symbol
        snapshot["ts"] = int(now_ts)
        snapshot["stale"] = True
        with DERIVATIVES_FLOW_CACHE_LOCK:
            DERIVATIVES_FLOW_CACHE["ts"] = now_ts
            DERIVATIVES_FLOW_CACHE["snapshot"] = dict(snapshot)
        return _normalize_derivatives_flow_snapshot(snapshot)

    try:
        oi_resp = _binance_request_get(
            "https://fapi.binance.com/fapi/v1/openInterest",
            params={"symbol": symbol},
            timeout=5,
            prefix="Binance derivatives openInterest",
        )
        oi_resp.raise_for_status()
        oi_data = oi_resp.json()
        current_oi = max(0.0, _safe_float((oi_data or {}).get("openInterest"), 0.0))
        previous_oi = _safe_float(previous.get("open_interest"), 0.0)
        snapshot["open_interest"] = current_oi
        if previous_oi > 0 and current_oi > 0:
            snapshot["open_interest_change"] = max(-1.0, min(1.0, (current_oi - previous_oi) / previous_oi))

        premium_resp = _binance_request_get(
            "https://fapi.binance.com/fapi/v1/premiumIndex",
            params={"symbol": symbol},
            timeout=5,
            prefix="Binance derivatives premiumIndex",
        )
        premium_resp.raise_for_status()
        premium_data = premium_resp.json()
        mark_price = max(0.0, _safe_float((premium_data or {}).get("markPrice"), 0.0))
        index_price = max(0.0, _safe_float((premium_data or {}).get("indexPrice"), 0.0))
        snapshot["mark_price"] = mark_price
        snapshot["index_price"] = index_price
        if mark_price > 0 and index_price > 0:
            snapshot["mark_premium_rate"] = max(-0.05, min(0.05, (mark_price - index_price) / index_price))
        snapshot["funding_rate_live"] = max(-0.05, min(0.05, _safe_float((premium_data or {}).get("lastFundingRate"), 0.0)))
        snapshot["funding_next_ts"] = _safe_int((premium_data or {}).get("nextFundingTime"), 0)

        taker_resp = _binance_request_get(
            "https://fapi.binance.com/futures/data/takerlongshortRatio",
            params={"symbol": symbol, "period": "5m", "limit": 2},
            timeout=5,
            prefix="Binance derivatives takerRatio",
        )
        taker_resp.raise_for_status()
        taker_rows = taker_resp.json()
        latest_taker = taker_rows[-1] if isinstance(taker_rows, list) and taker_rows else {}
        long_short_ratio = max(0.0, _safe_float(latest_taker.get("buySellRatio"), 1.0))
        buy_vol = max(0.0, _safe_float(latest_taker.get("buyVol"), 0.0))
        sell_vol = max(0.0, _safe_float(latest_taker.get("sellVol"), 0.0))
        total_taker_vol = buy_vol + sell_vol
        taker_buy_ratio = (buy_vol / total_taker_vol) if total_taker_vol > 0 else (long_short_ratio / (1.0 + long_short_ratio) if long_short_ratio > 0 else 0.5)
        snapshot["long_short_ratio"] = long_short_ratio
        snapshot["taker_buy_ratio"] = max(0.0, min(1.0, taker_buy_ratio))

        taker_pressure = (snapshot["taker_buy_ratio"] - 0.5) * 2.0
        oi_pressure = max(-1.0, min(1.0, snapshot["open_interest_change"] * 25.0))
        premium_pressure = max(-1.0, min(1.0, snapshot["mark_premium_rate"] * 800.0))
        crowded_funding = max(-1.0, min(1.0, snapshot["funding_rate_live"] * 500.0))
        snapshot["derivatives_pressure"] = max(
            -1.0,
            min(
                1.0,
                taker_pressure * 0.55
                + oi_pressure * 0.30
                + premium_pressure * 0.10
                - crowded_funding * 0.05,
            ),
        )

        snapshot = _normalize_derivatives_flow_snapshot(snapshot)
        with DERIVATIVES_FLOW_CACHE_LOCK:
            DERIVATIVES_FLOW_CACHE["ts"] = now_ts
            DERIVATIVES_FLOW_CACHE["snapshot"] = dict(snapshot)
        return snapshot
    except Exception as e:
        fallback = previous if cached_snapshot else snapshot
        fallback = _normalize_derivatives_flow_snapshot(fallback)
        fallback["stale"] = True
        fallback["ts"] = int(now_ts)
        if now_ts - _safe_float(getattr(get_derivatives_flow_snapshot, "_last_err_ts", 0.0), 0.0) > 300:
            print(f"⚠️ 衍生品資料讀取失敗: {e}")
            get_derivatives_flow_snapshot._last_err_ts = now_ts
        return fallback


def _sanitize_pending_news_eval_item(item):
    if not isinstance(item, dict):
        return None

    raw_news = normalize_news_text(item.get("news", item.get("text", "")))
    prepared = _prepare_news_text_for_model(raw_news)
    if not raw_news or not prepared:
        return None

    predicted_bias = _sanitize_news_label(item.get("predicted_bias"))
    if predicted_bias is None:
        return None

    source = str(item.get("source", "News")).strip() or "News"
    entry_price = _safe_float(item.get("entry_price"), 0.0)
    entry_ts = _safe_float(item.get("entry_ts"), 0.0)
    due_ts = _safe_float(item.get("due_ts"), 0.0)
    if entry_price <= 0 or entry_ts <= 0:
        return None
    if due_ts <= entry_ts:
        due_ts = entry_ts + NEWS_EVAL_HORIZON_SEC

    raw_key = str(item.get("news_key", "")).strip().lower()
    news_key = raw_key or f"{source.lower()}|{prepared[:180]}"
    ai_confidence = max(0.0, min(1.0, _safe_float(item.get("ai_confidence"), 0.0)))

    return {
        "news_key": news_key[:220],
        "source": source[:60],
        "news": raw_news[:240],
        "predicted_bias": predicted_bias,
        "ai_confidence": ai_confidence,
        "entry_price": entry_price,
        "entry_ts": entry_ts,
        "due_ts": due_ts,
    }


def _load_pending_news_eval_queue():
    global NEWS_EVAL_PENDING

    if isinstance(NEWS_EVAL_PENDING, list):
        return NEWS_EVAL_PENDING

    try:
        with open(NEWS_EVAL_PENDING_PATH, "rb") as f:
            raw_queue = pickle.load(f)
    except Exception:
        raw_queue = []

    if not isinstance(raw_queue, list):
        raw_queue = []

    queue = []
    seen = set()
    for item in raw_queue:
        clean = _sanitize_pending_news_eval_item(item)
        if clean is None:
            continue
        dedupe_key = (clean["news_key"], int(clean["entry_ts"]))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        queue.append(clean)

    NEWS_EVAL_PENDING = queue[-NEWS_EVAL_QUEUE_MAX:]
    if len(NEWS_EVAL_PENDING) != len(raw_queue):
        _save_pending_news_eval_queue(NEWS_EVAL_PENDING)
    return NEWS_EVAL_PENDING


def _save_pending_news_eval_queue(queue=None):
    global NEWS_EVAL_PENDING

    if queue is None:
        queue = NEWS_EVAL_PENDING if isinstance(NEWS_EVAL_PENDING, list) else []

    clean_queue = []
    for item in queue:
        clean = _sanitize_pending_news_eval_item(item)
        if clean is not None:
            clean_queue.append(clean)

    NEWS_EVAL_PENDING = clean_queue[-NEWS_EVAL_QUEUE_MAX:]
    try:
        _write_pickle_atomic(NEWS_EVAL_PENDING_PATH, NEWS_EVAL_PENDING)
    except Exception as e:
        print(f"⚠️ 儲存新聞待驗證隊列失敗: {e}")


def _queue_news_prediction_for_evaluation(news_text, predicted_bias, ai_confidence=None, source="News"):
    if not INCREMENTAL_LEARNING_ENABLED:
        return False

    raw_news = normalize_news_text(news_text)
    prepared = _prepare_news_text_for_model(raw_news)
    clean_bias = _sanitize_news_label(predicted_bias)
    entry_price = _safe_float(WS_PRICE, 0.0)
    now_ts = time.time()
    if not raw_news or not prepared or clean_bias is None or entry_price <= 0:
        return False

    queue = _load_pending_news_eval_queue()
    news_key = f"{str(source or 'News').strip().lower()}|{prepared[:180]}"

    for item in reversed(queue):
        if item.get("news_key") != news_key:
            continue
        if now_ts - _safe_float(item.get("entry_ts"), 0.0) < NEWS_EVAL_HORIZON_SEC:
            return False
        break

    queue.append(
        {
            "news_key": news_key,
            "source": str(source or "News"),
            "news": raw_news,
            "predicted_bias": clean_bias,
            "ai_confidence": max(0.0, min(1.0, _safe_float(ai_confidence, 0.0))),
            "entry_price": entry_price,
            "entry_ts": now_ts,
            "due_ts": now_ts + NEWS_EVAL_HORIZON_SEC,
        }
    )
    _save_pending_news_eval_queue(queue)
    return True


def _classify_news_market_move(move_rate):
    move = _safe_float(move_rate, 0.0)
    abs_move = abs(move)
    if abs_move < NEWS_EVAL_MIN_MOVE_RATE:
        return 0
    if move >= NEWS_EVAL_STRONG_MOVE_RATE:
        return 2
    if move >= NEWS_EVAL_MIN_MOVE_RATE:
        return 1
    if move <= -NEWS_EVAL_STRONG_MOVE_RATE:
        return -2
    return -1


def _is_news_prediction_correct(predicted_bias, actual_bias):
    predicted = _sanitize_news_label(predicted_bias)
    actual = _sanitize_news_label(actual_bias)
    if predicted is None or actual is None:
        return None
    if predicted == 0 or actual == 0:
        return predicted == actual
    return (predicted > 0) == (actual > 0)


def _maybe_retrain_news_model(labeled_total=None):
    if not INCREMENTAL_LEARNING_ENABLED:
        return False

    total = max(0, _safe_int(labeled_total, 0))
    bucket = total // max(1, MIN_PREDICTIONS_FOR_RETRAIN)
    if bucket <= 0:
        return False

    state = getattr(_maybe_retrain_news_model, "_state", {"last_bucket": 0, "last_run_ts": 0.0})
    now_ts = time.time()
    if bucket <= _safe_int(state.get("last_bucket"), 0):
        return False
    if (now_ts - _safe_float(state.get("last_run_ts"), 0.0)) < NEWS_RETRAIN_MIN_INTERVAL_SEC:
        return False

    incremental_train_news_model()
    _maybe_retrain_news_model._state = {
        "last_bucket": bucket,
        "last_run_ts": now_ts,
    }
    return True


def _process_pending_news_evaluations(current_price):
    now_ts = time.time()
    last_run_ts = _safe_float(getattr(_process_pending_news_evaluations, "_last_run_ts", 0.0), 0.0)
    if (now_ts - last_run_ts) < NEWS_EVAL_PROCESS_INTERVAL_SEC:
        return 0
    _process_pending_news_evaluations._last_run_ts = now_ts

    price_ref = _safe_float(current_price, 0.0) or _safe_float(WS_PRICE, 0.0)
    if price_ref <= 0:
        return 0

    queue = _load_pending_news_eval_queue()
    if not queue:
        return 0

    keep = []
    evaluated = 0
    stale = 0
    for item in queue:
        due_ts = _safe_float(item.get("due_ts"), 0.0)
        if due_ts > now_ts:
            keep.append(item)
            continue

        overdue_sec = now_ts - due_ts
        if overdue_sec > NEWS_EVAL_MAX_OVERDUE_SEC:
            stale += 1
            continue

        entry_price = _safe_float(item.get("entry_price"), 0.0)
        if entry_price <= 0:
            continue

        actual_move = (price_ref - entry_price) / max(entry_price, 1e-9)
        actual_bias = _classify_news_market_move(actual_move)
        predicted_bias = _sanitize_news_label(item.get("predicted_bias"))
        correct = _is_news_prediction_correct(predicted_bias, actual_bias)

        log_prediction_result(
            item.get("news", ""),
            predicted_bias,
            actual_market_move=round(actual_move, 6),
            correct=correct,
            actual_bias=actual_bias,
            ai_confidence=item.get("ai_confidence"),
            source=item.get("source", "News"),
            schedule_eval=False,
        )
        update_learning_buffer(item.get("news", ""), actual_bias)
        evaluated += 1

    if len(keep) != len(queue):
        _save_pending_news_eval_queue(keep)

    if stale > 0:
        print(f"🧹 已清除 {stale} 筆過期新聞驗證樣本")

    if evaluated > 0:
        stats = get_prediction_accuracy()
        retrained = _maybe_retrain_news_model(stats.get("total", 0))
        print(
            f"🧠 新聞驗證完成: {evaluated} 筆 | 準率 {stats.get('accuracy', 0)}% "
            f"({stats.get('correct', 0)}/{stats.get('total', 0)})"
            + (" | 已重訓新聞模型" if retrained else "")
        )

    return evaluated


def _is_truthy(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _is_real_copy_enabled() -> bool:
    return _is_truthy(os.getenv("BINANCE_REAL_COPY_ENABLED", "0"))


def _has_valid_tp_sl(trade) -> bool:
    if not isinstance(trade, dict):
        return False

    direction = str(trade.get("direction") or "")
    entry = _safe_float(trade.get("avg_entry", trade.get("entry")), 0.0)
    tp = _safe_float(trade.get("tp"), 0.0)
    sl = _safe_float(trade.get("sl"), 0.0)

    if entry <= 0 or tp <= 0 or sl <= 0:
        return False
    if direction == "long":
        return tp > entry and sl < tp
    if direction == "short":
        return tp < entry and sl > tp
    return False


def _get_binance_available_balance() -> float:
    """查詢 Binance 合約帳戶可用餘額（USDT）。失敗時回傳 0.0。"""
    snapshot = _get_binance_account_snapshot(log_on_error=True)
    return max(0.0, _safe_float(snapshot.get("available_balance"), 0.0))


def _get_binance_account_snapshot(log_on_error=False):
    snapshot = {
        "available_balance": 0.0,
        "wallet_balance": 0.0,
        "margin_balance": 0.0,
        "unrealized_profit": 0.0,
    }
    try:
        account = _binance_futures_signed_get("/fapi/v2/account")
        if not isinstance(account, dict):
            return snapshot

        snapshot["available_balance"] = max(0.0, _safe_float(account.get("availableBalance"), 0.0))
        snapshot["wallet_balance"] = max(0.0, _safe_float(account.get("totalWalletBalance"), 0.0))
        snapshot["margin_balance"] = max(0.0, _safe_float(account.get("totalMarginBalance"), 0.0))
        snapshot["unrealized_profit"] = _safe_float(account.get("totalUnrealizedProfit"), 0.0)

        for asset in (account.get("assets") or []):
            if not isinstance(asset, dict) or str(asset.get("asset") or "").upper() != "USDT":
                continue
            snapshot["available_balance"] = max(
                snapshot["available_balance"],
                _safe_float(asset.get("availableBalance"), snapshot["available_balance"]),
            )
            snapshot["wallet_balance"] = max(
                snapshot["wallet_balance"],
                _safe_float(asset.get("walletBalance"), snapshot["wallet_balance"]),
            )
            snapshot["margin_balance"] = max(
                snapshot["margin_balance"],
                _safe_float(asset.get("marginBalance"), snapshot["margin_balance"]),
            )
            snapshot["unrealized_profit"] = _safe_float(
                asset.get("unrealizedProfit"),
                snapshot["unrealized_profit"],
            )
            break

        if snapshot["wallet_balance"] <= 0 and snapshot["margin_balance"] > 0:
            snapshot["wallet_balance"] = max(
                0.0,
                snapshot["margin_balance"] - snapshot["unrealized_profit"],
            )
        if snapshot["margin_balance"] <= 0 and snapshot["wallet_balance"] > 0:
            snapshot["margin_balance"] = snapshot["wallet_balance"] + snapshot["unrealized_profit"]
        if snapshot["wallet_balance"] <= 0 and snapshot["available_balance"] > 0:
            snapshot["wallet_balance"] = snapshot["available_balance"]
    except Exception as e:
        if log_on_error:
            print(f"⚠️ 查詢 Binance 餘額失敗: {e}")
    return snapshot


def _get_binance_spot_market_graph(log_on_error=False):
    now_ts = time.time()
    with BINANCE_SPOT_MARKET_CACHE_LOCK:
        cached_ts = _safe_float(BINANCE_SPOT_MARKET_CACHE.get("ts"), 0.0)
        cached_graph = BINANCE_SPOT_MARKET_CACHE.get("graph") or {}
        if cached_graph and (now_ts - cached_ts) < BINANCE_SPOT_MARKET_CACHE_TTL_SEC:
            return cached_graph

    graph = {}
    try:
        exchange_info_res = _binance_request_get(
            "https://api.binance.com/api/v3/exchangeInfo",
            timeout=10,
            prefix="Binance spot exchangeInfo",
        )
        exchange_info = exchange_info_res.json() if exchange_info_res.ok else {}
        price_res = _binance_request_get(
            "https://api.binance.com/api/v3/ticker/price",
            timeout=10,
            prefix="Binance spot ticker",
        )
        prices = price_res.json() if price_res.ok else []

        price_map = {}
        for item in prices if isinstance(prices, list) else []:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol") or "").upper()
            price = _safe_float(item.get("price"), 0.0)
            if symbol and price > 0:
                price_map[symbol] = price

        for item in (exchange_info.get("symbols") or []) if isinstance(exchange_info, dict) else []:
            if not isinstance(item, dict):
                continue
            if str(item.get("status") or "").upper() != "TRADING":
                continue

            symbol = str(item.get("symbol") or "").upper()
            base_asset = str(item.get("baseAsset") or "").upper()
            quote_asset = str(item.get("quoteAsset") or "").upper()
            price = _safe_float(price_map.get(symbol), 0.0)
            if not symbol or not base_asset or not quote_asset or price <= 0:
                continue

            graph.setdefault(base_asset, []).append((quote_asset, price))
            graph.setdefault(quote_asset, []).append((base_asset, 1.0 / price))
    except Exception as e:
        if log_on_error:
            print(f"⚠️ 建立 Binance 現貨價格圖失敗: {e}")

    with BINANCE_SPOT_MARKET_CACHE_LOCK:
        BINANCE_SPOT_MARKET_CACHE["ts"] = now_ts
        BINANCE_SPOT_MARKET_CACHE["graph"] = graph
    return graph


def _convert_binance_asset_to_usdt(asset, amount, graph=None, max_hops=3):
    asset_code = str(asset or "").upper().strip()
    qty = max(0.0, _safe_float(amount, 0.0))
    if not asset_code or qty <= 0:
        return 0.0
    if asset_code == "USDT":
        return qty
    if asset_code in {"USDC", "FDUSD", "BUSD", "TUSD", "USDP"}:
        return qty

    market_graph = graph if isinstance(graph, dict) and graph else _get_binance_spot_market_graph(log_on_error=False)
    if not market_graph:
        return 0.0

    queue = deque([(asset_code, 1.0, 0)])
    visited = {asset_code}
    priority_assets = {"USDT", "USDC", "FDUSD", "BUSD", "BTC", "ETH", "BNB"}

    while queue:
        current_asset, factor, hops = queue.popleft()
        if current_asset == "USDT":
            return qty * factor
        if hops >= max_hops:
            continue

        neighbors = list(market_graph.get(current_asset) or [])
        neighbors.sort(key=lambda item: 0 if item[0] == "USDT" else (1 if item[0] in priority_assets else 2))
        for next_asset, edge_factor in neighbors:
            if next_asset in visited or edge_factor <= 0:
                continue
            visited.add(next_asset)
            queue.append((next_asset, factor * edge_factor, hops + 1))

    return 0.0


def _get_binance_spot_account_snapshot(log_on_error=False):
    snapshot = {
        "spot_total_assets_usdt": 0.0,
        "spot_asset_count": 0,
    }
    try:
        account = _binance_spot_signed_get("/api/v3/account")
        if not isinstance(account, dict):
            return snapshot

        graph = _get_binance_spot_market_graph(log_on_error=log_on_error)
        total_assets_usdt = 0.0
        asset_count = 0
        for item in (account.get("balances") or []):
            if not isinstance(item, dict):
                continue

            asset = str(item.get("asset") or "").upper()
            free_amount = _safe_float(item.get("free"), 0.0)
            locked_amount = _safe_float(item.get("locked"), 0.0)
            total_amount = max(0.0, free_amount + locked_amount)
            if not asset or total_amount <= 0:
                continue

            total_assets_usdt += _convert_binance_asset_to_usdt(asset, total_amount, graph=graph)
            asset_count += 1

        snapshot["spot_total_assets_usdt"] = max(0.0, total_assets_usdt)
        snapshot["spot_asset_count"] = asset_count
    except Exception as e:
        if log_on_error:
            print(f"⚠️ 查詢 Binance 現貨資產失敗: {e}")
    return snapshot


def _get_binance_total_assets_snapshot(log_on_error=False, force=False):
    now_ts = time.time()
    with BINANCE_PANEL_ASSET_CACHE_LOCK:
        cached_ts = _safe_float(BINANCE_PANEL_ASSET_CACHE.get("ts"), 0.0)
        cached_snapshot = dict(BINANCE_PANEL_ASSET_CACHE.get("snapshot") or {})
        if (not force) and (now_ts - cached_ts) < BINANCE_PANEL_ASSET_CACHE_TTL_SEC and cached_snapshot:
            return cached_snapshot

    futures_snapshot = _get_binance_account_snapshot(log_on_error=log_on_error)
    spot_snapshot = _get_binance_spot_account_snapshot(log_on_error=log_on_error)
    futures_total_assets_usdt = max(0.0, _safe_float(futures_snapshot.get("margin_balance"), 0.0))
    spot_total_assets_usdt = max(0.0, _safe_float(spot_snapshot.get("spot_total_assets_usdt"), 0.0))
    snapshot = {
        **futures_snapshot,
        "spot_total_assets_usdt": spot_total_assets_usdt,
        "futures_total_assets_usdt": futures_total_assets_usdt,
        "total_assets_usdt": spot_total_assets_usdt + futures_total_assets_usdt,
        "spot_asset_count": _safe_int(spot_snapshot.get("spot_asset_count"), 0),
    }

    with BINANCE_PANEL_ASSET_CACHE_LOCK:
        BINANCE_PANEL_ASSET_CACHE["ts"] = now_ts
        BINANCE_PANEL_ASSET_CACHE["snapshot"] = dict(snapshot)
    return snapshot


def _refresh_position_panel_account_state(force=False, log_on_error=False):
    snapshot = _get_binance_total_assets_snapshot(log_on_error=log_on_error, force=force)
    POSITION_PANEL_STATE["account_available_balance_usdt"] = max(0.0, _safe_float(snapshot.get("available_balance"), 0.0))
    POSITION_PANEL_STATE["account_wallet_balance_usdt"] = max(0.0, _safe_float(snapshot.get("wallet_balance"), 0.0))
    POSITION_PANEL_STATE["account_margin_balance_usdt"] = max(0.0, _safe_float(snapshot.get("margin_balance"), 0.0))
    POSITION_PANEL_STATE["binance_spot_total_assets_usdt"] = max(0.0, _safe_float(snapshot.get("spot_total_assets_usdt"), 0.0))
    POSITION_PANEL_STATE["binance_futures_total_assets_usdt"] = max(0.0, _safe_float(snapshot.get("futures_total_assets_usdt"), POSITION_PANEL_STATE.get("account_margin_balance_usdt", 0.0)))
    POSITION_PANEL_STATE["binance_total_assets_usdt"] = max(
        0.0,
        _safe_float(
            snapshot.get("total_assets_usdt"),
            POSITION_PANEL_STATE["binance_spot_total_assets_usdt"] + POSITION_PANEL_STATE["binance_futures_total_assets_usdt"],
        ),
    )
    return snapshot


def _compute_capital_usage_ratio(position_margin_usdt, account_snapshot=None):
    position_margin = max(0.0, _safe_float(position_margin_usdt, 0.0))
    if position_margin <= 0:
        return 0.0

    snapshot = account_snapshot if isinstance(account_snapshot, dict) else {}
    available_balance = max(0.0, _safe_float(snapshot.get("available_balance"), 0.0))
    wallet_balance = max(0.0, _safe_float(snapshot.get("wallet_balance"), 0.0))
    margin_balance = max(0.0, _safe_float(snapshot.get("margin_balance"), 0.0))

    capital_base = 0.0
    for candidate in (wallet_balance, margin_balance, available_balance + position_margin):
        if candidate > capital_base:
            capital_base = candidate

    if capital_base <= 0:
        return 0.0
    return max(0.0, position_margin / capital_base)


def _estimate_size_ratio_from_position_margin(position_margin_usdt, account_snapshot=None):
    position_margin = max(0.0, _safe_float(position_margin_usdt, 0.0))
    if position_margin <= 0:
        return 0.0

    snapshot = account_snapshot if isinstance(account_snapshot, dict) else {}
    available_after = max(0.0, _safe_float(snapshot.get("available_balance"), 0.0))
    available_before = available_after + position_margin
    balance_usage_cap = min(1.0, max(0.3, _safe_float(os.getenv("COPY_TRADE_BALANCE_USAGE_CAP", 0.92), 0.92)))
    notional_safety = min(1.0, max(0.7, _safe_float(os.getenv("COPY_TRADE_NOTIONAL_SAFETY", 0.985), 0.985)))
    denom = available_before * balance_usage_cap * notional_safety

    if denom > 1e-9:
        return max(0.0, min(1.0, position_margin / denom))

    capital_usage_ratio = _compute_capital_usage_ratio(position_margin, snapshot)
    if capital_usage_ratio <= 0:
        return 0.0

    scale = max(balance_usage_cap * notional_safety, 1e-6)
    return max(0.0, min(1.0, capital_usage_ratio / scale))


def _get_binance_symbol_leverage(default=DEFAULT_LEV) -> int:
    """讀取 Binance 目前該交易對實際槓桿。"""
    fallback = max(1, _safe_int(default, DEFAULT_LEV))
    try:
        rows = _binance_futures_signed_get("/fapi/v2/positionRisk", {"symbol": COPY_TRADE_SYMBOL})
        if isinstance(rows, list) and rows:
            lev = _safe_int(rows[0].get("leverage"), fallback)
            return max(1, lev)
    except Exception:
        pass
    return fallback


def _calc_copy_trade_qty(size_ratio, leverage=None, eth_price=None) -> float:
    """
    計算實際下單 ETH 數量。
    100% size_ratio = 使用全部可用餘額（以槓桿全開）。
    若查不到餘額，退回 COPY_TRADE_ETH_QTY 舊模式。
    """
    ratio = max(_safe_float(size_ratio, 0.0), 0.0)
    lev = max(
        1,
        min(
            COPY_TRADE_MAX_LEVERAGE,
            _safe_int(leverage or os.getenv("COPY_TRADE_LEVERAGE", DEFAULT_LEV), DEFAULT_LEV),
        ),
    )
    # 預留保證金緩衝，避免 all-in 因手續費/標記價滑動觸發 -2019
    balance_usage_cap = min(1.0, max(0.3, _safe_float(os.getenv("COPY_TRADE_BALANCE_USAGE_CAP", 0.92), 0.92)))
    notional_safety = min(1.0, max(0.7, _safe_float(os.getenv("COPY_TRADE_NOTIONAL_SAFETY", 0.985), 0.985)))

    # 嘗試用帳戶餘額計算
    balance = _get_binance_available_balance()
    price = _safe_float(eth_price, 0.0) if eth_price else _safe_float(WS_PRICE, 0.0)

    if balance > 0 and price > 0:
        # margin = balance * size_ratio（使用比例的可用資金）
        # notional = margin * leverage
        # qty = notional / price
        margin_used = balance * ratio * balance_usage_cap
        raw_qty = margin_used * lev * notional_safety / price
    else:
        # 舊模式 fallback
        base_qty = max(_safe_float(os.getenv("COPY_TRADE_ETH_QTY", 1.0), 1.0), COPY_TRADE_MIN_QTY)
        raw_qty = base_qty * max(ratio, 0.1)

    # ETHUSDT 永續最小步進通常為 0.001，向下取整避免超出精度。
    return max(COPY_TRADE_MIN_QTY, math.floor(raw_qty * 1000.0) / 1000.0)


def _calc_copy_trade_qty_with_buffer(size_ratio, leverage=None, eth_price=None, extra_buffer_ratio=1.0, enforce_min=True) -> float:
    """以額外緩衝重新估算可承受下單量，供 -2019 重試使用。"""
    ratio = max(_safe_float(size_ratio, 0.0), 0.0)
    lev = max(
        1,
        min(
            COPY_TRADE_MAX_LEVERAGE,
            _safe_int(leverage or os.getenv("COPY_TRADE_LEVERAGE", DEFAULT_LEV), DEFAULT_LEV),
        ),
    )
    balance_usage_cap = min(1.0, max(0.3, _safe_float(os.getenv("COPY_TRADE_BALANCE_USAGE_CAP", 0.92), 0.92)))
    notional_safety = min(1.0, max(0.7, _safe_float(os.getenv("COPY_TRADE_NOTIONAL_SAFETY", 0.985), 0.985)))
    buffer_ratio = min(1.0, max(0.3, _safe_float(extra_buffer_ratio, 1.0)))

    balance = _get_binance_available_balance()
    price = _safe_float(eth_price, 0.0) if eth_price else _safe_float(WS_PRICE, 0.0)

    if balance > 0 and price > 0:
        margin_used = balance * ratio * balance_usage_cap * buffer_ratio
        raw_qty = margin_used * lev * notional_safety / price
    else:
        base_qty = max(_safe_float(os.getenv("COPY_TRADE_ETH_QTY", 1.0), 1.0), COPY_TRADE_MIN_QTY)
        raw_qty = base_qty * max(ratio, 0.1) * buffer_ratio

    rounded_qty = math.floor(max(raw_qty, 0.0) * 1000.0) / 1000.0
    if enforce_min:
        return max(COPY_TRADE_MIN_QTY, rounded_qty)
    return rounded_qty


def _execute_binance_market_order_with_retry(order_params, initial_qty, min_qty, retry_steps, retry_decay, recalc_qty_fn=None):
    qty = max(min_qty, math.floor(_safe_float(initial_qty, 0.0) * 1000.0) / 1000.0)
    order_resp = None
    last_err = None

    for attempt in range(retry_steps + 1):
        current_qty = max(min_qty, math.floor(qty * 1000.0) / 1000.0)
        order_params["quantity"] = current_qty
        try:
            order_resp = _binance_futures_signed_request("POST", "/fapi/v1/order", order_params)
            qty = current_qty
            break
        except Exception as e:
            last_err = e
            if not _is_binance_insufficient_margin_error(e):
                break
            if attempt >= retry_steps:
                break

            candidates = []
            decayed_qty = math.floor((current_qty * retry_decay) * 1000.0) / 1000.0
            if decayed_qty > 0:
                candidates.append(decayed_qty)

            if callable(recalc_qty_fn):
                recalculated_qty = _safe_float(recalc_qty_fn(attempt + 1), 0.0)
                if recalculated_qty > 0:
                    candidates.append(math.floor(recalculated_qty * 1000.0) / 1000.0)

            if not candidates:
                break

            next_qty = min(candidates)
            if next_qty < min_qty:
                break
            qty = next_qty

    return order_resp, qty, last_err


def _format_binance_margin_failure(prefix, attempted_qty, leverage, price_ref, last_err):
    available_balance = _get_binance_available_balance()
    price = max(_safe_float(price_ref, 0.0), 0.0)
    approx_required_margin = 0.0
    if _safe_float(attempted_qty, 0.0) > 0 and price > 0 and max(_safe_int(leverage, 1), 1) > 0:
        approx_required_margin = (_safe_float(attempted_qty, 0.0) * price) / max(_safe_int(leverage, 1), 1)

    details = f"{prefix}: {last_err}"
    if available_balance > 0:
        details += f" | 可用保證金約 {available_balance:.2f} USDT"
    if approx_required_margin > 0:
        details += f" | 最後嘗試 {attempted_qty:.3f} ETH 約需保證金 {approx_required_margin:.2f} USDT @ {max(_safe_int(leverage, 1), 1)}x"
    return details


def _is_binance_insufficient_margin_error(err) -> bool:
    text = str(err or "")
    return "-2019" in text or "Margin is insufficient" in text


def _binance_futures_signed_request(method, path, params=None):
    api_key = str(os.getenv("BINANCE_API_KEY", "")).strip()
    api_secret = str(os.getenv("BINANCE_API_SECRET", "")).strip()
    if not api_key or not api_secret:
        raise RuntimeError("未設定 Binance API 金鑰")

    query = dict(params or {})
    query["timestamp"] = int(time.time() * 1000)
    query["recvWindow"] = 5000
    query_string = urlencode(query)
    signature = hmac.new(api_secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256).hexdigest()

    method_upper = str(method).upper()
    if method_upper == "GET":
        request_func = _binance_request_get
    elif method_upper == "POST":
        request_func = _binance_request_post
    elif method_upper == "DELETE":
        request_func = _binance_request_delete
    else:
        raise ValueError(f"Unsupported method: {method}")
    res = request_func(
        f"https://fapi.binance.com{path}",
        params={**query, "signature": signature},
        headers={"X-MBX-APIKEY": api_key},
        timeout=10,
        prefix="Binance futures signed",
    )
    data = res.json()
    if not res.ok:
        raise RuntimeError(f"Binance API 錯誤: {data}")
    if isinstance(data, dict) and data.get("code") not in (None, 0):
        raise RuntimeError(f"Binance API 錯誤: {data}")
    return data


def _binance_futures_signed_get(path, params=None):
    return _binance_futures_signed_request("GET", path, params)


def _binance_spot_signed_request(method, path, params=None):
    api_key = str(os.getenv("BINANCE_API_KEY", "")).strip()
    api_secret = str(os.getenv("BINANCE_API_SECRET", "")).strip()
    if not api_key or not api_secret:
        raise RuntimeError("未設定 Binance API 金鑰")

    query = dict(params or {})
    query["timestamp"] = int(time.time() * 1000)
    query["recvWindow"] = 5000
    query_string = urlencode(query)
    signature = hmac.new(api_secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256).hexdigest()

    method_upper = str(method).upper()
    if method_upper == "GET":
        request_func = _binance_request_get
    elif method_upper == "POST":
        request_func = _binance_request_post
    elif method_upper == "DELETE":
        request_func = _binance_request_delete
    else:
        raise ValueError(f"Unsupported method: {method}")

    res = request_func(
        f"https://api.binance.com{path}",
        params={**query, "signature": signature},
        headers={"X-MBX-APIKEY": api_key},
        timeout=10,
        prefix="Binance spot signed",
    )
    data = res.json()
    if not res.ok:
        raise RuntimeError(f"Binance Spot API 錯誤: {data}")
    if isinstance(data, dict) and data.get("code") not in (None, 0):
        raise RuntimeError(f"Binance Spot API 錯誤: {data}")
    return data


def _binance_spot_signed_get(path, params=None):
    return _binance_spot_signed_request("GET", path, params)


def _binance_futures_open_algo_orders(symbol=None, algo_type="CONDITIONAL"):
    params = {}
    if algo_type:
        params["algoType"] = algo_type
    if symbol:
        params["symbol"] = symbol
    return _binance_futures_signed_get("/fapi/v1/openAlgoOrders", params or None)


def _truthy_binance_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def _binance_order_type(order) -> str:
    if not isinstance(order, dict):
        return ""
    return str(order.get("type") or order.get("orderType") or "").upper()


def _binance_order_trigger_price(order) -> float:
    if not isinstance(order, dict):
        return 0.0

    for key in ("triggerPrice", "stopPrice", "price"):
        value = _safe_float(order.get(key), 0.0)
        if value > 0:
            return value
    return 0.0


def _is_binance_protection_order(order, close_side="") -> bool:
    if not isinstance(order, dict):
        return False

    if _binance_order_type(order) not in BINANCE_PROTECTION_ORDER_TYPES:
        return False

    if close_side and str(order.get("side") or "").upper() != str(close_side).upper():
        return False

    client_id = str(order.get("clientAlgoId") or order.get("clientOrderId") or "").lower()
    if client_id.startswith(BINANCE_PROTECTION_CLIENT_PREFIX):
        return True

    return _truthy_binance_flag(order.get("reduceOnly")) or _truthy_binance_flag(order.get("closePosition"))


def _is_binance_dual_side_mode() -> bool:
    try:
        payload = _binance_futures_signed_get("/fapi/v1/positionSide/dual")
        return bool(payload.get("dualSidePosition")) if isinstance(payload, dict) else False
    except Exception:
        return False


def _cancel_existing_binance_protection_orders(close_side, position_side, dual_side):
    try:
        open_orders = _binance_futures_signed_get("/fapi/v1/openOrders", {"symbol": COPY_TRADE_SYMBOL})
    except Exception:
        open_orders = []

    for order in open_orders if isinstance(open_orders, list) else []:
        if not _is_binance_protection_order(order, close_side):
            continue

        order_ps = str(order.get("positionSide") or "").upper()
        if dual_side and order_ps and order_ps != position_side:
            continue

        order_id = order.get("orderId")
        if order_id is None:
            continue

        try:
            _binance_futures_signed_request(
                "DELETE",
                "/fapi/v1/order",
                {"symbol": COPY_TRADE_SYMBOL, "orderId": int(order_id)},
            )
        except Exception:
            pass

    try:
        open_algo_orders = _binance_futures_open_algo_orders(symbol=COPY_TRADE_SYMBOL, algo_type="CONDITIONAL")
    except Exception:
        open_algo_orders = []

    for order in open_algo_orders if isinstance(open_algo_orders, list) else []:
        if not _is_binance_protection_order(order, close_side):
            continue

        order_ps = str(order.get("positionSide") or "").upper()
        if dual_side and order_ps and order_ps != position_side:
            continue

        algo_id = order.get("algoId")
        if algo_id is None:
            continue

        try:
            _binance_futures_signed_request("DELETE", "/fapi/v1/algoOrder", {"algoId": int(algo_id)})
        except Exception:
            pass


def _submit_binance_protection_order(close_side, position_side, dual_side, qty, order_type, trigger_price):
    if _safe_float(trigger_price, 0.0) <= 0:
        return

    trigger_px = round(_safe_float(trigger_price, 0.0), 2)
    order_type = str(order_type or "").upper()
    order_tag = "tp" if order_type.startswith("TAKE_PROFIT") else "sl"
    client_order_base = f"{BINANCE_PROTECTION_CLIENT_PREFIX}{order_tag}_{int(time.time() * 1000)}"
    # Binance 官方文件把 -4120 定義為這類保護單需改走 Algo Order endpoint。
    base_params = {
        "algoType": "CONDITIONAL",
        "symbol": COPY_TRADE_SYMBOL,
        "side": close_side,
        "type": order_type,
        "triggerPrice": trigger_px,
        "workingType": "MARK_PRICE",
        "priceProtect": "TRUE",
    }
    if dual_side:
        base_params["positionSide"] = position_side

    primary_error = None
    if order_type in {"STOP_MARKET", "TAKE_PROFIT_MARKET"}:
        primary_params = dict(base_params)
        primary_params["closePosition"] = "true"
        primary_params["clientAlgoId"] = f"{client_order_base}_cp"
        try:
            _binance_futures_signed_request("POST", "/fapi/v1/algoOrder", primary_params)
            return
        except Exception as e:
            primary_error = e

    fallback_params = dict(base_params)
    fallback_params["clientAlgoId"] = f"{client_order_base}_qty"
    fallback_params["quantity"] = qty
    if order_type in {"STOP", "TAKE_PROFIT"}:
        fallback_params["price"] = trigger_px
        fallback_params["timeInForce"] = "GTC"
    if not dual_side:
        fallback_params["reduceOnly"] = "true"

    try:
        _binance_futures_signed_request("POST", "/fapi/v1/algoOrder", fallback_params)
    except Exception as e:
        if primary_error is not None:
            raise RuntimeError(f"{e}; 原始錯誤: {primary_error}")
        raise


def update_copy_trade_tp_sl(tp=None, sl=None):
    if not _is_real_copy_enabled():
        return False, "實單未啟用，僅更新本地 TP/SL"

    positions = _binance_futures_signed_get("/fapi/v2/positionRisk")
    active_rows = []
    for item in positions if isinstance(positions, list) else []:
        if not isinstance(item, dict):
            continue
        if str(item.get("symbol") or "").upper() != COPY_TRADE_SYMBOL:
            continue
        amt = _safe_float(item.get("positionAmt"), 0.0)
        if abs(amt) <= 1e-9:
            continue
        active_rows.append(item)

    if not active_rows:
        return False, "Binance 無持倉，僅更新本地 TP/SL"

    row = max(active_rows, key=lambda x: abs(_safe_float(x.get("positionAmt"), 0.0)))
    position_amt = _safe_float(row.get("positionAmt"), 0.0)
    direction = "long" if position_amt > 0 else "short"
    dual_side = _is_binance_dual_side_mode()
    position_side = str(row.get("positionSide") or ("LONG" if direction == "long" else "SHORT")).upper()
    close_side = "SELL" if direction == "long" else "BUY"
    qty = max(abs(position_amt), 0.001)
    _cancel_existing_binance_protection_orders(close_side, position_side, dual_side)

    if _safe_float(tp, 0.0) > 0:
        _submit_binance_protection_order(close_side, position_side, dual_side, qty, "TAKE_PROFIT_MARKET", tp)
    if _safe_float(sl, 0.0) > 0:
        _submit_binance_protection_order(close_side, position_side, dual_side, qty, "STOP_MARKET", sl)

    return True, "✅ Binance TP/SL 已同步更新"


def execute_copy_trade_open(direction, size_ratio, tp=None, sl=None):
    if direction not in {"long", "short"}:
        return False, "⚠️ 跟單失敗：方向無效"

    if not _get_follow_mode_enabled():
        return False, "⏹️ 跟單未開啟，已略過 Binance 自動開單"

    if not _is_real_copy_enabled():
        return False, "⚠️ 已開啟跟單，但未啟用實單（請設定 BINANCE_REAL_COPY_ENABLED=1）"

    # 【防呆】已有持倉禁止重複開倉
    if active_trade.get("open"):
        return False, f"⚠️ 已有 {active_trade.get('direction')} 倉位開啟中，禁止同時開單"

    global _LEVERAGE_CAP
    desired_leverage = max(
        1,
        min(COPY_TRADE_MAX_LEVERAGE, _safe_int(os.getenv("COPY_TRADE_LEVERAGE", DEFAULT_LEV), DEFAULT_LEV)),
    )
    leverage_set_error = None
    leverage = desired_leverage
    side = "BUY" if direction == "long" else "SELL"
    dual_side = _is_binance_dual_side_mode()
    position_side = "LONG" if direction == "long" else "SHORT"

    # 若已知子帳戶槓桿上限，直接套用而不再請求 set_leverage
    if 0 < _LEVERAGE_CAP < desired_leverage:
        leverage = _LEVERAGE_CAP
    else:
        try:
            set_resp = _binance_futures_signed_request(
                "POST",
                "/fapi/v1/leverage",
                {"symbol": COPY_TRADE_SYMBOL, "leverage": desired_leverage},
            )
            if isinstance(set_resp, dict):
                leverage = max(1, _safe_int(set_resp.get("leverage"), desired_leverage))
        except Exception as e:
            err_str = str(e)
            leverage_set_error = err_str
            # -4421: 子帳戶槓桿超出限制，記下交易所當前實際槓桿作為上限
            if "-4421" in err_str:
                actual = _get_binance_symbol_leverage(default=desired_leverage)
                _LEVERAGE_CAP = actual
                leverage = actual

    # 以交易所實際槓桿為準，避免訊息顯示與實際下單不一致
    leverage = min(COPY_TRADE_MAX_LEVERAGE, _get_binance_symbol_leverage(default=leverage))
    qty = _calc_copy_trade_qty(size_ratio, leverage=leverage, eth_price=_safe_float(WS_PRICE, 0.0))

    order_params = {
        "symbol": COPY_TRADE_SYMBOL,
        "side": side,
        "type": "MARKET",
        "quantity": qty,
    }
    if dual_side:
        order_params["positionSide"] = position_side

    retry_steps = max(0, _safe_int(os.getenv("COPY_TRADE_MARGIN_RETRY_STEPS", 6), 6))
    retry_decay = min(0.95, max(0.5, _safe_float(os.getenv("COPY_TRADE_MARGIN_RETRY_DECAY", 0.85), 0.85)))
    min_qty = COPY_TRADE_MIN_QTY

    def _recalc_open_qty(retry_attempt):
        # 每次 -2019 後用最新餘額重算，並逐步增加緩衝，避免固定比例遞減仍高估可承受部位。
        buffer_ratio = max(0.35, 0.82 - (0.08 * retry_attempt))
        return _calc_copy_trade_qty_with_buffer(
            size_ratio,
            leverage=leverage,
            eth_price=_safe_float(WS_PRICE, 0.0),
            extra_buffer_ratio=buffer_ratio,
            enforce_min=False,
        )

    order_resp, qty, last_err = _execute_binance_market_order_with_retry(
        order_params,
        initial_qty=qty,
        min_qty=min_qty,
        retry_steps=retry_steps,
        retry_decay=retry_decay,
        recalc_qty_fn=_recalc_open_qty,
    )

    if order_resp is None:
        if _is_binance_insufficient_margin_error(last_err):
            return False, _format_binance_margin_failure("❌ Binance 自動開單失敗", qty, leverage, _safe_float(WS_PRICE, 0.0), last_err)
        return False, f"❌ Binance 自動開單失敗: {last_err}"

    # 優先用交易所回傳均價更新本地進場，降低「進場價未更新」視覺延遲。
    try:
        avg_price = _safe_float(order_resp.get("avgPrice"), 0.0) if isinstance(order_resp, dict) else 0.0
        if avg_price > 0:
            active_trade["entry"] = avg_price
            active_trade["avg_entry"] = avg_price
            sync_position_panel(_safe_float(WS_PRICE, avg_price))
    except Exception:
        pass

    close_side = "SELL" if direction == "long" else "BUY"
    cond_errors = []

    try:
        if direction == "long":
            if _safe_float(tp, 0.0) > 0:
                _submit_binance_protection_order(close_side, position_side, dual_side, qty, "TAKE_PROFIT_MARKET", tp)
            if _safe_float(sl, 0.0) > 0:
                _submit_binance_protection_order(close_side, position_side, dual_side, qty, "STOP_MARKET", sl)
        else:
            if _safe_float(tp, 0.0) > 0:
                _submit_binance_protection_order(close_side, position_side, dual_side, qty, "TAKE_PROFIT_MARKET", tp)
            if _safe_float(sl, 0.0) > 0:
                _submit_binance_protection_order(close_side, position_side, dual_side, qty, "STOP_MARKET", sl)
    except Exception as e:
        cond_errors.append(str(e))

    order_id = order_resp.get("orderId") if isinstance(order_resp, dict) else None
    executed_qty = _safe_float(order_resp.get("executedQty"), 0.0) if isinstance(order_resp, dict) else 0.0
    orig_qty = _safe_float(order_resp.get("origQty"), qty) if isinstance(order_resp, dict) else qty
    display_qty = executed_qty if executed_qty > 0 else (orig_qty if orig_qty > 0 else qty)
    active_trade["position_qty"] = max(0.0, display_qty)
    msg = (
        f"✅ Binance 已自動開單 | 方向: {direction} | 數量: {display_qty:.3f} ETH | 槓桿: {leverage}x"
        f" | orderId: {order_id}"
    )
    if desired_leverage != leverage:
        msg += f"\nℹ️ 槓桿請求 {desired_leverage}x，交易所實際 {leverage}x"
    if leverage_set_error:
        msg += f"\n⚠️ 設定槓桿失敗，沿用交易所槓桿：{leverage_set_error}"
    if cond_errors:
        msg += f"\n⚠️ TP/SL 掛單失敗: {cond_errors[-1]}"
    return True, msg


def _extract_tp_sl_from_binance_orders(orders, direction, position_side, entry_price):
    tp = 0.0
    sl = 0.0
    side_match = {str(position_side or "").upper(), "BOTH", ""}

    for order in orders or []:
        if not isinstance(order, dict):
            continue
        if str(order.get("symbol") or "").upper() != COPY_TRADE_SYMBOL:
            continue

        order_position_side = str(order.get("positionSide") or "").upper()
        if order_position_side not in side_match:
            continue

        order_type = _binance_order_type(order)
        if order_type not in BINANCE_PROTECTION_ORDER_TYPES:
            continue

        trigger_price = _binance_order_trigger_price(order)
        if trigger_price <= 0:
            continue

        if direction == "long":
            if order_type.startswith("TAKE_PROFIT"):
                tp = trigger_price if tp <= 0 else max(tp, trigger_price)
            elif order_type.startswith("STOP"):
                sl = trigger_price if sl <= 0 else max(sl, trigger_price)
        elif direction == "short":
            if order_type.startswith("TAKE_PROFIT"):
                tp = trigger_price if tp <= 0 else min(tp, trigger_price)
            elif order_type.startswith("STOP"):
                sl = trigger_price if sl <= 0 else min(sl, trigger_price)

    return tp, sl


def _infer_remote_close_reason(direction, current_price, tp, sl):
    direction = str(direction or "")
    px = _safe_float(current_price, 0.0)
    tp = _safe_float(tp, 0.0)
    sl = _safe_float(sl, 0.0)
    tolerance_rate = 0.0015
    distance_candidates = []

    if direction == "long":
        if tp > 0 and px >= tp * (1 - tolerance_rate):
            return "TP"
        if sl > 0 and px <= sl * (1 + tolerance_rate):
            return "SL"
    elif direction == "short":
        if tp > 0 and px <= tp * (1 + tolerance_rate):
            return "TP"
        if sl > 0 and px >= sl * (1 - tolerance_rate):
            return "SL"

    if tp > 0:
        distance_candidates.append(("TP", abs(px - tp) / max(tp, 1e-9)))
    if sl > 0:
        distance_candidates.append(("SL", abs(px - sl) / max(sl, 1e-9)))

    if distance_candidates:
        reason, distance = min(distance_candidates, key=lambda item: item[1])
        if distance <= 0.004:
            return reason

    return "MANUAL"


def _build_trade_close_message(reason, direction, current_price, candle_high=0.0, candle_low=0.0, source=""):
    reason_text = str(reason or "").upper()
    direction_text = str(direction or "")
    current_price = _safe_float(current_price, 0.0)
    candle_high = _safe_float(candle_high, current_price)
    candle_low = _safe_float(candle_low, current_price)
    source_text = f" | {source}" if source else ""

    if reason_text == "TP":
        return (
            f"✅ TP 命中（{direction_text}）{source_text}\n"
            f"當前: {current_price:.2f} | 1m高低: {candle_high:.2f}/{candle_low:.2f}\n"
            f"已關閉倉位，等待下一筆交易"
        )
    if reason_text == "SL":
        return (
            f"❌ SL 命中（{direction_text}）{source_text}\n"
            f"當前: {current_price:.2f} | 1m高低: {candle_high:.2f}/{candle_low:.2f}\n"
            f"已關閉倉位，等待下一筆交易"
        )
    return (
        f"ℹ️ 倉位已關閉（{direction_text}）{source_text}\n"
        f"當前: {current_price:.2f} | 1m高低: {candle_high:.2f}/{candle_low:.2f}\n"
        f"請檢查是否為手動平倉或交易所端觸發"
    )


def sync_active_trade_from_binance(send_notice=False):
    try:
        positions = _binance_futures_signed_get("/fapi/v2/positionRisk")
    except Exception as e:
        return False, f"Binance 同步失敗: {e}"

    try:
        standard_orders = _binance_futures_signed_get("/fapi/v1/openOrders", {"symbol": COPY_TRADE_SYMBOL})
    except Exception:
        standard_orders = []

    try:
        algo_orders = _binance_futures_open_algo_orders(symbol=COPY_TRADE_SYMBOL, algo_type="CONDITIONAL")
    except Exception:
        algo_orders = []

    orders = []
    if isinstance(standard_orders, list):
        orders.extend(standard_orders)
    if isinstance(algo_orders, list):
        orders.extend(algo_orders)

    active_rows = []
    for item in positions if isinstance(positions, list) else []:
        if not isinstance(item, dict):
            continue
        if str(item.get("symbol") or "").upper() != COPY_TRADE_SYMBOL:
            continue
        position_amt = _safe_float(item.get("positionAmt"), 0.0)
        if abs(position_amt) <= 1e-9:
            continue
        active_rows.append(item)

    if not active_rows:
        had_local_open = bool(active_trade.get("open"))
        local_direction = str(active_trade.get("direction") or "")
        close_price = _safe_float(
            WS_PRICE,
            _safe_float(
                POSITION_PANEL_STATE.get("binance_mark_price"),
                _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), 0.0),
            ),
        )
        close_reason = "MANUAL"

        # Binance 開倉後有時會有短暫查詢空窗；在保護視窗內不要把本地倉位誤清空。
        if had_local_open:
            open_ts = _safe_float(active_trade.get("open_time"), 0.0)
            if open_ts > 0 and (time.time() - open_ts) < BINANCE_POSITION_SYNC_GRACE_SEC:
                return False, "⌛ Binance 持倉同步延遲，暫不重置本地倉位"
            close_reason = _infer_remote_close_reason(
                local_direction,
                close_price,
                active_trade.get("tp"),
                active_trade.get("sl"),
            )
            record_position_close(close_reason, close_price, close_price, close_price)
            if close_reason in {"TP", "SL"}:
                _finalize_pending_training_sample(
                    _load_pending_training_sample_state(),
                    1 if close_reason == "TP" else 0,
                    close_reason=close_reason,
                    close_price=close_price,
                    atr=0.0,
                )

        _reset_active_trade_state()
        sync_position_panel(close_price)
        if had_local_open:
            msg = _build_trade_close_message(
                close_reason,
                local_direction,
                close_price,
                close_price,
                close_price,
                source="Binance 同步",
            )
            _send_trade_notification(msg, priority=True)
        else:
            msg = f"✅ 已同步 Binance：目前 {COPY_TRADE_SYMBOL} 無持倉"
            if send_notice:
                send_telegram(msg, priority=True)
        return True, msg

    row = max(active_rows, key=lambda item: abs(_safe_float(item.get("positionAmt"), 0.0)))
    position_amt = _safe_float(row.get("positionAmt"), 0.0)
    actual_qty = abs(position_amt)
    entry_price = _safe_float(row.get("entryPrice"), 0.0)
    mark_price = _safe_float(row.get("markPrice"), _safe_float(WS_PRICE, entry_price))
    notional = abs(_safe_float(row.get("notional"), abs(position_amt) * mark_price))
    leverage = max(1, _safe_int(row.get("leverage"), DEFAULT_LEV))
    position_margin = notional / leverage if leverage > 0 else 0.0
    unrealized_pnl_usdt = _safe_float(row.get("unRealizedProfit"), 0.0)
    snapshot_ts = int(time.time())
    direction = "long" if position_amt > 0 else "short"
    preserve_local_state = (
        bool(active_trade.get("open"))
        and str(active_trade.get("direction") or "") == direction
        and _safe_float(active_trade.get("size"), 0.0) > 0
    )
    position_side = str(row.get("positionSide") or "BOTH").upper()
    account_snapshot = _refresh_position_panel_account_state(force=True, log_on_error=False)
    size_ratio = _estimate_size_ratio_from_position_margin(position_margin, account_snapshot)
    capital_usage_ratio = _compute_capital_usage_ratio(position_margin, account_snapshot)
    if size_ratio <= 0:
        base_qty = max(_safe_float(os.getenv("COPY_TRADE_ETH_QTY", 1.0), 1.0), 1e-9)
        size_ratio = actual_qty / base_qty
    if capital_usage_ratio <= 0:
        capital_usage_ratio = min(1.0, size_ratio)
    tp, sl = _extract_tp_sl_from_binance_orders(orders, direction, position_side, entry_price)

    def _is_valid_tp_level(value):
        level = _safe_float(value, 0.0)
        if level <= 0 or entry_price <= 0:
            return False
        if direction == "long":
            return level > entry_price
        return level < entry_price

    def _is_valid_sl_level(value):
        level = _safe_float(value, 0.0)
        if level <= 0:
            return False
        if direction == "long":
            return tp <= 0 or level < tp
        return tp <= 0 or level > tp

    if not _is_valid_tp_level(tp):
        for candidate in (
            active_trade.get("tp"),
            POSITION_PANEL_STATE.get("tp"),
        ):
            if _is_valid_tp_level(candidate):
                tp = _safe_float(candidate, 0.0)
                break
        # 最終 fallback：依 SL 推算合理 TP（風險報酬比 1:1.5）
        if not _is_valid_tp_level(tp) and entry_price > 0 and _is_valid_sl_level(sl):
            risk = abs(sl - entry_price)
            if direction == "long":
                tp = entry_price + risk * 1.5
            else:
                tp = entry_price - risk * 1.5

    if not _is_valid_sl_level(sl):
        for candidate in (
            active_trade.get("sl"),
            POSITION_PANEL_STATE.get("sl"),
        ):
            if _is_valid_sl_level(candidate):
                sl = _safe_float(candidate, 0.0)
                break

    max_size, min_size = _resolve_scaling_bounds(
        size_ratio,
        active_trade.get("max_size") if preserve_local_state else None,
        active_trade.get("min_size") if preserve_local_state else None,
    )

    active_trade["direction"] = direction
    active_trade["entry"] = entry_price
    active_trade["avg_entry"] = entry_price
    active_trade["tp"] = tp
    active_trade["sl"] = sl
    active_trade["open"] = True
    active_trade["size"] = size_ratio
    active_trade["position_qty"] = actual_qty
    active_trade["max_size"] = max_size
    active_trade["min_size"] = min_size
    active_trade["add_count"] = max(0, _safe_int(active_trade.get("add_count"), 0)) if preserve_local_state else 0
    active_trade["reduce_count"] = max(0, _safe_int(active_trade.get("reduce_count"), 0)) if preserve_local_state else 0
    active_trade["quick_reduce_count"] = max(0, _safe_int(active_trade.get("quick_reduce_count"), 0)) if preserve_local_state else 0
    active_trade["quick_reduce_ts"] = _safe_float(active_trade.get("quick_reduce_ts"), 0.0) if preserve_local_state else 0.0
    active_trade["daily_min_size_enforce_ts"] = _safe_float(active_trade.get("daily_min_size_enforce_ts"), 0.0) if preserve_local_state else 0.0
    active_trade["last_adjust_ts"] = _safe_float(active_trade.get("last_adjust_ts"), 0.0) if preserve_local_state else 0.0
    active_trade["scale_add_paused"] = bool(active_trade.get("scale_add_paused", False)) if preserve_local_state else False
    active_trade["scale_add_pause_reason"] = str(active_trade.get("scale_add_pause_reason") or "") if preserve_local_state else ""
    active_trade["scale_add_pause_ts"] = _safe_float(active_trade.get("scale_add_pause_ts"), 0.0) if preserve_local_state else 0.0
    active_trade["last_scale_skip_notify_key"] = str(active_trade.get("last_scale_skip_notify_key") or "") if preserve_local_state else ""
    active_trade["last_scale_skip_notify_ts"] = _safe_float(active_trade.get("last_scale_skip_notify_ts"), 0.0) if preserve_local_state else 0.0
    prev_open_time = _safe_float(active_trade.get("open_time"), 0.0)
    active_trade["open_time"] = prev_open_time if preserve_local_state and prev_open_time > 0 else time.time()
    active_trade["tp_sl_adjusted_4h"] = bool(active_trade.get("tp_sl_adjusted_4h", False)) if preserve_local_state else False
    active_trade["time_horizon"] = _normalize_trade_time_horizon(active_trade.get("time_horizon") if preserve_local_state else "short")
    if preserve_local_state and bool(active_trade.get("break_even_active", False)):
        _sync_break_even_state_from_sl(direction, entry_price, sl, preserve_existing=True)
    else:
        _sync_break_even_state_from_sl(direction, entry_price, sl, preserve_existing=False)

    POSITION_PANEL_STATE["pair"] = "ETHUSDT"
    POSITION_PANEL_STATE["lev"] = leverage
    POSITION_PANEL_STATE["size"] = capital_usage_ratio
    POSITION_PANEL_STATE["size_ratio"] = size_ratio
    POSITION_PANEL_STATE["capital_usage_ratio"] = capital_usage_ratio
    POSITION_PANEL_STATE["binance_qty"] = actual_qty
    POSITION_PANEL_STATE["binance_mark_price"] = mark_price
    POSITION_PANEL_STATE["binance_mark_price_ts"] = snapshot_ts
    POSITION_PANEL_STATE["position_notional_usdt"] = notional
    POSITION_PANEL_STATE["position_margin_usdt"] = position_margin
    POSITION_PANEL_STATE["binance_unrealized_pnl_usdt"] = unrealized_pnl_usdt
    POSITION_PANEL_STATE["binance_unrealized_pnl_ts"] = snapshot_ts

    sync_position_panel(mark_price)

    tp_text = f"{tp:.2f}" if tp > 0 else "未抓到"
    sl_text = f"{sl:.2f}" if sl > 0 else "未抓到"
    msg = (
        f"✅ 已同步 Binance 倉位\n"
        f"方向: {direction} | 數量: {actual_qty:.4f} ETH | 資金使用率: {capital_usage_ratio*100:.2f}%\n"
        f"進場: {entry_price:.2f} | 現價: {mark_price:.2f} | 槓桿: {leverage}x\n"
        f"TP: {tp_text} | SL: {sl_text}"
    )
    if send_notice:
        send_telegram(msg, priority=True)
    return True, msg


def _estimate_panel_financials(entry_price, size_ratio, lev):
    prev_size = max(0.0, _safe_float(POSITION_PANEL_STATE.get("size_ratio", POSITION_PANEL_STATE.get("size")), 0.0))
    prev_notional = max(0.0, _safe_float(POSITION_PANEL_STATE.get("position_notional_usdt"), 0.0))
    prev_margin = max(0.0, _safe_float(POSITION_PANEL_STATE.get("position_margin_usdt"), 0.0))

    size_ratio = max(0.0, _safe_float(size_ratio, 0.0))
    lev = max(1, _safe_int(lev, DEFAULT_LEV))
    entry_price = max(0.0, _safe_float(entry_price, 0.0))

    if prev_size > 1e-9 and prev_notional > 0:
        scale = size_ratio / prev_size if size_ratio > 0 else 0.0
        notional = prev_notional * scale
        margin = prev_margin * scale if prev_margin > 0 else notional / lev
        return max(0.0, notional), max(0.0, margin)

    margin = PANEL_DEFAULT_MAX_MARGIN_USDT * max(size_ratio, 0.1 if size_ratio > 0 else 0.0)
    notional = margin * lev

    # 若舊檔已有更接近實際值，保守沿用
    if POSITION_PANEL_STATE.get("position_notional_usdt") and size_ratio > 0:
        notional = max(notional, _safe_float(POSITION_PANEL_STATE.get("position_notional_usdt"), 0.0))
        margin = max(margin, _safe_float(POSITION_PANEL_STATE.get("position_margin_usdt"), 0.0))

    # 進場價過低時避免出現過小 notional
    if entry_price > 0:
        notional = max(notional, entry_price * max(size_ratio, 0.05) * 0.5)
        margin = max(margin, notional / lev)

    return max(0.0, notional), max(0.0, margin)


def _sanitize_telegram_text(msg):
    raw = str(msg or "")
    safe_text = raw.replace("<", "").replace(">", "").replace("&", "and")
    fallback_text = raw.replace("<", "").replace(">", "")
    return safe_text, fallback_text


def _discord_webhook_base_url(webhook_url: str) -> str:
    parsed = urlparse(str(webhook_url or "").strip())
    if not parsed.scheme or not parsed.netloc or not parsed.path:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def _schedule_discord_message_delete(webhook_url: str, message_id: str, delay_sec: int):
    base_url = _discord_webhook_base_url(webhook_url)
    msg_id = str(message_id or "").strip()
    if not base_url or not msg_id or delay_sec <= 0:
        return

    def _delete_message():
        try:
            HTTP_SESSION.delete(f"{base_url}/messages/{msg_id}", timeout=8)
        except Exception as e:
            print("Discord auto-delete error:", e)

    timer = threading.Timer(delay_sec, _delete_message)
    timer.daemon = True
    timer.start()


def _post_discord_webhook(webhook_url: str, content: str, timeout: int = 5):
    url = str(webhook_url or "").strip()
    if not url:
        return

    payload = {"content": str(content or "")}
    if DISCORD_AUTO_DELETE_SEC <= 0:
        HTTP_SESSION.post(url, json=payload, timeout=timeout)
        return

    # 需要 wait=true 才能拿到 message id，供後續刪除
    res = HTTP_SESSION.post(url, json=payload, params={"wait": "true"}, timeout=timeout)
    res.raise_for_status()

    message_id = ""
    try:
        body = res.json() if res is not None else {}
        if isinstance(body, dict):
            message_id = str(body.get("id", "") or "")
    except Exception:
        message_id = ""

    if message_id:
        _schedule_discord_message_delete(url, message_id, DISCORD_AUTO_DELETE_SEC)


def _post_telegram_message(chat_id, text, reply_markup=None, timeout=5):
    if not TELEGRAM_TOKEN or chat_id is None:
        return None

    payload = {
        "chat_id": chat_id,
        "text": text,
    }
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup

    try:
        return HTTP_SESSION.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json=payload,
            timeout=timeout,
        )
    except Exception:
        return None


def _send_telegram_message(chat_id, msg, include_control_panel=False, timeout=5):
    safe_text, fallback_text = _sanitize_telegram_text(msg)
    reply_markup = _build_control_panel_keyboard(chat_id) if include_control_panel and _is_private_chat_id(chat_id) else None

    res = _post_telegram_message(chat_id, safe_text, reply_markup=reply_markup, timeout=timeout)
    if res is not None and res.status_code == 400 and fallback_text != safe_text:
        res = _post_telegram_message(chat_id, fallback_text, reply_markup=reply_markup, timeout=timeout)
    return res


def _start_panel_realtime_publisher():
    global PANEL_REALTIME_WORKER_STARTED
    if PANEL_REALTIME_WORKER_STARTED:
        return

    def _worker():
        global PANEL_REALTIME_LAST_ERROR_TS
        while True:
            PANEL_REALTIME_PUBLISH_EVENT.wait()
            PANEL_REALTIME_PUBLISH_EVENT.clear()
            with PANEL_REALTIME_QUEUE_LOCK:
                if not PANEL_REALTIME_PUBLISH_QUEUE:
                    continue
                payload = dict(PANEL_REALTIME_PUBLISH_QUEUE[-1])

            headers = {}
            if POSITION_PANEL_REALTIME_TOKEN:
                headers["X-Panel-Token"] = POSITION_PANEL_REALTIME_TOKEN
                headers["Authorization"] = f"Bearer {POSITION_PANEL_REALTIME_TOKEN}"

            try:
                _, _, publish_url = _current_panel_realtime_urls()
                if not publish_url:
                    continue
                res = HTTP_SESSION.post(
                    publish_url,
                    json=payload,
                    headers=headers,
                    timeout=POSITION_PANEL_REALTIME_TIMEOUT_SEC,
                )
                if res.status_code >= 400:
                    body = str(res.text or "").strip()
                    if len(body) > 200:
                        body = body[:200] + "..."
                    raise RuntimeError(f"HTTP {res.status_code} {body}")
            except Exception as e:
                now_ts = time.time()
                if now_ts - PANEL_REALTIME_LAST_ERROR_TS >= 30:
                    PANEL_REALTIME_LAST_ERROR_TS = now_ts
                    print(f"⚠️ 即時面板推送失敗: {e}")

    thread = threading.Thread(target=_worker, name="panel-realtime-publisher", daemon=True)
    thread.start()
    PANEL_REALTIME_WORKER_STARTED = True


def _queue_panel_realtime_publish(payload):
    global PANEL_REALTIME_LAST_SIGNATURE, PANEL_REALTIME_LAST_ENQUEUE_TS
    _, _, publish_url = _current_panel_realtime_urls()
    if not publish_url or not isinstance(payload, dict):
        return

    try:
        signature = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return

    now_ts = time.time()
    if signature == PANEL_REALTIME_LAST_SIGNATURE and (now_ts - PANEL_REALTIME_LAST_ENQUEUE_TS) < POSITION_PANEL_REALTIME_HEARTBEAT_SEC:
        return

    _start_panel_realtime_publisher()
    PANEL_REALTIME_LAST_SIGNATURE = signature
    PANEL_REALTIME_LAST_ENQUEUE_TS = now_ts
    with PANEL_REALTIME_QUEUE_LOCK:
        PANEL_REALTIME_PUBLISH_QUEUE.clear()
        PANEL_REALTIME_PUBLISH_QUEUE.append(payload)
    PANEL_REALTIME_PUBLISH_EVENT.set()


def record_position_close(reason, current_price, candle_high=0.0, candle_low=0.0):
    reason_text = str(reason or "").upper()
    if reason_text not in {"TP", "SL", "MANUAL", "MAX_HOLD"}:
        reason_text = "CLOSE"

    ts_now = int(time.time())
    POSITION_PANEL_STATE["last_close_reason"] = reason_text
    POSITION_PANEL_STATE["last_close_price"] = round(_safe_float(current_price, 0.0), 4)
    POSITION_PANEL_STATE["last_close_ts"] = ts_now
    POSITION_PANEL_STATE["last_close_candle_high"] = round(_safe_float(candle_high, 0.0), 4)
    POSITION_PANEL_STATE["last_close_candle_low"] = round(_safe_float(candle_low, 0.0), 4)

    hits = POSITION_PANEL_STATE.get("close_hits")
    if not isinstance(hits, list):
        hits = []
    hits.insert(
        0,
        {
            "reason": reason_text,
            "price": round(_safe_float(current_price, 0.0), 4),
            "ts": ts_now,
        },
    )
    POSITION_PANEL_STATE["close_hits"] = hits[:10]


def _build_sl_strategy_review(direction, entry, tp, sl, close_price, atr_ref, context, *, stop_atr, planned_rr, stop_overshoot, alignment_score):
    """Classify why an SL happened and return executable guard hints."""
    direction = _normalize_trade_direction(direction)
    context = context if isinstance(context, dict) else {}
    expected_sign = 1 if direction == "long" else -1

    score = _safe_float(context.get("score"), 0.5)
    ai_prob = _safe_float(context.get("ai_prob"), score)
    ai_long_prob = _safe_float(context.get("ai_long_prob"), ai_prob)
    ai_short_prob = _safe_float(context.get("ai_short_prob"), 1.0 - ai_prob)
    net_edge = _safe_float(context.get("net_edge_rate_est"), 0.0)
    risk_rate = _safe_float(context.get("risk_rate"), abs(entry - sl) / max(entry, 1e-9))
    reward_rate = _safe_float(context.get("reward_rate"), abs(tp - entry) / max(entry, 1e-9))
    sr_bias = _safe_float(context.get("sr_bias"), 0.0)
    support_hits = _safe_int(context.get("support_hits"), 0)
    resistance_hits = _safe_int(context.get("resistance_hits"), 0)
    repeated_support_tests = _safe_int(context.get("repeated_support_tests"), 0)
    repeated_resistance_tests = _safe_int(context.get("repeated_resistance_tests"), 0)
    repeated_test_pressure = _safe_float(context.get("repeated_test_pressure"), 0.0)
    breakout = _safe_int(context.get("breakout"), 0)
    htf = _safe_int(context.get("htf"), 0)
    mid_trend = _safe_int(context.get("mid_trend"), 0)
    macro_bias = _safe_float(context.get("macro_bias"), 0.0)
    derivatives_pressure = _safe_float(context.get("derivatives_pressure"), 0.0)
    rsi_15m = _safe_float(context.get("rsi_15m"), 50.0)
    ema50_deviation = _safe_float(context.get("ema50_deviation_15m"), 0.0)
    taker_buy_ratio = _safe_float(context.get("taker_buy_ratio"), 0.5)
    content_override = context.get("content_override") if isinstance(context.get("content_override"), dict) else {}
    host_opening_logic = context.get("host_opening_logic") if isinstance(context.get("host_opening_logic"), dict) else {}
    learned_entry_logic = context.get("learned_entry_logic") if isinstance(context.get("learned_entry_logic"), dict) else {}

    issue_codes = []
    issue_details = []
    optimization_actions = []

    def add_issue(code, detail, action):
        if code not in issue_codes:
            issue_codes.append(code)
            issue_details.append(detail)
            optimization_actions.append(action)

    if stop_atr < max(0.50, _safe_float(os.getenv("SL_REVIEW_MIN_STOP_ATR", "0.65"), 0.65)):
        add_issue("sl_too_tight", f"SL距離只有 {stop_atr:.2f} ATR", "提高最小SL距離或等回踩後再進場")
    if planned_rr < max(1.1, _safe_float(os.getenv("TRADE_MIN_ACCEPT_RR", 1.8), 1.8)):
        add_issue("rr_too_low", f"進場RR {planned_rr:.2f} 不足", "降低追單，要求TP空間覆蓋SL與成本")
    if net_edge <= _safe_float(os.getenv("SL_REVIEW_MIN_EDGE_RATE", "0.0012"), 0.0012):
        add_issue("edge_too_low", f"期望值 {net_edge*100:.3f}% 偏低", "提高AI期望值門檻，同方向需重新確認")
    if risk_rate < _safe_float(os.getenv("SL_REVIEW_MIN_RISK_RATE", "0.003"), 0.003):
        add_issue("risk_distance_too_small", f"風險距離 {risk_rate*100:.3f}% 過小", "避免把SL放在短線雜訊範圍內")
    if alignment_score <= 0:
        add_issue("indicator_not_aligned", f"多項指標同向不足 score={alignment_score:+d}", "同方向再進場需等4H/30m/SR至少兩項同向")

    if direction == "long":
        direction_prob = ai_long_prob
        if resistance_hits >= 1 and breakout != 1:
            add_issue("long_into_resistance", f"多單進場時上方壓力 {resistance_hits} 個且未突破", "多單靠近壓力需等突破確認或回踩支撐")
        if repeated_resistance_tests >= 2 and breakout != 1:
            add_issue("failed_resistance_break", f"壓力連測 {repeated_resistance_tests} 次未突破", "壓力連測未破時禁止追多")
        if sr_bias < -0.12 or support_hits == 0:
            add_issue("weak_support_for_long", f"支撐偏置 {sr_bias:+.2f} / 支撐數 {support_hits}", "多單需靠近支撐承接或掃低收回")
        if htf < 0 or mid_trend < 0:
            add_issue("long_against_trend", f"4H={htf} 30m={mid_trend}", "逆高週期多單需縮小倉位並等短線轉強")
        if rsi_15m >= 68 or ema50_deviation > 0.018:
            add_issue("long_chased_high", f"RSI={rsi_15m:.1f} EMA50偏離={ema50_deviation*100:.2f}%", "多單避免急拉後追高，改等回踩")
        if derivatives_pressure < -0.12 or taker_buy_ratio < 0.45:
            add_issue("derivatives_against_long", f"衍生品壓力 {derivatives_pressure:+.2f} 買盤 {taker_buy_ratio:.2f}", "衍生品反向時多單降權")
    else:
        direction_prob = ai_short_prob
        if support_hits >= 1 and breakout != -1:
            add_issue("short_into_support", f"空單進場時下方支撐 {support_hits} 個且未跌破", "空單靠近支撐需等跌破確認或反彈失敗")
        if repeated_support_tests >= 2 and breakout != -1:
            add_issue("failed_support_break", f"支撐連測 {repeated_support_tests} 次未跌破", "支撐連測未破時禁止追空")
        if sr_bias > 0.12 or resistance_hits == 0:
            add_issue("weak_resistance_for_short", f"支撐壓力偏置 {sr_bias:+.2f} / 壓力數 {resistance_hits}", "空單需靠近壓力反彈失敗或跌破支撐")
        if htf > 0 or mid_trend > 0:
            add_issue("short_against_trend", f"4H={htf} 30m={mid_trend}", "逆高週期空單需縮小倉位並等短線轉弱")
        if rsi_15m <= 32 or ema50_deviation < -0.018:
            add_issue("short_chased_low", f"RSI={rsi_15m:.1f} EMA50偏離={ema50_deviation*100:.2f}%", "空單避免急跌後追空，改等反彈壓力")
        if derivatives_pressure > 0.12 or taker_buy_ratio > 0.55:
            add_issue("derivatives_against_short", f"衍生品壓力 {derivatives_pressure:+.2f} 買盤 {taker_buy_ratio:.2f}", "衍生品反向時空單降權")

    if direction_prob < max(0.42, _safe_float(os.getenv("TRADE_MIN_DIRECTION_WIN_PROB", 0.42), 0.42)):
        add_issue("direction_probability_low", f"方向勝率 {direction_prob:.2f} 偏低", "提高該方向最低勝率門檻")
    if abs(repeated_test_pressure) >= 0.15 and repeated_test_pressure * expected_sign < 0:
        add_issue("repeated_test_pressure_against", f"連續測試壓力 {repeated_test_pressure:+.2f} 反向", "連續測試壓力反向時等待確認，不提前開")
    if macro_bias * expected_sign < -0.35:
        add_issue("macro_against", f"宏觀 {macro_bias:+.2f} 反向", "宏觀反向時提高RR與期望值門檻")
    if content_override.get("applied") and str(content_override.get("direction")) != direction:
        add_issue("host_signal_conflict", f"MLX主訊號={content_override.get('direction')}", "MLX主訊號反向時禁止同方向開倉")
    elif content_override.get("usable") and not content_override.get("applied"):
        add_issue("host_signal_not_confirmed", "MLX方向未通過衝突檢查", "MLX方向未確認時不可作為開單理由")
    if host_opening_logic:
        host_direction = str(host_opening_logic.get("direction") or "neutral")
        host_confidence = _safe_float(host_opening_logic.get("confidence"), 0.0)
        if host_direction not in {"neutral", direction}:
            add_issue(
                "mlx_opening_logic_conflict",
                f"MLX開單主邏輯={host_direction} 信心={host_confidence:.2f}",
                "MLX開單主邏輯反向時禁止同方向開倉",
            )
        elif host_direction == "neutral" or host_confidence < 0.42:
            add_issue(
                "mlx_opening_logic_weak",
                f"MLX開單主邏輯={host_direction} 信心={host_confidence:.2f}",
                "MLX開單主邏輯未明確時等待區間邊界或量能確認",
            )
    if learned_entry_logic:
        long_setup = _safe_float(learned_entry_logic.get("long_setup"), 0.0)
        short_setup = _safe_float(learned_entry_logic.get("short_setup"), 0.0)
        if direction == "long" and long_setup < short_setup + 0.25:
            add_issue("learned_logic_not_supporting_long", f"MLX多空分 {long_setup:.2f}/{short_setup:.2f}", "多單需等MLX學習邏輯重新偏多")
        if direction == "short" and short_setup < long_setup + 0.25:
            add_issue("learned_logic_not_supporting_short", f"MLX多空分 {long_setup:.2f}/{short_setup:.2f}", "空單需等MLX學習邏輯重新偏空")

    if stop_overshoot >= 0.25:
        add_issue("slippage_or_fast_break", f"觸發超出SL {stop_overshoot:.2f} ATR", "波動急放大時降低倉位並加大確認")

    replay_snapshot = {
        "score": round(score, 4),
        "ai_prob": round(ai_prob, 4),
        "direction_prob": round(direction_prob, 4),
        "net_edge_rate_est": round(net_edge, 6),
        "risk_rate": round(risk_rate, 6),
        "reward_rate": round(reward_rate, 6),
        "htf": htf,
        "mid_trend": mid_trend,
        "breakout": breakout,
        "sr_bias": round(sr_bias, 4),
        "support_hits": support_hits,
        "resistance_hits": resistance_hits,
        "repeated_support_tests": repeated_support_tests,
        "repeated_resistance_tests": repeated_resistance_tests,
        "repeated_test_pressure": round(repeated_test_pressure, 4),
        "macro_bias": round(macro_bias, 4),
        "derivatives_pressure": round(derivatives_pressure, 4),
        "rsi_15m": round(rsi_15m, 2),
        "ema50_deviation_15m": round(ema50_deviation, 5),
        "taker_buy_ratio": round(taker_buy_ratio, 4),
        "primary_indicator": str(context.get("primary_indicator") or ""),
        "strategy_version": str(context.get("strategy_version") or STRATEGY_VERSION),
    }
    severity = min(5, max(1, len(issue_codes)))
    return {
        "issue_codes": issue_codes,
        "issue_details": issue_details,
        "optimization_actions": optimization_actions[:8],
        "replay_snapshot": replay_snapshot,
        "severity": severity,
    }


def _review_stop_loss_event(direction, entry, tp, sl, close_price, candle_high, candle_low, atr, context=None):
    """Review the SL plan at execution time and retain a short re-entry guard."""
    direction = _normalize_trade_direction(direction)
    context = context if isinstance(context, dict) else {}
    entry = max(0.0, _safe_float(entry, 0.0))
    tp = max(0.0, _safe_float(tp, 0.0))
    sl = max(0.0, _safe_float(sl, 0.0))
    close_price = max(0.0, _safe_float(close_price, 0.0))
    atr_ref = max(_safe_float(atr, 0.0), entry * 0.001, 1e-6)
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    stop_atr = risk / atr_ref
    planned_rr = reward / max(risk, 1e-6)
    stop_overshoot = (
        max(0.0, sl - close_price) if direction == "long" else max(0.0, close_price - sl)
    ) / atr_ref

    expected_sign = 1 if direction == "long" else -1
    htf = _safe_int(context.get("htf"), 0)
    mid_trend = _safe_int(context.get("mid_trend"), 0)
    sr_bias = _safe_float(context.get("sr_bias"), 0.0)
    macro_bias = _safe_float(context.get("macro_bias"), 0.0)
    derivatives_pressure = _safe_float(context.get("derivatives_pressure"), 0.0)
    td_exhaustion = bool(context.get("td_exhaustion", False))
    alignment_score = 0
    indicators = []
    for label, value, threshold in (
        ("4H趨勢", htf, 0.0),
        ("30m動能", mid_trend, 0.0),
        ("支撐壓力", sr_bias, 0.10),
        ("宏觀", macro_bias, 0.20),
        ("衍生品", derivatives_pressure, 0.10),
    ):
        signed_value = _safe_float(value, 0.0) * expected_sign
        if signed_value > threshold:
            alignment_score += 1
            indicators.append(f"{label}同向")
        elif signed_value < -threshold:
            alignment_score -= 1
            indicators.append(f"{label}逆向")
        else:
            indicators.append(f"{label}中性")
    if td_exhaustion:
        alignment_score -= 1
        indicators.append("15m九轉衰竭")

    min_rr = max(1.1, _safe_float(os.getenv("TRADE_MIN_ACCEPT_RR", 1.8), 1.8))
    issues = []
    if stop_atr < 0.50:
        issues.append(f"SL過近 {stop_atr:.2f} ATR")
    if planned_rr < min_rr:
        issues.append(f"RR不足 {planned_rr:.2f}")
    if alignment_score <= 0:
        issues.append(f"技術面逆向/不足 score={alignment_score:+d}")
    if stop_overshoot >= 0.25:
        issues.append(f"觸發超出SL {stop_overshoot:.2f} ATR")

    strategy_review = _build_sl_strategy_review(
        direction,
        entry,
        tp,
        sl,
        close_price,
        atr_ref,
        context,
        stop_atr=stop_atr,
        planned_rr=planned_rr,
        stop_overshoot=stop_overshoot,
        alignment_score=alignment_score,
    )
    for detail in strategy_review.get("issue_details", []):
        if detail not in issues:
            issues.append(detail)

    requires_revalidation = bool(issues)
    verdict = "需重新確認" if requires_revalidation else "SL設定合理，屬正常風控出場"
    review = {
        "ts": int(time.time()),
        "direction": direction,
        "entry": round(entry, 4),
        "tp": round(tp, 4),
        "sl": round(sl, 4),
        "close_price": round(close_price, 4),
        "candle_high": round(_safe_float(candle_high, 0.0), 4),
        "candle_low": round(_safe_float(candle_low, 0.0), 4),
        "stop_atr": round(stop_atr, 3),
        "planned_rr": round(planned_rr, 3),
        "stop_overshoot_atr": round(stop_overshoot, 3),
        "alignment_score": alignment_score,
        "indicators": indicators,
        "issues": issues,
        "issue_codes": strategy_review.get("issue_codes", []),
        "optimization_actions": strategy_review.get("optimization_actions", []),
        "replay_snapshot": strategy_review.get("replay_snapshot", {}),
        "strategy_optimization_severity": strategy_review.get("severity", 1),
        "requires_revalidation": requires_revalidation,
        "verdict": verdict,
    }
    POSITION_PANEL_STATE["last_sl_review"] = review
    return review


def _recent_sl_review_guard_reason(final):
    if not _is_truthy(os.getenv("TRADE_SL_REVIEW_GUARD_ENABLED", "1")) or final.startswith("觀望"):
        return ""
    review = POSITION_PANEL_STATE.get("last_sl_review")
    if not isinstance(review, dict) or not review.get("requires_revalidation"):
        return ""
    guard_sec = max(60.0, _safe_float(os.getenv("TRADE_SL_REVIEW_GUARD_SEC", 900), 900))
    if time.time() - _safe_float(review.get("ts"), 0.0) > guard_sec:
        return ""
    same_direction = (
        ("做多" in final and review.get("direction") == "long")
        or ("做空" in final and review.get("direction") == "short")
    )
    if not same_direction:
        return ""
    issue_codes = review.get("issue_codes") if isinstance(review.get("issue_codes"), list) else []
    issue_code = str(issue_codes[0]) if issue_codes else ""
    code_block_map = {
        "sl_too_tight": "SL距離過近",
        "rr_too_low": "RR不足",
        "edge_too_low": "期望值不足",
        "risk_distance_too_small": "風險距離太小",
        "indicator_not_aligned": "指標未共振",
        "long_into_resistance": "多單打到壓力",
        "failed_resistance_break": "壓力未突破",
        "weak_support_for_long": "多單缺少支撐",
        "long_against_trend": "多單逆勢",
        "long_chased_high": "多單追高",
        "derivatives_against_long": "衍生品反多",
        "short_into_support": "空單打到支撐",
        "failed_support_break": "支撐未跌破",
        "weak_resistance_for_short": "空單缺少壓力",
        "short_against_trend": "空單逆勢",
        "short_chased_low": "空單追低",
        "derivatives_against_short": "衍生品反空",
        "direction_probability_low": "方向勝率偏低",
        "repeated_test_pressure_against": "連續測試壓力反向",
        "macro_against": "宏觀反向",
        "host_signal_conflict": "MLX主訊號衝突",
        "host_signal_not_confirmed": "MLX方向未確認",
        "learned_logic_not_supporting_long": "MLX不支持多單",
        "learned_logic_not_supporting_short": "MLX不支持空單",
        "slippage_or_fast_break": "快速破位",
    }
    issue = code_block_map.get(issue_code) or str((review.get("issues") or ["等待15m重新確認"])[0])
    return f"觀望（SL後檢討防護-{issue}）"


def _build_sl_review_context_from_live(
    *,
    htf=0,
    mid_trend=0,
    sr_analysis=None,
    macro_bias=0.0,
    derivatives_flow=None,
    td_exhaustion=False,
    decision=None,
):
    sr_analysis = sr_analysis if isinstance(sr_analysis, dict) else {}
    derivatives_flow = derivatives_flow if isinstance(derivatives_flow, dict) else {}
    decision = decision if isinstance(decision, dict) else {}
    return {
        "strategy_version": STRATEGY_VERSION,
        "htf": htf,
        "mid_trend": mid_trend,
        "sr_bias": sr_analysis.get("bias"),
        "support_hits": sr_analysis.get("support_hits"),
        "resistance_hits": sr_analysis.get("resistance_hits"),
        "macro_bias": macro_bias,
        "derivatives_pressure": derivatives_flow.get("derivatives_pressure", decision.get("derivatives_pressure")),
        "taker_buy_ratio": derivatives_flow.get("taker_buy_ratio", decision.get("taker_buy_ratio")),
        "open_interest_change": derivatives_flow.get("open_interest_change", decision.get("open_interest_change")),
        "td_exhaustion": td_exhaustion,
        "score": decision.get("score"),
        "ai_prob": decision.get("ai_prob"),
        "ai_long_prob": decision.get("ai_long_prob"),
        "ai_short_prob": decision.get("ai_short_prob"),
        "net_edge_rate_est": decision.get("net_edge_rate_est"),
        "risk_rate": decision.get("risk_rate"),
        "reward_rate": decision.get("reward_rate"),
        "rsi_15m": decision.get("rsi_15m"),
        "ema50_deviation_15m": decision.get("ema50_deviation_15m"),
        "breakout": decision.get("breakout"),
        "repeated_support_tests": decision.get("repeated_support_tests"),
        "repeated_resistance_tests": decision.get("repeated_resistance_tests"),
        "repeated_test_pressure": decision.get("repeated_test_pressure"),
        "content_override": decision.get("content_override"),
        "learned_entry_logic": decision.get("learned_entry_logic"),
        "primary_indicator": decision.get("primary_indicator"),
    }


TAIPEI_TZ = datetime.timezone(datetime.timedelta(hours=8))


def _taipei_trade_date(now=None):
    now = now or datetime.datetime.now(TAIPEI_TZ)
    if now.tzinfo is None:
        now = now.replace(tzinfo=TAIPEI_TZ)
    return now.astimezone(TAIPEI_TZ).date().isoformat()


def _daily_min_trade_due(now=None):
    if not _is_truthy(os.getenv("DAILY_MIN_TRADE_ENABLED", "1")):
        return False
    now = now or datetime.datetime.now(TAIPEI_TZ)
    if now.tzinfo is None:
        now = now.replace(tzinfo=TAIPEI_TZ)
    today = _taipei_trade_date(now)
    if POSITION_PANEL_STATE.get("daily_trade_date") != today:
        POSITION_PANEL_STATE["daily_trade_date"] = today
        POSITION_PANEL_STATE["daily_trade_opened"] = False
        POSITION_PANEL_STATE["daily_trade_source"] = ""
    due_hour = min(23, max(0, _safe_int(os.getenv("DAILY_MIN_TRADE_HOUR", 22), 22)))
    due_minute = min(59, max(0, _safe_int(os.getenv("DAILY_MIN_TRADE_MINUTE", 30), 30)))
    return not bool(POSITION_PANEL_STATE.get("daily_trade_opened", False)) and (now.hour, now.minute) >= (due_hour, due_minute)


def _mark_daily_trade_opened(source):
    existing_source = str(POSITION_PANEL_STATE.get("daily_trade_source") or "")
    if source == "restored_position" and existing_source == "daily_minimum":
        source = existing_source
    POSITION_PANEL_STATE["daily_trade_date"] = _taipei_trade_date()
    POSITION_PANEL_STATE["daily_trade_opened"] = True
    POSITION_PANEL_STATE["daily_trade_source"] = str(source or "signal")
    POSITION_PANEL_STATE["daily_trade_opened_ts"] = int(time.time())


def _is_daily_min_position():
    source = str(POSITION_PANEL_STATE.get("daily_trade_source") or "")
    if source == "daily_minimum":
        return True
    if not _is_truthy(os.getenv("RESTORED_DAILY_MIN_MANAGEMENT_ENABLED", "1")):
        return False
    if source != "restored_position":
        return False
    if not bool(POSITION_PANEL_STATE.get("daily_trade_opened", False)):
        return False
    if str(POSITION_PANEL_STATE.get("daily_trade_date") or "") != _taipei_trade_date():
        return False
    opened_ts = _safe_float(POSITION_PANEL_STATE.get("daily_trade_opened_ts"), 0.0)
    if opened_ts <= 0:
        return False
    opened_dt = datetime.datetime.fromtimestamp(opened_ts, TAIPEI_TZ)
    due_hour = min(23, max(0, _safe_int(os.getenv("DAILY_MIN_TRADE_HOUR", 22), 22)))
    due_minute = min(59, max(0, _safe_int(os.getenv("DAILY_MIN_TRADE_MINUTE", 30), 30)))
    return (opened_dt.hour, opened_dt.minute) >= (due_hour, due_minute)


def _daily_min_2024_style_profile(df_1d=None, df_15m=None):
    if not _is_truthy(os.getenv("DAILY_MIN_2024_STYLE_RELEASE_ENABLED", "0")):
        return {"active": False, "reason": "disabled"}

    frame = df_1d if df_1d is not None and len(df_1d) >= 45 else df_15m
    if frame is None or len(frame) < 80 or "close" not in frame:
        return {"active": False, "reason": "insufficient_data"}

    lookback = max(45, _safe_int(os.getenv("DAILY_MIN_2024_STYLE_LOOKBACK_BARS", 120), 120))
    closes = pd.to_numeric(frame["close"].tail(lookback), errors="coerce").dropna()
    if len(closes) < 45:
        return {"active": False, "reason": "insufficient_closes"}

    first = _safe_float(closes.iloc[0], 0.0)
    last = _safe_float(closes.iloc[-1], 0.0)
    if first <= 0 or last <= 0:
        return {"active": False, "reason": "invalid_price"}

    diffs = closes.diff().abs().dropna()
    path = _safe_float(diffs.sum(), 0.0)
    trend_efficiency = abs(last - first) / max(path, 1e-9)
    peak = closes.cummax()
    drawdown = _safe_float(((closes / peak) - 1.0).min(), 0.0)
    returns = closes.pct_change().dropna()
    realized_vol = _safe_float(returns.std(), 0.0)
    net_change = (last / first) - 1.0
    anchor_change = net_change
    if df_1d is not None and len(df_1d) >= 160 and "close" in df_1d:
        anchor_lookback = max(160, _safe_int(os.getenv("DAILY_MIN_2024_STYLE_ANCHOR_BARS", 240), 240))
        anchor_closes = pd.to_numeric(df_1d["close"].tail(anchor_lookback), errors="coerce").dropna()
        if len(anchor_closes) >= 120 and _safe_float(anchor_closes.iloc[0], 0.0) > 0:
            anchor_change = (_safe_float(anchor_closes.iloc[-1], 0.0) / _safe_float(anchor_closes.iloc[0], 1.0)) - 1.0

    max_efficiency = _safe_float(os.getenv("DAILY_MIN_2024_STYLE_MAX_EFFICIENCY", 0.075), 0.075)
    min_drawdown = _safe_float(os.getenv("DAILY_MIN_2024_STYLE_MIN_DRAWDOWN", 0.16), 0.16)
    min_realized_vol = _safe_float(os.getenv("DAILY_MIN_2024_STYLE_MIN_REALIZED_VOL", 0.018), 0.018)
    max_abs_net_change = _safe_float(os.getenv("DAILY_MIN_2024_STYLE_MAX_ABS_NET_CHANGE", 0.55), 0.55)
    min_anchor_change = _safe_float(os.getenv("DAILY_MIN_2024_STYLE_MIN_ANCHOR_CHANGE", 0.18), 0.18)

    active = bool(
        trend_efficiency <= max_efficiency
        and abs(drawdown) >= min_drawdown
        and realized_vol >= min_realized_vol
        and abs(net_change) <= max_abs_net_change
        and anchor_change >= min_anchor_change
    )
    return {
        "active": active,
        "reason": "low_efficiency_high_swing" if active else "not_matched",
        "trend_efficiency": round(trend_efficiency, 4),
        "drawdown": round(drawdown, 4),
        "realized_vol": round(realized_vol, 4),
        "net_change": round(net_change, 4),
        "anchor_change": round(anchor_change, 4),
    }


def _frame_close_change(frame, lookback):
    if frame is None or len(frame) < 2 or "close" not in frame:
        return 0.0
    closes = pd.to_numeric(frame["close"].tail(max(2, int(lookback))), errors="coerce").dropna()
    if len(closes) < 2:
        return 0.0
    first = _safe_float(closes.iloc[0], 0.0)
    last = _safe_float(closes.iloc[-1], 0.0)
    return (last / first - 1.0) if first > 0 else 0.0


def _latest_frame_value(frame, column, default=0.0):
    if frame is None or len(frame) == 0 or column not in frame:
        return default
    return _safe_float(frame[column].iloc[-1], default)


def classify_market_strategy_profile(df_1mth=None, df_1w=None, df_1d=None, df_4h=None):
    """Classify macro trend first, then select the indicator family for entries."""
    daily = df_1d if df_1d is not None and len(df_1d) > 0 else df_4h
    anchor_change = _frame_close_change(df_1mth, 12)
    weekly_change = _frame_close_change(df_1w, 52)
    daily_change = _frame_close_change(daily, 120)
    four_hour_change = _frame_close_change(df_4h, 120)

    daily_close = _latest_frame_value(daily, "close", 0.0)
    daily_ema50 = _latest_frame_value(daily, "ema50", daily_close)
    daily_ema200 = _latest_frame_value(daily, "ema200", daily_ema50)
    weekly_close = _latest_frame_value(df_1w, "close", daily_close)
    weekly_ema50 = _latest_frame_value(df_1w, "ema50", weekly_close)
    monthly_close = _latest_frame_value(df_1mth, "close", weekly_close)
    monthly_ema50 = _latest_frame_value(df_1mth, "ema50", monthly_close)

    adx = _latest_frame_value(daily, "adx14", 0.0)
    atr_rate = _latest_frame_value(daily, "atr14", 0.0) / max(daily_close, 1e-9)
    bb_pos = _latest_frame_value(daily, "bb_pos", 0.5)
    rsi = _latest_frame_value(daily, "rsi14", 50.0)
    supertrend_dir = _safe_int(_latest_frame_value(daily, "supertrend_dir", 0.0), 0)
    ichimoku_bias = _safe_int(_latest_frame_value(daily, "ichimoku_bias", 0.0), 0)

    macro_score = 0
    macro_score += 1 if monthly_close >= monthly_ema50 else -1
    macro_score += 1 if weekly_close >= weekly_ema50 else -1
    macro_score += 1 if daily_close >= daily_ema50 else -1
    macro_score += 1 if daily_ema50 >= daily_ema200 else -1
    macro_score += 1 if weekly_change > 0 else -1

    high_vol = atr_rate >= _safe_float(os.getenv("MARKET_PROFILE_HIGH_ATR_RATE", 0.045), 0.045) or adx >= _safe_float(os.getenv("MARKET_PROFILE_HIGH_ADX", 28), 28)
    range_like = (
        abs(daily_change) <= _safe_float(os.getenv("MARKET_PROFILE_RANGE_MAX_DAILY_CHANGE", 0.22), 0.22)
        and adx <= _safe_float(os.getenv("MARKET_PROFILE_RANGE_MAX_ADX", 22), 22)
    )
    bear_like = macro_score <= -2 or (weekly_change <= -0.18 and daily_close < daily_ema50)
    bull_like = macro_score >= 2 or (weekly_change >= 0.18 and daily_close > daily_ema50)

    if bear_like:
        phase = "bear"
        indicator_family = "supertrend_ema_adx"
    elif range_like:
        phase = "range_base"
        indicator_family = "rsi_vwap_bollinger"
    elif bull_like and high_vol:
        phase = "bull_high_vol"
        indicator_family = "ema_adx_atr"
    elif bull_like:
        phase = "bull"
        indicator_family = "ema_supertrend_ichimoku"
    else:
        phase = "range_base"
        indicator_family = "rsi_vwap_bollinger"

    return {
        "phase": phase,
        "indicator_family": indicator_family,
        "macro_score": macro_score,
        "monthly_change": round(anchor_change, 4),
        "weekly_change": round(weekly_change, 4),
        "daily_change": round(daily_change, 4),
        "four_hour_change": round(four_hour_change, 4),
        "adx": round(adx, 4),
        "atr_rate": round(atr_rate, 5),
        "rsi": round(rsi, 2),
        "bb_pos": round(bb_pos, 4),
        "supertrend_dir": supertrend_dir,
        "ichimoku_bias": ichimoku_bias,
        "daily_close_above_ema50": bool(daily_close >= daily_ema50),
        "weekly_close_above_ema50": bool(weekly_close >= weekly_ema50),
        "monthly_close_above_ema50": bool(monthly_close >= monthly_ema50),
    }


def _market_profile_score_adjustment(profile, direction, *, price, df_15m, df_4h):
    phase = str((profile or {}).get("phase") or "range_base")
    family = str((profile or {}).get("indicator_family") or "")
    direction_sign = 1 if direction == "long" else -1
    adjustment = 0.0
    reasons = []

    close_15m = _latest_frame_value(df_15m, "close", price)
    ema50_15m = _latest_frame_value(df_15m, "ema50", close_15m)
    vwap_15m = _latest_frame_value(df_15m, "vwap", close_15m)
    rsi_15m = _latest_frame_value(df_15m, "rsi14", 50.0)
    bb_pos_15m = _latest_frame_value(df_15m, "bb_pos", 0.5)
    supertrend_4h = _safe_int(_latest_frame_value(df_4h, "supertrend_dir", 0.0), 0)
    adx_4h = _latest_frame_value(df_4h, "adx14", 0.0)
    atr_rate_4h = _latest_frame_value(df_4h, "atr14", 0.0) / max(_latest_frame_value(df_4h, "close", price), 1e-9)
    ichimoku_4h = _safe_int(_latest_frame_value(df_4h, "ichimoku_bias", 0.0), 0)

    if phase == "bear":
        if direction == "short" and close_15m < ema50_15m and supertrend_4h <= 0 and adx_4h >= 18:
            adjustment += 0.09
            reasons.append("熊市EMA/Supertrend/ADX同向做空")
        elif direction == "long":
            adjustment -= 0.10
            reasons.append("熊市降低逆勢多單")
    elif phase == "range_base":
        if direction == "long" and rsi_15m <= 38 and (bb_pos_15m <= 0.30 or close_15m < vwap_15m):
            adjustment += 0.08
            reasons.append("震盪築底RSI/VWAP/Bollinger低位做多")
        elif direction == "short" and rsi_15m >= 62 and (bb_pos_15m >= 0.70 or close_15m > vwap_15m):
            adjustment += 0.07
            reasons.append("震盪築底RSI/VWAP/Bollinger高位做空")
        elif adx_4h < 18:
            adjustment -= 0.04
            reasons.append("低ADX震盪降低追突破")
    elif phase == "bull":
        if direction == "long" and close_15m > ema50_15m and supertrend_4h >= 0 and ichimoku_4h >= 0:
            adjustment += 0.10
            reasons.append("牛市EMA/Supertrend/Ichimoku同向做多")
        elif direction == "short":
            adjustment -= 0.08
            reasons.append("牛市降低逆勢空單")
    elif phase == "bull_high_vol":
        if direction == "long" and close_15m > ema50_15m and adx_4h >= 22:
            adjustment += 0.08
            reasons.append("高波動牛市EMA+ADX順勢做多")
        elif direction == "short" and supertrend_4h < 0 and adx_4h >= 24:
            adjustment += 0.03
            reasons.append("高波動牛市只允許強ADX回落空")
        if atr_rate_4h >= 0.045:
            adjustment *= 0.85
            reasons.append("ATR偏高降低追價權重")

    return {
        "adjustment": max(-0.14, min(0.14, adjustment)),
        "reasons": reasons,
        "indicator_family": family,
    }


def _build_daily_min_trade_plan(
    price,
    atr,
    df_15m,
    df_5m,
    htf,
    mid_trend,
    macro_bias=0.0,
    news_bias=0.0,
    breakout=0,
    volume_spike=False,
    regime="range",
    candlestick_turning=None,
    df_1d=None,
):
    """Create the small end-of-day trade from current structure, never a blind side."""
    entry = max(0.0, _safe_float(price, 0.0))
    regime_name = str(regime or "range")
    recent_high = _safe_float(df_15m["high"].tail(20).max(), entry)
    recent_low = _safe_float(df_15m["low"].tail(20).min(), entry)
    range_pos = max(0.0, min(1.0, (entry - recent_low) / max(recent_high - recent_low, 1e-6)))
    macro_score = _safe_float(macro_bias, 0.0) + _safe_float(news_bias, 0.0) * 0.25

    long_score = 0.0
    short_score = 0.0
    if regime_name in {"bear_trend", "bear_trend_strong"}:
        short_score += 0.55 if regime_name == "bear_trend_strong" else 0.35
    elif regime_name in {"bull_trend", "bull_trend_strong"}:
        long_score += 0.55 if regime_name == "bull_trend_strong" else 0.35
    long_score += 0.30 if _safe_int(htf, 0) == 1 else -0.20
    short_score += 0.30 if _safe_int(htf, 0) == -1 else -0.20
    long_score += 0.35 if _safe_int(mid_trend, 0) == 1 else -0.25
    short_score += 0.35 if _safe_int(mid_trend, 0) == -1 else -0.25
    long_score += max(-0.55, min(0.55, macro_score * 0.28))
    short_score += max(-0.55, min(0.55, -macro_score * 0.28))
    if range_pos >= 0.70:
        short_score += 0.95
        long_score -= 0.75
    elif range_pos <= 0.30:
        long_score += 0.95
        short_score -= 0.75
    if _safe_int(breakout, 0) == 1 and volume_spike:
        long_score += 0.55
    elif _safe_int(breakout, 0) == -1 and volume_spike:
        short_score += 0.55
    turn = candlestick_turning if isinstance(candlestick_turning, dict) else {}
    turn_direction = str(turn.get("direction") or "neutral")
    turn_count = _safe_int(turn.get("simultaneous_count"), 0)
    turn_confidence = _safe_float(turn.get("confidence"), 0.0)
    if (
        _is_truthy(os.getenv("TRADE_DAILY_MIN_USE_CANDLE_TURNING", "0"))
        and turn_count >= 2
        and turn_confidence >= max(0.42, _safe_float(os.getenv("TRADE_MULTI_TF_CANDLE_TURN_MIN_CONF", 0.48), 0.48))
    ):
        if turn_direction == "short":
            short_score += min(1.25, 0.55 + turn_confidence * 0.70)
            long_score -= min(0.95, 0.35 + turn_confidence * 0.50)
        elif turn_direction == "long" and long_score >= short_score:
            long_score += min(0.45, 0.15 + turn_confidence * 0.25)
    direction = "long" if long_score >= short_score else "short"
    if (
        regime_name == "bull_trend_strong"
        and direction == "long"
        and not (_safe_int(breakout, 0) == 1 and bool(volume_spike))
    ):
        direction = "short"
    atr_ref = max(_safe_float(atr, 0.0), entry * 0.001, 1e-6)
    if direction == "long":
        structural_sl = _safe_float(df_15m["low"].tail(10).min(), entry)
        structural_sl = min(structural_sl, _safe_float(df_5m["low"].tail(6).min(), entry))
        risk = max(entry - structural_sl, atr_ref * 0.5)
        if regime_name == "range":
            range_floor = max(0.0, recent_low - atr_ref * _safe_float(os.getenv("RANGE_SL_ATR_BUFFER", 0.35), 0.35))
            range_risk = max(entry - range_floor, atr_ref * _safe_float(os.getenv("RANGE_SL_MIN_ATR", 0.85), 0.85))
            risk = min(
                max(risk, range_risk),
                entry * _safe_float(os.getenv("RANGE_SL_MAX_RISK_RATE", 0.012), 0.012),
            )
        sl = entry - risk
        tp = entry + risk * 1.5
        final = "⏰ 每日保底做多"
    else:
        structural_sl = _safe_float(df_15m["high"].tail(10).max(), entry)
        structural_sl = max(structural_sl, _safe_float(df_5m["high"].tail(6).max(), entry))
        risk = max(structural_sl - entry, atr_ref * 0.5)
        if regime_name == "range":
            range_ceiling = recent_high + atr_ref * _safe_float(os.getenv("RANGE_SL_ATR_BUFFER", 0.35), 0.35)
            range_risk = max(range_ceiling - entry, atr_ref * _safe_float(os.getenv("RANGE_SL_MIN_ATR", 0.85), 0.85))
            risk = min(
                max(risk, range_risk),
                entry * _safe_float(os.getenv("RANGE_SL_MAX_RISK_RATE", 0.012), 0.012),
            )
        sl = entry + risk
        tp = entry - risk * 1.5
        final = "⏰ 每日保底做空"
    sign = 1 if direction == "long" else -1
    against_macro = macro_score * sign < -0.25
    local_confirmed = (_safe_int(breakout, 0) == sign) or bool(volume_spike)
    probe_size = max(
        0.001,
        min(0.03, _safe_float(os.getenv("DAILY_MIN_TRADE_PROBE_SIZE_RATIO", 0.001), 0.001)),
    )
    base_size = probe_size
    turn_matches_trade = turn_direction == direction
    max_size = base_size
    if turn_matches_trade and turn_direction == "long" and turn_count >= 2 and turn_confidence >= 0.48 and local_confirmed:
        max_size = min(
            0.12,
            max(base_size, _safe_float(os.getenv("DAILY_MIN_TRADE_TURN_LONG_MAX_SIZE_RATIO", base_size * 1.6), base_size * 1.6)),
        )
    elif turn_matches_trade and turn_direction == "short" and turn_count >= 2 and turn_confidence >= 0.48 and local_confirmed:
        max_size = min(
            0.06,
            max(base_size, _safe_float(os.getenv("DAILY_MIN_TRADE_TURN_SHORT_MAX_SIZE_RATIO", base_size * 1.25), base_size * 1.25)),
        )
    if against_macro:
        base_size = min(
            base_size,
            max(0.02, _safe_float(os.getenv("DAILY_MIN_TRADE_AGAINST_MACRO_SIZE_RATIO", 0.025), 0.025)),
        )
    if regime_name == "range":
        base_size = min(
            base_size,
            max(probe_size, _safe_float(os.getenv("DAILY_MIN_RANGE_SIZE_RATIO", base_size * 0.75), base_size * 0.75)),
        )
    style_profile = _daily_min_2024_style_profile(df_1d=df_1d, df_15m=df_15m)
    fast_release = bool(style_profile.get("active"))
    max_hold_sec = 0.0
    if fast_release:
        max_hold_sec = max(
            3600.0,
            _safe_float(os.getenv("DAILY_MIN_2024_STYLE_MAX_HOLD_SEC", 6 * 3600), 6 * 3600),
        )
        if not (turn_matches_trade and local_confirmed and turn_count >= 2 and turn_confidence >= 0.55):
            max_size = base_size
    return {
        "direction": direction,
        "final": final,
        "sl": sl,
        "tp": tp,
        "position_size": base_size,
        "max_position_size": max_size,
        "max_hold_sec": max_hold_sec,
        "against_macro": against_macro,
        "local_confirmed": local_confirmed,
        "macro_wait_recommended": bool(against_macro and not local_confirmed),
        "candlestick_turn_direction": turn_direction,
        "style_2024_profile": style_profile,
    }


def _recent_tp_sl_stats(limit=5):
    hits = POSITION_PANEL_STATE.get("close_hits")
    if not isinstance(hits, list):
        return {"total": 0, "tp": 0, "sl": 0}

    total = 0
    tp = 0
    sl = 0
    for hit in hits:
        if total >= max(1, _safe_int(limit, 5)):
            break
        reason = str(hit.get("reason") if isinstance(hit, dict) else "").upper()
        if reason not in {"TP", "SL"}:
            continue
        total += 1
        if reason == "TP":
            tp += 1
        elif reason == "SL":
            sl += 1

    return {"total": total, "tp": tp, "sl": sl}


def _format_tp_sl_win_rate_line(performance_stats=None, *, min_startup_samples=None, recent_limit=None):
    stats = performance_stats if isinstance(performance_stats, dict) else performance
    startup_total = max(0, _safe_int(stats.get("total"), 0))
    startup_wins = max(0, _safe_int(stats.get("win"), 0))
    min_samples = max(
        2,
        _safe_int(
            min_startup_samples
            if min_startup_samples is not None
            else os.getenv("TRADE_STARTUP_WIN_RATE_MIN_SAMPLES", 3),
            3,
        ),
    )

    if startup_total >= min_samples:
        startup_rate = startup_wins / max(startup_total, 1)
        return f"啟動後TP/SL勝率: {startup_rate:.2%}（{startup_wins}/{startup_total}）"

    lookback = max(
        min_samples,
        _safe_int(
            recent_limit
            if recent_limit is not None
            else os.getenv("TRADE_RECENT_WIN_RATE_LOOKBACK", 8),
            8,
        ),
    )
    recent = _recent_tp_sl_stats(lookback)
    if recent["total"] > 0:
        recent_rate = recent["tp"] / max(recent["total"], 1)
        mlx_recent = _recent_mlx_evaluated_stats(max(lookback * 3, 24))
        mlx_text = ""
        if mlx_recent["total"] > recent["total"]:
            mlx_rate = mlx_recent["win"] / max(mlx_recent["total"], 1)
            mlx_text = f"｜MLX近期評估 {mlx_rate:.2%}（{mlx_recent['win']}/{mlx_recent['total']}）"
        return (
            f"最近TP/SL勝率: {recent_rate:.2%}（{recent['tp']}/{recent['total']}）"
            f"{mlx_text}"
            f"｜啟動後樣本不足（{startup_total}/{min_samples}）"
        )

    return f"TP/SL勝率: 樣本不足（啟動後 {startup_total}/{min_samples}）"


def _recent_mlx_evaluated_stats(limit=24):
    try:
        with sqlite3.connect(str(MLX_LEARNING_DB_PATH), timeout=5) as connection:
            rows = connection.execute(
                """
                SELECT success
                FROM analysis_episode
                WHERE evaluated_at IS NOT NULL
                  AND success IS NOT NULL
                  AND direction IN ('long', 'short')
                ORDER BY evaluated_at DESC
                LIMIT ?
                """,
                (max(1, _safe_int(limit, 24)),),
            ).fetchall()
    except Exception:
        return {"total": 0, "win": 0, "loss": 0}
    total = len(rows)
    win = sum(1 for row in rows if _safe_int(row[0], 0) == 1)
    return {"total": total, "win": win, "loss": max(0, total - win)}


def _format_direction_color_text(text):
    text = str(text or "中性")
    bullish_markers = ("強多", "多頭", "偏多", "利多", "上漲", "做多", "買盤")
    bearish_markers = ("強空", "空頭", "偏空", "利空", "下跌", "做空", "賣壓")
    if any(marker in text for marker in bullish_markers):
        return f"🔴 {text}"
    if any(marker in text for marker in bearish_markers):
        return f"🟢 {text}"
    return f"⚪ {text}"


def _recent_sl_guard_reason(final, score, net_edge_rate_est, risk_rate, macro_bias, mid_trend, sr_bias):
    if not _is_truthy(os.getenv("TRADE_RECENT_SL_GUARD_ENABLED", "1")):
        return ""
    if final.startswith("觀望"):
        return ""

    lookback = max(3, _safe_int(os.getenv("TRADE_RECENT_SL_GUARD_LOOKBACK", 5), 5))
    max_sl = max(2, _safe_int(os.getenv("TRADE_RECENT_SL_GUARD_MAX_SL", 4), 4))
    stats = _recent_tp_sl_stats(lookback)
    if stats["total"] < lookback or stats["sl"] < max_sl:
        return ""

    min_score_gap = max(0.20, _safe_float(os.getenv("TRADE_SL_GUARD_MIN_SCORE_GAP", 0.36), 0.36))
    min_edge = max(0.0005, _safe_float(os.getenv("TRADE_SL_GUARD_MIN_EDGE_RATE", 0.003), 0.003))
    min_risk = max(0.001, _safe_float(os.getenv("TRADE_SL_GUARD_MIN_RISK_RATE", 0.003), 0.003))
    score_gap = abs(_safe_float(score, 0.5) - 0.5)

    if score_gap < min_score_gap:
        return f"觀望（連續SL防護-信號強度不足 {stats['sl']}/{stats['total']}）"
    if _safe_float(net_edge_rate_est, 0.0) < min_edge:
        return f"觀望（連續SL防護-期望值不足 {stats['sl']}/{stats['total']}）"
    if _safe_float(risk_rate, 0.0) < min_risk:
        return f"觀望（連續SL防護-停損距離過近 {stats['sl']}/{stats['total']}）"

    sr_bias = _safe_float(sr_bias, 0.0)
    macro_bias = _safe_float(macro_bias, 0.0)
    if "做空" in final and (mid_trend == 1 or macro_bias > 0.4 or sr_bias > 0.12):
        return f"觀望（連續SL防護-空單逆向共振 {stats['sl']}/{stats['total']}）"
    if "做多" in final and (mid_trend == -1 or macro_bias < -0.4 or sr_bias < -0.12):
        return f"觀望（連續SL防護-多單逆向共振 {stats['sl']}/{stats['total']}）"

    return ""


def sync_position_panel(current_price=None):
    _refresh_position_panel_account_state(force=False, log_on_error=False)
    entry_price = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), 0.0)
    last_price = _safe_float(current_price if current_price is not None else WS_PRICE, 0.0)
    direction = str(active_trade.get("direction") or "")
    lev = _safe_int(POSITION_PANEL_STATE.get("lev"), DEFAULT_LEV) or DEFAULT_LEV
    pair = str(POSITION_PANEL_STATE.get("pair") or DEFAULT_PAIR)
    funding_rate = _safe_float(POSITION_PANEL_STATE.get("funding_rate"), 0.0)
    fee_round_trip_rate = _safe_float(POSITION_PANEL_STATE.get("fee_round_trip_rate"), 0.001)
    hold_hours = max(0.0, (time.time() - _safe_float(active_trade.get("open_time"), time.time())) / 3600.0) if active_trade.get("open") else 0.0
    funding_cost_rate_est = max(0.0, abs(funding_rate) * min(hold_hours, 8.0) / 8.0) if active_trade.get("open") else 0.0
    size_ratio = max(0.0, _safe_float(active_trade.get("size"), 0.0)) if active_trade.get("open") else 0.0
    position_qty = _get_active_trade_position_qty() if active_trade.get("open") else 0.0
    capital_usage_ratio = max(0.0, _safe_float(POSITION_PANEL_STATE.get("capital_usage_ratio"), 0.0)) if active_trade.get("open") else 0.0
    max_size = round(_safe_float(active_trade.get("max_size"), 1.0), 4)
    min_size = round(_safe_float(active_trade.get("min_size"), 0.1), 4)
    scale_add_room = round(max(0.0, max_size - size_ratio), 4) if active_trade.get("open") else 0.0
    scale_reduce_room = round(max(0.0, size_ratio - min_size), 4) if active_trade.get("open") else 0.0
    latest_news = POSITION_PANEL_STATE.get("latest_news", [])
    if not isinstance(latest_news, list):
        latest_news = []

    if active_trade.get("open"):
        if not last_price:
            last_price = entry_price

        binance_mark_price = _safe_float(POSITION_PANEL_STATE.get("binance_mark_price"), 0.0)
        if binance_mark_price <= 0:
            binance_mark_price = last_price

        if position_qty > 0 and binance_mark_price > 0:
            position_notional_usdt = position_qty * binance_mark_price
            position_margin_usdt = position_notional_usdt / lev if lev > 0 else 0.0
        else:
            position_notional_usdt, position_margin_usdt = _estimate_panel_financials(entry_price, size_ratio, lev)

        if capital_usage_ratio <= 0 and position_margin_usdt > 0:
            capital_usage_ratio = _compute_capital_usage_ratio(
                position_margin_usdt,
                {
                    "available_balance": _safe_float(POSITION_PANEL_STATE.get("account_available_balance_usdt"), 0.0),
                    "wallet_balance": _safe_float(POSITION_PANEL_STATE.get("account_wallet_balance_usdt"), 0.0),
                    "margin_balance": _safe_float(POSITION_PANEL_STATE.get("account_margin_balance_usdt"), 0.0),
                },
            )
        if capital_usage_ratio <= 0:
            capital_usage_ratio = min(1.0, size_ratio)

        if entry_price > 0 and position_qty > 0:
            estimated_unrealized = (
                (last_price - entry_price) * position_qty
                if direction == "long"
                else (entry_price - last_price) * position_qty
            )
        elif entry_price > 0 and position_notional_usdt > 0:
            raw_move = (
                (last_price - entry_price) / entry_price
                if direction == "long"
                else (entry_price - last_price) / entry_price
            )
            estimated_unrealized = raw_move * position_notional_usdt
        else:
            estimated_unrealized = 0.0

        POSITION_PANEL_STATE["size"] = capital_usage_ratio
        POSITION_PANEL_STATE["size_ratio"] = size_ratio
        POSITION_PANEL_STATE["capital_usage_ratio"] = capital_usage_ratio
        POSITION_PANEL_STATE["binance_qty"] = position_qty
        POSITION_PANEL_STATE["position_notional_usdt"] = position_notional_usdt
        POSITION_PANEL_STATE["position_margin_usdt"] = position_margin_usdt
        POSITION_PANEL_STATE["estimated_unrealized_pnl_usdt"] = estimated_unrealized

        payload = {
            "open": True,
            "direction": direction or "long",
            "entry": round(entry_price, 4),
            "tp": round(_safe_float(active_trade.get("tp"), 0.0), 4),
            "sl": round(_safe_float(active_trade.get("sl"), 0.0), 4),
            "break_even_active": bool(active_trade.get("break_even_active", False)),
            "break_even_target": round(_safe_float(active_trade.get("break_even_target"), 0.0), 4),
            "break_even_ts": _safe_int(active_trade.get("break_even_ts"), 0),
            "size": round(capital_usage_ratio, 4),
            "size_ratio": round(size_ratio, 4),
            "capital_usage_ratio": round(capital_usage_ratio, 4),
            "open_since_ts": _safe_int(active_trade.get("open_time"), int(time.time())),
            "time_horizon": _normalize_trade_time_horizon(active_trade.get("time_horizon")),
            "max_hold_sec": round(_trade_max_hold_sec(active_trade.get("time_horizon")), 2),
            "max_size": max_size,
            "min_size": min_size,
            "scale_add_room": scale_add_room,
            "scale_reduce_room": scale_reduce_room,
            "add_count": _safe_int(active_trade.get("add_count"), 0),
            "reduce_count": _safe_int(active_trade.get("reduce_count"), 0),
            "quick_reduce_count": _safe_int(active_trade.get("quick_reduce_count"), 0),
            "quick_reduce_ts": _safe_int(active_trade.get("quick_reduce_ts"), 0),
            "daily_min_size_enforce_ts": _safe_int(active_trade.get("daily_min_size_enforce_ts"), 0),
            "last_adjust_ts": _safe_int(active_trade.get("last_adjust_ts"), 0),
            "scale_add_paused": bool(active_trade.get("scale_add_paused", False)),
            "scale_add_pause_reason": str(active_trade.get("scale_add_pause_reason") or ""),
            "scale_add_pause_ts": _safe_int(active_trade.get("scale_add_pause_ts"), 0),
            "tp_sl_adjusted_4h": bool(active_trade.get("tp_sl_adjusted_4h", False)),
            "last_close_reason": POSITION_PANEL_STATE.get("last_close_reason", ""),
            "last_close_price": round(_safe_float(POSITION_PANEL_STATE.get("last_close_price"), 0.0), 4),
            "last_close_ts": _safe_int(POSITION_PANEL_STATE.get("last_close_ts"), 0),
            "last_close_candle_high": round(_safe_float(POSITION_PANEL_STATE.get("last_close_candle_high"), 0.0), 4),
            "last_close_candle_low": round(_safe_float(POSITION_PANEL_STATE.get("last_close_candle_low"), 0.0), 4),
            "close_hits": POSITION_PANEL_STATE.get("close_hits", [])[:10],
            "last_sl_review": POSITION_PANEL_STATE.get("last_sl_review", {}),
            "daily_trade_date": str(POSITION_PANEL_STATE.get("daily_trade_date") or ""),
            "daily_trade_opened": bool(POSITION_PANEL_STATE.get("daily_trade_opened", False)),
            "daily_trade_source": str(POSITION_PANEL_STATE.get("daily_trade_source") or ""),
            "daily_trade_opened_ts": _safe_int(POSITION_PANEL_STATE.get("daily_trade_opened_ts"), 0),
            "latest_news": latest_news[:8],
            "binance_qty": round(position_qty, 6),
            "position_notional_usdt": round(position_notional_usdt, 4),
            "position_margin_usdt": round(position_margin_usdt, 4),
            "binance_entry_price": round(entry_price, 4),
            "binance_mark_price": round(binance_mark_price, 4),
            "binance_mark_price_ts": _safe_int(POSITION_PANEL_STATE.get("binance_mark_price_ts"), 0),
            "binance_break_even_price": round(entry_price * (1 + fee_round_trip_rate / 2), 4) if direction == "long" else round(entry_price * (1 - fee_round_trip_rate / 2), 4),
            "binance_unrealized_pnl_usdt": round(_safe_float(POSITION_PANEL_STATE.get("binance_unrealized_pnl_usdt"), estimated_unrealized), 4),
            "binance_unrealized_pnl_ts": _safe_int(POSITION_PANEL_STATE.get("binance_unrealized_pnl_ts"), 0),
            "estimated_unrealized_pnl_usdt": round(estimated_unrealized, 4),
            "account_available_balance_usdt": round(_safe_float(POSITION_PANEL_STATE.get("account_available_balance_usdt"), 0.0), 4),
            "account_wallet_balance_usdt": round(_safe_float(POSITION_PANEL_STATE.get("account_wallet_balance_usdt"), 0.0), 4),
            "account_margin_balance_usdt": round(_safe_float(POSITION_PANEL_STATE.get("account_margin_balance_usdt"), 0.0), 4),
            "binance_spot_total_assets_usdt": round(_safe_float(POSITION_PANEL_STATE.get("binance_spot_total_assets_usdt"), 0.0), 4),
            "binance_futures_total_assets_usdt": round(_safe_float(POSITION_PANEL_STATE.get("binance_futures_total_assets_usdt"), 0.0), 4),
            "binance_total_assets_usdt": round(_safe_float(POSITION_PANEL_STATE.get("binance_total_assets_usdt"), 0.0), 4),
            "fee_round_trip_rate": round(fee_round_trip_rate, 6),
            "funding_rate": round(funding_rate, 8),
            "funding_next_ts": _safe_int(POSITION_PANEL_STATE.get("funding_next_ts"), 0),
            "funding_cost_rate_est": round(funding_cost_rate_est, 6),
            "total_cost_rate_est": round(fee_round_trip_rate + funding_cost_rate_est, 6),
            "cost_eval_hold_hours": round(hold_hours, 2),
            "pair": pair,
            "lev": lev,
            "ts": int(time.time()),
        }
    else:
        payload = {
            "open": False,
            "direction": direction or str(POSITION_PANEL_STATE.get("direction") or "long"),
            "entry": round(entry_price or _safe_float(POSITION_PANEL_STATE.get("entry"), 0.0), 4),
            "tp": round(_safe_float(active_trade.get("tp"), _safe_float(POSITION_PANEL_STATE.get("tp"), 0.0)), 4),
            "sl": round(_safe_float(active_trade.get("sl"), _safe_float(POSITION_PANEL_STATE.get("sl"), 0.0)), 4),
            "break_even_active": False,
            "break_even_target": 0.0,
            "break_even_ts": 0,
            "size": 0.0,
            "size_ratio": 0.0,
            "capital_usage_ratio": 0.0,
            "open_since_ts": 0,
            "time_horizon": _normalize_trade_time_horizon(active_trade.get("time_horizon")),
            "max_hold_sec": round(_trade_max_hold_sec(active_trade.get("time_horizon")), 2),
            "max_size": round(_safe_float(active_trade.get("max_size"), 1.0), 4),
            "min_size": round(_safe_float(active_trade.get("min_size"), 0.1), 4),
            "scale_add_room": 0.0,
            "scale_reduce_room": 0.0,
            "add_count": 0,
            "reduce_count": 0,
            "quick_reduce_count": 0,
            "quick_reduce_ts": 0,
            "daily_min_size_enforce_ts": 0,
            "last_adjust_ts": 0,
            "scale_add_paused": False,
            "scale_add_pause_reason": "",
            "scale_add_pause_ts": 0,
            "tp_sl_adjusted_4h": False,
            "last_close_reason": POSITION_PANEL_STATE.get("last_close_reason", ""),
            "last_close_price": round(_safe_float(POSITION_PANEL_STATE.get("last_close_price"), 0.0), 4),
            "last_close_ts": _safe_int(POSITION_PANEL_STATE.get("last_close_ts"), 0),
            "last_close_candle_high": round(_safe_float(POSITION_PANEL_STATE.get("last_close_candle_high"), 0.0), 4),
            "last_close_candle_low": round(_safe_float(POSITION_PANEL_STATE.get("last_close_candle_low"), 0.0), 4),
            "close_hits": POSITION_PANEL_STATE.get("close_hits", [])[:10],
            "last_sl_review": POSITION_PANEL_STATE.get("last_sl_review", {}),
            "daily_trade_date": str(POSITION_PANEL_STATE.get("daily_trade_date") or ""),
            "daily_trade_opened": bool(POSITION_PANEL_STATE.get("daily_trade_opened", False)),
            "daily_trade_source": str(POSITION_PANEL_STATE.get("daily_trade_source") or ""),
            "daily_trade_opened_ts": _safe_int(POSITION_PANEL_STATE.get("daily_trade_opened_ts"), 0),
            "latest_news": latest_news[:8],
            "binance_qty": 0.0,
            "position_notional_usdt": 0.0,
            "position_margin_usdt": 0.0,
            "binance_entry_price": 0.0,
            "binance_mark_price": round(_safe_float(POSITION_PANEL_STATE.get("binance_mark_price"), last_price), 4),
            "binance_mark_price_ts": _safe_int(POSITION_PANEL_STATE.get("binance_mark_price_ts"), 0),
            "binance_break_even_price": 0.0,
            "binance_unrealized_pnl_usdt": 0.0,
            "binance_unrealized_pnl_ts": 0,
            "estimated_unrealized_pnl_usdt": 0.0,
            "account_available_balance_usdt": round(_safe_float(POSITION_PANEL_STATE.get("account_available_balance_usdt"), 0.0), 4),
            "account_wallet_balance_usdt": round(_safe_float(POSITION_PANEL_STATE.get("account_wallet_balance_usdt"), 0.0), 4),
            "account_margin_balance_usdt": round(_safe_float(POSITION_PANEL_STATE.get("account_margin_balance_usdt"), 0.0), 4),
            "binance_spot_total_assets_usdt": round(_safe_float(POSITION_PANEL_STATE.get("binance_spot_total_assets_usdt"), 0.0), 4),
            "binance_futures_total_assets_usdt": round(_safe_float(POSITION_PANEL_STATE.get("binance_futures_total_assets_usdt"), 0.0), 4),
            "binance_total_assets_usdt": round(_safe_float(POSITION_PANEL_STATE.get("binance_total_assets_usdt"), 0.0), 4),
            "fee_round_trip_rate": round(fee_round_trip_rate, 6),
            "funding_rate": round(funding_rate, 8),
            "funding_next_ts": _safe_int(POSITION_PANEL_STATE.get("funding_next_ts"), 0),
            "funding_cost_rate_est": 0.0,
            "total_cost_rate_est": round(fee_round_trip_rate, 6),
            "cost_eval_hold_hours": 0.0,
            "pair": pair,
            "lev": lev,
            "ts": int(time.time()),
        }

    POSITION_PANEL_STATE.update(payload)
    _write_json_atomic(POSITION_PANEL_FILE, payload)
    _queue_panel_realtime_publish(payload)


def _execute_copy_trade_scale(direction, delta_ratio, reduce=False, mark_price=None):
    """對既有倉位執行實單補倉/減倉。"""
    if direction not in {"long", "short"}:
        return False, "方向無效"

    if not _get_follow_mode_enabled() or not _is_real_copy_enabled():
        return False, "實單未啟用"

    try:
        positions = _binance_futures_signed_get("/fapi/v2/positionRisk")
    except Exception as e:
        return False, f"查詢持倉失敗: {e}"

    active_rows = []
    for item in positions if isinstance(positions, list) else []:
        if not isinstance(item, dict):
            continue
        if str(item.get("symbol") or "").upper() != COPY_TRADE_SYMBOL:
            continue
        amt = _safe_float(item.get("positionAmt"), 0.0)
        if abs(amt) <= 1e-9:
            continue
        active_rows.append(item)

    if not active_rows:
        return False, "Binance 無持倉，無法縮放"

    row = max(active_rows, key=lambda x: abs(_safe_float(x.get("positionAmt"), 0.0)))
    position_amt = _safe_float(row.get("positionAmt"), 0.0)
    if abs(position_amt) <= 1e-9:
        return False, "持倉數量為 0"

    leverage = max(1, _safe_int(row.get("leverage"), DEFAULT_LEV))
    price_ref = _safe_float(mark_price, 0.0) or _safe_float(row.get("markPrice"), 0.0) or _safe_float(WS_PRICE, 0.0)
    if reduce:
        qty = _calc_copy_trade_qty(delta_ratio, leverage=leverage, eth_price=price_ref)
    else:
        qty = _calc_copy_trade_qty_with_buffer(
            delta_ratio,
            leverage=leverage,
            eth_price=price_ref,
            enforce_min=False,
        )

    # 減倉不可超過現有倉位
    if reduce:
        qty = min(qty, abs(position_amt))

    qty = math.floor(max(0.0, qty) * 1000.0) / 1000.0
    if qty < COPY_TRADE_MIN_QTY:
        if not reduce:
            active_trade["scale_add_paused"] = True
            active_trade["scale_add_pause_reason"] = f"補倉量 {qty:.3f} ETH 低於最低補倉顆數 {COPY_TRADE_MIN_QTY:.3f} ETH"
            active_trade["scale_add_pause_ts"] = time.time()
            return False, f"補倉已暫停：{active_trade['scale_add_pause_reason']}"
        return False, f"下單量低於最小值 {COPY_TRADE_MIN_QTY:.3f}"

    dual_side = _is_binance_dual_side_mode()
    position_side = "LONG" if direction == "long" else "SHORT"

    if reduce:
        side = "SELL" if direction == "long" else "BUY"
    else:
        side = "BUY" if direction == "long" else "SELL"

    order_params = {
        "symbol": COPY_TRADE_SYMBOL,
        "side": side,
        "type": "MARKET",
        "quantity": qty,
    }
    if dual_side:
        order_params["positionSide"] = position_side
    elif reduce:
        order_params["reduceOnly"] = "true"

    try:
        if reduce:
            _binance_futures_signed_request("POST", "/fapi/v1/order", order_params)
            return True, f"減倉實單成功 qty={qty:.3f}"

        retry_steps = max(0, _safe_int(os.getenv("COPY_TRADE_MARGIN_RETRY_STEPS", 6), 6))
        retry_decay = min(0.95, max(0.5, _safe_float(os.getenv("COPY_TRADE_MARGIN_RETRY_DECAY", 0.85), 0.85)))

        def _recalc_scale_qty(retry_attempt):
            buffer_ratio = max(0.35, 0.82 - (0.08 * retry_attempt))
            return _calc_copy_trade_qty_with_buffer(
                delta_ratio,
                leverage=leverage,
                eth_price=price_ref,
                extra_buffer_ratio=buffer_ratio,
                enforce_min=False,
            )

        order_resp, final_qty, last_err = _execute_binance_market_order_with_retry(
            order_params,
            initial_qty=qty,
            min_qty=COPY_TRADE_MIN_QTY,
            retry_steps=retry_steps,
            retry_decay=retry_decay,
            recalc_qty_fn=_recalc_scale_qty,
        )
        if order_resp is None:
            if _is_binance_insufficient_margin_error(last_err):
                fallback_qty = math.floor(max(0.0, _safe_float(_recalc_scale_qty(retry_steps + 1), 0.0)) * 1000.0) / 1000.0
                if fallback_qty < COPY_TRADE_MIN_QTY:
                    active_trade["scale_add_paused"] = True
                    active_trade["scale_add_pause_reason"] = f"補倉量 {fallback_qty:.3f} ETH 低於最低補倉顆數 {COPY_TRADE_MIN_QTY:.3f} ETH"
                    active_trade["scale_add_pause_ts"] = time.time()
                    return False, f"補倉已暫停：{active_trade['scale_add_pause_reason']}"
                return False, _format_binance_margin_failure("補倉實單失敗", final_qty, leverage, price_ref, last_err)
            return False, f"補倉實單失敗: {last_err}"
        return True, f"補倉實單成功 qty={final_qty:.3f}"
    except Exception as e:
        return False, f"{('減倉' if reduce else '補倉')}實單失敗: {e}"


def _enforce_daily_min_trade_size(planned_size, current_price):
    if not (_get_follow_mode_enabled() and _is_real_copy_enabled()):
        return ""
    if not active_trade.get("open"):
        return ""
    direction = str(active_trade.get("direction") or "")
    actual_size = max(0.0, _safe_float(active_trade.get("size"), 0.0))
    target_size = max(
        0.0,
        min(
            0.10,
            _safe_float(
                planned_size,
                _safe_float(os.getenv("DAILY_MIN_TRADE_SIZE_RATIO", 0.05), 0.05),
            ),
        ),
    )
    max_actual = min(
        0.12,
        max(
            target_size + 0.025,
            target_size * max(1.15, _safe_float(os.getenv("DAILY_MIN_TRADE_SIZE_TOLERANCE_MULT", 1.35), 1.35)),
        ),
    )
    if direction not in {"long", "short"} or actual_size <= max_actual:
        return ""
    now_ts = time.time()
    last_ts = _safe_float(active_trade.get("daily_min_size_enforce_ts"), 0.0)
    cooldown = max(30.0, _safe_float(os.getenv("DAILY_MIN_TRADE_SIZE_ENFORCE_COOLDOWN_SEC", 60), 60))
    if now_ts - last_ts < cooldown:
        return ""
    active_trade["daily_min_size_enforce_ts"] = now_ts
    reduce_delta = actual_size - target_size
    if reduce_delta <= 0:
        return ""
    ok, msg = _execute_copy_trade_scale(
        direction,
        reduce_delta,
        reduce=True,
        mark_price=current_price,
    )
    if not ok:
        warning = (
            f"⚠️ 每日最低單倉位校正失敗\n"
            f"計畫: {target_size*100:.1f}% | 實際: {actual_size*100:.1f}%\n"
            f"{msg}"
        )
        send_private_telegram(warning, priority=True)
        return warning
    sync_active_trade_from_binance(send_notice=False)
    corrected = _safe_float(active_trade.get("size"), target_size)
    if corrected > max_actual:
        active_trade["daily_min_size_enforce_ts"] = 0.0
        sync_position_panel(current_price)
    notice = (
        f"🧯 每日最低單倉位已校正\n"
        f"計畫: {target_size*100:.1f}% | 原實際: {actual_size*100:.1f}% | 校正後: {corrected*100:.1f}%\n"
        f"{msg}"
    )
    send_telegram(notice, priority=True)
    return notice


def _estimate_trade_cost_rate_est(hold_hours=None) -> float:
    fee_round_trip_rate = max(0.0, _safe_float(POSITION_PANEL_STATE.get("fee_round_trip_rate"), 0.001))
    est_slippage_rate = max(0.0, _safe_float(os.getenv("TRADE_EST_SLIPPAGE_RATE", 0.0004), 0.0004))
    hours = _safe_float(hold_hours, _safe_float(os.getenv("TRADE_EST_HOLD_HOURS", 6.0), 6.0))
    funding_rate_abs = abs(_safe_float(POSITION_PANEL_STATE.get("funding_rate"), 0.0))
    funding_cost_rate_est = max(0.0, funding_rate_abs * max(hours / 8.0, 0.0))
    return fee_round_trip_rate + est_slippage_rate + funding_cost_rate_est


def _is_break_even_or_better(direction, entry, sl) -> bool:
    direction = str(direction or "")
    entry = _safe_float(entry, 0.0)
    sl = _safe_float(sl, 0.0)

    if direction not in {"long", "short"} or entry <= 0 or sl <= 0:
        return False

    tolerance = entry * 0.0001
    if direction == "long":
        return sl >= (entry - tolerance)
    return sl <= (entry + tolerance)


def _set_break_even_state(active: bool, target=0.0, ts=None):
    active_trade["break_even_active"] = bool(active)
    active_trade["break_even_target"] = _safe_float(target, 0.0) if active else 0.0
    active_trade["break_even_ts"] = _safe_float(ts, time.time()) if active else 0.0


def _sync_break_even_state_from_sl(direction, entry, sl, preserve_existing=True, now_ts=None):
    if not _is_break_even_or_better(direction, entry, sl):
        _set_break_even_state(False)
        return False

    existing_target = _safe_float(active_trade.get("break_even_target"), 0.0) if preserve_existing else 0.0
    existing_ts = _safe_float(active_trade.get("break_even_ts"), 0.0) if preserve_existing else 0.0
    target = existing_target if existing_target > 0 else _safe_float(sl, 0.0)
    now_value = _safe_float(now_ts, 0.0)
    ts = existing_ts if existing_ts > 0 else (now_value if now_value > 0 else time.time())
    _set_break_even_state(True, target=target, ts=ts)
    return True


def _estimate_break_even_buffer_rate(hold_hours=None) -> float:
    est_cost_rate = max(0.0, _estimate_trade_cost_rate_est(hold_hours=hold_hours))
    min_buffer_rate = max(
        0.0010,
        _safe_float_env_names(("AUTO_BREAK_EVEN_MIN_BUFFER_RATE", "TRADE_BREAK_EVEN_BUFFER_PCT"), 0.0010),
    )
    extra_buffer_rate = max(
        0.0004,
        _safe_float_env_names(("AUTO_BREAK_EVEN_EXTRA_BUFFER_RATE",), 0.0006),
    )
    return max(min_buffer_rate, est_cost_rate + extra_buffer_rate)


def _calc_break_even_stop_price(direction, entry, current_price=0.0, tp=0.0, hold_hours=None):
    direction = str(direction or "")
    entry = _safe_float(entry, 0.0)
    current_price = _safe_float(current_price, 0.0)
    tp = _safe_float(tp, 0.0)

    if direction not in {"long", "short"} or entry <= 0:
        return 0.0, 0.0

    buffer_rate = _estimate_break_even_buffer_rate(hold_hours=hold_hours)
    if direction == "long":
        target = entry * (1 + buffer_rate)
        if tp > 0:
            target = min(target, tp * 0.998)
        if current_price > 0 and target >= current_price:
            return 0.0, buffer_rate
    else:
        target = entry * (1 - buffer_rate)
        if tp > 0:
            target = max(target, tp * 1.002)
        if current_price > 0 and target <= current_price:
            return 0.0, buffer_rate

    return target, buffer_rate


SCALING_MARKET_STATE = {
    "ts": 0.0,
    "price": 0.0,
    "atr": 0.0,
    "htf": 0,
    "mid_trend": 0,
    "regime": "range",
    "breakout": 0,
    "support_hits": 0,
    "resistance_hits": 0,
    "sr_bias": 0.0,
    "volume_ratio": 0.0,
    "volume_spike": False,
    "buy_pressure": False,
    "sell_pressure": False,
}


def _update_scaling_market_state(
    price,
    atr,
    htf,
    mid_trend,
    regime,
    breakout,
    sr_analysis=None,
    volume_ratio=0.0,
    volume_spike=False,
    buy_pressure=False,
    sell_pressure=False,
):
    sr_payload = sr_analysis if isinstance(sr_analysis, dict) else {}
    SCALING_MARKET_STATE.update(
        {
            "ts": time.time(),
            "price": _safe_float(price, 0.0),
            "atr": max(0.0, _safe_float(atr, 0.0)),
            "htf": _safe_int(htf, 0),
            "mid_trend": _safe_int(mid_trend, 0),
            "regime": str(regime or "range"),
            "breakout": _safe_int(breakout, 0),
            "support_hits": max(0, _safe_int(sr_payload.get("support_hits"), 0)),
            "resistance_hits": max(0, _safe_int(sr_payload.get("resistance_hits"), 0)),
            "sr_bias": _safe_float(sr_payload.get("bias"), 0.0),
            "volume_ratio": max(0.0, _safe_float(volume_ratio, 0.0)),
            "volume_spike": bool(volume_spike),
            "buy_pressure": bool(buy_pressure),
            "sell_pressure": bool(sell_pressure),
        }
    )


def _derive_scaling_bounds(base_size):
    size = max(0.0, _safe_float(base_size, 0.0))
    if size <= 0:
        return 1.0, 0.1

    add_buffer = max(0.04, _safe_float(os.getenv("SCALE_MAX_ADD_BUFFER", 0.16), 0.16))
    max_multiplier = max(1.15, _safe_float(os.getenv("SCALE_MAX_SIZE_MULTIPLIER", 1.8), 1.8))
    hold_ratio = min(0.9, max(0.35, _safe_float(os.getenv("SCALE_MIN_HOLD_RATIO", 0.55), 0.55)))
    floor_size = 0.1 if size >= 0.1 else size

    max_size = min(1.0, max(size + add_buffer, size * max_multiplier))
    min_size = max(floor_size, min(size, size * hold_ratio))

    if min_size > max_size:
        min_size = min(size, max_size)

    return round(max_size, 6), round(min_size, 6)


def _cap_initial_position_size(size_ratio):
    size = max(0.0, _safe_float(size_ratio, 0.0))
    if size <= 0:
        return 0.0

    default_cap = _safe_float(os.getenv("TRADE_MAX_OPEN_SIZE_RATIO", 0.55), 0.55)
    max_open = _safe_float(os.getenv("TRADE_INITIAL_MAX_OPEN_SIZE_RATIO", default_cap), default_cap)
    max_open = min(0.95, max(0.05, max_open))

    if _is_truthy(os.getenv("TRADE_AUTO_SCALE_ENABLED", "0")):
        reserve = min(0.5, max(0.0, _safe_float(os.getenv("TRADE_SCALE_RESERVE_RATIO", 0.20), 0.20)))
        max_open = min(max_open, max(0.05, 1.0 - reserve))

    return min(size, max_open)


def _resolve_scaling_bounds(size_ratio, raw_max_size=None, raw_min_size=None):
    size = max(0.0, _safe_float(size_ratio, 0.0))
    default_max, default_min = _derive_scaling_bounds(size)

    max_candidate = _safe_float(raw_max_size, default_max)
    min_candidate = _safe_float(raw_min_size, default_min)
    floor_size = 0.1 if size >= 0.1 else size

    if max_candidate <= 0:
        max_size = default_max
    else:
        max_size = min(1.0, max(size, max_candidate))

    if min_candidate <= 0:
        min_size = default_min
    else:
        min_size = max(floor_size, min(size, min_candidate))

    if min_size > max_size:
        min_size = min(size, max_size)

    return round(max_size, 6), round(min_size, 6)


def _calc_scaling_progress(direction, entry, current_price, tp, sl):
    entry = _safe_float(entry, 0.0)
    px = _safe_float(current_price, 0.0)
    tp = _safe_float(tp, 0.0)
    sl = _safe_float(sl, 0.0)

    if direction not in {"long", "short"} or entry <= 0 or px <= 0 or tp <= 0 or sl <= 0:
        return {}

    if direction == "long":
        initial_risk_abs = max(entry - sl, 1e-9)
        target_abs = max(tp - entry, 1e-9)
        drawdown_progress = max(entry - px, 0.0) / initial_risk_abs
        profit_progress = max(px - entry, 0.0) / target_abs
        earned_r_multiple = max(px - entry, 0.0) / initial_risk_abs
    else:
        initial_risk_abs = max(sl - entry, 1e-9)
        target_abs = max(entry - tp, 1e-9)
        drawdown_progress = max(px - entry, 0.0) / initial_risk_abs
        profit_progress = max(entry - px, 0.0) / target_abs
        earned_r_multiple = max(entry - px, 0.0) / initial_risk_abs

    return {
        "move_abs": abs(px - entry),
        "initial_risk_abs": initial_risk_abs,
        "initial_risk_rate": initial_risk_abs / max(entry, 1e-9),
        "drawdown_progress": drawdown_progress,
        "profit_progress": profit_progress,
        "earned_r_multiple": earned_r_multiple,
    }


def _get_scaling_trend_score(direction):
    sign = 1 if direction == "long" else -1
    ctx = SCALING_MARKET_STATE
    score = 0.0

    htf = _safe_int(ctx.get("htf"), 0)
    mid_trend = _safe_int(ctx.get("mid_trend"), 0)
    breakout = _safe_int(ctx.get("breakout"), 0)
    regime = str(ctx.get("regime") or "range")
    sr_bias = _safe_float(ctx.get("sr_bias"), 0.0)

    if htf == sign:
        score += 1.2
    elif htf == -sign:
        score -= 1.2

    if mid_trend == sign:
        score += 1.0
    elif mid_trend == -sign:
        score -= 1.0

    if sign == 1:
        if regime == "bull_trend_strong":
            score += 1.1
        elif regime == "bull_trend":
            score += 0.6
        elif regime == "bear_trend_strong":
            score -= 1.1
        elif regime == "bear_trend":
            score -= 0.6
        score += max(-0.5, min(0.5, sr_bias))
    else:
        if regime == "bear_trend_strong":
            score += 1.1
        elif regime == "bear_trend":
            score += 0.6
        elif regime == "bull_trend_strong":
            score -= 1.1
        elif regime == "bull_trend":
            score -= 0.6
        score += max(-0.5, min(0.5, -sr_bias))

    if breakout == sign:
        score += 0.6
    elif breakout == -sign:
        score -= 0.8

    return score


def _has_scaling_opposing_pressure(direction):
    resistance_hits = max(0, _safe_int(SCALING_MARKET_STATE.get("resistance_hits"), 0))
    support_hits = max(0, _safe_int(SCALING_MARKET_STATE.get("support_hits"), 0))
    sr_bias = _safe_float(SCALING_MARKET_STATE.get("sr_bias"), 0.0)
    breakout = _safe_int(SCALING_MARKET_STATE.get("breakout"), 0)

    if direction == "long":
        return resistance_hits >= 2 or sr_bias <= -0.35 or breakout == -1
    return support_hits >= 2 or sr_bias >= 0.35 or breakout == 1


def maybe_activate_auto_break_even(current_price, atr=None, now_ts=None):
    if not _is_truthy(os.getenv("AUTO_BREAK_EVEN_ENABLED", "1")):
        return False
    if not active_trade.get("open"):
        return False

    direction = str(active_trade.get("direction") or "")
    entry = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), 0.0)
    tp = _safe_float(active_trade.get("tp"), 0.0)
    sl = _safe_float(active_trade.get("sl"), 0.0)
    px = _safe_float(current_price, 0.0)

    if direction not in {"long", "short"} or entry <= 0 or tp <= 0 or sl <= 0 or px <= 0:
        return False

    resolved_now_ts = _safe_float(now_ts, 0.0)
    if resolved_now_ts <= 0:
        resolved_now_ts = time.time()

    if _sync_break_even_state_from_sl(direction, entry, sl, preserve_existing=True, now_ts=resolved_now_ts):
        return False

    open_ts = _safe_float(active_trade.get("open_time"), 0.0)
    min_hold_sec = max(0.0, _safe_float_env_names(("AUTO_BREAK_EVEN_MIN_HOLD_SEC",), 600.0))
    daily_min_position = _is_daily_min_position()
    if daily_min_position:
        min_hold_sec = min(
            min_hold_sec,
            max(60.0, _safe_float(os.getenv("DAILY_MIN_BREAK_EVEN_MIN_HOLD_SEC", 120), 120)),
        )
    if open_ts > 0 and (resolved_now_ts - open_ts) < min_hold_sec:
        return False
    hold_hours = max(0.0, (resolved_now_ts - open_ts) / 3600.0) if open_ts > 0 else 0.0

    progress = _calc_scaling_progress(direction, entry, px, tp, sl)
    if not progress:
        return False

    require_reduce_first = _is_truthy(os.getenv("AUTO_BREAK_EVEN_REQUIRE_REDUCE_FIRST", "0"))
    if require_reduce_first and max(0, _safe_int(active_trade.get("reduce_count"), 0)) < 1:
        return False

    target, buffer_rate = _calc_break_even_stop_price(
        direction,
        entry,
        current_price=px,
        tp=tp,
        hold_hours=hold_hours,
    )
    if target <= 0:
        return False

    trigger_r = max(
        1.10,
        _safe_float_env_names(("AUTO_BREAK_EVEN_TRIGGER_R", "TRADE_BREAK_EVEN_TRIGGER_R"), 1.35),
    )
    trigger_tp_progress = min(
        0.95,
        max(0.45, _safe_float_env_names(("AUTO_BREAK_EVEN_TRIGGER_TP_PROGRESS",), 0.55)),
    )
    min_edge_rate = max(0.0008, _safe_float_env_names(("AUTO_BREAK_EVEN_MIN_EDGE_RATE",), 0.0018))
    min_profit_rate = max(
        buffer_rate + min_edge_rate,
        _safe_float_env_names(("AUTO_BREAK_EVEN_MIN_PROFIT_RATE",), 0.0065),
    )
    min_gap_atr_factor = max(0.25, _safe_float_env_names(("AUTO_BREAK_EVEN_MIN_GAP_ATR_FACTOR",), 0.35))
    if daily_min_position:
        trigger_r = min(trigger_r, max(0.45, _safe_float(os.getenv("DAILY_MIN_BREAK_EVEN_TRIGGER_R", 0.65), 0.65)))
        trigger_tp_progress = min(trigger_tp_progress, max(0.25, _safe_float(os.getenv("DAILY_MIN_BREAK_EVEN_TP_PROGRESS", 0.32), 0.32)))
        min_profit_rate = min(min_profit_rate, max(buffer_rate + 0.0005, _safe_float(os.getenv("DAILY_MIN_BREAK_EVEN_MIN_PROFIT_RATE", 0.0022), 0.0022)))
        min_gap_atr_factor = min(min_gap_atr_factor, max(0.12, _safe_float(os.getenv("DAILY_MIN_BREAK_EVEN_MIN_GAP_ATR_FACTOR", 0.18), 0.18)))

    favorable_move_rate = (px - entry) / entry if direction == "long" else (entry - px) / entry
    gap_abs = (px - target) if direction == "long" else (target - px)
    atr_value = max(0.0, _safe_float(atr, 0.0))
    min_gap_abs = max(entry * min_edge_rate, atr_value * min_gap_atr_factor)
    trend_score = _get_scaling_trend_score(direction)
    opposing_pressure = _has_scaling_opposing_pressure(direction)
    earned_r_multiple = _safe_float(progress.get("earned_r_multiple"), 0.0)
    profit_progress = _safe_float(progress.get("profit_progress"), 0.0)

    # 趨勢順暢時不要過早把 SL 推到成本附近，避免正常回踩把單洗掉。
    if trend_score >= 1.8 and not opposing_pressure:
        trigger_r = max(trigger_r, 1.55)
        trigger_tp_progress = max(trigger_tp_progress, 0.68)
        min_profit_rate = max(min_profit_rate, 0.0085)
        min_gap_abs = max(min_gap_abs, atr_value * 0.45)

    if favorable_move_rate < min_profit_rate:
        return False
    if gap_abs < min_gap_abs:
        return False
    be_ready = (
        (earned_r_multiple >= trigger_r and profit_progress >= trigger_tp_progress)
        or earned_r_multiple >= (trigger_r + 0.45)
        or profit_progress >= min(0.97, trigger_tp_progress + 0.20)
    )
    if not be_ready:
        return False

    old_sl = sl
    new_sl = round(target, 2)
    if direction == "long" and new_sl <= old_sl + 1e-9:
        return False
    if direction == "short" and new_sl >= old_sl - 1e-9:
        return False

    active_trade["sl"] = float(new_sl)
    _set_break_even_state(True, target=new_sl, ts=resolved_now_ts)
    sync_position_panel(px)

    sync_msg = ""
    if _get_follow_mode_enabled() and _is_real_copy_enabled():
        try:
            _, binance_msg = update_copy_trade_tp_sl(active_trade.get("tp"), new_sl)
            sync_msg = f"\n{binance_msg}"
        except Exception as e:
            sync_msg = f"\n⚠️ Binance 保本 SL 同步失敗: {e}"

    send_telegram(
        f"🛡️ 自動保本啟動\n"
        f"方向: {direction} | 現價: {px:.2f}\n"
        f"進場: {entry:.2f} | TP: {tp:.2f}\n"
        f"SL: {old_sl:.2f} → {new_sl:.2f}\n"
        f"觸發條件: R={earned_r_multiple:.2f} | TP進度={profit_progress*100:.1f}% | 趨勢分數={trend_score:.2f}"
        f"{sync_msg}",
        priority=True,
    )
    return True


def maybe_take_quick_profit_reduce(current_price, atr=None, now_ts=None):
    if not _is_truthy(os.getenv("QUICK_PROFIT_REDUCE_ENABLED", "1")):
        return False
    if not active_trade.get("open"):
        return False
    direction = str(active_trade.get("direction") or "")
    entry = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), 0.0)
    size = max(0.0, _safe_float(active_trade.get("size"), 0.0))
    min_size = max(0.0, _safe_float(active_trade.get("min_size"), 0.0))
    progress = _calc_scaling_progress(direction, entry, current_price, active_trade.get("tp"), active_trade.get("sl"))
    if direction not in {"long", "short"} or not progress or size <= min_size + 1e-9:
        return False

    now_ts = _safe_float(now_ts, 0.0) or time.time()
    last_ts = _safe_float(active_trade.get("quick_reduce_ts"), 0.0)
    cooldown = max(180.0, _safe_float(os.getenv("QUICK_PROFIT_REDUCE_COOLDOWN_SEC", 900), 900))
    if now_ts - last_ts < cooldown:
        return False
    max_count = max(0, _safe_int(os.getenv("QUICK_PROFIT_REDUCE_MAX_COUNT", 1), 1))
    current_count = max(0, _safe_int(active_trade.get("quick_reduce_count"), 0))
    if current_count >= max_count:
        return False

    earned_r = _safe_float(progress.get("earned_r_multiple"), 0.0)
    profit_progress = _safe_float(progress.get("profit_progress"), 0.0)
    daily_min_position = _is_daily_min_position()
    trigger_r = max(0.25, _safe_float(os.getenv("QUICK_PROFIT_REDUCE_TRIGGER_R", 0.65), 0.65))
    trigger_progress = max(0.15, _safe_float(os.getenv("QUICK_PROFIT_REDUCE_TP_PROGRESS", 0.35), 0.35))
    if daily_min_position:
        trigger_r = min(trigger_r, max(0.20, _safe_float(os.getenv("DAILY_MIN_QUICK_REDUCE_TRIGGER_R", 0.35), 0.35)))
        trigger_progress = min(trigger_progress, max(0.12, _safe_float(os.getenv("DAILY_MIN_QUICK_REDUCE_TP_PROGRESS", 0.22), 0.22)))
    if earned_r < trigger_r and profit_progress < trigger_progress:
        return False

    step = max(0.02, _safe_float(os.getenv("QUICK_PROFIT_REDUCE_STEP", 0.06), 0.06))
    delta = min(step, size - min_size)
    if delta <= 0:
        return False

    real_copy_enabled = _get_follow_mode_enabled() and _is_real_copy_enabled()
    if real_copy_enabled:
        min_delta_ratio = max(0.01, _safe_float(os.getenv("QUICK_PROFIT_REDUCE_MIN_DELTA_RATIO", 0.02), 0.02))
        if delta < min_delta_ratio:
            active_trade["quick_reduce_count"] = max_count
            active_trade["quick_reduce_ts"] = now_ts
            active_trade["last_adjust_ts"] = now_ts
            sync_position_panel(current_price)
            return False

    if real_copy_enabled:
        ok, scale_msg = _execute_copy_trade_scale(direction, delta, reduce=True, mark_price=current_price)
        if not ok:
            if "下單量低於最小值" in str(scale_msg):
                active_trade["quick_reduce_count"] = max_count
                active_trade["quick_reduce_ts"] = now_ts
                active_trade["last_adjust_ts"] = now_ts
                sync_position_panel(current_price)
                print(f"🔕 快速鎖利減倉略過但不推 Telegram: {scale_msg}")
                return False
            _notify_scale_skip(
                f"⚠️ 快速鎖利減倉略過：{scale_msg}",
                private=True,
                key=f"quick_reduce_order:{scale_msg}",
                now_ts=now_ts,
            )
            return False
        sync_active_trade_from_binance(send_notice=False)
    else:
        active_trade["size"] = max(min_size, size - delta)
        scale_msg = "虛擬減倉完成"

    active_trade["quick_reduce_count"] = current_count + 1
    active_trade["quick_reduce_ts"] = now_ts
    active_trade["last_adjust_ts"] = now_ts
    sync_position_panel(current_price)
    send_telegram(
        f"➖ 快速鎖利減倉\n"
        f"方向: {direction} | 現價: {current_price:.2f}\n"
        f"R={earned_r:.2f} | TP進度={profit_progress*100:.1f}%\n"
        f"減倉目標: -{delta*100:.1f}% | 目前倉位: {_safe_float(active_trade.get('size'), 0.0)*100:.1f}%\n"
        f"{scale_msg}",
        priority=True,
    )
    return True


def _assess_scaling_action(direction, entry, current_price, tp, sl, reduce=False, risk_cut=False):
    """回傳 (ok, reason, metrics)；在補倉/減倉前做成本合理性檢查。"""
    entry = _safe_float(entry, 0.0)
    px = _safe_float(current_price, 0.0)
    tp = _safe_float(tp, 0.0)
    sl = _safe_float(sl, 0.0)

    if direction not in {"long", "short"} or entry <= 0 or px <= 0 or tp <= 0 or sl <= 0:
        return False, "缺少有效方向或 TP/SL，略過調倉", {}

    total_cost = _estimate_trade_cost_rate_est()
    fee_round_trip_rate = max(0.0, _safe_float(POSITION_PANEL_STATE.get("fee_round_trip_rate"), 0.001))
    est_slippage_rate = max(0.0, _safe_float(os.getenv("TRADE_EST_SLIPPAGE_RATE", 0.0004), 0.0004))

    if direction == "long":
        reward_rate = max((tp - px) / px, 0.0)
        risk_rate = max((px - sl) / px, 1e-9)
        pnl_rate = (px - entry) / entry
    else:
        reward_rate = max((px - tp) / px, 0.0)
        risk_rate = max((sl - px) / px, 1e-9)
        pnl_rate = (entry - px) / entry

    rr_now = reward_rate / max(risk_rate, 1e-9)

    if not reduce:
        min_rr = max(1.05, _safe_float(os.getenv("SCALE_ADD_MIN_RR", 1.35), 1.35))
        min_net_edge = max(0.0003, _safe_float(os.getenv("SCALE_ADD_MIN_NET_EDGE_RATE", 0.0009), 0.0009))
        net_edge = reward_rate - total_cost

        metrics = {
            "reward_rate": reward_rate,
            "risk_rate": risk_rate,
            "rr": rr_now,
            "total_cost": total_cost,
            "net_edge": net_edge,
        }

        if rr_now < min_rr:
            return False, f"RR不足({rr_now:.2f} < {min_rr:.2f})", metrics
        if net_edge < min_net_edge:
            return False, f"報酬扣成本後不足({net_edge*100:.3f}% < {min_net_edge*100:.3f}%)", metrics
        return True, "補倉成本檢查通過", metrics

    # 減倉評估：止盈減倉要能鎖定淨利；結構失效時允許風險降倉。
    one_way_exit_cost = max(0.0, fee_round_trip_rate * 0.5 + est_slippage_rate * 0.5)
    lock_net_rate = pnl_rate - one_way_exit_cost
    min_lock_rate = max(0.0002, _safe_float(os.getenv("SCALE_REDUCE_MIN_LOCK_NET_RATE", 0.0008), 0.0008))
    max_risk_cut_loss = max(0.0005, _safe_float(os.getenv("SCALE_REDUCE_MAX_RISK_CUT_LOSS_RATE", 0.018), 0.018))

    metrics = {
        "pnl_rate": pnl_rate,
        "one_way_exit_cost": one_way_exit_cost,
        "lock_net_rate": lock_net_rate,
        "risk_cut": bool(risk_cut),
    }

    if risk_cut:
        if pnl_rate < -max_risk_cut_loss:
            return False, f"風險降倉浮虧過大({pnl_rate*100:.3f}% < -{max_risk_cut_loss*100:.3f}%)", metrics
        return True, "結構失效風險降倉通過", metrics
    if lock_net_rate < min_lock_rate:
        return False, f"可鎖定淨利不足({lock_net_rate*100:.3f}% < {min_lock_rate*100:.3f}%)", metrics
    return True, "減倉成本檢查通過", metrics


def _maybe_relax_sl_after_scale_add(direction, current_price, atr=None, initial_risk_abs=0.0):
    """補倉後小幅放寬 SL，但用風險上限避免無限制攤平。"""
    if not _is_truthy(os.getenv("SCALE_ADD_RELAX_SL_ENABLED", "1")):
        return False, ""

    direction = str(direction or "")
    entry = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), 0.0)
    old_sl = _safe_float(active_trade.get("sl"), 0.0)
    tp = _safe_float(active_trade.get("tp"), 0.0)
    px = _safe_float(current_price, 0.0)

    if direction not in {"long", "short"} or entry <= 0 or old_sl <= 0 or px <= 0:
        return False, ""

    atr_value = max(0.0, _safe_float(atr, _safe_float(SCALING_MARKET_STATE.get("atr"), 0.0)))
    risk_abs = max(_safe_float(initial_risk_abs, 0.0), abs(entry - old_sl), entry * 0.001)
    relax_atr_factor = max(0.0, _safe_float(os.getenv("SCALE_ADD_SL_RELAX_ATR_FACTOR", 0.35), 0.35))
    relax_risk_factor = max(0.0, _safe_float(os.getenv("SCALE_ADD_SL_RELAX_RISK_FACTOR", 0.12), 0.12))
    relax_abs = max(atr_value * relax_atr_factor, risk_abs * relax_risk_factor, entry * 0.0008)

    max_risk_multiplier = max(1.0, _safe_float(os.getenv("SCALE_ADD_SL_MAX_RISK_MULTIPLIER", 1.28), 1.28))
    max_distance_rate = max(0.002, _safe_float(os.getenv("SCALE_ADD_SL_MAX_DISTANCE_RATE", 0.045), 0.045))
    max_risk_abs = min(risk_abs * max_risk_multiplier, entry * max_distance_rate)
    if max_risk_abs <= abs(entry - old_sl) + 1e-9:
        return False, "SL 已達補倉放寬上限"

    if direction == "long":
        candidate = old_sl - relax_abs
        floor_sl = entry - max_risk_abs
        new_sl = max(candidate, floor_sl)
        if tp > 0 and new_sl >= min(entry, tp):
            return False, ""
        if new_sl >= old_sl - 1e-9:
            return False, "SL 已達補倉放寬上限"
    else:
        candidate = old_sl + relax_abs
        ceiling_sl = entry + max_risk_abs
        new_sl = min(candidate, ceiling_sl)
        if tp > 0 and new_sl <= max(entry, tp):
            return False, ""
        if new_sl <= old_sl + 1e-9:
            return False, "SL 已達補倉放寬上限"

    new_sl = round(float(new_sl), 2)
    active_trade["sl"] = float(new_sl)
    _sync_break_even_state_from_sl(direction, entry, new_sl, preserve_existing=False)

    sync_msg = ""
    if _get_follow_mode_enabled() and _is_real_copy_enabled():
        try:
            _, binance_msg = update_copy_trade_tp_sl(active_trade.get("tp"), new_sl)
            sync_msg = f"\n{binance_msg}"
        except Exception as e:
            sync_msg = f"\n⚠️ Binance SL 放寬同步失敗: {e}"

    return True, f"SL: {old_sl:.2f} → {new_sl:.2f}{sync_msg}"


def _assess_mlx_learned_scaling_logic(direction, progress, trend_score, opposing_pressure):
    """Apply learned live-session rules to scaling: no blind averaging down."""
    direction = str(direction or "")
    if direction not in {"long", "short"} or not isinstance(progress, dict):
        return {
            "add_ok": False,
            "reduce_ok": False,
            "risk_cut": False,
            "add_reason": "方向或進度無效",
            "reduce_reason": "方向或進度無效",
        }

    support_hits = max(0, _safe_int(SCALING_MARKET_STATE.get("support_hits"), 0))
    resistance_hits = max(0, _safe_int(SCALING_MARKET_STATE.get("resistance_hits"), 0))
    sr_bias = _safe_float(SCALING_MARKET_STATE.get("sr_bias"), 0.0)
    breakout = _safe_int(SCALING_MARKET_STATE.get("breakout"), 0)
    regime = str(SCALING_MARKET_STATE.get("regime") or "range")
    volume_ratio = max(0.0, _safe_float(SCALING_MARKET_STATE.get("volume_ratio"), 0.0))
    volume_spike = bool(SCALING_MARKET_STATE.get("volume_spike"))
    buy_pressure = bool(SCALING_MARKET_STATE.get("buy_pressure"))
    sell_pressure = bool(SCALING_MARKET_STATE.get("sell_pressure"))
    drawdown_progress = _safe_float(progress.get("drawdown_progress"), 0.0)
    profit_progress = _safe_float(progress.get("profit_progress"), 0.0)
    earned_r_multiple = _safe_float(progress.get("earned_r_multiple"), 0.0)

    if direction == "long":
        regime_aligned = regime not in {"bear_trend", "bear_trend_strong"}
        volume_confirmed = volume_ratio >= 1.05 and (volume_spike or buy_pressure or breakout == 1)
        direction_reconfirmed = (
            support_hits >= 1
            or sr_bias >= 0.16
            or breakout == 1
            or trend_score >= 1.4
        )
        structure_invalid = (
            breakout == -1
            or sr_bias <= -0.28
            or (opposing_pressure and trend_score < 1.0)
        )
        target_pressure = resistance_hits >= 1 or sr_bias <= -0.10
    else:
        regime_aligned = regime not in {"bull_trend", "bull_trend_strong"}
        volume_confirmed = volume_ratio >= 1.05 and (volume_spike or sell_pressure or breakout == -1)
        direction_reconfirmed = (
            resistance_hits >= 1
            or sr_bias <= -0.16
            or breakout == -1
            or trend_score >= 1.4
        )
        structure_invalid = (
            breakout == 1
            or sr_bias >= 0.28
            or (opposing_pressure and trend_score < 1.0)
        )
        target_pressure = support_hits >= 1 or sr_bias >= 0.10

    # Learned rule: add only when the original direction is valid again.
    add_ok = bool(direction_reconfirmed and volume_confirmed and regime_aligned and not structure_invalid)
    add_reason = "方向、量能與主要趨勢重新成立，允許補倉" if add_ok else "方向、量能或主要趨勢未重新成立，禁止盲目補倉"
    if structure_invalid:
        add_reason = "結構失效，禁止補倉"

    # Learned rule: reduce at target pressure, or cut risk when structure fails.
    risk_cut = bool(structure_invalid and drawdown_progress > 0.12)
    profit_reduce = bool(target_pressure and (profit_progress >= 0.28 or earned_r_multiple >= 0.65))
    reduce_ok = bool(risk_cut or profit_reduce)
    if risk_cut:
        reduce_reason = "結構失效，先降倉控風險"
    elif profit_reduce:
        reduce_reason = "到壓力/支撐目標區，先減倉鎖利"
    else:
        reduce_reason = "未到減倉條件"

    return {
        "add_ok": add_ok,
        "reduce_ok": reduce_ok,
        "risk_cut": risk_cut,
        "add_reason": add_reason,
        "reduce_reason": reduce_reason,
        "direction_reconfirmed": direction_reconfirmed,
        "structure_invalid": structure_invalid,
        "target_pressure": target_pressure,
        "support_hits": support_hits,
        "resistance_hits": resistance_hits,
        "sr_bias": sr_bias,
        "breakout": breakout,
        "volume_ratio": volume_ratio,
        "volume_confirmed": volume_confirmed,
        "regime_aligned": regime_aligned,
    }


def _notify_scale_skip(message, private=False, priority=True, key=None, now_ts=None):
    """Limit repeated scale skip alerts while preserving the first visible warning."""
    now_ts = _safe_float(now_ts, 0.0) or time.time()
    cooldown = max(300.0, _safe_float(os.getenv("SCALE_SKIP_NOTIFY_COOLDOWN_SEC", "3600"), 3600))
    key = str(key or message or "scale_skip")
    last_key = str(active_trade.get("last_scale_skip_notify_key") or "")
    last_ts = _safe_float(active_trade.get("last_scale_skip_notify_ts"), 0.0)
    if key == last_key and now_ts - last_ts < cooldown:
        print(f"🔕 調倉略過通知已限頻: {message}")
        return False

    active_trade["last_scale_skip_notify_key"] = key
    active_trade["last_scale_skip_notify_ts"] = now_ts
    if private:
        send_private_telegram(message, priority=priority)
    else:
        send_telegram(message, priority=priority)
    return True


def manage_position_scaling(current_price, atr=None, now_ts=None):
    """持倉中的補倉/減倉管理（虛擬倉位）。"""
    if not _is_truthy(os.getenv("TRADE_AUTO_SCALE_ENABLED", "0")):
        return
    if not active_trade.get("open"):
        return

    now_ts = _safe_float(now_ts, 0.0)
    if now_ts <= 0:
        now_ts = time.time()
    cooldown = max(90.0, _safe_float(os.getenv("SCALE_COOLDOWN_SEC", 240), 240))
    add_step = max(0.03, _safe_float(os.getenv("SCALE_ADD_STEP", 0.08), 0.08))
    reduce_step = max(0.03, _safe_float(os.getenv("SCALE_REDUCE_STEP", 0.08), 0.08))
    max_add_count = max(0, _safe_int(os.getenv("SCALE_MAX_ADD_COUNT", 2), 2))
    max_reduce_count = max(0, _safe_int(os.getenv("SCALE_MAX_REDUCE_COUNT", 2), 2))
    min_add_drawdown = max(0.05, _safe_float(os.getenv("SCALE_ADD_MIN_DRAWDOWN_PROGRESS", 0.22), 0.22))
    max_add_drawdown = max(min_add_drawdown, _safe_float(os.getenv("SCALE_ADD_MAX_DRAWDOWN_PROGRESS", 0.58), 0.58))
    min_reduce_progress = max(0.10, _safe_float(os.getenv("SCALE_REDUCE_MIN_PROFIT_PROGRESS", 0.42), 0.42))
    max_reduce_progress = max(min_reduce_progress, _safe_float(os.getenv("SCALE_REDUCE_MAX_PROFIT_PROGRESS", 0.82), 0.82))
    min_reduce_r_multiple = max(0.2, _safe_float(os.getenv("SCALE_REDUCE_MIN_R_MULTIPLE", 0.9), 0.9))
    min_add_trend_score = _safe_float(os.getenv("SCALE_ADD_MIN_TREND_SCORE", 1.8), 1.8)
    max_reduce_trend_score = _safe_float(os.getenv("SCALE_REDUCE_MAX_TREND_SCORE", 2.1), 2.1)
    min_move_rate = max(0.0008, _safe_float(os.getenv("SCALE_MIN_MOVE_RATE", 0.0025), 0.0025))
    min_move_atr_factor = max(0.0, _safe_float(os.getenv("SCALE_MIN_MOVE_ATR_FACTOR", 0.35), 0.35))

    last_adjust = _safe_float(active_trade.get("last_adjust_ts"), 0.0)
    if now_ts - last_adjust < cooldown:
        return

    direction = active_trade.get("direction")
    entry = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), current_price)
    size = max(0.0, _safe_float(active_trade.get("size"), 0.0))
    max_size, min_size = _resolve_scaling_bounds(
        size,
        active_trade.get("max_size"),
        active_trade.get("min_size"),
    )
    add_count = int(active_trade.get("add_count", 0))
    reduce_count = int(active_trade.get("reduce_count", 0))
    scale_add_paused = bool(active_trade.get("scale_add_paused", False))
    break_even_active = bool(active_trade.get("break_even_active", False))
    trade_source = str(active_trade.get("trade_source") or "")
    turn_direction = str(active_trade.get("candlestick_turn_direction") or "neutral")
    turn_count = max(0, _safe_int(active_trade.get("candlestick_turn_count"), 0))
    turn_confidence = max(0.0, _safe_float(active_trade.get("candlestick_turn_confidence"), 0.0))
    real_copy_enabled = _get_follow_mode_enabled() and _is_real_copy_enabled()
    progress = _calc_scaling_progress(direction, entry, current_price, active_trade.get("tp"), active_trade.get("sl"))

    if direction not in {"long", "short"} or not progress:
        return

    atr_value = max(_safe_float(atr, _safe_float(SCALING_MARKET_STATE.get("atr"), 0.0)), 0.0)
    min_move_abs = max(entry * min_move_rate, atr_value * min_move_atr_factor)
    if progress["move_abs"] < min_move_abs:
        return

    trend_score = _get_scaling_trend_score(direction)
    opposing_pressure = _has_scaling_opposing_pressure(direction)
    drawdown_progress = _safe_float(progress.get("drawdown_progress"), 0.0)
    profit_progress = _safe_float(progress.get("profit_progress"), 0.0)
    earned_r_multiple = _safe_float(progress.get("earned_r_multiple"), 0.0)
    learned_scaling = _assess_mlx_learned_scaling_logic(direction, progress, trend_score, opposing_pressure)
    daily_min_add_ok = bool(
        trade_source != "daily_minimum"
        or (
            turn_direction == direction
            and turn_count >= 2
            and turn_confidence >= 0.48
            and learned_scaling.get("volume_confirmed")
        )
    )

    add_trigger = (
        not break_even_active
        and not scale_add_paused
        and daily_min_add_ok
        and add_count < max_add_count
        and size < max_size - 1e-9
        and min_add_drawdown <= drawdown_progress <= max_add_drawdown
        and learned_scaling.get("add_ok")
    )

    reduce_trigger = (
        reduce_count < max_reduce_count
        and size > min_size + 1e-9
        and (
            (
                min_reduce_progress <= profit_progress <= max_reduce_progress
                and earned_r_multiple >= min_reduce_r_multiple
                and learned_scaling.get("reduce_ok")
            )
            or learned_scaling.get("risk_cut")
        )
    )

    # 補倉：逆勢回踩時逐步加碼（有上限）
    if add_trigger:
        delta = min(add_step, max_size - size)
        if delta > 0:
            ok_scale, scale_reason, scale_metrics = _assess_scaling_action(
                direction,
                entry,
                current_price,
                active_trade.get("tp"),
                active_trade.get("sl"),
                reduce=False,
            )
            if not ok_scale:
                detail = ""
                if isinstance(scale_metrics, dict) and scale_metrics:
                    detail = (
                        f" | reward={_safe_float(scale_metrics.get('reward_rate'), 0.0)*100:.3f}%"
                        f" | risk={_safe_float(scale_metrics.get('risk_rate'), 0.0)*100:.3f}%"
                        f" | RR={_safe_float(scale_metrics.get('rr'), 0.0):.2f}"
                )
                note = f"⚠️ 補倉略過（成本檢查）: {scale_reason}{detail}"
                _notify_scale_skip(
                    note,
                    private=real_copy_enabled,
                    key=f"add_cost:{scale_reason}",
                    now_ts=now_ts,
                )
                return

            if real_copy_enabled:
                ok, scale_msg = _execute_copy_trade_scale(direction, delta, reduce=False, mark_price=current_price)
                if not ok:
                    if active_trade.get("scale_add_paused"):
                        sync_position_panel(current_price)
                    _notify_scale_skip(
                        f"⚠️ 補倉略過：{scale_msg}",
                        private=True,
                        key=f"add_order:{scale_msg}",
                        now_ts=now_ts,
                    )
                    return
                sync_active_trade_from_binance(send_notice=False)
                active_trade["add_count"] = add_count + 1
                active_trade["last_adjust_ts"] = now_ts
                active_trade["max_size"] = max_size
                active_trade["min_size"] = min_size
                relaxed_sl, relax_msg = _maybe_relax_sl_after_scale_add(
                    direction,
                    current_price,
                    atr=atr_value,
                    initial_risk_abs=_safe_float(progress.get("initial_risk_abs"), 0.0),
                )
                if not relaxed_sl:
                    update_copy_trade_tp_sl(active_trade.get("tp"), active_trade.get("sl"))
                sync_position_panel(current_price)
                synced_size = _safe_float(active_trade.get("size"), size + delta)
                synced_entry = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), entry)
                sl_note = f"\n🛡️ 補倉後放寬止損: {relax_msg}" if relaxed_sl and relax_msg else ""
                send_telegram(
                    f"➕ 補倉（{direction}）\n"
                    f"現價: {current_price:.2f} | 目標加倉: +{int(delta*100)}%\n"
                    f"進場均價: {synced_entry:.2f} | TP: {_safe_float(active_trade.get('tp'), 0.0):.2f} | SL: {_safe_float(active_trade.get('sl'), 0.0):.2f}\n"
                    f"倉位(同步後): {int(size*100)}% → {int(synced_size*100)}%\n"
                    f"MLX調倉邏輯: {learned_scaling.get('add_reason')}\n"
                    f"{scale_msg}"
                    f"{sl_note}",
                    priority=True,
                )
                return

            new_size = size + delta
            # 均價更新（虛擬倉位）
            new_entry = ((entry * size) + (current_price * delta)) / max(new_size, 1e-9)
            active_trade["entry"] = float(new_entry)
            active_trade["avg_entry"] = float(new_entry)
            active_trade["size"] = float(new_size)
            active_trade["max_size"] = max_size
            active_trade["min_size"] = min_size
            active_trade["add_count"] = add_count + 1
            active_trade["last_adjust_ts"] = now_ts
            relaxed_sl, relax_msg = _maybe_relax_sl_after_scale_add(
                direction,
                current_price,
                atr=atr_value,
                initial_risk_abs=_safe_float(progress.get("initial_risk_abs"), 0.0),
            )
            tp_text = f"{_safe_float(active_trade.get('tp'), 0.0):.2f}" if active_trade.get("tp") is not None else "N/A"
            sl_text = f"{_safe_float(active_trade.get('sl'), 0.0):.2f}" if active_trade.get("sl") is not None else "N/A"
            sl_note = f"\n🛡️ 補倉後放寬止損: {relax_msg}" if relaxed_sl and relax_msg else ""
            sync_position_panel(current_price)
            send_telegram(
                f"➕ 補倉（{direction}）\n"
                f"現價: {current_price:.2f} | 加倉: +{int(delta*100)}%\n"
                f"進場均價: {new_entry:.2f} | TP: {tp_text} | SL: {sl_text}\n"
                f"倉位: {int(size*100)}% → {int(new_size*100)}%"
                f"\nMLX調倉邏輯: {learned_scaling.get('add_reason')}"
                f"{sl_note}",
                priority=True,
            )
            return

    # 減倉：有利方向浮盈時鎖定部分利潤（保留底倉）
    if reduce_trigger:
        delta = min(reduce_step, size - min_size)
        if delta > 0:
            ok_scale, scale_reason, scale_metrics = _assess_scaling_action(
                direction,
                entry,
                current_price,
                active_trade.get("tp"),
                active_trade.get("sl"),
                reduce=True,
                risk_cut=bool(learned_scaling.get("risk_cut")),
            )
            if not ok_scale:
                detail = ""
                if isinstance(scale_metrics, dict) and scale_metrics:
                    detail = (
                        f" | 浮盈={_safe_float(scale_metrics.get('pnl_rate'), 0.0)*100:.3f}%"
                        f" | 出場成本={_safe_float(scale_metrics.get('one_way_exit_cost'), 0.0)*100:.3f}%"
                        f" | 淨鎖利={_safe_float(scale_metrics.get('lock_net_rate'), 0.0)*100:.3f}%"
                )
                note = f"⚠️ 減倉略過（成本檢查）: {scale_reason}{detail}"
                _notify_scale_skip(
                    note,
                    private=real_copy_enabled,
                    key=f"reduce_cost:{scale_reason}",
                    now_ts=now_ts,
                )
                return

            if real_copy_enabled:
                ok, scale_msg = _execute_copy_trade_scale(direction, delta, reduce=True, mark_price=current_price)
                if not ok:
                    if "下單量低於最小值" in str(scale_msg):
                        active_trade["reduce_count"] = max_reduce_count
                        active_trade["last_adjust_ts"] = now_ts
                        sync_position_panel(current_price)
                        print(f"🔕 減倉略過但不推 Telegram: {scale_msg}")
                        return
                    _notify_scale_skip(
                        f"⚠️ 減倉略過：{scale_msg}",
                        private=True,
                        key=f"reduce_order:{scale_msg}",
                        now_ts=now_ts,
                    )
                    return
                sync_active_trade_from_binance(send_notice=False)
                active_trade["reduce_count"] = reduce_count + 1
                active_trade["last_adjust_ts"] = now_ts
                active_trade["max_size"] = max_size
                active_trade["min_size"] = min_size
                update_copy_trade_tp_sl(active_trade.get("tp"), active_trade.get("sl"))
                sync_position_panel(current_price)
                synced_size = _safe_float(active_trade.get("size"), max(min_size, size - delta))
                synced_entry = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), entry)
                send_telegram(
                    f"➖ 減倉（{direction}）\n"
                    f"現價: {current_price:.2f} | 目標減倉: -{int(delta*100)}%\n"
                    f"進場均價: {synced_entry:.2f} | TP: {_safe_float(active_trade.get('tp'), 0.0):.2f} | SL: {_safe_float(active_trade.get('sl'), 0.0):.2f}\n"
                    f"倉位(同步後): {int(size*100)}% → {int(synced_size*100)}%\n"
                    f"MLX調倉邏輯: {learned_scaling.get('reduce_reason')}\n"
                    f"{scale_msg}",
                    priority=True,
                )
                return

            new_size = size - delta
            tp_text = f"{_safe_float(active_trade.get('tp'), 0.0):.2f}" if active_trade.get("tp") is not None else "N/A"
            sl_text = f"{_safe_float(active_trade.get('sl'), 0.0):.2f}" if active_trade.get("sl") is not None else "N/A"
            active_trade["size"] = float(new_size)
            active_trade["max_size"] = max_size
            active_trade["min_size"] = min_size
            active_trade["reduce_count"] = int(active_trade.get("reduce_count", 0)) + 1
            active_trade["last_adjust_ts"] = now_ts
            sync_position_panel(current_price)
            send_telegram(
                f"➖ 減倉（{direction}）\n"
                f"現價: {current_price:.2f} | 減倉: -{int(delta*100)}%\n"
                f"進場均價: {entry:.2f} | TP: {tp_text} | SL: {sl_text}\n"
                f"倉位: {int(size*100)}% → {int(new_size*100)}%\n"
                f"MLX調倉邏輯: {learned_scaling.get('reduce_reason')}",
                priority=True,
            )


def maybe_shrink_tp_after_hold(current_price=None, now_ts=None):
    if not active_trade.get("open"):
        return False
    if active_trade.get("tp_sl_adjusted_4h", False):
        return False

    open_ts = _safe_float(active_trade.get("open_time"), 0.0)
    if open_ts <= 0:
        return False

    resolved_now_ts = _safe_float(now_ts, 0.0)
    if resolved_now_ts <= 0:
        resolved_now_ts = time.time()
    if (resolved_now_ts - open_ts) < 4 * 3600:
        return False

    entry_ref = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), 0.0)
    old_tp = _safe_float(active_trade.get("tp"), 0.0)
    current_price = _safe_float(current_price, 0.0)

    if old_tp <= 0 or entry_ref <= 0:
        active_trade["tp_sl_adjusted_4h"] = True
        return False

    direction = str(active_trade.get("direction") or "")
    if direction == "long":
        new_tp = entry_ref + (old_tp - entry_ref) * 0.6
    elif direction == "short":
        new_tp = entry_ref - (entry_ref - old_tp) * 0.6
    else:
        return False

    active_trade["tp"] = round(new_tp, 2)
    active_trade["tp_sl_adjusted_4h"] = True
    sync_position_panel(current_price or entry_ref)

    sync_msg = ""
    if _get_follow_mode_enabled() and _is_real_copy_enabled():
        try:
            _, binance_msg = update_copy_trade_tp_sl(active_trade.get("tp"), active_trade.get("sl"))
            sync_msg = f"\n{binance_msg}"
        except Exception as e:
            sync_msg = f"\n⚠️ Binance TP 同步失敗: {e}"

    if not _is_truthy(os.getenv("ETH_BOT_DISABLE_LIVE", "0")):
        print(f"⏱️ 持倉已超4小時，只縮TP | 新TP: {active_trade['tp']:.2f} | SL維持: {active_trade['sl']:.2f}")
    send_telegram(
        f"⏱️ 持倉已超4小時，只縮減止盈\n"
        f"方向: {active_trade['direction']} | 進場均價: {entry_ref:.2f}\n"
        f"TP: {old_tp:.2f} → {active_trade['tp']:.2f}\n"
        f"SL 維持: {active_trade['sl']:.2f}"
        f"{sync_msg}",
        priority=True,
    )
    return True


def get_signal_direction(signal):
    s = str(signal or "")
    if "做多" in s:
        return "long"
    if "做空" in s:
        return "short"
    return None


def _daily_anchor_guard_should_wait(final, score, decision=None):
    if str(final or "").startswith("觀望"):
        return False

    direction = get_signal_direction(final)
    if direction not in {"long", "short"}:
        return False

    decision = decision if isinstance(decision, dict) else {}
    market_profile = decision.get("market_profile") if isinstance(decision.get("market_profile"), dict) else {}
    market_phase = str(market_profile.get("phase") or "range_base")
    risk_rate = _safe_float(decision.get("risk_rate"), 0.0)
    host_logic = decision.get("host_opening_logic") if isinstance(decision.get("host_opening_logic"), dict) else {}
    host_mode = str(host_logic.get("mode") or "")

    if market_phase == "bull":
        return False
    if market_phase == "bear":
        max_bear_short_risk = max(
            0.003,
            _safe_float(os.getenv("DAILY_MIN_ANCHOR_BEAR_SHORT_MAX_RISK_RATE", 0.015), 0.015),
        )
        news_bias = _safe_float(decision.get("news_bias"), 0.0)
        event_risk = _safe_int(decision.get("event_risk"), 0)
        if (
            direction == "short"
            and host_mode == "breakdown_after_support_tests"
            and 0 < risk_rate <= max_bear_short_risk
            and news_bias <= 0
            and event_risk <= 0
        ):
            return False
        support_hits = _safe_int(decision.get("support_hits"), 0)
        max_tested_bear_short_risk = max(
            max_bear_short_risk,
            _safe_float(os.getenv("DAILY_MIN_ANCHOR_BEAR_TESTED_SHORT_MAX_RISK_RATE", 0.025), 0.025),
        )
        if (
            direction == "short"
            and host_mode == "breakdown_after_support_tests"
            and support_hits >= 1
            and 0 < risk_rate <= max_tested_bear_short_risk
            and news_bias <= 0
            and event_risk <= 0
        ):
            return False
        return True

    if market_phase == "range_base" and risk_rate > 0:
        if direction == "long" and host_mode == "support_reclaim":
            return True
        max_range_risk = max(
            0.002,
            _safe_float(os.getenv("DAILY_MIN_ANCHOR_RANGE_MAX_RISK_RATE", 0.008), 0.008),
        )
        if risk_rate > max_range_risk:
            return True
    if market_phase == "bull_high_vol" and risk_rate > 0:
        max_risk = max(
            0.003,
            _safe_float(os.getenv("DAILY_MIN_ANCHOR_BULL_HIGH_VOL_MAX_RISK_RATE", 0.012), 0.012),
        )
        if risk_rate > max_risk:
            return True

    score_value = _safe_float(score, 0.5)
    directional_gap = (score_value - 0.5) if direction == "long" else (0.5 - score_value)
    min_score_gap = max(0.08, _safe_float(os.getenv("DAILY_MIN_ANCHOR_ALLOW_SCORE_GAP", 0.12), 0.12))
    min_edge = max(0.0002, _safe_float(os.getenv("DAILY_MIN_ANCHOR_ALLOW_NET_EDGE_RATE", 0.0015), 0.0015))
    min_host_conf = max(0.30, _safe_float(os.getenv("DAILY_MIN_ANCHOR_ALLOW_HOST_CONF", 0.52), 0.52))

    net_edge = _safe_float(decision.get("net_edge_rate_est"), 0.0)
    host_direction = str(host_logic.get("direction") or "neutral")
    host_conf = _safe_float(host_logic.get("confidence"), 0.0)
    host_applied = bool(decision.get("host_logic_applied", False))
    repeated_support = _safe_int(decision.get("repeated_support_tests"), 0)
    repeated_resistance = _safe_int(decision.get("repeated_resistance_tests"), 0)
    volume_spike = bool(decision.get("volume_spike"))
    buy_pressure = bool(decision.get("buy_pressure"))
    sell_pressure = bool(decision.get("sell_pressure"))
    profile_adjustment = decision.get("market_profile_adjustment") if isinstance(decision.get("market_profile_adjustment"), dict) else {}
    profile_edge = _safe_float(profile_adjustment.get("adjustment"), 0.0)

    high_score = directional_gap >= min_score_gap
    positive_edge = net_edge >= min_edge
    host_confirms = host_applied and host_direction == direction and host_conf >= min_host_conf
    pressure_break = (
        direction == "long"
        and repeated_resistance >= 2
        or direction == "short"
        and repeated_support >= 2
    )
    flow_confirms = volume_spike and ((direction == "long" and buy_pressure) or (direction == "short" and sell_pressure))
    profile_confirms = profile_edge >= 0.05

    if high_score and (positive_edge or host_confirms or pressure_break or flow_confirms or profile_confirms):
        return False

    return market_phase != "bull" or (direction == "long" and market_phase in {"bear", "bull_high_vol"})


def _same_entry_confirm_candle(prev_candle_id, candle_id):
    if prev_candle_id is None or candle_id is None:
        return False
    return str(prev_candle_id) == str(candle_id)


def _get_entry_confirm_candle_id(df_5m):
    try:
        if df_5m is None or len(df_5m) == 0:
            return None
        if "time" in df_5m.columns:
            candle_time = df_5m["time"].iloc[-1]
            if candle_time is not None:
                return int(float(candle_time))
        idx = df_5m.index[-1]
        if hasattr(idx, "timestamp"):
            return int(idx.timestamp())
        return str(idx)
    except Exception:
        return None


def _evaluate_pending_entry_confirmation(pending, direction, price, score, candle_id, now_ts):
    if not pending:
        return False, "建立待確認訊號"

    if direction != pending.get("direction"):
        return False, "方向改變，重置待確認訊號"

    pending_price = _safe_float(pending.get("price"), 0.0)
    if pending_price <= 0 or price <= 0:
        return False, "價格資料不足"

    max_age = max(60.0, _safe_float(os.getenv("TRADE_ENTRY_CONFIRM_MAX_AGE_SEC", 420), 420))
    min_wait = max(0.0, _safe_float(os.getenv("TRADE_ENTRY_CONFIRM_MIN_WAIT_SEC", 45), 45))
    max_chase = max(0.0, _safe_float(os.getenv("TRADE_ENTRY_CONFIRM_MAX_CHASE_RATE", 0.0018), 0.0018))
    max_reversal = max(0.0, _safe_float(os.getenv("TRADE_ENTRY_CONFIRM_MAX_REVERSAL_RATE", 0.0025), 0.0025))
    require_new_candle = str(os.getenv("TRADE_ENTRY_CONFIRM_REQUIRE_NEW_5M", "1") or "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    age = max(0.0, now_ts - _safe_float(pending.get("ts"), now_ts))
    if age > max_age:
        return False, "待確認訊號逾時"
    if age < min_wait:
        return False, f"等待確認中 {age:.0f}/{min_wait:.0f}s"
    if require_new_candle and _same_entry_confirm_candle(pending.get("candle_id"), candle_id):
        return False, "等待下一根 5m K 確認"

    move_rate = (price - pending_price) / pending_price
    if direction == "long":
        if move_rate > max_chase:
            return False, f"多單已追高 {move_rate*100:.2f}%"
        if move_rate < -max_reversal:
            return False, f"多單回落過深 {move_rate*100:.2f}%"
        if score <= 0.5:
            return False, "多單分數轉弱"
    else:
        if move_rate < -max_chase:
            return False, f"空單已追低 {abs(move_rate)*100:.2f}%"
        if move_rate > max_reversal:
            return False, f"空單反彈過深 {move_rate*100:.2f}%"
        if score >= 0.5:
            return False, "空單分數轉弱"

    return True, f"延遲確認通過 {age:.0f}s | 價格變化 {move_rate*100:+.2f}%"


def auto_fix_trade_plan(signal, entry, sl, tp, atr):
    """最終開單前修正 TP/SL，避免方向錯誤或風險距離過小。"""
    direction = get_signal_direction(signal)
    if direction is None:
        return signal, sl, tp

    entry = _safe_float(entry, 0.0)
    sl = _safe_float(sl, entry)
    tp = _safe_float(tp, entry)
    atr = max(_safe_float(atr, 0.0), 0.0)

    # 最小風險距離：避免 SL/TP 太近造成雜訊掃損
    min_risk = max(entry * 0.0008, atr * 0.35, 0.3)

    if direction == "long":
        if sl >= entry - min_risk:
            sl = entry - min_risk
        risk = max(entry - sl, min_risk)
        min_tp = entry + max(risk * 1.4, min_risk * 1.2)
        if tp <= min_tp:
            tp = min_tp
    else:
        if sl <= entry + min_risk:
            sl = entry + min_risk
        risk = max(sl - entry, min_risk)
        min_tp = entry - max(risk * 1.4, min_risk * 1.2)
        if tp <= 0 or tp >= min_tp:
            tp = min_tp

    return signal, float(sl), float(tp)


def get_macro_bias():
    global MACRO_CACHE

    now = time.time()

    # 🔥 低延遲模式（接近金十）
    if now - MACRO_CACHE["ts"] < 3:
        return MACRO_CACHE["sp"], MACRO_CACHE["nq"], MACRO_CACHE["btc"], MACRO_CACHE["dxy"], MACRO_CACHE.get("news", 0), MACRO_CACHE.get("event", 0), MACRO_CACHE.get("news_list", [])

    # ===== SP500 =====
    try:
        sp_url = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=ES=F"
        sp_data = HTTP_SESSION.get(sp_url, timeout=3).json()
        sp_price = float(sp_data["quoteResponse"]["result"][0]["regularMarketPrice"])
        sp_prev = float(sp_data["quoteResponse"]["result"][0]["regularMarketPreviousClose"])
        sp_change = (sp_price - sp_prev) / sp_prev
    except:
        sp_change = 0

    # ===== NASDAQ (NQ) =====
    try:
        nq_url = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=NQ=F"
        nq_data = HTTP_SESSION.get(nq_url, timeout=3).json()
        nq_price = float(nq_data["quoteResponse"]["result"][0]["regularMarketPrice"])
        nq_prev = float(nq_data["quoteResponse"]["result"][0]["regularMarketPreviousClose"])
        nq_change = (nq_price - nq_prev) / nq_prev
    except:
        nq_change = 0

    # ===== BTC（相對性核心）=====
    try:
        btc_rows, _ = _fetch_market_kline_rows("BTCUSDT", "1h", limit=25, timeout=5, prefix="BTC宏觀K線")
        btc_open = _safe_float(btc_rows[0][1], 0.0) if btc_rows else 0.0
        btc_close = _safe_float(btc_rows[-1][4], 0.0) if btc_rows else 0.0
        btc_change = (btc_close - btc_open) / btc_open if btc_open > 0 and btc_close > 0 else 0
    except:
        btc_change = 0

    # ===== DXY =====
    try:
        dxy_url = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=DX-Y.NYB"
        dxy_data = HTTP_SESSION.get(dxy_url, timeout=3).json()
        dxy_price = float(dxy_data["quoteResponse"]["result"][0]["regularMarketPrice"])
        dxy_prev = float(dxy_data["quoteResponse"]["result"][0]["regularMarketPreviousClose"])
        dxy_change = (dxy_price - dxy_prev) / dxy_prev
    except:
        dxy_change = 0

    # ===== 即時新聞聚合（僅 RSS，已移除 Binance / Jin10） =====
    news_bias, event_risk, news_list = refresh_rss_news_cache(force=False)

    MACRO_CACHE = {"sp": sp_change, "nq": nq_change, "btc": btc_change, "dxy": dxy_change, "news": news_bias, "event": event_risk, "news_list": news_list, "ts": now}
    return sp_change, nq_change, btc_change, dxy_change, news_bias, event_risk, news_list

# ===== Online Model Persistence =====
ONLINE_MODEL_PATH = ai_data_path("online_model.pkl")

# =============================
# 全域
# =============================
WS_PRICE = None
WS_PRICE_TS = 0.0

# ===== 勝率統計 =====
performance = {
    "total": 0,
    "win": 0,
    "loss": 0
}

# ===== 交易狀態（真實交易管理） =====
active_trade = {
    "direction": None,
    "entry": None,
    "avg_entry": None,
    "tp": None,
    "sl": None,
    "break_even_active": False,
    "break_even_target": 0.0,
    "break_even_ts": 0.0,
    "open": False,
    "size": 0.0,
    "position_qty": 0.0,
    "max_size": 1.0,
    "min_size": 0.15,
    "add_count": 0,
    "reduce_count": 0,
    "quick_reduce_count": 0,
    "quick_reduce_ts": 0.0,
    "daily_min_size_enforce_ts": 0.0,
    "last_adjust_ts": 0.0,
    "scale_add_paused": False,
    "scale_add_pause_reason": "",
    "scale_add_pause_ts": 0.0,
    "last_scale_skip_notify_key": "",
    "last_scale_skip_notify_ts": 0.0,
    "open_time": None,
    "tp_sl_adjusted_4h": False,
    "time_horizon": "short",
    "trade_source": "",
    "candlestick_turn_direction": "neutral",
    "candlestick_turn_count": 0,
    "candlestick_turn_confidence": 0.0,
}


def _reset_active_trade_state():
    """清除舊持倉狀態，避免重啟後殘留舊單影響重新開倉。"""
    active_trade["direction"] = None
    active_trade["entry"] = None
    active_trade["avg_entry"] = None
    active_trade["tp"] = None
    active_trade["sl"] = None
    active_trade["break_even_active"] = False
    active_trade["break_even_target"] = 0.0
    active_trade["break_even_ts"] = 0.0
    active_trade["open"] = False
    active_trade["size"] = 0.0
    active_trade["position_qty"] = 0.0
    active_trade["trade_source"] = ""
    active_trade["candlestick_turn_direction"] = "neutral"
    active_trade["candlestick_turn_count"] = 0
    active_trade["candlestick_turn_confidence"] = 0.0
    active_trade["add_count"] = 0
    active_trade["reduce_count"] = 0
    active_trade["quick_reduce_count"] = 0
    active_trade["quick_reduce_ts"] = 0.0
    active_trade["daily_min_size_enforce_ts"] = 0.0
    active_trade["last_adjust_ts"] = 0.0
    active_trade["scale_add_paused"] = False
    active_trade["scale_add_pause_reason"] = ""
    active_trade["scale_add_pause_ts"] = 0.0
    active_trade["last_scale_skip_notify_key"] = ""
    active_trade["last_scale_skip_notify_ts"] = 0.0
    active_trade["open_time"] = None
    active_trade["tp_sl_adjusted_4h"] = False
    active_trade["time_horizon"] = "short"
    _clear_pending_training_sample_state()


def restore_active_trade_from_panel():
    if not POSITION_PANEL_FILE.exists():
        return

    try:
        raw = json.loads(POSITION_PANEL_FILE.read_text(encoding="utf-8"))
    except Exception:
        return

    if not isinstance(raw, dict) or not raw.get("open"):
        return

    entry = _safe_float(raw.get("entry"), 0.0)
    tp = _safe_float(raw.get("tp"), 0.0)
    sl = _safe_float(raw.get("sl"), 0.0)
    size = max(0.0, _safe_float(raw.get("size_ratio", raw.get("size")), 0.0))
    position_qty = max(0.0, _safe_float(raw.get("binance_qty"), 0.0))
    direction = str(raw.get("direction") or "")

    if direction not in {"long", "short"} or entry <= 0 or tp <= 0 or sl <= 0 or size <= 0:
        return

    restored_horizon = _normalize_trade_time_horizon(raw.get("time_horizon"))
    restore_max_age_sec = max(48 * 3600, _trade_max_hold_sec(restored_horizon) + 24 * 3600)
    state_ts = _safe_int(raw.get("ts"), 0)
    if state_ts > 0 and (time.time() - state_ts) > restore_max_age_sec:
        return

    active_trade["direction"] = direction
    active_trade["entry"] = entry
    active_trade["avg_entry"] = entry
    active_trade["tp"] = tp
    active_trade["sl"] = sl
    active_trade["open"] = True
    active_trade["size"] = min(1.0, size)
    active_trade["position_qty"] = position_qty
    max_size, min_size = _resolve_scaling_bounds(
        active_trade["size"],
        raw.get("max_size"),
        raw.get("min_size"),
    )
    active_trade["max_size"] = max_size
    active_trade["min_size"] = min_size
    active_trade["add_count"] = max(0, _safe_int(raw.get("add_count"), 0))
    active_trade["reduce_count"] = max(0, _safe_int(raw.get("reduce_count"), 0))
    active_trade["quick_reduce_count"] = max(0, _safe_int(raw.get("quick_reduce_count"), 0))
    active_trade["quick_reduce_ts"] = _safe_float(raw.get("quick_reduce_ts"), 0.0)
    active_trade["daily_min_size_enforce_ts"] = _safe_float(raw.get("daily_min_size_enforce_ts"), 0.0)
    active_trade["last_adjust_ts"] = _safe_float(raw.get("last_adjust_ts"), 0.0)
    active_trade["scale_add_paused"] = bool(raw.get("scale_add_paused", False))
    active_trade["scale_add_pause_reason"] = str(raw.get("scale_add_pause_reason") or "")
    active_trade["scale_add_pause_ts"] = _safe_float(raw.get("scale_add_pause_ts"), 0.0)
    active_trade["open_time"] = _safe_float(raw.get("open_since_ts"), state_ts or time.time())
    active_trade["tp_sl_adjusted_4h"] = bool(raw.get("tp_sl_adjusted_4h", False))
    active_trade["time_horizon"] = restored_horizon
    stored_be_active = bool(raw.get("break_even_active", False))
    stored_be_target = _safe_float(raw.get("break_even_target"), 0.0)
    stored_be_ts = _safe_float(raw.get("break_even_ts"), 0.0)
    if stored_be_active or _is_break_even_or_better(direction, entry, sl):
        _set_break_even_state(True, target=(stored_be_target if stored_be_target > 0 else sl), ts=(stored_be_ts if stored_be_ts > 0 else (state_ts or time.time())))
    else:
        _set_break_even_state(False)
    print(
        f"♻️ 已從 position.json 還原持倉 | 方向: {direction} | 進場: {entry:.2f} | "
        f"TP: {tp:.2f} | SL: {sl:.2f} | 倉位: {size:.2%}"
    )


if LIVE_RUNTIME_ENABLED:
    restore_active_trade_from_panel()

    # 重啟後預設清除舊持倉，避免「重新開倉」時仍卡在舊單狀態。
    if str(os.getenv("CLEAR_OLD_TRADE_ON_START", "1")).strip() == "1" and active_trade.get("open"):
        if _read_pending_training_sample_state_raw():
            print("♻️ 偵測到待學習中的舊持倉，保留持倉狀態以延續 TP/SL 學習")
        else:
            print("♻️ 偵測到重啟後舊持倉，已清除舊單狀態以允許重新開倉")
            _reset_active_trade_state()
            sync_position_panel(WS_PRICE)


# =============================
# KLINE CACHE（避免打爆API）
# =============================
KLINE_CACHE = {}
KLINE_TTL = {
    "1M": 60*60*12,
    "1w": 60*60*6,
    "1d": 60*60,
    "12h": 60*30,
    "4h": 60*60,
    "1h": 60*10,
    "30m": 60*5,
    "15m": 60*3,
    "5m": 60*2,
    "1m": 10
}
TRADINGVIEW_FAILURE_COOLDOWN = {}
BINANCE_RATE_LIMIT_STATE = {"until": 0.0, "status": 0, "reason": ""}
BINANCE_RATE_LIMIT_LOCK = threading.Lock()


def _parse_retry_after_seconds(value, default_sec):
    text = str(value or "").strip()
    if not text:
        return float(default_sec)
    try:
        return max(1.0, float(text))
    except Exception:
        return float(default_sec)


def _binance_rate_limit_remaining_sec():
    with BINANCE_RATE_LIMIT_LOCK:
        until = _safe_float(BINANCE_RATE_LIMIT_STATE.get("until"), 0.0)
        return max(0.0, until - time.time())


def _raise_if_binance_rate_limited(prefix="Binance"):
    remain = _binance_rate_limit_remaining_sec()
    if remain > 0:
        raise RuntimeError(f"{prefix} rate-limit cooldown {remain:.0f}s")


def _note_binance_rate_limit_response(response, prefix="Binance"):
    status = _safe_int(getattr(response, "status_code", 0), 0)
    if status not in {418, 429}:
        return
    default_sec = 900 if status == 418 else 60
    retry_sec = _parse_retry_after_seconds(getattr(response, "headers", {}).get("Retry-After"), default_sec)
    retry_sec = max(float(default_sec), retry_sec) if status == 418 else max(10.0, retry_sec)
    until = time.time() + retry_sec
    with BINANCE_RATE_LIMIT_LOCK:
        BINANCE_RATE_LIMIT_STATE.update(
            {
                "until": until,
                "status": status,
                "reason": str(prefix),
            }
        )
    print(f"⚠️ {prefix} 觸發 {status}，暫停 Binance 請求 {int(retry_sec)} 秒，避免升級封鎖")


def _binance_request_get(url, *, params=None, headers=None, timeout=10, prefix="Binance"):
    _raise_if_binance_rate_limited(prefix)
    response = HTTP_SESSION.get(url, params=params, headers=headers, timeout=timeout)
    _note_binance_rate_limit_response(response, prefix=prefix)
    return response


def _binance_request_post(url, *, params=None, headers=None, timeout=10, prefix="Binance"):
    _raise_if_binance_rate_limited(prefix)
    response = HTTP_SESSION.post(url, params=params, headers=headers, timeout=timeout)
    _note_binance_rate_limit_response(response, prefix=prefix)
    return response


def _binance_request_delete(url, *, params=None, headers=None, timeout=10, prefix="Binance"):
    _raise_if_binance_rate_limited(prefix)
    response = HTTP_SESSION.delete(url, params=params, headers=headers, timeout=timeout)
    _note_binance_rate_limit_response(response, prefix=prefix)
    return response


def _tradingview_cooldown_key(symbol, interval):
    return f"{str(symbol or '').upper()}:{str(interval)}"


def _is_tradingview_in_cooldown(symbol, interval):
    key = _tradingview_cooldown_key(symbol, interval)
    until = _safe_float(TRADINGVIEW_FAILURE_COOLDOWN.get(key), 0.0)
    return until > time.time()


def _mark_tradingview_failure(symbol, interval):
    cooldown = max(15.0, _safe_float(os.getenv("TRADINGVIEW_FAILURE_COOLDOWN_SEC", 90), 90))
    TRADINGVIEW_FAILURE_COOLDOWN[_tradingview_cooldown_key(symbol, interval)] = time.time() + cooldown

# =============================
# WebSocket（tick級）
# =============================
def ws_price_stream():
    def on_message(ws, msg):
        global WS_PRICE, WS_PRICE_TS
        data = json.loads(msg)
        WS_PRICE = float(data["p"])
        WS_PRICE_TS = time.time()

    ws = websocket.WebSocketApp(
        "wss://fstream.binance.com/ws/ethusdt@aggTrade",
        on_message=on_message
    )

    while True:
        try:
            ws.run_forever()
        except:
            time.sleep(2)

if LIVE_RUNTIME_ENABLED:
    threading.Thread(target=ws_price_stream, daemon=True).start()


def _validated_market_price(kline_price, reference_price=None, now_ts=None):
    """優先用即時成交價核對；不可用時至少核對 1m/5m 價格一致性。"""
    now_ts = _safe_float(now_ts, time.time())
    kline_price = max(0.0, _safe_float(kline_price, 0.0))
    ws_price = max(0.0, _safe_float(WS_PRICE, 0.0))
    ws_age = max(0.0, now_ts - _safe_float(WS_PRICE_TS, 0.0))
    max_age = max(5.0, _safe_float(os.getenv("MARKET_PRICE_VALIDATION_MAX_AGE_SEC", 30), 30))
    max_deviation = max(0.0005, _safe_float(os.getenv("MARKET_PRICE_VALIDATION_MAX_DEVIATION_RATE", 0.003), 0.003))
    reference_price = max(0.0, _safe_float(reference_price, 0.0))
    validation_price = ws_price if ws_price > 0 and ws_age <= max_age else reference_price
    validation_source = "WebSocket" if validation_price == ws_price and ws_price > 0 else "TradingView-5m"
    if kline_price <= 0 or validation_price <= 0:
        raise RuntimeError("行情交叉驗證缺少可用參考價格")
    deviation = abs(validation_price - kline_price) / max(validation_price, kline_price, 1e-9)
    if deviation > max_deviation:
        raise RuntimeError(
            f"行情交叉驗證價差過大 TradingView-1m={kline_price:.2f} "
            f"{validation_source}={validation_price:.2f} "
            f"deviation={deviation*100:.3f}%"
        )
    return ws_price if ws_price > 0 and ws_age <= max_age else kline_price

# =============================
# Indicators
# =============================
def calc_indicators(df):
    df["ma25"] = df["close"].rolling(25).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["signal"] = df["macd"].ewm(span=9).mean()
    delta = df["close"].diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi14"] = (100 - 100 / (1 + relative_strength)).fillna(50.0)

    # ===== Volume v2 =====
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    # VWAP（簡化版：以收盤加權）
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-9)

    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    atr_ref = df["atr14"].replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_ref
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_ref
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    df["adx14"] = dx.ewm(alpha=1 / 14, adjust=False).mean().fillna(0.0)
    df["plus_di14"] = plus_di.fillna(0.0)
    df["minus_di14"] = minus_di.fillna(0.0)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_mid + bb_std * 2
    df["bb_lower"] = bb_mid - bb_std * 2
    df["bb_pos"] = ((close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)).clip(0, 1).fillna(0.5)

    hl2 = (high + low) / 2
    st_mult = _safe_float(os.getenv("SUPERTREND_MULTIPLIER", 3.0), 3.0)
    upper_basic = hl2 + st_mult * df["atr14"]
    lower_basic = hl2 - st_mult * df["atr14"]
    final_upper = upper_basic.copy()
    final_lower = lower_basic.copy()
    supertrend = pd.Series(index=df.index, dtype=float)
    supertrend_dir = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i == 0:
            supertrend.iloc[i] = upper_basic.iloc[i]
            supertrend_dir.iloc[i] = 1.0
            continue
        if upper_basic.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = upper_basic.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]
        if lower_basic.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = lower_basic.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]
        if supertrend.iloc[i - 1] == final_upper.iloc[i - 1]:
            supertrend.iloc[i] = final_lower.iloc[i] if close.iloc[i] > final_upper.iloc[i] else final_upper.iloc[i]
        else:
            supertrend.iloc[i] = final_upper.iloc[i] if close.iloc[i] < final_lower.iloc[i] else final_lower.iloc[i]
        supertrend_dir.iloc[i] = 1.0 if close.iloc[i] >= supertrend.iloc[i] else -1.0
    df["supertrend"] = supertrend.ffill()
    df["supertrend_dir"] = supertrend_dir.fillna(0.0)

    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    df["ichimoku_tenkan"] = tenkan
    df["ichimoku_kijun"] = kijun
    df["ichimoku_span_a"] = span_a
    df["ichimoku_span_b"] = span_b
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)
    df["ichimoku_bias"] = np.where(close > cloud_top, 1, np.where(close < cloud_bottom, -1, 0))

    return df


def get_td_sequential_setup(df, completed_only=True):
    """Return the current TD Setup count using close vs. the close four bars earlier.

    Only completed candles are used by default.  This keeps a live, unfinished
    candle from repeatedly enabling/disabling the entry filter.
    """
    if df is None or "close" not in df.columns:
        return {
            "up_count": 0,
            "down_count": 0,
            "up_9": False,
            "down_9": False,
            "up_sequence_start": None,
            "down_sequence_start": None,
        }

    closes = df["close"].dropna()
    if completed_only and len(closes) > 0:
        closes = closes.iloc[:-1]
    if len(closes) < 13:
        return {
            "up_count": 0,
            "down_count": 0,
            "up_9": False,
            "down_9": False,
            "up_sequence_start": None,
            "down_sequence_start": None,
        }

    up_count = 0
    down_count = 0
    up_sequence_start = None
    down_sequence_start = None
    for index in range(4, len(closes)):
        close = _safe_float(closes.iloc[index], 0.0)
        compare_close = _safe_float(closes.iloc[index - 4], 0.0)
        if close > compare_close:
            up_count += 1
            up_sequence_start = closes.index[index - up_count + 1]
        else:
            up_count = 0
            up_sequence_start = None

        if close < compare_close:
            down_count += 1
            down_sequence_start = closes.index[index - down_count + 1]
        else:
            down_count = 0
            down_sequence_start = None

    return {
        "up_count": up_count,
        "down_count": down_count,
        "up_9": up_count >= 9,
        "down_9": down_count >= 9,
        "up_sequence_start": up_sequence_start,
        "down_sequence_start": down_sequence_start,
    }

# =============================
# Market Regime（市場狀態）
# =============================
def detect_market_regime(df_1h, df_4h):
    close_1h = max(_safe_float(df_1h["close"].iloc[-1], 0.0), 1e-9)
    close_4h = max(_safe_float(df_4h["close"].iloc[-1], 0.0), 1e-9)
    ma25_1h = max(_safe_float(df_1h["ma25"].iloc[-1], close_1h), 1e-9)
    ma25_4h = max(_safe_float(df_4h["ma25"].iloc[-1], close_4h), 1e-9)
    trend_1h = close_1h - ma25_1h
    trend_4h = close_4h - ma25_4h
    trend_1h_rate = trend_1h / close_1h
    trend_4h_rate = trend_4h / close_4h

    # ===== 4H 強度（新增）=====
    strength_4h = ma25_4h - _safe_float(df_4h["ma25"].iloc[-5], ma25_4h)
    strength_4h_rate = strength_4h / ma25_4h

    # ===== 波動（用1H判斷市場活躍度）=====
    vol = (
        _safe_float(df_1h["high"].iloc[-1], close_1h)
        - _safe_float(df_1h["low"].iloc[-1], close_1h)
    ) / close_1h

    # 震盪不是只有價格剛好等於均線。用主播式區間邏輯：
    # 4H 乖離小、MA 斜率小、波動收斂，或 1H/4H 方向互打時，先視為區間。
    range_trend_threshold = max(0.002, _safe_float(os.getenv("MARKET_RANGE_TREND_RATE", 0.006), 0.006))
    range_strength_threshold = max(0.001, _safe_float(os.getenv("MARKET_RANGE_STRENGTH_RATE", 0.003), 0.003))
    range_vol_threshold = max(0.004, _safe_float(os.getenv("MARKET_RANGE_VOL_RATE", 0.012), 0.012))
    weak_4h = (
        abs(trend_4h_rate) <= range_trend_threshold
        and abs(strength_4h_rate) <= range_strength_threshold
        and vol <= range_vol_threshold
    )
    timeframe_conflict = (
        trend_1h_rate * trend_4h_rate < 0
        and abs(strength_4h_rate) <= range_strength_threshold * 1.4
        and vol <= range_vol_threshold * 1.15
    )
    if weak_4h or timeframe_conflict:
        return "range"

    # ===== v2 分類（強弱趨勢）=====
    # 多頭
    if trend_4h > 0:
        if strength_4h > 0 and vol > 0.008:
            return "bull_trend_strong"
        return "bull_trend"

    # 空頭
    if trend_4h < 0:
        if strength_4h < 0 and vol > 0.008:
            return "bear_trend_strong"
        return "bear_trend"

    return "range"


MLX_ANALYSIS_WINDOWS = {
    "monthly": {"label": "月線全部", "bars": None},
    "weekly": {"label": "週線2年", "bars": 104},
    "daily": {"label": "日線1年", "bars": 365},
    "four_hour": {"label": "4H一個月", "bars": 180},
    "one_hour": {"label": "1H一週", "bars": 168},
    "fifteen_min": {"label": "15m一天", "bars": 96},
}


def build_higher_timeframe_context(
    df_4h,
    df_1d=None,
    df_1w=None,
    df_1h=None,
    df_15m=None,
    df_1mth=None,
):
    def summarize(frame, label, short_bars, medium_bars, long_bars, window_label=None):
        empty = {
            f"{label}_trend": 0,
            f"{label}_strength_pct": 0.0,
            f"{label}_short_change_pct": 0.0,
            f"{label}_medium_change_pct": 0.0,
            f"{label}_long_change_pct": 0.0,
            f"{label}_window_change_pct": 0.0,
            f"{label}_range_pos": 0.5,
            f"{label}_samples": 0,
            f"{label}_coverage_pct": 0.0,
            f"{label}_support": None,
            f"{label}_resistance": None,
            f"{label}_window_label": window_label or "",
        }
        if frame is None or "close" not in frame.columns:
            return empty
        completed = frame.iloc[:-1].copy() if len(frame) > 1 else frame.copy()
        closes = pd.to_numeric(completed["close"], errors="coerce").dropna()
        if len(closes) < 4:
            empty[f"{label}_samples"] = int(len(closes))
            return empty

        def change_pct(bars):
            available_bars = min(max(1, int(bars)), len(closes) - 1)
            prior = float(closes.iloc[-(available_bars + 1)])
            return (close - prior) / max(abs(prior), 1e-9) * 100

        close = float(closes.iloc[-1])
        effective_long_bars = len(closes) - 1 if long_bars is None else int(long_bars)
        effective_long_bars = max(1, min(effective_long_bars, len(closes)))
        baseline = float(closes.tail(effective_long_bars).mean())
        range_rows = completed.tail(min(effective_long_bars, len(completed)))
        high = float(pd.to_numeric(range_rows["high"], errors="coerce").max())
        low = float(pd.to_numeric(range_rows["low"], errors="coerce").min())
        trend = 1 if close > baseline else -1 if close < baseline else 0
        short_change = change_pct(short_bars)
        medium_change = change_pct(medium_bars)
        long_change = change_pct(effective_long_bars)
        support, resistance = _calc_support_resistance_levels(
            completed, lookback=min(effective_long_bars, len(completed))
        )
        coverage_target = max(effective_long_bars + 1, 1)
        if long_bars is None:
            coverage = 100.0 if len(closes) >= 4 else 0.0
        else:
            coverage = min(100.0, len(closes) / coverage_target * 100)
        return {
            f"{label}_trend": trend,
            f"{label}_strength_pct": medium_change,
            f"{label}_short_change_pct": short_change,
            f"{label}_medium_change_pct": medium_change,
            f"{label}_long_change_pct": long_change,
            f"{label}_window_change_pct": long_change,
            f"{label}_range_pos": max(0.0, min(1.0, (close - low) / max(high - low, 1e-9))),
            f"{label}_samples": int(len(closes)),
            f"{label}_coverage_pct": coverage,
            f"{label}_support": support,
            f"{label}_resistance": resistance,
            f"{label}_window_label": window_label or "",
        }

    context = {}
    context.update(summarize(df_15m, "fifteen_min", 4, 32, 96, "15m一天"))
    context.update(summarize(df_1h, "one_hour", 24, 72, 168, "1H一週"))
    context.update(summarize(df_4h, "four_hour", 6, 42, 180, "4H一個月"))
    context.update(summarize(df_1d, "daily", 7, 90, 365, "日線1年"))
    context.update(summarize(df_1w, "weekly", 4, 26, 104, "週線2年"))
    context.update(summarize(df_1mth, "monthly", 1, 12, None, "月線全部"))
    context["analysis_windows"] = {
        key: value["label"] for key, value in MLX_ANALYSIS_WINDOWS.items()
    }
    candle_value = None
    if df_4h is not None and len(df_4h) > 1 and "time" in df_4h.columns:
        candle_value = df_4h["time"].iloc[-2]
    context["candle_key"] = f"ETHUSDT:4h:{candle_value}"
    return context

# =============================
# FVG
# =============================
def calc_fvg(df):
    if len(df) < 3:
        return None, None

    for i in range(len(df)-2, 1, -1):
        c1 = df.iloc[i-2]
        c3 = df.iloc[i]

        if c3["low"] > c1["high"]:
            return c1["high"], c3["low"]

        if c3["high"] < c1["low"]:
            return c3["high"], c1["low"]

    return None, None

# =============================
# Triangle Pattern（三角收斂）
# =============================
def detect_triangle(df, lookback=20):
    if len(df) < lookback:
        return 0

    highs = df["high"].tail(lookback).values
    lows = df["low"].tail(lookback).values

    # 線性回歸斜率
    x = np.arange(len(highs))
    high_slope = np.polyfit(x, highs, 1)[0]
    low_slope = np.polyfit(x, lows, 1)[0]

    # 上下降、下上升 → 收斂
    if high_slope < 0 and low_slope > 0:
        return 1   # 三角收斂

    return 0


def _detect_candlestick_pattern(df):
    """回傳 (pattern_name, bias)；bias: 1=偏多, -1=偏空, 0=中性。"""
    if df is None or len(df) < 3:
        return "不足", 0

    c1 = df.iloc[-1]
    c2 = df.iloc[-2]

    o1, h1, l1, p1 = [float(c1[k]) for k in ("open", "high", "low", "close")]
    o2, h2, l2, p2 = [float(c2[k]) for k in ("open", "high", "low", "close")]

    body1 = abs(p1 - o1)
    range1 = max(h1 - l1, 1e-9)
    upper_wick = h1 - max(o1, p1)
    lower_wick = min(o1, p1) - l1

    if (p2 < o2) and (p1 > o1) and (o1 <= p2) and (p1 >= o2):
        return "看漲吞沒", 1
    if (p2 > o2) and (p1 < o1) and (o1 >= p2) and (p1 <= o2):
        return "看跌吞沒", -1

    if body1 / range1 <= 0.15:
        return "十字星", 0

    if lower_wick >= body1 * 2.0 and upper_wick <= body1 * 0.8 and p1 > o1:
        return "錘子線", 1
    if upper_wick >= body1 * 2.0 and lower_wick <= body1 * 0.8 and p1 < o1:
        return "流星線", -1

    if p1 > o1:
        return "陽線延續", 1
    if p1 < o1:
        return "陰線延續", -1
    return "中性K", 0


def _range_position_label(range_pos):
    pos = max(0.0, min(1.0, _safe_float(range_pos, 0.5)))
    if pos <= 0.15:
        return "區間低位/支撐附近"
    if pos <= 0.35:
        return "偏低位"
    if pos < 0.65:
        return "中位震盪"
    if pos < 0.85:
        return "偏高位"
    return "區間高位/壓力附近"


def _trend_label(trend):
    value = _safe_int(trend, 0)
    if value > 0:
        return "偏多"
    if value < 0:
        return "偏空"
    return "中性"


def _pattern_bias_label(pattern_bias):
    value = _safe_int(pattern_bias, 0)
    if value > 0:
        return "K線偏多"
    if value < 0:
        return "K線偏空"
    return "K線中性"


def _timeframe_bias_score(trend, range_pos, pattern_bias, window_change_pct):
    pos = max(0.0, min(1.0, _safe_float(range_pos, 0.5)))
    score = _safe_int(trend, 0) * 0.55
    score += _safe_int(pattern_bias, 0) * 0.25
    change = _safe_float(window_change_pct, 0.0)
    if abs(change) >= 0.6:
        score += (1 if change > 0 else -1) * min(0.35, abs(change) / 12.0)
    if pos >= 0.82:
        score -= 0.18
    elif pos <= 0.18:
        score += 0.18
    return max(-1.0, min(1.0, score))


def _host_style_action_hint(tf_name, bias_score, range_pos):
    pos = max(0.0, min(1.0, _safe_float(range_pos, 0.5)))
    if tf_name in ("1M", "1W", "1D"):
        if bias_score > 0.25:
            return "高週期給多方背景，但高位不追多"
        if bias_score < -0.25:
            return "高週期給空方背景，但低位不追空"
        return "高週期未定向，短線降倉或等突破"
    if tf_name in ("4H", "1H"):
        if bias_score > 0.25:
            return "用回踩支撐找多，壓力前先減倉"
        if bias_score < -0.25:
            return "用反彈壓力找空，支撐前先減倉"
        return "方向混沌，等區間邊界或放量突破"
    if bias_score > 0.25:
        return "15m只找多單觸發，不在壓力高位硬追" if pos < 0.82 else "15m偏多但已近壓力，等回踩"
    if bias_score < -0.25:
        return "15m只找空單觸發，不在支撐低位硬追" if pos > 0.18 else "15m偏空但已近支撐，等反彈"
    return "15m沒有觸發，等待量能或結構確認"


def _build_timeframe_kline_view(higher_tf_context, pattern_map=None):
    context = higher_tf_context if isinstance(higher_tf_context, dict) else {}
    patterns = pattern_map if isinstance(pattern_map, dict) else {}
    specs = [
        ("1M", "monthly", "宏觀大區間"),
        ("1W", "weekly", "高週期趨勢"),
        ("1D", "daily", "主要風險背景"),
        ("4H", "four_hour", "波段方向"),
        ("1H", "one_hour", "日內方向"),
        ("15m", "fifteen_min", "進出場觸發"),
    ]
    view = {}
    summary_parts = []
    for tf_name, prefix, role in specs:
        pattern_name, pattern_bias = patterns.get(prefix, ("未提供", 0))
        trend = context.get(f"{prefix}_trend", 0)
        range_pos = _safe_float(context.get(f"{prefix}_range_pos"), 0.5)
        window_change = _safe_float(context.get(f"{prefix}_window_change_pct"), 0.0)
        support = context.get(f"{prefix}_support")
        resistance = context.get(f"{prefix}_resistance")
        bias_score = _timeframe_bias_score(trend, range_pos, pattern_bias, window_change)
        if bias_score > 0.25:
            bias_label = "偏多"
        elif bias_score < -0.25:
            bias_label = "偏空"
        else:
            bias_label = "中性"
        range_label = _range_position_label(range_pos)
        action = _host_style_action_hint(tf_name, bias_score, range_pos)
        support_text = f"{_safe_float(support):.2f}" if support else "無"
        resistance_text = f"{_safe_float(resistance):.2f}" if resistance else "無"
        text = (
            f"{role}：{_trend_label(trend)}，{pattern_name}/{_pattern_bias_label(pattern_bias)}，"
            f"{range_label}，窗口變化{window_change:.2f}%，支撐{support_text}，壓力{resistance_text}；"
            f"{action}"
        )
        view[tf_name] = {
            "role": role,
            "bias": bias_label,
            "trend": _trend_label(trend),
            "pattern": pattern_name,
            "pattern_bias": _pattern_bias_label(pattern_bias),
            "range_pos": round(range_pos, 4),
            "range_label": range_label,
            "window_change_pct": round(window_change, 4),
            "support": support,
            "resistance": resistance,
            "action": action,
            "text": text,
        }
        summary_parts.append(f"{tf_name}={text}")

    high_tf_score = (
        _timeframe_bias_score(
            context.get("monthly_trend", 0),
            context.get("monthly_range_pos", 0.5),
            patterns.get("monthly", ("", 0))[1],
            context.get("monthly_window_change_pct", 0.0),
        )
        * 0.30
        + _timeframe_bias_score(
            context.get("weekly_trend", 0),
            context.get("weekly_range_pos", 0.5),
            patterns.get("weekly", ("", 0))[1],
            context.get("weekly_window_change_pct", 0.0),
        )
        * 0.35
        + _timeframe_bias_score(
            context.get("daily_trend", 0),
            context.get("daily_range_pos", 0.5),
            patterns.get("daily", ("", 0))[1],
            context.get("daily_window_change_pct", 0.0),
        )
        * 0.35
    )
    mid_tf_score = (
        _timeframe_bias_score(
            context.get("four_hour_trend", 0),
            context.get("four_hour_range_pos", 0.5),
            patterns.get("four_hour", ("", 0))[1],
            context.get("four_hour_window_change_pct", 0.0),
        )
        * 0.55
        + _timeframe_bias_score(
            context.get("one_hour_trend", 0),
            context.get("one_hour_range_pos", 0.5),
            patterns.get("one_hour", ("", 0))[1],
            context.get("one_hour_window_change_pct", 0.0),
        )
        * 0.45
    )
    low_tf_score = _timeframe_bias_score(
        context.get("fifteen_min_trend", 0),
        context.get("fifteen_min_range_pos", 0.5),
        patterns.get("fifteen_min", ("", 0))[1],
        context.get("fifteen_min_window_change_pct", 0.0),
    )
    conflict = (
        (high_tf_score > 0.25 and mid_tf_score < -0.25)
        or (high_tf_score < -0.25 and mid_tf_score > 0.25)
    )
    turning_point = _score_multi_tf_candlestick_turning(context, patterns)
    host_style_order = (
        "先看月/週/日定大背景，再看4H/1H判斷主要方向與支撐壓力，"
        "最後只用15m找進場觸發；高位不追多、低位不追空，週期衝突時降倉或等待。"
    )
    return {
        "order": host_style_order,
        "high_tf_score": round(high_tf_score, 4),
        "mid_tf_score": round(mid_tf_score, 4),
        "low_tf_score": round(low_tf_score, 4),
        "conflict": bool(conflict),
        "turning_point": turning_point,
        "views": view,
        "summary": "；".join(summary_parts),
    }


def _score_multi_tf_candlestick_turning(higher_tf_context, pattern_map=None):
    """多時段K線反轉型態同時出現時，量化變盤機率。"""
    context = higher_tf_context if isinstance(higher_tf_context, dict) else {}
    patterns = pattern_map if isinstance(pattern_map, dict) else {}
    specs = [
        ("15m", "fifteen_min", 0.70),
        ("1H", "one_hour", 0.90),
        ("4H", "four_hour", 1.15),
        ("1D", "daily", 1.35),
        ("1W", "weekly", 1.05),
        ("1M", "monthly", 0.80),
    ]
    bullish_reversal = {"看漲吞沒", "錘子線"}
    bearish_reversal = {"看跌吞沒", "流星線"}
    bullish_score = 0.0
    bearish_score = 0.0
    bullish_count = 0
    bearish_count = 0
    reasons = []

    for label, prefix, weight in specs:
        pattern_name, pattern_bias = patterns.get(prefix, ("", 0))
        pattern_name = str(pattern_name or "")
        trend = _safe_int(context.get(f"{prefix}_trend"), 0)
        range_pos = max(0.0, min(1.0, _safe_float(context.get(f"{prefix}_range_pos"), 0.5)))
        window_change = _safe_float(context.get(f"{prefix}_window_change_pct"), 0.0)

        if pattern_name in bullish_reversal:
            if trend <= 0 or range_pos <= 0.42 or window_change < 0:
                score = weight * (1.15 if range_pos <= 0.30 else 1.0)
                bullish_score += score
                bullish_count += 1
                reasons.append(f"{label}{pattern_name}低位/跌後偏多")
        elif pattern_name in bearish_reversal:
            if trend >= 0 or range_pos >= 0.58 or window_change > 0:
                score = weight * (1.15 if range_pos >= 0.70 else 1.0)
                bearish_score += score
                bearish_count += 1
                reasons.append(f"{label}{pattern_name}高位/漲後偏空")
        elif pattern_name == "十字星":
            if range_pos >= 0.74 or (trend > 0 and window_change > 0):
                bearish_score += weight * 0.55
                bearish_count += 1
                reasons.append(f"{label}高位十字星偏空變盤")
            elif range_pos <= 0.26 or (trend < 0 and window_change < 0):
                bullish_score += weight * 0.55
                bullish_count += 1
                reasons.append(f"{label}低位十字星偏多變盤")

    direction = "neutral"
    raw_score = bullish_score - bearish_score
    if raw_score >= 0.65 and bullish_count >= 1:
        direction = "long"
    elif raw_score <= -0.65 and bearish_count >= 1:
        direction = "short"

    simultaneous_count = bullish_count if direction == "long" else bearish_count if direction == "short" else max(bullish_count, bearish_count)
    simultaneous = simultaneous_count >= 2
    confidence = 0.0
    if direction != "neutral":
        confidence = min(0.88, 0.30 + abs(raw_score) * 0.16 + max(0, simultaneous_count - 1) * 0.12)
        if simultaneous:
            confidence = min(0.92, confidence + 0.10)

    return {
        "direction": direction,
        "score": round(max(-1.0, min(1.0, raw_score / 3.0)), 4),
        "confidence": round(confidence, 4),
        "simultaneous": bool(simultaneous),
        "simultaneous_count": int(simultaneous_count),
        "bullish_count": int(bullish_count),
        "bearish_count": int(bearish_count),
        "reasons": reasons[:8],
    }


def _calc_support_resistance_levels(df, lookback=60):
    if df is None or len(df) < 20:
        return None, None

    window = df.tail(max(20, min(lookback, len(df))))
    support = float(np.percentile(window["low"], 12))
    resistance = float(np.percentile(window["high"], 88))

    if support <= 0 or resistance <= 0 or resistance <= support:
        return None, None
    return support, resistance


def _build_range_trade_reference(
    support_15m,
    resistance_15m,
    support_1h,
    resistance_1h,
    atr,
):
    supports = sorted(
        value
        for value in (_safe_float(support_15m, 0.0), _safe_float(support_1h, 0.0))
        if value > 0
    )
    resistances = sorted(
        value
        for value in (
            _safe_float(resistance_15m, 0.0),
            _safe_float(resistance_1h, 0.0),
        )
        if value > 0
    )
    if not supports or not resistances or supports[-1] >= resistances[0]:
        return {}
    support_low, support_high = supports[0], supports[-1]
    resistance_low, resistance_high = resistances[0], resistances[-1]
    buffer = max(_safe_float(atr, 0.0) * 0.5, resistance_high * 0.0005)
    target_buffer = buffer * 0.5
    return {
        "support_low": round(support_low, 2),
        "support_high": round(support_high, 2),
        "resistance_low": round(resistance_low, 2),
        "resistance_high": round(resistance_high, 2),
        "long_entry": round(support_high, 2),
        "long_tp": round(resistance_low - target_buffer, 2),
        "long_sl": round(support_low - buffer, 2),
        "short_entry": round(resistance_low, 2),
        "short_tp": round(support_high + target_buffer, 2),
        "short_sl": round(resistance_high + buffer, 2),
    }


def _score_host_opening_logic(
    *,
    price,
    timeframe_kline_view,
    range_pos,
    htf,
    mid_trend,
    breakout,
    regime,
    volume_spike,
    buy_pressure,
    sell_pressure,
    sweep_high,
    sweep_low,
    support_hits,
    resistance_hits,
    repeated_support_tests,
    repeated_resistance_tests,
    repeated_test_pressure,
    macro_bias,
):
    """主播思維開單主訊號：大週期定背景，中週期定方向，15m只找觸發。"""
    view = timeframe_kline_view if isinstance(timeframe_kline_view, dict) else {}
    high_score = _safe_float(view.get("high_tf_score"), 0.0)
    mid_score = _safe_float(view.get("mid_tf_score"), 0.0)
    low_score = _safe_float(view.get("low_tf_score"), 0.0)
    pos = max(0.0, min(1.0, _safe_float(range_pos, 0.5)))
    support_hits = _safe_int(support_hits, 0)
    resistance_hits = _safe_int(resistance_hits, 0)
    repeated_support_tests = _safe_int(repeated_support_tests, 0)
    repeated_resistance_tests = _safe_int(repeated_resistance_tests, 0)
    repeated_test_pressure = _safe_float(repeated_test_pressure, 0.0)
    macro_bias = _safe_float(macro_bias, 0.0)

    long_score = 0.0
    short_score = 0.0
    long_reasons = []
    short_reasons = []

    def add(direction, points, reason):
        nonlocal long_score, short_score
        if direction == "long":
            long_score += points
            long_reasons.append(reason)
        else:
            short_score += points
            short_reasons.append(reason)

    if high_score > 0.20:
        add("long", min(0.9, 0.45 + high_score * 0.55), "高週期背景偏多")
    elif high_score < -0.20:
        add("short", min(0.9, 0.45 + abs(high_score) * 0.55), "高週期背景偏空")
    else:
        long_score += 0.05
        short_score += 0.05

    if mid_score > 0.18:
        add("long", min(1.1, 0.55 + mid_score * 0.70), "4H/1H方向偏多")
    elif mid_score < -0.18:
        add("short", min(1.1, 0.55 + abs(mid_score) * 0.70), "4H/1H方向偏空")

    if low_score > 0.18:
        add("long", min(0.75, 0.35 + low_score * 0.45), "15m觸發偏多")
    elif low_score < -0.18:
        add("short", min(0.75, 0.35 + abs(low_score) * 0.45), "15m觸發偏空")

    if htf == 1:
        add("long", 0.30, "4H在MA上方")
    elif htf == -1:
        add("short", 0.30, "4H在MA下方")
    if mid_trend == 1:
        add("long", 0.35, "30m動能支持多方")
    elif mid_trend == -1:
        add("short", 0.35, "30m動能支持空方")

    if pos <= 0.24:
        add("long", 0.45, "低位靠近支撐，不追空")
        short_score -= 0.35
    elif pos >= 0.76:
        add("short", 0.45, "高位靠近壓力，不追多")
        long_score -= 0.35

    if support_hits > 0:
        add("long", min(0.55, support_hits * 0.18), "多週期支撐靠近")
    if resistance_hits > 0:
        add("short", min(0.55, resistance_hits * 0.18), "多週期壓力靠近")

    if repeated_support_tests >= 2:
        if breakout == -1 and (volume_spike or sell_pressure) and not sweep_low:
            add("short", min(1.0, repeated_support_tests * 0.22), "支撐連測後放量跌破")
            long_score -= 0.45
        else:
            add("long", min(0.75, repeated_support_tests * 0.16), "支撐連測未破，偏等承接")
            short_score -= 0.35
    if repeated_resistance_tests >= 2:
        if breakout == 1 and (volume_spike or buy_pressure) and not sweep_high:
            add("long", min(1.0, repeated_resistance_tests * 0.22), "壓力連測後放量突破")
            short_score -= 0.45
        else:
            add("short", min(0.75, repeated_resistance_tests * 0.16), "壓力連測未破，偏等反彈失敗")
            long_score -= 0.35

    if breakout == 1:
        if volume_spike or buy_pressure:
            add("long", 0.55, "短線突破且有量/買盤確認")
        else:
            long_score -= 0.30
            short_score += 0.20
            short_reasons.append("突破量能不足，防假突破")
    elif breakout == -1:
        if volume_spike or sell_pressure:
            add("short", 0.55, "短線跌破且有量/賣壓確認")
        else:
            short_score -= 0.30
            long_score += 0.20
            long_reasons.append("跌破量能不足，防假跌破")

    if sweep_low:
        add("long", 0.45, "掃低後收回，偏假跌破")
        short_score -= 0.35
    if sweep_high:
        add("short", 0.45, "掃高後收回，偏假突破")
        long_score -= 0.35

    if regime in {"bull_trend", "bull_trend_strong"}:
        add("long", 0.35 if regime == "bull_trend" else 0.55, "趨勢結構偏多")
    elif regime in {"bear_trend", "bear_trend_strong"}:
        add("short", 0.35 if regime == "bear_trend" else 0.55, "趨勢結構偏空")

    if macro_bias > 0.35:
        add("long", 0.18, "宏觀輔助偏多")
    elif macro_bias < -0.35:
        add("short", 0.18, "宏觀輔助偏空")

    long_score += max(0.0, -repeated_test_pressure) * 0.35
    short_score += max(0.0, repeated_test_pressure) * 0.35

    long_score = max(0.0, long_score)
    short_score = max(0.0, short_score)
    edge = long_score - short_score
    direction = "neutral"
    if edge >= 1.05:
        direction = "long"
    elif edge <= -1.05:
        direction = "short"
    confidence = min(0.88, abs(edge) / 3.8 + max(long_score, short_score) / 10.0)
    if confidence < 0.42:
        direction = "neutral"

    mode = "wait"
    if direction == "long":
        if repeated_resistance_tests >= 2 and breakout == 1:
            mode = "breakout_after_pressure_tests"
        elif pos <= 0.35 or support_hits > 0 or repeated_support_tests >= 2:
            mode = "support_reclaim"
        elif mid_score > 0.25 and low_score > 0.15:
            mode = "trend_pullback_long"
    elif direction == "short":
        if repeated_support_tests >= 2 and breakout == -1:
            mode = "breakdown_after_support_tests"
        elif pos >= 0.65 or resistance_hits > 0 or repeated_resistance_tests >= 2:
            mode = "resistance_rejection"
        elif mid_score < -0.25 and low_score < -0.15:
            mode = "trend_pullback_short"

    reasons = long_reasons if direction == "long" else short_reasons if direction == "short" else []
    return {
        "direction": direction,
        "confidence": round(confidence, 4),
        "long_score": round(long_score, 4),
        "short_score": round(short_score, 4),
        "edge": round(edge, 4),
        "mode": mode,
        "reasons": reasons[:8],
        "high_tf_score": round(high_score, 4),
        "mid_tf_score": round(mid_score, 4),
        "low_tf_score": round(low_score, 4),
        "range_pos": round(pos, 4),
    }


def _count_consecutive_level_tests(df, level, side, *, tolerance=0.0015):
    level = _safe_float(level, 0.0)
    if df is None or level <= 0 or len(df) == 0:
        return 0

    tol = max(level * max(_safe_float(tolerance, 0.0), 0.0), 1e-9)
    count = 0
    rows = df.tail(len(df))
    for _, row in rows.iloc[::-1].iterrows():
        high = _safe_float(row.get("high"), 0.0)
        low = _safe_float(row.get("low"), 0.0)
        close = _safe_float(row.get("close"), 0.0)
        if high <= 0 or low <= 0 or close <= 0:
            break

        if side == "support":
            touched = low <= level + tol and close >= level - tol
            broken_away = close < level - tol or low > level + tol
        else:
            touched = high >= level - tol and close <= level + tol
            broken_away = close > level + tol or high < level - tol

        if touched:
            count += 1
            continue
        if broken_away:
            break
        break
    return count


def analyze_repeated_level_tests(price, df_5m, df_15m, support, resistance, atr=0.0):
    price = max(_safe_float(price, 0.0), 0.0)
    atr = max(_safe_float(atr, 0.0), 0.0)
    if price <= 0:
        return {
            "support_tests": 0,
            "resistance_tests": 0,
            "pressure": 0.0,
            "near_support": False,
            "near_resistance": False,
        }

    support = _safe_float(support, 0.0)
    resistance = _safe_float(resistance, 0.0)
    tolerance = max(0.0015, min(0.006, atr / max(price, 1e-9) * 0.65))

    support_tests = 0
    resistance_tests = 0
    if support > 0:
        support_tests += _count_consecutive_level_tests(df_5m, support, "support", tolerance=tolerance)
        support_tests += _count_consecutive_level_tests(df_15m, support, "support", tolerance=tolerance)
    if resistance > 0:
        resistance_tests += _count_consecutive_level_tests(df_5m, resistance, "resistance", tolerance=tolerance)
        resistance_tests += _count_consecutive_level_tests(df_15m, resistance, "resistance", tolerance=tolerance)

    near_support = support > 0 and abs(price - support) / price <= tolerance * 1.5
    near_resistance = resistance > 0 and abs(resistance - price) / price <= tolerance * 1.5
    pressure = (resistance_tests - support_tests) * 0.035
    if not near_support and support_tests > 0:
        pressure = min(pressure + support_tests * 0.012, pressure)
    if not near_resistance and resistance_tests > 0:
        pressure = max(pressure - resistance_tests * 0.012, pressure)

    return {
        "support_tests": support_tests,
        "resistance_tests": resistance_tests,
        "pressure": pressure,
        "near_support": near_support,
        "near_resistance": near_resistance,
        "tolerance_rate": tolerance,
    }


def analyze_multi_tf_sr_frames(price, frame_map, tf_cfg=None):
    """多週期支撐壓力 + K線型態分析，可直接餵入已準備好的 K 線資料。"""
    tf_cfg = tf_cfg or [
        ("月線全部", "1M", 1500, 1.5),
        ("周線2年", "1w", 104, 1.3),
        ("日線1年", "1d", 365, 1.1),
        ("12h", "12h", 160, 1.0),
        ("4h一個月", "4h", 180, 0.9),
    ]

    lines = []
    score = 0.0
    support_hits = 0
    resistance_hits = 0

    px = max(_safe_float(price, 0.0), 0.0)
    near_threshold = 0.007  # 0.7%

    for label, interval, limit, weight in tf_cfg:
        df = None
        if isinstance(frame_map, dict):
            df = frame_map.get(interval)

        if df is None or len(df) < 20:
            continue

        if limit and len(df) > limit:
            df = df.tail(limit)

        support, resistance = _calc_support_resistance_levels(df, lookback=min(limit, 80))
        pattern, bias = _detect_candlestick_pattern(df)

        if support is None or resistance is None:
            continue

        dist_s = (px - support) / px if px > 0 else 999.0
        dist_r = (resistance - px) / px if px > 0 else 999.0

        sr_mark = ""
        if 0 <= dist_s <= near_threshold:
            support_hits += 1
            score += 0.20 * weight
            sr_mark = "近支撐"
        elif 0 <= dist_r <= near_threshold:
            resistance_hits += 1
            score -= 0.20 * weight
            sr_mark = "近壓力"

        score += 0.08 * weight * bias
        lines.append(
            f"{label} S:{support:.2f} / R:{resistance:.2f} | 型態:{pattern}{('｜' + sr_mark) if sr_mark else ''}"
        )

    return {
        "bias": score,
        "support_hits": support_hits,
        "resistance_hits": resistance_hits,
        "lines": lines[:5],
    }


def analyze_multi_tf_sr(price):
    """多週期支撐壓力 + K線型態分析（月線全部/週2年/日1年/12h/4h一個月）。"""
    tf_cfg = [
        ("月線全部", "1M", 1500, 1.5),
        ("周線2年", "1w", 104, 1.3),
        ("日線1年", "1d", 365, 1.1),
        ("12h", "12h", 160, 1.0),
        ("4h一個月", "4h", 180, 0.9),
    ]
    frame_map = {}
    for _, interval, limit, _ in tf_cfg:
        try:
            frame_map[interval] = get_kline(interval, limit=limit)
        except Exception:
            frame_map[interval] = None
    return analyze_multi_tf_sr_frames(price, frame_map, tf_cfg=tf_cfg)

# =============================
# AI（Meta Model）
# =============================
MODEL_PATH = ai_data_path("model.pkl")
DATA_PATH = ai_data_path("ai_data.csv")
BACKTEST_DATA_PATH = ai_data_path("backtest_ai_data.csv")
ONLINE_SCALER_PATH = ai_data_path("online_scaler.pkl")
ONLINE_MODEL_META_PATH = ai_data_path("online_model_meta.json")
AI_LEARNING_PIPELINE_VERSION = 2
STRATEGY_VERSION = str(os.getenv("ETH_BOT_STRATEGY_VERSION", "mlx-actual-v1")).strip() or "mlx-actual-v1"
AI_LEARNING_META_PATH = ai_data_path("ai_learning_meta.json")
MODEL_FEATURE_COLUMNS = [
    "htf",
    "htf_strength",
    "mid_trend",
    "macd",
    "hist",
    "price_vs_ma",
    "breakout",
    "fvg",
    "volatility",
    "trend_strength",
    "range_pos",
    "sp",
    "nq",
    "btc",
    "dxy",
    "macro",
    "regime",
    "triangle",
    "event_risk",
    "volume_spike",
    "volume_ratio",
    "buy_pressure",
    "absorption",
    "sweep_high",
    "sweep_low",
    "multi_tf_sr_bias",
    "multi_tf_support_hits",
    "multi_tf_resistance_hits",
    "candlestick_turn_score",
    "candlestick_turn_confidence",
    "candlestick_turn_count",
    "open_interest_change",
    "mark_premium_rate",
    "funding_rate_live",
    "taker_buy_ratio",
    "derivatives_pressure",
]
LEGACY_MODEL_FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS[:25]
ONLINE_MODEL_MIN_SAMPLES = max(12, _safe_int(os.getenv("ONLINE_MODEL_MIN_SAMPLES", 24), 24))
ONLINE_MODEL_FULL_WEIGHT_SAMPLES = max(
    ONLINE_MODEL_MIN_SAMPLES + 16,
    _safe_int(os.getenv("ONLINE_MODEL_FULL_WEIGHT_SAMPLES", 120), 120),
)
MLX_REPLACE_MODEL_ENABLED = _is_truthy(os.getenv("MLX_REPLACE_MODEL_ENABLED", "0"))
MLX_REPLACE_MODEL_MIN_SAMPLES = max(
    20,
    _safe_int(os.getenv("MLX_REPLACE_MODEL_MIN_SAMPLES", 60), 60),
)
MLX_REPLACE_MODEL_MIN_WEIGHT = max(
    5.0,
    _safe_float(os.getenv("MLX_REPLACE_MODEL_MIN_WEIGHT", 20.0), 20.0),
)
MLX_REPLACE_MODEL_MAX_SWING = min(
    0.40,
    max(0.05, _safe_float(os.getenv("MLX_REPLACE_MODEL_MAX_SWING", 0.28), 0.28)),
)
MODEL_RETRAIN_INTERVAL_SEC = max(120.0, _safe_float(os.getenv("MODEL_RETRAIN_INTERVAL_SEC", 300), 300))
AI_LOG_FLUSH_SIZE = max(1, _safe_int(os.getenv("AI_LOG_FLUSH_SIZE", 1), 1))
BACKTEST_SAMPLE_WEIGHT = min(1.0, max(0.05, _safe_float(os.getenv("BACKTEST_AI_SAMPLE_WEIGHT", 0.35), 0.35)))
BACKTEST_MAX_ROWS = max(100, _safe_int(os.getenv("BACKTEST_AI_MAX_ROWS", 1200), 1200))
PENDING_TRAINING_SAMPLE_MAX_AGE_SEC = max(
    3600.0,
    _safe_float(os.getenv("PENDING_TRAINING_SAMPLE_MAX_AGE_SEC", 172800), 172800),
)
SL_FOLLOWUP_REVIEWS_PATH = data_path("sl_followup_reviews.json")
SL_FOLLOWUP_REVIEW_DELAY_SEC = max(120.0, _safe_float(os.getenv("SL_FOLLOWUP_REVIEW_DELAY_SEC", 900), 900))
SL_FOLLOWUP_RECOVERY_ATR = max(0.15, _safe_float(os.getenv("SL_FOLLOWUP_RECOVERY_ATR", 0.6), 0.6))
SL_FOLLOWUP_MAX_AGE_SEC = max(
    SL_FOLLOWUP_REVIEW_DELAY_SEC * 2,
    _safe_float(os.getenv("SL_FOLLOWUP_MAX_AGE_SEC", 86400), 86400),
)
model = None


def _persist_batch_model(trained_model):
    _write_pickle_atomic(MODEL_PATH, trained_model)


def _fit_batch_model_from_dataframe(df, sample_weight=None):
    work = df.tail(1500).copy() if len(df) > 1500 else df.copy()
    X = work.drop(columns=["label"])
    y = work["label"]
    fit_sample_weight = None
    if sample_weight is not None:
        fit_sample_weight = np.asarray(sample_weight, dtype=float)
        if fit_sample_weight.shape[0] != len(df):
            fit_sample_weight = None
        elif len(df) > 1500:
            fit_sample_weight = fit_sample_weight[-len(work):]

    trained_model = RandomForestClassifier(
        n_estimators=180,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
    )
    if fit_sample_weight is not None:
        trained_model.fit(X, y, sample_weight=fit_sample_weight)
    else:
        trained_model.fit(X, y)
    return trained_model, len(work)


def _try_rebuild_batch_model_from_history(min_rows=10, reason=""):
    global model

    df, sample_weight = _load_weighted_training_dataframe(include_backtest=True)
    if df is None or len(df) < max(1, int(min_rows)):
        return False

    try:
        model, used_rows = _fit_batch_model_from_dataframe(df, sample_weight=sample_weight)
        _persist_batch_model(model)
        note = f" | 原因: {reason}" if reason else ""
        print(f"♻️ 已用 {used_rows} 筆歷史樣本重建 batch model{note}")
        return True
    except Exception as e:
        print(f"⚠️ 重建 batch model 失敗: {e}")
        model = None
        return False


def _new_online_model():
    # sklearn 的 partial_fit 不支援 class_weight="balanced"；
    # 改為在更新時動態提供 sample_weight。
    return SGDClassifier(loss="log_loss", alpha=1e-4, random_state=42)


def _new_online_scaler():
    return StandardScaler()


def _build_online_balanced_sample_weights(labels):
    y = np.array([1 if int(label) > 0 else 0 for label in labels], dtype=int)
    if y.size == 0:
        return np.array([], dtype=float)

    counts = {0: int(np.sum(y == 0)), 1: int(np.sum(y == 1))}
    present_classes = sum(1 for count in counts.values() if count > 0)
    if present_classes <= 0:
        return np.ones(y.size, dtype=float)

    total = float(y.size)
    class_weights = {}
    for cls, count in counts.items():
        if count <= 0:
            class_weights[cls] = 1.0
            continue
        class_weights[cls] = min(4.0, max(1.0, total / (present_classes * float(count))))

    return np.array([class_weights[int(label)] for label in y], dtype=float)


def _compute_online_sample_weight(label):
    clean_label = 1 if int(label) > 0 else 0
    future_counts = {
        0: max(0, _safe_int(online_label_counts.get(0), 0)),
        1: max(0, _safe_int(online_label_counts.get(1), 0)),
    }
    future_counts[clean_label] += 1

    present_classes = sum(1 for count in future_counts.values() if count > 0)
    total = future_counts[0] + future_counts[1]
    label_count = future_counts[clean_label]
    if present_classes <= 0 or total <= 0 or label_count <= 0:
        return 1.0

    return float(min(4.0, max(1.0, total / (present_classes * float(label_count)))))


# Online learning model
online_model = _new_online_model()
online_scaler = _new_online_scaler()
online_initialized = False
online_sample_count = 0
online_label_counts = {0: 0, 1: 0}
last_batch_train_ts = 0.0


def _align_feature_frame(frame, estimator):
    if estimator is None or not hasattr(estimator, "feature_names_in_"):
        return frame

    expected_cols = list(estimator.feature_names_in_)
    aligned = frame.copy()

    for col in expected_cols:
        if col not in aligned.columns:
            aligned[col] = 0

    extra_cols = [col for col in aligned.columns if col not in expected_cols]
    if extra_cols:
        aligned = aligned.drop(columns=extra_cols)

    return aligned[expected_cols]


def _has_expected_feature_schema(estimator) -> bool:
    cols = list(getattr(estimator, "feature_names_in_", []))
    return cols == MODEL_FEATURE_COLUMNS


def _normalize_feature_payload(features):
    payload = features if isinstance(features, dict) else {}
    normalized = {}
    for col in MODEL_FEATURE_COLUMNS:
        normalized[col] = _safe_float(payload.get(col), 0.0)
    return normalized


def _normalize_trade_direction(direction):
    return "short" if str(direction or "").strip().lower() == "short" else "long"


def _build_directional_learning_features(features, direction):
    """將空單樣本轉為與多單同語意的特徵空間，避免勝敗標籤互相污染。"""
    payload = _normalize_feature_payload(features)
    if _normalize_trade_direction(direction) != "short":
        return payload

    transformed = dict(payload)
    signed_cols = [
        "htf",
        "mid_trend",
        "macd",
        "hist",
        "price_vs_ma",
        "breakout",
        "sp",
        "nq",
        "btc",
        "dxy",
        "macro",
        "regime",
        "multi_tf_sr_bias",
        "candlestick_turn_score",
        "mark_premium_rate",
        "funding_rate_live",
        "derivatives_pressure",
    ]

    for col in signed_cols:
        transformed[col] = -_safe_float(payload.get(col), 0.0)

    transformed["range_pos"] = max(0.0, min(1.0, 1.0 - _safe_float(payload.get("range_pos"), 0.0)))
    transformed["buy_pressure"] = max(0.0, min(1.0, 1.0 - _safe_float(payload.get("buy_pressure"), 0.0)))
    transformed["taker_buy_ratio"] = max(0.0, min(1.0, 1.0 - _safe_float(payload.get("taker_buy_ratio"), 0.5)))
    transformed["sweep_high"] = _safe_float(payload.get("sweep_low"), 0.0)
    transformed["sweep_low"] = _safe_float(payload.get("sweep_high"), 0.0)
    transformed["multi_tf_support_hits"] = _safe_float(payload.get("multi_tf_resistance_hits"), 0.0)
    transformed["multi_tf_resistance_hits"] = _safe_float(payload.get("multi_tf_support_hits"), 0.0)
    return _normalize_feature_payload(transformed)


def _sanitize_pending_training_sample(sample):
    if not isinstance(sample, dict):
        return None

    entry_ts = _safe_float(sample.get("entry_ts"), 0.0)
    if entry_ts <= 0 or (time.time() - entry_ts) > PENDING_TRAINING_SAMPLE_MAX_AGE_SEC:
        return None

    direction = _normalize_trade_direction(sample.get("direction"))
    features = sample.get("features") if isinstance(sample.get("features"), dict) else None
    learn_features = sample.get("learn_features") if isinstance(sample.get("learn_features"), dict) else None

    if not isinstance(features, dict) and not isinstance(learn_features, dict):
        return None

    if not isinstance(features, dict):
        features = dict(learn_features)
    if not isinstance(learn_features, dict):
        learn_features = _build_directional_learning_features(features, direction)

    return {
        "features": _normalize_feature_payload(features),
        "learn_features": _normalize_feature_payload(learn_features),
        "direction": direction,
        "entry_ts": entry_ts,
        "mlx_episode_id": _safe_int(sample.get("mlx_episode_id"), 0),
    }


def _load_pending_training_sample_state():
    sample = _sanitize_pending_training_sample(_read_pending_training_sample_state_raw())
    if sample is None:
        _clear_pending_training_sample_state()
    return sample


def _save_pending_training_sample_state(sample):
    clean = _sanitize_pending_training_sample(sample)
    if clean is None:
        _clear_pending_training_sample_state()
        return None

    _write_json_atomic(PENDING_TRAINING_SAMPLE_PATH, clean)
    return clean


def _build_actual_trade_mlx_market(decision, direction, source, daily_min_trade=False):
    decision = decision if isinstance(decision, dict) else {}
    higher_timeframe = decision.get("higher_timeframe")
    market = dict(higher_timeframe) if isinstance(higher_timeframe, dict) else {}
    for key in (
        "price",
        "score",
        "ai_prob",
        "ai_long_prob",
        "ai_short_prob",
        "htf",
        "mid_trend",
        "daily_trend",
        "weekly_trend",
        "regime",
        "breakout",
        "triangle",
        "macro_bias",
        "volume_spike",
        "rsi_15m",
        "derivatives_pressure",
        "support_hits",
        "resistance_hits",
    ):
        if key in decision:
            market[key] = decision.get(key)
    market["macro"] = decision.get("macro_bias", market.get("macro"))
    market["direction"] = direction
    market["source"] = source
    market["daily_min_trade"] = bool(daily_min_trade)
    market["primary_reason"] = "每日最低一單" if daily_min_trade else "實單策略觸發"
    market["strategy_version"] = STRATEGY_VERSION
    features = decision.get("features")
    if isinstance(features, dict):
        market["features"] = dict(features)
        market.setdefault("sr_bias", features.get("multi_tf_sr_bias"))
    return market


def _maybe_backfill_pending_training_sample(
    pending_sample,
    df_4h,
    df_1h,
    df_30m,
    df_15m,
    df_5m,
    price,
    sr_analysis,
    sp_change,
    nq_change,
    btc_change,
    dxy_change,
    news_bias,
    event_risk,
    last_signal=None,
    losing_streak=0,
):
    if not active_trade.get("open"):
        return pending_sample
    if pending_sample and _safe_int(pending_sample.get("mlx_episode_id"), 0) > 0:
        return pending_sample

    direction = _normalize_trade_direction(active_trade.get("direction"))
    open_ts = _safe_float(
        pending_sample.get("entry_ts") if isinstance(pending_sample, dict) else None,
        _safe_float(active_trade.get("open_time"), 0.0),
    )
    if open_ts <= 0 or direction not in {"long", "short"}:
        return pending_sample

    try:
        decision = build_trade_signal_snapshot(
            df_4h=df_4h,
            df_1h=df_1h,
            df_30m=df_30m,
            df_15m=df_15m,
            df_5m=df_5m,
            price=price,
            sr_analysis=sr_analysis,
            sp_change=sp_change,
            nq_change=nq_change,
            btc_change=btc_change,
            dxy_change=dxy_change,
            news_bias=news_bias,
            event_risk=event_risk,
            last_signal=last_signal,
            losing_streak=losing_streak,
        )
        features = decision.get("features")
        if not isinstance(features, dict):
            return pending_sample
        mlx_episode_id = record_actual_trade_open(
            direction=direction,
            entry_price=_safe_float(active_trade.get("avg_entry", active_trade.get("entry")), price),
            tp_price=_safe_float(active_trade.get("tp"), 0.0),
            sl_price=_safe_float(active_trade.get("sl"), 0.0),
            market=_build_actual_trade_mlx_market(
                decision,
                direction,
                source="restored_position",
            ),
            reason_text="重啟後依目前持倉狀態補建實單分析",
            opened_at=open_ts,
            source="restored_position",
        )
        existing_features = (
            pending_sample.get("features")
            if isinstance(pending_sample, dict) and isinstance(pending_sample.get("features"), dict)
            else features
        )
        existing_learn_features = (
            pending_sample.get("learn_features")
            if isinstance(pending_sample, dict) and isinstance(pending_sample.get("learn_features"), dict)
            else _build_directional_learning_features(existing_features, direction)
        )
        rebuilt = _save_pending_training_sample_state(
            {
                "features": dict(existing_features),
                "learn_features": dict(existing_learn_features),
                "direction": direction,
                "entry_ts": open_ts,
                "mlx_episode_id": _safe_int(mlx_episode_id, 0),
            }
        )
        if rebuilt:
            print("♻️ 已依目前開倉狀態補回待學習樣本")
            return rebuilt
    except Exception as e:
        print(f"⚠️ 補回待學習樣本失敗: {e}")

    return pending_sample


def _load_sl_followup_reviews():
    if not SL_FOLLOWUP_REVIEWS_PATH.exists():
        return []

    try:
        raw = json.loads(SL_FOLLOWUP_REVIEWS_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠️ 載入 SL 後續學習佇列失敗: {e}")
        return []

    items = raw.get("reviews") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []

    now_ts = time.time()
    reviews = []
    for item in items:
        if not isinstance(item, dict):
            continue

        review_after_ts = _safe_float(item.get("review_after_ts"), 0.0)
        close_ts = _safe_float(item.get("close_ts"), 0.0)
        close_price = max(0.0, _safe_float(item.get("close_price"), 0.0))
        atr = max(0.0, _safe_float(item.get("atr"), 0.0))
        features = item.get("features") if isinstance(item.get("features"), dict) else {}

        if review_after_ts <= 0 or close_price <= 0:
            continue
        if close_ts > 0 and (now_ts - close_ts) > SL_FOLLOWUP_MAX_AGE_SEC:
            continue

        reviews.append(
            {
                "direction": _normalize_trade_direction(item.get("direction")),
                "features": _normalize_feature_payload(features),
                "close_price": close_price,
                "close_ts": close_ts,
                "atr": atr,
                "review_after_ts": review_after_ts,
            }
        )

    return reviews[-200:]


SL_FOLLOWUP_REVIEWS = _load_sl_followup_reviews()


def _save_sl_followup_reviews():
    payload = {"reviews": SL_FOLLOWUP_REVIEWS[-200:]}
    _write_json_atomic(SL_FOLLOWUP_REVIEWS_PATH, payload)


def _queue_sl_followup_review(features, direction, close_price, close_ts, atr):
    global SL_FOLLOWUP_REVIEWS

    sample = {
        "direction": _normalize_trade_direction(direction),
        "features": _normalize_feature_payload(features),
        "close_price": max(0.0, _safe_float(close_price, 0.0)),
        "close_ts": _safe_float(close_ts, time.time()),
        "atr": max(0.0, _safe_float(atr, 0.0)),
        "review_after_ts": _safe_float(close_ts, time.time()) + SL_FOLLOWUP_REVIEW_DELAY_SEC,
    }

    if sample["close_price"] <= 0:
        return

    SL_FOLLOWUP_REVIEWS.append(sample)
    SL_FOLLOWUP_REVIEWS = SL_FOLLOWUP_REVIEWS[-200:]
    _save_sl_followup_reviews()


def _read_ai_learning_meta():
    if not AI_LEARNING_META_PATH.exists():
        return {}

    try:
        raw = json.loads(AI_LEARNING_META_PATH.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _write_ai_learning_meta(extra=None):
    payload = {
        "version": int(AI_LEARNING_PIPELINE_VERSION),
        "updated_at": datetime.datetime.now().isoformat(),
    }
    if isinstance(extra, dict):
        payload.update(extra)
    _write_json_atomic(AI_LEARNING_META_PATH, payload)


def _archive_ai_learning_artifact(path_str, backup_tag):
    path = Path(path_str)
    if not path.exists():
        return ""

    backup_name = f"{path.stem}.v{AI_LEARNING_PIPELINE_VERSION - 1}_backup_{backup_tag}{path.suffix}"
    backup_path = path.with_name(backup_name)

    try:
        os.replace(path, backup_path)
        return str(backup_path)
    except Exception as e:
        print(f"⚠️ 備份學習檔案失敗 {path.name}: {e}")
        return ""


def _ensure_ai_learning_pipeline_version():
    meta = _read_ai_learning_meta()
    current_version = _safe_int(meta.get("version"), 0) if isinstance(meta, dict) else 0
    if current_version >= AI_LEARNING_PIPELINE_VERSION:
        return

    backup_tag = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    archived = []
    for artifact in [DATA_PATH, MODEL_PATH, ONLINE_MODEL_PATH, ONLINE_SCALER_PATH, ONLINE_MODEL_META_PATH]:
        backup_path = _archive_ai_learning_artifact(artifact, backup_tag)
        if backup_path:
            archived.append(backup_path)

    _write_ai_learning_meta(
        {
            "backup_tag": backup_tag,
            "archived": archived,
        }
    )

    if archived:
        print("♻️ AI 學習邏輯已切換版本；舊資料與模型已備份並重建")
    else:
        print("♻️ AI 學習邏輯版本已初始化")


def _coerce_training_dataframe_columns(df):
    if df is None or df.empty:
        return None

    work = df.copy()

    if "label" not in work.columns:
        if not all(isinstance(col, (int, np.integer)) for col in work.columns):
            return None
        feature_count = work.shape[1] - 1
        if feature_count == len(MODEL_FEATURE_COLUMNS):
            work.columns = MODEL_FEATURE_COLUMNS + ["label"]
        elif feature_count == len(LEGACY_MODEL_FEATURE_COLUMNS):
            work.columns = LEGACY_MODEL_FEATURE_COLUMNS + ["label"]
        else:
            return None
    else:
        feature_cols = [col for col in work.columns if col != "label"]
        if all(col in MODEL_FEATURE_COLUMNS for col in feature_cols):
            pass
        elif all(str(col).startswith("feature_") for col in feature_cols):
            ordered = sorted(feature_cols, key=lambda col: int(str(col).split("_")[1]))
            if len(ordered) == len(MODEL_FEATURE_COLUMNS):
                rename_cols = MODEL_FEATURE_COLUMNS
            elif len(ordered) == len(LEGACY_MODEL_FEATURE_COLUMNS):
                rename_cols = LEGACY_MODEL_FEATURE_COLUMNS
            else:
                return None
            work = work.rename(columns={src: dst for src, dst in zip(ordered, rename_cols)})
        else:
            return None

    for col in MODEL_FEATURE_COLUMNS:
        if col not in work.columns:
            work[col] = 0.0

    extra_cols = [col for col in work.columns if col not in MODEL_FEATURE_COLUMNS + ["label"]]
    if extra_cols:
        work = work.drop(columns=extra_cols)

    return work[MODEL_FEATURE_COLUMNS + ["label"]]


def _load_training_dataframe_from_path(path_str):
    if not os.path.exists(path_str):
        return None

    try:
        df = pd.read_csv(path_str)
        df = _coerce_training_dataframe_columns(df)
        if df is None:
            df = pd.read_csv(path_str, header=None)
            df = _coerce_training_dataframe_columns(df)
    except Exception as e:
        print(f"⚠️ 讀取訓練數據失敗 ({Path(path_str).name}): {e}")
        return None

    try:
        df = df.dropna(how="all")
        df = df.apply(pd.to_numeric, errors="coerce")
        if "label" not in df.columns:
            return None

        feature_cols = [col for col in df.columns if col != "label"]
        if not feature_cols:
            return None

        df = df.dropna(subset=feature_cols + ["label"])
        if df.empty:
            return None

        df["label"] = df["label"].astype(int)
        return df
    except Exception as e:
        print(f"⚠️ 清洗訓練數據失敗 ({Path(path_str).name}): {e}")
        return None


def _load_training_dataframe():
    return _load_training_dataframe_from_path(DATA_PATH)


def _load_weighted_training_dataframe(include_backtest=False):
    frames = []
    weights = []

    live_df = _load_training_dataframe_from_path(DATA_PATH)
    if live_df is not None and not live_df.empty:
        frames.append(live_df)
        weights.append(np.ones(len(live_df), dtype=float))

    if include_backtest:
        backtest_df = _load_training_dataframe_from_path(BACKTEST_DATA_PATH)
        if backtest_df is not None and not backtest_df.empty:
            if len(backtest_df) > BACKTEST_MAX_ROWS:
                backtest_df = backtest_df.tail(BACKTEST_MAX_ROWS).copy()
            frames.append(backtest_df)
            weights.append(np.full(len(backtest_df), BACKTEST_SAMPLE_WEIGHT, dtype=float))

    if not frames:
        return None, None

    df = pd.concat(frames, ignore_index=True)
    sample_weight = np.concatenate(weights).astype(float) if weights else None
    return df, sample_weight


def _reset_online_learning_state(reason=""):
    global online_model, online_scaler, online_initialized, online_sample_count, online_label_counts
    if reason:
        print(reason)
    online_model = _new_online_model()
    online_scaler = _new_online_scaler()
    online_initialized = False
    online_sample_count = 0
    online_label_counts = {0: 0, 1: 0}


def _save_online_learning_state():
    meta = {
        "sample_count": int(online_sample_count),
        "label_counts": {
            "0": int(online_label_counts.get(0, 0)),
            "1": int(online_label_counts.get(1, 0)),
        },
        "saved_at": datetime.datetime.now().isoformat(),
    }
    try:
        ensure_parent_dir(ONLINE_MODEL_PATH)
        ensure_parent_dir(ONLINE_SCALER_PATH)
        ensure_parent_dir(ONLINE_MODEL_META_PATH)
        _write_pickle_atomic(ONLINE_MODEL_PATH, online_model)
        _write_pickle_atomic(ONLINE_SCALER_PATH, online_scaler)
        _write_json_atomic(Path(ONLINE_MODEL_META_PATH), meta)
    except Exception as e:
        print(f"⚠️ 儲存 online model 狀態失敗: {e}")


def _bootstrap_online_model_from_history(max_rows=400):
    global online_model, online_scaler, online_initialized, online_sample_count, online_label_counts

    df = _load_training_dataframe()
    if df is None or df.empty:
        return False

    work = df.tail(max(1, int(max_rows))).copy()
    X_raw = work.drop(columns=["label"])
    y = work["label"].astype(int).to_numpy()
    sample_weight = _build_online_balanced_sample_weights(y)

    try:
        _reset_online_learning_state()
        X = _prepare_online_feature_frame(X_raw, fit_scaler=True)
        online_model.partial_fit(X, y, classes=np.array([0, 1]), sample_weight=sample_weight)
        online_initialized = True
        online_sample_count = int(len(y))
        online_label_counts = {
            0: int(np.sum(y == 0)),
            1: int(np.sum(y == 1)),
        }
        _save_online_learning_state()
        print(f"♻️ 已用 {len(y)} 筆歷史樣本重建 online model")
        return True
    except Exception as e:
        _reset_online_learning_state(f"⚠️ 重建 online model 失敗: {e}")
        return False


def _prepare_online_feature_frame(frame, fit_scaler=False):
    X = frame.copy()
    if hasattr(online_scaler, "feature_names_in_"):
        X = _align_feature_frame(X, online_scaler)
    elif online_initialized and hasattr(online_model, "feature_names_in_"):
        X = _align_feature_frame(X, online_model)
    else:
        X = _align_feature_frame(X, type("FeatureSchema", (), {"feature_names_in_": MODEL_FEATURE_COLUMNS})())

    if fit_scaler:
        online_scaler.partial_fit(X)

    if hasattr(online_scaler, "n_samples_seen_"):
        scaled = online_scaler.transform(X)
        return pd.DataFrame(scaled, columns=X.columns)
    return X


def _predict_estimator_probability(estimator, frame):
    if estimator is None or not hasattr(estimator, "predict_proba"):
        return None
    try:
        return float(estimator.predict_proba(_align_feature_frame(frame, estimator))[0][1])
    except Exception:
        return None


def _predict_estimator_probability_from_features(estimator, features):
    if estimator is None or not hasattr(estimator, "predict_proba"):
        return None
    expected_cols = list(getattr(estimator, "feature_names_in_", []))
    if not expected_cols:
        return None
    try:
        x = np.asarray(
            [[_safe_float(features.get(col), 0.0) for col in expected_cols]],
            dtype=np.float32,
        )
        if (
            _is_truthy(os.getenv("BACKTEST_FAST_MODEL_PREDICT", "0"))
            and isinstance(estimator, RandomForestClassifier)
            and hasattr(estimator, "estimators_")
        ):
            classes = list(getattr(estimator, "classes_", []))
            if 1 not in classes:
                return None
            class_idx = classes.index(1)
            max_trees = max(1, _safe_int(os.getenv("BACKTEST_FAST_MODEL_TREES", 24), 24))
            total = 0.0
            count = 0
            for tree in list(estimator.estimators_)[:max_trees]:
                total += float(tree.predict_proba(x, check_input=False)[0][class_idx])
                count += 1
            if count > 0:
                return total / count
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classes = list(getattr(estimator, "classes_", []))
            proba = estimator.predict_proba(x)[0]
            class_idx = classes.index(1) if 1 in classes else min(1, len(proba) - 1)
            return float(proba[class_idx])
    except Exception:
        return None


def _predict_legacy_trade_probability(features):
    normalized = _normalize_feature_payload(features)
    X_raw = pd.DataFrame([normalized])

    batch_prob = None
    if model is not None and _has_expected_feature_schema(model):
        if _is_truthy(os.getenv("BACKTEST_FAST_MODEL_PREDICT", "0")):
            batch_prob = _predict_estimator_probability_from_features(model, normalized)
        if batch_prob is None:
            batch_prob = _predict_estimator_probability(model, X_raw)

    online_prob = None
    skip_online_fast = (
        _is_truthy(os.getenv("BACKTEST_FAST_MODEL_PREDICT", "0"))
        and _is_truthy(os.getenv("BACKTEST_FAST_MODEL_SKIP_ONLINE", "1"))
    )
    online_ready = (
        not skip_online_fast
        and online_initialized
        and online_sample_count >= ONLINE_MODEL_MIN_SAMPLES
        and _has_expected_feature_schema(online_model)
        and hasattr(online_scaler, "n_samples_seen_")
    )
    if online_ready:
        X_online = _prepare_online_feature_frame(X_raw, fit_scaler=False)
        online_prob = _predict_estimator_probability(online_model, X_online)
    elif batch_prob is None and not skip_online_fast and online_initialized and hasattr(online_scaler, "n_samples_seen_"):
        X_online = _prepare_online_feature_frame(X_raw, fit_scaler=False)
        online_prob = _predict_estimator_probability(online_model, X_online)

    if batch_prob is not None and online_prob is not None:
        ramp = min(
            1.0,
            max(
                0.0,
                (online_sample_count - ONLINE_MODEL_MIN_SAMPLES)
                / max(1, ONLINE_MODEL_FULL_WEIGHT_SAMPLES - ONLINE_MODEL_MIN_SAMPLES),
            ),
        )
        online_weight = 0.20 + 0.45 * ramp
        return batch_prob * (1 - online_weight) + online_prob * online_weight
    if batch_prob is not None:
        return batch_prob
    if online_prob is not None:
        return online_prob
    return 0.5


def _predict_mlx_learning_probability(features, direction="setup"):
    if not MLX_REPLACE_MODEL_ENABLED:
        return None
    try:
        result = predict_mlx_replacement_probability(
            _normalize_feature_payload(features),
            direction=direction,
        )
    except Exception as exc:
        print(f"⚠️ MLX learning replacement probability failed: {exc}")
        return None
    if not isinstance(result, dict):
        return None
    sample_count = _safe_int(result.get("sample_count"), 0)
    effective_weight = _safe_float(result.get("effective_weight"), 0.0)
    if sample_count < MLX_REPLACE_MODEL_MIN_SAMPLES or effective_weight < MLX_REPLACE_MODEL_MIN_WEIGHT:
        return None
    probability = max(0.05, min(0.95, _safe_float(result.get("probability"), 0.5)))
    return 0.5 + max(-MLX_REPLACE_MODEL_MAX_SWING, min(MLX_REPLACE_MODEL_MAX_SWING, probability - 0.5))


def _predict_trade_probability(features, direction="setup"):
    mlx_prob = _predict_mlx_learning_probability(features, direction=direction)
    if mlx_prob is not None:
        return mlx_prob
    return _predict_legacy_trade_probability(features)


def _predict_directional_trade_bias(features):
    long_features = _build_directional_learning_features(features, "long")
    short_features = _build_directional_learning_features(features, "short")
    long_prob = _predict_trade_probability(long_features, direction="long")
    short_prob = _predict_trade_probability(short_features, direction="short")

    long_edge = max(0.0, long_prob - 0.5)
    short_edge = max(0.0, short_prob - 0.5)

    if long_edge <= 0 and short_edge <= 0:
        return 0.5, long_prob, short_prob

    bias_prob = 0.5 + (long_edge - short_edge)
    return max(0.05, min(bias_prob, 0.95)), long_prob, short_prob


def _compute_macro_bias(sp_change, nq_change, btc_change, dxy_change, news_bias, event_risk):
    macro_bias = 0.0

    if btc_change > 0.002:
        macro_bias += 1.5
    elif btc_change < -0.002:
        macro_bias -= 1.5

    if nq_change > 0.0015:
        macro_bias += 1.2
    elif nq_change < -0.0015:
        macro_bias -= 1.2

    if sp_change > 0.0015:
        macro_bias += 0.6
    elif sp_change < -0.0015:
        macro_bias -= 0.6

    if dxy_change > 0.0015:
        macro_bias -= 1.0
    elif dxy_change < -0.0015:
        macro_bias += 1.0

    macro_bias += _safe_float(news_bias, 0.0) * 0.8

    if _safe_int(event_risk, 0) >= 1:
        macro_bias *= 1.2
    if _safe_int(event_risk, 0) >= 2:
        macro_bias *= 1.5

    return float(macro_bias)


def _score_macro_indicator_alignment(
    *,
    direction,
    host_mode,
    sp_change,
    nq_change,
    btc_change,
    dxy_change,
    news_bias,
    event_risk,
    macro_bias,
    htf,
    mid_trend,
    breakout,
    regime,
    sr_bias,
    support_hits,
    resistance_hits,
    repeated_test_pressure,
    derivatives_pressure,
    taker_buy_ratio,
    volume_spike,
    buy_pressure,
    sell_pressure,
    sweep_high,
    sweep_low,
    range_pos,
    timeframe_kline_view,
):
    """Combine news, global risk assets, derivatives, SR, and K-lines into a final quality gate."""
    direction = str(direction or "").lower()
    host_mode = str(host_mode or "wait")
    sign = 1 if direction == "long" else -1 if direction == "short" else 0
    if sign == 0:
        return {"score": 0.0, "aligned": 0, "against": 0, "reasons": [], "hard_block": False}

    view = timeframe_kline_view if isinstance(timeframe_kline_view, dict) else {}
    pos = max(0.0, min(1.0, _safe_float(range_pos, 0.5)))
    taker_buy_ratio = max(0.0, min(1.0, _safe_float(taker_buy_ratio, 0.5)))
    event_risk = _safe_int(event_risk, 0)

    score = 0.0
    aligned = 0
    against = 0
    reasons = []

    def add(value, reason, *, threshold=0.0):
        nonlocal score, aligned, against
        value = _safe_float(value, 0.0)
        if abs(value) <= threshold:
            return
        score += value
        if value > 0:
            aligned += 1
        else:
            against += 1
        reasons.append(reason)

    macro_component = max(-1.2, min(1.2, _safe_float(macro_bias, 0.0) * 0.36)) * sign
    add(macro_component, "宏觀分數同向" if macro_component > 0 else "宏觀分數逆向", threshold=0.12)

    market_component = 0.0
    for value, weight, inverse in (
        (btc_change, 170.0, False),
        (nq_change, 140.0, False),
        (sp_change, 85.0, False),
        (dxy_change, 120.0, True),
    ):
        component = _safe_float(value, 0.0) * weight
        if inverse:
            component *= -1
        market_component += max(-0.45, min(0.45, component))
    market_component *= sign
    add(market_component, "全球股市/美元/BTC同向" if market_component > 0 else "全球股市/美元/BTC逆向", threshold=0.10)

    news_component = max(-0.85, min(0.85, _safe_float(news_bias, 0.0) * 0.42)) * sign
    add(news_component, "新聞情緒同向" if news_component > 0 else "新聞情緒逆向", threshold=0.10)

    high_tf = max(-1.0, min(1.0, _safe_float(view.get("high_tf_score"), 0.0)))
    mid_tf = max(-1.0, min(1.0, _safe_float(view.get("mid_tf_score"), 0.0)))
    low_tf = max(-1.0, min(1.0, _safe_float(view.get("low_tf_score"), 0.0)))
    tf_views = view.get("views") if isinstance(view.get("views"), dict) else {}
    range_positions = [
        _safe_float((tf_views.get(tf) or {}).get("range_pos"), 0.5)
        for tf in ("15m", "1H", "4H", "1D")
        if isinstance(tf_views.get(tf), dict)
    ]
    upper_zone_count = sum(1 for value in range_positions if value >= 0.68)
    lower_zone_count = sum(1 for value in range_positions if value <= 0.32)
    kline_component = (high_tf * 0.45 + mid_tf * 0.55 + low_tf * 0.30) * sign
    add(kline_component, "多週期K線同向" if kline_component > 0 else "多週期K線逆向", threshold=0.12)

    trend_component = 0.0
    trend_component += 0.35 if htf == sign else -0.30
    trend_component += 0.45 if mid_trend == sign else -0.38
    if regime in {"bull_trend", "bull_trend_strong"}:
        trend_component += 0.35 * sign
    elif regime in {"bear_trend", "bear_trend_strong"}:
        trend_component -= 0.35 * sign
    add(trend_component, "大中週期趨勢同向" if trend_component > 0 else "大中週期趨勢逆向", threshold=0.10)

    trigger_component = 0.0
    if breakout == sign:
        trigger_component += 0.45
    elif breakout == -sign:
        trigger_component -= 0.45
    if direction == "long":
        trigger_component += 0.28 if buy_pressure else -0.12
        trigger_component += 0.30 if sweep_low else 0.0
        trigger_component -= 0.35 if sweep_high else 0.0
        trigger_component -= 0.28 if pos >= 0.78 else 0.0
        trigger_component += 0.22 if pos <= 0.32 else 0.0
    else:
        trigger_component += 0.28 if sell_pressure else -0.12
        trigger_component += 0.30 if sweep_high else 0.0
        trigger_component -= 0.35 if sweep_low else 0.0
        trigger_component -= 0.28 if pos <= 0.22 else 0.0
        trigger_component += 0.22 if pos >= 0.68 else 0.0
    if volume_spike:
        trigger_component += 0.25 if trigger_component > 0 else -0.08
    add(trigger_component, "短線觸發同向" if trigger_component > 0 else "短線觸發逆向", threshold=0.10)

    sr_component = _safe_float(sr_bias, 0.0) * 0.55 * sign
    if direction == "long":
        sr_component += min(0.35, _safe_int(support_hits, 0) * 0.12)
        sr_component -= min(0.35, _safe_int(resistance_hits, 0) * 0.12)
    else:
        sr_component += min(0.35, _safe_int(resistance_hits, 0) * 0.12)
        sr_component -= min(0.35, _safe_int(support_hits, 0) * 0.12)
    sr_component += max(-0.35, min(0.35, _safe_float(repeated_test_pressure, 0.0) * 0.45 * sign))
    add(sr_component, "支撐壓力同向" if sr_component > 0 else "支撐壓力逆向", threshold=0.10)

    flow_component = max(-0.65, min(0.65, _safe_float(derivatives_pressure, 0.0) * 0.65)) * sign
    flow_component += max(-0.35, min(0.35, (taker_buy_ratio - 0.5) * 1.4)) * sign
    add(flow_component, "合約資金流同向" if flow_component > 0 else "合約資金流逆向", threshold=0.10)

    strong_trigger = (
        breakout == sign
        and volume_spike
        and ((direction == "long" and buy_pressure and not sweep_high) or (direction == "short" and sell_pressure and not sweep_low))
    )
    hard_block = False
    min_score = 0.55
    if host_mode in {"trend_pullback_long", "trend_pullback_short"}:
        min_score = 1.15
    elif host_mode in {"support_reclaim", "resistance_rejection"}:
        min_score = 0.85
    elif host_mode in {"breakout_after_pressure_tests", "breakdown_after_support_tests"}:
        min_score = 0.70

    if event_risk >= 2 and score < max(min_score, 1.20) and not strong_trigger:
        hard_block = True
        reasons.append("高時事風險未確認")
    if (
        direction == "short"
        and pos <= 0.30
        and _safe_float(macro_bias, 0.0) > -0.45
        and _safe_float(derivatives_pressure, 0.0) > -0.12
        and _safe_float(news_bias, 0.0) > -0.35
    ):
        hard_block = True
        reasons.append("低位追空缺少宏觀/資金流確認")
    if (
        direction == "short"
        and lower_zone_count >= 2
        and _safe_float(macro_bias, 0.0) > -0.65
        and _safe_float(derivatives_pressure, 0.0) > -0.18
    ):
        hard_block = True
        reasons.append("多週期低位追空未確認")
    if (
        direction == "short"
        and host_mode == "breakdown_after_support_tests"
        and mid_trend != -1
        and not (_safe_float(derivatives_pressure, 0.0) < -0.25 and _safe_float(taker_buy_ratio, 0.5) < 0.46)
    ):
        hard_block = True
        reasons.append("跌破支撐但30m動能未同向")
    if (
        direction == "short"
        and host_mode == "breakdown_after_support_tests"
        and lower_zone_count >= 2
        and not (_safe_float(derivatives_pressure, 0.0) < -0.20 and strong_trigger)
    ):
        hard_block = True
        reasons.append("多週期低位跌破不追空")
    if (
        direction == "long"
        and pos >= 0.70
        and _safe_float(macro_bias, 0.0) < 0.45
        and _safe_float(derivatives_pressure, 0.0) < 0.12
        and _safe_float(news_bias, 0.0) < 0.35
    ):
        hard_block = True
        reasons.append("高位追多缺少宏觀/資金流確認")
    if (
        direction == "long"
        and upper_zone_count >= 2
        and _safe_float(macro_bias, 0.0) < 0.65
        and _safe_float(derivatives_pressure, 0.0) < 0.18
    ):
        hard_block = True
        reasons.append("多週期高位追多未確認")
    if (
        direction == "long"
        and host_mode == "trend_pullback_long"
        and upper_zone_count >= 3
        and not (strong_trigger and _safe_float(derivatives_pressure, 0.0) > 0.20)
    ):
        hard_block = True
        reasons.append("多週期高位趨勢多未確認突破")
    if against >= 4 and aligned <= 2 and not strong_trigger:
        hard_block = True
        reasons.append("多項指標逆向")
    if score < min_score and not strong_trigger:
        hard_block = True
        reasons.append("整體共振不足")

    return {
        "score": round(score, 4),
        "min_score": round(min_score, 4),
        "aligned": aligned,
        "against": against,
        "reasons": reasons[:8],
        "hard_block": bool(hard_block),
        "strong_trigger": bool(strong_trigger),
    }


def build_trade_signal_snapshot(
    *,
    df_4h,
    df_1h,
    df_30m,
    df_15m,
    df_5m,
    price,
    sr_analysis=None,
    sp_change=0.0,
    nq_change=0.0,
    btc_change=0.0,
    dxy_change=0.0,
    news_bias=0.0,
    event_risk=0,
    last_signal=None,
    losing_streak=0,
    min_accept_rr=None,
    min_net_edge_rate=None,
    est_slippage_rate=None,
    est_hold_hours=None,
    fee_round_trip_rate=None,
    funding_rate=None,
    derivatives_flow=None,
    df_1d=None,
    df_1w=None,
    df_1mth=None,
):
    sr_analysis = sr_analysis if isinstance(sr_analysis, dict) else {"bias": 0.0, "support_hits": 0, "resistance_hits": 0, "lines": []}

    if any(frame is None or len(frame) < 30 for frame in (df_4h, df_1h, df_30m, df_15m, df_5m)):
        return {
            "features": {},
            "score": 0.5,
            "final": "觀望",
            "sl": None,
            "tp": None,
            "position_size": 0.0,
            "ai_prob": 0.5,
            "ai_long_prob": 0.5,
            "ai_short_prob": 0.5,
            "macro_bias": 0.0,
            "fake_breakout": False,
            "triangle": 0,
            "fvg_low": None,
            "fvg_high": None,
            "point_explain": "",
            "htf": 0,
            "htf_strength": 0.0,
            "mid_trend": 0,
            "breakout": 0,
            "regime": "range",
            "atr": 0.0,
            "derivatives_pressure": 0.0,
            "open_interest_change": 0.0,
            "mark_premium_rate": 0.0,
            "funding_rate_live": 0.0,
            "taker_buy_ratio": 0.5,
            "rr_at_entry": 0.0,
            "risk_rate": 0.0,
            "reward_rate": 0.0,
            "net_edge_rate_est": 0.0,
            "total_trade_cost_rate_est": 0.0,
            "fee_round_trip_rate": 0.0,
            "funding_cost_rate_est": 0.0,
            "content_override": {"enabled": True, "usable": False, "applied": False},
            "host_opening_logic": {"direction": "neutral", "confidence": 0.0, "reasons": []},
            "host_logic_applied": False,
            "macro_indicator_alignment": {"score": 0.0, "aligned": 0, "against": 0, "reasons": []},
            "market_profile": {"phase": "unknown", "indicator_family": "none"},
        }

    price = _safe_float(price, _safe_float(df_5m["close"].iloc[-1], 0.0))
    derivatives_flow = _normalize_derivatives_flow_snapshot(derivatives_flow)
    higher_timeframe = build_higher_timeframe_context(
        df_4h,
        df_1d,
        df_1w,
        df_1h=df_1h,
        df_15m=df_15m,
        df_1mth=df_1mth,
    )
    market_profile = classify_market_strategy_profile(df_1mth=df_1mth, df_1w=df_1w, df_1d=df_1d, df_4h=df_4h)
    timeframe_patterns = {
        "fifteen_min": _detect_candlestick_pattern(df_15m),
        "one_hour": _detect_candlestick_pattern(df_1h),
        "four_hour": _detect_candlestick_pattern(df_4h),
        "daily": _detect_candlestick_pattern(df_1d),
        "weekly": _detect_candlestick_pattern(df_1w),
    }
    timeframe_kline_view = _build_timeframe_kline_view(higher_timeframe, timeframe_patterns)
    higher_timeframe["timeframe_kline_view"] = timeframe_kline_view
    higher_timeframe["timeframe_kline_summary"] = timeframe_kline_view.get("summary", "")
    candlestick_turning = (
        timeframe_kline_view.get("turning_point")
        if isinstance(timeframe_kline_view.get("turning_point"), dict)
        else {}
    )
    candlestick_turn_score = _safe_float(candlestick_turning.get("score"), 0.0)
    candlestick_turn_confidence = _safe_float(candlestick_turning.get("confidence"), 0.0)
    candlestick_turn_count = _safe_int(candlestick_turning.get("simultaneous_count"), 0)
    candlestick_turn_direction = str(candlestick_turning.get("direction") or "neutral")
    regime = detect_market_regime(df_1h, df_4h)

    trend_4h = df_4h["close"].iloc[-1] - df_4h["ma25"].iloc[-1]
    strength_4h = df_4h["ma25"].iloc[-1] - df_4h["ma25"].iloc[-5]
    htf = 1 if trend_4h > 0 else -1
    htf_strength = abs(strength_4h)

    mid_trend = 1 if df_30m["macd"].iloc[-1] > df_30m["signal"].iloc[-1] else -1
    fvg_low, fvg_high = calc_fvg(df_15m)
    atr = float(df_15m["high"].iloc[-1] - df_15m["low"].iloc[-1]) if len(df_15m) > 0 else 0.0

    recent_high_5m = df_5m["high"].iloc[-5:-1].max()
    recent_low_5m = df_5m["low"].iloc[-5:-1].min()
    breakout = 0
    if price > recent_high_5m:
        breakout = 1
    elif price < recent_low_5m:
        breakout = -1

    prev_high = df_5m["high"].iloc[-2]
    prev_low = df_5m["low"].iloc[-2]
    sweep_high = bool(price > recent_high_5m and df_5m["close"].iloc[-1] < prev_high)
    sweep_low = bool(price < recent_low_5m and df_5m["close"].iloc[-1] > prev_low)

    macro_bias = _compute_macro_bias(sp_change, nq_change, btc_change, dxy_change, news_bias, event_risk)

    recent_high_15 = df_15m["high"].tail(20).max()
    recent_low_15 = df_15m["low"].tail(20).min()
    repeated_level_tests = analyze_repeated_level_tests(
        price,
        df_5m.tail(80),
        df_15m.tail(48),
        recent_low_15,
        recent_high_15,
        atr=atr,
    )
    repeated_support_tests = _safe_int(repeated_level_tests.get("support_tests"), 0)
    repeated_resistance_tests = _safe_int(repeated_level_tests.get("resistance_tests"), 0)
    repeated_test_pressure = _safe_float(repeated_level_tests.get("pressure"), 0.0)
    triangle = detect_triangle(df_15m)

    vol_now = df_15m["volume"].iloc[-1]
    vol_ma = df_15m["vol_ma20"].iloc[-1] if "vol_ma20" in df_15m.columns else df_15m["volume"].rolling(20).mean().iloc[-1]
    volume_spike = bool(vol_now > vol_ma * 1.5)
    volume_ratio = vol_now / (vol_ma + 1e-9)
    buy_pressure = bool(df_15m["close"].iloc[-1] > df_15m["open"].iloc[-1])
    sell_pressure = bool(df_15m["close"].iloc[-1] < df_15m["open"].iloc[-1])

    prev_close = df_15m["close"].iloc[-2]
    absorption = False
    if volume_spike:
        if buy_pressure and price < prev_close:
            absorption = True
        if sell_pressure and price > prev_close:
            absorption = True

    open_interest_change = _safe_float(derivatives_flow.get("open_interest_change"), 0.0)
    mark_premium_rate = _safe_float(derivatives_flow.get("mark_premium_rate"), 0.0)
    funding_rate_live = _safe_float(derivatives_flow.get("funding_rate_live"), 0.0)
    taker_buy_ratio = max(0.0, min(1.0, _safe_float(derivatives_flow.get("taker_buy_ratio"), 0.5)))
    derivatives_pressure = max(-1.0, min(1.0, _safe_float(derivatives_flow.get("derivatives_pressure"), 0.0)))

    features = {
        "htf": htf,
        "htf_strength": htf_strength,
        "mid_trend": mid_trend,
        "macd": df_15m["macd"].iloc[-1],
        "hist": df_15m["macd"].iloc[-1] - df_15m["signal"].iloc[-1],
        "price_vs_ma": df_15m["close"].iloc[-1] - df_15m["ma25"].iloc[-1],
        "breakout": breakout,
        "fvg": (fvg_high - fvg_low) if fvg_low else 0,
        "volatility": df_15m["high"].iloc[-1] - df_15m["low"].iloc[-1],
        "trend_strength": abs(df_15m["ma25"].iloc[-1] - df_15m["ma25"].iloc[-5]),
        "range_pos": (price - recent_low_15) / (recent_high_15 - recent_low_15 + 1e-6),
        "sp": sp_change,
        "nq": nq_change,
        "btc": btc_change,
        "dxy": dxy_change,
        "macro": macro_bias,
        "regime": {
            "bull_trend_strong": 2,
            "bull_trend": 1,
            "range": 0,
            "bear_trend": -1,
            "bear_trend_strong": -2,
        }[regime],
        "triangle": triangle,
        "event_risk": event_risk,
        "volume_spike": int(volume_spike),
        "volume_ratio": volume_ratio,
        "buy_pressure": int(buy_pressure),
        "absorption": int(absorption),
        "sweep_high": int(sweep_high),
        "sweep_low": int(sweep_low),
        "multi_tf_sr_bias": _safe_float(sr_analysis.get("bias"), 0.0),
        "multi_tf_support_hits": _safe_int(sr_analysis.get("support_hits"), 0),
        "multi_tf_resistance_hits": _safe_int(sr_analysis.get("resistance_hits"), 0),
        "repeated_support_tests": repeated_support_tests,
        "repeated_resistance_tests": repeated_resistance_tests,
        "repeated_test_pressure": repeated_test_pressure,
        "candlestick_turn_score": candlestick_turn_score,
        "candlestick_turn_confidence": candlestick_turn_confidence,
        "candlestick_turn_count": candlestick_turn_count,
        "open_interest_change": open_interest_change,
        "mark_premium_rate": mark_premium_rate,
        "funding_rate_live": funding_rate_live,
        "taker_buy_ratio": taker_buy_ratio,
        "derivatives_pressure": derivatives_pressure,
    }

    ai_prob = 0.5
    ai_long_prob = 0.5
    ai_short_prob = 0.5
    try:
        ai_prob, ai_long_prob, ai_short_prob = _predict_directional_trade_bias(features)
    except Exception:
        ai_prob = 0.5
        ai_long_prob = 0.5
        ai_short_prob = 0.5

    if abs(ai_prob - 0.5) < 0.05:
        rule_score = 0.0
        rule_score += 0.25 if htf == 1 else -0.25
        rule_score += 0.15 if mid_trend == 1 else -0.15
        if breakout == 1:
            rule_score += 0.25
        elif breakout == -1:
            rule_score -= 0.25
        rule_score += max(-0.25, min(0.25, repeated_test_pressure))
        if _is_truthy(os.getenv("TRADE_USE_MULTI_TF_CANDLE_TURNING", "1")) and candlestick_turn_count >= 2:
            rule_score += max(-0.18, min(0.18, candlestick_turn_score * candlestick_turn_confidence))
        rule_score += macro_bias * 0.1
        if triangle == 1:
            rule_score += 0.05
        ai_prob = 0.5 + rule_score

    ai_prob = max(0.05, min(ai_prob, 0.95))
    score = 0.75 * ai_prob + 0.25 * last_signal if last_signal is not None else ai_prob

    confluence = 0.0
    score_direction = 1 if score >= 0.5 else -1
    confluence += 1.0 if mid_trend == score_direction else -0.7
    if breakout == score_direction:
        confluence += 0.9
    elif breakout == -score_direction:
        confluence -= 0.9
    if score_direction == 1 and repeated_resistance_tests >= 2:
        if breakout == 1 and (volume_spike or buy_pressure) and not sweep_high:
            confluence += min(1.2, repeated_resistance_tests * 0.25)
        else:
            confluence -= min(0.7, repeated_resistance_tests * 0.16)
    elif score_direction == -1 and repeated_support_tests >= 2:
        if breakout == -1 and (volume_spike or sell_pressure) and not sweep_low:
            confluence += min(1.2, repeated_support_tests * 0.25)
        else:
            confluence -= min(0.7, repeated_support_tests * 0.16)
    elif score_direction == 1 and repeated_support_tests >= 2:
        confluence -= min(0.8, repeated_support_tests * 0.18)
    elif score_direction == -1 and repeated_resistance_tests >= 2:
        confluence -= min(0.8, repeated_resistance_tests * 0.18)
    confluence += 0.7 if htf == score_direction else -0.4
    score += confluence * 0.06

    if volume_spike:
        if buy_pressure:
            score += 0.06
        elif sell_pressure:
            score -= 0.06

    if absorption:
        score *= 0.9
    if sweep_high:
        score -= 0.12
    if sweep_low:
        score += 0.12

    if regime == "bull_trend_strong":
        score += 0.25
    elif regime == "bull_trend":
        score += 0.12
    elif regime == "bear_trend_strong":
        score -= 0.25
    elif regime == "bear_trend":
        score -= 0.12

    if regime == "range":
        score = 0.5 + (score - 0.5) * 0.4

    if macro_bias > 0.6 and score > 0.5:
        score += 0.06
    elif macro_bias < -0.6 and score < 0.5:
        score -= 0.06
    else:
        score = 0.5 + (score - 0.5) * 0.92

    score = max(0.05, min(score, 0.95))

    if news_bias:
        score += max(-0.10, min(0.10, _safe_float(news_bias, 0.0) * 0.04))
        score = max(0.05, min(score, 0.95))

    sr_bias = _safe_float(sr_analysis.get("bias"), 0.0)
    score += max(-0.14, min(0.14, sr_bias * 0.10))
    score += max(-0.18, min(0.18, repeated_test_pressure))
    if (
        _is_truthy(os.getenv("TRADE_USE_MULTI_TF_CANDLE_TURNING", "1"))
        and candlestick_turn_count >= 2
        and candlestick_turn_confidence >= max(
            0.42,
            min(0.80, _safe_float(os.getenv("TRADE_MULTI_TF_CANDLE_TURN_MIN_CONF", 0.48), 0.48)),
        )
    ):
        turn_adjust = max(
            -0.10,
            min(
                0.10,
                candlestick_turn_score
                * candlestick_turn_confidence
                * _safe_float(os.getenv("TRADE_MULTI_TF_CANDLE_TURN_WEIGHT", 0.16), 0.16),
            ),
        )
        score += turn_adjust
    score = max(0.05, min(score, 0.95))

    if abs(derivatives_pressure) >= 0.10:
        score += max(-0.06, min(0.06, derivatives_pressure * 0.045))
        score = max(0.05, min(score, 0.95))

    profile_direction = "long" if score >= 0.5 else "short"
    market_profile_adjustment = _market_profile_score_adjustment(
        market_profile,
        profile_direction,
        price=price,
        df_15m=df_15m,
        df_4h=df_4h,
    )
    score += _safe_float(market_profile_adjustment.get("adjustment"), 0.0)
    score = max(0.05, min(score, 0.95))

    auxiliary_score = score
    host_opening_logic = _score_host_opening_logic(
        price=price,
        timeframe_kline_view=timeframe_kline_view,
        range_pos=features.get("range_pos"),
        htf=htf,
        mid_trend=mid_trend,
        breakout=breakout,
        regime=regime,
        volume_spike=volume_spike,
        buy_pressure=buy_pressure,
        sell_pressure=sell_pressure,
        sweep_high=sweep_high,
        sweep_low=sweep_low,
        support_hits=_safe_int(sr_analysis.get("support_hits"), 0),
        resistance_hits=_safe_int(sr_analysis.get("resistance_hits"), 0),
        repeated_support_tests=repeated_support_tests,
        repeated_resistance_tests=repeated_resistance_tests,
        repeated_test_pressure=repeated_test_pressure,
        macro_bias=macro_bias,
    )
    host_logic_applied = False
    if _is_truthy(os.getenv("TRADE_HOST_OPENING_LOGIC_ENABLED", "1")):
        host_direction = str(host_opening_logic.get("direction") or "neutral")
        host_confidence = max(0.0, min(0.88, _safe_float(host_opening_logic.get("confidence"), 0.0)))
        if host_direction in {"long", "short"} and host_confidence >= max(
            0.38,
            min(0.70, _safe_float(os.getenv("TRADE_HOST_OPENING_LOGIC_MIN_CONF", 0.42), 0.42)),
        ):
            max_swing = max(
                0.12,
                min(0.36, _safe_float(os.getenv("TRADE_HOST_OPENING_LOGIC_MAX_SWING", 0.30), 0.30)),
            )
            target_score = 0.5 + (max_swing * host_confidence if host_direction == "long" else -max_swing * host_confidence)
            blend = max(0.25, min(0.82, _safe_float(os.getenv("TRADE_HOST_OPENING_LOGIC_BLEND", 0.62), 0.62)))
            score = score * (1.0 - blend) + target_score * blend
            score = max(0.05, min(score, 0.95))
            host_logic_applied = True
            prob_floor = max(0.46, min(0.80, 0.50 + host_confidence * 0.28))
            if host_direction == "long":
                ai_long_prob = max(ai_long_prob, prob_floor)
                ai_prob = max(ai_prob, score)
            else:
                ai_short_prob = max(ai_short_prob, prob_floor)
                ai_prob = min(ai_prob, score)

    learned_entry_logic = _score_mlx_learned_entry_logic(
        score,
        range_pos=features.get("range_pos"),
        htf=htf,
        mid_trend=mid_trend,
        breakout=breakout,
        regime=regime,
        volume_spike=volume_spike,
        buy_pressure=buy_pressure,
        sell_pressure=sell_pressure,
        sweep_high=sweep_high,
        sweep_low=sweep_low,
        support_hits=_safe_int(sr_analysis.get("support_hits"), 0),
        resistance_hits=_safe_int(sr_analysis.get("resistance_hits"), 0),
        repeated_support_tests=repeated_support_tests,
        repeated_resistance_tests=repeated_resistance_tests,
        repeated_test_pressure=repeated_test_pressure,
        macro_bias=macro_bias,
        derivatives_pressure=derivatives_pressure,
    )
    score = _safe_float(learned_entry_logic.get("score"), score)
    learned_logic_confidence = max(0.0, min(0.82, _safe_float(learned_entry_logic.get("confidence"), 0.0)))
    learned_logic_direction = str(learned_entry_logic.get("direction") or "")
    if learned_logic_direction == "long":
        ai_long_prob = max(ai_long_prob, learned_logic_confidence)
        ai_prob = max(ai_prob, score)
    elif learned_logic_direction == "short":
        ai_short_prob = max(ai_short_prob, learned_logic_confidence)
        ai_prob = min(ai_prob, score)
    raw_content_override = _load_binance_host_content_override_signal()
    content_override = _assess_host_content_override(
        raw_content_override,
        htf=htf,
        mid_trend=mid_trend,
        macro_bias=macro_bias,
        sr_bias=sr_bias,
        support_hits=_safe_int(sr_analysis.get("support_hits"), 0),
        resistance_hits=_safe_int(sr_analysis.get("resistance_hits"), 0),
        derivatives_pressure=derivatives_pressure,
    )
    if content_override.get("applied"):
        override_direction = str(content_override.get("direction") or "")
        override_quality = max(0.0, min(1.0, _safe_float(content_override.get("quality"), 0.0)))
        primary_mode = content_override.get("mode") == "primary"
        max_swing_default = 0.34 if primary_mode else 0.22
        max_swing = max(
            0.08,
            min(
                0.40,
                _safe_float(
                    os.getenv("TRADE_HOST_CONTENT_OVERRIDE_MAX_SCORE_SWING", max_swing_default),
                    max_swing_default,
                ),
            ),
        )
        target_score = 0.5 + (max_swing * override_quality if override_direction == "long" else -max_swing * override_quality)
        if primary_mode:
            aux_cap = max(
                0.0,
                min(
                    0.10,
                    _safe_float(os.getenv("TRADE_HOST_CONTENT_AUXILIARY_SCORE_CAP", 0.06), 0.06),
                ),
            )
            auxiliary_adjustment = max(-aux_cap, min(aux_cap, auxiliary_score - 0.5))
            score = target_score + auxiliary_adjustment
        else:
            blend = max(0.15, min(0.75, _safe_float(os.getenv("TRADE_HOST_CONTENT_OVERRIDE_BLEND", 0.55), 0.55)))
            score = score * (1.0 - blend) + target_score * blend
        score = max(0.05, min(score, 0.95))
        prob_floor = max(0.42, min(0.82, 0.52 + override_quality * (0.22 if primary_mode else 0.18)))
        if override_direction == "long":
            ai_long_prob = max(ai_long_prob, prob_floor)
            ai_prob = max(ai_prob, min(0.90, score))
        elif override_direction == "short":
            ai_short_prob = max(ai_short_prob, prob_floor)
            ai_prob = min(ai_prob, max(0.10, score))

    primary_indicator = (
        "mlx_host_strategy"
        if content_override.get("applied")
        else "mlx_opening_logic"
        if host_logic_applied
        else "mlx_learned_logic"
    )

    if _is_truthy(os.getenv("TRADE_USE_CONFLUENCE_PROB_FLOOR", "1")):
        long_confluence = 0.0
        short_confluence = 0.0
        if htf == 1:
            long_confluence += 1.0
        else:
            short_confluence += 1.0
        if mid_trend == 1:
            long_confluence += 1.0
        else:
            short_confluence += 1.0
        if breakout == 1:
            long_confluence += 1.0
        elif breakout == -1:
            short_confluence += 1.0
        if regime == "bull_trend_strong":
            long_confluence += 1.2
        elif regime == "bull_trend":
            long_confluence += 0.7
        elif regime == "bear_trend_strong":
            short_confluence += 1.2
        elif regime == "bear_trend":
            short_confluence += 0.7
        if macro_bias > 0.6:
            long_confluence += 0.7
        elif macro_bias < -0.6:
            short_confluence += 0.7
        if sr_bias > 0.10:
            long_confluence += 0.4
        elif sr_bias < -0.10:
            short_confluence += 0.4
        if repeated_resistance_tests >= 2:
            if learned_entry_logic.get("resistance_break_confirmed"):
                long_confluence += min(1.5, repeated_resistance_tests * 0.3)
            else:
                short_confluence += min(0.8, repeated_resistance_tests * 0.18)
        if repeated_support_tests >= 2:
            if learned_entry_logic.get("support_break_confirmed"):
                short_confluence += min(1.5, repeated_support_tests * 0.3)
            else:
                long_confluence += min(0.8, repeated_support_tests * 0.18)
        if derivatives_pressure > 0.12:
            long_confluence += 0.4
        elif derivatives_pressure < -0.12:
            short_confluence += 0.4
        if volume_spike and buy_pressure:
            long_confluence += 0.4
        elif volume_spike and sell_pressure:
            short_confluence += 0.4
        if candlestick_turn_count >= 2 and candlestick_turn_confidence >= 0.48:
            if candlestick_turn_direction == "long":
                long_confluence += min(0.7, 0.25 + candlestick_turn_confidence * 0.45)
            elif candlestick_turn_direction == "short":
                short_confluence += min(0.7, 0.25 + candlestick_turn_confidence * 0.45)
        if score >= 0.80:
            long_confluence += 1.0
        elif score <= 0.20:
            short_confluence += 1.0

        confluence_trigger = max(3.5, _safe_float(os.getenv("TRADE_CONFLUENCE_PROB_FLOOR_TRIGGER", 4.0), 4.0))
        base_floor = max(0.42, min(0.75, _safe_float(os.getenv("TRADE_CONFLUENCE_PROB_FLOOR_BASE", 0.52), 0.52)))
        if long_confluence >= confluence_trigger and score >= 0.80:
            confluence_floor = min(0.72, base_floor + (long_confluence - confluence_trigger) * 0.04)
            ai_long_prob = max(ai_long_prob, confluence_floor)
        if short_confluence >= confluence_trigger and score <= 0.20:
            confluence_floor = min(0.72, base_floor + (short_confluence - confluence_trigger) * 0.04)
            ai_short_prob = max(ai_short_prob, confluence_floor)

    entry = price
    min_accept_rr = max(1.1, _safe_float(min_accept_rr if min_accept_rr is not None else os.getenv("TRADE_MIN_ACCEPT_RR", 1.8), 1.8))
    min_net_edge_rate = max(0.0005, _safe_float(min_net_edge_rate if min_net_edge_rate is not None else os.getenv("TRADE_MIN_NET_EDGE_RATE", 0.0012), 0.0012))
    min_entry_risk_rate = max(0.001, _safe_float(os.getenv("TRADE_MIN_ENTRY_RISK_RATE", 0.003), 0.003))
    min_direction_win_prob = max(0.05, min(0.75, _safe_float(os.getenv("TRADE_MIN_DIRECTION_WIN_PROB", 0.42), 0.42)))
    conflict_min_edge_rate = max(min_net_edge_rate, _safe_float(os.getenv("TRADE_CONFLICT_MIN_EDGE_RATE", 0.0025), 0.0025))
    conflict_short_max_score = min(0.45, _safe_float(os.getenv("TRADE_CONFLICT_SHORT_MAX_SCORE", 0.20), 0.20))
    conflict_long_min_score = max(0.55, _safe_float(os.getenv("TRADE_CONFLICT_LONG_MIN_SCORE", 0.80), 0.80))
    est_slippage_rate = max(0.0, _safe_float(est_slippage_rate if est_slippage_rate is not None else os.getenv("TRADE_EST_SLIPPAGE_RATE", 0.0004), 0.0004))
    est_hold_hours = max(0.0, _safe_float(est_hold_hours if est_hold_hours is not None else os.getenv("TRADE_EST_HOLD_HOURS", 6.0), 6.0))
    fee_round_trip_rate = max(0.0, _safe_float(fee_round_trip_rate if fee_round_trip_rate is not None else POSITION_PANEL_STATE.get("fee_round_trip_rate"), 0.001))
    resolved_funding_rate = funding_rate if funding_rate is not None else (
        funding_rate_live if funding_rate_live else POSITION_PANEL_STATE.get("funding_rate")
    )
    funding_rate_abs = abs(_safe_float(resolved_funding_rate, 0.0))
    funding_cost_rate_est = max(0.0, funding_rate_abs * max(est_hold_hours / 8.0, 0.0))
    total_trade_cost_rate_est = fee_round_trip_rate + est_slippage_rate + funding_cost_rate_est

    final = "觀望"
    sl = None
    tp = None
    position_size = 0.0
    macro_indicator_alignment = {"score": 0.0, "aligned": 0, "against": 0, "reasons": []}
    rr_at_entry = 0.0
    risk_rate = 0.0
    reward_rate = 0.0
    net_edge_rate_est = 0.0
    rsi_15m = max(0.0, min(100.0, _safe_float(df_15m["rsi14"].iloc[-1], 50.0)))
    ema50_15m = max(1e-9, _safe_float(df_15m["ema50"].iloc[-1], price))
    ema50_deviation = (price - ema50_15m) / ema50_15m

    pullback_long = bool(regime in ["bear_trend", "bear_trend_strong"] and mid_trend == 1 and (volume_spike or breakout == 1))
    pullback_short = bool(regime in ["bull_trend", "bull_trend_strong"] and mid_trend == -1 and (volume_spike or breakout == -1))

    fake_breakout = False
    if breakout != 0 and not volume_spike:
        fake_breakout = True
    if absorption or sweep_high or sweep_low:
        fake_breakout = True
    if breakout == 1 and btc_change < 0:
        fake_breakout = True
    if breakout == -1 and btc_change > 0:
        fake_breakout = True

    entry_threshold = max(0.08, 0.14 - _safe_int(event_risk, 0) * 0.02)
    if regime == "range":
        entry_threshold += 0.05

    range_pos = max(0.0, min(1.0, _safe_float(features.get("range_pos"), 0.5)))
    support_hits = _safe_int(sr_analysis.get("support_hits"), 0)
    resistance_hits = _safe_int(sr_analysis.get("resistance_hits"), 0)
    if regime == "range" and breakout == 0:
        range_low = _safe_float(df_15m["low"].tail(20).min(), entry)
        range_high = _safe_float(df_15m["high"].tail(20).max(), entry)
        range_buffer = max(atr * 0.35, entry * 0.001)
        if range_pos <= 0.28 and (buy_pressure or sweep_low or support_hits > 0):
            final = "↔️ 震盪下緣做多"
            sl = range_low - range_buffer
            tp = range_high - range_buffer * 0.5
            position_size = _cap_initial_position_size(
                max(0.01, _safe_float(os.getenv("TRADE_RANGE_POSITION_SIZE", 0.05), 0.05))
            )
        elif range_pos >= 0.72 and (sell_pressure or sweep_high or resistance_hits > 0):
            final = "↔️ 震盪上緣做空"
            sl = range_high + range_buffer
            tp = range_low + range_buffer * 0.5
            position_size = _cap_initial_position_size(
                max(0.01, _safe_float(os.getenv("TRADE_RANGE_POSITION_SIZE", 0.05), 0.05))
            )

    if final.startswith("觀望") and abs(score - 0.5) > entry_threshold:
        if abs(score - 0.5) < 0.12:
            final = "觀望（低信心）"

        if triangle == 1:
            upper = df_15m["high"].tail(20).max()
            lower = df_15m["low"].tail(20).min()
            range_size = upper - lower

            if price > upper - range_size * 0.2 and breakout == 0:
                final = "🔺 三角上緣做空"
                sl = upper
                risk = sl - entry
                tp = entry - risk * 1.8
            elif price < lower + range_size * 0.2 and breakout == 0:
                final = "🔻 三角下緣做多"
                sl = lower
                risk = entry - sl
                tp = entry + risk * 1.8
            elif breakout == 1:
                final = "🚀 三角突破做多"
                sl = lower
                risk = entry - sl
                tp = entry + risk * 2.5
            elif breakout == -1:
                final = "🚀 三角跌破做空"
                sl = upper
                risk = sl - entry
                tp = entry - risk * 2.5
        else:
            if score > 0.52:
                final = "🚀 做多"
                recent_low_pb = df_15m["low"].tail(10).min()
                sl = recent_low_pb
                risk = entry - sl
                # Backtests show normal trends also need enough extension to cover
                # false starts and costs; 2.0R was negative across split windows.
                rr = (
                    2.6
                    if regime.endswith("strong")
                    else max(
                        2.0,
                        _safe_float(os.getenv("TRADE_NORMAL_TREND_RR", 2.6), 2.6),
                    )
                )
                tp = entry + risk * rr
            elif score < 0.48:
                final = "🚀 做空"
                recent_high_pb = df_15m["high"].tail(10).max()
                sl = recent_high_pb
                risk = sl - entry
                rr = (
                    2.6
                    if regime.endswith("strong")
                    else max(
                        2.0,
                        _safe_float(os.getenv("TRADE_NORMAL_TREND_RR", 2.6), 2.6),
                    )
                )
                tp = entry - risk * rr
            elif pullback_long and score >= 0.45:
                final = "↩️ 反彈做多"
                recent_low_pb = min(df_5m["low"].tail(6).min(), df_15m["low"].tail(6).min())
                sl = recent_low_pb
                risk = max(entry - sl, atr * 0.45, entry * 0.001)
                tp = entry + risk * 1.8
            elif pullback_short and score <= 0.55:
                final = "↩️ 反彈做空"
                recent_high_pb = max(df_5m["high"].tail(6).max(), df_15m["high"].tail(6).max())
                sl = recent_high_pb
                risk = max(sl - entry, atr * 0.45, entry * 0.001)
                tp = entry - risk * 1.8

            confidence = abs(score - 0.5) * 2
            if regime in ["bull_trend_strong", "bear_trend_strong"]:
                base = 0.35
            elif regime in ["bull_trend", "bear_trend"]:
                base = 0.25
            else:
                base = 0.15

            if confidence > 0.7:
                position_size = base
            elif confidence > 0.5:
                position_size = base * 0.7
            elif confidence > 0.3:
                position_size = base * 0.5
            else:
                position_size = base * 0.3

            if losing_streak >= 3:
                position_size *= 0.5

            if "三角" in final:
                if "突破" in final:
                    position_size *= 1.2
                else:
                    position_size *= 0.7

            counter_trend_entry = (
                ("做多" in final and regime in {"bear_trend", "bear_trend_strong"})
                or ("做空" in final and regime in {"bull_trend", "bull_trend_strong"})
            )
            if counter_trend_entry:
                position_size = min(
                    position_size,
                    max(0.01, _safe_float(os.getenv("TRADE_COUNTER_TREND_MAX_SIZE_RATIO", 0.02), 0.02)),
                )
            turn_opposes_entry = (
                candlestick_turn_count >= 2
                and candlestick_turn_confidence >= 0.60
                and (
                    ("做多" in final and candlestick_turn_direction == "short")
                    or ("做空" in final and candlestick_turn_direction == "long")
                )
            )
            if turn_opposes_entry:
                position_size = min(
                    position_size,
                    max(0.01, _safe_float(os.getenv("TRADE_OPPOSING_TURN_MAX_SIZE_RATIO", 0.02), 0.02)),
                )

            position_size = _cap_initial_position_size(position_size)

    if not final.startswith("觀望"):
        final, sl, tp = auto_fix_trade_plan(final, entry, sl, tp, atr)

    point_explain = ""
    if not final.startswith("觀望") and sl is not None and tp is not None and entry > 0:
        sr_bias = _safe_float(sr_analysis.get("bias"), 0.0)
        support_hits = _safe_int(sr_analysis.get("support_hits"), 0)
        resistance_hits = _safe_int(sr_analysis.get("resistance_hits"), 0)
        if "做多" in final:
            risk_rate = max((entry - sl) / entry, 1e-9)
            reward_rate = max((tp - entry) / entry, 0.0)
            direction_win_prob = max(0.05, min(0.95, _safe_float(ai_long_prob, ai_prob)))
        else:
            risk_rate = max((sl - entry) / entry, 1e-9)
            reward_rate = max((entry - tp) / entry, 0.0)
            direction_win_prob = max(0.05, min(0.95, _safe_float(ai_short_prob, 1.0 - ai_prob)))

        rr_at_entry = reward_rate / max(risk_rate, 1e-9)
        min_reward_rate_needed = total_trade_cost_rate_est + min_net_edge_rate
        net_edge_rate_est = (
            direction_win_prob * reward_rate
            - (1.0 - direction_win_prob) * risk_rate
            - total_trade_cost_rate_est
        )
        point_explain = (
            f"📐 點位計算: SL=近10根高低點/結構位，TP=風險×RR\n"
            f"成本估算: 手續費{fee_round_trip_rate*100:.3f}% + 滑價{est_slippage_rate*100:.3f}% + 資金費{funding_cost_rate_est*100:.3f}% ≈ {total_trade_cost_rate_est*100:.3f}%\n"
            f"風險/報酬: risk={risk_rate*100:.3f}% | reward={reward_rate*100:.3f}% | RR={rr_at_entry:.2f} | AI期望值={net_edge_rate_est*100:.3f}%"
        )
        if rr_at_entry < min_accept_rr:
            final = "觀望（RR不足）"
        elif risk_rate < min_entry_risk_rate:
            final = "觀望（停損距離過近）"
        elif direction_win_prob < min_direction_win_prob:
            final = "觀望（方向勝率不足）"
        elif reward_rate < min_reward_rate_needed:
            final = "觀望（報酬不足覆蓋成本）"
        elif _is_truthy(os.getenv("TRADE_REQUIRE_AI_EDGE", "1")):
            min_expected_edge_rate = max(
                0.0002,
                _safe_float(os.getenv("TRADE_MIN_EXPECTED_EDGE_RATE", min_net_edge_rate), min_net_edge_rate),
            )
            if net_edge_rate_est < min_expected_edge_rate:
                final = "觀望（AI期望值不足）"
        if (
            not final.startswith("觀望")
            and "做多" in final
            and _is_truthy(os.getenv("TRADE_TIGHTEN_LOW_WINRATE_LONGS", "1"))
            and not content_override.get("applied")
        ):
            long_setup = _safe_float(learned_entry_logic.get("long_setup"), 0.0)
            short_setup = _safe_float(learned_entry_logic.get("short_setup"), 0.0)
            support_confirm = bool(learned_entry_logic.get("support_hold_unconfirmed")) or bool(sweep_low)
            breakout_confirm = breakout == 1 and (volume_spike or buy_pressure) and not sweep_high
            min_long_prob = max(
                min_direction_win_prob,
                _safe_float(os.getenv("TRADE_TIGHT_LONG_MIN_PROB", 0.48), 0.48),
            )
            min_long_setup_edge = max(
                0.15,
                _safe_float(os.getenv("TRADE_TIGHT_LONG_MIN_SETUP_EDGE", 0.35), 0.35),
            )
            if direction_win_prob < min_long_prob:
                final = "觀望（多單歷史勝率偏低）"
            elif long_setup < short_setup + min_long_setup_edge and not (support_confirm or breakout_confirm):
                final = "觀望（多單缺少支撐承接或突破確認）"
            elif resistance_hits >= 1 and not breakout_confirm and score < 0.72:
                final = "觀望（多單壓力未突破）"
        if not final.startswith("觀望"):
            if "做空" in final:
                conflict_count = 0
                if mid_trend == 1:
                    conflict_count += 1
                if macro_bias > 0.4:
                    conflict_count += 1
                if sr_bias > 0.12 or support_hits >= 1:
                    conflict_count += 1
                if derivatives_pressure > 0.12:
                    conflict_count += 1
                if conflict_count >= 2 and (score > conflict_short_max_score or net_edge_rate_est < conflict_min_edge_rate):
                    final = "觀望（空單逆向共振不足）"
            elif "做多" in final:
                conflict_count = 0
                if mid_trend == -1:
                    conflict_count += 1
                if macro_bias < -0.4:
                    conflict_count += 1
                if sr_bias < -0.12 or resistance_hits >= 1:
                    conflict_count += 1
                if derivatives_pressure < -0.12:
                    conflict_count += 1
                if conflict_count >= 2 and (score < conflict_long_min_score or net_edge_rate_est < conflict_min_edge_rate):
                    final = "觀望（多單逆向共振不足）"
        if (
            not final.startswith("觀望")
            and _is_truthy(os.getenv("TRADE_USE_30M_MACD_PRIMARY", "0"))
            and not content_override.get("applied")
        ):
            if "做多" in final and mid_trend != 1:
                final = "觀望（主指標30m MACD未支持做多）"
            elif "做空" in final and mid_trend != -1:
                final = "觀望（主指標30m MACD未支持做空）"
        macro_indicator_alignment = {"score": 0.0, "aligned": 0, "against": 0, "reasons": []}
        if (
            not final.startswith("觀望")
            and _is_truthy(os.getenv("TRADE_USE_MACRO_INDICATOR_ALIGNMENT_GATE", "1"))
            and not content_override.get("applied")
        ):
            direction_name = "long" if "做多" in final else "short"
            macro_indicator_alignment = _score_macro_indicator_alignment(
                direction=direction_name,
                host_mode=str(host_opening_logic.get("mode") or "wait"),
                sp_change=sp_change,
                nq_change=nq_change,
                btc_change=btc_change,
                dxy_change=dxy_change,
                news_bias=news_bias,
                event_risk=event_risk,
                macro_bias=macro_bias,
                htf=htf,
                mid_trend=mid_trend,
                breakout=breakout,
                regime=regime,
                sr_bias=sr_bias,
                support_hits=support_hits,
                resistance_hits=resistance_hits,
                repeated_test_pressure=repeated_test_pressure,
                derivatives_pressure=derivatives_pressure,
                taker_buy_ratio=taker_buy_ratio,
                volume_spike=volume_spike,
                buy_pressure=buy_pressure,
                sell_pressure=sell_pressure,
                sweep_high=sweep_high,
                sweep_low=sweep_low,
                range_pos=features.get("range_pos"),
                timeframe_kline_view=timeframe_kline_view,
            )
            if macro_indicator_alignment.get("hard_block"):
                final = "觀望（MLX宏觀指標共振不足）"
                position_size = 0.0
        if (
            not final.startswith("觀望")
            and _is_truthy(os.getenv("TRADE_USE_MLX_BACKTEST_PROFILE_FILTER", "1"))
            and not content_override.get("applied")
        ):
            host_mode = str(host_opening_logic.get("mode") or "wait")
            direction_name = "long" if "做多" in final else "short"
            profile_ok = False
            if (
                host_mode == "breakdown_after_support_tests"
                and direction_name == "short"
                and regime in {"bear_trend", "bear_trend_strong", "bull_trend"}
            ):
                profile_ok = True
            elif host_mode == "trend_pullback_long" and direction_name == "long" and regime == "bull_trend_strong":
                profile_ok = True
            elif (
                regime == "range"
                and direction_name == "long"
                and host_mode in {"support_reclaim", "breakout_after_pressure_tests"}
            ):
                profile_ok = True
            elif (
                regime == "range"
                and direction_name == "short"
                and host_mode in {"resistance_rejection", "breakdown_after_support_tests"}
            ):
                profile_ok = True
            if not profile_ok:
                final = "觀望（MLX回測輪廓不佳）"
                position_size = 0.0
    final = _classify_wait_state(
        final,
        repeated_support_tests=repeated_support_tests,
        repeated_resistance_tests=repeated_resistance_tests,
        learned_entry_logic=learned_entry_logic,
        breakout=breakout,
        volume_spike=volume_spike,
    )
    return {
        "features": features,
        "score": score,
        "auxiliary_score": auxiliary_score,
        "primary_indicator": primary_indicator,
        "final": final,
        "sl": sl,
        "tp": tp,
        "position_size": position_size,
        "ai_prob": ai_prob,
        "ai_long_prob": ai_long_prob,
        "ai_short_prob": ai_short_prob,
        "macro_bias": macro_bias,
        "fake_breakout": fake_breakout,
        "triangle": triangle,
        "fvg_low": fvg_low,
        "fvg_high": fvg_high,
        "point_explain": point_explain,
        "htf": htf,
        "htf_strength": htf_strength,
        "mid_trend": mid_trend,
        "breakout": breakout,
        "regime": regime,
        "atr": atr,
        "volume_spike": volume_spike,
        "volume_ratio": volume_ratio,
        "buy_pressure": buy_pressure,
        "sell_pressure": sell_pressure,
        "absorption": absorption,
        "sweep_high": sweep_high,
        "sweep_low": sweep_low,
        "candlestick_turning": candlestick_turning,
        "candlestick_turn_score": candlestick_turn_score,
        "candlestick_turn_confidence": candlestick_turn_confidence,
        "candlestick_turn_count": candlestick_turn_count,
        "entry_threshold": entry_threshold,
        "pullback_long": pullback_long,
        "pullback_short": pullback_short,
        "total_trade_cost_rate_est": total_trade_cost_rate_est,
        "fee_round_trip_rate": fee_round_trip_rate,
        "funding_cost_rate_est": funding_cost_rate_est,
        "rr_at_entry": rr_at_entry,
        "risk_rate": risk_rate,
        "reward_rate": reward_rate,
        "net_edge_rate_est": net_edge_rate_est,
        "rsi_15m": rsi_15m,
        "ema50_deviation_15m": ema50_deviation,
        "open_interest_change": open_interest_change,
        "mark_premium_rate": mark_premium_rate,
        "funding_rate_live": funding_rate_live,
        "taker_buy_ratio": taker_buy_ratio,
        "derivatives_pressure": derivatives_pressure,
        "derivatives_flow_stale": bool(derivatives_flow.get("stale", False)),
        "content_override": content_override,
        "host_opening_logic": host_opening_logic,
        "host_logic_applied": bool(host_logic_applied),
        "macro_indicator_alignment": macro_indicator_alignment,
        "learned_entry_logic": learned_entry_logic,
        "market_profile": market_profile,
        "market_profile_phase": market_profile.get("phase"),
        "market_profile_indicator_family": market_profile.get("indicator_family"),
        "market_profile_adjustment": market_profile_adjustment,
        "timeframe_kline_view": timeframe_kline_view,
        "timeframe_kline_summary": timeframe_kline_view.get("summary", ""),
        "higher_timeframe": higher_timeframe,
        "support_hits": _safe_int(sr_analysis.get("support_hits"), 0),
        "resistance_hits": _safe_int(sr_analysis.get("resistance_hits"), 0),
        "repeated_support_tests": repeated_support_tests,
        "repeated_resistance_tests": repeated_resistance_tests,
        "repeated_test_pressure": repeated_test_pressure,
    }


def load_model():
    global model, online_model, online_scaler, online_initialized, online_sample_count, online_label_counts

    _ensure_ai_learning_pipeline_version()
    batch_rebuild_reason = ""
    online_rebuild_reason = ""

    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                with open(MODEL_PATH, "rb") as f:
                    model = pickle.load(f)
        except Exception as e:
            model = None
            batch_rebuild_reason = str(e)

    if model is not None and not _has_expected_feature_schema(model):
        model = None
        batch_rebuild_reason = "批次模型特徵欄位為舊版"

    if model is None and batch_rebuild_reason:
        _try_rebuild_batch_model_from_history(min_rows=10, reason=batch_rebuild_reason)

    online_loaded = False
    _reset_online_learning_state()
    if os.path.exists(ONLINE_MODEL_PATH):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                with open(ONLINE_MODEL_PATH, "rb") as f:
                    online_model = pickle.load(f)
            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                with open(ONLINE_SCALER_PATH, "rb") as f:
                    online_scaler = pickle.load(f)
            if os.path.exists(ONLINE_MODEL_META_PATH):
                with open(ONLINE_MODEL_META_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                online_sample_count = max(0, _safe_int(meta.get("sample_count"), 0))
                label_counts = meta.get("label_counts") if isinstance(meta, dict) else {}
                online_label_counts = {
                    0: max(0, _safe_int((label_counts or {}).get("0"), 0)),
                    1: max(0, _safe_int((label_counts or {}).get("1"), 0)),
                }
            if not _has_expected_feature_schema(online_model):
                raise ValueError("online_model 特徵欄位為舊版")
            if list(getattr(online_scaler, "feature_names_in_", [])) != MODEL_FEATURE_COLUMNS:
                raise ValueError("online_scaler 特徵欄位為舊版")
            online_initialized = True
            online_loaded = True
        except Exception as e:
            online_rebuild_reason = str(e)
            _reset_online_learning_state()

    if not online_loaded:
        rebuilt_online = _bootstrap_online_model_from_history()
        if rebuilt_online and online_rebuild_reason:
            print(f"♻️ online model 已改用當前環境重建: {online_rebuild_reason}")

    # 加載新聞模型
    load_news_model()

def update_online_model(features, label):
    global online_model, online_initialized, online_sample_count, online_label_counts

    clean_label = 1 if int(label) > 0 else 0
    payload = _normalize_feature_payload(features)
    X = pd.DataFrame([payload])
    y = np.array([clean_label])
    sample_weight = np.array([_compute_online_sample_weight(clean_label)], dtype=float)

    try:
        X = _prepare_online_feature_frame(X, fit_scaler=True)
        if not online_initialized:
            online_model.partial_fit(X, y, classes=np.array([0, 1]), sample_weight=sample_weight)
            online_initialized = True
        else:
            online_model.partial_fit(X, y, sample_weight=sample_weight)
    except Exception as e:
        print("⚠️ online_model error, reset model:", e)
        _reset_online_learning_state()
        X = _prepare_online_feature_frame(pd.DataFrame([payload]), fit_scaler=True)
        online_model.partial_fit(X, y, classes=np.array([0, 1]), sample_weight=sample_weight)
        online_initialized = True

    online_sample_count += 1
    online_label_counts[clean_label] = online_label_counts.get(clean_label, 0) + 1

    _save_online_learning_state()

def train_model():
    global model, last_batch_train_ts
    _flush_log_buffer(force=True)
    df, sample_weight = _load_weighted_training_dataframe(include_backtest=True)
    if df is None or len(df) < 50:
        return False

    model, _ = _fit_batch_model_from_dataframe(df, sample_weight=sample_weight)

    def _save_model():
        try:
            _persist_batch_model(model)
        except:
            pass
    threading.Thread(target=_save_model, daemon=True).start()
    last_batch_train_ts = time.time()
    print("✅ AI 更新")
    return True


def maybe_train_model_periodically(force=False):
    global last_batch_train_ts
    now_ts = time.time()
    if not force and (now_ts - last_batch_train_ts) < MODEL_RETRAIN_INTERVAL_SEC:
        return False
    return train_model()


def retrain_model():
    """強制重新訓練 AI 模型"""
    global model, last_batch_train_ts
    print("🔄 開始重新訓練 AI 模型...")
    _flush_log_buffer(force=True)

    df, sample_weight = _load_weighted_training_dataframe(include_backtest=True)
    if df is None:
        print("⚠️ 沒有訓練數據檔案")
        return False

    if len(df) < 10:
        print("⚠️ 訓練數據不足（至少需要10筆）")
        return False

    # 確保有 label 列
    if "label" not in df.columns:
        print("⚠️ 數據缺少 label 列")
        return

    if len(df) > 1500:
        print("✅ 使用最近1500筆數據訓練")

    model, _ = _fit_batch_model_from_dataframe(df, sample_weight=sample_weight)

    try:
        _persist_batch_model(model)
        print("✅ AI 模型重新訓練完成並保存")
    except Exception as e:
        print(f"⚠️ 保存模型失敗: {e}")
    last_batch_train_ts = time.time()
    return True

log_buffer = []


def _flush_log_buffer(force=False):
    global log_buffer
    if not log_buffer:
        return
    if not force and len(log_buffer) < AI_LOG_FLUSH_SIZE:
        return

    _ensure_training_data_schema()
    df = pd.DataFrame(log_buffer)
    ensure_parent_dir(DATA_PATH)
    if os.path.exists(DATA_PATH):
        df.to_csv(DATA_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(DATA_PATH, index=False)
    log_buffer = []


def _ensure_training_data_schema_for_path(path_str):
    if not os.path.exists(path_str):
        return

    try:
        probe = pd.read_csv(path_str, nrows=0)
        if list(probe.columns) == MODEL_FEATURE_COLUMNS + ["label"]:
            return
    except Exception:
        pass

    try:
        df = _load_training_dataframe_from_path(path_str)
        if df is None or df.empty:
            return
        ensure_parent_dir(path_str)
        df.to_csv(path_str, index=False)
        print(f"♻️ 已將訓練資料欄位遷移到目前特徵結構 ({Path(path_str).name})")
    except Exception as e:
        print(f"⚠️ 訓練資料欄位遷移失敗 ({Path(path_str).name}): {e}")


def _ensure_training_data_schema():
    _ensure_training_data_schema_for_path(DATA_PATH)
    _ensure_training_data_schema_for_path(BACKTEST_DATA_PATH)


def log_data(features, label):
    global log_buffer

    payload = _normalize_feature_payload(features)
    log_buffer.append({**payload, "label": 1 if int(label) > 0 else 0})
    _flush_log_buffer(force=False)


def _process_sl_followup_reviews(df_1m, current_price):
    global SL_FOLLOWUP_REVIEWS

    if not SL_FOLLOWUP_REVIEWS:
        return 0

    now_ts = time.time()
    current_px = max(0.0, _safe_float(current_price, 0.0))
    learned = 0
    keep = []
    queue_changed = False

    for item in SL_FOLLOWUP_REVIEWS:
        review_after_ts = _safe_float(item.get("review_after_ts"), 0.0)
        close_ts = _safe_float(item.get("close_ts"), 0.0)

        if review_after_ts > now_ts:
            keep.append(item)
            continue
        if close_ts > 0 and (now_ts - close_ts) > SL_FOLLOWUP_MAX_AGE_SEC:
            continue

        close_price = max(0.0, _safe_float(item.get("close_price"), 0.0))
        if close_price <= 0:
            continue

        atr_ref = max(_safe_float(item.get("atr"), 0.0), close_price * 0.001, 1e-6)
        post_high = current_px if current_px > 0 else close_price
        post_low = current_px if current_px > 0 else close_price

        try:
            if df_1m is not None and len(df_1m) > 0 and "time" in df_1m.columns:
                future = df_1m[df_1m["time"] >= int(close_ts * 1000)]
                if len(future) > 0:
                    post_high = max(post_high, _safe_float(future["high"].max(), post_high))
                    low_candidate = max(0.0, _safe_float(future["low"].min(), close_price))
                    post_low = min(post_low, low_candidate) if post_low > 0 else low_candidate
        except Exception:
            pass

        direction = _normalize_trade_direction(item.get("direction"))
        if direction == "long":
            favorable_move = max(0.0, post_high - close_price)
        else:
            favorable_move = max(0.0, close_price - post_low)

        if favorable_move < atr_ref * SL_FOLLOWUP_RECOVERY_ATR:
            retry_item = dict(item)
            retry_item["review_after_ts"] = now_ts + SL_FOLLOWUP_REVIEW_DELAY_SEC
            keep.append(retry_item)
            queue_changed = True
            continue

        sample_features = item.get("features")
        if not isinstance(sample_features, dict):
            continue

        log_data(sample_features, 1)
        update_online_model(sample_features, 1)
        learned += 1
        print(
            f"🧠 SL 後續走勢修正學習 | {direction} | close={close_price:.2f} "
            f"| favorable={favorable_move:.2f} | atr_ref={atr_ref:.2f}"
        )

    if queue_changed or len(keep) != len(SL_FOLLOWUP_REVIEWS):
        SL_FOLLOWUP_REVIEWS = keep
        _save_sl_followup_reviews()

    return learned


def _finalize_pending_training_sample(pending_sample, label, close_reason="", close_price=0.0, atr=0.0):
    record_strategy_outcome(
        label,
        close_reason=close_reason,
        close_price=close_price,
        strategy_version=STRATEGY_VERSION,
    )
    if not pending_sample or not isinstance(pending_sample, dict):
        _clear_pending_training_sample_state()
        return None

    direction = _normalize_trade_direction(pending_sample.get("direction"))
    mlx_episode_id = _safe_int(pending_sample.get("mlx_episode_id"), 0)
    if mlx_episode_id > 0:
        try:
            update_actual_trade_outcome(
                mlx_episode_id,
                close_price,
                1 if int(label) > 0 else 0,
                closed_at=time.time(),
            )
        except Exception as exc:
            print(f"⚠️ MLX實單結果回寫失敗: {exc}")
    sample_features = pending_sample.get("learn_features")

    if not isinstance(sample_features, dict):
        base_features = pending_sample.get("features")
        if isinstance(base_features, dict):
            sample_features = _build_directional_learning_features(base_features, direction)

    if isinstance(sample_features, dict):
        clean_label = 1 if int(label) > 0 else 0
        log_data(sample_features, clean_label)
        update_online_model(sample_features, clean_label)
        if str(close_reason or "").upper() == "SL" and clean_label == 0:
            _queue_sl_followup_review(sample_features, direction, close_price, time.time(), atr)

    _clear_pending_training_sample_state()
    return None

def send_telegram(msg, priority=False, include_private=True):
    global LAST_TELEGRAM_TS

    now = time.time()

    # ===== 只有低優先才限流 =====
    if not priority and now - LAST_TELEGRAM_TS < 10:
        return False

    if include_private:
        targets = _get_notification_chat_ids()
    else:
        targets = []  # 群聊推播已停用

    if not targets and _is_truthy(os.getenv("TELEGRAM_FALLBACK_PRIVATE_WHEN_NO_BROADCAST_TARGET", "1")):
        private_target = _resolve_private_chat_id_for_controls()
        if private_target:
            targets = [private_target]

    if not TELEGRAM_TOKEN or not targets:
        print("⚠️ Telegram 目標未設定，略過發送")
        return False

    try:
        sent_count = 0
        for chat_id in targets:
            res = _send_telegram_message(chat_id, msg, include_control_panel=True)

            if res is None or res.status_code != 200:
                status = getattr(res, "status_code", "no-response")
                body = getattr(res, "text", "")
                delivery = _note_telegram_delivery_event(
                    chat_id=chat_id,
                    ok=False,
                    status_code=status,
                    body=body,
                    error="sendMessage returned no response" if res is None else None,
                    context="eth.send_telegram.broadcast",
                )
                print(f"❌ Telegram 發送失敗 [{chat_id}]:", status, body)

                if delivery.get("remove_chat") or _is_telegram_chat_not_found(status, body):
                    _remove_notification_chat(chat_id)
                    continue

                try:
                    retry_after = max(1, int(delivery.get("retry_after", 0) or 0))
                    time.sleep(min(30, retry_after))
                    res2 = _send_telegram_message(chat_id, msg, include_control_panel=True)
                    retry_status = getattr(res2, "status_code", "no-response")
                    retry_body = getattr(res2, "text", "")
                    retry_delivery = _note_telegram_delivery_event(
                        chat_id=chat_id,
                        ok=res2 is not None and res2.status_code == 200,
                        status_code=retry_status,
                        body=retry_body,
                        error="retry sendMessage returned no response" if res2 is None else None,
                        context="eth.send_telegram.broadcast_retry",
                    )
                    print(f"🔁 retry [{chat_id}]:", retry_status)
                    if res2 is not None and res2.status_code == 200:
                        sent_count += 1
                    elif retry_delivery.get("remove_chat"):
                        _remove_notification_chat(chat_id)
                except Exception as e:
                    _note_telegram_delivery_event(
                        chat_id=chat_id,
                        ok=False,
                        status_code="exception",
                        error=e,
                        context="eth.send_telegram.broadcast_retry",
                    )
                    print(f"❌ retry失敗 [{chat_id}]:", e)
            else:
                _note_telegram_delivery_event(
                    chat_id=chat_id,
                    ok=True,
                    status_code=res.status_code,
                    body=getattr(res, "text", ""),
                    context="eth.send_telegram.broadcast",
                )
                sent_count += 1

        if sent_count > 0:
            print(f"✅ Telegram 已送出 ({sent_count}/{len(targets)})")

        # Discord只發「進場通知」
        try:
            if DISCORD_WEBHOOK and "進場" in msg:
                _post_discord_webhook(DISCORD_WEBHOOK, msg, timeout=5)
        except Exception as e:
            print("Discord error:", e)

        LAST_TELEGRAM_TS = now

    except Exception as e:
        print("❌ Telegram error:", e, "| msg:", msg[:50])
        return False

    return sent_count > 0


def send_private_telegram(msg, priority=False):
    global LAST_TELEGRAM_TS

    now = time.time()
    if not priority and now - LAST_TELEGRAM_TS < 10:
        return False

    dedupe_key = ""
    dedupe_cache = {}
    if _is_truthy(os.getenv("TELEGRAM_PRIVATE_DEDUPE_ENABLED", "1")):
        dedupe_text = str(msg or "").strip()
        if dedupe_text:
            dedupe_key = hashlib.sha256(dedupe_text.encode("utf-8", errors="ignore")).hexdigest()
            dedupe_cache = getattr(send_private_telegram, "_dedupe_cache", {})
            cooldown = max(
                30.0,
                _safe_float(
                    os.getenv(
                        "TELEGRAM_PRIVATE_PRIORITY_DEDUPE_SEC" if priority else "TELEGRAM_PRIVATE_DEDUPE_SEC",
                        180 if priority else 60,
                    ),
                    180 if priority else 60,
                ),
            )
            last_sent = _safe_float(dedupe_cache.get(dedupe_key), 0.0)
            if now - last_sent < cooldown:
                if now - _safe_float(getattr(send_private_telegram, "_last_dedupe_log_ts", 0.0), 0.0) > 60:
                    print("🔕 私聊重複通知已略過")
                    send_private_telegram._last_dedupe_log_ts = now
                return False

    target = _resolve_private_chat_id_for_controls()
    if not TELEGRAM_TOKEN or not target:
        print("⚠️ 私聊目標未設定，略過發送")
        return False

    try:
        res = _send_telegram_message(target, msg, include_control_panel=True)

        if res is None or res.status_code != 200:
            status = getattr(res, "status_code", "no-response")
            body = getattr(res, "text", "")
            delivery = _note_telegram_delivery_event(
                chat_id=target,
                ok=False,
                status_code=status,
                body=body,
                error="sendMessage returned no response" if res is None else None,
                context="eth.send_private_telegram",
            )
            print(f"❌ 私聊發送失敗 [{target}]", status, body)

            if delivery.get("remove_chat") or _is_telegram_chat_not_found(status, body):
                _remove_notification_chat(target)

            return False

        _note_telegram_delivery_event(
            chat_id=target,
            ok=True,
            status_code=res.status_code,
            body=getattr(res, "text", ""),
            context="eth.send_private_telegram",
        )
        LAST_TELEGRAM_TS = now
        if dedupe_key:
            dedupe_cache[dedupe_key] = now
            if len(dedupe_cache) > 300:
                cutoff = now - 3600.0
                dedupe_cache = {key: ts for key, ts in dedupe_cache.items() if _safe_float(ts, 0.0) >= cutoff}
            send_private_telegram._dedupe_cache = dedupe_cache
        summary = str(msg or "").strip().splitlines()[0][:48]
        print(f"✅ 私聊通知已送出: {summary}" if summary else "✅ 私聊通知已送出")
        return True
    except Exception as e:
        _note_telegram_delivery_event(
            chat_id=target,
            ok=False,
            status_code="exception",
            error=e,
            context="eth.send_private_telegram",
        )
        print("❌ 私聊通知錯誤:", e)
        return False


def _send_trade_notification(msg, priority=True):
    delivered = send_telegram(msg, priority=priority)
    if delivered:
        return True
    return send_private_telegram(msg, priority=priority)


# ===== AI分析（MLX / OpenAI fallback） =====
OPENAI_API_KEY = _get_required_env("OPENAI_API_KEY", "", mask=True)
if MLX_AGENT_ENABLED:
    print(f"✅ MLX AI agent: {MLX_MODEL} @ {MLX_AGENT_BASE_URL}")
if OPENAI_PAID_API_ENABLED and OPENAI_API_KEY:
    print(
        f"✅ OpenAI 模型: 分析={OPENAI_CHAT_MODEL} | 翻譯={OPENAI_TRANSLATION_MODEL} | reasoning={OPENAI_REASONING_EFFORT}"
    )
else:
    print("🟢 OpenAI 付費 API 已停用，AI 將只使用本地模型與免費 fallback")


_MLX_AUTO_ANALYSIS_LOCK = threading.Lock()


def ask_ai_analysis(prompt, market_context=None, question="", learning_limit=None):
    market_context = market_context if isinstance(market_context, dict) else {}
    learning_context = build_mlx_learning_context(market_context, limit=learning_limit)
    learned_prompt = prompt
    if learning_context:
        learned_prompt = f"{prompt}\n\n{learning_context}"
    messages = [
        {
            "role": "system",
            "content": (
                "你是一個專業 ETH 交易分析師。只根據使用者提供的市場資料分析，"
                "清楚區分事實與推論，不承諾獲利，並用繁體中文簡潔回答。"
            ),
        },
        {"role": "user", "content": learned_prompt},
    ]
    mlx_error = ""

    if MLX_AGENT_ENABLED:
        try:
            response = requests.post(
                f"{MLX_AGENT_BASE_URL}/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": MLX_MODEL,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 900,
                },
                timeout=max(10, MLX_AGENT_TIMEOUT_SEC),
            )
            response.raise_for_status()
            text = _extract_openai_chat_text(response.json())
            if text:
                recorded = record_mlx_analysis(question or prompt, text, market_context)
                if str(question).startswith("auto-shadow:") and not recorded:
                    raise ValueError("MLX 影子開單缺少有效的 TP/SL，未寫入學習資料")
                return text
            mlx_error = "本地模型回傳空內容"
        except Exception as exc:
            mlx_error = str(exc)
            print(f"⚠️ MLX AI agent 請求失敗: {exc}")

    if OPENAI_PAID_API_ENABLED and OPENAI_API_KEY:
        try:
            payload = _build_openai_chat_payload(
                OPENAI_CHAT_MODEL,
                [
                    {"role": _openai_instruction_role(OPENAI_CHAT_MODEL), "content": messages[0]["content"]},
                    messages[1],
                ],
                temperature=0.3,
            )
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            text = _extract_openai_chat_text(response.json())
            if text:
                recorded = record_mlx_analysis(question or prompt, text, market_context)
                if str(question).startswith("auto-shadow:") and not recorded:
                    raise ValueError("影子開單缺少有效的 TP/SL，未寫入學習資料")
                return text
            return "AI分析失敗: OpenAI 回傳空內容"
        except Exception as exc:
            return f"AI分析失敗: {exc}"

    if mlx_error:
        return f"AI分析失敗: 本地 MLX agent 無法使用（{mlx_error}）"
    return "AI分析已停用：MLX agent 與 OpenAI 付費 API 均未啟用。"


def _start_mlx_auto_analysis(period_key, market_context):
    if not MLX_AGENT_ENABLED or not _is_truthy(os.getenv("MLX_AUTO_ANALYSIS_ENABLED", "1")):
        return False
    if not claim_auto_analysis(period_key):
        return False

    context = dict(market_context or {})

    def worker():
        try:
            with _MLX_AUTO_ANALYSIS_LOCK:
                prompt = f"""
你是 ETH 影子交易分析 Agent。根據以下資料建立影子交易預測。
這是無資金風險的研究分析，絕對不會送出真實委託。
影子開單數量不設上限，可同時建立多筆做多或做空預測。每筆先等待指定進場價觸發，
成交後持續追蹤到 TP 或 SL 其中一個先到。TP 先到才算成功，SL 先到算失敗，
不再以固定一小時後的漲跌判定。至少建立一筆；只有存在不同、可說明的交易依據時
才增加筆數，不可為湊數而重複相同內容。

價格: {context.get('price')}
15m RSI14: {context.get('rsi_15m')}，EMA50乖離: {context.get('ema50_deviation_15m')}%
1H K線: {context.get('one_hour_pattern')}，動能方向: {context.get('mid_trend')}
4H趨勢: {context.get('htf')}
4H K線: {context.get('four_hour_pattern')}
日線趨勢: {context.get('daily_trend')}，K線: {context.get('daily_pattern')}，30日變化: {context.get('daily_medium_change_pct')}%
月線趨勢: {context.get('monthly_trend')}，K線: {context.get('monthly_pattern')}，全部變化: {context.get('monthly_window_change_pct')}%，區間位置: {context.get('monthly_range_pos')}
週線趨勢: {context.get('weekly_trend')}，K線: {context.get('weekly_pattern')}，2年變化: {context.get('weekly_window_change_pct')}%，區間位置: {context.get('weekly_range_pos')}
日線1年變化: {context.get('daily_window_change_pct')}%，日線區間位置: {context.get('daily_range_pos')}
4H一個月變化: {context.get('four_hour_window_change_pct')}%，4H區間位置: {context.get('four_hour_range_pos')}
1H一週變化: {context.get('one_hour_window_change_pct')}%，1H區間位置: {context.get('one_hour_range_pos')}
15m一天變化: {context.get('fifteen_min_window_change_pct')}%，15m區間位置: {context.get('fifteen_min_range_pos')}
MLX多區間K線讀法順序: {context.get('host_style_kline_order')}
MLX多區間K線判讀:
{context.get('timeframe_kline_summary')}
多週期是否衝突: {context.get('timeframe_conflict')}
市場狀態: {context.get('regime')}
突破: {context.get('breakout')}
宏觀偏向: {context.get('macro')}
量能放大: {context.get('volume_spike')}
15m ATR參考: {context.get('atr_15m')}
支撐候選（只能用來組支撐區）: 15m={context.get('range_support_15m')}；1H={context.get('range_support_1h')}
壓力候選（只能用來組壓力區）: 15m={context.get('range_resistance_15m')}；1H={context.get('range_resistance_1h')}
多週期支撐壓力:
{context.get('sr_lines')}
策略計算的合法震盪參考方案（價位必須照抄）:
{context.get('range_trade_reference')}

學習目標：
1. 用 MLX 多區間K線讀法順序判讀：先看月/週/日定大背景，再看4H/1H判斷主要方向與支撐壓力，最後只用15m找進場觸發。
2. 分別說明15m、1H、4H、日線、週線、月線K線看到哪些因素支持做多或做空。
3. 高位不追多、低位不追空；不同週期互相衝突時要降低信心、縮小倉位或等待，不可硬開。
4. 從多週期價位找出目前有效的支撐區與壓力區，不可只給單一模糊價位。
5. 若市場為震盪，做多應靠近支撐、做空應靠近壓力；SL放在區間外並預留ATR緩衝，
   TP放在對側區間之前。必須給出具體多空兩套TP/SL與區間失效條件。
6. 支撐區只能由支撐候選組成，壓力區只能由壓力候選組成，禁止交叉混用。
7. 價位必須滿足：震盪做多 SL < 進場 < TP；震盪做空 TP < 進場 < SL。
   多單SL必須低於支撐區下緣；空單SL必須高於壓力區上緣。
8. 震盪支撐、壓力、進場、TP、SL必須逐字使用「策略計算的合法震盪參考方案」，
   不可自行交換或另算價位。

請先輸出一段嚴格 JSON，不可省略欄位，格式如下：
```json
{{
  "direction": "做多或做空",
  "primary_reason": "趨勢/支撐壓力/震盪/突破/新聞/量能/多週期衝突",
  "confidence": 0.0,
  "market_regime": "trend/range/breakout/fake_breakout_risk/news_driven/high_tf_conflict/higher_tf_transition",
  "support_zone": [支撐下緣數字, 支撐上緣數字],
  "resistance_zone": [壓力下緣數字, 壓力上緣數字],
  "range_long": {{"entry": 數字, "tp": 數字, "sl": 數字, "invalidation": "文字"}},
  "range_short": {{"entry": 數字, "tp": 數字, "sl": 數字, "invalidation": "文字"}},
  "factors": ["最多8個因素"],
  "timeframe_view": {{
    "15m": "做多/做空因素",
    "1h": "做多/做空因素",
    "4h": "做多/做空因素",
    "1d": "做多/做空因素",
    "1w": "做多/做空因素",
    "1M": "做多/做空因素"
  }}
}}
```
confidence 請用 0 到 1 的小數；direction 不可輸出觀望。

JSON 後再嚴格用以下格式輸出：
時段判讀：15m=...；1H=...；4H=...；日線=...；週線=...；月線=...
有效支撐區：價格-價格
有效壓力區：價格-價格
震盪做多：進場...；TP...；SL...
震盪做空：進場...；TP...；SL...
區間失效條件：...
多方機率：整數%
空方機率：整數%
影子開單1：做多或做空；進場=數字；TP=數字；SL=數字；依據=...
影子開單2：做多或做空；進場=數字；TP=數字；SL=數字；依據=...（依需要繼續列出，筆數不限）
多空機率合計必須為 100%。不得輸出觀望，不得聲稱已送出真實委託。
每筆做多必須 SL < 進場 < TP；每筆做空必須 TP < 進場 < SL。
"""
                result = ask_ai_analysis(
                    prompt,
                    market_context=context,
                    question=f"auto-shadow:{period_key}",
                    learning_limit=4,
                )
                if str(result).startswith(("AI分析失敗", "AI分析已停用")):
                    release_auto_analysis(period_key)
                    print(f"⚠️ MLX 自動影子分析失敗，保留重試: {period_key}")
                else:
                    print(f"🧠 MLX 自動影子分析已記錄: {period_key}")
        except Exception as exc:
            release_auto_analysis(period_key)
            print(f"⚠️ MLX 自動影子分析錯誤: {exc}")

    try:
        threading.Thread(target=worker, daemon=True, name="mlx-auto-analysis").start()
        return True
    except Exception:
        release_auto_analysis(period_key)
        return False


def _build_bot_help_text():
    return (
        "可用指令：\n"
        "/start - 開始使用並顯示控制面板\n"
        "/help - 顯示這份說明\n"
        "/whoami - 顯示你的 Telegram user id\n"
        "/settings - 顯示跟單與控制面板設定\n"
        "/panel 或 /menu - 開啟倉位面板\n"
        "/follow - 開關跟單\n"
        "/sync - 同步幣安倉位\n"
        "/tp 2300 - 設定止盈\n"
        "/sl 2350 - 設定止損\n"
        "/tpsl 2300 2350 - 同時設定 TP/SL\n"
        "/ai 問題 - AI 分析市場\n"
        "/news - 取得最新新聞\n"
        "/fix - 即時修正錯誤\n"
        "/restart - 重啟 bot"
    )


def _build_bot_settings_text():
    return (
        "設定入口：\n"
        "- 用 /panel 或 /menu 開啟控制面板\n"
        "- 用 /follow 切換跟單開關\n"
        "- 用 /sync 同步幣安倉位\n"
        "- 用 /whoami 查自己的 Telegram user id\n"
        "- 用 /tp、/sl、/tpsl 調整持倉保護\n"
        "- 若要手動平倉，請使用控制面板上的「⛔ 一鍵平倉」"
    )


def _build_whoami_text(context=None):
    context = context if isinstance(context, dict) else {}
    chat_id = context.get("chat_id")
    user_id = context.get("user_id")
    username = str(context.get("username", "") or "").strip()
    first_name = str(context.get("first_name", "") or "").strip()
    chat_type = str(context.get("chat_type", "") or "").strip()

    whitelist_user_id = user_id
    if whitelist_user_id in (None, "") and _is_private_chat_id(chat_id):
        whitelist_user_id = chat_id

    lines = ["你的 Telegram 資訊"]
    if whitelist_user_id not in (None, ""):
        lines.append(f"user_id: {whitelist_user_id}")
    else:
        lines.append("user_id: 取得失敗")

    if chat_id not in (None, ""):
        lines.append(f"chat_id: {chat_id}")

    if chat_type:
        lines.append(f"chat_type: {chat_type}")

    if username:
        lines.append(f"username: @{username}")

    if first_name:
        lines.append(f"first_name: {first_name}")

    if whitelist_user_id not in (None, ""):
        lines.append("")
        lines.append("白名單範例：")
        lines.append(f"POSITION_PANEL_ALLOWED_TELEGRAM_USER_IDS={whitelist_user_id}")
        lines.append("白名單要填 user_id，不是 @username。")
    else:
        lines.append("")
        lines.append("建議在和 bot 的私聊視窗內重新送一次 /whoami。")

    return "\n".join(lines)


REALTIME_FIX_CHECK_LABELS = {
    "conflict_markers": "程式衝突標記",
    "dependency_constraints": "套件依賴",
    "runtime_memory": "記憶體使用與優化",
    "panel_tunnel_health": "面板連線與隧道",
    "runtime_storage": "執行資料儲存空間",
    "runtime_json": "執行狀態 JSON",
    "entry_confirm_runtime": "進場確認執行狀態",
    "entry_confirm_candle_id": "進場確認 K 線追蹤",
    "py_compile": "Python 語法檢查",
    "import_smoke": "模組載入測試",
    "telegram_policy": "Telegram 設定",
    "telegram_watch_risk": "Telegram 發送風險",
    "model_health": "交易模型健康狀態",
}
REALTIME_FIX_STATUS_LABELS = {
    "ok": "正常",
    "fixed": "已自動修復",
    "error": "異常",
}
REALTIME_FIX_COMMANDS = {
    "/fix",
    "/fix_errors",
    "/repair",
    "/修復錯誤",
    "修復錯誤",
    "修正錯誤",
    "即時修復",
    "即時修正錯誤",
}


def _recent_program_log_errors(limit=5, line_window=260):
    try:
        if not PROGRAM_LOG_FILE.exists():
            return []
        lines = PROGRAM_LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()[-int(line_window):]
    except Exception:
        return []

    keywords = ("Traceback", "Exception", "Error", "ERROR", "錯誤", "失敗", "❌")
    skips = ("Telegram 已送出", "私聊通知已送出", "新聞監控中", "持倉監控", "ERROR_RATE")
    findings = []
    seen = set()
    for raw_line in reversed(lines):
        line = str(raw_line or "").strip()
        if not line:
            continue
        if any(skip in line for skip in skips):
            continue
        if not any(keyword in line for keyword in keywords):
            continue
        compact = re.sub(r"\s+", " ", line)
        if compact in seen:
            continue
        seen.add(compact)
        findings.append(compact[:240])
        if len(findings) >= limit:
            break
    return list(reversed(findings))


def _format_realtime_fix_reply(report, command_output="", elapsed_sec=0.0):
    report = report if isinstance(report, dict) else {}
    status = str(report.get("status") or "error")
    status_label = REALTIME_FIX_STATUS_LABELS.get(status, status)
    auto_fix_count = int(report.get("auto_fix_count") or 0)
    checks = report.get("checks") if isinstance(report.get("checks"), list) else []

    fixed_items = [item for item in checks if item.get("status") == "fixed"]
    error_items = [item for item in checks if item.get("status") == "error"]

    lines = [
        "🛠️ 即時錯誤修正完成",
        f"狀態: {status_label}",
        f"自動修復: {auto_fix_count} 項",
        f"耗時: {elapsed_sec:.1f} 秒",
    ]

    if fixed_items:
        lines.append("")
        lines.append("已修復:")
        for item in fixed_items[:6]:
            name = str(item.get("name") or "")
            label = REALTIME_FIX_CHECK_LABELS.get(name, name or "未知項目")
            detail = str(item.get("detail") or "").strip()
            lines.append(f"- {label}: {detail[:160]}")

    if error_items:
        lines.append("")
        lines.append("仍異常:")
        for item in error_items[:6]:
            name = str(item.get("name") or "")
            label = REALTIME_FIX_CHECK_LABELS.get(name, name or "未知項目")
            detail = str(item.get("detail") or item.get("error") or "").strip()
            lines.append(f"- {label}: {detail[:220]}")

    recent_errors = _recent_program_log_errors(limit=4)
    if recent_errors:
        lines.append("")
        lines.append("最近錯誤摘要:")
        for line in recent_errors:
            lines.append(f"- {line}")

    if not fixed_items and not error_items and not recent_errors:
        lines.append("")
        lines.append("目前沒有偵測到需要即時修正的錯誤。")

    if not checks and command_output:
        lines.append("")
        lines.append("maintenance 輸出:")
        lines.append(str(command_output).strip()[:800])

    return "\n".join(lines)[:3600]


def _run_realtime_error_fix():
    now = time.time()
    if getattr(_run_realtime_error_fix, "_running", False):
        return "⏳ 即時錯誤修正正在執行中，完成後會回報。"

    cooldown_sec = max(30.0, _safe_float(os.getenv("REALTIME_FIX_COOLDOWN_SEC", "120"), 120.0))
    last_run_ts = _safe_float(getattr(_run_realtime_error_fix, "_last_run_ts", 0.0), 0.0)
    if last_run_ts and now - last_run_ts < cooldown_sec:
        remain = int(max(1, cooldown_sec - (now - last_run_ts)))
        return f"⏳ 剛執行過即時修正，請 {remain} 秒後再試。"

    setattr(_run_realtime_error_fix, "_running", True)
    started = time.time()
    try:
        pycache_dir = data_path("..", "pycache").resolve()
        ensure_parent_dir(pycache_dir / ".keep")
        env = os.environ.copy()
        env["PYTHONPYCACHEPREFIX"] = str(pycache_dir)
        timeout_sec = int(max(90.0, _safe_float(os.getenv("REALTIME_FIX_TIMEOUT_SEC", "240"), 240.0)))
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_DIR / "maintenance.py"),
                "--skip-smoke-backtest",
                "--no-notify",
            ],
            cwd=str(REPO_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        output = result.stdout or ""
        report = _read_json_file(MAINTENANCE_REPORT_FILE, {})
        if not isinstance(report, dict) or not report:
            try:
                report = json.loads(output)
            except Exception:
                report = {
                    "status": "error" if result.returncode else "ok",
                    "checks": [],
                    "auto_fix_count": 0,
                }
        if result.returncode != 0 and report.get("status") != "error":
            report["status"] = "error"
        return _format_realtime_fix_reply(report, output, time.time() - started)
    except subprocess.TimeoutExpired as exc:
        output = str(getattr(exc, "stdout", "") or "").strip()
        return (
            "⚠️ 即時錯誤修正逾時\n"
            f"timeout={int(_safe_float(os.getenv('REALTIME_FIX_TIMEOUT_SEC', '240'), 240))}s\n"
            f"{output[:800]}"
        ).strip()
    except Exception as exc:
        return f"⚠️ 即時錯誤修正啟動失敗: {exc}"
    finally:
        setattr(_run_realtime_error_fix, "_running", False)
        setattr(_run_realtime_error_fix, "_last_run_ts", time.time())


# ===== Telegram 指令（AI分析） =====
def handle_ai_command(text, context=None):
    text = str(text or "").strip()
    context = context if isinstance(context, dict) else {}
    chat_id = context.get("chat_id")
    is_private_chat = _is_private_chat_id(chat_id)

    if text.startswith("/start"):
        if is_private_chat:
            send_control_panel(chat_id)
        return "歡迎使用 ETH bot\n\n" + _build_bot_help_text()

    if text.startswith("/help"):
        return _build_bot_help_text()

    if text.startswith("/whoami"):
        return _build_whoami_text(context)

    if text.startswith("/settings"):
        if is_private_chat:
            send_control_panel(chat_id)
        return _build_bot_settings_text()

    if str(text).strip() in {FOLLOW_BUTTON_TEXT_DISABLED, FOLLOW_BUTTON_TEXT_ENABLED} or text.startswith("/follow"):
        enabled = _toggle_follow_mode_enabled()
        if enabled and _is_real_copy_enabled():
            return "✅ 跟單已開啟（Binance 自動開單已啟用）"
        if enabled:
            return "✅ 跟單已開啟（目前僅訊號，請設 BINANCE_REAL_COPY_ENABLED=1 後重啟啟用實單）"
        return "⏹️ 跟單已關閉"

    if text.startswith("/sync") or str(text).strip() == "同步幣安倉位":
        ok, message = sync_active_trade_from_binance(send_notice=False)
        return message

    if text.startswith("/tp ") or text.startswith("/sl ") or text.startswith("/tpsl "):
        if not active_trade.get("open"):
            return "目前無持倉，無法調整 TP/SL"

        entry = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), 0.0)
        direction = str(active_trade.get("direction") or "")
        curr_tp = _safe_float(active_trade.get("tp"), 0.0)
        curr_sl = _safe_float(active_trade.get("sl"), 0.0)

        parts = text.split()
        new_tp = curr_tp
        new_sl = curr_sl

        if text.startswith("/tp "):
            if len(parts) < 2:
                return "用法: /tp 2300"
            new_tp = _safe_float(parts[1], 0.0)
        elif text.startswith("/sl "):
            if len(parts) < 2:
                return "用法: /sl 2350"
            new_sl = _safe_float(parts[1], 0.0)
        else:
            if len(parts) < 3:
                return "用法: /tpsl 2300 2350"
            new_tp = _safe_float(parts[1], 0.0)
            new_sl = _safe_float(parts[2], 0.0)

        if new_tp <= 0 or new_sl <= 0:
            return "TP/SL 需為正數"

        if direction == "long":
            if new_tp <= entry:
                return f"多單 TP 必須 > 進場價 {entry:.2f}"
        elif direction == "short":
            if new_tp >= entry:
                return f"空單 TP 必須 < 進場價 {entry:.2f}"
        else:
            return "持倉方向未知，無法調整 TP/SL"

        if direction == "long":
            if new_sl <= 0 or new_sl >= new_tp:
                return f"多單 SL 必須 > 0 且 < TP {new_tp:.2f}"
        elif direction == "short":
            if new_sl <= 0 or new_sl <= new_tp:
                return f"空單 SL 必須 > TP {new_tp:.2f}"

        active_trade["tp"] = float(new_tp)
        active_trade["sl"] = float(new_sl)
        _sync_break_even_state_from_sl(direction, entry, new_sl, preserve_existing=False)
        sync_position_panel(_safe_float(WS_PRICE, entry))

        sync_msg = ""
        if _get_follow_mode_enabled() and _is_real_copy_enabled():
            try:
                ok, binance_msg = update_copy_trade_tp_sl(new_tp, new_sl)
                sync_msg = f"\n{binance_msg}"
            except Exception as e:
                sync_msg = f"\n⚠️ Binance TP/SL 同步失敗: {e}"

        return (
            f"✅ 已更新 TP/SL\n"
            f"方向: {direction}\n"
            f"進場: {entry:.2f}\n"
            f"TP: {new_tp:.2f}\n"
            f"SL: {new_sl:.2f}"
            f"{sync_msg}"
        )

    if text == "⛔ 一鍵平倉":
        if not active_trade.get("open"):
            return "目前無持倉"

        _, binance_msg = _perform_manual_close()
        result = "✅ 已執行一鍵平倉"
        if binance_msg:
            result += f"\n{binance_msg}"
        return result

    if text.startswith("/ai"):
        question = text.replace("/ai", "").strip()

        if context is None:
            context = {}

        # 如果沒輸入問題 → 自動分析當前市場
        if not question:
            question = "請根據以下數據判斷是否應該做多或做空，並給出理由與策略"

        if question.lower() in {"stats", "status", "學習", "學習狀態"}:
            stats = get_mlx_learning_stats()
            top_factors = stats.get("top_factors") or []
            factor_line = "尚無已驗證因子"
            if top_factors:
                factor_line = "；".join(
                    f"{item['name']} {item['winrate']:.1f}%({item['wins']}/{item['total']})"
                    for item in top_factors[:5]
                )
            grade_stats = stats.get("shadow_grades") or []
            grade_line = "尚無已驗證分級"
            if grade_stats:
                grade_line = "；".join(
                    f"{item['name']}級 {item['winrate']:.1f}%({item['wins']}/{item['total']})"
                    for item in grade_stats
                )
            regime_stats = stats.get("market_regimes") or []
            regime_line = "尚無已驗證盤型"
            if regime_stats:
                regime_line = "；".join(
                    f"{item['name']} {item['winrate']:.1f}%({item['wins']}/{item['total']})"
                    for item in regime_stats[:5]
                )
            reason_stats = stats.get("primary_reasons") or []
            reason_line = "尚無已驗證主因"
            if reason_stats:
                reason_line = "；".join(
                    f"{item['name']} {item['winrate']:.1f}%({item['wins']}/{item['total']})"
                    for item in reason_stats[:5]
                )
            version_stats = stats.get("strategy_versions") or []
            version_line = "尚無版本統計"
            if version_stats:
                version_line = "；".join(
                    f"{item['name']} {item['winrate']:.1f}%({item['wins']}/{item['evaluated']})"
                    for item in version_stats[:5]
                )
            sl_stats = sl_review_summary(limit=3)
            sl_line = (
                f"累積 {sl_stats['total']} 筆；近{sl_stats['recent_checked']}筆需重審 "
                f"{sl_stats['revalidation']} 筆"
            )
            if sl_stats["top_issues"]:
                sl_line += "；常見問題 " + "、".join(
                    f"{issue}({count})" for issue, count in sl_stats["top_issues"]
                )
            return (
                "🧠 MLX 學習狀態\n"
                f"累積分析: {stats['total']}\n"
                f"已驗證: {stats['evaluated']}\n"
                f"成功案例: {stats['successful']}\n"
                f"結構化分析: {stats.get('structured_analyses', 0)}\n"
                f"既有模型匯入: {stats.get('imported', 0)}\n"
                f"可用學習案例: {stats.get('context_total', stats['total'])}\n"
                f"高週期觀察: {stats.get('higher_tf_observations', 0)}\n"
                f"歷史變盤案例: {stats.get('turning_points', 0)}\n"
                f"非變盤對照: {stats.get('contrast_examples', 0)}\n"
                f"自動影子分析: {stats.get('auto_analyses', 0)}\n"
                f"實單分析: {stats.get('actual_trades', 0)} 筆"
                f"（已驗證 {stats.get('actual_trades_evaluated', 0)}，"
                f"成功 {stats.get('actual_trades_successful', 0)}）\n"
                f"影子單分級: {grade_line}\n"
                f"因子勝率: {factor_line}\n"
                f"主因勝率: {reason_line}\n"
                f"盤型勝率: {regime_line}\n"
                f"策略版本: {version_line}\n"
                f"SL檢討: {sl_line}\n"
                f"驗證準確率: {stats['accuracy']:.1f}%\n"
                "新影子單驗證: TP先到才成功，SL先到即失敗\n"
                f"舊案例驗證週期: {stats['evaluation_hours']:.0f} 小時"
            )

        # 注入你系統數據（核心升級）
        prompt = f"""
你是一個專業ETH交易分析師，請根據以下即時市場數據進行分析：

【市場數據】
價格: {context.get('price')}
AI分數: {context.get('score')}
HTF趨勢: {context.get('htf')}
日線趨勢: {context.get('daily_trend')}（強度 {context.get('daily_strength_pct')}%）
週線趨勢: {context.get('weekly_trend')}（強度 {context.get('weekly_strength_pct')}%）
月線全部: 趨勢 {context.get('monthly_trend')}，變化 {context.get('monthly_window_change_pct')}%，區間位置 {context.get('monthly_range_pos')}
週線2年: 趨勢 {context.get('weekly_trend')}，變化 {context.get('weekly_window_change_pct')}%，區間位置 {context.get('weekly_range_pos')}
日線1年: 趨勢 {context.get('daily_trend')}，變化 {context.get('daily_window_change_pct')}%，區間位置 {context.get('daily_range_pos')}
4H一個月: 趨勢 {context.get('four_hour_trend')}，變化 {context.get('four_hour_window_change_pct')}%，區間位置 {context.get('four_hour_range_pos')}
1H一週: 趨勢 {context.get('one_hour_trend')}，變化 {context.get('one_hour_window_change_pct')}%，區間位置 {context.get('one_hour_range_pos')}
15m一天: 趨勢 {context.get('fifteen_min_trend')}，變化 {context.get('fifteen_min_window_change_pct')}%，區間位置 {context.get('fifteen_min_range_pos')}
市場狀態: {context.get('regime')}
Breakout: {context.get('breakout')}
Triangle: {context.get('triangle')}
Macro: {context.get('macro')}
Volume Spike: {context.get('volume_spike')}

【問題】
{question}

請輸出：
先輸出嚴格 JSON：
```json
{{
  "direction": "做多/做空/觀望",
  "primary_reason": "趨勢/支撐壓力/震盪/突破/新聞/量能/多週期衝突",
  "confidence": 0.0,
  "market_regime": "trend/range/breakout/fake_breakout_risk/news_driven/high_tf_conflict/higher_tf_transition",
  "support_zone": [支撐下緣數字, 支撐上緣數字],
  "resistance_zone": [壓力下緣數字, 壓力上緣數字],
  "entry_zone": [進場下緣數字, 進場上緣數字],
  "tp": 數字,
  "sl": 數字,
  "factors": ["最多8個因素"]
}}
```
再輸出：
1. 當前市場結構
2. 是否建議做多/做空/觀望
3. 進場區間
4. 止盈止損建議
"""

        return ask_ai_analysis(prompt, market_context=context, question=question)

    if text.startswith("/news"):
        try:
            _, _, news_list = refresh_rss_news_cache(force=True)
            if news_list:
                preview = "\n".join([f"- {n}" for n in news_list[:12]])
                return f"📰 最新即時訊息\n{preview}"
            return "📰 目前沒有抓到新的即時訊息"
        except Exception as e:
            return f"📰 新聞讀取失敗: {e}"

    command_name = text.split(maxsplit=1)[0] if text else ""
    if command_name in REALTIME_FIX_COMMANDS or text in REALTIME_FIX_COMMANDS:
        if not is_private_chat:
            return "即時錯誤修正只允許在私聊執行。"
        return _run_realtime_error_fix()

    if text.startswith("/panel") or text.startswith("/menu"):
        send_control_panel(chat_id)
        return "✅ 已送出倉位面板 / 跟單設定 / 一鍵平倉控制面板"

    if is_private_chat and text:
        if text.startswith("/"):
            return "無法識別的指令。\n\n" + _build_bot_help_text()
        return "已收到訊息。若要操作 bot，請使用以下指令：\n\n" + _build_bot_help_text()

    return None

if LIVE_RUNTIME_ENABLED:
    load_model()

# =============================
# API（簡化 + CACHE）
# =============================
def _log_kline_source_failure(source_name, exc, prefix="K線"):
    now_ts = time.time()
    state = getattr(_log_kline_source_failure, "_state", {})
    key = f"{prefix}:{source_name}:{type(exc).__name__}:{str(exc)[:80]}"
    last_ts = _safe_float(state.get(key), 0.0) if isinstance(state, dict) else 0.0
    if now_ts - last_ts >= 300:
        print(f"⚠️ {prefix}來源失敗，改試下一個: {source_name} | {exc}")
        if not isinstance(state, dict):
            state = {}
        state[key] = now_ts
        setattr(_log_kline_source_failure, "_state", state)


def _log_kline_fallback(source_name, prefix="K線"):
    now_ts = time.time()
    state = getattr(_log_kline_fallback, "_state", {})
    key = f"{prefix}:{source_name}"
    last_ts = _safe_float(state.get(key), 0.0) if isinstance(state, dict) else 0.0
    if now_ts - last_ts >= 300:
        print(f"♻️ Futures {prefix}不可用，改用 {source_name}")
        if not isinstance(state, dict):
            state = {}
        state[key] = now_ts
        setattr(_log_kline_fallback, "_state", state)


def _tradingview_session_id(prefix):
    digest = hashlib.sha1(f"{prefix}:{time.time()}:{os.getpid()}:{threading.get_ident()}".encode()).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _tradingview_message(method, params):
    payload = json.dumps({"m": method, "p": params}, separators=(",", ":"))
    return f"~m~{len(payload)}~m~{payload}"


def _parse_tradingview_messages(raw):
    messages = []
    text = str(raw or "")
    cursor = 0
    while True:
        start = text.find("~m~", cursor)
        if start < 0:
            break
        length_start = start + 3
        length_end = text.find("~m~", length_start)
        if length_end < 0:
            break
        try:
            payload_len = int(text[length_start:length_end])
        except Exception:
            cursor = length_end + 3
            continue
        payload_start = length_end + 3
        payload = text[payload_start : payload_start + payload_len]
        cursor = payload_start + payload_len
        try:
            messages.append(json.loads(payload))
        except Exception:
            continue
    return messages


def _tradingview_symbol(symbol):
    symbol = str(symbol or "ETHUSDT").upper().strip()
    if symbol in TRADINGVIEW_SYMBOL_MAP:
        return TRADINGVIEW_SYMBOL_MAP[symbol]
    if symbol.endswith("USDT"):
        return f"BINANCE:{symbol}.P"
    return f"BINANCE:{symbol}"


def _tradingview_bar_count(interval, limit, start_time_ms=None, end_time_ms=None):
    requested = max(1, min(10000, _safe_int(limit, 100)))
    interval_ms = KLINE_INTERVAL_MS.get(str(interval), 60 * 1000)
    if start_time_ms is not None:
        end_ms = int(_safe_float(end_time_ms, time.time() * 1000))
        span = max(0, end_ms - int(_safe_float(start_time_ms, 0.0)))
        requested = max(requested, int(span // max(1, interval_ms)) + 8)
    return max(1, min(10000, requested))


def _tradingview_requested_bars(interval, limit, start_time_ms=None, end_time_ms=None):
    requested = max(1, _safe_int(limit, 100))
    if str(interval) == "1M" and start_time_ms is None:
        requested = min(requested, max(60, _safe_int(os.getenv("TRADINGVIEW_MONTHLY_MAX_BARS", 120), 120)))
    interval_ms = KLINE_INTERVAL_MS.get(str(interval), 60 * 1000)
    if start_time_ms is not None:
        end_ms = int(_safe_float(end_time_ms, time.time() * 1000))
        span = max(0, end_ms - int(_safe_float(start_time_ms, 0.0)))
        requested = max(requested, int(span // max(1, interval_ms)) + 8)
    return max(1, requested)


def _tradingview_min_acceptable_bars(interval, requested_bars, start_time_ms=None):
    if start_time_ms is not None:
        return max(1, min(24, _safe_int(requested_bars, 100)))
    if str(interval) == "1M":
        return max(24, min(60, _safe_int(requested_bars, 100)))
    if str(interval) == "1w":
        return max(52, min(120, _safe_int(requested_bars, 100)))
    return max(30, min(_safe_int(requested_bars, 100), _safe_int(os.getenv("TRADINGVIEW_MIN_ACCEPTABLE_BARS", 120), 120)))


def _filter_tradingview_rows(parsed_all, start_time_ms=None, end_time_ms=None):
    parsed = []
    start_ms = int(_safe_float(start_time_ms, 0.0)) if start_time_ms is not None else None
    end_ms = int(_safe_float(end_time_ms, 0.0)) if end_time_ms is not None else None
    for row in parsed_all:
        open_ms = int(row[0])
        if start_ms is not None and open_ms < start_ms:
            continue
        if end_ms is not None and open_ms > end_ms:
            continue
        parsed.append(row)
    return parsed


def _parse_tradingview_series_rows(series, interval):
    parsed = []
    interval_ms = KLINE_INTERVAL_MS.get(str(interval), 60 * 1000)
    if not isinstance(series, list):
        return parsed
    for bar in series:
        values = bar.get("v") if isinstance(bar, dict) else None
        if not isinstance(values, list) or len(values) < 5:
            continue
        open_ms = int(_safe_float(values[0], 0.0) * 1000)
        volume = _safe_float(values[5], 0.0) if len(values) > 5 else 0.0
        parsed.append(
            [
                open_ms,
                str(_safe_float(values[1], 0.0)),
                str(_safe_float(values[2], 0.0)),
                str(_safe_float(values[3], 0.0)),
                str(_safe_float(values[4], 0.0)),
                str(volume),
                open_ms + interval_ms - 1,
                "0",
                0,
                "0",
                "0",
                "0",
            ]
        )
    parsed.sort(key=lambda row: row[0])
    return parsed


def _fetch_tradingview_kline_rows(symbol, interval, limit=100, start_time_ms=None, end_time_ms=None, timeout=10):
    tv_interval = TRADINGVIEW_INTERVAL_MAP.get(str(interval))
    if not tv_interval:
        raise RuntimeError(f"TradingView 不支援週期: {interval}")

    chart_session = _tradingview_session_id("cs")
    tv_symbol = _tradingview_symbol(symbol)
    requested_bars = _tradingview_requested_bars(interval, limit, start_time_ms=start_time_ms, end_time_ms=end_time_ms)
    min_acceptable_bars = _tradingview_min_acceptable_bars(interval, requested_bars, start_time_ms=start_time_ms)
    max_paged_bars = max(
        10000,
        min(750000, _safe_int(os.getenv("TRADINGVIEW_MAX_PAGED_BARS", 600000), 600000)),
    )
    if requested_bars > max_paged_bars:
        raise RuntimeError(
            f"TradingView requested {requested_bars} bars exceeds TRADINGVIEW_MAX_PAGED_BARS={max_paged_bars}"
        )
    bar_count = min(10000, requested_bars)
    ws = None
    try:
        ws = websocket.create_connection(
            TRADINGVIEW_WS_URL,
            timeout=max(3, _safe_int(timeout, 10)),
            header=["Origin: https://www.tradingview.com"],
        )
        symbol_payload = json.dumps(
            {"symbol": tv_symbol, "adjustment": "splits", "session": "extended"},
            separators=(",", ":"),
        )
        for method, params in (
            ("set_auth_token", ["unauthorized_user_token"]),
            ("chart_create_session", [chart_session, ""]),
            ("resolve_symbol", [chart_session, "symbol_1", f"={symbol_payload}"]),
            ("create_series", [chart_session, "s1", "s1", "symbol_1", tv_interval, bar_count]),
        ):
            ws.send(_tradingview_message(method, params))

        timeout_sec = max(5, _safe_int(timeout, 10))
        if requested_bars > 10000:
            timeout_sec = max(timeout_sec, min(240, 15 + requested_bars // 2500))
        try:
            ws.settimeout(max(5, min(15, timeout_sec)))
        except Exception:
            pass
        deadline = time.time() + timeout_sec
        accumulated_rows = {}
        last_loaded_count = 0
        last_oldest_open = None
        requested_more = False
        while time.time() < deadline:
            try:
                raw = ws.recv()
            except Exception as exc:
                if accumulated_rows:
                    parsed_all = [accumulated_rows[key] for key in sorted(accumulated_rows)]
                    parsed = _filter_tradingview_rows(parsed_all, start_time_ms=start_time_ms, end_time_ms=end_time_ms)
                    if len(parsed) >= min_acceptable_bars:
                        return parsed[-max(1, min(max_paged_bars, _safe_int(limit, len(parsed)))) :]
                    oldest_open = min(accumulated_rows)
                    raise RuntimeError(
                        f"TradingView returned only {len(parsed)} usable {interval} bars; "
                        f"oldest_open={oldest_open}, requested_start={start_time_ms}, requested_bars={requested_bars}"
                    ) from exc
                raise
            for message in _parse_tradingview_messages(raw):
                method = message.get("m")
                if method == "critical_error":
                    raise RuntimeError(f"TradingView critical_error: {message.get('p')}")
                if method != "timescale_update":
                    continue
                payload = message.get("p")
                if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], dict):
                    continue
                series = (payload[1].get("s1") or {}).get("s")
                if not isinstance(series, list):
                    continue
                parsed_all = _parse_tradingview_series_rows(series, interval)
                if not parsed_all:
                    continue
                for row in parsed_all:
                    accumulated_rows[int(row[0])] = row
                parsed_all = [accumulated_rows[key] for key in sorted(accumulated_rows)]
                oldest_open = int(parsed_all[0][0])
                loaded_count = len(parsed_all)
                needs_older_start = start_time_ms is not None and oldest_open > int(_safe_float(start_time_ms, 0.0))
                needs_more_limit = start_time_ms is None and loaded_count < min(requested_bars, max_paged_bars)
                if (needs_older_start or needs_more_limit) and loaded_count < max_paged_bars:
                    no_progress = (
                        requested_more
                        and loaded_count <= last_loaded_count
                        and oldest_open == last_oldest_open
                    )
                    if no_progress:
                        if needs_older_start:
                            raise RuntimeError(
                                "TradingView pagination made no progress before requested start time"
                            )
                        needs_more_limit = False
                    last_loaded_count = loaded_count
                    last_oldest_open = oldest_open
                    request_count = min(10000, max(1, min(requested_bars, max_paged_bars) - loaded_count))
                    if request_count > 0 and not no_progress:
                        ws.send(_tradingview_message("request_more_data", [chart_session, "s1", request_count]))
                        requested_more = True
                        continue
                if needs_older_start and loaded_count >= max_paged_bars:
                    raise RuntimeError(
                        f"TradingView pagination stopped at {loaded_count} bars before requested start time; "
                        f"increase TRADINGVIEW_MAX_PAGED_BARS above {max_paged_bars} or use a shorter window"
                    )

                parsed = _filter_tradingview_rows(parsed_all, start_time_ms=start_time_ms, end_time_ms=end_time_ms)
                if parsed:
                    return parsed[-max(1, min(max_paged_bars, _safe_int(limit, 100))) :]
        if accumulated_rows:
            parsed_all = [accumulated_rows[key] for key in sorted(accumulated_rows)]
            parsed = _filter_tradingview_rows(parsed_all, start_time_ms=start_time_ms, end_time_ms=end_time_ms)
            if len(parsed) >= min_acceptable_bars:
                return parsed[-max(1, min(max_paged_bars, _safe_int(limit, len(parsed)))) :]
        raise RuntimeError("TradingView K線逾時或無資料")
    finally:
        try:
            if ws is not None:
                ws.close()
        except Exception:
            pass


def _fetch_binance_kline_rows(symbol, interval, limit=100, start_time_ms=None, end_time_ms=None, timeout=10, prefix="K線"):
    params = {
        "symbol": str(symbol or "ETHUSDT").upper(),
        "interval": str(interval),
        "limit": max(1, min(1500, _safe_int(limit, 100))),
    }
    if start_time_ms is not None:
        params["startTime"] = int(_safe_float(start_time_ms, 0.0))
    if end_time_ms is not None:
        params["endTime"] = int(_safe_float(end_time_ms, 0.0))

    errors = []
    for source_name, url in BINANCE_KLINE_SOURCES:
        try:
            response = _binance_request_get(
                url,
                params=params,
                timeout=timeout,
                prefix=f"{prefix}:{source_name}",
            )
            response.raise_for_status()
            rows = response.json()
            if isinstance(rows, list) and rows:
                if source_name != "futures":
                    _log_kline_fallback(source_name, prefix=prefix)
                return rows, source_name
            errors.append(f"{source_name}: empty")
        except Exception as exc:
            errors.append(f"{source_name}: {exc}")
            _log_kline_source_failure(source_name, exc, prefix=prefix)
            continue
    raise RuntimeError("; ".join(errors) or f"{prefix}來源全部失敗")


KRAKEN_INTERVAL_MAP = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60,
    "4h": 240, "1d": 1440, "1w": 10080, "1M": 21600,
}
KRAKEN_KLINE_CACHE = {}
KRAKEN_REQUEST_LOCK = threading.Lock()
KRAKEN_LAST_REQUEST_TS = 0.0
COINBASE_GRANULARITY_MAP = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "1d": 86400}
TWELVE_DATA_INTERVAL_MAP = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "4h": "4h", "12h": "12h", "1d": "1day",
    "1w": "1week", "1M": "1month",
}
TWELVE_DATA_KLINE_CACHE = {}
TWELVE_DATA_REQUEST_LOCK = threading.Lock()
TWELVE_DATA_USAGE_STATE = {"day": "", "count": 0, "last_request_ts": 0.0}
TWELVE_DATA_USAGE_PATH = data_path("api_token_usage.json")


def _load_twelve_data_usage_state():
    try:
        payload = json.loads(TWELVE_DATA_USAGE_PATH.read_text(encoding="utf-8"))
        item = payload.get("twelve_data") if isinstance(payload, dict) else {}
        if isinstance(item, dict):
            TWELVE_DATA_USAGE_STATE.update(
                {
                    "day": str(item.get("day") or ""),
                    "count": max(0, _safe_int(item.get("count"), 0)),
                    "last_request_ts": max(0.0, _safe_float(item.get("last_request_ts"), 0.0)),
                }
            )
    except Exception:
        pass


def _save_twelve_data_usage_state():
    try:
        ensure_parent_dir(TWELVE_DATA_USAGE_PATH)
        payload = {}
        if TWELVE_DATA_USAGE_PATH.exists():
            try:
                payload = json.loads(TWELVE_DATA_USAGE_PATH.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        if not isinstance(payload, dict):
            payload = {}
        payload["twelve_data"] = {
            "day": TWELVE_DATA_USAGE_STATE["day"],
            "count": TWELVE_DATA_USAGE_STATE["count"],
            "last_request_ts": TWELVE_DATA_USAGE_STATE["last_request_ts"],
            "daily_limit": max(1, _safe_int(os.getenv("TWELVE_DATA_DAILY_REQUEST_LIMIT", 800), 800)),
        }
        tmp_path = TWELVE_DATA_USAGE_PATH.with_name(
            f".{TWELVE_DATA_USAGE_PATH.name}.{os.getpid()}.tmp"
        )
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(TWELVE_DATA_USAGE_PATH)
    except Exception as exc:
        print(f"⚠️ Twelve Data 用量狀態寫入失敗: {exc}")


_load_twelve_data_usage_state()


def _fetch_coinbase_kline_rows(symbol, interval, limit=100, timeout=10):
    granularity = COINBASE_GRANULARITY_MAP.get(str(interval))
    if granularity is None:
        raise RuntimeError(f"Coinbase不支援週期 {interval}")
    product = "BTC-USD" if str(symbol or "").upper().startswith("BTC") else "ETH-USD"
    response = requests.get(
        f"https://api.exchange.coinbase.com/products/{product}/candles",
        params={"granularity": granularity},
        headers={"User-Agent": "ETH-bot/1.0"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    rows = []
    for item in reversed(payload if isinstance(payload, list) else []):
        if not isinstance(item, list) or len(item) < 6:
            continue
        open_ms = int(_safe_float(item[0], 0.0) * 1000)
        rows.append([open_ms, item[3], item[2], item[1], item[4], item[5], open_ms + granularity * 1000 - 1, "0", 0, "0", "0", "0"])
    if not rows:
        raise RuntimeError("Coinbase candles empty")
    return rows[-max(1, min(300, _safe_int(limit, 100))) :]


def _fetch_twelve_data_kline_rows(
    symbol, interval, limit=100, start_time_ms=None, end_time_ms=None, timeout=10
):
    api_key = str(os.getenv("TWELVE_DATA_API_KEY", "") or "").strip()
    if not api_key:
        raise RuntimeError("Twelve Data API key 未設定")
    td_interval = TWELVE_DATA_INTERVAL_MAP.get(str(interval))
    if td_interval is None:
        raise RuntimeError(f"Twelve Data不支援週期 {interval}")

    pair = "BTC/USD" if str(symbol or "").upper().startswith("BTC") else "ETH/USD"
    selected_limit = max(1, min(5000, _safe_int(limit, 100)))
    cache_key = (
        pair, str(interval), selected_limit,
        int(_safe_float(start_time_ms, 0.0)), int(_safe_float(end_time_ms, 0.0)),
    )
    cached = TWELVE_DATA_KLINE_CACHE.get(cache_key)
    cache_ttl = max(10, KLINE_TTL.get(str(interval), 10))
    if cached and time.time() - _safe_float(cached[0], 0.0) < cache_ttl:
        return cached[1]

    params = {
        "symbol": pair,
        "interval": td_interval,
        "outputsize": selected_limit,
        "timezone": "UTC",
        "order": "ASC",
        "apikey": api_key,
    }
    if start_time_ms is not None:
        params["start_date"] = datetime.datetime.fromtimestamp(
            _safe_float(start_time_ms, 0.0) / 1000, tz=datetime.timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")
    if end_time_ms is not None:
        params["end_date"] = datetime.datetime.fromtimestamp(
            _safe_float(end_time_ms, 0.0) / 1000, tz=datetime.timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")

    with TWELVE_DATA_REQUEST_LOCK:
        today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        if TWELVE_DATA_USAGE_STATE["day"] != today:
            TWELVE_DATA_USAGE_STATE.update({"day": today, "count": 0})
        daily_limit = max(1, _safe_int(os.getenv("TWELVE_DATA_DAILY_REQUEST_LIMIT", 800), 800))
        if TWELVE_DATA_USAGE_STATE["count"] >= daily_limit:
            raise RuntimeError(f"Twelve Data每日請求上限已達 {daily_limit}")
        min_gap = max(0.25, _safe_float(os.getenv("TWELVE_DATA_REQUEST_MIN_GAP_SEC", 8.0), 8.0))
        wait_sec = min_gap - (time.time() - TWELVE_DATA_USAGE_STATE["last_request_ts"])
        if wait_sec > 0:
            time.sleep(wait_sec)
        response = requests.get(
            "https://api.twelvedata.com/time_series", params=params,
            headers={"User-Agent": "ETH-bot/1.0"}, timeout=timeout,
        )
        TWELVE_DATA_USAGE_STATE["last_request_ts"] = time.time()
        TWELVE_DATA_USAGE_STATE["count"] += 1
        _save_twelve_data_usage_state()
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("status") == "error":
        raise RuntimeError(f"Twelve Data error: {(payload or {}).get('message', 'invalid response')}")
    values = payload.get("values") if isinstance(payload.get("values"), list) else []
    interval_ms = KLINE_INTERVAL_MS.get(str(interval), 60_000)
    rows = []
    for item in values:
        if not isinstance(item, dict):
            continue
        try:
            stamp = datetime.datetime.strptime(str(item.get("datetime")), "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=datetime.timezone.utc
            )
            open_ms = int(stamp.timestamp() * 1000)
            rows.append([
                open_ms, item["open"], item["high"], item["low"], item["close"],
                item.get("volume") or "0", open_ms + interval_ms - 1,
                "0", 0, "0", "0", "0",
            ])
        except (KeyError, TypeError, ValueError):
            continue
    rows.sort(key=lambda row: row[0])
    if not rows:
        raise RuntimeError("Twelve Data candles empty")
    rows = rows[-selected_limit:]
    TWELVE_DATA_KLINE_CACHE[cache_key] = (time.time(), rows)
    return rows


def _fetch_kraken_kline_rows(symbol, interval, limit=100, start_time_ms=None, end_time_ms=None, timeout=10):
    global KRAKEN_LAST_REQUEST_TS
    kraken_interval = KRAKEN_INTERVAL_MAP.get(str(interval))
    if kraken_interval is None:
        raise RuntimeError(f"Kraken不支援週期 {interval}")
    pair = "XBTUSD" if str(symbol or "").upper().startswith("BTC") else "ETHUSD"
    cache_key = (pair, str(interval), int(_safe_float(start_time_ms, 0.0)), int(_safe_float(end_time_ms, 0.0)))
    cached = KRAKEN_KLINE_CACHE.get(cache_key)
    cache_ttl = KLINE_TTL.get(str(interval), 10)
    selected_limit = max(1, min(720, _safe_int(limit, 100)))
    if cached and time.time() - _safe_float(cached[0], 0.0) < cache_ttl and len(cached[1]) >= min(limit, 720):
        if start_time_ms is not None:
            return cached[1][:selected_limit]
        return cached[1][-selected_limit:]
    params = {"pair": pair, "interval": kraken_interval}
    if start_time_ms is not None:
        params["since"] = max(0, int(_safe_float(start_time_ms, 0.0) / 1000))
    with KRAKEN_REQUEST_LOCK:
        min_gap = max(1.0, _safe_float(os.getenv("KRAKEN_REQUEST_MIN_GAP_SEC", 3.0), 3.0))
        wait_sec = min_gap - (time.time() - KRAKEN_LAST_REQUEST_TS)
        if wait_sec > 0:
            time.sleep(wait_sec)
        response = requests.get("https://api.kraken.com/0/public/OHLC", params=params, timeout=timeout)
        KRAKEN_LAST_REQUEST_TS = time.time()
        first_payload = response.json() if response.ok else {}
        if any("Too many requests" in str(item) for item in first_payload.get("error", [])):
            time.sleep(max(5.0, min_gap * 2.0))
            response = requests.get("https://api.kraken.com/0/public/OHLC", params=params, timeout=timeout)
            KRAKEN_LAST_REQUEST_TS = time.time()
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(f"Kraken OHLC error: {payload['error']}")
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    raw_rows = next((value for key, value in result.items() if key != "last" and isinstance(value, list)), [])
    rows = []
    for item in raw_rows:
        if not isinstance(item, list) or len(item) < 7:
            continue
        open_ms = int(_safe_float(item[0], 0.0) * 1000)
        if end_time_ms is not None and open_ms > int(_safe_float(end_time_ms, 0.0)):
            continue
        close_ms = open_ms + kraken_interval * 60 * 1000 - 1
        rows.append([open_ms, item[1], item[2], item[3], item[4], item[6], close_ms, "0", 0, "0", "0", "0"])
    if str(interval) == "1M" and rows:
        monthly = {}
        for row in rows:
            stamp = datetime.datetime.fromtimestamp(row[0] / 1000, tz=datetime.timezone.utc)
            key = (stamp.year, stamp.month)
            if key not in monthly:
                monthly[key] = list(row)
            else:
                bucket = monthly[key]
                bucket[2] = str(max(_safe_float(bucket[2], 0.0), _safe_float(row[2], 0.0)))
                bucket[3] = str(min(_safe_float(bucket[3], 0.0), _safe_float(row[3], 0.0)))
                bucket[4] = row[4]
                bucket[5] = str(_safe_float(bucket[5], 0.0) + _safe_float(row[5], 0.0))
                bucket[6] = row[6]
        rows = list(monthly.values())
    if not rows:
        raise RuntimeError("Kraken OHLC empty")
    KRAKEN_KLINE_CACHE[cache_key] = (time.time(), rows)
    if start_time_ms is not None:
        return rows[:selected_limit]
    return rows[-selected_limit:]


def _fetch_market_kline_rows(
    symbol,
    interval,
    limit=100,
    start_time_ms=None,
    end_time_ms=None,
    timeout=10,
    prefix="K線",
    source_preference=None,
):
    errors = []
    source_preference = str(
        source_preference or os.getenv("MARKET_KLINE_SOURCE_PREFERENCE", "kraken_first")
    ).lower()
    if str(interval) == "12h":
        try:
            source_start = None
            if start_time_ms is not None:
                source_start = int(_safe_float(start_time_ms, 0.0)) - KLINE_INTERVAL_MS["12h"]
            source_rows, source_name = _fetch_market_kline_rows(
                symbol,
                "4h",
                limit=max(12, _safe_int(limit, 100) * 3 + 6),
                start_time_ms=source_start,
                end_time_ms=end_time_ms,
                timeout=timeout,
                prefix=f"{prefix}-4h合成",
            )
            buckets = {}
            for row in source_rows:
                try:
                    open_ms = int(_safe_float(row[0], 0.0))
                    bucket_ms = (open_ms // KLINE_INTERVAL_MS["12h"]) * KLINE_INTERVAL_MS["12h"]
                    bucket = buckets.setdefault(
                        bucket_ms,
                        {
                            "open_time": bucket_ms,
                            "open": _safe_float(row[1], 0.0),
                            "high": _safe_float(row[2], 0.0),
                            "low": _safe_float(row[3], 0.0),
                            "close": _safe_float(row[4], 0.0),
                            "volume": 0.0,
                            "last_open": open_ms,
                        },
                    )
                    if open_ms < bucket.get("first_open", open_ms):
                        bucket["open"] = _safe_float(row[1], bucket["open"])
                        bucket["first_open"] = open_ms
                    bucket["high"] = max(_safe_float(bucket.get("high"), 0.0), _safe_float(row[2], 0.0))
                    bucket["low"] = min(_safe_float(bucket.get("low"), _safe_float(row[3], 0.0)), _safe_float(row[3], 0.0))
                    if open_ms >= _safe_int(bucket.get("last_open"), 0):
                        bucket["close"] = _safe_float(row[4], bucket["close"])
                        bucket["last_open"] = open_ms
                    bucket["volume"] = _safe_float(bucket.get("volume"), 0.0) + _safe_float(row[5], 0.0)
                except Exception:
                    continue
            parsed = []
            for bucket_ms in sorted(buckets):
                if start_time_ms is not None and bucket_ms < int(_safe_float(start_time_ms, 0.0)):
                    continue
                if end_time_ms is not None and bucket_ms > int(_safe_float(end_time_ms, 0.0)):
                    continue
                bucket = buckets[bucket_ms]
                parsed.append(
                    [
                        bucket_ms,
                        str(_safe_float(bucket.get("open"), 0.0)),
                        str(_safe_float(bucket.get("high"), 0.0)),
                        str(_safe_float(bucket.get("low"), 0.0)),
                        str(_safe_float(bucket.get("close"), 0.0)),
                        str(_safe_float(bucket.get("volume"), 0.0)),
                        bucket_ms + KLINE_INTERVAL_MS["12h"] - 1,
                        "0",
                        0,
                        "0",
                        "0",
                        "0",
                    ]
                )
            if parsed:
                return parsed[-max(1, min(1500, _safe_int(limit, 100))) :], f"{source_name}_resampled_12h"
            errors.append("tradingview_4h_resampled_12h: empty")
        except Exception as exc:
            errors.append(f"tradingview_4h_resampled_12h: {exc}")
            _log_kline_source_failure("tradingview_4h_resampled_12h", exc, prefix=prefix)
        if not ALLOW_BINANCE_MARKET_DATA_FALLBACK:
            raise RuntimeError("; ".join(errors) or f"{prefix} TradingView 12h 合成失敗")

    if source_preference == "kraken_first":
        try:
            if str(interval) in COINBASE_GRANULARITY_MAP and start_time_ms is None and end_time_ms is None:
                rows = _fetch_coinbase_kline_rows(symbol, interval, limit=limit, timeout=timeout)
                return rows, "coinbase"
            else:
                rows = _fetch_kraken_kline_rows(
                    symbol, interval, limit=limit, start_time_ms=start_time_ms,
                    end_time_ms=end_time_ms, timeout=timeout,
                )
                return rows, "kraken"
        except Exception as exc:
            errors.append(f"kraken_first: {exc}")
            _log_kline_source_failure("kraken", exc, prefix=prefix)

    # Formal API-key fallback. Keep it ahead of TradingView so the bot does
    # not depend on an unauthenticated web session when primary feeds fail.
    if str(os.getenv("TWELVE_DATA_API_KEY", "") or "").strip():
        try:
            rows = _fetch_twelve_data_kline_rows(
                symbol, interval, limit=limit, start_time_ms=start_time_ms,
                end_time_ms=end_time_ms, timeout=timeout,
            )
            return rows, "twelve_data"
        except Exception as exc:
            errors.append(f"twelve_data: {exc}")
            _log_kline_source_failure("twelve_data", exc, prefix=prefix)

    if source_preference == "binance_first" and ALLOW_BINANCE_MARKET_DATA_FALLBACK:
        try:
            rows, source = _fetch_binance_kline_rows(
                symbol,
                interval,
                limit=limit,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                timeout=timeout,
                prefix=prefix,
            )
            if rows:
                return rows, f"binance_{source}"
            errors.append("binance_first: empty")
        except Exception as exc:
            errors.append(f"binance_first: {exc}")

    if _is_tradingview_in_cooldown(symbol, interval):
        errors.append("tradingview: cooldown")
    else:
        try:
            rows = _fetch_tradingview_kline_rows(
                symbol,
                interval,
                limit=limit,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                timeout=timeout,
            )
            if rows:
                TRADINGVIEW_FAILURE_COOLDOWN.pop(_tradingview_cooldown_key(symbol, interval), None)
                return rows, "tradingview"
            errors.append("tradingview: empty")
            _mark_tradingview_failure(symbol, interval)
        except Exception as exc:
            errors.append(f"tradingview: {exc}")
            _mark_tradingview_failure(symbol, interval)
            _log_kline_source_failure("tradingview", exc, prefix=prefix)

    if not ALLOW_BINANCE_MARKET_DATA_FALLBACK:
        raise RuntimeError("; ".join(errors) or f"{prefix} TradingView 來源失敗")

    rows, source = _fetch_binance_kline_rows(
        symbol,
        interval,
        limit=limit,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        timeout=timeout,
        prefix=prefix,
    )
    return rows, f"binance_{source}"


def get_kline(interval, limit=100):
    now = time.time()
    limit = max(1, min(1500, int(limit)))

    if interval in KLINE_CACHE:
        data, ts = KLINE_CACHE[interval]
        if now - ts < KLINE_TTL.get(interval, 10) and len(data) >= limit:
            return data

    try:
        data, _ = _fetch_market_kline_rows("ETHUSDT", interval, limit=limit, timeout=10, prefix=f"{interval} K線")
    except Exception:
        if interval in KLINE_CACHE:
            data, _ = KLINE_CACHE[interval]
            print(f"⚠️ {interval} K線來源失敗，沿用快取")
            return data
        raise

    df = pd.DataFrame(data, columns=[
        "time","open","high","low","close","volume",
        "_","_","_","_","_","_"
    ])
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)

    df = calc_indicators(df)

    KLINE_CACHE[interval] = (df, now)
    return df

# =============================
# 主邏輯（AI接管）
# =============================
def run_bot():
    global performance

    load_model()  # 加載所有模型，包括新聞模型
    if model is None:
        retrain_model()

    last_signal = None
    last_trade_time = 0
    TRADE_COOLDOWN = 300  # 冷卻加長（防洗單）
    MIN_PRICE_CHANGE = 0.002  # 至少0.2%價格變動才允許新單
    MIN_SIGNAL_DIFF = 0.05  # 信號差異門檻
    MIN_ACCEPT_RR = max(1.1, _safe_float(os.getenv("TRADE_MIN_ACCEPT_RR", 1.8), 1.8))
    MIN_NET_EDGE_RATE = max(0.0005, _safe_float(os.getenv("TRADE_MIN_NET_EDGE_RATE", 0.0012), 0.0012))
    EST_SLIPPAGE_RATE = max(0.0, _safe_float(os.getenv("TRADE_EST_SLIPPAGE_RATE", 0.0004), 0.0004))
    EST_HOLD_HOURS_FOR_COST = max(0.0, _safe_float(os.getenv("TRADE_EST_HOLD_HOURS", 6.0), 6.0))
    last_trade_signal = None  # 避免同一訊號重複開單
    losing_streak = 0
    MAX_LOSS_STREAK = 3
    last_entry_price = None
    last_direction = None
    last_binance_sync_ts = 0.0
    volume_spike = None
    pending_entry_confirmation = None
    entry_confirm_enabled = str(os.getenv("TRADE_ENTRY_CONFIRM_ENABLED", "1") or "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    print(
        "✅ 進場延遲確認: "
        f"{'啟用' if entry_confirm_enabled else '停用'} | "
        f"wait={os.getenv('TRADE_ENTRY_CONFIRM_MIN_WAIT_SEC', '45')}s | "
        f"new_5m={os.getenv('TRADE_ENTRY_CONFIRM_REQUIRE_NEW_5M', '1')}"
    )
    # trade_open 移除，改用 active_trade 控制是否可開單

    last_update_id = None

    # ===== V7 防洗單（訊號記憶）=====
    last_signal_cache = None
    if _get_follow_mode_enabled() and _is_real_copy_enabled():
        try:
            ok, _ = sync_active_trade_from_binance(send_notice=False)
            if ok and active_trade.get("open"):
                last_binance_sync_ts = time.time()
        except Exception as e:
            print(f"⚠️ 啟動時同步 Binance 倉位失敗: {e}")

    pending_training_sample = _load_pending_training_sample_state()
    if pending_training_sample and active_trade.get("open"):
        print("♻️ 已還原待學習樣本，將接續追蹤目前持倉的 TP/SL 結果")
    elif pending_training_sample and not active_trade.get("open"):
        pending_training_sample = None
        _clear_pending_training_sample_state()

    # 重啟時已存在的部位同樣算作當日已開單，避免每日保底重複加開。
    if active_trade.get("open"):
        _mark_daily_trade_opened("restored_position")
        sync_position_panel(_safe_float(WS_PRICE, active_trade.get("avg_entry", active_trade.get("entry", 0.0))))

    while True:
        # 每輪先建立空決策，避免新增的前置流程誤用尚未產生的交易快照，
        # 導致主迴圈持續失敗而無法評估或開倉。
        decision = {}
        try:
            # ===== Telegram 指令接收 =====
            commands, last_update_id = fetch_telegram_commands(last_update_id)

            for command in commands:
                try:
                    text = str(command.get("text", "") or "")
                    chat_id = command.get("chat_id")

                    if not text:
                        continue

                    _remember_notification_chat(chat_id)

                    if _handle_control_callback(text, chat_id):
                        continue

                    # AI / 新聞指令
                    context = {
                        "strategy_version": STRATEGY_VERSION,
                        "price": price if 'price' in locals() else None,
                        "score": score if 'score' in locals() else None,
                        "htf": htf if 'htf' in locals() else None,
                        "regime": regime if 'regime' in locals() else None,
                        "breakout": breakout if 'breakout' in locals() else None,
                        "triangle": triangle if 'triangle' in locals() else None,
                        "macro": macro_bias if 'macro_bias' in locals() else None,
                        "volume_spike": volume_spike if 'volume_spike' in locals() else None,
                        "chat_id": chat_id,
                        "user_id": command.get("user_id"),
                        "username": command.get("username"),
                        "first_name": command.get("first_name"),
                        "chat_type": command.get("chat_type"),
                    }
                    if isinstance(locals().get("higher_tf_context"), dict):
                        context.update(higher_tf_context)

                    reply = handle_ai_command(text, context)

                    if reply:
                        _send_telegram_message(chat_id, reply, include_control_panel=True)
                except:
                    pass
            # ===== HTF（方向）=====
            df_1mth = get_kline("1M", max(60, _safe_int(os.getenv("MONTHLY_KLINE_LIMIT", 120), 120)))
            df_1w = get_kline("1w", 110)
            df_1d = get_kline("1d", 370)
            df_4h = get_kline("4h", 200)
            df_1h = get_kline("1h", 170)

            # ===== Market Regime =====
            regime = detect_market_regime(df_1h, df_4h)

            # ===== 4H v2（方向 + 強度）=====
            trend_4h = df_4h["close"].iloc[-1] - df_4h["ma25"].iloc[-1]
            strength_4h = df_4h["ma25"].iloc[-1] - df_4h["ma25"].iloc[-5]

            htf = 1 if trend_4h > 0 else -1
            htf_strength = abs(strength_4h)

            # ===== MID（策略）=====
            df_30m = get_kline("30m")
            df_15m = get_kline("15m", 100)
            td_setup_15m = get_td_sequential_setup(df_15m)
            higher_tf_context = build_higher_timeframe_context(
                df_4h,
                df_1d,
                df_1w,
                df_1h=df_1h,
                df_15m=df_15m,
                df_1mth=df_1mth,
            )
            timeframe_patterns = {
                "fifteen_min": _detect_candlestick_pattern(df_15m),
                "one_hour": _detect_candlestick_pattern(df_1h),
                "four_hour": _detect_candlestick_pattern(df_4h),
                "daily": _detect_candlestick_pattern(df_1d),
                "weekly": _detect_candlestick_pattern(df_1w),
                "monthly": _detect_candlestick_pattern(df_1mth),
            }
            timeframe_kline_view = _build_timeframe_kline_view(
                higher_tf_context,
                timeframe_patterns,
            )
            higher_tf_context["timeframe_kline_view"] = timeframe_kline_view
            higher_tf_context["timeframe_kline_summary"] = timeframe_kline_view.get("summary", "")
            if record_higher_timeframe_context(higher_tf_context):
                print(
                    "🧭 已收集MLX多週期資料 | "
                    f"15m={higher_tf_context['fifteen_min_trend']:+d} | "
                    f"1H={higher_tf_context['one_hour_trend']:+d} | "
                    f"4H={higher_tf_context['four_hour_trend']:+d} | "
                    f"日線={higher_tf_context['daily_trend']:+d} | "
                    f"週線={higher_tf_context['weekly_trend']:+d} | "
                    f"月線={higher_tf_context['monthly_trend']:+d}"
                )

            mid_trend = 1 if df_30m["macd"].iloc[-1] > df_30m["signal"].iloc[-1] else -1
            fvg_low, fvg_high = calc_fvg(df_15m)

            # ===== LTF（進場）=====
            df_5m = get_kline("5m", 50)
            df_1m = get_kline("1m", 50)

            breakout = 0
            recent_high = df_5m["high"].iloc[-5:-1].max()
            recent_low = df_5m["low"].iloc[-5:-1].min()
            price = _validated_market_price(
                df_1m["close"].iloc[-1],
                reference_price=df_5m["close"].iloc[-1],
            )
            # Shadow analysis must stay independent from Binance.  The live
            # strategy may validate against Binance WebSocket prices, but
            # shadow entries and grading use the externally routed 1m candle.
            shadow_price = max(0.0, _safe_float(df_1m["close"].iloc[-1], 0.0))
            sr_analysis = analyze_multi_tf_sr(price)

            # ===== Macro（時事）=====
            sp_change, nq_change, btc_change, dxy_change, news_bias, event_risk, news_list = get_macro_bias()
            derivatives_flow = get_derivatives_flow_snapshot(COPY_TRADE_SYMBOL)
            if not derivatives_flow.get("stale"):
                POSITION_PANEL_STATE["funding_rate"] = _safe_float(derivatives_flow.get("funding_rate_live"), POSITION_PANEL_STATE.get("funding_rate", 0.0))
                POSITION_PANEL_STATE["funding_next_ts"] = _safe_int(derivatives_flow.get("funding_next_ts"), POSITION_PANEL_STATE.get("funding_next_ts", 0))
            atr = float(df_15m["high"].iloc[-1] - df_15m["low"].iloc[-1]) if len(df_15m) > 0 else 0.0

            if price > recent_high:
                breakout = 1
            elif price < recent_low:
                breakout = -1

            completed_15m = df_15m["time"].iloc[-2] if len(df_15m) > 1 else "unknown"
            range_support_15m, range_resistance_15m = _calc_support_resistance_levels(
                df_15m, lookback=96
            )
            range_support_1h, range_resistance_1h = _calc_support_resistance_levels(
                df_1h, lookback=168
            )
            atr_15m_avg = float(
                (df_15m["high"] - df_15m["low"]).tail(14).mean()
            )
            range_trade_reference = _build_range_trade_reference(
                range_support_15m,
                range_resistance_15m,
                range_support_1h,
                range_resistance_1h,
                atr_15m_avg,
            )
            auto_market_context = {
                "strategy_version": STRATEGY_VERSION,
                "price": shadow_price,
                "market_data_source": "external_kline_only",
                "analysis_timeframe": "1h",
                "sampling_timeframe": "15m",
                "htf": htf,
                "mid_trend": mid_trend,
                "one_hour_pattern": timeframe_patterns["one_hour"][0],
                "four_hour_pattern": timeframe_patterns["four_hour"][0],
                "daily_pattern": timeframe_patterns["daily"][0],
                "weekly_pattern": timeframe_patterns["weekly"][0],
                "daily_trend": higher_tf_context.get("daily_trend"),
                "daily_medium_change_pct": higher_tf_context.get("daily_medium_change_pct"),
                "weekly_trend": higher_tf_context.get("weekly_trend"),
                "weekly_medium_change_pct": higher_tf_context.get("weekly_medium_change_pct"),
                "monthly_pattern": timeframe_patterns["monthly"][0],
                "monthly_trend": higher_tf_context.get("monthly_trend"),
                "monthly_window_change_pct": higher_tf_context.get("monthly_window_change_pct"),
                "monthly_range_pos": higher_tf_context.get("monthly_range_pos"),
                "weekly_window_change_pct": higher_tf_context.get("weekly_window_change_pct"),
                "weekly_range_pos": higher_tf_context.get("weekly_range_pos"),
                "daily_window_change_pct": higher_tf_context.get("daily_window_change_pct"),
                "daily_range_pos": higher_tf_context.get("daily_range_pos"),
                "four_hour_window_change_pct": higher_tf_context.get("four_hour_window_change_pct"),
                "four_hour_range_pos": higher_tf_context.get("four_hour_range_pos"),
                "one_hour_trend": higher_tf_context.get("one_hour_trend"),
                "one_hour_window_change_pct": higher_tf_context.get("one_hour_window_change_pct"),
                "one_hour_range_pos": higher_tf_context.get("one_hour_range_pos"),
                "fifteen_min_trend": higher_tf_context.get("fifteen_min_trend"),
                "fifteen_min_window_change_pct": higher_tf_context.get("fifteen_min_window_change_pct"),
                "fifteen_min_range_pos": higher_tf_context.get("fifteen_min_range_pos"),
                "regime": regime,
                "breakout": breakout,
                "macro": _compute_macro_bias(
                    sp_change, nq_change, btc_change, dxy_change, news_bias, event_risk
                ),
                "volume_spike": bool(
                    df_15m["volume"].iloc[-1]
                    > df_15m["vol_ma20"].iloc[-1] * 1.5
                ),
                "rsi_15m": round(_safe_float(df_15m["rsi14"].iloc[-1], 50.0), 2),
                "ema50_deviation_15m": round(
                    (
                        shadow_price
                        / max(_safe_float(df_15m["ema50"].iloc[-1], shadow_price), 1e-9)
                        - 1
                    )
                    * 100,
                    3,
                ),
                "rsi_bucket": (
                    "overbought"
                    if _safe_float(df_15m["rsi14"].iloc[-1], 50.0) >= 70
                    else "oversold"
                    if _safe_float(df_15m["rsi14"].iloc[-1], 50.0) <= 30
                    else "neutral"
                ),
                "atr_15m": round(atr_15m_avg, 4),
                "range_support_15m": range_support_15m,
                "range_resistance_15m": range_resistance_15m,
                "range_support_1h": range_support_1h,
                "range_resistance_1h": range_resistance_1h,
                "range_trade_reference": range_trade_reference,
                "sr_bias": _safe_float(sr_analysis.get("bias"), 0.0),
                "sr_lines": "\n".join(sr_analysis.get("lines") or []),
                "timeframe_kline_view": timeframe_kline_view,
                "timeframe_kline_summary": timeframe_kline_view.get("summary", ""),
                "host_style_kline_order": timeframe_kline_view.get("order", ""),
                "timeframe_conflict": bool(timeframe_kline_view.get("conflict", False)),
            }
            _start_mlx_auto_analysis(
                f"ETHUSDT:15m:{completed_15m}",
                auto_market_context,
            )

            _process_sl_followup_reviews(df_1m, price)
            _process_pending_news_evaluations(price)

            scaling_vol_now = _safe_float(df_15m["volume"].iloc[-1], 0.0)
            scaling_vol_ma = _safe_float(
                df_15m["vol_ma20"].iloc[-1]
                if "vol_ma20" in df_15m.columns
                else df_15m["volume"].rolling(20).mean().iloc[-1],
                0.0,
            )
            scaling_volume_ratio = scaling_vol_now / (scaling_vol_ma + 1e-9)
            scaling_buy_pressure = bool(df_15m["close"].iloc[-1] > df_15m["open"].iloc[-1])
            scaling_sell_pressure = bool(df_15m["close"].iloc[-1] < df_15m["open"].iloc[-1])
            _update_scaling_market_state(
                price=price,
                atr=atr,
                htf=htf,
                mid_trend=mid_trend,
                regime=regime,
                breakout=breakout,
                sr_analysis=sr_analysis,
                volume_ratio=scaling_volume_ratio,
                volume_spike=scaling_vol_now > scaling_vol_ma * 1.5,
                buy_pressure=scaling_buy_pressure,
                sell_pressure=scaling_sell_pressure,
            )

            pending_training_sample = _maybe_backfill_pending_training_sample(
                pending_training_sample,
                df_4h=df_4h,
                df_1h=df_1h,
                df_30m=df_30m,
                df_15m=df_15m,
                df_5m=df_5m,
                price=price,
                sr_analysis=sr_analysis,
                sp_change=sp_change,
                nq_change=nq_change,
                btc_change=btc_change,
                dxy_change=dxy_change,
                news_bias=news_bias,
                event_risk=event_risk,
                last_signal=last_signal,
                losing_streak=losing_streak,
            )

            # ===== 🔥 即時新聞推送（任何時候都推送，不依賴是否持倉）=====
            if not hasattr(run_bot, "last_news_set"):
                run_bot.last_news_set = set()

            if not hasattr(run_bot, "startup_news_snapshot_sent"):
                run_bot.startup_news_snapshot_sent = False

            if news_list:
                POSITION_PANEL_STATE["latest_news"] = build_panel_news_items(news_list, limit=5)
                new_news = []

                for n in news_list:
                    if n and n not in run_bot.last_news_set:
                        new_news.append(n)

                if not run_bot.startup_news_snapshot_sent and news_list:
                    snapshot_header = (
                        "🧭 啟動新聞快照\n"
                        "━━━━━━━━━━━━━━\n"
                        f"已抓到 {len(news_list)} 則 RSS 即時訊息\n"
                        "━━━━━━━━━━━━━━"
                    )
                    print("\n" + snapshot_header)
                    run_bot.startup_news_snapshot_sent = True

                new_news = new_news[:15]

                if new_news:
                    now_time = datetime.datetime.now().strftime("%H:%M:%S")

                    for n in new_news:
                        msg_news = build_news_message(n, now_time)

                        # 🔥 過濾中性新聞
                        if "📊 解讀: 中性" in msg_news:
                            continue

                        print("\n" + msg_news)
                        if DISCORD_NEWS:
                            try:
                                _post_discord_webhook(DISCORD_NEWS, msg_news, timeout=5)
                            except Exception as _e:
                                print("Discord news error:", _e)
                        run_bot.last_news_set.add(n)

                    if len(run_bot.last_news_set) > 500:
                        run_bot.last_news_set = set(list(run_bot.last_news_set)[-200:])

            # ===== 即時新聞摘要（顯示在主訊號內）=====
            news_text = ""
            if news_list:
                news_text = "📰 重點新聞:\n"
                for n in news_list[:5]:
                    raw_item = str(n)
                    m = re.match(r"^\[([^\]]+)\]\s*(.*)$", raw_item)
                    if m:
                        src = m.group(1).strip()
                        body = m.group(2).strip()
                    else:
                        src = "News"
                        body = raw_item

                    zh_body = translate_news_to_zh(body)
                    preview = zh_body[:100] + ("..." if len(zh_body) > 100 else "")
                    news_text += f"- {preview}\n"

            # ===== 真實交易管理（TP/SL） =====
            evaluate_mlx_learning(shadow_price)
            _process_binance_host_learning(
                price,
                {
                    "price": price,
                    "regime": regime,
                    "htf": htf,
                    "mid_trend": mid_trend,
                    "breakout": breakout,
                    "macro_bias": _compute_macro_bias(
                        sp_change, nq_change, btc_change, dxy_change, news_bias, event_risk
                    ),
                    "news_bias": news_bias,
                    "event_risk": event_risk,
                    "symbol": "ETHUSDT",
                },
            )
            _process_binance_host_live_learning(
                price,
                {
                    "price": price,
                    "regime": regime,
                    "htf": htf,
                    "mid_trend": mid_trend,
                    "breakout": breakout,
                    "macro_bias": _compute_macro_bias(
                        sp_change, nq_change, btc_change, dxy_change, news_bias, event_risk
                    ),
                    "news_bias": news_bias,
                    "event_risk": event_risk,
                    "symbol": "ETHUSDT",
                },
            )
            if active_trade["open"]:
                position_direction = str(active_trade.get("direction") or "")
                td_position_warning = (
                    (position_direction == "long" and td_setup_15m["up_9"])
                    or (position_direction == "short" and td_setup_15m["down_9"])
                )
                if td_position_warning:
                    td_side = "上漲" if position_direction == "long" else "下跌"
                    td_count = td_setup_15m["up_count"] if position_direction == "long" else td_setup_15m["down_count"]
                    td_start = td_setup_15m["up_sequence_start"] if position_direction == "long" else td_setup_15m["down_sequence_start"]
                    td_warning_key = f"{position_direction}:{td_side}:{td_start}"
                    if getattr(run_bot, "last_td_position_warning_key", None) != td_warning_key:
                        warning = (
                            f"⚠️ 15m {td_side}九轉持倉警示\n"
                            f"目前{('多單' if position_direction == 'long' else '空單')}遇到{td_side} Setup {td_count}，趨勢可能衰竭。\n"
                            "僅警示：不會自動平倉或反手，請依 TP/SL 與其他訊號管理。"
                        )
                        print(warning)
                        _send_trade_notification(warning, priority=True)
                        run_bot.last_td_position_warning_key = td_warning_key

                # 實單跟單時定期回同步 Binance，確保面板與開倉點位/持倉量一致。
                if _get_follow_mode_enabled() and _is_real_copy_enabled() and (time.time() - last_binance_sync_ts) >= 12:
                    try:
                        ok, _ = sync_active_trade_from_binance(send_notice=False)
                        if ok:
                            last_binance_sync_ts = time.time()
                    except Exception:
                        pass

                current = price
                candle_high = float(df_1m["high"].iloc[-1]) if len(df_1m) > 0 else current
                candle_low = float(df_1m["low"].iloc[-1]) if len(df_1m) > 0 else current
                open_ts = _safe_float(active_trade.get("open_time"), 0.0)
                trade_horizon = _normalize_trade_time_horizon(active_trade.get("time_horizon"))
                custom_max_hold_sec = _safe_float(active_trade.get("max_hold_sec"), 0.0)
                max_hold_sec = custom_max_hold_sec if custom_max_hold_sec > 0 else _trade_max_hold_sec(trade_horizon)
                held_sec = time.time() - open_ts if open_ts > 0 else 0.0
                if open_ts > 0 and held_sec >= max_hold_sec:
                    held_direction = str(active_trade.get("direction") or "long")
                    held_entry = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), 0.0)
                    profitable = (
                        (held_direction == "long" and current > held_entry)
                        or (held_direction == "short" and current < held_entry)
                    )
                    horizon_label = _trade_time_horizon_label(trade_horizon)
                    max_hold_hours = max_hold_sec / 3600.0
                    max_hold_text = "7天" if max_hold_sec >= 7 * 24 * 3600 else "24小時"
                    closed, close_msg = _close_trade_at_max_hold(
                        current,
                        candle_high,
                        candle_low,
                        max_hold_label=f"{horizon_label}{max_hold_text}",
                    )
                    if closed:
                        if profitable:
                            performance["win"] += 1
                            outcome_label = 1
                            outcome_text = "盈利"
                        else:
                            performance["loss"] += 1
                            outcome_label = 0
                            outcome_text = "未盈利"
                        performance["total"] += 1
                        pending_training_sample = _finalize_pending_training_sample(
                            pending_training_sample, outcome_label, close_reason="MAX_HOLD", close_price=current, atr=atr,
                        )
                        last_signal_cache = None
                        _send_trade_notification(
                            _build_trade_close_message("MAX_HOLD", held_direction, current, candle_high, candle_low)
                            + f"\n⏱️ {horizon_label}持倉已達 {max_hold_text} 上限（{held_sec/3600.0:.1f}/{max_hold_hours:.1f}h），{outcome_text}也結束。\n"
                            + close_msg,
                            priority=True,
                        )
                    elif time.time() - getattr(run_bot, "last_max_hold_error_ts", 0.0) > 300:
                        print(close_msg)
                        _send_trade_notification(close_msg, priority=True)
                        run_bot.last_max_hold_error_ts = time.time()
                    continue
                has_valid_tp_sl = _has_valid_tp_sl(active_trade)

                if active_trade["direction"] == "long":
                    tp_hit = has_valid_tp_sl and ((current >= active_trade["tp"]) or (candle_high >= active_trade["tp"]))
                    sl_hit = has_valid_tp_sl and ((current <= active_trade["sl"]) or (candle_low <= active_trade["sl"]))

                    # 同根K同時觸發時採保守：先算SL，避免回測偏樂觀
                    if sl_hit:
                        performance["loss"] += 1
                        performance["total"] += 1
                        pending_training_sample = _finalize_pending_training_sample(
                            pending_training_sample,
                            0,
                            close_reason="SL",
                            close_price=current,
                            atr=atr,
                        )
                        record_position_close("SL", current, candle_high, candle_low)
                        sl_review = _review_stop_loss_event(
                            active_trade["direction"], active_trade.get("avg_entry", active_trade.get("entry")),
                            active_trade.get("tp"), active_trade.get("sl"), current, candle_high, candle_low, atr,
                            _build_sl_review_context_from_live(
                                htf=htf,
                                mid_trend=mid_trend,
                                sr_analysis=sr_analysis,
                                macro_bias=macro_bias,
                                derivatives_flow=derivatives_flow,
                                td_exhaustion=td_setup_15m["up_9"],
                                decision=locals().get("decision"),
                            ),
                        )
                        record_sl_review(sl_review)
                        active_trade["open"] = False
                        active_trade["size"] = 0.0
                        active_trade["position_qty"] = 0.0
                        active_trade["add_count"] = 0
                        active_trade["reduce_count"] = 0
                        active_trade["quick_reduce_count"] = 0
                        active_trade["quick_reduce_ts"] = 0.0
                        active_trade["daily_min_size_enforce_ts"] = 0.0
                        active_trade["open_time"] = None
                        active_trade["tp_sl_adjusted_4h"] = False
                        active_trade["time_horizon"] = "short"
                        _set_break_even_state(False)
                        last_signal_cache = None
                        losing_streak += 1
                        sync_position_panel(current)
                        print("❌ SL 命中")
                        _send_trade_notification(
                            _build_trade_close_message("SL", active_trade["direction"], current, candle_high, candle_low)
                            + f"\n\n🔎 SL檢討：{sl_review['verdict']}\n"
                            + f"風險 {sl_review['stop_atr']:.2f} ATR｜RR {sl_review['planned_rr']:.2f}｜技術分數 {sl_review['alignment_score']:+d}\n"
                            + "；".join(sl_review["issues"] or sl_review["indicators"])
                            + (
                                "\n策略優化: " + "；".join((sl_review.get("optimization_actions") or [])[:3])
                                if sl_review.get("optimization_actions") else ""
                            ),
                            priority=True,
                        )

                    elif tp_hit:
                        performance["win"] += 1
                        performance["total"] += 1
                        pending_training_sample = _finalize_pending_training_sample(
                            pending_training_sample,
                            1,
                            close_reason="TP",
                            close_price=current,
                            atr=atr,
                        )
                        record_position_close("TP", current, candle_high, candle_low)
                        active_trade["open"] = False
                        active_trade["size"] = 0.0
                        active_trade["position_qty"] = 0.0
                        active_trade["add_count"] = 0
                        active_trade["reduce_count"] = 0
                        active_trade["quick_reduce_count"] = 0
                        active_trade["quick_reduce_ts"] = 0.0
                        active_trade["daily_min_size_enforce_ts"] = 0.0
                        active_trade["open_time"] = None
                        active_trade["tp_sl_adjusted_4h"] = False
                        active_trade["time_horizon"] = "short"
                        _set_break_even_state(False)
                        last_signal_cache = None
                        losing_streak = 0
                        sync_position_panel(current)
                        print("✅ TP 命中")
                        _send_trade_notification(
                            _build_trade_close_message("TP", active_trade["direction"], current, candle_high, candle_low),
                            priority=True,
                        )

                elif active_trade["direction"] == "short":
                    tp_hit = has_valid_tp_sl and ((current <= active_trade["tp"]) or (candle_low <= active_trade["tp"]))
                    sl_hit = has_valid_tp_sl and ((current >= active_trade["sl"]) or (candle_high >= active_trade["sl"]))

                    # 同根K同時觸發時採保守：先算SL，避免回測偏樂觀
                    if sl_hit:
                        performance["loss"] += 1
                        performance["total"] += 1
                        pending_training_sample = _finalize_pending_training_sample(
                            pending_training_sample,
                            0,
                            close_reason="SL",
                            close_price=current,
                            atr=atr,
                        )
                        record_position_close("SL", current, candle_high, candle_low)
                        sl_review = _review_stop_loss_event(
                            active_trade["direction"], active_trade.get("avg_entry", active_trade.get("entry")),
                            active_trade.get("tp"), active_trade.get("sl"), current, candle_high, candle_low, atr,
                            _build_sl_review_context_from_live(
                                htf=htf,
                                mid_trend=mid_trend,
                                sr_analysis=sr_analysis,
                                macro_bias=macro_bias,
                                derivatives_flow=derivatives_flow,
                                td_exhaustion=td_setup_15m["down_9"],
                                decision=locals().get("decision"),
                            ),
                        )
                        record_sl_review(sl_review)
                        active_trade["open"] = False
                        active_trade["size"] = 0.0
                        active_trade["position_qty"] = 0.0
                        active_trade["add_count"] = 0
                        active_trade["reduce_count"] = 0
                        active_trade["quick_reduce_count"] = 0
                        active_trade["quick_reduce_ts"] = 0.0
                        active_trade["daily_min_size_enforce_ts"] = 0.0
                        active_trade["open_time"] = None
                        active_trade["tp_sl_adjusted_4h"] = False
                        active_trade["time_horizon"] = "short"
                        _set_break_even_state(False)
                        last_signal_cache = None
                        losing_streak += 1
                        sync_position_panel(current)
                        print("❌ SL 命中")
                        _send_trade_notification(
                            _build_trade_close_message("SL", active_trade["direction"], current, candle_high, candle_low)
                            + f"\n\n🔎 SL檢討：{sl_review['verdict']}\n"
                            + f"風險 {sl_review['stop_atr']:.2f} ATR｜RR {sl_review['planned_rr']:.2f}｜技術分數 {sl_review['alignment_score']:+d}\n"
                            + "；".join(sl_review["issues"] or sl_review["indicators"])
                            + (
                                "\n策略優化: " + "；".join((sl_review.get("optimization_actions") or [])[:3])
                                if sl_review.get("optimization_actions") else ""
                            ),
                            priority=True,
                        )

                    elif tp_hit:
                        performance["win"] += 1
                        performance["total"] += 1
                        pending_training_sample = _finalize_pending_training_sample(
                            pending_training_sample,
                            1,
                            close_reason="TP",
                            close_price=current,
                            atr=atr,
                        )
                        record_position_close("TP", current, candle_high, candle_low)
                        active_trade["open"] = False
                        active_trade["size"] = 0.0
                        active_trade["position_qty"] = 0.0
                        active_trade["add_count"] = 0
                        active_trade["reduce_count"] = 0
                        active_trade["quick_reduce_count"] = 0
                        active_trade["quick_reduce_ts"] = 0.0
                        active_trade["daily_min_size_enforce_ts"] = 0.0
                        active_trade["open_time"] = None
                        active_trade["tp_sl_adjusted_4h"] = False
                        active_trade["time_horizon"] = "short"
                        _set_break_even_state(False)
                        last_signal_cache = None
                        losing_streak = 0
                        sync_position_panel(current)
                        print("✅ TP 命中")
                        _send_trade_notification(
                            _build_trade_close_message("TP", active_trade["direction"], current, candle_high, candle_low),
                            priority=True,
                        )

            # 命中止盈止損前提下，持倉中允許補倉/減倉
            if active_trade["open"]:
                if _is_daily_min_position():
                    _enforce_daily_min_trade_size(
                        _safe_float(os.getenv("DAILY_MIN_TRADE_SIZE_RATIO", 0.05), 0.05),
                        current,
                    )
                quick_reduced = maybe_take_quick_profit_reduce(current, atr=atr)
                be_triggered = maybe_activate_auto_break_even(current, atr=atr)
                if not quick_reduced and not be_triggered:
                    manage_position_scaling(current, atr=atr)

            # ===== 持倉超過4小時，只縮減止盈範圍 =====
            if active_trade["open"]:
                maybe_shrink_tp_after_hold(current_price=price)

            # ===== 核心限制：未平倉禁止開新單，但新聞照常推 =====
            if active_trade["open"]:
                if not hasattr(run_bot, "last_position_status_ts"):
                    run_bot.last_position_status_ts = 0
                if not hasattr(run_bot, "last_news_monitor_ts"):
                    run_bot.last_news_monitor_ts = 0

                if time.time() - run_bot.last_position_status_ts > 15:
                    monitor_entry = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), 0.0)
                    print(
                        f"📡 持倉監控 | 方向: {active_trade['direction']} | 倉位: {int(_safe_float(active_trade.get('size'), 0)*100)}% | 現價: {price:.2f} | 進場均價: {monitor_entry:.2f} | TP: {active_trade['tp']:.2f} | SL: {active_trade['sl']:.2f}"
                    )
                    run_bot.last_position_status_ts = time.time()

                if time.time() - run_bot.last_news_monitor_ts > 30:
                    latest_news_preview = " | ".join(news_list[:4]) if news_list else "目前無新快訊（RSS 暫無資料）"
                    print(f"📰 新聞監控中 | {latest_news_preview}")
                    run_bot.last_news_monitor_ts = time.time()

                sync_position_panel(price)
                time.sleep(0.8)
                continue

            decision = build_trade_signal_snapshot(
                df_4h=df_4h,
                df_1h=df_1h,
                df_30m=df_30m,
                df_15m=df_15m,
                df_5m=df_5m,
                price=price,
                sr_analysis=sr_analysis,
                sp_change=sp_change,
                nq_change=nq_change,
                btc_change=btc_change,
                dxy_change=dxy_change,
                news_bias=news_bias,
                event_risk=event_risk,
                last_signal=last_signal,
                losing_streak=losing_streak,
                min_accept_rr=MIN_ACCEPT_RR,
                min_net_edge_rate=MIN_NET_EDGE_RATE,
                est_slippage_rate=EST_SLIPPAGE_RATE,
                est_hold_hours=EST_HOLD_HOURS_FOR_COST,
                derivatives_flow=derivatives_flow,
                df_1d=df_1d,
                df_1w=df_1w,
                df_1mth=df_1mth,
            )

            features = decision["features"]
            score = decision["score"]
            final = decision["final"]
            sl = decision["sl"]
            tp = decision["tp"]
            position_size = decision["position_size"]
            ai_prob = decision["ai_prob"]
            ai_long_prob = decision["ai_long_prob"]
            ai_short_prob = decision["ai_short_prob"]
            macro_bias = decision["macro_bias"]
            fake_breakout = decision["fake_breakout"]
            triangle = decision["triangle"]
            fvg_low = decision["fvg_low"]
            fvg_high = decision["fvg_high"]
            point_explain = decision["point_explain"]
            htf = decision["htf"]
            htf_strength = decision["htf_strength"]
            mid_trend = decision["mid_trend"]
            breakout = decision["breakout"]
            regime = decision["regime"]
            atr = decision["atr"]
            derivatives_pressure = _safe_float(decision.get("derivatives_pressure"), 0.0)
            taker_buy_ratio = _safe_float(decision.get("taker_buy_ratio"), 0.5)
            open_interest_change = _safe_float(decision.get("open_interest_change"), 0.0)
            net_edge_rate_est = _safe_float(decision.get("net_edge_rate_est"), 0.0)
            risk_rate = _safe_float(decision.get("risk_rate"), 0.0)
            repeated_support_tests = _safe_int(decision.get("repeated_support_tests"), 0)
            repeated_resistance_tests = _safe_int(decision.get("repeated_resistance_tests"), 0)
            repeated_test_pressure = _safe_float(decision.get("repeated_test_pressure"), 0.0)
            content_override = decision.get("content_override") if isinstance(decision.get("content_override"), dict) else {}
            host_opening_logic = decision.get("host_opening_logic") if isinstance(decision.get("host_opening_logic"), dict) else {}
            host_logic_applied = bool(decision.get("host_logic_applied", False))
            learned_entry_logic = decision.get("learned_entry_logic") if isinstance(decision.get("learned_entry_logic"), dict) else {}
            entry = price
            daily_min_trade = _daily_min_trade_due()
            daily_plan = {}
            if (
                daily_min_trade
                and _is_truthy(os.getenv("DAILY_MIN_DUE_ALLOW_QUALITY_SIGNAL", "1"))
                and not str(final or "").startswith("觀望")
                and get_signal_direction(final) in {"long", "short"}
                and not _daily_anchor_guard_should_wait(final, score, decision)
            ):
                daily_min_trade = False
            if daily_min_trade:
                daily_plan = _build_daily_min_trade_plan(
                    price,
                    atr,
                    df_15m,
                    df_5m,
                    htf,
                    mid_trend,
                    macro_bias=macro_bias,
                    news_bias=news_bias,
                    breakout=breakout,
                    volume_spike=bool(decision.get("volume_spike")),
                    regime=regime,
                    candlestick_turning=decision.get("candlestick_turning"),
                    df_1d=df_1d,
                )
                final = daily_plan["final"]
                sl = daily_plan["sl"]
                tp = daily_plan["tp"]
                position_size = daily_plan["position_size"]
                if final.startswith("觀望"):
                    daily_min_trade = False

            if (
                not daily_min_trade
                and _is_truthy(os.getenv("DAILY_MIN_ANCHOR_GUARD_ENABLED", "1"))
                and not bool(POSITION_PANEL_STATE.get("daily_trade_opened", False))
                and not _daily_min_trade_due()
                and not str(final).startswith("觀望")
            ):
                market_profile = decision.get("market_profile") if isinstance(decision.get("market_profile"), dict) else {}
                market_phase = str(market_profile.get("phase") or "range_base")
                final_direction = get_signal_direction(final)
                if market_phase == "bear" and final_direction == "long":
                    final = "觀望（熊市禁止非每日多單）"
                    position_size = 0.0
                elif _daily_anchor_guard_should_wait(final, score, decision):
                    final = "觀望（每日單錨定-等待保底）"
                    position_size = 0.0

            if not daily_min_trade:
                sl_guard_reason = _recent_sl_guard_reason(
                    final,
                    score,
                    net_edge_rate_est,
                    risk_rate,
                    macro_bias,
                    mid_trend,
                    _safe_float(sr_analysis.get("bias"), 0.0),
                )
                if sl_guard_reason:
                    final = sl_guard_reason
                sl_review_guard_reason = _recent_sl_review_guard_reason(final)
                if sl_review_guard_reason:
                    final = sl_review_guard_reason

            # ===== 中文時事解讀 =====
            macro_text = "中性"
            if event_risk >= 2:
                macro_text += "｜⚠️重大事件"
            elif event_risk == 1:
                macro_text += "｜⚠️波動增加"
            if macro_bias > 0:
                macro_text = "偏多（美股↑ / 美元↓ / 新聞利多）" + macro_text[2:] if macro_text.startswith("中性") else macro_text
            elif macro_bias < 0:
                macro_text = "偏空（美股↓ / 美元↑ / 新聞利空）" + macro_text[2:] if macro_text.startswith("中性") else macro_text

            # ===== AI判斷依據說明（技術來源強化版）=====
            reason = []

            # HTF 趨勢（MA25）
            if htf == 1:
                reason.append("多頭趨勢（4H MA25 上方）")
            else:
                reason.append("空頭趨勢（4H MA25 下方）")

            # MID 動能（MACD）
            if mid_trend == 1:
                reason.append("動能偏多（30m MACD > Signal）")
            else:
                reason.append("動能偏空（30m MACD < Signal）")

            # Breakout（結構突破）
            if breakout == 1:
                reason.append("突破高點（5m 結構突破）")
            elif breakout == -1:
                reason.append("跌破低點（5m 結構跌破）")

            # Regime（市場結構）
            if regime == "bull_trend_strong":
                reason.append("強多趨勢（1H+4H 同向 + 高波動）")
            elif regime == "bear_trend_strong":
                reason.append("強空趨勢（1H+4H 同向 + 高波動）")
            elif regime == "range":
                reason.append("盤整（趨勢不一致）")

            # Triangle（三角收斂）
            if triangle == 1:
                reason.append("三角收斂（高低點收斂）")

            # FVG（流動性缺口）
            if fvg_low and fvg_high:
                reason.append(f"FVG缺口（{fvg_low:.0f}-{fvg_high:.0f}）")

            # Macro（時事）
            if macro_bias > 0:
                reason.append("宏觀偏多（美股↑ / DXY↓）")
            elif macro_bias < 0:
                reason.append("宏觀偏空（美股↓ / DXY↑）")

            # News（時事）
            if news_bias > 0:
                reason.append("市場利多新聞（ETF/採用/上漲）")
            elif news_bias < 0:
                reason.append("市場利空新聞（監管/駭客/拋售）")

            # Event risk（重大波動）
            if event_risk >= 2:
                reason.append("重大事件（高波動風險）")
            elif event_risk == 1:
                reason.append("事件風險（波動提升）")

            # AI輸出
            reason.append(f"AI概率（{ai_prob:.2f}）")
            reason.append(f"AI長/短勝率（{ai_long_prob:.2f}/{ai_short_prob:.2f}）")
            if abs(derivatives_pressure) >= 0.10:
                flow_text = "偏多" if derivatives_pressure > 0 else "偏空"
                reason.append(
                    f"衍生品流向{flow_text}（壓力{derivatives_pressure:+.2f} / 買盤{taker_buy_ratio:.2f} / OI{open_interest_change*100:+.2f}%）"
                )
            if net_edge_rate_est:
                reason.append(f"AI期望值（{net_edge_rate_est*100:+.3f}%）")
            reason.append(
                f"多週期SR偏置（{_safe_float(sr_analysis.get('bias'), 0.0):+.2f} | 支撐{_safe_int(sr_analysis.get('support_hits'), 0)} / 壓力{_safe_int(sr_analysis.get('resistance_hits'), 0)}）"
            )
            if repeated_support_tests >= 2 or repeated_resistance_tests >= 2:
                reason.append(
                    f"連續測試壓力（支撐{repeated_support_tests} / 壓力{repeated_resistance_tests} | 方向壓力{repeated_test_pressure:+.2f}）"
                )
            if host_opening_logic:
                host_dir = {"long": "偏多", "short": "偏空", "neutral": "中性"}.get(
                    str(host_opening_logic.get("direction") or "neutral"),
                    "中性",
                )
                host_reasons = "、".join((host_opening_logic.get("reasons") or [])[:2])
                host_state = "主訊號" if host_logic_applied else "觀察"
                reason.append(
                    f"MLX開單邏輯{host_dir}（{host_state} | 信心{_safe_float(host_opening_logic.get('confidence'), 0.0):.2f}"
                    + (f" | {host_reasons}" if host_reasons else "")
                    + "）"
                )
            if learned_entry_logic:
                learned_dir = {"long": "偏多", "short": "偏空", "neutral": "中性"}.get(
                    str(learned_entry_logic.get("direction") or "neutral"),
                    "中性",
                )
                learned_reasons = "、".join((learned_entry_logic.get("reasons") or [])[:2])
                reason.append(
                    f"MLX學習邏輯{learned_dir}（多{_safe_float(learned_entry_logic.get('long_setup'), 0.0):.2f}/空{_safe_float(learned_entry_logic.get('short_setup'), 0.0):.2f}"
                    + (f" | {learned_reasons}" if learned_reasons else "")
                    + "）"
                )
            if content_override.get("usable"):
                override_dir_text = "偏多" if content_override.get("direction") == "long" else "偏空"
                if content_override.get("applied") and content_override.get("mode") == "primary":
                    override_state = "MLX主訊號"
                elif content_override.get("applied"):
                    override_state = "MLX覆蓋"
                else:
                    override_state = "MLX學習未覆蓋"
                reason.append(
                    f"MLX{override_dir_text}（{override_state} | 強度{_safe_int(content_override.get('strength'), 0)} | 品質{_safe_float(content_override.get('quality'), 0.0):.2f} | 驗證{_safe_float(content_override.get('validation_accuracy'), 0.0):.1f}%/{_safe_int(content_override.get('validation_evaluated'), 0)}筆 | 衝突{_safe_int(content_override.get('conflicts'), 0)}）"
                )

            # ===== 市場狀態中文轉換 =====
            regime_map = {
                "bull_trend_strong": "強多趨勢",
                "bull_trend": "多頭趨勢",
                "range": "盤整",
                "bear_trend_strong": "強空趨勢",
                "bear_trend": "空頭趨勢"
            }

            regime_text = regime_map.get(regime, regime)
            colored_regime_text = _format_direction_color_text(regime_text)
            colored_macro_text = _format_direction_color_text(macro_text)

            # 九轉只作為反向開倉的保守濾網，不以單一訊號自動反手。
            td_entry_block_reason = ""
            if not daily_min_trade and "做多" in final and td_setup_15m["up_9"]:
                td_entry_block_reason = f"觀望（15m上漲九轉 Setup {td_setup_15m['up_count']}）"
            elif not daily_min_trade and "做空" in final and td_setup_15m["down_9"]:
                td_entry_block_reason = f"觀望（15m下跌九轉 Setup {td_setup_15m['down_count']}）"
            if td_entry_block_reason:
                final = td_entry_block_reason
                reason.append("九轉反向開倉濾網（等待其他確認）")
            elif daily_min_trade:
                daily_note = "每日最低一單：22:30 後小倉位結構單"
                if isinstance(daily_plan, dict) and daily_plan.get("against_macro"):
                    daily_note += "（逆宏觀已縮小倉位）"
                reason.append(daily_note)
            elif td_setup_15m["up_9"]:
                reason.append(f"15m上漲九轉 Setup {td_setup_15m['up_count']}（封鎖新多單）")
            elif td_setup_15m["down_9"]:
                reason.append(f"15m下跌九轉 Setup {td_setup_15m['down_count']}（封鎖新空單）")

            reason_text = " | ".join(reason)

            # ===== 統一輸出訊號格式 =====
            if "做多" in final:
                display_signal = "🚀 做多"
            elif "做空" in final:
                display_signal = "🚀 做空"
            else:
                display_signal = final

            # ===== 訊息格式（進場優先顯示）=====
            msg = ""

            if not final.startswith("觀望"):
                msg += f"📍 進場: {entry:.2f}\n"

                if tp is not None:
                    msg += f"🎯 止盈: {tp:.2f}\n"
                else:
                    msg += "🎯 止盈: N/A\n"

                if sl is not None:
                    msg += f"🛑 止損: {sl:.2f}\n"
                else:
                    msg += "🛑 止損: N/A\n"

                msg += f"💰 倉位: {int(position_size*100)}%\n\n"

            msg += (
                f"🤖 AI信號：{display_signal}\n"
                f"📊 信心值: {ai_prob:.2f}\n"
                f"📈 {_format_tp_sl_win_rate_line(performance)}\n"
                f"🌍 市場狀態: {colored_regime_text}\n"
                f"📰 時事判斷: {colored_macro_text}\n"
                f"{news_text}"
                f"🧠 判斷依據: {reason_text}"
            )
            if point_explain:
                msg += f"\n{point_explain}"
            sr_lines = sr_analysis.get("lines") if isinstance(sr_analysis, dict) else []
            if isinstance(sr_lines, list) and sr_lines:
                msg += "\n\n🧱 多週期支撐/壓力（K線型態）\n" + "\n".join([f"- {line}" for line in sr_lines[:5]])
            # Fix spam log（觀望不要一直 print）
            if not final.startswith("觀望"):
                print(msg)

            # 強制單也必須再次經過自動修正，避免繞過前面的保護
            if not final.startswith("觀望"):
                final, sl, tp = auto_fix_trade_plan(final, entry, sl, tp, atr)

            # ===== 開單頻率 + 訊號去重（核心修正）=====
            now_ts = time.time()

            TRADE_COOLDOWN = 300  # 拉長避免過度交易

            # 先做去重與冷卻判斷，再決定是否跳過

            # ===== 同方向去重（用方向，不用字串） =====
            current_direction = get_signal_direction(final)
            last_direction_simple = get_signal_direction(last_trade_signal) if last_trade_signal else None

            # ===== 防洗單 v6 =====
            if not daily_min_trade and current_direction == last_direction_simple:
                # 價格變動太小 → 不開單
                if last_entry_price is not None:
                    price_change = abs(price - last_entry_price) / price
                    if price_change < MIN_PRICE_CHANGE:
                        final = "觀望（防洗單-價格過近）"

                # 信號變化太小 → 不開單
                if last_signal is not None:
                    if abs(score - last_signal) < MIN_SIGNAL_DIFF:
                        final = "觀望（防洗單-信號重複）"

            if not final.startswith("觀望") and not daily_min_trade:
                # ===== 冷卻防洗單 =====
                if now_ts - last_trade_time < TRADE_COOLDOWN:
                    final = "觀望（冷卻中）"

                # ===== 價格變動過小 =====
                elif last_entry_price is not None:
                    price_change = abs(price - last_entry_price) / price
                    if price_change < MIN_PRICE_CHANGE:
                        final = "觀望（價格未達門檻）"

            sync_position_panel(price)

            # ===== 最終過濾 =====
            if final.startswith("觀望"):
                pending_entry_confirmation = None
                continue

            # ===== 最終安全檢查：拒絕假突破低信心單 =====
            if not daily_min_trade and fake_breakout and abs(score - 0.5) < 0.22:
                pending_entry_confirmation = None
                continue

            # ===== 多週期壓力/支撐阻擋：靠近關鍵反向區域先觀望 =====
            support_hits = _safe_int(sr_analysis.get("support_hits"), 0)
            resistance_hits = _safe_int(sr_analysis.get("resistance_hits"), 0)
            if (
                not daily_min_trade
                and "做多" in final
                and resistance_hits >= 2
                and repeated_resistance_tests < 2
                and score < 0.72
            ):
                pending_entry_confirmation = None
                continue
            if (
                not daily_min_trade
                and "做空" in final
                and support_hits >= 2
                and repeated_support_tests < 2
                and score > 0.28
            ):
                pending_entry_confirmation = None
                continue

            if not final.startswith("觀望"):
                direction = "long" if "做多" in final else "short"

                if entry_confirm_enabled and not daily_min_trade:
                    candle_id = _get_entry_confirm_candle_id(df_5m)
                    confirmed, confirm_msg = _evaluate_pending_entry_confirmation(
                        pending_entry_confirmation,
                        direction,
                        price,
                        score,
                        candle_id,
                        now_ts,
                    )
                    if not confirmed:
                        keep_waiting = (
                            pending_entry_confirmation
                            and direction == pending_entry_confirmation.get("direction")
                            and str(confirm_msg).startswith("等待")
                        )
                        if not keep_waiting:
                            pending_entry_confirmation = {
                                "direction": direction,
                                "price": float(price),
                                "score": float(score),
                                "final": final,
                                "ts": now_ts,
                                "candle_id": candle_id,
                            }
                        print(f"⏳ 進場延遲確認 | {direction} | {confirm_msg} | 價格 {price:.2f}")
                        sync_position_panel(price)
                        continue
                    print(f"✅ 進場延遲確認 | {direction} | {confirm_msg}")
                    pending_entry_confirmation = None

                # 保險：再次確認沒有持倉
                if active_trade["open"]:
                    continue

                # 防止同一訊號重複刷
                if not daily_min_trade and last_signal_cache == msg:
                    continue

                # ===== 建立真實交易 =====

                active_trade["direction"] = direction
                active_trade["entry"] = float(entry)
                active_trade["avg_entry"] = float(entry)
                active_trade["tp"] = tp
                active_trade["sl"] = sl
                active_trade["trade_source"] = "daily_minimum" if daily_min_trade else "signal"
                active_trade["candlestick_turn_direction"] = str(
                    (decision.get("candlestick_turning") or {}).get("direction") or "neutral"
                )
                active_trade["candlestick_turn_count"] = _safe_int(decision.get("candlestick_turn_count"), 0)
                active_trade["candlestick_turn_confidence"] = _safe_float(
                    decision.get("candlestick_turn_confidence"), 0.0
                )
                base_size = _safe_float(position_size, 0.0)
                if base_size <= 0:
                    base_size = 0.2
                base_size = _cap_initial_position_size(base_size)
                planned_open_size = base_size
                active_trade["size"] = float(min(1.0, max(base_size, 0.1)))
                if daily_min_trade:
                    active_trade["size"] = float(max(base_size, 0.01))
                max_size, min_size = _derive_scaling_bounds(active_trade["size"])
                if daily_min_trade and daily_plan.get("max_position_size") is not None:
                    max_size = max(
                        active_trade["size"],
                        _safe_float(daily_plan.get("max_position_size"), active_trade["size"]),
                    )
                active_trade["max_size"] = max_size
                active_trade["min_size"] = min_size
                active_trade["add_count"] = 0
                active_trade["reduce_count"] = 0
                active_trade["quick_reduce_count"] = 0
                active_trade["quick_reduce_ts"] = 0.0
                active_trade["daily_min_size_enforce_ts"] = 0.0
                active_trade["last_adjust_ts"] = 0.0
                active_trade["scale_add_paused"] = False
                active_trade["scale_add_pause_reason"] = ""
                active_trade["scale_add_pause_ts"] = 0.0
                active_trade["last_scale_skip_notify_key"] = ""
                active_trade["last_scale_skip_notify_ts"] = 0.0
                active_trade["open_time"] = time.time()
                active_trade["tp_sl_adjusted_4h"] = False
                active_trade["time_horizon"] = _infer_trade_time_horizon(
                    final=final,
                    regime=regime,
                    htf=htf,
                    mid_trend=mid_trend,
                    daily_min_trade=daily_min_trade,
                )
                active_trade["max_hold_sec"] = _safe_float(
                    daily_plan.get("max_hold_sec") if daily_min_trade else 0.0,
                    0.0,
                )
                _set_break_even_state(False)
                # 注意：active_trade["open"] 尚未設為 True，等待跟單確認後再設定

                print("📤 發送 Telegram")
                send_telegram(msg, priority=True)
                last_signal_cache = msg
                last_trade_time = now_ts
                last_trade_signal = final
                last_entry_price = price
                last_direction = final

                if _get_follow_mode_enabled():
                    copy_ok, copy_msg = execute_copy_trade_open(
                        direction=direction,
                        size_ratio=active_trade.get("size", 0.0),
                        tp=active_trade.get("tp"),
                        sl=active_trade.get("sl"),
                    )
                    send_private_telegram(copy_msg, priority=True)
                    if copy_ok:
                        learn_features = _build_directional_learning_features(features, direction)
                        active_trade["open"] = True
                        _mark_daily_trade_opened("daily_minimum" if daily_min_trade else "signal")
                        sync_active_trade_from_binance(send_notice=False)
                        if daily_min_trade:
                            correction_msg = _enforce_daily_min_trade_size(planned_open_size, price)
                            if correction_msg:
                                msg += f"\n\n{correction_msg}"
                        mlx_episode_id = record_actual_trade_open(
                            direction=direction,
                            entry_price=_safe_float(active_trade.get("avg_entry", active_trade.get("entry")), entry),
                            tp_price=_safe_float(active_trade.get("tp"), tp or 0.0),
                            sl_price=_safe_float(active_trade.get("sl"), sl or 0.0),
                            market=_build_actual_trade_mlx_market(
                                decision,
                                direction,
                                source="copy_trade",
                                daily_min_trade=daily_min_trade,
                            ),
                            reason_text=reason_text,
                            opened_at=now_ts,
                            source="copy_trade",
                        )
                        pending_training_sample = {
                            "features": dict(features),
                            "learn_features": dict(learn_features),
                            "direction": direction,
                            "entry_ts": now_ts,
                            "mlx_episode_id": _safe_int(mlx_episode_id, 0),
                        }
                        pending_training_sample = _save_pending_training_sample_state(pending_training_sample)
                    else:
                        # 開單失敗，清除本地倉位狀態，避免面板顯示假倉位
                        pending_training_sample = None
                        _reset_active_trade_state()
                        sync_position_panel(price)
                else:
                    # 未開啟跟單，僅本地追蹤
                    learn_features = _build_directional_learning_features(features, direction)
                    mlx_episode_id = record_actual_trade_open(
                        direction=direction,
                        entry_price=entry,
                        tp_price=tp or 0.0,
                        sl_price=sl or 0.0,
                        market=_build_actual_trade_mlx_market(
                            decision,
                            direction,
                            source="local_tracking",
                            daily_min_trade=daily_min_trade,
                        ),
                        reason_text=reason_text,
                        opened_at=now_ts,
                        source="local_tracking",
                    )
                    pending_training_sample = {
                        "features": dict(features),
                        "learn_features": dict(learn_features),
                        "direction": direction,
                        "entry_ts": now_ts,
                        "mlx_episode_id": _safe_int(mlx_episode_id, 0),
                    }
                    pending_training_sample = _save_pending_training_sample_state(pending_training_sample)
                    active_trade["open"] = True
                    _mark_daily_trade_opened("daily_minimum" if daily_min_trade else "signal")
                    sync_position_panel(price)

            # ===== 學習標籤改為真實 TP/SL，避免用 1.2 秒短噪音污染模型 =====
            new_price = WS_PRICE if WS_PRICE else price

            # 定期重訓 batch model；每次平倉樣本都會先即時落盤
            maybe_train_model_periodically(force=False)

            # ===== 更新信號（平滑 + 防洗單記錄）=====
            last_signal = score
            # ===== 每日策略勝率巡檢（預設台北時間 23:50） =====
            report_time = str(os.getenv("STRATEGY_DAILY_REPORT_TIME", "23:50") or "23:50").strip()
            try:
                report_hour, report_minute = [int(part) for part in report_time.split(":", 1)]
            except (TypeError, ValueError):
                report_hour, report_minute = 23, 50
            report_hour = max(0, min(23, report_hour))
            report_minute = max(0, min(59, report_minute))
            report_now = datetime.datetime.now(ZoneInfo("Asia/Taipei"))
            report_date = report_now.strftime("%Y-%m-%d")
            if (
                (report_now.hour, report_now.minute) >= (report_hour, report_minute)
                and not daily_report_was_sent(report_date)
                and time.time() - getattr(run_bot, "last_daily_report_attempt_ts", 0.0) >= 300
            ):
                run_bot.last_daily_report_attempt_ts = time.time()
                report = build_daily_strategy_report()
                if send_telegram(report):
                    mark_daily_report_sent(report_date)

            sync_position_panel(new_price)
            time.sleep(0.8)

        except Exception as e:
            err_text = repr(e)
            if "TradingView" in err_text or "tradingview" in err_text:
                now_ts = time.time()
                last_ts = _safe_float(getattr(run_bot, "last_tradingview_error_log_ts", 0.0), 0.0)
                if now_ts - last_ts >= 300:
                    print("⚠️ TradingView 暫時不可用，已改用快取/等待下一輪:", err_text)
                    run_bot.last_tradingview_error_log_ts = now_ts
            else:
                print("error:", err_text)
            time.sleep(3)

# =============================
# start
# =============================
if __name__ == "__main__":
    print("🔥 AI 接管版啟動")
    run_bot()
