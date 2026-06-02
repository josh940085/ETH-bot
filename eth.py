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
import pandas as pd
import numpy as np
import threading
import websocket
import json
import pickle
import os
import re
import html
try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None
from collections import deque
from pathlib import Path
from urllib.parse import urlencode, urlparse, urlunparse
import xml.etree.ElementTree as ET

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
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

TELEGRAM_TOKEN = _get_required_env("TELEGRAM_TOKEN", "", mask=True)
TELEGRAM_CHAT_ID = _get_required_env("TELEGRAM_CHAT_ID", "", warn_if_missing=False)
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
DEFAULT_PAIR = "ETHUSDT"
DEFAULT_LEV = 10
COPY_TRADE_MAX_LEVERAGE = 5
PANEL_DEFAULT_MAX_MARGIN_USDT = 100.0
DEFAULT_MINI_APP_URL = "https://josh940085.github.io/ETH-bot/"
COPY_TRADE_SYMBOL = "ETHUSDT"
COPY_TRADE_MIN_QTY = 0.01
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
    active_trade["open_time"] = None
    active_trade["tp_sl_adjusted_4h"] = False
    _set_break_even_state(False)
    _clear_pending_training_sample_state()
    sync_position_panel(current)
    return current, binance_close_msg


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
        }

    close_hits = raw.get("close_hits")
    if not isinstance(close_hits, list):
        close_hits = []

    return {
        "last_close_reason": str(raw.get("last_close_reason", "") or "").upper(),
        "last_close_price": float(raw.get("last_close_price", 0.0) or 0.0),
        "last_close_ts": int(raw.get("last_close_ts", 0) or 0),
        "last_close_candle_high": float(raw.get("last_close_candle_high", 0.0) or 0.0),
        "last_close_candle_low": float(raw.get("last_close_candle_low", 0.0) or 0.0),
        "close_hits": close_hits[:10],
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
        if OPENAI_API_KEY:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "你是專業翻譯員。請把輸入內容翻成繁體中文，只輸出翻譯結果，不要補充說明。"
                    },
                    {
                        "role": "user",
                        "content": short_src
                    }
                ],
                "temperature": 0
            }

            res = requests.post(url, headers=headers, json=payload, timeout=6)
            data = res.json() if res is not None else {}
            zh = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

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

        # 4. 外匯分析 - 替換失效來源
        ("https://www.forexlive.com/feed/", "ForexLive"),
        ("https://www.investing.com/rss/forex.rss", "Technical Analysis"),
    ]

    aggregated = []
    for feed_url, source_name in feeds:
        try:
            aggregated.extend(fetch_rss_news(feed_url, source_name))
        except Exception as e:
            now_err = time.time()
            key = f"rss_err_{source_name.lower()}"
            last_err = getattr(fetch_macro_rss_news, key, 0)
            if now_err - last_err > 60:
                print(f"⚠️ {source_name} RSS error:", repr(e))
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

    try:
        oi_resp = HTTP_SESSION.get(
            "https://fapi.binance.com/fapi/v1/openInterest",
            params={"symbol": symbol},
            timeout=5,
        )
        oi_resp.raise_for_status()
        oi_data = oi_resp.json()
        current_oi = max(0.0, _safe_float((oi_data or {}).get("openInterest"), 0.0))
        previous_oi = _safe_float(previous.get("open_interest"), 0.0)
        snapshot["open_interest"] = current_oi
        if previous_oi > 0 and current_oi > 0:
            snapshot["open_interest_change"] = max(-1.0, min(1.0, (current_oi - previous_oi) / previous_oi))

        premium_resp = HTTP_SESSION.get(
            "https://fapi.binance.com/fapi/v1/premiumIndex",
            params={"symbol": symbol},
            timeout=5,
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

        taker_resp = HTTP_SESSION.get(
            "https://fapi.binance.com/futures/data/takerlongshortRatio",
            params={"symbol": symbol, "period": "5m", "limit": 2},
            timeout=5,
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
        exchange_info_res = HTTP_SESSION.get("https://api.binance.com/api/v3/exchangeInfo", timeout=10)
        exchange_info = exchange_info_res.json() if exchange_info_res.ok else {}
        price_res = HTTP_SESSION.get("https://api.binance.com/api/v3/ticker/price", timeout=10)
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
        request_func = HTTP_SESSION.get
    elif method_upper == "POST":
        request_func = HTTP_SESSION.post
    elif method_upper == "DELETE":
        request_func = HTTP_SESSION.delete
    else:
        raise ValueError(f"Unsupported method: {method}")
    res = request_func(
        f"https://fapi.binance.com{path}",
        params={**query, "signature": signature},
        headers={"X-MBX-APIKEY": api_key},
        timeout=10,
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
        request_func = HTTP_SESSION.get
    elif method_upper == "POST":
        request_func = HTTP_SESSION.post
    elif method_upper == "DELETE":
        request_func = HTTP_SESSION.delete
    else:
        raise ValueError(f"Unsupported method: {method}")

    res = request_func(
        f"https://api.binance.com{path}",
        params={**query, "signature": signature},
        headers={"X-MBX-APIKEY": api_key},
        timeout=10,
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
    active_trade["last_adjust_ts"] = _safe_float(active_trade.get("last_adjust_ts"), 0.0) if preserve_local_state else 0.0
    active_trade["scale_add_paused"] = bool(active_trade.get("scale_add_paused", False)) if preserve_local_state else False
    active_trade["scale_add_pause_reason"] = str(active_trade.get("scale_add_pause_reason") or "") if preserve_local_state else ""
    active_trade["scale_add_pause_ts"] = _safe_float(active_trade.get("scale_add_pause_ts"), 0.0) if preserve_local_state else 0.0
    prev_open_time = _safe_float(active_trade.get("open_time"), 0.0)
    active_trade["open_time"] = prev_open_time if preserve_local_state and prev_open_time > 0 else time.time()
    active_trade["tp_sl_adjusted_4h"] = bool(active_trade.get("tp_sl_adjusted_4h", False)) if preserve_local_state else False
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
    if reason_text not in {"TP", "SL", "MANUAL"}:
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
            "max_size": round(_safe_float(active_trade.get("max_size"), 1.0), 4),
            "min_size": round(_safe_float(active_trade.get("min_size"), 0.1), 4),
            "add_count": _safe_int(active_trade.get("add_count"), 0),
            "reduce_count": _safe_int(active_trade.get("reduce_count"), 0),
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
            "max_size": round(_safe_float(active_trade.get("max_size"), 1.0), 4),
            "min_size": round(_safe_float(active_trade.get("min_size"), 0.1), 4),
            "add_count": 0,
            "reduce_count": 0,
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
}


def _update_scaling_market_state(price, atr, htf, mid_trend, regime, breakout, sr_analysis=None):
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


def _assess_scaling_action(direction, entry, current_price, tp, sl, reduce=False):
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

    # 減倉評估：至少要能鎖定一段淨利，避免手續費把利潤吃掉。
    one_way_exit_cost = max(0.0, fee_round_trip_rate * 0.5 + est_slippage_rate * 0.5)
    lock_net_rate = pnl_rate - one_way_exit_cost
    min_lock_rate = max(0.0002, _safe_float(os.getenv("SCALE_REDUCE_MIN_LOCK_NET_RATE", 0.0008), 0.0008))

    metrics = {
        "pnl_rate": pnl_rate,
        "one_way_exit_cost": one_way_exit_cost,
        "lock_net_rate": lock_net_rate,
    }

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

    add_trigger = (
        not break_even_active
        and not scale_add_paused
        and add_count < max_add_count
        and size < max_size - 1e-9
        and min_add_drawdown <= drawdown_progress <= max_add_drawdown
        and trend_score >= min_add_trend_score
        and not opposing_pressure
    )

    reduce_trigger = (
        reduce_count < max_reduce_count
        and size > min_size + 1e-9
        and min_reduce_progress <= profit_progress <= max_reduce_progress
        and earned_r_multiple >= min_reduce_r_multiple
        and (opposing_pressure or trend_score <= max_reduce_trend_score)
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
                if real_copy_enabled:
                    send_private_telegram(note, priority=True)
                else:
                    send_telegram(note, priority=True)
                return

            if real_copy_enabled:
                ok, scale_msg = _execute_copy_trade_scale(direction, delta, reduce=False, mark_price=current_price)
                if not ok:
                    if active_trade.get("scale_add_paused"):
                        sync_position_panel(current_price)
                    send_private_telegram(f"⚠️ 補倉略過：{scale_msg}", priority=True)
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
                if real_copy_enabled:
                    send_private_telegram(note, priority=True)
                else:
                    send_telegram(note, priority=True)
                return

            if real_copy_enabled:
                ok, scale_msg = _execute_copy_trade_scale(direction, delta, reduce=True, mark_price=current_price)
                if not ok:
                    send_private_telegram(f"⚠️ 減倉略過：{scale_msg}", priority=True)
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
                f"倉位: {int(size*100)}% → {int(new_size*100)}%",
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
        btc_url = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
        btc_data = HTTP_SESSION.get(btc_url, timeout=3).json()
        btc_change = float(btc_data["priceChangePercent"]) / 100
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
    "last_adjust_ts": 0.0,
    "scale_add_paused": False,
    "scale_add_pause_reason": "",
    "scale_add_pause_ts": 0.0,
    "open_time": None,
    "tp_sl_adjusted_4h": False,
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
    active_trade["add_count"] = 0
    active_trade["reduce_count"] = 0
    active_trade["last_adjust_ts"] = 0.0
    active_trade["scale_add_paused"] = False
    active_trade["scale_add_pause_reason"] = ""
    active_trade["scale_add_pause_ts"] = 0.0
    active_trade["open_time"] = None
    active_trade["tp_sl_adjusted_4h"] = False
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

    state_ts = _safe_int(raw.get("ts"), 0)
    if state_ts > 0 and (time.time() - state_ts) > 48 * 3600:
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
    active_trade["last_adjust_ts"] = _safe_float(raw.get("last_adjust_ts"), 0.0)
    active_trade["scale_add_paused"] = bool(raw.get("scale_add_paused", False))
    active_trade["scale_add_pause_reason"] = str(raw.get("scale_add_pause_reason") or "")
    active_trade["scale_add_pause_ts"] = _safe_float(raw.get("scale_add_pause_ts"), 0.0)
    active_trade["open_time"] = _safe_float(raw.get("open_since_ts"), state_ts or time.time())
    active_trade["tp_sl_adjusted_4h"] = bool(raw.get("tp_sl_adjusted_4h", False))
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

# =============================
# WebSocket（tick級）
# =============================
def ws_price_stream():
    def on_message(ws, msg):
        global WS_PRICE
        data = json.loads(msg)
        WS_PRICE = float(data["p"])

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

# =============================
# Indicators
# =============================
def calc_indicators(df):
    df["ma25"] = df["close"].rolling(25).mean()

    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["signal"] = df["macd"].ewm(span=9).mean()

    # ===== Volume v2 =====
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    # VWAP（簡化版：以收盤加權）
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-9)

    return df

# =============================
# Market Regime（市場狀態）
# =============================
def detect_market_regime(df_1h, df_4h):
    trend_1h = df_1h["close"].iloc[-1] - df_1h["ma25"].iloc[-1]
    trend_4h = df_4h["close"].iloc[-1] - df_4h["ma25"].iloc[-1]

    # ===== 4H 強度（新增）=====
    strength_4h = df_4h["ma25"].iloc[-1] - df_4h["ma25"].iloc[-5]

    # ===== 波動（用1H判斷市場活躍度）=====
    vol = (df_1h["high"].iloc[-1] - df_1h["low"].iloc[-1]) / df_1h["close"].iloc[-1]

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


def _calc_support_resistance_levels(df, lookback=60):
    if df is None or len(df) < 20:
        return None, None

    window = df.tail(max(20, min(lookback, len(df))))
    support = float(np.percentile(window["low"], 12))
    resistance = float(np.percentile(window["high"], 88))

    if support <= 0 or resistance <= 0 or resistance <= support:
        return None, None
    return support, resistance


def analyze_multi_tf_sr_frames(price, frame_map, tf_cfg=None):
    """多週期支撐壓力 + K線型態分析，可直接餵入已準備好的 K 線資料。"""
    tf_cfg = tf_cfg or [
        ("月線", "1M", 80, 1.5),
        ("周線", "1w", 120, 1.3),
        ("日線", "1d", 180, 1.1),
        ("12h", "12h", 160, 1.0),
        ("4h", "4h", 140, 0.9),
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
    """多週期支撐壓力 + K線型態分析（月/週/日/12h/4h）。"""
    tf_cfg = [
        ("月線", "1M", 80, 1.5),
        ("周線", "1w", 120, 1.3),
        ("日線", "1d", 180, 1.1),
        ("12h", "12h", 160, 1.0),
        ("4h", "4h", 140, 0.9),
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
    if pending_sample or not active_trade.get("open"):
        return pending_sample

    direction = _normalize_trade_direction(active_trade.get("direction"))
    open_ts = _safe_float(active_trade.get("open_time"), 0.0)
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
        rebuilt = _save_pending_training_sample_state(
            {
                "features": dict(features),
                "learn_features": _build_directional_learning_features(features, direction),
                "direction": direction,
                "entry_ts": open_ts,
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


def _predict_trade_probability(features):
    normalized = _normalize_feature_payload(features)
    X_raw = pd.DataFrame([normalized])

    batch_prob = None
    if model is not None and _has_expected_feature_schema(model):
        batch_prob = _predict_estimator_probability(model, X_raw)

    online_prob = None
    online_ready = (
        online_initialized
        and online_sample_count >= ONLINE_MODEL_MIN_SAMPLES
        and _has_expected_feature_schema(online_model)
        and hasattr(online_scaler, "n_samples_seen_")
    )
    if online_ready:
        X_online = _prepare_online_feature_frame(X_raw, fit_scaler=False)
        online_prob = _predict_estimator_probability(online_model, X_online)
    elif batch_prob is None and online_initialized and hasattr(online_scaler, "n_samples_seen_"):
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


def _predict_directional_trade_bias(features):
    long_features = _build_directional_learning_features(features, "long")
    short_features = _build_directional_learning_features(features, "short")
    long_prob = _predict_trade_probability(long_features)
    short_prob = _predict_trade_probability(short_features)

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
        }

    price = _safe_float(price, _safe_float(df_5m["close"].iloc[-1], 0.0))
    derivatives_flow = _normalize_derivatives_flow_snapshot(derivatives_flow)
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
    score = max(0.05, min(score, 0.95))

    if abs(derivatives_pressure) >= 0.10:
        score += max(-0.06, min(0.06, derivatives_pressure * 0.045))
        score = max(0.05, min(score, 0.95))

    entry = price
    min_accept_rr = max(1.1, _safe_float(min_accept_rr if min_accept_rr is not None else os.getenv("TRADE_MIN_ACCEPT_RR", 1.8), 1.8))
    min_net_edge_rate = max(0.0005, _safe_float(min_net_edge_rate if min_net_edge_rate is not None else os.getenv("TRADE_MIN_NET_EDGE_RATE", 0.0012), 0.0012))
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
    rr_at_entry = 0.0
    risk_rate = 0.0
    reward_rate = 0.0
    net_edge_rate_est = 0.0

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

    if abs(score - 0.5) > entry_threshold:
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
                rr = 2.6 if regime.endswith("strong") else 2.0
                tp = entry + risk * rr
            elif score < 0.48:
                final = "🚀 做空"
                recent_high_pb = df_15m["high"].tail(10).max()
                sl = recent_high_pb
                risk = sl - entry
                rr = 2.6 if regime.endswith("strong") else 2.0
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

            position_size = _cap_initial_position_size(position_size)

    if final != "觀望":
        final, sl, tp = auto_fix_trade_plan(final, entry, sl, tp, atr)

    point_explain = ""
    if not final.startswith("觀望") and sl is not None and tp is not None and entry > 0:
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
        elif reward_rate < min_reward_rate_needed:
            final = "觀望（報酬不足覆蓋成本）"
        elif _is_truthy(os.getenv("TRADE_REQUIRE_AI_EDGE", "0")):
            min_expected_edge_rate = max(
                0.0002,
                _safe_float(os.getenv("TRADE_MIN_EXPECTED_EDGE_RATE", min_net_edge_rate), min_net_edge_rate),
            )
            if net_edge_rate_est < min_expected_edge_rate:
                final = "觀望（AI期望值不足）"

    return {
        "features": features,
        "score": score,
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
        "open_interest_change": open_interest_change,
        "mark_premium_rate": mark_premium_rate,
        "funding_rate_live": funding_rate_live,
        "taker_buy_ratio": taker_buy_ratio,
        "derivatives_pressure": derivatives_pressure,
        "derivatives_flow_stale": bool(derivatives_flow.get("stale", False)),
        "support_hits": _safe_int(sr_analysis.get("support_hits"), 0),
        "resistance_hits": _safe_int(sr_analysis.get("resistance_hits"), 0),
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
    if not pending_sample or not isinstance(pending_sample, dict):
        _clear_pending_training_sample_state()
        return None

    direction = _normalize_trade_direction(pending_sample.get("direction"))
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
        print("✅ 私聊通知已送出")
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


# ===== AI分析（OpenClaw / OpenAI） =====
OPENAI_API_KEY = _get_required_env("OPENAI_API_KEY", "", mask=True)

def ask_ai_analysis(prompt):
    if not OPENAI_API_KEY:
        return "AI分析失敗: 未設定 OPENAI_API_KEY"

    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "你是一個專業ETH交易分析師"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=10).json()
        return res["choices"][0]["message"]["content"]
    except Exception as e:
        return f"AI分析失敗: {e}"


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

        # 注入你系統數據（核心升級）
        prompt = f"""
你是一個專業ETH交易分析師，請根據以下即時市場數據進行分析：

【市場數據】
價格: {context.get('price')}
AI分數: {context.get('score')}
HTF趨勢: {context.get('htf')}
市場狀態: {context.get('regime')}
Breakout: {context.get('breakout')}
Triangle: {context.get('triangle')}
Macro: {context.get('macro')}
Volume Spike: {context.get('volume_spike')}

【問題】
{question}

請輸出：
1. 當前市場結構
2. 是否建議做多/做空/觀望
3. 進場區間
4. 止盈止損建議
"""

        return ask_ai_analysis(prompt)

    if text.startswith("/news"):
        try:
            _, _, news_list = refresh_rss_news_cache(force=True)
            if news_list:
                preview = "\n".join([f"- {n}" for n in news_list[:12]])
                return f"📰 最新即時訊息\n{preview}"
            return "📰 目前沒有抓到新的即時訊息"
        except Exception as e:
            return f"📰 新聞讀取失敗: {e}"

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
def get_kline(interval, limit=100):
    now = time.time()

    if interval in KLINE_CACHE:
        data, ts = KLINE_CACHE[interval]
        if now - ts < KLINE_TTL.get(interval, 10):
            return data

    url = "https://fapi.binance.com/fapi/v1/klines"
    data = requests.get(url, params={
        "symbol": "ETHUSDT",
        "interval": interval,
        "limit": limit
    }).json()

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
    # trade_open 移除，改用 active_trade 控制是否可開單

    # ===== 每日報告 =====
    last_report_time = 0

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

    while True:
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

                    reply = handle_ai_command(text, context)

                    if reply:
                        _send_telegram_message(chat_id, reply, include_control_panel=True)
                except:
                    pass
            # ===== HTF（方向）=====
            df_4h = get_kline("4h")
            df_1h = get_kline("1h")

            # ===== Market Regime =====
            regime = detect_market_regime(df_1h, df_4h)

            # ===== 4H v2（方向 + 強度）=====
            trend_4h = df_4h["close"].iloc[-1] - df_4h["ma25"].iloc[-1]
            strength_4h = df_4h["ma25"].iloc[-1] - df_4h["ma25"].iloc[-5]

            htf = 1 if trend_4h > 0 else -1
            htf_strength = abs(strength_4h)

            # ===== MID（策略）=====
            df_30m = get_kline("30m")
            df_15m = get_kline("15m")

            mid_trend = 1 if df_30m["macd"].iloc[-1] > df_30m["signal"].iloc[-1] else -1
            fvg_low, fvg_high = calc_fvg(df_15m)

            # ===== LTF（進場）=====
            df_5m = get_kline("5m", 50)
            df_1m = get_kline("1m", 50)

            breakout = 0
            recent_high = df_5m["high"].iloc[-5:-1].max()
            recent_low = df_5m["low"].iloc[-5:-1].min()
            price = WS_PRICE if WS_PRICE else df_1m["close"].iloc[-1]
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

            _process_sl_followup_reviews(df_1m, price)
            _process_pending_news_evaluations(price)

            _update_scaling_market_state(
                price=price,
                atr=atr,
                htf=htf,
                mid_trend=mid_trend,
                regime=regime,
                breakout=breakout,
                sr_analysis=sr_analysis,
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
                    send_telegram(snapshot_header, priority=True, include_private=False)
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
            if active_trade["open"]:
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
                        active_trade["open"] = False
                        active_trade["size"] = 0.0
                        active_trade["position_qty"] = 0.0
                        active_trade["add_count"] = 0
                        active_trade["reduce_count"] = 0
                        active_trade["open_time"] = None
                        active_trade["tp_sl_adjusted_4h"] = False
                        _set_break_even_state(False)
                        last_signal_cache = None
                        losing_streak += 1
                        sync_position_panel(current)
                        print("❌ SL 命中")
                        _send_trade_notification(
                            _build_trade_close_message("SL", active_trade["direction"], current, candle_high, candle_low),
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
                        active_trade["open_time"] = None
                        active_trade["tp_sl_adjusted_4h"] = False
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
                        active_trade["open"] = False
                        active_trade["size"] = 0.0
                        active_trade["position_qty"] = 0.0
                        active_trade["add_count"] = 0
                        active_trade["reduce_count"] = 0
                        active_trade["open_time"] = None
                        active_trade["tp_sl_adjusted_4h"] = False
                        _set_break_even_state(False)
                        last_signal_cache = None
                        losing_streak += 1
                        sync_position_panel(current)
                        print("❌ SL 命中")
                        _send_trade_notification(
                            _build_trade_close_message("SL", active_trade["direction"], current, candle_high, candle_low),
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
                        active_trade["open_time"] = None
                        active_trade["tp_sl_adjusted_4h"] = False
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
                be_triggered = maybe_activate_auto_break_even(current, atr=atr)
                if not be_triggered:
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
            entry = price

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

            reason_text = " | ".join(reason)

            # ===== 市場狀態中文轉換 =====
            regime_map = {
                "bull_trend_strong": "強多趨勢",
                "bull_trend": "多頭趨勢",
                "range": "盤整",
                "bear_trend_strong": "強空趨勢",
                "bear_trend": "空頭趨勢"
            }

            regime_text = regime_map.get(regime, regime)

            # ===== 統一輸出訊號格式 =====
            if "做多" in final:
                display_signal = "🚀 做多"
            elif "做空" in final:
                display_signal = "🚀 做空"
            else:
                display_signal = final

            # ===== 訊息格式（進場優先顯示）=====
            msg = ""

            if final != "觀望":
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
                f"📈 勝率: {(performance['win']/performance['total'] if performance['total']>0 else 0):.2%}\n"
                f"🌍 市場狀態: {regime_text}\n"
                f"📰 時事判斷: {macro_text}\n"
                f"{news_text}"
                f"🧠 判斷依據: {reason_text}"
            )
            if point_explain:
                msg += f"\n{point_explain}"
            sr_lines = sr_analysis.get("lines") if isinstance(sr_analysis, dict) else []
            if isinstance(sr_lines, list) and sr_lines:
                msg += "\n\n🧱 多週期支撐/壓力（K線型態）\n" + "\n".join([f"- {line}" for line in sr_lines[:5]])
            # Fix spam log（觀望不要一直print）
            if final != "觀望":
                print(msg)

            # 強制單也必須再次經過自動修正，避免繞過前面的保護
            if final != "觀望":
                final, sl, tp = auto_fix_trade_plan(final, entry, sl, tp, atr)

            # ===== 開單頻率 + 訊號去重（核心修正）=====
            now_ts = time.time()

            TRADE_COOLDOWN = 300  # 拉長避免過度交易

            # 先做去重與冷卻判斷，再決定是否跳過

            # ===== 同方向去重（用方向，不用字串） =====
            current_direction = get_signal_direction(final)
            last_direction_simple = get_signal_direction(last_trade_signal) if last_trade_signal else None

            # ===== 防洗單 v6 =====
            if current_direction == last_direction_simple:
                # 價格變動太小 → 不開單
                if last_entry_price is not None:
                    price_change = abs(price - last_entry_price) / price
                    if price_change < MIN_PRICE_CHANGE:
                        final = "觀望（防洗單-價格過近）"

                # 信號變化太小 → 不開單
                if last_signal is not None:
                    if abs(score - last_signal) < MIN_SIGNAL_DIFF:
                        final = "觀望（防洗單-信號重複）"

            if final != "觀望":
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
                continue

            # ===== 最終安全檢查：拒絕假突破低信心單 =====
            if fake_breakout and abs(score - 0.5) < 0.22:
                continue

            # ===== 多週期壓力/支撐阻擋：靠近關鍵反向區域先觀望 =====
            support_hits = _safe_int(sr_analysis.get("support_hits"), 0)
            resistance_hits = _safe_int(sr_analysis.get("resistance_hits"), 0)
            if "做多" in final and resistance_hits >= 2 and score < 0.72:
                continue
            if "做空" in final and support_hits >= 2 and score > 0.28:
                continue

            if final != "觀望":

                # 保險：再次確認沒有持倉
                if active_trade["open"]:
                    continue

                # 防止同一訊號重複刷
                if last_signal_cache == msg:
                    continue

                # ===== 建立真實交易 =====
                direction = "long" if "做多" in final else "short"

                active_trade["direction"] = direction
                active_trade["entry"] = float(entry)
                active_trade["avg_entry"] = float(entry)
                active_trade["tp"] = tp
                active_trade["sl"] = sl
                base_size = _safe_float(position_size, 0.0)
                if base_size <= 0:
                    base_size = 0.2
                base_size = _cap_initial_position_size(base_size)
                active_trade["size"] = float(min(1.0, max(base_size, 0.1)))
                max_size, min_size = _derive_scaling_bounds(active_trade["size"])
                active_trade["max_size"] = max_size
                active_trade["min_size"] = min_size
                active_trade["add_count"] = 0
                active_trade["reduce_count"] = 0
                active_trade["last_adjust_ts"] = 0.0
                active_trade["scale_add_paused"] = False
                active_trade["scale_add_pause_reason"] = ""
                active_trade["scale_add_pause_ts"] = 0.0
                active_trade["open_time"] = time.time()
                active_trade["tp_sl_adjusted_4h"] = False
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
                        pending_training_sample = {
                            "features": dict(features),
                            "learn_features": dict(learn_features),
                            "direction": direction,
                            "entry_ts": now_ts,
                        }
                        pending_training_sample = _save_pending_training_sample_state(pending_training_sample)
                        active_trade["open"] = True
                        sync_active_trade_from_binance(send_notice=False)
                    else:
                        # 開單失敗，清除本地倉位狀態，避免面板顯示假倉位
                        pending_training_sample = None
                        _reset_active_trade_state()
                        sync_position_panel(price)
                else:
                    # 未開啟跟單，僅本地追蹤
                    learn_features = _build_directional_learning_features(features, direction)
                    pending_training_sample = {
                        "features": dict(features),
                        "learn_features": dict(learn_features),
                        "direction": direction,
                        "entry_ts": now_ts,
                    }
                    pending_training_sample = _save_pending_training_sample_state(pending_training_sample)
                    active_trade["open"] = True
                    sync_position_panel(price)

            # ===== 學習標籤改為真實 TP/SL，避免用 1.2 秒短噪音污染模型 =====
            new_price = WS_PRICE if WS_PRICE else price

            # 定期重訓 batch model；每次平倉樣本都會先即時落盤
            maybe_train_model_periodically(force=False)

            # ===== 更新信號（平滑 + 防洗單記錄）=====
            last_signal = score
            # ===== 每日報告 =====
            if time.time() - last_report_time > 3600:  # 每1小時
                winrate = performance["win"] / performance["total"] if performance["total"] > 0 else 0

                report = (
                    f"📊 交易報告\n"
                    f"總交易: {performance['total']}\n"
                    f"勝率: {winrate:.2%}\n"
                    f"勝: {performance['win']} / 敗: {performance['loss']}"
                )

                send_telegram(report)
                last_report_time = time.time()

            sync_position_panel(new_price)
            time.sleep(0.8)

        except Exception as e:
            print("error:", repr(e))
            time.sleep(3)

# =============================
# start
# =============================
if __name__ == "__main__":
    print("🔥 AI 接管版啟動")
    run_bot()
