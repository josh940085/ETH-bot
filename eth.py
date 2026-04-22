# ===== 移除 gevent（穩定版）=====
import requests
import datetime
import time
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
import subprocess
import hmac
import hashlib
try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None
from collections import deque
from pathlib import Path
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from threading import Lock

# ===== Macro / News Engine =====

MACRO_CACHE = {"sp": 0, "nq": 0, "btc": 0, "dxy": 0, "news": 0, "event": 0, "news_list": [], "ts": 0}
NEWS_CACHE = {"news": 0, "event": 0, "news_list": [], "ts": 0}

# ===== AI 新聞分類 =====
NEWS_MODEL_PATH = "news_model.pkl"
NEWS_VECTORIZER_PATH = "news_vectorizer.pkl"
NEWS_PERFORMANCE_LOG = "news_predictions.jsonl"   # 記錄所有預測結果用於評估
NEWS_LEARNING_BUFFER = "learning_buffer.pkl"       # 增量學習緩衝區
news_model = None
news_vectorizer = None

# 增量學習配置
INCREMENTAL_LEARNING_ENABLED = True
MIN_PREDICTIONS_FOR_RETRAIN = 50  # 每50個預測後考慮重新訓練

HTTP_SESSION = requests.Session()
HTTP_SESSION.headers.update({"User-Agent": "Mozilla/5.0"})

TRANSLATION_CACHE = {}
BOT_SOFT_RESTART_REQUESTED = False
TELEGRAM_STATE_PATH = Path(__file__).resolve().parent / ".telegram_state.json"
POSITION_JSON_PATH = Path(__file__).resolve().parent / "docs" / "position.json"

# ===== Environment variables / secrets =====
def _load_local_env():
    """簡易讀取 .env（不依賴 python-dotenv）。"""
    try:
        env_path = Path(__file__).resolve().parent / ".env"
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
                # 以 .env 為準，避免既有空值/舊值蓋掉最新設定
                os.environ[key] = value
    except Exception as e:
        print(f"⚠️ .env 載入失敗: {e}")


def _get_required_env(name, default=None, mask=False):
    value = os.getenv(name, default)
    if value is None or str(value).strip() == "":
        print(f"⚠️ 缺少環境變數: {name}")
        return default
    if mask:
        print(f"✅ 已載入 {name}")
    return value


def _get_env_with_alias(primary, aliases=None, default=""):
    keys = [primary] + list(aliases or [])
    for k in keys:
        v = os.getenv(k)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return default


_load_local_env()

TELEGRAM_TOKEN = _get_required_env("TELEGRAM_TOKEN", "", mask=True)
TELEGRAM_CHAT_ID = _get_required_env("TELEGRAM_CHAT_ID", "")
TELEGRAM_PRIVATE_CHAT_ID = _get_env_with_alias("TELEGRAM_PRIVATE_CHAT_ID", ["TELEGRAM_USER_CHAT_ID"], "")
TELEGRAM_PRIVATE_CHAT_LOCK = str(os.getenv("TELEGRAM_PRIVATE_CHAT_LOCK", "0")).strip() == "1"
if not TELEGRAM_PRIVATE_CHAT_ID:
    print("⚠️ 缺少環境變數: TELEGRAM_PRIVATE_CHAT_ID")
# ===== Telegram =====
LAST_TELEGRAM_TS = 0

# ===== Discord（同步通知） =====
DISCORD_WEBHOOK = _get_required_env("DISCORD_WEBHOOK", "", mask=True)

# ===== Binance 實單跟單（API） =====
BINANCE_API_KEY = _get_required_env("BINANCE_API_KEY", "", mask=True)
BINANCE_API_SECRET = _get_required_env("BINANCE_API_SECRET", "", mask=True)
BINANCE_BASE_URL = str(os.getenv("BINANCE_BASE_URL", "https://fapi.binance.com")).rstrip("/")
BINANCE_REAL_COPY_ENABLED = str(os.getenv("BINANCE_REAL_COPY_ENABLED", "0")).strip() == "1"
BINANCE_SYMBOL = str(os.getenv("BINANCE_SYMBOL", "ETHUSDT")).strip() or "ETHUSDT"
try:
    BINANCE_LEVERAGE = int(float(_get_env_with_alias("BINANCE_LEVERAGE", ["COPY_TRADE_LEVERAGE"], "10")))
except Exception:
    BINANCE_LEVERAGE = 10
# === USDT 名目模式（傳統） ===
try:
    BINANCE_ORDER_NOTIONAL_USDT = max(10.0, float(_get_env_with_alias("BINANCE_ORDER_NOTIONAL_USDT", ["COPY_TRADE_USDT"], "150")))
except Exception:
    BINANCE_ORDER_NOTIONAL_USDT = 150.0

# === ETH 數量模式（新） ===
# 若設定 COPY_TRADE_ETH_QTY，100% 倉位就是該數量的 ETH
try:
    eth_qty_str = os.getenv("COPY_TRADE_ETH_QTY", "").strip()
    BINANCE_ORDER_ETH_QTY = float(eth_qty_str) if eth_qty_str else 0.0
except Exception:
    BINANCE_ORDER_ETH_QTY = 0.0

# ETH 模式下的最小開倉數量（例子：0.009）
try:
    min_eth_str = os.getenv("COPY_TRADE_MIN_ETH_QTY", "").strip()
    BINANCE_MIN_ETH_QTY = float(min_eth_str) if min_eth_str else 0.001
except Exception:
    BINANCE_MIN_ETH_QTY = 0.001

try:
    BINANCE_MIN_NOTIONAL_USDT = max(5.0, float(os.getenv("BINANCE_MIN_NOTIONAL_USDT", "20")))
except Exception:
    BINANCE_MIN_NOTIONAL_USDT = 20.0
try:
    BINANCE_QTY_STEP = max(0.0001, float(os.getenv("BINANCE_QTY_STEP", "0.001")))
except Exception:
    BINANCE_QTY_STEP = 0.001
try:
    BINANCE_MIN_QTY = max(0.0001, float(os.getenv("BINANCE_MIN_QTY", "0.001")))
except Exception:
    BINANCE_MIN_QTY = 0.001

# 100% 倉位基準改為「帳戶總資產的 1/3」
BINANCE_USE_ACCOUNT_THIRD = str(os.getenv("COPY_TRADE_USE_ACCOUNT_THIRD", "1")).strip() == "1"

_BINANCE_POSITION_MODE_CACHE = {"dual": None, "ts": 0.0}
_BINANCE_ACCOUNT_CACHE = {"asset": 0.0, "ts": 0.0}


def _is_binance_copy_ready():
    # 由跟單按鈕（FOLLOW_MODE_ENABLED）控制，API Key 需存在
    # BINANCE_REAL_COPY_ENABLED=0 可強制停用（安全開關）
    if BINANCE_REAL_COPY_ENABLED is False and os.getenv("BINANCE_REAL_COPY_ENABLED", "").strip() == "0":
        return False
    try:
        follow_on = FOLLOW_MODE_ENABLED
    except NameError:
        follow_on = False
    if not follow_on:
        return False
    return bool(BINANCE_API_KEY and BINANCE_API_SECRET)


def _is_private_chat(chat_id):
    try:
        return int(chat_id) > 0
    except Exception:
        return False


def _with_forced_remove_reply_keyboard(payload, chat_id):
    """私聊訊息強制移除 reply keyboard（不覆蓋既有 inline keyboard）。"""
    try:
        normalized_chat_id = str(chat_id or "").strip()
        if normalized_chat_id and _is_private_chat(normalized_chat_id):
            if "reply_markup" not in payload:
                payload["reply_markup"] = {"remove_keyboard": True}
    except Exception:
        pass
    return payload


def _resolve_follow_private_chat_id():
    preferred = str(TELEGRAM_PRIVATE_CHAT_ID or "").strip()
    if preferred and _is_private_chat(preferred):
        return preferred
    return ""


def _binance_sign_params(params):
    q = urlencode(params, doseq=True)
    signature = hmac.new(
        BINANCE_API_SECRET.encode("utf-8"),
        q.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{q}&signature={signature}"


def _binance_request(method, path, params=None, signed=False, timeout=6):
    params = dict(params or {})
    headers = {}
    if BINANCE_API_KEY:
        headers["X-MBX-APIKEY"] = BINANCE_API_KEY

    if signed:
        params["timestamp"] = int(time.time() * 1000)
        params.setdefault("recvWindow", 5000)
        payload = _binance_sign_params(params)
    else:
        payload = urlencode(params, doseq=True)

    url = f"{BINANCE_BASE_URL}{path}"
    if payload:
        url = f"{url}?{payload}"

    if method == "POST":
        resp = HTTP_SESSION.post(url, headers=headers, timeout=timeout)
    elif method == "DELETE":
        resp = HTTP_SESSION.delete(url, headers=headers, timeout=timeout)
    else:
        resp = HTTP_SESSION.get(url, headers=headers, timeout=timeout)

    data = resp.json()
    if resp.status_code >= 400:
        raise Exception(f"binance_http_{resp.status_code}: {data}")
    return data


def _round_down_step(value, step):
    if step <= 0:
        return value
    return max(0.0, (int(value / step)) * step)


def _round_up_step(value, step):
    if step <= 0:
        return value
    # 避免浮點誤差導致多跳一格
    return max(0.0, math.ceil((value - 1e-12) / step) * step)


def _get_binance_total_asset_usdt(force=False):
    """讀取 Binance 合約總資產（優先 totalMarginBalance，次選 totalWalletBalance）。"""
    now = time.time()
    cached_asset = _safe_float(_BINANCE_ACCOUNT_CACHE.get("asset"), 0.0)
    cached_ts = _safe_float(_BINANCE_ACCOUNT_CACHE.get("ts"), 0.0)
    if (not force) and cached_asset > 0 and now - cached_ts < 10:
        return cached_asset

    try:
        data = _binance_request("GET", "/fapi/v2/account", {}, signed=True)
        total_margin = _safe_float(data.get("totalMarginBalance"), 0.0)
        total_wallet = _safe_float(data.get("totalWalletBalance"), 0.0)
        asset = total_margin if total_margin > 0 else total_wallet
        if asset > 0:
            _BINANCE_ACCOUNT_CACHE["asset"] = asset
            _BINANCE_ACCOUNT_CACHE["ts"] = now
        return asset
    except Exception as e:
        print(f"⚠️ 讀取 Binance 總資產失敗: {e}")
        return cached_asset


def _binance_qty_from_size_ratio(size_ratio, current_price, enforce_min_notional=False):
    """根據倉位比例計算下單數量。預設 100% = 帳戶總資產 1/3（USDT）。"""
    ratio = max(0.0, min(float(size_ratio), 1.0))
    px = max(0.0, _safe_float(current_price, 0.0))
    if ratio <= 0 or px <= 0:
        return 0.0

    # 新模式：100% = 帳戶總資產的 1/3（USDT）
    if BINANCE_USE_ACCOUNT_THIRD and _is_binance_copy_ready():
        total_asset = _get_binance_total_asset_usdt(force=False)
        if total_asset > 0:
            notional = (total_asset / 3.0) * ratio
            if enforce_min_notional and notional > 0:
                notional = max(notional, BINANCE_MIN_NOTIONAL_USDT)
            raw_qty = notional / px
            min_qty = BINANCE_MIN_QTY
        elif BINANCE_ORDER_ETH_QTY > 0:
            # API 讀資產失敗時，退回舊 ETH 顆數模式
            raw_qty = BINANCE_ORDER_ETH_QTY * ratio
            min_qty = BINANCE_MIN_ETH_QTY
        else:
            # API 讀資產失敗時，退回舊 USDT 名目模式
            notional = BINANCE_ORDER_NOTIONAL_USDT * ratio
            if enforce_min_notional and notional > 0:
                notional = max(notional, BINANCE_MIN_NOTIONAL_USDT)
            raw_qty = notional / px
            min_qty = BINANCE_MIN_QTY
    # 舊模式：若設定 ETH 顆數，使用 ETH 模式；否則用 USDT 名目
    elif BINANCE_ORDER_ETH_QTY > 0:
        # ETH 顆數模式：100% = BINANCE_ORDER_ETH_QTY
        raw_qty = BINANCE_ORDER_ETH_QTY * ratio
        # ETH 模式下的最小量
        min_qty = BINANCE_MIN_ETH_QTY
    else:
        # USDT 名目模式（傳統）
        notional = BINANCE_ORDER_NOTIONAL_USDT * ratio
        if enforce_min_notional and notional > 0:
            notional = max(notional, BINANCE_MIN_NOTIONAL_USDT)
        raw_qty = notional / px
        # USDT 模式下的最小量
        min_qty = BINANCE_MIN_QTY

    if enforce_min_notional:
        qty = _round_up_step(raw_qty, BINANCE_QTY_STEP)
    else:
        qty = _round_down_step(raw_qty, BINANCE_QTY_STEP)
    
    if qty < min_qty:
        return 0.0
    return qty


def _binance_set_leverage_once():
    if getattr(_binance_set_leverage_once, "done", False):
        return True
    if not _is_binance_copy_ready():
        return False
    try:
        _binance_request(
            "POST",
            "/fapi/v1/leverage",
            {"symbol": BINANCE_SYMBOL, "leverage": BINANCE_LEVERAGE},
            signed=True,
        )
        _binance_set_leverage_once.done = True
        return True
    except Exception as e:
        print(f"⚠️ 設定 Binance 槓桿失敗: {e}")
        return False


def _binance_is_dual_side_mode(force=False):
    """查詢帳戶是否為 Hedge Mode（dualSidePosition=true）。"""
    now = time.time()
    cached = _BINANCE_POSITION_MODE_CACHE.get("dual")
    if (not force) and cached is not None and now - _BINANCE_POSITION_MODE_CACHE.get("ts", 0.0) < 300:
        return bool(cached)

    try:
        data = _binance_request("GET", "/fapi/v1/positionSide/dual", {}, signed=True)
        raw = data.get("dualSidePosition")
        dual = bool(raw) if isinstance(raw, bool) else str(raw).strip().lower() == "true"
        _BINANCE_POSITION_MODE_CACHE["dual"] = dual
        _BINANCE_POSITION_MODE_CACHE["ts"] = now
        return dual
    except Exception as e:
        print(f"⚠️ 讀取 Binance 倉位模式失敗: {e}")
        if cached is not None:
            return bool(cached)
        return False


def _binance_finalize_order_params(params, position_side=None, reduce_only=False):
    """根據 Binance 倉位模式補齊參數，避免 -4061。"""
    if _binance_is_dual_side_mode():
        if position_side in ("LONG", "SHORT"):
            params["positionSide"] = position_side
    elif reduce_only:
        params["reduceOnly"] = "true"
    return params


def _binance_place_market_order(side, qty, reduce_only=False, position_side=None):
    """在 Binance 下市價單（MARKET）。平倉/減倉使用此函式以避免 -4061。"""
    if qty <= 0:
        return False, "qty<=0"
    if not _is_binance_copy_ready():
        return False, "binance_copy_not_ready"

    try:
        _binance_set_leverage_once()
        params = {
            "symbol": BINANCE_SYMBOL,
            "side": side,
            "type": "MARKET",
            "quantity": f"{qty:.6f}",
        }
        params = _binance_finalize_order_params(params, position_side=position_side, reduce_only=reduce_only)

        data = _binance_request("POST", "/fapi/v1/order", params, signed=True)
        return True, data
    except Exception as e:
        return False, str(e)


def _binance_place_limit_order(side, qty, price, reduce_only=False, position_side=None):
    """在 Binance 下限價單（LIMIT）。開倉/補倉使用此函式。"""
    if qty <= 0:
        return False, "qty<=0"
    if not _is_binance_copy_ready():
        return False, "binance_copy_not_ready"

    try:
        _binance_set_leverage_once()
        params = {
            "symbol": BINANCE_SYMBOL,
            "side": side,
            "type": "LIMIT",
            "quantity": f"{qty:.6f}",
            "price": f"{price:.2f}",
            "timeInForce": "GTC",
        }
        params = _binance_finalize_order_params(params, position_side=position_side, reduce_only=reduce_only)
        data = _binance_request("POST", "/fapi/v1/order", params, signed=True)
        return True, data
    except Exception as e:
        return False, str(e)


def _binance_get_order(order_id):
    """查詢單一訂單狀態。"""
    try:
        if not order_id:
            return False, "missing_order_id"
        data = _binance_request(
            "GET",
            "/fapi/v1/order",
            {"symbol": BINANCE_SYMBOL, "orderId": int(order_id)},
            signed=True,
        )
        return True, data
    except Exception as e:
        return False, str(e)


def _binance_wait_limit_fill_then_maybe_cancel(order_id, timeout_sec=20):
    """等待限價單成交；逾時未成交則取消。"""
    started = time.time()
    last_order = None

    while time.time() - started < max(1.0, float(timeout_sec)):
        ok, order_data = _binance_get_order(order_id)
        if ok:
            last_order = order_data
            status = str(order_data.get("status") or "").upper()
            if status in ("FILLED", "CANCELED", "EXPIRED", "REJECTED"):
                return ok, order_data, False
        time.sleep(0.8)

    # 超時，嘗試取消未成交訂單
    ok_cancel, cancel_data = _binance_cancel_order(order_id)
    if ok_cancel:
        ok, order_data = _binance_get_order(order_id)
        if ok:
            return True, order_data, True
    return False, cancel_data, True


def _binance_place_stop_market_close(direction, stop_price):
    """掛 Binance 原生止損單（STOP_MARKET + closePosition）。"""
    px = _safe_float(stop_price, 0.0)
    if px <= 0:
        return False, "stop_price_invalid", None
    if not _is_binance_copy_ready():
        return False, "binance_copy_not_ready", None

    try:
        _binance_set_leverage_once()
        side = "SELL" if direction == "long" else "BUY"
        position_side = "LONG" if direction == "long" else "SHORT"
        params = {
            "symbol": BINANCE_SYMBOL,
            "side": side,
            "type": "STOP_MARKET",
            "stopPrice": f"{px:.2f}",
            "closePosition": "true",
            "workingType": "MARK_PRICE",
            "priceProtect": "TRUE",
        }
        params = _binance_finalize_order_params(params, position_side=position_side, reduce_only=False)
        data = _binance_request("POST", "/fapi/v1/order", params, signed=True)
        return True, data, data.get("orderId")
    except Exception as e:
        return False, str(e), None


def _binance_cancel_order(order_id):
    if not order_id:
        return True, "no_order_id"
    try:
        data = _binance_request(
            "DELETE",
            "/fapi/v1/order",
            {"symbol": BINANCE_SYMBOL, "orderId": int(order_id)},
            signed=True,
        )
        return True, data
    except Exception as e:
        msg = str(e)
        # 已成交/已取消時，視為可接受
        if "-2011" in msg or "Unknown order" in msg:
            return True, msg
        return False, msg


def _clear_native_stop_tracking():
    active_trade["binance_sl_order_id"] = None
    active_trade["binance_sl_price"] = 0.0


def _cancel_native_stop_loss_order():
    order_id = active_trade.get("binance_sl_order_id")
    if not order_id:
        _clear_native_stop_tracking()
        return True, "no_native_stop"

    ok, data = _binance_cancel_order(order_id)
    _clear_native_stop_tracking()
    return ok, data


def sync_binance_set_native_stop_loss(direction, stop_price):
    """同步原生止損單：先撤舊單，再掛新 STOP_MARKET。"""
    if not _is_binance_copy_ready():
        return True, "skip_not_enabled"

    _cancel_native_stop_loss_order()
    ok, data, order_id = _binance_place_stop_market_close(direction, stop_price)
    if ok:
        active_trade["binance_sl_order_id"] = order_id
        active_trade["binance_sl_price"] = _safe_float(stop_price, 0.0)
    return ok, data


def _fetch_binance_position_entry(direction):
    """查詢 Binance 實際持倉均價（entryPrice），未持倉或失敗時回傳 0.0。"""
    try:
        data = _binance_request("GET", "/fapi/v2/positionRisk", {"symbol": BINANCE_SYMBOL}, signed=True)
        target_side = "LONG" if direction == "long" else "SHORT"
        for pos in (data if isinstance(data, list) else []):
            if pos.get("symbol") != BINANCE_SYMBOL:
                continue
            # Hedge Mode：比對 positionSide；One-way Mode：忽略
            if _binance_is_dual_side_mode():
                if pos.get("positionSide") != target_side:
                    continue
            ep = _safe_float(pos.get("entryPrice"), 0.0)
            amt = _safe_float(pos.get("positionAmt"), 0.0)
            if abs(amt) > 1e-9 and ep > 0:
                return ep
        return 0.0
    except Exception as e:
        print(f"⚠️ 查詢 Binance 進場均價失敗: {e}")
        return 0.0


def sync_binance_open_position(direction, size_ratio, current_price):
    if not _is_binance_copy_ready():
        return True, "skip_not_enabled"

    qty = _binance_qty_from_size_ratio(size_ratio, current_price, enforce_min_notional=True)
    if qty <= 0:
        return False, "quantity_too_small"
    
    # 檢查實際名目是否達到最小值（避免 -4164）
    notional = qty * current_price
    if notional < BINANCE_MIN_NOTIONAL_USDT:
        return False, f"notional_too_small_{notional:.2f}<{BINANCE_MIN_NOTIONAL_USDT}"

    side = "BUY" if direction == "long" else "SELL"
    position_side = "LONG" if direction == "long" else "SHORT"
    # 開倉用市價單：確保即時成交，避免限價單超時取消
    ok, data = _binance_place_market_order(side, qty, reduce_only=False, position_side=position_side)
    if ok:
        executed_qty = _safe_float((data or {}).get("executedQty"), 0.0) if isinstance(data, dict) else 0.0
        active_trade["binance_qty"] = float(executed_qty if executed_qty > 0 else qty)
        # 從訂單回傳的 avgPrice 取得實際成交價
        try:
            avg_price = float((data or {}).get("avgPrice") or 0)
        except (ValueError, TypeError):
            avg_price = 0.0
        # avgPrice=0 時（少見）查詢 positionRisk 取得進場均價
        if avg_price <= 0:
            avg_price = _fetch_binance_position_entry(direction)
        if avg_price > 0:
            active_trade["entry"] = avg_price
            active_trade["avg_entry"] = avg_price
        return True, data
    return False, data


def sync_binance_adjust_position(direction, action, delta_ratio, current_price):
    if not _is_binance_copy_ready():
        return True, "skip_not_enabled"

    enforce_min_notional = action == "add"
    qty = _binance_qty_from_size_ratio(delta_ratio, current_price, enforce_min_notional=enforce_min_notional)
    if qty <= 0:
        return False, "quantity_too_small"

    # 檢查實際名目是否達到最小值（避免 -4164）
    notional = qty * current_price
    if enforce_min_notional and notional < BINANCE_MIN_NOTIONAL_USDT:
        return False, f"notional_too_small_{notional:.2f}<{BINANCE_MIN_NOTIONAL_USDT}"

    tracked_qty = max(0.0, _safe_float(active_trade.get("binance_qty"), 0.0))
    if action == "reduce" and tracked_qty > 0:
        qty = min(qty, tracked_qty)
        qty = _round_down_step(qty, BINANCE_QTY_STEP)
        if qty < BINANCE_MIN_QTY:
            return False, "reduce_qty_too_small"

    if action == "add":
        side = "BUY" if direction == "long" else "SELL"
        position_side = "LONG" if direction == "long" else "SHORT"
        # 補倉用限價單
        ok, data = _binance_place_limit_order(side, qty, current_price, position_side=position_side)
        if ok:
            active_trade["binance_qty"] = tracked_qty + qty
            # 補倉後從 Binance 同步實際均價（取代虛擬計算）
            actual_ep = _fetch_binance_position_entry(direction)
            if actual_ep > 0:
                active_trade["entry"] = actual_ep
                active_trade["avg_entry"] = actual_ep
            return True, data
        return False, data
    else:
        side = "SELL" if direction == "long" else "BUY"
        position_side = "LONG" if direction == "long" else "SHORT"
        # 減倉用市價單（limit+reduceOnly 會觸發 -4061）
        ok, data = _binance_place_market_order(side, qty, reduce_only=True, position_side=position_side)
        if ok:
            active_trade["binance_qty"] = max(0.0, tracked_qty - qty)
            return True, data
        return False, data


def sync_binance_close_position(current_price, reason="TP/SL"):
    if not _is_binance_copy_ready():
        return True, "skip_not_enabled"

    direction = active_trade.get("direction")
    if direction not in ("long", "short"):
        return False, "direction_invalid"

    qty = max(0.0, _safe_float(active_trade.get("binance_qty"), 0.0))
    if qty < BINANCE_MIN_QTY:
        # 若沒有追蹤到實際數量，退回用當前 size 推估
        qty = _binance_qty_from_size_ratio(_safe_float(active_trade.get("size"), 0.0), current_price)
    if qty <= 0:
        return False, "close_qty_too_small"

    # 止盈/手動平倉前先撤掉原生止損單，避免殘單。
    _cancel_native_stop_loss_order()

    side = "SELL" if direction == "long" else "BUY"
    position_side = "LONG" if direction == "long" else "SHORT"
    # 平倉用市價單（limit+reduceOnly 會觸發 -4061）
    ok, data = _binance_place_market_order(side, qty, reduce_only=True, position_side=position_side)
    if ok:
        active_trade["binance_qty"] = 0.0
        _clear_native_stop_tracking()
        return True, data

    send_telegram(f"⚠️ Binance 平倉失敗（{reason}）：{data}", priority=True)
    return False, data


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
    if news_model is not None:
        return  # 已訓練

    texts = [item[0] for item in NEWS_TRAINING_DATA]
    labels = [item[1] for item in NEWS_TRAINING_DATA]

    # 改進的特徵提取：使用雙詞組合 + 子線性 TF
    news_vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),      # 捕捉詞組
        min_df=1,
        sublinear_tf=True,        # 改進詞頻計算
        stop_words='english'
    )
    X = news_vectorizer.fit_transform(texts)
    y = np.array(labels)

    # 使用集成投票模型（Gradient Boosting + Random Forest + Logistic Regression）
    news_model = VotingClassifier(
        estimators=[
            ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)),
            ('lr', LogisticRegression(max_iter=200, random_state=42))
        ],
        voting='soft'  # 使用概率投票
    )
    news_model.fit(X, y)

    # 保存模型
    try:
        with open(NEWS_MODEL_PATH, "wb") as f:
            pickle.dump(news_model, f)
        with open(NEWS_VECTORIZER_PATH, "wb") as f:
            pickle.dump(news_vectorizer, f)
    except:
        pass

def load_news_model():
    global news_model, news_vectorizer
    try:
        with open(NEWS_MODEL_PATH, "rb") as f:
            news_model = pickle.load(f)
        with open(NEWS_VECTORIZER_PATH, "rb") as f:
            news_vectorizer = pickle.load(f)
    except:
        train_news_model()

def predict_news_sentiment(text):
    """預測新聞情緒（舊函數，保持兼容性）"""
    global news_model, news_vectorizer
    if news_model is None:
        load_news_model()
    if news_model is None:
        return 0  # 預設中性

    X = news_vectorizer.transform([text])
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
        X = news_vectorizer.transform([text])
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
def log_prediction_result(news_text, predicted_bias, actual_market_move=None, correct=None):
    """記錄預測結果用於增量學習和精準度評估"""
    try:
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "news": news_text[:150],
            "predicted_bias": predicted_bias,
            "actual_move": actual_market_move,
            "is_correct": correct
        }
        
        with open(NEWS_PERFORMANCE_LOG, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except:
        pass


def get_prediction_accuracy():
    """計算模型預測準確度"""
    try:
        total = 0
        correct = 0
        
        with open(NEWS_PERFORMANCE_LOG, "r") as f:
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
        return {"accuracy": round(accuracy, 2), "total": total, "correct": correct}
    except:
        return {"accuracy": 0, "total": 0, "correct": 0}


def update_learning_buffer(news_text, true_label):
    """將新樣本添加到增量學習緩衝區"""
    try:
        buffer = []
        try:
            with open(NEWS_LEARNING_BUFFER, "rb") as f:
                buffer = pickle.load(f)
        except:
            buffer = []
        
        buffer.append((news_text, true_label))
        
        # 緩衝區最多保留 200 個樣本
        if len(buffer) > 200:
            buffer = buffer[-200:]
        
        with open(NEWS_LEARNING_BUFFER, "wb") as f:
            pickle.dump(buffer, f)
    except:
        pass


def incremental_train_news_model():
    """增量學習：結合原始訓練數據 + 學習緩衝區新樣本進行重新訓練"""
    global news_model, news_vectorizer
    
    texts = [item[0] for item in NEWS_TRAINING_DATA]
    labels = [item[1] for item in NEWS_TRAINING_DATA]
    
    # 讀取學習緩衝區的新樣本
    try:
        with open(NEWS_LEARNING_BUFFER, "rb") as f:
            buffer = pickle.load(f)
            for text, label in buffer:
                texts.append(text)
                labels.append(label)
    except:
        pass
    
    # 防止訓練數據過多導致過擬合
    if len(texts) > 500:
        texts = texts[-500:]
        labels = labels[-500:]
    
    # 重新訓練模型
    try:
        news_vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            stop_words='english'
        )
        X = news_vectorizer.fit_transform(texts)
        y = np.array(labels)

        news_model = VotingClassifier(
            estimators=[
                ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)),
                ('lr', LogisticRegression(max_iter=200, random_state=42))
            ],
            voting='soft'
        )
        news_model.fit(X, y)

        # 保存更新的模型
        with open(NEWS_MODEL_PATH, "wb") as f:
            pickle.dump(news_model, f)
        with open(NEWS_VECTORIZER_PATH, "wb") as f:
            pickle.dump(news_vectorizer, f)
        
        print(f"✓ 增量學習完成：使用 {len(texts)} 個樣本重新訓練模型")
    except Exception as e:
        print(f"✗ 增量學習失敗: {e}")


# 新聞情緒/事件分析（更穩定的分類）
def analyze_news_text(raw_text):
    """更穩定的新聞分類：拆分多空 / 事件 / 影響，避免單一關鍵字誤判。"""
    text = str(raw_text or "").strip()

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

    # 記錄預測結果用於增量學習評估
    log_prediction_result(text, final_bias)

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
    bias = analysis["bias"]

    # ===== 根據利多/利空選擇對應 emoji 與標題 =====
    if bias >= 2:
        header_emoji = "🟢"
        header_label = "市場利多快訊（即時）"
    elif bias == 1:
        header_emoji = "🟢"
        header_label = "市場輕微利多快訊（即時）"
    elif bias == -1:
        header_emoji = "🔴"
        header_label = "市場輕微利空快訊（即時）"
    elif bias <= -2:
        header_emoji = "🔴"
        header_label = "市場利空快訊（即時）"
    else:
        header_emoji = "🟡"
        header_label = "市場快訊（即時）"
    
    # ===== 顯示 AI 學習狀態 =====
    accuracy_info = get_prediction_accuracy()
    accuracy_str = f"準率: {accuracy_info['accuracy']}% ({accuracy_info['correct']}/{accuracy_info['total']})" if accuracy_info['total'] > 0 else "準率: 初始化中"

    return (
        f"{header_emoji} {header_label}\n"
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
                analysis = analyze_news_text(text)
                news_bias += int(analysis.get("bias", 0))
                event_risk += int(analysis.get("event_risk", 0))
                news_list.append(f"[{src}] {text[:200]}")

        # 若本輪沒有新快訊，回退為近期標題，避免監控面板長期顯示「暫無資料」
        if not news_list and latest_news:
            news_list = latest_news

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


def manage_position_scaling(current_price, atr=None):
    """持倉中的補倉/減倉管理（虛擬倉位）。"""
    if not active_trade.get("open"):
        return

    now_ts = time.time()
    cooldown = 120
    add_step = 0.12
    reduce_step = 0.12
    max_add_count = 20

    last_adjust = _safe_float(active_trade.get("last_adjust_ts"), 0.0)
    if now_ts - last_adjust < cooldown:
        return

    direction = active_trade.get("direction")
    entry = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), current_price)
    size = max(0.0, _safe_float(active_trade.get("size"), 0.0))
    max_size = max(1.0, _safe_float(active_trade.get("max_size"), 1.0))
    min_size = _safe_float(active_trade.get("min_size"), 0.15)
    add_count = int(active_trade.get("add_count", 0))

    # 以進場價附近與小幅浮盈作為調整條件，避免無限頻繁調倉
    if direction == "long":
        add_trigger = current_price <= entry * 0.997
        reduce_trigger = current_price >= entry * 1.004
    elif direction == "short":
        add_trigger = current_price >= entry * 1.003
        reduce_trigger = current_price <= entry * 0.996
    else:
        return

    # 補倉：逆勢回踩時逐步加碼（有上限）
    if add_trigger and add_count < max_add_count and size < max_size - 1e-9:
        delta = min(add_step, max_size - size)
        if delta > 0:
            ok_sync, sync_msg = sync_binance_adjust_position(direction, "add", delta, current_price)
            if not ok_sync:
                send_telegram(f"⚠️ Binance 補倉失敗，已取消本次補倉: {sync_msg}", priority=True)
                return
            new_size = size + delta
            # 均價更新（虛擬倉位）
            new_entry = ((entry * size) + (current_price * delta)) / max(new_size, 1e-9)
            tp_text = f"{_safe_float(active_trade.get('tp'), 0.0):.2f}" if active_trade.get("tp") is not None else "N/A"
            sl_text = f"{_safe_float(active_trade.get('sl'), 0.0):.2f}" if active_trade.get("sl") is not None else "N/A"
            active_trade["entry"] = float(new_entry)
            active_trade["avg_entry"] = float(new_entry)
            active_trade["size"] = float(new_size)
            active_trade["add_count"] = add_count + 1
            active_trade["last_adjust_ts"] = now_ts
            send_telegram(
                f"➕ 補倉（{direction}）\n"
                f"現價: {current_price:.2f} | 加倉: +{int(delta*100)}%\n"
                f"進場均價: {new_entry:.2f} | TP: {tp_text} | SL: {sl_text}\n"
                f"倉位: {int(size*100)}% → {int(new_size*100)}%",
                priority=True,
            )
            _send_private_telegram_text(
                f"➕ 補倉（{direction}）\n"
                f"現價: {current_price:.2f} | 加倉: +{int(delta*100)}%\n"
                f"進場均價: {new_entry:.2f} | TP: {tp_text} | SL: {sl_text}\n"
                f"倉位: {int(size*100)}% → {int(new_size*100)}%"
            )
            send_follow_action_alert("add", direction, current_price, delta, new_size, new_entry, tp_text, sl_text)
            refresh_position_panel_from_active_trade()
            return

    # 減倉：有利方向浮盈時鎖定部分利潤（保留底倉）
    if reduce_trigger and size > min_size + 1e-9:
        delta = min(reduce_step, size - min_size)
        if delta > 0:
            ok_sync, sync_msg = sync_binance_adjust_position(direction, "reduce", delta, current_price)
            if not ok_sync:
                send_telegram(f"⚠️ Binance 減倉失敗，已取消本次減倉: {sync_msg}", priority=True)
                return
            new_size = size - delta
            tp_text = f"{_safe_float(active_trade.get('tp'), 0.0):.2f}" if active_trade.get("tp") is not None else "N/A"
            sl_text = f"{_safe_float(active_trade.get('sl'), 0.0):.2f}" if active_trade.get("sl") is not None else "N/A"
            active_trade["size"] = float(new_size)
            active_trade["reduce_count"] = int(active_trade.get("reduce_count", 0)) + 1
            active_trade["last_adjust_ts"] = now_ts
            send_telegram(
                f"➖ 減倉（{direction}）\n"
                f"現價: {current_price:.2f} | 減倉: -{int(delta*100)}%\n"
                f"進場均價: {entry:.2f} | TP: {tp_text} | SL: {sl_text}\n"
                f"倉位: {int(size*100)}% → {int(new_size*100)}%",
                priority=True,
            )
            _send_private_telegram_text(
                f"➖ 減倉（{direction}）\n"
                f"現價: {current_price:.2f} | 減倉: -{int(delta*100)}%\n"
                f"進場均價: {entry:.2f} | TP: {tp_text} | SL: {sl_text}\n"
                f"倉位: {int(size*100)}% → {int(new_size*100)}%"
            )
            send_follow_action_alert("reduce", direction, current_price, delta, new_size, entry, tp_text, sl_text)
            refresh_position_panel_from_active_trade()


def maybe_decay_take_profit(current_price):
    """同一張單持倉超過 4 小時後，逐步降低止盈位。"""
    if not active_trade.get("open"):
        return

    direction = active_trade.get("direction")
    if direction not in ("long", "short"):
        return

    entry = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), 0.0)
    tp = _safe_float(active_trade.get("tp"), 0.0)
    open_ts = _safe_float(active_trade.get("open_ts"), 0.0)
    if entry <= 0 or tp <= 0 or open_ts <= 0:
        return

    start_after = 4 * 60 * 60
    every = 60 * 60
    decay_ratio = 0.18

    held_sec = time.time() - open_ts
    if held_sec < start_after:
        return

    expected_count = int((held_sec - start_after) // every) + 1
    current_count = int(active_trade.get("tp_decay_count", 0))
    steps = expected_count - current_count
    if steps <= 0:
        return

    old_tp = tp
    for _ in range(steps):
        if direction == "long":
            dist = max(tp - entry, 0.0)
            if dist <= 0:
                break
            tp = entry + dist * (1.0 - decay_ratio)
            tp = max(tp, current_price * 1.0004)
        else:
            dist = max(entry - tp, 0.0)
            if dist <= 0:
                break
            tp = entry - dist * (1.0 - decay_ratio)
            tp = min(tp, current_price * 0.9996)

    active_trade["tp"] = float(tp)
    active_trade["tp_decay_count"] = expected_count

    if abs(tp - old_tp) > 1e-9:
        hours = held_sec / 3600.0
        send_telegram(
            f"⏱️ 持倉超時下修止盈（{direction}）\n"
            f"持倉: {hours:.1f}h | 現價: {current_price:.2f}\n"
            f"TP: {old_tp:.2f} → {tp:.2f}",
            priority=True,
        )
        refresh_position_panel_from_active_trade()
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
        if tp >= min_tp:
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
ONLINE_MODEL_PATH = "/Volumes/SSD/trading/online_model.pkl"

# =============================
# 全域
# =============================
WS_LOCK = threading.Lock()
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
    "open": False,
    "size": 0.0,
    "max_size": 1.0,
    "min_size": 0.15,
    "add_count": 0,
    "reduce_count": 0,
    "last_adjust_ts": 0.0,
    "open_ts": 0.0,
    "tp_decay_count": 0,
    "binance_qty": 0.0,
    "binance_sl_order_id": None,
    "binance_sl_price": 0.0,
    "last_close_reason": "",
    "last_close_price": 0.0,
    "last_close_ts": 0,
    "last_close_candle_high": 0.0,
    "last_close_candle_low": 0.0,
    "close_hits": [],
}


def _record_close_hit(reason, current_price, candle_high, candle_low):
    """記錄最近一次平倉原因與命中價格資訊，供 mini app 顯示。"""
    active_trade["last_close_reason"] = str(reason or "").upper()
    active_trade["last_close_price"] = _safe_float(current_price, 0.0)
    active_trade["last_close_candle_high"] = _safe_float(candle_high, 0.0)
    active_trade["last_close_candle_low"] = _safe_float(candle_low, 0.0)
    hit_ts = int(time.time())
    active_trade["last_close_ts"] = hit_ts

    # 保留最近命中紀錄（供 mini app 顯示過去 TP/SL）
    hits = active_trade.get("close_hits")
    if not isinstance(hits, list):
        hits = []
    hits.append(
        {
            "reason": str(reason or "").upper(),
            "price": _safe_float(current_price, 0.0),
            "candle_high": _safe_float(candle_high, 0.0),
            "candle_low": _safe_float(candle_low, 0.0),
            "ts": hit_ts,
        }
    )
    active_trade["close_hits"] = hits[-10:]


def _reset_active_trade_after_manual_close(close_price):
    """手動平倉成功後重置持倉狀態。"""
    px = _safe_float(close_price, 0.0)
    _record_close_hit("MANUAL", px, px, px)
    active_trade["open"] = False
    active_trade["size"] = 0.0
    active_trade["add_count"] = 0
    active_trade["reduce_count"] = 0
    active_trade["open_ts"] = 0.0
    active_trade["tp_decay_count"] = 0
    active_trade["binance_qty"] = 0.0

# =============================
# KLINE CACHE（避免打爆API）
# =============================

# ===== 跟單模式 =====
FOLLOW_MODE_ENABLED = False
_FOLLOW_BUTTON_MSG_ID = None  # 跟單按鈕訊息 ID（用於更新按鈕文字）
KLINE_CACHE = {}
KLINE_TTL = {
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
        except Exception as e:
            print(f"⚠️ WebSocket 斷線，2 秒後重連: {e}")
            time.sleep(2)

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

# =============================
# AI（Meta Model）
# =============================
MODEL_PATH = "/Volumes/SSD/trading/model.pkl"
DATA_PATH = "/Volumes/SSD/trading/ai_data.csv"
model = None
# Online learning model
online_model = SGDClassifier(loss="log_loss")
online_initialized = False

def load_model():
    global model, online_model, online_initialized

    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            print(f"⚠️ 加載模型失敗: {e}")
            model = None

    if os.path.exists(ONLINE_MODEL_PATH):
        try:
            with open(ONLINE_MODEL_PATH, "rb") as f:
                online_model = pickle.load(f)
            online_initialized = True
        except:
            print("⚠️ 舊模型不相容，重置 online_model")
            online_model = SGDClassifier(loss="log_loss")
            online_initialized = False

    # 加載新聞模型
    load_news_model()

def update_online_model(features, label):
    global online_model, online_initialized

    X = pd.DataFrame([features])
    y = np.array([label])

    # ===== FIX: 強制對齊 feature columns =====
    if online_initialized and hasattr(online_model, "feature_names_in_"):
        expected_cols = list(online_model.feature_names_in_)

        # 補缺的欄位
        for col in expected_cols:
            if col not in X.columns:
                X[col] = 0

        # 移除多餘欄位
        X = X[expected_cols]

    try:
        if not online_initialized:
            online_model.partial_fit(X, y, classes=np.array([0, 1]))
            online_initialized = True
        else:
            online_model.partial_fit(X, y)
    except Exception as e:
        print("⚠️ online_model error, reset model:", e)
        online_model = SGDClassifier(loss="log_loss")
        online_model.partial_fit(X, y, classes=np.array([0, 1]))
        online_initialized = True

    # persist online model
    def _save():
        try:
            with open(ONLINE_MODEL_PATH, "wb") as f:
                pickle.dump(online_model, f)
        except:
            pass

    threading.Thread(target=_save, daemon=True).start()

def train_model():
    global model
    if not os.path.exists(DATA_PATH):
        return

    df = pd.read_csv(DATA_PATH)
    if len(df) < 50:
        return

    X = df.drop(columns=["label"])
    y = df["label"]

    model = RandomForestClassifier(n_estimators=120)
    model.fit(X, y)

    def _save_model():
        try:
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(model, f)
        except:
            pass
    threading.Thread(target=_save_model, daemon=True).start()
    print("✅ AI 更新")

def retrain_model():
    """強制重新訓練 AI 模型"""
    global model
    print("🔄 開始重新訓練 AI 模型...")

    if not os.path.exists(DATA_PATH):
        print("⚠️ 沒有訓練數據檔案")
        return

    try:
        df = pd.read_csv(DATA_PATH, header=None)
        # 數據沒有標題，直接使用最後一列作為 label
        df.columns = [f"feature_{i}" for i in range(len(df.columns) - 1)] + ["label"]
    except Exception as e:
        print(f"⚠️ 讀取訓練數據失敗: {e}")
        return

    if len(df) < 10:
        print("⚠️ 訓練數據不足（至少需要10筆）")
        return

    # 只使用最近的1000筆數據來加速訓練
    if len(df) > 1000:
        df = df.tail(1000)
        print(f"✅ 使用最近1000筆數據訓練")

    # 確保有 label 列
    if "label" not in df.columns:
        print("⚠️ 數據缺少 label 列")
        return

    X = df.drop(columns=["label"])
    y = df["label"]

    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X, y)

    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        print("✅ AI 模型重新訓練完成並保存")
    except Exception as e:
        print(f"⚠️ 保存模型失敗: {e}")

log_buffer = []

def log_data(features, label):
    global log_buffer

    log_buffer.append({**features, "label": label})

    if len(log_buffer) >= 20:
        df = pd.DataFrame(log_buffer)

        if os.path.exists(DATA_PATH):
            df.to_csv(DATA_PATH, mode="a", header=False, index=False)
        else:
            df.to_csv(DATA_PATH, index=False)

        log_buffer = []


def _send_private_telegram_text(msg):
    
    """若有設定私聊 chat id，則同步發送文字訊息到私聊。"""
    private_chat_id = str(TELEGRAM_PRIVATE_CHAT_ID or "").strip()
    group_chat_id = str(TELEGRAM_CHAT_ID or "").strip()
    if not TELEGRAM_TOKEN or not private_chat_id:
        return
    if private_chat_id == group_chat_id:
        return

    try:
        payload = _with_forced_remove_reply_keyboard(
            {
                "chat_id": private_chat_id,
                "text": str(msg),
            },
            private_chat_id,
        )
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json=payload,
            timeout=5,
        )
    except Exception as e:
        print(f"⚠️ 私聊訊息發送失敗: {e}")

def send_telegram(msg, priority=False):
    global LAST_TELEGRAM_TS

    now = time.time()

    # ===== 只有低優先才限流 =====
    if not priority and now - LAST_TELEGRAM_TS < 10:
        return

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram 未設定，略過發送")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    # V7 安全版 + 避免特殊字元炸掉
    safe_msg = str(msg).replace("<", "").replace(">", "").replace("&", "and")

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": safe_msg
    }
    payload = _with_forced_remove_reply_keyboard(payload, TELEGRAM_CHAT_ID)

    try:
        res = requests.post(url, json=payload, timeout=5)
        if res.status_code == 400:
            # fallback without any特殊字元問題
            payload["text"] = str(msg).replace("<", "").replace(">", "")
            payload = _with_forced_remove_reply_keyboard(payload, TELEGRAM_CHAT_ID)
            res = requests.post(url, json=payload, timeout=5)
        sent_message_id = None

        if res.status_code != 200:
            print("❌ Telegram 發送失敗:", res.status_code, res.text)
        else:
            print("✅ Telegram 已送出")
            try:
                body = res.json()
                sent_message_id = body.get("result", {}).get("message_id")
            except Exception:
                sent_message_id = None

        # ===== retry（避免偶發失敗） =====
        if res.status_code != 200:
            try:
                time.sleep(1)
                payload = _with_forced_remove_reply_keyboard(payload, TELEGRAM_CHAT_ID)
                res2 = requests.post(url, json=payload, timeout=5)
                print("🔁 retry:", res2.status_code)
            except Exception as e:
                print("❌ retry失敗:", e)

        # Discord只發「進場通知」
        try:
            if DISCORD_WEBHOOK and "進場" in msg:
                requests.post(DISCORD_WEBHOOK, json={"content": msg}, timeout=5)
        except Exception as e:
            print("Discord error:", e)

        LAST_TELEGRAM_TS = now

    except Exception as e:
        print("❌ Telegram error:", e, "| msg:", msg[:50])


WEBAPP_BASE_URL = "https://josh940085.github.io/ETH-bot/index.html"
WEBAPP_VERSION = str(os.getenv("WEBAPP_VERSION", "20260421-1")).strip() or "20260421-1"


def _build_private_bottom_keyboard():
    """建立私聊底部鍵盤（倉位面板 + 跟單設定 + 手動平倉），避免被 remove_keyboard 收掉。"""
    has_open_position = active_trade.get("open") and active_trade.get("direction") in ("long", "short")
    if has_open_position:
        direction = active_trade.get("direction")
        entry = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), 0.0)
        tp = _safe_float(active_trade.get("tp"), 0.0)
        sl = _safe_float(active_trade.get("sl"), 0.0)
        size = max(0.0, _safe_float(active_trade.get("size"), 0.0))
        size_pct = int(size * 100)
        dir_param = "long" if direction == "long" else "short"
        url = (
            f"{WEBAPP_BASE_URL}"
            f"?v={WEBAPP_VERSION}"
            f"&dir={dir_param}"
            f"&entry={entry:.2f}"
            f"&tp={tp:.2f}"
            f"&sl={sl:.2f}"
            f"&size={size_pct}"
            f"&pair=ETHUSDT"
            f"&lev=10"
        )
    else:
        url = (
            f"{WEBAPP_BASE_URL}"
            f"?v={WEBAPP_VERSION}"
            f"&restart=1"
            f"&pair=ETHUSDT"
            f"&lev=10"
            f"&t={int(time.time())}"
        )

    keyboard_rows = [[
        {"text": "📊 倉位面板", "web_app": {"url": url}},
        {"text": "👥 跟單設定"},
    ]]
    if has_open_position:
        keyboard_rows.append([
            {"text": "🧾 手動平倉"},
        ])

    return {
        "keyboard": keyboard_rows,
        "resize_keyboard": True,
        "persistent": True,
    }


def send_position_keyboard(direction, entry, tp, sl, size, entry_display=None, tp_display=None, sl_display=None, is_update=False):
    """進場後在 Telegram 發出倉位面板按鈕（私聊用 Web App，群組/頻道用 URL 按鈕）。
    entry_display, tp_display, sl_display: 若提供則使用此字串確保訊息與網址一致。"""
    if not TELEGRAM_TOKEN:
        return

    try:
        dir_param  = "long" if direction == "long" else "short"
        size_pct   = int(float(size) * 100)
        
        # 使用傳入的顯示價格，確保與訊息一致
        entry_str = entry_display if entry_display else f"{entry:.2f}"
        tp_str = tp_display if tp_display else (f"{tp:.2f}" if tp is not None else "0.0")
        sl_str = sl_display if sl_display else (f"{sl:.2f}" if sl is not None else "0.0")
        
        url = (
            f"{WEBAPP_BASE_URL}"
            f"?v={WEBAPP_VERSION}"
            f"&dir={dir_param}"
            f"&entry={entry_str}"
            f"&tp={tp_str}"
            f"&sl={sl_str}"
            f"&size={size_pct}"
            f"&pair=ETHUSDT"
            f"&lev=10"
        )
        
        main_chat_id = str(TELEGRAM_CHAT_ID or "").strip()
        private_chat_id = str(TELEGRAM_PRIVATE_CHAT_ID or "").strip()

        # 群組/頻道：inline URL 按鈕
        group_keyboard = {
            "inline_keyboard": [[
                {"text": "📊 倉位面板", "url": url}
            ]]
        }
        group_text = "📊 倉位面板已更新，點擊按鈕查看最新數據" if is_update else "📊 倉位已建立，點擊按鈕查看即時面板"

        # 私聊：底部鍵盤（倉位面板 web_app + 跟單設定 + 手動平倉）
        private_keyboard = {
            "keyboard": [[
                {"text": "📊 倉位面板", "web_app": {"url": url}},
                {"text": "👥 跟單設定"},
            ], [
                {"text": "🧾 手動平倉"},
            ]],
            "resize_keyboard": True,
            "persistent": True,
        }
        private_text = "📊 倉位面板已更新" if is_update else "📊 倉位已建立，面板顯示在下方"

        # 主 chat
        if main_chat_id:
            if _is_private_chat(main_chat_id):
                requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                    json={
                        "chat_id": main_chat_id,
                        "text": private_text,
                        "reply_markup": private_keyboard,
                    },
                    timeout=5,
                )
            else:
                requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                    json={
                        "chat_id": main_chat_id,
                        "text": group_text,
                        "reply_markup": group_keyboard,
                    },
                    timeout=5,
                )

        # 私聊額外通知：TELEGRAM_PRIVATE_CHAT_ID 與主 chat 不同時才發送
        if private_chat_id and _is_private_chat(private_chat_id) and private_chat_id != main_chat_id:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={
                    "chat_id": private_chat_id,
                    "text": private_text,
                    "reply_markup": private_keyboard,
                },
                timeout=5,
            )
    except Exception as e:
        print(f"⚠️ 倉位面板按鈕發送失敗: {e}")


def refresh_position_panel_from_active_trade():
    """依照目前 active_trade 狀態重送面板按鈕，保持 URL 參數與交易狀態同步。"""
    if not active_trade.get("open"):
        return

    direction = active_trade.get("direction")
    if direction not in ("long", "short"):
        return

    entry = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), 0.0)
    size = max(0.0, _safe_float(active_trade.get("size"), 0.0))
    tp = active_trade.get("tp")
    sl = active_trade.get("sl")

    if entry <= 0 or size <= 0:
        return

    entry_str = f"{entry:.2f}"
    tp_str = f"{_safe_float(tp, 0.0):.2f}" if tp is not None else "0.0"
    sl_str = f"{_safe_float(sl, 0.0):.2f}" if sl is not None else "0.0"

    send_position_keyboard(
        direction,
        entry,
        tp,
        sl,
        size,
        entry_display=entry_str,
        tp_display=tp_str,
        sl_display=sl_str,
        is_update=True,
    )
    write_position_json()
    push_position_json()


_last_position_push_ts = 0
_POSITION_PUSH_MIN_INTERVAL = 5  # 最短推送間隔（秒），可透過環境變數 POSITION_PUSH_INTERVAL 調整
_position_push_lock = Lock()
_position_push_git_lock = Lock()


def write_position_json():
    """將目前 active_trade 狀態寫入 docs/position.json，供 mini app / web app 即時讀取。"""
    try:
        is_open = bool(active_trade.get("open", False))
        # 使用 avg_entry（平均進場價），與 refresh_position_panel_from_active_trade 保持一致
        entry = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), 0.0)
        tp = _safe_float(active_trade.get("tp"), 0.0)
        sl = _safe_float(active_trade.get("sl"), 0.0)
        size = max(0.0, _safe_float(active_trade.get("size"), 0.0))
        tracked_qty = max(0.0, _safe_float(active_trade.get("binance_qty"), 0.0))
        lev = max(1.0, _safe_float(BINANCE_LEVERAGE, 10.0))
        position_notional = tracked_qty * entry if (tracked_qty > 0 and entry > 0) else 0.0
        position_margin = (position_notional / lev) if position_notional > 0 else 0.0
        raw_hits = active_trade.get("close_hits")
        close_hits = []
        if isinstance(raw_hits, list):
            for item in raw_hits[-10:]:
                if not isinstance(item, dict):
                    continue
                close_hits.append(
                    {
                        "reason": str(item.get("reason") or "").upper(),
                        "price": round(_safe_float(item.get("price"), 0.0), 4),
                        "candle_high": round(_safe_float(item.get("candle_high"), 0.0), 4),
                        "candle_low": round(_safe_float(item.get("candle_low"), 0.0), 4),
                        "ts": int(_safe_float(item.get("ts"), 0.0)),
                    }
                )
        data = {
            "open": is_open,
            "direction": active_trade.get("direction") or "",
            "entry": round(entry, 4),
            "tp": round(tp, 4),
            "sl": round(sl, 4),
            "size": round(size, 4),
            "last_close_reason": str(active_trade.get("last_close_reason") or ""),
            "last_close_price": round(_safe_float(active_trade.get("last_close_price"), 0.0), 4),
            "last_close_ts": int(_safe_float(active_trade.get("last_close_ts"), 0.0)),
            "last_close_candle_high": round(_safe_float(active_trade.get("last_close_candle_high"), 0.0), 4),
            "last_close_candle_low": round(_safe_float(active_trade.get("last_close_candle_low"), 0.0), 4),
            "close_hits": close_hits,
            "binance_qty": round(tracked_qty, 6),
            "position_notional_usdt": round(position_notional, 4),
            "position_margin_usdt": round(position_margin, 4),
            "pair": "ETHUSDT",
            "lev": int(lev),
            "ts": int(time.time()),
        }
        POSITION_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        POSITION_JSON_PATH.write_text(
            json.dumps(data, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"⚠️ 寫入 position.json 失敗: {e}")


def push_position_json():
    """在背景執行緒中 git commit + push docs/position.json（有節流保護）。"""
    global _last_position_push_ts
    try:
        min_interval = max(3.0, float(os.getenv("POSITION_PUSH_INTERVAL", str(_POSITION_PUSH_MIN_INTERVAL))))
    except Exception:
        min_interval = float(_POSITION_PUSH_MIN_INTERVAL)
    now = time.time()
    with _position_push_lock:
        if now - _last_position_push_ts < min_interval:
            return
        _last_position_push_ts = now

    repo_dir = str(POSITION_JSON_PATH.parent.parent)
    rel_path = "docs/position.json"

    def _push():
        def _run_git(args, timeout=30):
            return subprocess.run(
                ["git"] + args,
                cwd=repo_dir,
                timeout=timeout,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

        try:
            with _position_push_git_lock:
                _run_git(["add", rel_path], timeout=10)

                result = _run_git(["commit", "-m", "chore: update position data"], timeout=10)
                commit_out = (result.stdout or "").strip()

                if result.returncode != 0 and "nothing to commit" not in commit_out:
                    print(f"⚠️ 提交 position.json 失敗: {commit_out}")
                    return

                push_result = _run_git(["push"], timeout=30)
                if push_result.returncode == 0:
                    return

                push_out = (push_result.stdout or "").strip()
                non_ff = (
                    "non-fast-forward" in push_out
                    or "fetch first" in push_out
                    or "tip of your current branch is behind" in push_out
                )

                if not non_ff:
                    print(f"⚠️ 推送 position.json 失敗: {push_out}")
                    return

                # 遠端比本地新：先 rebase 同步後再重推
                branch_res = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], timeout=10)
                branch = (branch_res.stdout or "").strip() or "HEAD"

                pull_result = _run_git(["pull", "--rebase", "--autostash", "origin", branch], timeout=45)
                if pull_result.returncode != 0:
                    print(f"⚠️ 自動同步遠端失敗: {(pull_result.stdout or '').strip()}")
                    return

                retry_push = _run_git(["push"], timeout=30)
                if retry_push.returncode != 0:
                    print(f"⚠️ 推送 position.json 失敗(重試後): {(retry_push.stdout or '').strip()}")
        except Exception as e:
            print(f"⚠️ 推送 position.json 失敗: {e}")

    threading.Thread(target=_push, daemon=True).start()


def remove_position_keyboard():
    """平倉後更新底部 web_app 鍵盤為待機狀態（無倉位面板），讓使用者可重啟。"""
    if not TELEGRAM_TOKEN:
        return

    main_chat_id = str(TELEGRAM_CHAT_ID or "").strip()
    private_chat_id = str(TELEGRAM_PRIVATE_CHAT_ID or "").strip()

    target_ids = []
    if main_chat_id and _is_private_chat(main_chat_id):
        target_ids.append(main_chat_id)
    if private_chat_id and _is_private_chat(private_chat_id) and private_chat_id not in target_ids:
        target_ids.append(private_chat_id)

    restart_url = (
        f"{WEBAPP_BASE_URL}"
        f"?v={WEBAPP_VERSION}"
        f"&restart=1"
        f"&pair=ETHUSDT"
        f"&lev=10"
        f"&t={int(time.time())}"
    )
    restart_keyboard = {
        "keyboard": [[
            {"text": "📊 倉位面板", "web_app": {"url": restart_url}},
            {"text": "👥 跟單設定"},
        ]],
        "resize_keyboard": True,
        "persistent": True,
    }

    for cid in target_ids:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={
                    "chat_id": cid,
                    "text": "📋 倉位已平倉，面板已重置",
                    "reply_markup": restart_keyboard,
                },
                timeout=5,
            )
        except Exception as e:
            print(f"⚠️ 重置底部面板失敗: {e}")

    write_position_json()
    push_position_json()




# ===== AI分析（OpenClaw / OpenAI） =====
OPENAI_API_KEY = _get_required_env("OPENAI_API_KEY", "", mask=True)


# ===== 跟單模式輔助函數 =====
def toggle_follow_mode():
    """切換跟單模式開/關，回傳切換後狀態。"""
    global FOLLOW_MODE_ENABLED
    FOLLOW_MODE_ENABLED = not FOLLOW_MODE_ENABLED
    return FOLLOW_MODE_ENABLED


def send_follow_button(is_update=False):
    """發送或更新跟單切換按鈕（inline keyboard）。"""
    global _FOLLOW_BUTTON_MSG_ID
    private_chat_id = _resolve_follow_private_chat_id()
    if not TELEGRAM_TOKEN or not private_chat_id:
        return

    btn_text = "✅ 跟單中（點擊關閉幣安下單）" if FOLLOW_MODE_ENABLED else "📈 開啟跟單（幣安自動下單）"

    try:
        if is_update and _FOLLOW_BUTTON_MSG_ID:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageReplyMarkup",
                json={
                    "chat_id": private_chat_id,
                    "message_id": _FOLLOW_BUTTON_MSG_ID,
                    "reply_markup": {
                        "inline_keyboard": [[
                            {"text": btn_text, "callback_data": "toggle_follow"}
                        ]]
                    },
                },
                timeout=5,
            )
        else:
            resp = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={
                    "chat_id": private_chat_id,
                    "text": "👥 跟單模式：開啟後，Bot 每次開倉/補倉/減倉/平倉都會同步在你的幣安帳戶下對應市價單",
                    "reply_markup": {
                        "inline_keyboard": [[
                            {"text": btn_text, "callback_data": "toggle_follow"}
                        ]]
                    },
                },
                timeout=5,
            )
            result = resp.json()
            if result.get("ok"):
                _FOLLOW_BUTTON_MSG_ID = result["result"]["message_id"]
    except Exception as e:
        print(f"⚠️ 跟單按鈕發送失敗: {e}")


def send_follow_action_alert(action, direction, current_price, delta, new_size, entry, tp_text, sl_text):
    """跟單模式下，補倉/減倉時推送跟單提醒訊息。"""
    if not FOLLOW_MODE_ENABLED:
        return
    private_chat_id = _resolve_follow_private_chat_id()
    if not TELEGRAM_TOKEN or not private_chat_id:
        return

    action_emoji = "➕" if action == "add" else "➖"
    action_zh   = "補倉" if action == "add" else "減倉"
    dir_zh      = "做多" if direction == "long" else "做空"

    try:
        payload = _with_forced_remove_reply_keyboard(
            {
                "chat_id": private_chat_id,
                "text": (
                    f"👥 跟單提醒 | {action_emoji} {action_zh}\n"
                    f"━━━━━━━━━━━━━━\n"
                    f"方向: {dir_zh} | 現價: {current_price:.2f}\n"
                    f"操作: {action_zh} {int(delta * 100)}%\n"
                    f"新倉位: {int(new_size * 100)}% | 均價: {entry:.2f}\n"
                    f"TP: {tp_text} | SL: {sl_text}\n"
                    f"━━━━━━━━━━━━━━"
                ),
            },
            private_chat_id,
        )
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json=payload,
            timeout=5,
        )
    except Exception as e:
        print(f"⚠️ 跟單私聊提醒發送失敗: {e}")


def _process_follow_callback(cq_data, cq_id, cq_msg_id, chat_id):
    """處理 inline button 的 callback_query（跟單切換）。"""
    private_chat_id = _resolve_follow_private_chat_id()
    if cq_data != "toggle_follow":
        return
    if not private_chat_id or str(chat_id) != str(private_chat_id):
        return

    is_enabled = toggle_follow_mode()
    has_keys = bool(BINANCE_API_KEY and BINANCE_API_SECRET)
    if is_enabled and not has_keys:
        # API Key 未設定，回滾並告知
        toggle_follow_mode()  # 再切回 False
        is_enabled = False
        if cq_id:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery",
                data={"callback_query_id": cq_id, "text": "❌ 尚未設定 BINANCE_API_KEY / BINANCE_API_SECRET，無法啟動跟單", "show_alert": True},
                timeout=5,
            )
        return
    status_text = "✅ 跟單模式已開啟！後續 Bot 操作將在幣安帳戶同步下單" if is_enabled else "⏹️ 跟單模式已關閉，幣安下單停止"
    btn_text    = "✅ 跟單中（點擊關閉幣安下單）" if is_enabled else "📈 開啟跟單（幣安自動下單）"

    # 開啟跟單且目前已有倉位 → 立刻在幣安同步開倉
    if is_enabled and active_trade.get("open") and active_trade.get("direction"):
        cur_price = WS_PRICE or active_trade.get("entry", 0)
        if cur_price and cur_price > 0:
            cur_size = _safe_float(active_trade.get("size"), 0.2)
            ok_sync, sync_msg = sync_binance_open_position(active_trade["direction"], cur_size, cur_price)
            if ok_sync:
                status_text += "\n📌 已在幣安同步開立現有倉位"
            else:
                status_text += f"\n⚠️ 幣安同步開倉失敗: {sync_msg}"

    try:
        if cq_id:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery",
                data={"callback_query_id": cq_id, "text": status_text, "show_alert": True},
                timeout=5,
            )

        msg_id = None
        try:
            msg_id = int(cq_msg_id) if cq_msg_id else _FOLLOW_BUTTON_MSG_ID
        except (TypeError, ValueError):
            msg_id = _FOLLOW_BUTTON_MSG_ID

        if msg_id and private_chat_id:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageReplyMarkup",
                json={
                    "chat_id": private_chat_id,
                    "message_id": msg_id,
                    "reply_markup": {
                        "inline_keyboard": [[
                            {"text": btn_text, "callback_data": "toggle_follow"}
                        ]]
                    },
                },
                timeout=5,
            )
    except Exception as e:
        print(f"⚠️ 跟單回調處理失敗: {e}")

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


def _parse_telegram_state_payload(raw):
    try:
        payload = json.loads(raw) if str(raw).strip() else {}
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _read_telegram_state_locked():
    if not TELEGRAM_STATE_PATH.exists():
        return {}
    try:
        with TELEGRAM_STATE_PATH.open("r", encoding="utf-8") as fh:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
            raw = fh.read()
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return _parse_telegram_state_payload(raw)
    except Exception:
        return {}


def _update_telegram_state_locked(mutator):
    try:
        TELEGRAM_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TELEGRAM_STATE_PATH.open("a+", encoding="utf-8") as fh:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)

            fh.seek(0)
            payload = _parse_telegram_state_payload(fh.read())
            result = mutator(payload)

            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(payload, ensure_ascii=False))
            fh.flush()
            os.fsync(fh.fileno())

            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            return result
    except Exception:
        return None


def load_last_update_id():
    payload = _read_telegram_state_locked()
    value = payload.get("last_update_id")
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def save_last_update_id(update_id):
    _update_telegram_state_locked(lambda payload: payload.__setitem__("last_update_id", int(update_id)))


def request_supervisor_restart():
    def _mutate(payload):
        payload["restart_requested"] = True
        payload["restart_requested_at"] = int(time.time())
        return True

    return bool(_update_telegram_state_locked(_mutate))


def pop_pending_commands():
    def _mutate(payload):
        pending = payload.get("pending_commands")
        commands = pending if isinstance(pending, list) else []
        payload["pending_commands"] = []
        return commands

    result = _update_telegram_state_locked(_mutate)
    return result if isinstance(result, list) else []


# ===== Telegram 指令（AI分析） =====
def handle_ai_command(text, context=None, chat_id=None):
    global BOT_SOFT_RESTART_REQUESTED

    if text.startswith("/restart"):
        if os.getenv("BOT_SUPERVISOR") == "1":
            if request_supervisor_restart():
                return "♻️ 已收到 /restart，將由 program.py 執行同步並重啟。"
            return "⚠️ /restart 失敗：無法寫入重啟請求。"

        BOT_SOFT_RESTART_REQUESTED = True
        return "♻️ 已收到 /restart，將在本程序內執行軟重啟。"

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

    if text.startswith("/follow"):
        private_chat_id = _resolve_follow_private_chat_id()
        if not private_chat_id:
            return "⚠️ 尚未設定 TELEGRAM_PRIVATE_CHAT_ID，無法啟用私聊跟單。"
        if not _is_private_chat(chat_id) or str(chat_id) != str(private_chat_id):
            return "🔒 跟單功能僅限私聊使用，請到私聊對話輸入 /follow。"

        is_enabled = toggle_follow_mode()
        if is_enabled:
            if not (BINANCE_API_KEY and BINANCE_API_SECRET):
                toggle_follow_mode()  # 回滾
                return "❌ 尚未設定 BINANCE_API_KEY / BINANCE_API_SECRET，無法啟動跟單。"
            # 若目前已有倉位，立刻同步到幣安
            sync_note = ""
            if active_trade.get("open") and active_trade.get("direction"):
                cur_price = WS_PRICE or active_trade.get("entry", 0)
                if cur_price and cur_price > 0:
                    cur_size = _safe_float(active_trade.get("size"), 0.2)
                    ok_sync, sync_msg = sync_binance_open_position(active_trade["direction"], cur_size, cur_price)
                    if ok_sync:
                        sync_note = "\n📌 已在幣安同步開立現有倉位"
                    else:
                        sync_note = f"\n⚠️ 幣安同步開倉失敗: {sync_msg}"
            send_follow_button(is_update=False)
            return f"✅ 跟單模式已開啟！每次補倉/減倉都會推送同步提醒。{sync_note}"
        else:
            send_follow_button(is_update=True)
            return "⏹️ 跟單模式已關閉。"

    if text.startswith("/binance"):
        ready = _is_binance_copy_ready()
        mode = "已啟用" if BINANCE_REAL_COPY_ENABLED else "未啟用"
        key_ok = "有" if BINANCE_API_KEY else "無"
        secret_ok = "有" if BINANCE_API_SECRET else "無"
        if BINANCE_USE_ACCOUNT_THIRD:
            asset = _get_binance_total_asset_usdt(force=True)
            baseline = f"總資產1/3（約 {asset/3:.2f} USDT）" if asset > 0 else "總資產1/3（讀取中）"
        elif BINANCE_ORDER_ETH_QTY > 0:
            baseline = f"固定 {BINANCE_ORDER_ETH_QTY:.4f} ETH"
        else:
            baseline = f"固定 {BINANCE_ORDER_NOTIONAL_USDT:.2f} USDT"
        return (
            "🔗 Binance 實單跟單狀態\n"
            f"模式: {mode}\n"
            f"API Key: {key_ok}\n"
            f"API Secret: {secret_ok}\n"
            f"Symbol: {BINANCE_SYMBOL}\n"
            f"Leverage: {BINANCE_LEVERAGE}x\n"
            f"100%倉位基準: {baseline}\n"
            f"可下單: {'是' if ready else '否（請確認 BINANCE_REAL_COPY_ENABLED=1 與 API 金鑰）'}"
        )

    normalized = str(text or "").strip()
    if normalized in ("倉位面板", "📊 倉位面板", "開啟倉位面板", "📊 開啟倉位面板"):
        if active_trade.get("open"):
            refresh_position_panel_from_active_trade()
            return "📊 已更新倉位面板，請點底部「倉位面板」查看。"
        write_position_json()
        push_position_json()
        return "📋 目前無持倉，已同步最新狀態到面板。"

    return None

load_model()

# =============================
# API（簡化 + CACHE）
# =============================
def get_kline(interval, limit=100):
    now = time.time()
    cached_df = None

    if interval in KLINE_CACHE:
        data, ts = KLINE_CACHE[interval]
        cached_df = data
        if now - ts < KLINE_TTL.get(interval, 10):
            return data

    url = "https://fapi.binance.com/fapi/v1/klines"
    try:
        resp = HTTP_SESSION.get(
            url,
            params={
                "symbol": "ETHUSDT",
                "interval": interval,
                "limit": limit,
            },
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or not data:
            raise ValueError(f"invalid_kline_payload_{interval}")
    except Exception as e:
        if cached_df is not None:
            print(f"⚠️ 讀取 {interval} K 線失敗，改用快取: {e}")
            return cached_df
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
    global performance, BOT_SOFT_RESTART_REQUESTED, KLINE_CACHE, MACRO_CACHE, NEWS_CACHE

    load_model()  # 加載所有模型，包括新聞模型
    retrain_model()  # 啟動時重新訓練 AI 模型

    last_signal = None
    last_trade_time = 0
    TRADE_COOLDOWN = 300  # 冷卻加長（防洗單）
    MIN_PRICE_CHANGE = 0.002  # 至少0.2%價格變動才允許新單
    MIN_SIGNAL_DIFF = 0.05  # 信號差異門檻
    last_trade_signal = None  # 避免同一訊號重複開單
    losing_streak = 0
    MAX_LOSS_STREAK = 3
    last_entry_price = None
    last_direction = None
    # trade_open 移除，改用 active_trade 控制是否可開單

    # ===== 每日報告 =====
    last_report_time = 0

    last_update_id = load_last_update_id()

    # ===== V7 防洗單（訊號記憶）=====
    last_signal_cache = None

    while True:
        try:
            if BOT_SOFT_RESTART_REQUESTED:
                BOT_SOFT_RESTART_REQUESTED = False
                KLINE_CACHE.clear()
                MACRO_CACHE = {"sp": 0, "nq": 0, "btc": 0, "dxy": 0, "news": 0, "event": 0, "news_list": [], "ts": 0}
                NEWS_CACHE = {"news": 0, "event": 0, "news_list": [], "ts": 0}
                last_signal = None
                last_trade_time = 0
                last_trade_signal = None
                last_entry_price = None
                last_direction = None
                last_signal_cache = None
                load_model()
                print("♻️ 已完成軟重啟：模型重載、快取清空")

            # ===== Telegram 指令接收 =====
            if os.getenv("BOT_SUPERVISOR") == "1":
                updates = pop_pending_commands()
            else:
                params = {"timeout": 5}
                if last_update_id:
                    params["offset"] = last_update_id + 1

                try:
                    res = HTTP_SESSION.get(
                        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
                        params=params,
                        timeout=6
                    )
                    res.raise_for_status()
                    payload = res.json()
                    updates = payload.get("result", []) if isinstance(payload, dict) else []
                except Exception as e:
                    print(f"⚠️ 讀取 Telegram 更新失敗: {e}")
                    updates = []

            for u in updates:
                if os.getenv("BOT_SUPERVISOR") == "1":
                    text = u.get("text", "")
                    chat_id = u.get("chat_id")
                    last_update_id = u.get("update_id", last_update_id)
                    if last_update_id is not None:
                        save_last_update_id(last_update_id)
                else:
                    last_update_id = u.get("update_id")
                    if last_update_id is not None:
                        save_last_update_id(last_update_id)

                    # ===== 處理 inline 按鈕 callback_query（非 supervisor）=====
                    cq = u.get("callback_query")
                    if cq:
                        try:
                            _process_follow_callback(
                                cq.get("data", ""),
                                cq.get("id", ""),
                                cq.get("message", {}).get("message_id"),
                                cq.get("message", {}).get("chat", {}).get("id"),
                            )
                        except Exception:
                            pass
                        continue

                    msg_obj = u.get("message", {})
                    chat_id = msg_obj.get("chat", {}).get("id")

                    # ===== 處理 mini app web_app_data（手動平倉） =====
                    web_app_raw = msg_obj.get("web_app_data", {}).get("data", "")
                    if web_app_raw:
                        try:
                            payload = json.loads(str(web_app_raw)) if str(web_app_raw).strip().startswith("{") else {}
                            action = str(payload.get("action") or "").strip().lower()
                        except Exception:
                            action = ""

                        if action == "manual_close":
                            if not active_trade.get("open"):
                                requests.post(
                                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                                    json=_with_forced_remove_reply_keyboard(
                                        {"chat_id": chat_id, "text": "📋 目前無持倉，無需手動平倉。"},
                                        chat_id,
                                    ),
                                    timeout=5,
                                )
                                continue

                            close_px = _safe_float(WS_PRICE, 0.0)
                            if close_px <= 0:
                                close_px = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), 0.0)

                            ok_close, close_msg = sync_binance_close_position(close_px, reason="MANUAL")
                            if not ok_close:
                                requests.post(
                                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                                    json=_with_forced_remove_reply_keyboard(
                                        {"chat_id": chat_id, "text": f"⚠️ 手動平倉失敗：{close_msg}"},
                                        chat_id,
                                    ),
                                    timeout=5,
                                )
                                continue

                            _reset_active_trade_after_manual_close(close_px)
                            last_signal_cache = None
                            remove_position_keyboard()
                            send_telegram(
                                f"🧾 手動平倉完成（{active_trade.get('direction', '-') }）\n"
                                f"平倉價: {close_px:.2f}",
                                priority=True,
                            )
                            continue

                    text = msg_obj.get("text", "")

                try:
                    if not text:
                        continue

                    # ===== supervisor 模式下轉發的 callback 命令 =====
                    if text.startswith("__callback__:"):
                        raw_callback = text[len("__callback__:"):]
                        parts = raw_callback.rsplit(":", 2)
                        if len(parts) == 3:
                            cq_data, cq_id, cq_msg_id = parts
                            try:
                                _process_follow_callback(cq_data, cq_id, cq_msg_id, chat_id)
                            except Exception:
                                pass
                        continue

                    # 私聊鎖定：關閉一般對話，只保留面板與控制指令
                    private_chat_id = str(TELEGRAM_PRIVATE_CHAT_ID or "").strip()
                    text_norm = str(text or "").strip()
                    if TELEGRAM_PRIVATE_CHAT_LOCK and private_chat_id and str(chat_id) == private_chat_id:
                        allowed_exact = {"倉位面板", "📊 倉位面板", "開啟倉位面板", "📊 開啟倉位面板", "👥 跟單設定", "🧾 手動平倉"}
                        allowed_prefix = ("/follow", "/binance", "/restart", "/close")
                        if text_norm not in allowed_exact and (not any(text_norm.startswith(p) for p in allowed_prefix)):
                            continue

                    # 底部「跟單設定」按鈕 → 等同 /follow
                    if text_norm == "👥 跟單設定":
                        text = "/follow"
                        text_norm = "/follow"
                    elif text_norm == "🧾 手動平倉":
                        text = "/close"
                        text_norm = "/close"

                    # 文字指令手動平倉（/close）
                    if text_norm.startswith("/close"):
                        if not active_trade.get("open"):
                            reply = "📋 目前無持倉，無需手動平倉。"
                        else:
                            close_px = _safe_float(WS_PRICE, 0.0)
                            if close_px <= 0:
                                close_px = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), 0.0)
                            ok_close, close_msg = sync_binance_close_position(close_px, reason="MANUAL")
                            if not ok_close:
                                reply = f"⚠️ 手動平倉失敗：{close_msg}"
                            else:
                                _reset_active_trade_after_manual_close(close_px)
                                last_signal_cache = None
                                remove_position_keyboard()
                                reply = f"🧾 手動平倉完成\n平倉價: {close_px:.2f}"

                        payload = {"chat_id": chat_id, "text": reply}
                        if _is_private_chat(chat_id):
                            payload["reply_markup"] = _build_private_bottom_keyboard()
                        payload = _with_forced_remove_reply_keyboard(payload, chat_id)
                        requests.post(
                            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                            json=payload,
                            timeout=5,
                        )
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
                    }

                    reply = handle_ai_command(text, context, chat_id=chat_id)

                    if reply:
                        payload = {"chat_id": chat_id, "text": reply}
                        # 由底部按鈕觸發 /follow 或 /close 時，保留 mini app 鍵盤不被 remove_keyboard 清掉
                        if _is_private_chat(chat_id) and (text_norm.startswith("/follow") or text_norm.startswith("/close")):
                            payload["reply_markup"] = _build_private_bottom_keyboard()
                        payload = _with_forced_remove_reply_keyboard(payload, chat_id)
                        requests.post(
                            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                            json=payload,
                            timeout=5
                        )
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

            # ===== Macro（時事）=====
            sp_change, nq_change, btc_change, dxy_change, news_bias, event_risk, news_list = get_macro_bias()

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
                    send_telegram(snapshot_header, priority=True)
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
                        send_telegram(msg_news, priority=True)
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
                current = price
                candle_high = float(df_1m["high"].iloc[-1]) if len(df_1m) > 0 else current
                candle_low = float(df_1m["low"].iloc[-1]) if len(df_1m) > 0 else current

                # 同一張單持倉超過 4 小時，自動下修 TP
                maybe_decay_take_profit(current)

                if active_trade["direction"] == "long":
                    tp_hit = (current >= active_trade["tp"]) or (candle_high >= active_trade["tp"])
                    sl_hit = (current <= active_trade["sl"]) or (candle_low <= active_trade["sl"])

                    # 同根K同時觸發時採保守：先算SL，避免回測偏樂觀
                    if sl_hit:
                        # 已掛原生止損時，避免再重複送市價平倉。
                        if active_trade.get("binance_sl_order_id"):
                            _clear_native_stop_tracking()
                            active_trade["binance_qty"] = 0.0
                        else:
                            ok_close, close_msg = sync_binance_close_position(current, reason="SL")
                            if not ok_close:
                                print(f"⚠️ Binance 平倉失敗，維持持倉重試: {close_msg}")
                                continue
                        performance["loss"] += 1
                        performance["total"] += 1
                        _record_close_hit("SL", current, candle_high, candle_low)
                        active_trade["open"] = False
                        active_trade["size"] = 0.0
                        active_trade["add_count"] = 0
                        active_trade["reduce_count"] = 0
                        active_trade["open_ts"] = 0.0
                        active_trade["tp_decay_count"] = 0
                        remove_position_keyboard()
                        last_signal_cache = None
                        losing_streak += 1
                        print("❌ SL 命中")
                        send_telegram(
                            f"❌ SL 命中（{active_trade['direction']}）\n"
                            f"當前: {current:.2f} | 1m高低: {candle_high:.2f}/{candle_low:.2f}\n"
                            f"已關閉倉位，等待下一筆交易",
                            priority=True
                        )

                    elif tp_hit:
                        ok_close, close_msg = sync_binance_close_position(current, reason="TP")
                        if not ok_close:
                            print(f"⚠️ Binance 平倉失敗，維持持倉重試: {close_msg}")
                            continue
                        performance["win"] += 1
                        performance["total"] += 1
                        _record_close_hit("TP", current, candle_high, candle_low)
                        active_trade["open"] = False
                        active_trade["size"] = 0.0
                        active_trade["add_count"] = 0
                        active_trade["reduce_count"] = 0
                        active_trade["open_ts"] = 0.0
                        active_trade["tp_decay_count"] = 0
                        remove_position_keyboard()
                        last_signal_cache = None
                        losing_streak = 0
                        print("✅ TP 命中")
                        send_telegram(
                            f"✅ TP 命中（{active_trade['direction']}）\n"
                            f"當前: {current:.2f} | 1m高低: {candle_high:.2f}/{candle_low:.2f}\n"
                            f"已關閉倉位，等待下一筆交易",
                            priority=True
                        )

                elif active_trade["direction"] == "short":
                    tp_hit = (current <= active_trade["tp"]) or (candle_low <= active_trade["tp"])
                    sl_hit = (current >= active_trade["sl"]) or (candle_high >= active_trade["sl"])

                    # 同根K同時觸發時採保守：先算SL，避免回測偏樂觀
                    if sl_hit:
                        # 已掛原生止損時，避免再重複送市價平倉。
                        if active_trade.get("binance_sl_order_id"):
                            _clear_native_stop_tracking()
                            active_trade["binance_qty"] = 0.0
                        else:
                            ok_close, close_msg = sync_binance_close_position(current, reason="SL")
                            if not ok_close:
                                print(f"⚠️ Binance 平倉失敗，維持持倉重試: {close_msg}")
                                continue
                        performance["loss"] += 1
                        performance["total"] += 1
                        _record_close_hit("SL", current, candle_high, candle_low)
                        active_trade["open"] = False
                        active_trade["size"] = 0.0
                        active_trade["add_count"] = 0
                        active_trade["reduce_count"] = 0
                        active_trade["open_ts"] = 0.0
                        active_trade["tp_decay_count"] = 0
                        remove_position_keyboard()
                        last_signal_cache = None
                        losing_streak += 1
                        print("❌ SL 命中")
                        send_telegram(
                            f"❌ SL 命中（{active_trade['direction']}）\n"
                            f"當前: {current:.2f} | 1m高低: {candle_high:.2f}/{candle_low:.2f}\n"
                            f"已關閉倉位，等待下一筆交易",
                            priority=True
                        )

                    elif tp_hit:
                        ok_close, close_msg = sync_binance_close_position(current, reason="TP")
                        if not ok_close:
                            print(f"⚠️ Binance 平倉失敗，維持持倉重試: {close_msg}")
                            continue
                        performance["win"] += 1
                        performance["total"] += 1
                        _record_close_hit("TP", current, candle_high, candle_low)
                        active_trade["open"] = False
                        active_trade["size"] = 0.0
                        active_trade["add_count"] = 0
                        active_trade["reduce_count"] = 0
                        active_trade["open_ts"] = 0.0
                        active_trade["tp_decay_count"] = 0
                        remove_position_keyboard()
                        last_signal_cache = None
                        losing_streak = 0
                        print("✅ TP 命中")
                        send_telegram(
                            f"✅ TP 命中（{active_trade['direction']}）\n"
                            f"當前: {current:.2f} | 1m高低: {candle_high:.2f}/{candle_low:.2f}\n"
                            f"已關閉倉位，等待下一筆交易",
                            priority=True
                        )

            # 命中止盈止損前提下，持倉中允許補倉/減倉
            if active_trade["open"]:
                manage_position_scaling(current)

            # ===== 核心限制：未平倉禁止開新單，但新聞照常推 =====
            if active_trade["open"]:
                if not hasattr(run_bot, "last_position_status_ts"):
                    run_bot.last_position_status_ts = 0
                if not hasattr(run_bot, "last_news_monitor_ts"):
                    run_bot.last_news_monitor_ts = 0

                if time.time() - run_bot.last_position_status_ts > 15:
                    # 開啟跟單時，定期從 Binance 同步：進場均價 + 偵測原生止損是否已觸發
                    if _is_binance_copy_ready():
                        direction_now = active_trade.get("direction", "")
                        actual_ep = _fetch_binance_position_entry(direction_now)
                        if actual_ep > 0:
                            old_ep = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), 0.0)
                            if abs(actual_ep - old_ep) > 0.01:
                                active_trade["entry"] = actual_ep
                                active_trade["avg_entry"] = actual_ep
                                write_position_json()
                            push_position_json()
                        elif active_trade.get("binance_sl_order_id"):
                            # Binance 回傳持倉量為 0 → 原生止損單已觸發，Bot 同步清倉
                            print("⚠️ 偵測到 Binance 持倉已清空（原生止損觸發），Bot 同步平倉狀態")
                            current_for_sl = _safe_float(active_trade.get("sl"), price)
                            _clear_native_stop_tracking()
                            active_trade["binance_qty"] = 0.0
                            performance["loss"] += 1
                            performance["total"] += 1
                            _record_close_hit("SL", current_for_sl, current_for_sl, current_for_sl)
                            active_trade["open"] = False
                            active_trade["size"] = 0.0
                            active_trade["add_count"] = 0
                            active_trade["reduce_count"] = 0
                            active_trade["open_ts"] = 0.0
                            active_trade["tp_decay_count"] = 0
                            losing_streak += 1
                            last_signal_cache = None
                            remove_position_keyboard()  # 內部已呼叫 write/push_position_json
                            send_telegram(
                                f"❌ SL 命中（Binance 原生止損觸發，Bot 同步）\n"
                                f"已關閉倉位，等待下一筆交易",
                                priority=True
                            )
                            run_bot.last_position_status_ts = time.time()
                            continue
                    monitor_entry = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), 0.0)
                    print(
                        f"📡 持倉監控 | 方向: {active_trade['direction']} | 倉位: {int(_safe_float(active_trade.get('size'), 0)*100)}% | 現價: {price:.2f} | 進場均價: {monitor_entry:.2f} | TP: {active_trade['tp']:.2f} | SL: {active_trade['sl']:.2f}"
                    )
                    run_bot.last_position_status_ts = time.time()

                if time.time() - run_bot.last_news_monitor_ts > 30:
                    latest_news_preview = " | ".join(news_list[:4]) if news_list else "目前無新快訊（RSS 暫無資料）"
                    print(f"📰 新聞監控中 | {latest_news_preview}")
                    run_bot.last_news_monitor_ts = time.time()

                # 持倉心跳：每 5 分鐘定期推送 position.json，確保 mini app 抓到最新 ts
                if not hasattr(run_bot, "last_position_heartbeat_ts"):
                    run_bot.last_position_heartbeat_ts = 0
                if time.time() - run_bot.last_position_heartbeat_ts > 300:
                    write_position_json()
                    push_position_json()
                    run_bot.last_position_heartbeat_ts = time.time()

                time.sleep(0.8)
                continue

            if price > recent_high:
                breakout = 1
            elif price < recent_low:
                breakout = -1

            # ===== Liquidity Sweep（掃流動性 v7）=====
            sweep_high = False
            sweep_low = False

            prev_high = df_5m["high"].iloc[-2]
            prev_low = df_5m["low"].iloc[-2]

            # 掃上方流動性（假突破上影）
            if price > recent_high and df_5m["close"].iloc[-1] < prev_high:
                sweep_high = True

            # 掃下方流動性（假跌破下影）
            if price < recent_low and df_5m["close"].iloc[-1] > prev_low:
                sweep_low = True

            macro_bias = 0

            # BTC（最高權重）
            if btc_change > 0.002:
                macro_bias += 1.5
            elif btc_change < -0.002:
                macro_bias -= 1.5

            # ===== Macro v2（權重模型）=====
            # NASDAQ（權重最高）
            if nq_change > 0.0015:
                macro_bias += 1.2
            elif nq_change < -0.0015:
                macro_bias -= 1.2

            # SP500（輔助）
            if sp_change > 0.0015:
                macro_bias += 0.6
            elif sp_change < -0.0015:
                macro_bias -= 0.6

            # DXY（反向）
            if dxy_change > 0.0015:
                macro_bias -= 1
            elif dxy_change < -0.0015:
                macro_bias += 1

            # NEWS（新增）
            macro_bias += news_bias * 0.8
            # ===== 事件風險（波動放大器）=====
            if event_risk >= 1:
                macro_bias *= 1.2
            if event_risk >= 2:
                macro_bias *= 1.5

            # ===== Feature（升級版）=====
            recent_high = df_15m["high"].tail(20).max()
            recent_low = df_15m["low"].tail(20).min()
            triangle = detect_triangle(df_15m)

            # ===== Volume v2（量價分析）=====
            vol_now = df_15m["volume"].iloc[-1]
            vol_ma = df_15m["vol_ma20"].iloc[-1] if "vol_ma20" in df_15m.columns else df_15m["volume"].rolling(20).mean().iloc[-1]

            volume_spike = vol_now > vol_ma * 1.5
            volume_ratio = vol_now / (vol_ma + 1e-9)

            # 買賣壓（K線方向近似）
            buy_pressure = df_15m["close"].iloc[-1] > df_15m["open"].iloc[-1]
            sell_pressure = df_15m["close"].iloc[-1] < df_15m["open"].iloc[-1]

            # 吸籌 / 出貨（簡化：放量但不延續）
            prev_close = df_15m["close"].iloc[-2]
            absorption = False
            if volume_spike:
                if buy_pressure and price < prev_close:
                    absorption = True  # 出貨
                if sell_pressure and price > prev_close:
                    absorption = True  # 吸籌

            features = {
                "htf": htf,
                "htf_strength": htf_strength,
                "mid_trend": mid_trend,
                "macd": df_15m["macd"].iloc[-1],
                "hist": df_15m["macd"].iloc[-1] - df_15m["signal"].iloc[-1],
                "price_vs_ma": df_15m["close"].iloc[-1] - df_15m["ma25"].iloc[-1],
                "breakout": breakout,
                "fvg": (fvg_high - fvg_low) if fvg_low else 0,

                # 新增核心特徵
                "volatility": df_15m["high"].iloc[-1] - df_15m["low"].iloc[-1],
                "trend_strength": abs(df_15m["ma25"].iloc[-1] - df_15m["ma25"].iloc[-5]),
                "range_pos": (price - recent_low) / (recent_high - recent_low + 1e-6),

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
                    "bear_trend_strong": -2
                }[regime],
                "triangle": triangle,
                "event_risk": event_risk,
                "volume_spike": int(volume_spike),
                "volume_ratio": volume_ratio,
                "buy_pressure": int(buy_pressure),
                "absorption": int(absorption),
                "sweep_high": int(sweep_high),
                "sweep_low": int(sweep_low),
            }

            # ===== AI決策（強化版：避免卡0.5）=====
            ai_prob = 0.5

            try:
                X = pd.DataFrame([features])

                # Online model（主模型）
                if online_initialized:
                    ai_prob = online_model.predict_proba(X)[0][1]

                # 備用模型
                elif model:
                    ai_prob = model.predict_proba(X)[0][1]

            except Exception:
                ai_prob = 0.5

            # ===== fallback 強化（關鍵升級）=====
            # 當AI信心太低 → 用規則模型補強
            if abs(ai_prob - 0.5) < 0.05:

                rule_score = 0

                # 趨勢
                if htf == 1:
                    rule_score += 0.25
                else:
                    rule_score -= 0.25

                # 動能
                if mid_trend == 1:
                    rule_score += 0.15
                else:
                    rule_score -= 0.15

                # breakout
                if breakout == 1:
                    rule_score += 0.25
                elif breakout == -1:
                    rule_score -= 0.25

                # macro
                rule_score += macro_bias * 0.1

                # triangle
                if triangle == 1:
                    rule_score += 0.05

                ai_prob = 0.5 + rule_score

            # clamp
            ai_prob = max(0.05, min(ai_prob, 0.95))

            # ===== AI主導進場（v4 最終盈利版）=====

            # ===== 動態平滑（避免卡0.5）=====
            if last_signal is not None:
                score = 0.75 * ai_prob + 0.25 * last_signal
            else:
                score = ai_prob

            # ===== 自適應噪音（低信心才探索）=====
            if abs(score - 0.5) < 0.08:
                noise = np.random.uniform(-0.05, 0.05)
            else:
                noise = np.random.uniform(-0.02, 0.02)
            score = max(0.05, min(score + noise, 0.95))

            # ===== 小週期主導（Dual Flow v2）=====
            confluence = 0

            # 小週期優先（動能）
            if mid_trend == 1:
                confluence += 1
            elif mid_trend == -1:
                confluence += 1

            # breakout 直接加權（不再綁定HTF）
            if breakout != 0:
                confluence += 1

            # 大週期只做「方向加權」（不再卡死）
            if htf == mid_trend:
                confluence += 1
            else:
                # 不一致 → 視為回調（仍給權重）
                confluence += 0.5

            score += confluence * 0.08

            # ===== Volume bias =====
            if volume_spike:
                if buy_pressure:
                    score += 0.06
                elif sell_pressure:
                    score -= 0.06

            # 吸籌/出貨 → 反轉警告
            if absorption:
                score *= 0.9

            # ===== Sweep 反轉加權 =====
            if sweep_high:
                score -= 0.12
            if sweep_low:
                score += 0.12

            # ===== Regime強化（趨勢才重倉）=====
            if regime == "bull_trend_strong":
                score += 0.25
            elif regime == "bull_trend":
                score += 0.12
            elif regime == "bear_trend_strong":
                score -= 0.25
            elif regime == "bear_trend":
                score -= 0.12

            # ===== Range市場幾乎不做 =====
            if regime == "range":
                score = 0.5 + (score - 0.5) * 0.4

            # ===== Macro強化（避免逆勢）=====
            if macro_bias == 1 and score > 0.5:
                score += 0.08
            elif macro_bias == -1 and score < 0.5:
                score -= 0.08
            else:
                score *= 0.95

            # ===== 新聞影響強化（增加時事判斷權重）=====
            if news_text:
                analysis = analyze_news_text(news_text)
                news_bias = analysis["bias"]  # -2 到 2
                news_score_adjust = news_bias * 0.08  # 將新聞 bias 轉換為 score 調整（-0.16 到 0.16）
                score += news_score_adjust
                score = max(0.05, min(score, 0.95))  # 確保在範圍內

            # ===== 最終決策（含進場 / TP / SL / 倉位）=====
            entry = price
            atr = df_15m["high"].iloc[-1] - df_15m["low"].iloc[-1]

            # 預設
            final = "觀望"
            sl = None
            tp = None
            position_size = 0

            # ===== 提前進場機制（升級）=====
            early_entry = False
            if abs(score - 0.5) > 0.18 and htf == mid_trend:
                early_entry = True

            # ===== 回調模式（Pullback Trading）=====
            pullback_long = False
            pullback_short = False

            # 強多 → 回調做空
            if regime in ["bull_trend", "bull_trend_strong"] and mid_trend == -1:
                if volume_spike or breakout == -1:
                    pullback_short = True

            # 強空 → 回調做多
            if regime in ["bear_trend", "bear_trend_strong"] and mid_trend == 1:
                if volume_spike or breakout == 1:
                    pullback_long = True

            # ===== 假突破過濾（量價）=====
            fake_breakout = False
            if breakout != 0 and not volume_spike:
                fake_breakout = True
            if absorption or sweep_high or sweep_low:
                fake_breakout = True

            # ===== 相對性過濾（BTC）=====
            if breakout == 1 and btc_change < 0:
                fake_breakout = True
            if breakout == -1 and btc_change > 0:
                fake_breakout = True

            # 放寬條件，解決高分卻觀望問題
            if regime != "range" and abs(score - 0.5) > (0.05 - event_risk*0.03):

                # ===== 低信心過濾（避免亂單） =====
                if abs(score - 0.5) < 0.12:
                    final = "觀望（低信心）"

                # ===== 三角模式 v2（提前進場 + 突破加碼） =====
                if triangle == 1:

                    upper = df_15m["high"].tail(20).max()
                    lower = df_15m["low"].tail(20).min()
                    range_size = upper - lower

                    # 靠近上緣 → 做空
                    if price > upper - range_size * 0.2 and breakout == 0:
                        final = "🔺 三角上緣做空"

                        sl = upper
                        risk = sl - entry
                        tp = entry - risk * 1.8

                    # 靠近下緣 → 做多
                    elif price < lower + range_size * 0.2 and breakout == 0:
                        final = "🔻 三角下緣做多"

                        sl = lower
                        risk = entry - sl
                        tp = entry + risk * 1.8

                    # 突破 → 強訊號（加碼）
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

                    # 對稱多空門檻：調整 AI 信號邏輯，提升做空觸發機率
                    if score > 0.52:
                        final = "🚀 做多"

                        recent_low_15 = df_15m["low"].tail(10).min()
                        sl = recent_low_15

                        risk = entry - sl
                        rr = 2.2 if regime.endswith("strong") else 1.6
                        tp = entry + risk * rr

                    elif score < 0.48:
                        final = "🚀 做空"

                        recent_high_15 = df_15m["high"].tail(10).max()
                        sl = recent_high_15

                        risk = sl - entry
                        rr = 2.2 if regime.endswith("strong") else 1.6
                        tp = entry - risk * rr

                    # 反彈單：把回調旗標接入實際開單（偏保守門檻）
                    elif pullback_long and score >= 0.45:
                        final = "↩️ 反彈做多"
                        recent_low_pb = min(df_5m["low"].tail(6).min(), df_15m["low"].tail(6).min())
                        sl = recent_low_pb
                        risk = max(entry - sl, atr * 0.45, entry * 0.001)
                        tp = entry + risk * 1.5

                    elif pullback_short and score <= 0.55:
                        final = "↩️ 反彈做空"
                        recent_high_pb = max(df_5m["high"].tail(6).max(), df_15m["high"].tail(6).max())
                        sl = recent_high_pb
                        risk = max(sl - entry, atr * 0.45, entry * 0.001)
                        tp = entry - risk * 1.5

                    # ===== 倉位（動態） =====
                    confidence = abs(score - 0.5) * 2

                    if regime in ["bull_trend_strong", "bear_trend_strong"]:
                        base = 0.5
                    elif regime in ["bull_trend", "bear_trend"]:
                        base = 0.35
                    else:
                        base = 0.2

                    if confidence > 0.7:
                        position_size = base
                    elif confidence > 0.5:
                        position_size = base * 0.7
                    elif confidence > 0.3:
                        position_size = base * 0.5
                    else:
                        position_size = base * 0.3

                    if losing_streak >= MAX_LOSS_STREAK:
                        position_size *= 0.5

                    # 三角策略調整（提前單較小，突破加碼）
                    if "三角" in final:
                        if "突破" in final:
                            position_size *= 1.2
                        else:
                            position_size *= 0.7

            # ===== 修正長短單 TP/SL（方向 + 最小風險距離） =====
            if final != "觀望":
                final, sl, tp = auto_fix_trade_plan(final, entry, sl, tp, atr)

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
            
            # 提取訊息中的進場/止盈/止損價格（確保與網址一致）
            entry_display_str = f"{entry:.2f}" if final != "觀望" else None
            tp_display_str = f"{tp:.2f}" if (final != "觀望" and tp is not None) else "0.0"
            sl_display_str = f"{sl:.2f}" if (final != "觀望" and sl is not None) else "0.0"

            msg += (
                f"🤖 AI信號：{display_signal}\n"
                f"📊 信心值: {ai_prob:.2f}\n"
                f"📈 勝率: {(performance['win']/performance['total'] if performance['total']>0 else 0):.2%}\n"
                f"🌍 市場狀態: {regime_text}\n"
                f"📰 時事判斷: {macro_text}\n"
                f"{news_text}"
                f"🧠 判斷依據: {reason_text}"
            )
            # Fix spam log（觀望不要一直print）
            if final != "觀望":
                print(msg)

            # ===== 強制進場（解決AI高信心但被過濾掉） =====
            if final == "觀望":
                if score > 0.7:
                    final = "🚀 做多（強制）"
                    recent_low_15 = df_15m["low"].tail(10).min()
                    sl = recent_low_15
                    risk = entry - sl
                    tp = entry + risk * 2

                elif score < 0.3:
                    final = "🚀 做空（強制）"
                    recent_high_15 = df_15m["high"].tail(10).max()
                    sl = recent_high_15
                    risk = sl - entry
                    tp = entry - risk * 2

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

            # ===== 最終過濾 =====
            if final.startswith("觀望"):
                continue

            # ===== 最終安全檢查：拒絕假突破低信心單 =====
            if fake_breakout and abs(score - 0.5) < 0.22:
                continue

            if final != "觀望":

                # 保險：再次確認沒有持倉
                if active_trade["open"]:
                    continue

                # 防止同一訊號重複刷
                if last_signal_cache == msg:
                    continue

                print("📤 發送 Telegram")
                send_telegram(msg, priority=True)
                # 若設定私聊，同時發送開倉通知到私聊
                if TELEGRAM_PRIVATE_CHAT_ID:
                    try:
                        private_payload = _with_forced_remove_reply_keyboard(
                            {
                                "chat_id": TELEGRAM_PRIVATE_CHAT_ID,
                                "text": f"🔔 開倉通知 (私聊)\n\n{msg}",
                            },
                            TELEGRAM_PRIVATE_CHAT_ID,
                        )
                        requests.post(
                            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                            json=private_payload,
                            timeout=5,
                        )
                    except Exception as e:
                        print(f"⚠️ 私聊開倉通知發送失敗: {e}")
                last_signal_cache = msg
                last_trade_time = now_ts
                last_trade_signal = final
                last_entry_price = price
                last_direction = final

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
                open_size_ratio = float(min(1.0, max(base_size, 0.1)))
                ok_open, open_msg = sync_binance_open_position(direction, open_size_ratio, price)
                if not ok_open:
                    send_telegram(f"⚠️ Binance 開倉失敗，已取消本次交易: {open_msg}", priority=True)
                    continue
                
                # 若開倉成功，active_trade["entry"] 已被更新為實際成交價
                # 需要同步更新顯示文本，確保通知和 mini app 用相同的進場價

                active_trade["size"] = open_size_ratio
                active_trade["max_size"] = 1.0
                active_trade["min_size"] = max(0.1, active_trade["size"] * 0.3)
                active_trade["add_count"] = 0
                active_trade["reduce_count"] = 0
                active_trade["last_adjust_ts"] = 0.0
                active_trade["open_ts"] = time.time()
                active_trade["tp_decay_count"] = 0
                active_trade["last_close_reason"] = ""
                active_trade["last_close_price"] = 0.0
                active_trade["last_close_ts"] = 0
                active_trade["last_close_candle_high"] = 0.0
                active_trade["last_close_candle_low"] = 0.0
                active_trade["open"] = True

                # 止損改為 Binance 原生 STOP_MARKET；止盈維持 Bot 監控。
                ok_native_sl, native_sl_msg = sync_binance_set_native_stop_loss(direction, sl)
                if not ok_native_sl:
                    send_telegram(f"⚠️ 原生止損單掛單失敗，暫回退 Bot 監控止損: {native_sl_msg}", priority=True)

                # 用實際成交價（已更新的 entry）重新生成顯示字符串
                actual_entry = active_trade["entry"]
                actual_entry_display = f"{actual_entry:.2f}"
                send_position_keyboard(
                    direction,
                    actual_entry,
                    tp,
                    sl,
                    active_trade["size"],
                    entry_display=actual_entry_display,
                    tp_display=tp_display_str,
                    sl_display=sl_display_str,
                )
                send_follow_button(is_update=False)
                write_position_json()
                push_position_json()

            # ===== 記錄（未來價格）=====
            future_price = price
            time.sleep(1.2)

            new_price = WS_PRICE if WS_PRICE else price

            # （原簡易績效追蹤區塊已移除，現由真實交易管理統計）

            # ===== 更乾淨學習（避免噪音）=====
            # AI 標籤（根據實際進場方向與結果）
            if last_direction and last_entry_price:
                if "做多" in last_direction:
                    label = 1 if new_price > last_entry_price else 0
                elif "做空" in last_direction:
                    label = 1 if new_price < last_entry_price else 0
                else:
                    continue
            else:
                continue

            log_data(features, label)

            # ===== 即時學習（核心升級）=====
            update_online_model(features, label)

            # 每60秒嘗試訓練一次（更穩定）
            if int(time.time()) % 60 == 0:
                train_model()

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
