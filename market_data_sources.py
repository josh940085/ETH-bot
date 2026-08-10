"""Shared multi-exchange K-line (candlestick) fetchers.

Extracted from eth.py's live multi-exchange kline waterfall (Kraken ->
Coinbase -> Twelve Data -> TradingView scrape) so the live trading path and
backtest/historical data tooling read market data the same way instead of
diverging (live used this waterfall, backtests only read Binance archives).

Binance fetching is intentionally NOT included here: eth.py's Binance kline
fetch shares a rate-limit bucket with its order/account REST calls
(`_binance_request_get` / `BINANCE_RATE_LIMIT_STATE`), and backtests already
have a dedicated Binance archive/REST fetcher in market_history.py. Callers
that need a Binance leg supply their own fetcher.

Every non-Binance source here applies a fixed minimum request gap
(`*_REQUEST_MIN_GAP_SEC`) before calling out, not just a reactive backoff
after a 429/418 - this is deliberate so historical backfills (which can
issue hundreds of paginated requests) do not trip exchange rate limits.
"""

import datetime
import hashlib
import json
import os
import threading
import time

import requests

from runtime_paths import data_path, ensure_parent_dir

DEFAULT_PAIR = str(os.getenv("TRADE_SYMBOL", "BTCUSDT") or "BTCUSDT").strip().upper()

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
KLINE_TTL = {
    "1M": 60 * 60 * 12,
    "1w": 60 * 60 * 6,
    "1d": 60 * 60,
    "12h": 60 * 30,
    "4h": 60 * 60,
    "1h": 60 * 10,
    "30m": 60 * 5,
    "15m": 60 * 3,
    "5m": 60 * 2,
    "1m": 10,
}


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


def log_kline_source_failure(source_name, exc, prefix="K線"):
    now_ts = time.time()
    state = getattr(log_kline_source_failure, "_state", {})
    key = f"{prefix}:{source_name}:{type(exc).__name__}:{str(exc)[:80]}"
    last_ts = _safe_float(state.get(key), 0.0) if isinstance(state, dict) else 0.0
    if now_ts - last_ts >= 300:
        print(f"⚠️ {prefix}來源失敗，改試下一個: {source_name} | {exc}")
        if not isinstance(state, dict):
            state = {}
        state[key] = now_ts
        setattr(log_kline_source_failure, "_state", state)


def log_kline_fallback(source_name, prefix="K線"):
    now_ts = time.time()
    state = getattr(log_kline_fallback, "_state", {})
    key = f"{prefix}:{source_name}"
    last_ts = _safe_float(state.get(key), 0.0) if isinstance(state, dict) else 0.0
    if now_ts - last_ts >= 300:
        print(f"♻️ Futures {prefix}不可用，改用 {source_name}")
        if not isinstance(state, dict):
            state = {}
        state[key] = now_ts
        setattr(log_kline_fallback, "_state", state)


# ===== TradingView (WebSocket scrape) =====

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
TRADINGVIEW_FAILURE_COOLDOWN = {}


def tradingview_cooldown_key(symbol, interval):
    return f"{str(symbol or DEFAULT_PAIR).upper()}:{interval}"


def is_tradingview_in_cooldown(symbol, interval):
    key = tradingview_cooldown_key(symbol, interval)
    until = _safe_float(TRADINGVIEW_FAILURE_COOLDOWN.get(key), 0.0)
    return time.time() < until


def mark_tradingview_failure(symbol, interval):
    cooldown = max(15.0, _safe_float(os.getenv("TRADINGVIEW_FAILURE_COOLDOWN_SEC", 90), 90))
    TRADINGVIEW_FAILURE_COOLDOWN[tradingview_cooldown_key(symbol, interval)] = time.time() + cooldown


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
    symbol = str(symbol or DEFAULT_PAIR).upper().strip()
    if symbol in TRADINGVIEW_SYMBOL_MAP:
        return TRADINGVIEW_SYMBOL_MAP[symbol]
    if symbol.endswith("USDT"):
        return f"BINANCE:{symbol}.P"
    return f"BINANCE:{symbol}"


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


def fetch_tradingview_klines(symbol, interval, limit=100, start_time_ms=None, end_time_ms=None, timeout=10):
    import websocket  # local import: only TradingView needs this dependency

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


# ===== Kraken =====

KRAKEN_INTERVAL_MAP = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60,
    "4h": 240, "1d": 1440, "1w": 10080, "1M": 21600,
}
KRAKEN_KLINE_CACHE = {}
KRAKEN_REQUEST_LOCK = threading.Lock()
_KRAKEN_LAST_REQUEST_TS = {"ts": 0.0}


def fetch_kraken_klines(symbol, interval, limit=100, start_time_ms=None, end_time_ms=None, timeout=10):
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
        wait_sec = min_gap - (time.time() - _KRAKEN_LAST_REQUEST_TS["ts"])
        if wait_sec > 0:
            time.sleep(wait_sec)
        response = requests.get("https://api.kraken.com/0/public/OHLC", params=params, timeout=timeout)
        _KRAKEN_LAST_REQUEST_TS["ts"] = time.time()
        first_payload = response.json() if response.ok else {}
        if any("Too many requests" in str(item) for item in first_payload.get("error", [])):
            time.sleep(max(5.0, min_gap * 2.0))
            response = requests.get("https://api.kraken.com/0/public/OHLC", params=params, timeout=timeout)
            _KRAKEN_LAST_REQUEST_TS["ts"] = time.time()
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


# ===== Coinbase =====

COINBASE_GRANULARITY_MAP = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "1d": 86400}
COINBASE_REQUEST_LOCK = threading.Lock()
_COINBASE_LAST_REQUEST_TS = {"ts": 0.0}


def fetch_coinbase_klines(symbol, interval, limit=100, start_time_ms=None, end_time_ms=None, timeout=10):
    granularity = COINBASE_GRANULARITY_MAP.get(str(interval))
    if granularity is None:
        raise RuntimeError(f"Coinbase不支援週期 {interval}")
    product = "BTC-USD" if str(symbol or "").upper().startswith("BTC") else "ETH-USD"
    params = {"granularity": granularity}
    if start_time_ms is not None:
        params["start"] = datetime.datetime.fromtimestamp(
            _safe_float(start_time_ms, 0.0) / 1000, tz=datetime.timezone.utc
        ).isoformat()
    if end_time_ms is not None:
        params["end"] = datetime.datetime.fromtimestamp(
            _safe_float(end_time_ms, 0.0) / 1000, tz=datetime.timezone.utc
        ).isoformat()
    with COINBASE_REQUEST_LOCK:
        min_gap = max(0.25, _safe_float(os.getenv("COINBASE_REQUEST_MIN_GAP_SEC", 0.5), 0.5))
        wait_sec = min_gap - (time.time() - _COINBASE_LAST_REQUEST_TS["ts"])
        if wait_sec > 0:
            time.sleep(wait_sec)
        response = requests.get(
            f"https://api.exchange.coinbase.com/products/{product}/candles",
            params=params,
            headers={"User-Agent": "ETH-bot/1.0"},
            timeout=timeout,
        )
        _COINBASE_LAST_REQUEST_TS["ts"] = time.time()
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
    if start_time_ms is not None or end_time_ms is not None:
        return rows[: max(1, min(300, _safe_int(limit, len(rows))))]
    return rows[-max(1, min(300, _safe_int(limit, 100))) :]


# ===== Twelve Data =====

TWELVE_DATA_INTERVAL_MAP = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "4h": "4h", "12h": "12h", "1d": "1day",
    "1w": "1week", "1M": "1month",
}
TWELVE_DATA_KLINE_CACHE = {}
TWELVE_DATA_REQUEST_LOCK = threading.Lock()
TWELVE_DATA_USAGE_STATE = {"day": "", "count": 0, "last_request_ts": 0.0}
TWELVE_DATA_USAGE_PATH = data_path("api_token_usage.json")


def load_twelve_data_usage_state():
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


def save_twelve_data_usage_state():
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


load_twelve_data_usage_state()


def fetch_twelve_data_klines(symbol, interval, limit=100, start_time_ms=None, end_time_ms=None, timeout=10):
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
    date_format = "%Y-%m-%d" if str(interval) in {"1d", "1w", "1M"} else "%Y-%m-%d %H:%M:%S"
    if start_time_ms is not None:
        params["start_date"] = datetime.datetime.fromtimestamp(
            _safe_float(start_time_ms, 0.0) / 1000, tz=datetime.timezone.utc
        ).strftime(date_format)
    if end_time_ms is not None:
        params["end_date"] = datetime.datetime.fromtimestamp(
            _safe_float(end_time_ms, 0.0) / 1000, tz=datetime.timezone.utc
        ).strftime(date_format)

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
        save_twelve_data_usage_state()
    payload = response.json()
    if response.status_code >= 400:
        raise RuntimeError(
            f"Twelve Data HTTP {response.status_code}: "
            f"{(payload or {}).get('message', 'request failed') if isinstance(payload, dict) else 'request failed'}"
        )
    if not isinstance(payload, dict) or payload.get("status") == "error":
        raise RuntimeError(f"Twelve Data error: {(payload or {}).get('message', 'invalid response')}")
    values = payload.get("values") if isinstance(payload.get("values"), list) else []
    interval_ms = KLINE_INTERVAL_MS.get(str(interval), 60_000)
    rows = []
    for item in values:
        if not isinstance(item, dict):
            continue
        try:
            raw_datetime = str(item.get("datetime") or "").strip()
            try:
                stamp = datetime.datetime.strptime(raw_datetime, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                stamp = datetime.datetime.strptime(raw_datetime, "%Y-%m-%d")
            stamp = stamp.replace(tzinfo=datetime.timezone.utc)
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
