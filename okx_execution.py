"""Small, fail-closed OKX V5 execution adapter for the live bot.

The strategy continues to express position size in base asset (BTC).  OKX
SWAP orders use contracts, so this module always obtains ``ctVal`` and lot
size from the instrument endpoint before submitting an order.
"""

import base64
import datetime as dt
import hashlib
import hmac
import json
import math
import os
from urllib.parse import urlencode

import requests
from runtime_config import load_local_env

load_local_env(overwrite=False, names=(".env",))


BASE_URL = "https://www.okx.com"
DEFAULT_INST_ID = "BTC-USDT-SWAP"
CLIENT_PREFIX = "ethbot"


class OKXAPIError(RuntimeError):
    pass


def configured():
    return all(str(os.getenv(name, "") or "").strip() for name in (
        "OKX_API_KEY", "OKX_API_SECRET", "OKX_API_PASSPHRASE",
    ))


def inst_id():
    return str(os.getenv("OKX_INST_ID", DEFAULT_INST_ID) or DEFAULT_INST_ID).strip().upper()


def td_mode():
    return str(os.getenv("OKX_TD_MODE", "cross") or "cross").strip().lower()


def _timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _request(method, path, *, params=None, body=None, timeout=10):
    if not configured():
        raise OKXAPIError("OKX API 金鑰未完整設定")
    method = method.upper()
    query = urlencode({k: v for k, v in (params or {}).items() if v is not None})
    request_path = path + (f"?{query}" if query else "")
    body_text = json.dumps(body, separators=(",", ":"), ensure_ascii=False) if body is not None else ""
    timestamp = _timestamp()
    secret = str(os.getenv("OKX_API_SECRET") or "").encode()
    prehash = f"{timestamp}{method}{request_path}{body_text}".encode()
    sign = base64.b64encode(hmac.new(secret, prehash, hashlib.sha256).digest()).decode()
    headers = {
        "OK-ACCESS-KEY": str(os.getenv("OKX_API_KEY") or "").strip(),
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": str(os.getenv("OKX_API_PASSPHRASE") or "").strip(),
        "Content-Type": "application/json",
    }
    if str(os.getenv("OKX_DEMO_TRADING", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}:
        headers["x-simulated-trading"] = "1"
    response = requests.request(method, BASE_URL + request_path, data=body_text or None, headers=headers, timeout=timeout)
    try:
        payload = response.json()
    except Exception as exc:
        raise OKXAPIError(f"OKX HTTP {response.status_code}: 非 JSON 回應") from exc
    if response.status_code >= 400 or str(payload.get("code", "")) != "0":
        raise OKXAPIError(f"OKX API: HTTP {response.status_code} code={payload.get('code')} msg={payload.get('msg')}")
    for row in payload.get("data") or []:
        if isinstance(row, dict) and str(row.get("sCode", "0")) != "0":
            raise OKXAPIError(f"OKX 訂單遭拒 code={row.get('sCode')} msg={row.get('sMsg')}")
    return payload.get("data") or []


def _number(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def instrument():
    rows = _request("GET", "/api/v5/public/instruments", params={"instType": "SWAP", "instId": inst_id()})
    if not rows:
        raise OKXAPIError(f"找不到 OKX 合約 {inst_id()}")
    row = rows[0]
    ct_val = _number(row.get("ctVal"))
    lot_sz = _number(row.get("lotSz"))
    min_sz = _number(row.get("minSz"))
    if str(row.get("ctValCcy") or "").upper() != "BTC" or ct_val <= 0 or lot_sz <= 0 or min_sz <= 0:
        raise OKXAPIError("OKX 合約規格不支援或不安全")
    return {"ct_val": ct_val, "lot_sz": lot_sz, "min_sz": min_sz}


def contracts_for_base(base_qty, *, enforce_min=True):
    spec = instrument()
    raw = max(0.0, _number(base_qty)) / spec["ct_val"]
    contracts = math.floor(raw / spec["lot_sz"]) * spec["lot_sz"]
    if enforce_min:
        contracts = max(spec["min_sz"], contracts)
    return round(contracts, 8), spec


def base_for_contracts(contracts, spec=None):
    spec = spec or instrument()
    return max(0.0, _number(contracts)) * spec["ct_val"]


def account_snapshot():
    rows = _request("GET", "/api/v5/account/balance")
    details = (rows[0].get("details") or []) if rows else []
    usdt = next((row for row in details if str(row.get("ccy") or "").upper() == "USDT"), {})
    total_eq = _number((rows[0] if rows else {}).get("totalEq"))
    available = _number(usdt.get("availEq"), _number(usdt.get("availBal")))
    equity = _number(usdt.get("eq"), total_eq)
    return {"available_balance": max(0.0, available), "wallet_balance": max(0.0, equity), "margin_balance": max(0.0, equity), "unrealized_profit": 0.0}


def account_config():
    rows = _request("GET", "/api/v5/account/config")
    return rows[0] if rows else {}


def positions():
    return _request("GET", "/api/v5/account/positions", params={"instId": inst_id()})


def active_position():
    rows = [row for row in positions() if abs(_number(row.get("pos"))) > 1e-12]
    if not rows:
        return None
    # net mode reports a signed pos. Long/short mode carries posSide.
    return max(rows, key=lambda row: abs(_number(row.get("pos"))))


def position_direction(row):
    side = str((row or {}).get("posSide") or "").lower()
    pos = _number((row or {}).get("pos"))
    if side in {"long", "short"}:
        return side
    return "long" if pos > 0 else "short"


def position_mode():
    return str(account_config().get("posMode") or "net_mode").lower()


def set_leverage(leverage, direction=None):
    body = {"instId": inst_id(), "lever": str(int(leverage)), "mgnMode": td_mode()}
    if position_mode() == "long_short_mode":
        if direction not in {"long", "short"}:
            raise OKXAPIError("OKX long/short 模式缺少持倉方向")
        body["posSide"] = direction
    rows = _request("POST", "/api/v5/account/set-leverage", body=body)
    return max(1, int(_number((rows[0] if rows else {}).get("lever"), leverage)))


def _order_side(direction, reduce=False):
    if reduce:
        return "sell" if direction == "long" else "buy"
    return "buy" if direction == "long" else "sell"


def _client_id(tag):
    import time
    return f"{CLIENT_PREFIX}{tag}{time.time_ns():x}"[:32]


def place_market(direction, base_qty, *, reduce=False, enforce_min=False):
    contracts, spec = contracts_for_base(base_qty, enforce_min=enforce_min)
    if contracts <= 0:
        raise OKXAPIError("OKX 下單量低於合約最小張數")
    body = {"instId": inst_id(), "tdMode": td_mode(), "side": _order_side(direction, reduce), "ordType": "market", "sz": str(contracts), "clOrdId": _client_id("close" if reduce else "open")}
    if position_mode() == "long_short_mode":
        body["posSide"] = direction
    elif reduce:
        body["reduceOnly"] = True
    rows = _request("POST", "/api/v5/trade/order", body=body)
    row = rows[0] if rows else {}
    return {"order_id": row.get("ordId"), "contracts": contracts, "base_qty": base_for_contracts(contracts, spec)}


def pending_algos():
    return _request("GET", "/api/v5/trade/orders-algo-pending", params={"ordType": "conditional", "instId": inst_id()})


def cancel_protections():
    targets = [row for row in pending_algos() if str(row.get("algoClOrdId") or "").startswith(CLIENT_PREFIX)]
    if not targets:
        return
    _request("POST", "/api/v5/trade/cancel-algos", body=[{"instId": inst_id(), "algoId": str(row.get("algoId"))} for row in targets if row.get("algoId")])


def install_protections(direction, base_qty, tp, sl):
    if _number(tp) <= 0 or _number(sl) <= 0:
        raise OKXAPIError("缺少有效 TP/SL")
    contracts, _ = contracts_for_base(base_qty)
    side = _order_side(direction, reduce=True)
    cancel_protections()
    long_short = position_mode() == "long_short_mode"
    for kind, trigger in (("tp", tp), ("sl", sl)):
        body = {"instId": inst_id(), "tdMode": td_mode(), "side": side, "ordType": "conditional", "sz": str(contracts), "algoClOrdId": _client_id(kind)}
        if long_short:
            body["posSide"] = direction
        else:
            body["reduceOnly"] = True
        if kind == "tp":
            body.update({"tpTriggerPx": str(trigger), "tpOrdPx": "-1", "tpTriggerPxType": "mark"})
        else:
            body.update({"slTriggerPx": str(trigger), "slOrdPx": "-1", "slTriggerPxType": "mark"})
        _request("POST", "/api/v5/trade/order-algo", body=body)
    algos = pending_algos()
    found_tp = any(str(row.get("algoClOrdId") or "").startswith(CLIENT_PREFIX + "tp") for row in algos)
    found_sl = any(str(row.get("algoClOrdId") or "").startswith(CLIENT_PREFIX + "sl") for row in algos)
    if not (found_tp and found_sl):
        raise OKXAPIError("OKX 未確認完整 TP/SL 保護單")


def close_position(direction, base_qty):
    cancel_protections()
    return place_market(direction, base_qty, reduce=True)
