#!/usr/bin/env python3
import asyncio
import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from urllib.parse import parse_qsl

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


def _load_local_env():
    for name in (".env", "token.env"):
        path = Path(name)
        if not path.exists():
            continue
        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except Exception:
            continue


_load_local_env()


def _load_origins():
    raw = str(os.getenv("POSITION_PANEL_ALLOWED_ORIGINS", "*") or "").strip()
    if not raw or raw == "*":
        return ["*"]
    return [item.strip() for item in raw.split(",") if item.strip()]


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


def _load_allowed_user_ids():
    raw = str(
        os.getenv("POSITION_PANEL_ALLOWED_TELEGRAM_USER_IDS")
        or os.getenv("TELEGRAM_CHAT_ID")
        or ""
    ).strip()
    values = set()
    for item in raw.split(","):
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
        str(os.getenv("POSITION_PANEL_REALTIME_PORT", "") or "").strip()
        or str(os.getenv("PORT", "") or "").strip()
    )
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


PANEL_TOKEN = str(os.getenv("POSITION_PANEL_REALTIME_TOKEN", "") or "").strip()
TELEGRAM_TOKEN = str(os.getenv("TELEGRAM_TOKEN", "") or "").strip()
TELEGRAM_INIT_DATA_MAX_AGE_SEC = max(60, _safe_int_env("POSITION_PANEL_TELEGRAM_AUTH_MAX_AGE_SEC", 86400))
ALLOWED_TELEGRAM_USER_IDS = _load_allowed_user_ids()
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


def _token_valid(token: str) -> bool:
    if not PANEL_TOKEN:
        return True
    return str(token or "").strip() == PANEL_TOKEN


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


def _parse_telegram_init_data(init_data: str) -> dict:
    values = {}
    for key, value in parse_qsl(str(init_data or ""), keep_blank_values=True):
        if key:
            values[key] = value
    return values


def _validate_telegram_init_data(init_data: str):
    if not TELEGRAM_TOKEN:
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
    if auth_date <= 0 or (now_ts - auth_date) > TELEGRAM_INIT_DATA_MAX_AGE_SEC:
        return None

    data_check_string = "\n".join(f"{key}={payload[key]}" for key in sorted(payload.keys()))
    secret_key = hmac.new(b"WebAppData", TELEGRAM_TOKEN.encode("utf-8"), hashlib.sha256).digest()
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

    if ALLOWED_TELEGRAM_USER_IDS and user_id not in ALLOWED_TELEGRAM_USER_IDS:
        return None

    payload["user"] = user
    payload["user_id"] = user_id
    payload["auth_date"] = auth_date
    return payload


def _viewer_authorized_http(request: Request) -> bool:
    if PANEL_TOKEN and _token_valid(_extract_token(request)):
        return True
    if not TELEGRAM_TOKEN:
        return not PANEL_TOKEN
    return _validate_telegram_init_data(_extract_init_data_http(request)) is not None


def _viewer_authorized_ws(websocket: WebSocket) -> bool:
    token = str(websocket.query_params.get("token", "") or "").strip()
    if PANEL_TOKEN and _token_valid(token):
        return True
    if not TELEGRAM_TOKEN:
        return not PANEL_TOKEN
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
    return {"ok": True, "service": "panel-realtime"}


@app.get("/healthz")
async def healthz():
    return {"ok": True, "ts": int(time.time())}


@app.get("/api/panel/state")
async def get_panel_state(request: Request):
    if not _viewer_authorized_http(request):
        raise HTTPException(status_code=401, detail="unauthorized")

    async with STATE_LOCK:
        payload = dict(LATEST_STATE)
    return JSONResponse(payload)


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
    )
