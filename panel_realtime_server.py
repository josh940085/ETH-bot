#!/usr/bin/env python3
import asyncio
import json
import os
import time

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


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
    if not _token_valid(_extract_token(request)):
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
    token = str(websocket.query_params.get("token", "") or "").strip()
    if not _token_valid(token):
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
