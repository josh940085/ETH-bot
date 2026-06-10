#!/usr/bin/env python3
import asyncio
import base64
import hashlib
import hmac
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import parse_qsl

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response


def _read_local_env_values():
    values = {}
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
                if key:
                    values[key] = value
        except Exception:
            continue
    return values


def _load_local_env():
    for key, value in _read_local_env_values().items():
        if key not in os.environ:
            os.environ[key] = value


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


def _safe_bool_env(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw not in {"0", "false", "no", "off"}


def _runtime_env_value(name: str, default=""):
    local_values = _read_local_env_values()
    if name in local_values:
        return local_values[name]
    return os.getenv(name, default)


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
    return FileResponse(position_path, media_type="application/json")


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
        access_log=_safe_bool_env("POSITION_PANEL_ACCESS_LOG_ENABLED", False),
    )
