"""Small localhost-only bridge from ETH-bot to the self-hosted n8n workflow."""

import os
import time

import requests

from runtime_config import env_bool
from runtime_paths import REPO_DIR


DEFAULT_N8N_NOTIFICATION_URL = "http://127.0.0.1:5678/webhook/eth-bot-notifications"
N8N_WEBHOOK_SECRET_FILE = REPO_DIR / ".runtime" / "n8n-home" / ".webhook_secret"
_LAST_ERROR_LOG_TS = 0.0


def n8n_notifications_enabled() -> bool:
    if env_bool("ETH_BOT_DISABLE_LIVE", False):
        return False
    return env_bool("N8N_NOTIFICATIONS_ENABLED", True)


def _get_webhook_secret() -> str:
    try:
        return N8N_WEBHOOK_SECRET_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def post_n8n_notification(destination, payload, *, wait_for_response=False, timeout=5, session=None):
    """Ask n8n to deliver one notification; callers retain direct-send fallback.

    A read timeout is delivery-ambiguous: n8n may already have sent the message.
    Return a successful sentinel in that case so callers do not send the same
    notification directly as a fallback.
    """
    global _LAST_ERROR_LOG_TS

    if not n8n_notifications_enabled():
        return None

    target = str(destination or "").strip().lower()
    if target not in {"telegram", "discord_news", "discord_trade"}:
        return None
    if not isinstance(payload, dict) or not payload:
        return None

    url = str(os.getenv("N8N_NOTIFICATION_URL", DEFAULT_N8N_NOTIFICATION_URL) or "").strip()
    if not url:
        return None
    webhook_secret = _get_webhook_secret()
    if not webhook_secret:
        return None

    client = session if session is not None else requests
    try:
        response = client.post(
            url,
            json={
                "destination": target,
                "payload": payload,
                "wait_for_response": bool(wait_for_response),
                "source": "eth-bot",
                "secret": webhook_secret,
                "ts": int(time.time()),
            },
            timeout=max(1, int(timeout)),
        )
        if 200 <= int(response.status_code) < 300:
            return response
        raise RuntimeError(f"HTTP {response.status_code}")
    except requests.exceptions.Timeout as exc:
        now_ts = time.time()
        if now_ts - _LAST_ERROR_LOG_TS >= 60:
            print(f"⚠️ n8n 回覆逾時，為避免重複通知不啟用直送備援: {exc}")
            _LAST_ERROR_LOG_TS = now_ts
        ambiguous_response = requests.Response()
        ambiguous_response.status_code = 200
        ambiguous_response.reason = "n8n delivery status unknown after timeout"
        ambiguous_response._content = b""
        return ambiguous_response
    except Exception as exc:
        now_ts = time.time()
        if now_ts - _LAST_ERROR_LOG_TS >= 60:
            print(f"⚠️ n8n 通知暫不可用，改用原直送備援: {exc}")
            _LAST_ERROR_LOG_TS = now_ts
        return None
